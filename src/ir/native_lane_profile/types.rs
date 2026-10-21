// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Scalar types, signatures and intrinsic rows for the candidate lane profile.

use crate::ast::{self, TypeAnn};
use crate::ir::{BinOp, ValueId};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneProfileRejection {
    pub construct: &'static str,
    /// Byte offset, in the checked source text, of the innermost source node
    /// or top-level item the walker refused. `None` for IR-only refusals.
    pub at: Option<usize>,
    /// The function whose signature or body was being checked, when known.
    pub function: Option<String>,
}
impl LaneProfileRejection {
    pub(super) fn new(construct: &'static str) -> Self {
        LaneProfileRejection {
            construct,
            at: None,
            function: None,
        }
    }
    /// Record the innermost location only: the first (deepest) caller wins.
    pub(super) fn located(mut self, at: usize) -> Self {
        self.at.get_or_insert(at);
        self
    }
    pub(super) fn in_function(mut self, name: &str) -> Self {
        self.function.get_or_insert_with(|| name.to_string());
        self
    }
}
pub(super) type Check<T> = Result<T, LaneProfileRejection>;
pub(super) fn reject<T>(construct: &'static str) -> Check<T> {
    Err(LaneProfileRejection::new(construct))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    I64,
    F64,
    Unit,
    /// The result of a store intrinsic. It is a statement where a statement
    /// fits and the defined i64 value 0 where a value is read: the MLIR
    /// lowering materialises `arith.constant 0`, and the marked compiler's
    /// value-position store lowering writes 0 into the result slot (see
    /// `nb_block_tail_store_zero` and the store arm of `nb_expr` in
    /// `examples/mindc_mind/main.mind`). Constant initialisers still refuse it.
    Store,
    Never,
}

/// The type a value read observes: a store result reads as its defined 0.
pub(super) fn value(ty: Ty) -> Ty {
    if ty == Ty::Store { Ty::I64 } else { ty }
}
#[derive(Clone, Debug)]
pub(super) struct Signature {
    pub(super) params: Vec<Ty>,
    pub(super) result: Ty,
}
pub(super) type Functions = BTreeMap<String, Signature>;
pub(super) type Values = BTreeMap<ValueId, Ty>;

pub(super) fn scalar(ann: &TypeAnn) -> Check<Ty> {
    match ann {
        TypeAnn::ScalarI64 | TypeAnn::ScalarBool => Ok(Ty::I64),
        TypeAnn::ScalarF64 => Ok(Ty::F64),
        TypeAnn::ScalarF32 => reject("lane.type.non_f64_float"),
        TypeAnn::ScalarI32
        | TypeAnn::ScalarU32
        | TypeAnn::Named(_)
        | TypeAnn::Tensor { .. }
        | TypeAnn::DiffTensor { .. }
        | TypeAnn::Slice { .. }
        | TypeAnn::Array { .. }
        | TypeAnn::Ref { .. }
        | TypeAnn::Generic { .. }
        | TypeAnn::Tuple { .. }
        | TypeAnn::SparseTensor { .. }
        | TypeAnn::RawPtr { .. }
        | TypeAnn::FnPtr { .. } => reject("lane.type.unsupported"),
    }
}

/// Exact terminal-call inventory for closure admission. This alone is NOT
/// admission: the combined source/IR predicate checks argument and result types.
pub(crate) fn intrinsic_arity(name: &str) -> Option<usize> {
    intrinsic_signature(name)
        .map(|s| s.params.len())
        .or_else(|| conversion_arity(name))
}

/// The scalar `as` conversions the source walker admits, as lowering spells
/// them (`src/eval/lower.rs`, `Node::As`): `x as i64` is `__mind_conv_i64(x)`
/// and `x as f64` is `__mind_conv_f64(x)`. Typed by the SOURCE value, because
/// the same callee converts (f64 source) or is the identity (i64 source).
/// IR only: an explicit source call to `__mind_conv_f64` is still unknown.
pub(super) fn conversion_type(name: &str, args: &[Ty]) -> Option<Ty> {
    let args: Vec<Ty> = args.iter().map(|&a| value(a)).collect();
    match (name, args.as_slice()) {
        ("__mind_conv_i64", [Ty::F64]) => Some(Ty::I64),
        ("__mind_conv_f64", [Ty::I64]) => Some(Ty::F64),
        _ => None,
    }
}
fn conversion_arity(name: &str) -> Option<usize> {
    matches!(name, "__mind_conv_f64").then_some(1)
}

fn intrinsic_signature(name: &str) -> Option<Signature> {
    use Ty::{F64, I64};
    // Reuse canonical i64 ABI rows only for the explicitly supported names.
    // The canonical contract marks stores DiscardOnly because the historical
    // frozen seed leaves their result slot unwritten. This profile is selected
    // only for a compiler carrying the lane marker, whose value-position store
    // lowering writes the C runtime's defined 0 (the value MLIR materialises),
    // so a store result is typed `Ty::Store`: a statement, or the value 0.
    // Positioned read/write is delivery-specific.
    let row = match name {
        "__mind_alloc" => Some("alloc"),
        "__mind_load_i64" => Some("load_i64"),
        "__mind_load_i8" => Some("load8"),
        "__mind_store_i64" => Some("store_i64"),
        "__mind_store_i8" => Some("store8"),
        "__mind_read" => Some("read"),
        "__mind_write" => Some("write"),
        _ => None,
    };
    if let Some(logical) = row {
        let contract = crate::intrinsic_contract::intrinsic_contract(logical)?;
        if !contract
            .parameters
            .iter()
            .all(|p| matches!(p, crate::intrinsic_contract::IntrinsicValueType::I64))
        {
            return None;
        }
        return Some(Signature {
            params: vec![I64; contract.parameters.len()],
            result: match contract.result {
                crate::intrinsic_contract::IntrinsicResult::I64 => I64,
                crate::intrinsic_contract::IntrinsicResult::DiscardOnly => Ty::Store,
            },
        });
    }
    let (params, result) = match name {
        "__mind_free" | "__mind_load_i32" | "__mind_conv_i64" | "__mind_conv_i32" => {
            (vec![I64], I64)
        }
        "__mind_realloc" => (vec![I64, I64], I64),
        "__mind_store_i32" => (vec![I64, I64], Ty::Store),
        "__mind_bits_to_f64" => (vec![I64], F64),
        "__mind_f64_to_bits" => (vec![F64], I64),
        // These are the verified raw Linux word ABI, not libc-compatible externs.
        "__mind_sys_getuid" | "__mind_sys_fork" => (vec![], I64),
        "__mind_sys_close"
        | "__mind_sys_fsync"
        | "__mind_sys_umask"
        | "__mind_sys_getpgid"
        | "__mind_sys_fchdir"
        | "__mind_sys_exit_group" => (vec![I64], I64),
        "__mind_sys_fstat"
        | "__mind_sys_flock"
        | "__mind_sys_pipe2"
        | "__mind_sys_dup2"
        | "__mind_sys_kill"
        | "__mind_sys_setpgid"
        | "__mind_sys_getrlimit"
        | "__mind_sys_setrlimit"
        | "__mind_sys_nanosleep"
        | "__mind_sys_clock_gettime"
        | "__mind_sys_memfd_create" => (vec![I64; 2], I64),
        "__mind_sys_fcntl"
        | "__mind_sys_lseek"
        | "__mind_sys_unlinkat"
        | "__mind_sys_close_range"
        | "__mind_sys_getdents64" => (vec![I64; 3], I64),
        "__mind_sys_openat"
        | "__mind_sys_newfstatat"
        | "__mind_sys_renameat"
        | "__mind_sys_wait4"
        | "__mind_sys_rt_sigaction" => (vec![I64; 4], I64),
        "__mind_sys_prctl" | "__mind_sys_execveat" | "__mind_sys_waitid" => (vec![I64; 5], I64),
        _ => return None,
    };
    Some(Signature { params, result })
}

pub(super) fn equal(actual: Ty, wanted: Ty) -> Check<()> {
    // A store result fits a statement slot (Unit) and an i64 value slot.
    let store_fits = actual == Ty::Store && matches!(wanted, Ty::I64 | Ty::Unit);
    if actual == wanted || actual == Ty::Never || store_fits {
        Ok(())
    } else {
        reject("lane.type.mismatch")
    }
}
pub(super) fn unify(a: Ty, b: Ty) -> Check<Ty> {
    if a == Ty::Never {
        return Ok(b);
    }
    if b == Ty::Never || a == b {
        return Ok(a);
    }
    match (a, b) {
        (Ty::Store, Ty::I64) | (Ty::I64, Ty::Store) => Ok(Ty::I64),
        _ => reject("lane.type.branch_mismatch"),
    }
}
pub(super) fn call_type(name: &str, args: &[Ty], functions: &Functions) -> Check<Ty> {
    let intrinsic = intrinsic_signature(name);
    let sig = functions
        .get(name)
        .or(intrinsic.as_ref())
        .ok_or(LaneProfileRejection::new("lane.call.unknown"))?;
    if args.len() != sig.params.len() {
        return reject("lane.call.arity");
    }
    for (&actual, &expected) in args.iter().zip(&sig.params) {
        equal(actual, expected)?;
    }
    Ok(sig.result)
}
pub(super) fn binary(op: &BinOp, left: Ty, right: Ty) -> Check<Ty> {
    let (left, right) = (value(left), value(right));
    equal(left, right)?;
    if !matches!(left, Ty::I64 | Ty::F64) {
        return reject("lane.binop.non_scalar");
    }
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => Ok(left),
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Ok(Ty::I64),
        BinOp::Mod | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
            equal(left, Ty::I64)?;
            Ok(Ty::I64)
        }
    }
}
pub(super) fn ast_bin(op: ast::BinOp) -> BinOp {
    match op {
        ast::BinOp::Add => BinOp::Add,
        ast::BinOp::Sub => BinOp::Sub,
        ast::BinOp::Mul => BinOp::Mul,
        ast::BinOp::Div => BinOp::Div,
        ast::BinOp::Mod => BinOp::Mod,
        ast::BinOp::Eq => BinOp::Eq,
        ast::BinOp::Ne => BinOp::Ne,
        ast::BinOp::Lt => BinOp::Lt,
        ast::BinOp::Le => BinOp::Le,
        ast::BinOp::Gt => BinOp::Gt,
        ast::BinOp::Ge => BinOp::Ge,
    }
}
