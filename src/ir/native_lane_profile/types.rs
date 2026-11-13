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
    /// The std seed module (`std/<m>.mind`) whose reached function was being
    /// checked. When set, `at` is an offset in that module's text, not in the
    /// composed user image.
    pub seed_module: Option<String>,
}
impl LaneProfileRejection {
    pub(super) fn new(construct: &'static str) -> Self {
        LaneProfileRejection {
            construct,
            at: None,
            function: None,
            seed_module: None,
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
    /// Mark a refusal raised inside a reached std seed function.
    pub(super) fn in_seed(mut self, module: &str, function: &str) -> Self {
        self.seed_module
            .get_or_insert_with(|| format!("std/{module}.mind"));
        self.in_function(function)
    }
}
pub(super) type Check<T> = Result<T, LaneProfileRejection>;
pub(super) fn reject<T>(construct: &'static str) -> Check<T> {
    Err(LaneProfileRejection::new(construct))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    I64,
    Bool,
    I32,
    U32,
    /// A C pointer of any allowed pointee (`*const u8`, `*mut u8`, `*mut i32`).
    /// The exact pointee is enforced at extern-signature admission
    /// (`admit_extern`); the lane value graph only needs "pointer origin".
    Ptr,
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
    if matches!(ty, Ty::Store | Ty::Bool) {
        Ty::I64
    } else {
        ty
    }
}

/// Width-erased IR view: i32/u32/ptr all travel in 64-bit slots, so legacy IR
/// cannot tell them apart. Source facts remain mandatory for real widths.
pub(super) fn word(ty: Ty) -> Ty {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Ptr => Ty::I64,
        other => other,
    }
}
#[derive(Clone, Debug)]
pub(super) struct Signature {
    pub(super) params: Vec<Ty>,
    pub(super) result: Ty,
}
impl Signature {
    pub(crate) fn arity(&self) -> usize {
        self.params.len()
    }
}
pub(super) type Functions = BTreeMap<String, Signature>;
pub(super) type Values = BTreeMap<ValueId, Ty>;

pub(super) fn scalar(ann: &TypeAnn) -> Check<Ty> {
    match ann {
        TypeAnn::ScalarI64 => Ok(Ty::I64),
        TypeAnn::ScalarBool => Ok(Ty::Bool),
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
/// them (`src/eval/lower.rs`, `Node::As`): `x as i64` is `__mind_conv_i64(x)`,
/// `x as f64` is `__mind_conv_f64(x)` and `x as u32` is `__mind_conv_u32(x)`.
/// Typed by the SOURCE value, because the same callee converts (f64 source) or
/// is the identity (i64 source). `x as u32` of an integer word keeps its low 32
/// bits on both backends (MLIR masks, the marked compiler `nb_mask_rax_w`), and
/// the result travels as a word like every IR integer; its float form is
/// refused at source (`lane.source.cast`).
/// IR only: an explicit source call to `__mind_conv_f64` is still unknown.
pub(super) fn conversion_type(name: &str, args: &[Ty]) -> Option<Ty> {
    let args: Vec<Ty> = args.iter().map(|&a| value(a)).collect();
    match (name, args.as_slice()) {
        ("__mind_conv_i64", [Ty::F64]) => Some(Ty::I64),
        ("__mind_conv_f64", [Ty::I64]) => Some(Ty::F64),
        ("__mind_conv_u32", [Ty::I64]) => Some(Ty::I64),
        _ => None,
    }
}
/// The runtime helpers `print` lowers to (src/eval/lower.rs, `Node::Print`)
/// and their arity: `printI64(v)`, `printNewline()`, `print_bytes(addr, len)`.
pub(super) const PRINT_HELPERS: [(&str, usize); 3] =
    [("printI64", 1), ("printNewline", 0), ("print_bytes", 2)];

/// IR type of a print helper call, for a print-marked compiler and only when
/// no tier defines the name. The source walk already refused a direct call to
/// an undefined helper and a `print` whose helper names have a definition, so
/// here a helper call can only come from an admitted `print` statement.
pub(super) fn print_helper_type(name: &str, args: &[Ty], callees: &Callees) -> Option<Ty> {
    if !callees.print || !callees.lookup(name).is_empty() {
        return None;
    }
    let args: Vec<Ty> = args.iter().map(|&a| value(a)).collect();
    match (name, args.as_slice()) {
        ("printI64", [Ty::I64]) | ("printNewline", []) => Some(Ty::Unit),
        ("print_bytes", [Ty::I64, Ty::I64]) => Some(Ty::I64),
        _ => None,
    }
}
fn conversion_arity(name: &str) -> Option<usize> {
    matches!(name, "__mind_conv_f64" | "__mind_conv_u32").then_some(1)
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
        "__mind_store_i16" => Some("store_i16"),
        "__mind_store_i8" => Some("store8"),
        "__mind_read" => Some("read"),
        "__mind_write" => Some("write"),
        "__mind_open" => Some("open"),
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
        // One-argument runtime trap. A reached seed body calls it as a
        // statement; the registered row is one i64 and an i64 result.
        "__mind_assert_fail" => (vec![I64], I64),
        "__mind_free" | "__mind_load_i32" | "__mind_conv_i64" | "__mind_conv_i32" => {
            (vec![I64], I64)
        }
        // Sign-extending 16-bit load (nb_emit_load_i16_sx) and the FrozenNative
        // argc/argv rows (seed ids 10/11; the bridge entry sets the argv slot).
        "__mind_load_i16" => (vec![I64], I64),
        "__mind_argc" => (vec![], I64),
        "__mind_argv" => (vec![I64], I64),
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
        // Reserved per-lane arm blocks (kind ledger: examples/mindc_mind/nb_part_dispatch.mind).
        // --- rows: ymm (Layer A + R-02) ---
        "__mind_vpaddd_i32x8"
        | "__mind_vpsubd_i32x8"
        | "__mind_vpaddb_i8x32"
        | "__mind_vmovdqu_i32x8"
        | "__mind_vpaddq_i64x4"
        | "__mind_vpsubq_i64x4" => (vec![I64; 2], I64),
        "__mind_vpshufb_i8x32" => (vec![I64; 3], I64),
        // --- end rows: ymm ---
        // --- rows: cpu_features ---
        "__mind_cpu_features_v1" => (vec![], I64),
        // --- end rows: cpu_features ---
        // --- rows: measure (M-A3) ---
        "__mind_rdtscp_v1" | "__mind_lfence_v1" => (vec![], I64),
        "__mind_sys_perf_event_open" => (vec![I64; 5], I64),
        // --- end rows: measure ---
        // --- rows: shm (A5) ---
        "__mind_cpu_relax" => (vec![], I64),
        "__mind_atomic_load_acquire_i64" => (vec![I64], I64),
        "__mind_atomic_store_release_i64" | "__mind_sys_munmap" => (vec![I64; 2], I64),
        "__mind_sys_sched_setaffinity" => (vec![I64; 3], I64),
        "__mind_sys_mmap" => (vec![I64; 6], I64),
        // --- end rows: shm ---
        // --- rows: ymm_bank (R-01b) ---
        // R-01b register bank (kinds 70-83): literal ymm register / displacement
        // operands, statement-position DiscardOnly rows; the self-host is the fence.
        "__mind_ymm_vmovdqu_ld"
        | "__mind_ymm_vmovdqu_st"
        | "__mind_ymm_vpaddd_i32x8_m"
        | "__mind_ymm_vpsubd_i32x8_m"
        | "__mind_ymm_vpaddb_i8x32_m"
        | "__mind_ymm_vpaddd_i32x8_r"
        | "__mind_ymm_vpsubd_i32x8_r"
        | "__mind_ymm_vpaddb_i8x32_r"
        | "__mind_ymm_vpshufb_i8x32_r"
        | "__mind_ymm_vpaddq_i64x4_m"
        | "__mind_ymm_vpsubq_i64x4_m"
        | "__mind_ymm_vpaddq_i64x4_r"
        | "__mind_ymm_vpsubq_i64x4_r"
        | "__mind_ymm_vpmuludq_i64x4_r"
        | "__mind_ymm_vpand_i64x4_r"
        | "__mind_ymm_vpor_i64x4_r"
        | "__mind_ymm_vpcmpgtq_i64x4_r"
        | "__mind_ymm_vpsrlq_i64x4_r"
        | "__mind_ymm_vpsllq_i64x4_r" => (vec![I64; 3], Ty::Store),
        "__mind_ymm_vpshufb_i8x32_m" => (vec![I64; 4], Ty::Store),
        // --- end rows: ymm_bank ---
        // --- rows: group_sum (0.10.3) ---
        "__mind_group_bank_i64" => (vec![I64; 5], I64),
        "__mind_group_acc32_i64" => (vec![I64; 6], I64),
        // --- end rows: group_sum ---
        _ => return None,
    };
    Some(Signature { params, result })
}

pub(super) fn equal(actual: Ty, wanted: Ty) -> Check<()> {
    // A store result fits a statement slot (Unit) and an i64 value slot.
    let store_fits = actual == Ty::Store && matches!(wanted, Ty::I64 | Ty::Unit);
    if actual == wanted
        || matches!((actual, wanted), (Ty::Bool, Ty::I64) | (Ty::I64, Ty::Bool))
        || actual == Ty::Never
        || store_fits
    {
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
        (Ty::Bool, Ty::I64) | (Ty::I64, Ty::Bool) => Ok(Ty::Bool),
        (Ty::Store, Ty::I64) | (Ty::I64, Ty::Store) => Ok(Ty::I64),
        _ => reject("lane.type.branch_mismatch"),
    }
}
/// The tiers of the call-resolution chain, in lookup order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Tier {
    User,
    Extern,
    Seed,
    Intrinsic,
}

/// The callee tables one lane-checked image resolves its calls against.
///
/// Lookup order is `user fn > extern > seed > intrinsic`. A name defined by
/// more than one tier is refused `lane.call.ambiguous`, never settled by that
/// order: the frozen compiler binds a bare name to the LAST definition in the
/// composed image (`[std seed blob][user region]`), so a seed body's call would
/// bind to a same-named user function. The order only fixes which tier is
/// consulted and reported first.
#[derive(Clone, Copy, Debug)]
pub(super) struct Callees<'a> {
    pub(super) user: &'a Functions,
    /// The `extern "C"` declarations the image admits (`admit_extern`).
    pub(super) externs: &'a Functions,
    /// The std seed functions the image reaches, typed like user functions.
    pub(super) seed: &'a Functions,
    /// The selected compiler carries the `nbprint1` mark: the builtin `print`
    /// statement lowers natively (examples/mindc_mind/nb_print.mind).
    pub(super) print: bool,
}
static NO_FUNCTIONS: Functions = BTreeMap::new();
impl<'a> From<&'a Functions> for Callees<'a> {
    /// The user tier alone (intrinsics are always consulted).
    fn from(user: &'a Functions) -> Self {
        Callees {
            user,
            externs: &NO_FUNCTIONS,
            seed: &NO_FUNCTIONS,
            print: false,
        }
    }
}
impl Callees<'_> {
    /// Every tier that defines `name`, in lookup order.
    pub(super) fn lookup(&self, name: &str) -> Vec<(Tier, Signature)> {
        let tables = [
            (Tier::User, self.user),
            (Tier::Extern, self.externs),
            (Tier::Seed, self.seed),
        ];
        let mut hits: Vec<(Tier, Signature)> = tables
            .iter()
            .filter_map(|(tier, table)| table.get(name).map(|sig| (*tier, sig.clone())))
            .collect();
        hits.extend(intrinsic_signature(name).map(|sig| (Tier::Intrinsic, sig)));
        hits
    }
}

pub(super) fn call_type<'a>(name: &str, args: &[Ty], callees: impl Into<Callees<'a>>) -> Check<Ty> {
    let sig = match callees.into().lookup(name).as_slice() {
        [] => return reject("lane.call.unknown"),
        [(_, sig)] => sig.clone(),
        _ => return reject("lane.call.ambiguous"),
    };
    if args.len() != sig.params.len() {
        return reject("lane.call.arity");
    }
    for (&actual, &expected) in args.iter().zip(&sig.params) {
        equal(actual, expected)?;
    }
    Ok(sig.result)
}

/// IR call typing under width erasure. Every integer/pointer argument and
/// result is compared as a 64-bit word, because legacy IR cannot express the
/// declared width. Source admission already proved the real signature.
pub(super) fn call_type_ir<'a>(
    name: &str,
    args: &[Ty],
    callees: impl Into<Callees<'a>>,
) -> Check<Ty> {
    let sig = match callees.into().lookup(name).as_slice() {
        [] => return reject("lane.call.unknown"),
        [(_, sig)] => sig.clone(),
        _ => return reject("lane.call.ambiguous"),
    };
    if args.len() != sig.params.len() {
        return reject("lane.call.arity");
    }
    for (&actual, &expected) in args.iter().zip(&sig.params) {
        equal(word(actual), word(expected))?;
    }
    Ok(word(sig.result))
}
pub(super) fn binary(op: &BinOp, left: Ty, right: Ty) -> Check<Ty> {
    // Bool operations have i1 width in MLIR, but word width in the seed.
    if left == Ty::Bool || right == Ty::Bool {
        return reject("lane.bool.width");
    }
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

#[cfg(test)]
mod chain_tests {
    use super::*;

    fn table(name: &str, params: Vec<Ty>, result: Ty) -> Functions {
        Functions::from([(name.to_string(), Signature { params, result })])
    }
    fn chain<'a>(user: &'a Functions, externs: &'a Functions, seed: &'a Functions) -> Callees<'a> {
        Callees {
            user,
            externs,
            seed,
            print: false,
        }
    }

    #[test]
    fn each_tier_alone_resolves_with_its_own_signature() {
        let none = Functions::new();
        let f = table("f", vec![Ty::I64], Ty::F64);
        for callees in [
            chain(&f, &none, &none),
            chain(&none, &f, &none),
            chain(&none, &none, &f),
        ] {
            assert_eq!(call_type("f", &[Ty::I64], callees), Ok(Ty::F64));
            assert_eq!(
                call_type("f", &[], callees).unwrap_err().construct,
                "lane.call.arity"
            );
            let wrong = call_type("f", &[Ty::F64], callees).unwrap_err();
            assert_eq!(wrong.construct, "lane.type.mismatch");
        }
        assert_eq!(call_type("__mind_alloc", &[Ty::I64], &none), Ok(Ty::I64));
        let unknown = call_type("g", &[], chain(&f, &f, &none)).unwrap_err();
        assert_eq!(unknown.construct, "lane.call.unknown");
    }

    #[test]
    fn lookup_order_is_user_extern_seed_intrinsic() {
        let alloc = table("__mind_alloc", vec![Ty::I64], Ty::I64);
        let tiers: Vec<Tier> = chain(&alloc, &alloc, &alloc)
            .lookup("__mind_alloc")
            .into_iter()
            .map(|(tier, _)| tier)
            .collect();
        assert_eq!(
            tiers,
            [Tier::User, Tier::Extern, Tier::Seed, Tier::Intrinsic]
        );
    }

    #[test]
    fn a_name_in_two_tiers_is_ambiguous_whatever_the_signatures() {
        let none = Functions::new();
        let f = table("f", vec![Ty::I64], Ty::I64);
        let same = f.clone();
        for callees in [
            chain(&f, &same, &none),
            chain(&f, &none, &same),
            chain(&none, &f, &same),
        ] {
            let err = call_type("f", &[Ty::I64], callees).unwrap_err();
            assert_eq!(err.construct, "lane.call.ambiguous");
        }
        // A tier that redefines an intrinsic is ambiguous with it.
        let alloc = table("__mind_alloc", vec![Ty::I64], Ty::I64);
        for callees in [chain(&alloc, &none, &none), chain(&none, &none, &alloc)] {
            let err = call_type("__mind_alloc", &[Ty::I64], callees).unwrap_err();
            assert_eq!(err.construct, "lane.call.ambiguous");
        }
    }

    #[test]
    fn seed_refusals_carry_the_module_and_the_first_function() {
        let r = LaneProfileRejection::new("lane.type.unsupported")
            .located(7)
            .in_seed("cli", "args_get")
            .in_seed("vec", "outer");
        assert_eq!(r.seed_module.as_deref(), Some("std/cli.mind"));
        assert_eq!(r.function.as_deref(), Some("args_get"));
        assert_eq!(r.at, Some(7));
    }
}
