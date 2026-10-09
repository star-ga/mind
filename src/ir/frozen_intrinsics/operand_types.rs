// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! What an intrinsic operand or result DENOTES (W1): the frozen emitter treats
//! every intrinsic operand and result as 64 raw bits, so admission proves each
//! one is an i64 slot rather than trusting the call's name.

use super::{FrozenResult, frozen_native_intrinsic, reject, visit_fn_instrs};
use crate::ir::frozen_profile::FrozenProfileRejection;
use crate::ir::{BinOp, Instr, ValueId};

/// Refuse a call to a frozen intrinsic whose operand is not PROVEN to denote an
/// i64 slot.
///
/// The frozen emitter loads every operand as 64 raw bits. For an i64 that is the
/// value; for an f64 it is the IEEE bit pattern (`__mind_wrap_i8(300.0)` is 0,
/// `__mind_alloc(16.0)` asks for ~4.6e18 bytes), and the MLIR backend refuses the
/// same call as a type mismatch. So admission keys on what each operand DENOTES,
/// proven positively per function scope: an id is an i64 slot only when every
/// definition of it, and every value that can flow into it, is one —
///
/// * an integer constant, an i64 / integer-named / declared-struct parameter or
///   call result (from the module's declared signatures), a value-returning
///   frozen intrinsic, integer arithmetic over two proven operands;
/// * an `if` result or merge whose two incoming values are proven, a loop-carried
///   variable whose pre-loop, back-edge, `break` and `continue` values are proven.
///
/// Everything else — a float, a bool, an `i32`, a comparison, an array element,
/// an undeclared callee, an id the scope never defines — is unproven, and an
/// unproven operand is refused (`call.intrinsic_operand_not_i64`). Symmetrically, a
/// value an intrinsic RETURNS must be consumed as an i64 wherever it flows
/// (`call.intrinsic_result_not_i64`). The loop facts are solved as a greatest fixpoint
/// (start from every id whose own definition qualifies and remove any id with an
/// unproven input until nothing changes), so a cycle through a back edge is
/// proven exactly when every value entering it is.
pub(crate) fn admit_intrinsic_operand_types(
    module: &crate::ir::IRModule,
) -> Result<(), FrozenProfileRejection> {
    admit_operands_in_scope(module, &module.instrs, None)?;
    admit_operands_in_nested_fns(module, &module.instrs)
}

fn admit_operands_in_nested_fns(
    module: &crate::ir::IRModule,
    instrs: &[Instr],
) -> Result<(), FrozenProfileRejection> {
    let mut result = Ok(());
    visit_all_fndefs(instrs, &mut |scope, body| {
        if result.is_ok() {
            result = admit_operands_in_scope(module, body, Some(scope));
        }
    });
    result
}

/// A visitor over one function: its scope and its body.
type FnVisitor<'v> = dyn FnMut(FnScope<'_>, &[Instr]) + 'v;

/// The function a scope belongs to: its name (the key of its declared
/// signature), its parameter count and its tail value.
#[derive(Clone, Copy)]
struct FnScope<'a> {
    name: &'a str,
    n_params: usize,
    ret_id: Option<ValueId>,
}

/// Every `FnDef` at any depth, each visited once with its own body.
fn visit_all_fndefs(instrs: &[Instr], f: &mut FnVisitor<'_>) {
    for instr in instrs {
        if let Instr::FnDef {
            name,
            params,
            body,
            ret_id,
            ..
        } = instr
        {
            let scope = FnScope {
                name,
                n_params: params.len(),
                ret_id: *ret_id,
            };
            f(scope, body);
            visit_all_fndefs(body, f);
            continue;
        }
        #[cfg(feature = "std-surface")]
        for nested in crate::ir::instr_bodies(instr) {
            visit_all_fndefs(nested, f);
        }
    }
}

/// One proof obligation on an id: it holds when every listed id is proven. An
/// empty list is an unconditional proof; `None` can never be proven.
type SlotFact = Option<Vec<ValueId>>;

fn admit_operands_in_scope(
    module: &crate::ir::IRModule,
    body: &[Instr],
    scope: Option<FnScope<'_>>,
) -> Result<(), FrozenProfileRejection> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut facts: BTreeMap<ValueId, Vec<SlotFact>> = BTreeMap::new();
    let mut calls: Vec<Vec<ValueId>> = Vec::new();
    let mut produced: BTreeSet<ValueId> = BTreeSet::new();
    visit_fn_instrs(body, &mut |instr| {
        if let Instr::Call {
            name, args, dst, ..
        } = instr
        {
            if let Some(row) = frozen_native_intrinsic(name) {
                calls.push(args.clone());
                if row.result == FrozenResult::Value {
                    produced.insert(*dst);
                }
            }
        }
        if let Some(dst) = crate::ir::instruction_dst(instr) {
            let fact = slot_fact(module, scope, instr);
            facts.entry(dst).or_default().push(fact);
        }
        for (id, fact) in region_facts(instr) {
            facts.entry(id).or_default().push(fact);
        }
    });
    if calls.is_empty() {
        return Ok(());
    }
    // Greatest fixpoint: assume every id whose facts are all satisfiable, then
    // withdraw any id that depends on an unproven one, transitively.
    let mut proven: BTreeSet<ValueId> = facts
        .iter()
        .filter(|(_, fs)| fs.iter().all(Option::is_some))
        .map(|(id, _)| *id)
        .collect();
    let mut users: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
    for (id, fs) in &facts {
        for needs in fs.iter().flatten() {
            for n in needs {
                users.entry(*n).or_default().push(*id);
            }
        }
    }
    let mut work: Vec<ValueId> = facts
        .keys()
        .chain(users.keys())
        .filter(|id| !proven.contains(id))
        .copied()
        .collect();
    while let Some(lost) = work.pop() {
        for user in users.get(&lost).into_iter().flatten() {
            if proven.remove(user) {
                work.push(*user);
            }
        }
    }
    if !calls
        .iter()
        .all(|args| args.iter().all(|a| proven.contains(a)))
    {
        return reject("call.intrinsic_operand_not_i64");
    }
    // The RESULT side: an intrinsic's i64 must stay an i64. Everything it flows
    // into (arithmetic, an `if` value, a loop-carried variable) must itself be a
    // proven slot, and wherever it is finally consumed the consumer must take an
    // i64: `let y: f64 = __mind_load_i64(p); if y == 1.5 {..}` would compare the
    // raw bits as a double natively, while the MLIR backend refuses the program.
    let mut work: Vec<ValueId> = produced.iter().copied().collect();
    while let Some(v) = work.pop() {
        for user in users.get(&v).into_iter().flatten() {
            if produced.insert(*user) {
                work.push(*user);
            }
        }
    }
    let mut sinks_ok = produced.is_subset(&proven)
        && scope
            .and_then(|s| s.ret_id)
            .is_none_or(|r| !produced.contains(&r) || fn_returns_i64_slot(module, scope));
    visit_fn_instrs(body, &mut |instr| {
        sinks_ok &= result_use_is_i64(module, scope, instr, &produced, &proven);
    });
    if sinks_ok {
        Ok(())
    } else {
        reject("call.intrinsic_result_not_i64")
    }
}

/// Does `instr` consume every intrinsic-produced value it reads as an i64?
fn result_use_is_i64(
    module: &crate::ir::IRModule,
    scope: Option<FnScope<'_>>,
    instr: &Instr,
    produced: &std::collections::BTreeSet<ValueId>,
    proven: &std::collections::BTreeSet<ValueId>,
) -> bool {
    let from_intrinsic = |v: &ValueId| produced.contains(v);
    match instr {
        // Intrinsic operands are held to `proven` above; a user callee must DECLARE
        // an i64-slot parameter where it receives one.
        Instr::Call { name, args, .. } => {
            frozen_native_intrinsic(name).is_some()
                || args.iter().enumerate().all(|(i, a)| {
                    !from_intrinsic(a)
                        || declared_param(module, name, i, args.len())
                            .is_some_and(|t| type_is_i64_slot(module, t))
                })
        }
        // Arithmetic flows on (its dst is held to `proven`); a comparison must
        // compare two i64 slots, never an i64 against a float.
        Instr::BinOp { op, lhs, rhs, .. } => {
            binop_is_integer_arith(op)
                || !(from_intrinsic(lhs) || from_intrinsic(rhs))
                || (proven.contains(lhs) && proven.contains(rhs))
        }
        Instr::Return { value } => {
            value.is_none_or(|v| !from_intrinsic(&v) || fn_returns_i64_slot(module, scope))
        }
        // Region results, merges and loop-carried values flow on; a condition must
        // be a comparison, not a raw intrinsic value.
        #[cfg(feature = "std-surface")]
        Instr::If { cond_id, .. } | Instr::While { cond_id, .. } => !from_intrinsic(cond_id),
        #[cfg(feature = "std-surface")]
        Instr::Break { .. } | Instr::Continue { .. } => true,
        // Any other consumer (an array index, a tensor, an output) is not proven.
        other => {
            let mut ok = true;
            crate::opt::ir_canonical::for_each_operand(other, |v| ok &= !from_intrinsic(&v));
            ok
        }
    }
}

fn fn_returns_i64_slot(module: &crate::ir::IRModule, scope: Option<FnScope<'_>>) -> bool {
    scope
        .and_then(|s| declared_ret(module, s.name))
        .is_some_and(|t| type_is_i64_slot(module, t))
}

/// The proof obligation an instruction's own `dst` carries.
fn slot_fact(module: &crate::ir::IRModule, scope: Option<FnScope<'_>>, instr: &Instr) -> SlotFact {
    match instr {
        Instr::ConstI64(..) => Some(Vec::new()),
        Instr::BinOp { op, lhs, rhs, .. } => binop_is_integer_arith(op).then(|| vec![*lhs, *rhs]),
        Instr::Param { index, .. } => {
            let param = scope.and_then(|s| declared_param(module, s.name, *index, s.n_params));
            param
                .is_some_and(|t| type_is_i64_slot(module, t))
                .then(Vec::new)
        }
        Instr::Call { name, .. } => {
            let slot = match frozen_native_intrinsic(name) {
                Some(row) => row.result == FrozenResult::Value,
                None => declared_ret(module, name).is_some_and(|t| type_is_i64_slot(module, t)),
            };
            slot.then(Vec::new)
        }
        #[cfg(feature = "std-surface")]
        Instr::If {
            then_result,
            else_result,
            ..
        } => Some(vec![*then_result, *else_result]),
        // Every other definition (a float, an array element, a tensor, a region
        // result, ...) is not an i64 slot this predicate has proven.
        _ => None,
    }
}

/// The facts a region adds for the ids it defines beyond `instruction_dst`: an
/// `if` merge needs both incoming values; a loop-carried variable's pre-loop id
/// (the block argument inside the loop) and its exit id need the back-edge value
/// and every `break` / `continue` value carried under the same name.
fn region_facts(instr: &Instr) -> Vec<(ValueId, SlotFact)> {
    #[cfg(feature = "std-surface")]
    match instr {
        Instr::If { merges, .. } => {
            return merges
                .iter()
                .map(|(merge, then_val, else_val)| (*merge, Some(vec![*then_val, *else_val])))
                .collect();
        }
        Instr::While {
            cond_instrs,
            body,
            live_vars,
            init_ids,
            exit_ids,
            ..
        } => {
            let mut jumps: Vec<(String, ValueId)> = Vec::new();
            for region in [cond_instrs.as_slice(), body.as_slice()] {
                visit_fn_instrs(region, &mut |i| {
                    if let Instr::Break { live } | Instr::Continue { live } = i {
                        jumps.extend(live.iter().cloned());
                    }
                });
            }
            let mut out = Vec::new();
            for (k, (var, post)) in live_vars.iter().enumerate() {
                let mut needs = vec![*post];
                needs.extend(jumps.iter().filter(|(n, _)| n == var).map(|(_, v)| *v));
                // The exit value can also be the pre-loop value (zero iterations); a
                // loop with no recorded pre-loop id proves nothing about its exit.
                let exit_fact = init_ids.get(k).map(|init| {
                    out.push((*init, Some(needs.clone())));
                    let mut with_init = needs.clone();
                    with_init.push(*init);
                    with_init
                });
                if let Some(exit) = exit_ids.get(k) {
                    out.push((*exit, exit_fact));
                }
            }
            return out;
        }
        _ => {}
    }
    let _ = instr;
    Vec::new()
}

/// Integer arithmetic keeps an i64 slot an i64 slot; a comparison yields a bool.
fn binop_is_integer_arith(op: &BinOp) -> bool {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => true,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => false,
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => true,
    }
}

/// Does a declared type denote ONE i64 slot on both backends? `i64`, the integer
/// names the MLIR backend lowers to `i64` (narrow ones are masked in the IR), a
/// raw pointer, and a struct this module declares (an aggregate is an i64 handle
/// in the IR). Exhaustive on purpose: a new `TypeAnn` variant must be decided here.
fn type_is_i64_slot(module: &crate::ir::IRModule, ty: &crate::ast::TypeAnn) -> bool {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarI64 | TypeAnn::RawPtr { .. } => true,
        TypeAnn::Named(name) => {
            matches!(
                name.as_str(),
                "i64" | "u64" | "i8" | "u8" | "i16" | "u16" | "isize" | "usize"
            ) || declares_struct(module, name)
        }
        TypeAnn::ScalarI32
        | TypeAnn::ScalarU32
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool
        | TypeAnn::Tensor { .. }
        | TypeAnn::DiffTensor { .. }
        | TypeAnn::Slice { .. }
        | TypeAnn::Array { .. }
        | TypeAnn::Ref { .. }
        | TypeAnn::Generic { .. }
        | TypeAnn::Tuple { .. }
        | TypeAnn::SparseTensor { .. }
        | TypeAnn::FnPtr { .. } => false,
    }
}

#[cfg(feature = "std-surface")]
fn declares_struct(module: &crate::ir::IRModule, name: &str) -> bool {
    module.struct_field_types.contains_key(name)
}

#[cfg(feature = "std-surface")]
/// Parameter `index` of `fn_name`'s declared signature, trusted only when the
/// signature lists exactly `arity` parameters: a signature recorded without an
/// implicit receiver would otherwise shift every index onto its neighbour's type.
fn declared_param<'m>(
    module: &'m crate::ir::IRModule,
    fn_name: &str,
    index: usize,
    arity: usize,
) -> Option<&'m crate::ast::TypeAnn> {
    let params = &module.fn_signatures.get(fn_name)?.0;
    if params.len() != arity {
        return None;
    }
    params.get(index)
}

#[cfg(feature = "std-surface")]
fn declared_ret<'m>(
    module: &'m crate::ir::IRModule,
    callee: &str,
) -> Option<&'m crate::ast::TypeAnn> {
    module.fn_signatures.get(callee)?.1.as_ref()
}

// Without `std-surface` the IR carries no declared types, so no parameter, call
// result or struct handle is proven: an operand must be a constant or arithmetic
// over constants. Refusal is the safe direction.
#[cfg(not(feature = "std-surface"))]
fn declares_struct(_module: &crate::ir::IRModule, _name: &str) -> bool {
    false
}

#[cfg(not(feature = "std-surface"))]
fn declared_param<'m>(
    _module: &'m crate::ir::IRModule,
    _fn_name: &str,
    _index: usize,
    _arity: usize,
) -> Option<&'m crate::ast::TypeAnn> {
    None
}

#[cfg(not(feature = "std-surface"))]
fn declared_ret<'m>(
    _module: &'m crate::ir::IRModule,
    _callee: &str,
) -> Option<&'m crate::ast::TypeAnn> {
    None
}
