// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! What an intrinsic operand or result DENOTES (W1): the frozen emitter treats
//! every intrinsic operand and result as 64 raw bits, so admission proves each
//! one is an i64 slot rather than trusting the call's name.

use std::collections::{BTreeMap, BTreeSet};

use super::program::{Key, Program};
use super::{FrozenResult, frozen_native_intrinsic, reject, visit_fn_instrs};
use crate::ir::frozen_profile::FrozenProfileRejection;
use crate::ir::{BinOp, Instr, ValueId};

/// Refuse a program in which a frozen-intrinsic operand is not PROVEN to denote
/// an i64 slot, an intrinsic's result is consumed as anything else, or a call
/// into a linked std function passes an unproven value for an i64 parameter.
///
/// The frozen emitter loads every operand as 64 raw bits. For an i64 that is the
/// value; for an f64 it is the IEEE bit pattern (`__mind_wrap_i8(300.0)` is 0,
/// `__mind_alloc(16.0)` asks for ~4.6e18 bytes), and the MLIR backend refuses the
/// same call as a type mismatch. So admission keys on what each operand DENOTES,
/// proven positively over the WHOLE program: a value is an i64 slot only when
/// every definition of it, and every value that can flow into it, is one —
///
/// * an integer constant, a value-returning frozen intrinsic, integer arithmetic
///   over two proven operands;
/// * a parameter declared an i64 slot whose argument at EVERY call site in the
///   program is proven (a declaration alone proves nothing: the native bridge runs
///   no type checker, so `fn w(x: i64)` called as `w(300.0)` receives the bits);
/// * a call result whose callee is declared to return an i64 slot and whose every
///   `return` value (and tail value) is proven;
/// * an `if` result or merge whose two incoming values are proven, a loop-carried
///   variable whose pre-loop, back-edge, `break` and `continue` values are proven.
///
/// Everything else — a float, a bool, an `i32`, a comparison, an array element,
/// an undeclared or ambiguous callee, an id the scope never defines — is
/// unproven, and an unproven operand is refused (`call.intrinsic_operand_not_i64`).
/// Symmetrically, a value an intrinsic RETURNS must be consumed as an i64
/// wherever it flows (`call.intrinsic_result_not_i64`). The facts are solved as
/// one greatest fixpoint over every scope of the program (start from every value
/// whose own definition qualifies and withdraw any value with an unproven input
/// until nothing changes), so a cycle through a back edge or through recursion
/// is proven exactly when every value entering it is.
pub(crate) fn admit_program_operand_types(
    prog: &Program<'_>,
) -> Result<(), FrozenProfileRejection> {
    let flow = collect_flow(prog);
    if flow.intrinsic_calls.is_empty() && flow.std_calls.is_empty() {
        return Ok(());
    }
    let facts = program_facts(prog, &flow);
    let mut users: BTreeMap<Key, Vec<Key>> = BTreeMap::new();
    for (id, fs) in &facts {
        for needs in fs.iter().flatten() {
            for n in needs {
                users.entry(*n).or_default().push(*id);
            }
        }
    }
    let proven = greatest_fixpoint(&facts, &users);
    if !flow
        .intrinsic_calls
        .iter()
        .all(|(s, args)| args.iter().all(|a| proven.contains(&(*s, *a))))
    {
        return reject("call.intrinsic_operand_not_i64");
    }
    // A call from the user program into a linked std function must pass a proven
    // value for every parameter the std function declares an i64 slot, with the
    // std function's own arity: std bodies are written against their declared
    // types, and the native bridge runs no type checker that would refuse
    // `mask32(1.5)` the way `mindc check` does.
    for (s, callee, args) in &flow.std_calls {
        let n_params = prog.scopes[*callee].func.map_or(usize::MAX, |f| f.n_params);
        let ok = args.len() == n_params
            && args.iter().enumerate().all(|(i, a)| {
                !prog.param_is_slot(*callee, i, n_params) || proven.contains(&(*s, *a))
            });
        if !ok {
            return reject("call.std_argument_not_i64");
        }
    }
    admit_result_sinks(prog, &flow, &users, &proven)
}

/// The RESULT side: an intrinsic's i64 must stay an i64. Everything it flows into
/// (arithmetic, an `if` value, a loop-carried variable, a parameter, a call
/// result) must itself be a proven slot, and wherever it is finally consumed the
/// consumer must take an i64: `let y: f64 = __mind_load_i64(p); if y == 1.5 {..}`
/// would compare the raw bits as a double natively, while the MLIR backend
/// refuses the program.
fn admit_result_sinks(
    prog: &Program<'_>,
    flow: &Flow,
    users: &BTreeMap<Key, Vec<Key>>,
    proven: &BTreeSet<Key>,
) -> Result<(), FrozenProfileRejection> {
    let mut produced = flow.produced.clone();
    let mut work: Vec<Key> = produced.iter().copied().collect();
    while let Some(v) = work.pop() {
        for user in users.get(&v).into_iter().flatten() {
            if produced.insert(*user) {
                work.push(*user);
            }
        }
    }
    if produced.is_empty() {
        return Ok(());
    }
    let mut sinks_ok = produced.is_subset(proven);
    for (s, scope) in prog.scopes.iter().enumerate() {
        if let Some(r) = scope.func.and_then(|f| f.ret_id) {
            sinks_ok &= !produced.contains(&(s, r)) || prog.ret_is_slot(s);
        }
        visit_fn_instrs(scope.body, &mut |instr| {
            sinks_ok &= result_use_is_i64(prog, s, instr, &produced, proven);
        });
    }
    if sinks_ok {
        Ok(())
    } else {
        reject("call.intrinsic_result_not_i64")
    }
}

/// What one walk over the program collects.
#[derive(Default)]
struct Flow {
    /// Every frozen-intrinsic call: its scope and operands.
    intrinsic_calls: Vec<(usize, Vec<ValueId>)>,
    /// Every value a value-returning intrinsic produces.
    produced: BTreeSet<Key>,
    /// Callee scope -> every call site of it: (caller scope, operands).
    call_sites: BTreeMap<usize, Vec<(usize, Vec<ValueId>)>>,
    /// Scope -> every `return` operand in it (`None` for a bare `return`).
    returns: BTreeMap<usize, Vec<Option<ValueId>>>,
    /// Calls from the user program into a linked std function:
    /// (caller scope, callee scope, operands).
    std_calls: Vec<(usize, usize, Vec<ValueId>)>,
}

fn collect_flow(prog: &Program<'_>) -> Flow {
    let mut flow = Flow::default();
    for (s, scope) in prog.scopes.iter().enumerate() {
        visit_fn_instrs(scope.body, &mut |instr| match instr {
            Instr::Call {
                name, args, dst, ..
            } => {
                if let Some(row) = frozen_native_intrinsic(name) {
                    flow.intrinsic_calls.push((s, args.clone()));
                    if row.result == FrozenResult::Value {
                        flow.produced.insert((s, *dst));
                    }
                } else if let Some(c) = prog.callee(name) {
                    flow.call_sites
                        .entry(c)
                        .or_default()
                        .push((s, args.clone()));
                    if scope.part == 0 && prog.scopes[c].part != 0 {
                        flow.std_calls.push((s, c, args.clone()));
                    }
                }
            }
            Instr::Return { value } => flow.returns.entry(s).or_default().push(*value),
            _ => {}
        });
    }
    flow
}

/// One proof obligation on a value: it holds when every listed value is proven.
/// An empty list is an unconditional proof; `None` can never be proven.
type SlotFact = Option<Vec<Key>>;

fn program_facts(prog: &Program<'_>, flow: &Flow) -> BTreeMap<Key, Vec<SlotFact>> {
    let mut facts: BTreeMap<Key, Vec<SlotFact>> = BTreeMap::new();
    for (s, scope) in prog.scopes.iter().enumerate() {
        visit_fn_instrs(scope.body, &mut |instr| {
            if let Some(dst) = crate::ir::instruction_dst(instr) {
                let fact = slot_fact(prog, flow, s, instr);
                facts.entry((s, dst)).or_default().push(fact);
            }
            for (id, fact) in region_facts(instr) {
                let fact = fact.map(|ids| ids.into_iter().map(|v| (s, v)).collect());
                facts.entry((s, id)).or_default().push(fact);
            }
        });
    }
    facts
}

/// Greatest fixpoint: assume every value whose facts are all satisfiable, then
/// withdraw any value that depends on an unproven one, transitively.
fn greatest_fixpoint(
    facts: &BTreeMap<Key, Vec<SlotFact>>,
    users: &BTreeMap<Key, Vec<Key>>,
) -> BTreeSet<Key> {
    let mut proven: BTreeSet<Key> = facts
        .iter()
        .filter(|(_, fs)| fs.iter().all(Option::is_some))
        .map(|(id, _)| *id)
        .collect();
    let mut work: Vec<Key> = facts
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
    proven
}

/// Does `instr` consume every intrinsic-produced value it reads as an i64?
fn result_use_is_i64(
    prog: &Program<'_>,
    s: usize,
    instr: &Instr,
    produced: &BTreeSet<Key>,
    proven: &BTreeSet<Key>,
) -> bool {
    let from_intrinsic = |v: &ValueId| produced.contains(&(s, *v));
    match instr {
        // Intrinsic operands are held to `proven` above; a user callee must DECLARE
        // an i64-slot parameter where it receives one.
        Instr::Call { name, args, .. } => {
            frozen_native_intrinsic(name).is_some()
                || args.iter().enumerate().all(|(i, a)| {
                    !from_intrinsic(a)
                        || prog
                            .callee(name)
                            .is_some_and(|c| prog.param_is_slot(c, i, args.len()))
                })
        }
        // Arithmetic flows on (its dst is held to `proven`); a comparison must
        // compare two i64 slots, never an i64 against a float.
        Instr::BinOp { op, lhs, rhs, .. } => {
            binop_is_integer_arith(op)
                || !(from_intrinsic(lhs) || from_intrinsic(rhs))
                || (proven.contains(&(s, *lhs)) && proven.contains(&(s, *rhs)))
        }
        Instr::Return { value } => value.is_none_or(|v| !from_intrinsic(&v) || prog.ret_is_slot(s)),
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

/// The proof obligation an instruction's own `dst` carries in scope `s`.
fn slot_fact(prog: &Program<'_>, flow: &Flow, s: usize, instr: &Instr) -> SlotFact {
    let local = |ids: &[ValueId]| Some(ids.iter().map(|v| (s, *v)).collect());
    match instr {
        Instr::ConstI64(..) => Some(Vec::new()),
        Instr::BinOp { op, lhs, rhs, .. } => {
            if binop_is_integer_arith(op) {
                local(&[*lhs, *rhs])
            } else {
                None
            }
        }
        Instr::Param { index, .. } => param_fact(prog, flow, s, *index),
        Instr::Call { name, args, .. } => match frozen_native_intrinsic(name) {
            Some(row) => (row.result == FrozenResult::Value).then(Vec::new),
            None => call_result_fact(prog, flow, prog.callee(name)?, args.len()),
        },
        #[cfg(feature = "std-surface")]
        Instr::If {
            then_result,
            else_result,
            ..
        } => local(&[*then_result, *else_result]),
        // Every other definition (a float, an array element, a tensor, a region
        // result, ...) is not an i64 slot this predicate has proven.
        _ => None,
    }
}

/// A parameter is proven when its function is the one unambiguous definition of
/// its name, declares it an i64 slot, and EVERY call site in the program passes
/// exactly the function's arity and a proven value at this index.
fn param_fact(prog: &Program<'_>, flow: &Flow, s: usize, index: usize) -> SlotFact {
    let f = prog.scopes[s].func?;
    if prog.callee(f.name) != Some(s) || !prog.param_is_slot(s, index, f.n_params) {
        return None;
    }
    let mut needs = Vec::new();
    for (caller, args) in flow.call_sites.get(&s).into_iter().flatten() {
        if args.len() != f.n_params {
            return None;
        }
        needs.push((*caller, args[index]));
    }
    Some(needs)
}

/// A call result is proven when the callee takes exactly this arity, declares an
/// i64-slot return, and every value it can return — each `return` operand and its
/// tail value — is proven. A bare `return` or a body with nothing to return is
/// never proven.
fn call_result_fact(prog: &Program<'_>, flow: &Flow, c: usize, arity: usize) -> SlotFact {
    let f = prog.scopes[c].func?;
    if f.n_params != arity || !prog.ret_is_slot(c) {
        return None;
    }
    let mut needs = Vec::new();
    for value in flow.returns.get(&c).into_iter().flatten() {
        needs.push((c, (*value)?));
    }
    needs.extend(f.ret_id.map(|r| (c, r)));
    (!needs.is_empty()).then_some(needs)
}

/// The facts a region adds for the ids it defines beyond `instruction_dst`: an
/// `if` merge needs both incoming values; a loop-carried variable's pre-loop id
/// (the block argument inside the loop) and its exit id need the back-edge value
/// and every `break` / `continue` value carried under the same name.
fn region_facts(instr: &Instr) -> Vec<(ValueId, Option<Vec<ValueId>>)> {
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

/// Does a declared type denote ONE i64 slot on both backends? Exactly `i64`, and a
/// struct this module declares (an aggregate is an i64 handle in the IR). Nothing
/// else, whatever its width: a `u64` slot compares unsigned under MLIR and signed
/// natively, a `u8`/`u16`/`u32` slot is masked by MLIR and not natively (measured:
/// `fn big(y: u64)` given `__mind_load_i64(p) == -1` and compared `> 5` is 1 under
/// MLIR, 2 natively), and `i8`/`i16`/`isize`/`usize`/raw pointers/aliases are not the
/// slot the corpus proves. Exhaustive on purpose: a new `TypeAnn` variant must be
/// decided here.
pub(super) fn type_is_i64_slot(module: &crate::ir::IRModule, ty: &crate::ast::TypeAnn) -> bool {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarI64 => true,
        TypeAnn::Named(name) => name == "i64" || declares_struct(module, name),
        TypeAnn::RawPtr { .. }
        | TypeAnn::ScalarI32
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
pub(super) fn declared_param<'m>(
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
pub(super) fn declared_ret<'m>(
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
pub(super) fn declared_param<'m>(
    _module: &'m crate::ir::IRModule,
    _fn_name: &str,
    _index: usize,
    _arity: usize,
) -> Option<&'m crate::ast::TypeAnn> {
    None
}

#[cfg(not(feature = "std-surface"))]
pub(super) fn declared_ret<'m>(
    _module: &'m crate::ir::IRModule,
    _callee: &str,
) -> Option<&'m crate::ast::TypeAnn> {
    None
}
