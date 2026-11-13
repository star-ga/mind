// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! IR admission walk for the candidate lane profile: the single lowering of
//! an admitted source, typed under the same call-resolution chain.

use super::admit::{extern_gate, source_facts};
use super::types::{
    Callees, Check, Functions, LaneProfileRejection, Ty, Values, binary, call_type_ir,
    conversion_type, equal, print_helper_type, reject, unify, value,
};
use super::{arrays, names};
use crate::ast;
use crate::ir::{IRModule, Instr, ValueId};
use std::collections::{BTreeMap, BTreeSet};

fn get(env: &Values, id: &ValueId) -> Check<Ty> {
    env.get(id)
        .copied()
        .ok_or(LaneProfileRejection::new("lane.ir.untyped_value"))
}
fn put(env: &mut Values, id: ValueId, ty: Ty) -> Check<()> {
    let ty = if ty == Ty::Bool { Ty::I64 } else { ty };
    if let Some(old) = env.insert(id, ty) {
        equal(ty, old)?;
    }
    Ok(())
}

struct Ir<'a> {
    callees: Callees<'a>,
    result: Ty,
    loops: Vec<BTreeMap<String, Ty>>,
    params: &'a [(String, ValueId)],
    /// `ConstArray` results of this function: read only by `ArrayLoad`, never
    /// typed in the value env, so any other use is an untyped value.
    arrays: BTreeSet<ValueId>,
}
impl Ir<'_> {
    fn walk(&mut self, body: &[Instr], env: &mut Values) -> Check<()> {
        for instr in body {
            match instr {
                Instr::ConstI64(id, _) => put(env, *id, Ty::I64)?,
                Instr::ConstF64(id, _) => put(env, *id, Ty::F64)?,
                Instr::ConstArray { dst, .. } => {
                    self.arrays.insert(*dst);
                }
                Instr::ArrayLoad { dst, base, index } => {
                    if !self.arrays.contains(base) {
                        return reject("lane.ir.array_base");
                    }
                    equal(get(env, index)?, Ty::I64)?;
                    put(env, *dst, Ty::I64)?;
                }
                Instr::Param { dst, name, index } => {
                    if self.params.get(*index) != Some(&(name.clone(), *dst)) {
                        return reject("lane.ir.parameter_binding");
                    }
                    get(env, dst)?;
                }
                Instr::BinOp { dst, op, lhs, rhs } => {
                    let ty = binary(op, get(env, lhs)?, get(env, rhs)?)?;
                    put(env, *dst, ty)?;
                }
                Instr::Call {
                    dst, name, args, ..
                } => {
                    let args = args
                        .iter()
                        .map(|a| get(env, a))
                        .collect::<Check<Vec<_>>>()?;
                    let ty = match conversion_type(name, &args)
                        .or_else(|| print_helper_type(name, &args, &self.callees))
                    {
                        Some(ty) => ty,
                        None => call_type_ir(name, &args, self.callees)?,
                    };
                    put(env, *dst, value(ty))?;
                }
                Instr::Return { value } => equal(
                    value
                        .as_ref()
                        .map(|v| get(env, v))
                        .transpose()?
                        .unwrap_or(Ty::Unit),
                    self.result,
                )?,
                Instr::Output(id) => {
                    equal(get(env, id)?, Ty::I64)?;
                }
                Instr::If {
                    cond_id,
                    cond_instrs,
                    then_instrs,
                    then_result,
                    else_instrs,
                    else_result,
                    dst,
                    branch_bindings,
                    merges,
                } => {
                    let mut cond = env.clone();
                    self.walk(cond_instrs, &mut cond)?;
                    equal(get(&cond, cond_id)?, Ty::I64)?;
                    let mut a = cond.clone();
                    self.walk(then_instrs, &mut a)?;
                    let mut b = cond;
                    self.walk(else_instrs, &mut b)?;
                    // A branch that surely leaves (return/break/continue)
                    // never reaches the merge, so its side is not typed there:
                    // lowering forwards the other branch's value in its place
                    // (`if c { x = 1; } else { return 0; }`).
                    let then_live = !diverges(then_instrs);
                    let else_live = !diverges(else_instrs);
                    // When BOTH leave, the merge is unreachable: every use of its
                    // ids is dead code (after this `if`, or on an enclosing
                    // branch that `diverges` therefore also marks as leaving).
                    let merged = |left: &ValueId, right: &ValueId| -> Check<Ty> {
                        match (then_live, else_live) {
                            (true, true) => unify(get(&a, left)?, get(&b, right)?),
                            (true, false) => get(&a, left),
                            (false, true) => get(&b, right),
                            (false, false) => Ok(Ty::Never),
                        }
                    };
                    put(env, *dst, merged(then_result, else_result)?)?;
                    for (id, left, right) in merges {
                        put(env, *id, merged(left, right)?)?;
                    }
                    for (_, id) in branch_bindings {
                        // A binding escaping a branch must be defined on both
                        // reaching paths or be an explicit validated merge/dst.
                        if !env.contains_key(id) {
                            put(env, *id, merged(id, id)?)?;
                        }
                    }
                }
                Instr::While {
                    cond_id,
                    cond_instrs,
                    body,
                    live_vars,
                    init_ids,
                    exit_ids,
                } => {
                    if live_vars.len() != init_ids.len() || live_vars.len() != exit_ids.len() {
                        return reject("lane.ir.loop_arity");
                    }
                    let mut inside = env.clone();
                    let mut carried = BTreeMap::new();
                    for ((name, post), init) in live_vars.iter().zip(init_ids) {
                        let ty = get(env, init)?;
                        put(&mut inside, *post, ty)?;
                        if carried.insert(name.clone(), ty).is_some() {
                            return reject("lane.ir.loop_binding");
                        }
                    }
                    self.loops.push(carried);
                    self.walk(cond_instrs, &mut inside)?;
                    equal(get(&inside, cond_id)?, Ty::I64)?;
                    self.walk(body, &mut inside)?;
                    self.loops.pop();
                    for (((_, post), init), exit) in live_vars.iter().zip(init_ids).zip(exit_ids) {
                        let ty = get(env, init)?;
                        equal(get(&inside, post)?, ty)?;
                        put(env, *exit, ty)?;
                    }
                }
                Instr::Break { live } | Instr::Continue { live } => {
                    let expected = self
                        .loops
                        .last()
                        .ok_or(LaneProfileRejection::new("lane.ir.loop_exit_scope"))?;
                    let mut seen = BTreeSet::new();
                    for (name, id) in live {
                        if !seen.insert(name) {
                            return reject("lane.ir.loop_binding");
                        }
                        let ty = get(env, id)?;
                        if let Some(want) = expected.get(name) {
                            equal(ty, *want)?;
                        }
                    }
                    if expected.keys().any(|name| !seen.contains(name)) {
                        return reject("lane.ir.loop_binding");
                    }
                }
                Instr::FnDef { .. } => return reject("lane.ir.nested_function"),
                Instr::ExternFnDecl { name, .. } => {
                    if !self.callees.externs.contains_key(name) {
                        return reject("lane.extern.unknown_symbol");
                    }
                }
                Instr::ArrayStore { .. }
                | Instr::ConstTensor(..)
                | Instr::ConstDenseTensor { .. }
                | Instr::Sum { .. }
                | Instr::Mean { .. }
                | Instr::Relu { .. }
                | Instr::ReluGrad { .. }
                | Instr::Reshape { .. }
                | Instr::ExpandDims { .. }
                | Instr::Squeeze { .. }
                | Instr::Transpose { .. }
                | Instr::Dot { .. }
                | Instr::MatMul { .. }
                | Instr::Conv2d { .. }
                | Instr::Conv2dGradInput { .. }
                | Instr::Conv2dGradFilter { .. }
                | Instr::Index { .. }
                | Instr::Slice { .. }
                | Instr::Gather { .. }
                | Instr::SparseAttr { .. }
                | Instr::Region { .. }
                | Instr::VecLoad { .. }
                | Instr::VecFma { .. }
                | Instr::VecReduceAdd { .. }
                | Instr::VecStore { .. }
                | Instr::VecLoadI32 { .. }
                | Instr::VecMulAddQ16 { .. }
                | Instr::VecReduceAddI64 { .. } => {
                    return Err(LaneProfileRejection::new(names::instr(instr)));
                }
            }
        }
        Ok(())
    }
}

/// True only when control surely leaves `instrs` before falling through: a
/// top-level `return`, `break` or `continue`, or an `if` both of whose branches
/// surely leave. Anything else (a loop, a call) may fall through.
pub(super) fn diverges(instrs: &[Instr]) -> bool {
    instrs.iter().any(|i| match i {
        Instr::Return { .. } | Instr::Break { .. } | Instr::Continue { .. } => true,
        Instr::If {
            then_instrs,
            else_instrs,
            ..
        } => diverges(then_instrs) && diverges(else_instrs),
        _ => false,
    })
}

/// Require the source and its single lowering together. The caller owns their
/// provenance/binding and the exact compiler-capability selection; this function
/// is not a validator for arbitrary independent source/IR pairs or proof data.
pub fn profile_lane_admits(source: &ast::Module, ir: &IRModule, extern_fixed: bool) -> Check<()> {
    extern_gate(source, extern_fixed)?;
    let no_seed = Functions::new();
    let facts = source_facts(source, &no_seed, false)?;
    check_ir(ir, &facts.user, facts.callees(&no_seed))
}

/// The IR leg for one lowering: every `FnDef` binds exactly one of
/// `functions` and is walked under `callees`.
pub(super) fn check_ir(ir: &IRModule, functions: &Functions, callees: Callees) -> Check<()> {
    if !ir.value_types.is_empty() {
        return reject("lane.ir.aggregate_metadata");
    }
    crate::ir::check_ssa_well_formed(ir).map_err(|_| LaneProfileRejection::new("lane.ir.ssa"))?;
    let mut seen = BTreeSet::new();
    let mut globals = Values::new();
    for instr in &ir.instrs {
        if let Instr::FnDef {
            name,
            params,
            ret_id,
            body,
            reap_threshold,
            value_types,
            semantic_types,
        } = instr
        {
            (|| -> Check<()> {
                let sig = functions
                    .get(name)
                    .ok_or(LaneProfileRejection::new("lane.ir.function_binding"))?;
                if !seen.insert(name.clone())
                    || params.len() != sig.params.len()
                    || reap_threshold.is_some()
                {
                    return reject("lane.ir.function_binding");
                }
                let mut env = Values::new();
                for ((_, id), ty) in params.iter().zip(&sig.params) {
                    put(&mut env, *id, *ty)?;
                }
                let mut walk = Ir {
                    callees,
                    result: sig.result,
                    loops: vec![],
                    params,
                    arrays: BTreeSet::new(),
                };
                walk.walk(body, &mut env)?;
                arrays::metadata(&walk.arrays, value_types)?;
                if let Some(ret) = ret_id {
                    equal(get(&env, ret)?, sig.result)?;
                }
                if let Some(meta) = semantic_types {
                    for (id, ty) in meta.values() {
                        use crate::types::{ScalarType, SemanticType};
                        let expected = match ty {
                            SemanticType::Scalar(ScalarType::I64 | ScalarType::Bool) => Ty::I64,
                            SemanticType::Scalar(ScalarType::F64) => Ty::F64,
                            _ if arrays::semantic(&walk.arrays, id, ty) => continue,
                            _ => return reject("lane.ir.semantic_type"),
                        };
                        equal(get(&env, id)?, expected)?;
                    }
                }
                Ok(())
            })()
            .map_err(|r| r.in_function(name))?;
        } else {
            Ir {
                callees,
                result: Ty::Unit,
                loops: vec![],
                params: &[],
                arrays: BTreeSet::new(),
            }
            .walk(std::slice::from_ref(instr), &mut globals)?;
        }
    }
    if seen.len() != functions.len() {
        return reject("lane.ir.function_binding");
    }
    Ok(())
}
