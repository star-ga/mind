// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Source and IR admission walk for the candidate lane profile.

use super::types::{
    Check, Functions, LaneProfileRejection, Signature, Ty, Values, ast_bin, binary, call_type,
    conversion_type, equal, reject, scalar, unify, value,
};
use crate::ast::{self, Literal, Node, TypeAnn};
use crate::ir::{IRModule, Instr, ValueId};
use std::collections::{BTreeMap, BTreeSet};

struct Source<'a> {
    functions: &'a Functions,
    result: Ty,
    loops: usize,
}
impl Source<'_> {
    fn block(&mut self, nodes: &[Node], env: &mut BTreeMap<String, Ty>) -> Check<Ty> {
        let mut result = Ty::Unit;
        for node in nodes {
            let next = self.node(node, env)?;
            if result != Ty::Never {
                result = next;
            }
        }
        Ok(result)
    }
    fn node(&mut self, node: &Node, env: &mut BTreeMap<String, Ty>) -> Check<Ty> {
        self.node_kind(node, env)
            .map_err(|r| r.located(node.span().start()))
    }
    fn node_kind(&mut self, node: &Node, env: &mut BTreeMap<String, Ty>) -> Check<Ty> {
        match node {
            Node::Lit(Literal::Int(_), _) => Ok(Ty::I64),
            Node::Lit(Literal::Float(_), _) => Ok(Ty::F64),
            Node::Lit(Literal::Ident(name), _) => env
                .get(name)
                .copied()
                .ok_or(LaneProfileRejection::new("lane.source.unbound")),
            Node::Lit(Literal::Str(_), _) => reject("lane.source.string"),
            Node::Paren(inner, _) => self.node(inner, env),
            Node::Binary {
                op, left, right, ..
            } => binary(&ast_bin(*op), self.node(left, env)?, self.node(right, env)?),
            Node::Bitwise { left, right, .. } | Node::Logical { left, right, .. } => {
                equal(self.node(left, env)?, Ty::I64)?;
                equal(self.node(right, env)?, Ty::I64)?;
                Ok(Ty::I64)
            }
            Node::Neg { operand, .. } => {
                let ty = value(self.node(operand, env)?);
                // Runtime f64 negation currently lowers through an integer
                // zero (eval/lower.rs), losing a uniformly typed subtraction.
                // Literal negation folds directly to ConstF64 and is safe here.
                if ty == Ty::F64 && !matches!(operand.as_ref(), Node::Lit(Literal::Float(_), _)) {
                    return reject("lane.source.float_neg_lowering");
                }
                if matches!(ty, Ty::I64 | Ty::F64) {
                    Ok(ty)
                } else {
                    reject("lane.source.neg_type")
                }
            }
            Node::Not { operand, .. } | Node::BitNot { operand, .. } => {
                equal(self.node(operand, env)?, Ty::I64)?;
                Ok(Ty::I64)
            }
            Node::Let {
                name,
                ann,
                value: init,
                ..
            } => {
                let ty = value(self.node(init, env)?);
                if let Some(ann) = ann {
                    equal(ty, scalar(ann)?)?;
                }
                if !matches!(ty, Ty::I64 | Ty::F64) {
                    return reject("lane.source.binding_type");
                }
                env.insert(name.clone(), ty);
                Ok(Ty::Unit)
            }
            Node::Assign { name, value, .. } => {
                let old = *env
                    .get(name)
                    .ok_or(LaneProfileRejection::new("lane.source.unbound"))?;
                equal(self.node(value, env)?, old)?;
                Ok(Ty::Unit)
            }
            Node::Call { callee, args, .. } => {
                let types = args
                    .iter()
                    .map(|a| self.node(a, env))
                    .collect::<Check<Vec<_>>>()?;
                call_type(callee, &types, self.functions)
            }
            Node::Return { value, .. } => {
                let ty = match value {
                    Some(v) => self.node(v, env)?,
                    None => Ty::Unit,
                };
                equal(ty, self.result)?;
                Ok(Ty::Never)
            }
            Node::Block { stmts, .. } => self.block(stmts, &mut env.clone()),
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                equal(self.node(cond, env)?, Ty::I64)?;
                let a = self.block(then_branch, &mut env.clone())?;
                let b = match else_branch {
                    Some(b) => self.block(b, &mut env.clone())?,
                    None => Ty::Unit,
                };
                // Statement ifs can have a value on just one branch; both
                // values are discarded. Value contexts reject Unit downstream.
                if a == Ty::Unit || b == Ty::Unit {
                    Ok(Ty::Unit)
                } else {
                    unify(a, b)
                }
            }
            Node::While { cond, body, .. } => {
                equal(self.node(cond, env)?, Ty::I64)?;
                self.loops += 1;
                self.block(body, &mut env.clone())?;
                self.loops -= 1;
                Ok(Ty::Unit)
            }
            Node::For {
                var,
                start,
                end,
                body,
                attrs,
                ..
            } => {
                if !attrs.is_empty() {
                    return reject("lane.source.loop_attributes");
                }
                equal(self.node(start, env)?, Ty::I64)?;
                equal(self.node(end, env)?, Ty::I64)?;
                let mut local = env.clone();
                local.insert(var.clone(), Ty::I64);
                self.loops += 1;
                self.block(body, &mut local)?;
                self.loops -= 1;
                Ok(Ty::Unit)
            }
            Node::Break { .. } | Node::Continue { .. } => {
                if self.loops == 0 {
                    reject("lane.source.loop_exit_scope")
                } else {
                    Ok(Ty::Never)
                }
            }
            Node::As { expr, ty, .. } => {
                let from = value(self.node(expr, env)?);
                // Narrow source casts retain width semantics in subsequent
                // expressions, but legacy scalar IR erases that information.
                // Only full-width i64 <-> f64 conversions are admitted: the
                // marked compiler and MLIR both convert i64 -> f64 with IEEE
                // round-to-nearest-even and f64 -> i64 by SATURATING truncation
                // (NaN -> 0, out of range -> i64::MIN / i64::MAX). See
                // `conversion_type` for the IR side of the same pairs.
                match ty {
                    TypeAnn::ScalarI64 if matches!(from, Ty::I64 | Ty::F64 | Ty::Never) => {
                        Ok(Ty::I64)
                    }
                    TypeAnn::ScalarF64 if matches!(from, Ty::I64 | Ty::Never) => Ok(Ty::F64),
                    _ => reject("lane.source.cast"),
                }
            }
            // `print` lowers to the `printI64`/`printNewline`/`print_bytes`
            // runtime calls (src/eval/lower.rs, `Node::Print`). No native-lane
            // compiler implements them and the IR walker types none of them,
            // so admitting `print` here only moved its refusal to an unlocated
            // IR call. Refuse it at its source site instead.
            Node::Print { .. } => reject("lane.source.print"),
            Node::ExternBlock { .. } => reject("lane.extern.unimplemented"),
            Node::FnDef(..) => reject("lane.source.nested_function"),
            Node::Const { .. }
            | Node::Export { .. }
            | Node::Import { .. }
            | Node::Tuple { .. }
            | Node::CallGrad { .. }
            | Node::CallTensorSum { .. }
            | Node::CallTensorMean { .. }
            | Node::CallReshape { .. }
            | Node::CallExpandDims { .. }
            | Node::CallSqueeze { .. }
            | Node::CallTranspose { .. }
            | Node::CallIndex { .. }
            | Node::CallSlice { .. }
            | Node::CallSliceStride { .. }
            | Node::CallGather { .. }
            | Node::CallDot { .. }
            | Node::CallMatMul { .. }
            | Node::TensorMatmul { .. }
            | Node::TensorElemwise { .. }
            | Node::CallTensorRelu { .. }
            | Node::CallTensorRand { .. }
            | Node::CallTensorConv2d { .. }
            | Node::LetTuple { .. }
            | Node::ArrayLit { .. }
            | Node::MapLit { .. }
            | Node::SetLit { .. }
            | Node::ForEach { .. }
            | Node::MethodCall { .. }
            | Node::FieldAccess { .. }
            | Node::ExternConst { .. }
            | Node::TypeAlias { .. }
            | Node::StructDef { .. }
            | Node::EnumDef { .. }
            | Node::Assert { .. }
            | Node::StructLit { .. }
            | Node::IndexAccess { .. }
            | Node::SliceRange { .. }
            | Node::IndexAssign { .. }
            | Node::FieldAssign { .. }
            | Node::Match { .. }
            | Node::Try { .. }
            | Node::Ref { .. }
            | Node::Region { .. }
            | Node::Closure(..)
            | Node::TraitDef { .. }
            | Node::ImplBlock { .. } => reject("lane.source.unsupported"),
        }
    }
}

fn source_facts(source: &ast::Module) -> Check<Functions> {
    let mut functions = Functions::new();
    for node in &source.items {
        if let Node::FnDef(fd, _) = node {
            let at = node.span().start();
            let sig = (|| -> Check<Signature> {
                if fd.name.starts_with("__mind_")
                    || !fd.type_params.is_empty()
                    || fd.reap_threshold.is_some()
                    || !fd.attrs.is_empty()
                {
                    return reject("lane.source.function_contract");
                }
                Ok(Signature {
                    params: fd
                        .params
                        .iter()
                        .map(|p| scalar(&p.ty))
                        .collect::<Check<_>>()?,
                    result: fd
                        .ret_type
                        .as_ref()
                        .map(scalar)
                        .transpose()?
                        .unwrap_or(Ty::Unit),
                })
            })()
            .map_err(|r| r.in_function(&fd.name).located(at))?;
            if functions.insert(fd.name.clone(), sig).is_some() {
                return Err(LaneProfileRejection::new("lane.source.duplicate_function")
                    .in_function(&fd.name)
                    .located(at));
            }
        }
    }
    let mut globals = BTreeMap::new();
    for node in &source.items {
        if let Node::Const {
            name,
            ty,
            value,
            attrs,
            ..
        } = node
        {
            (|| -> Check<()> {
                if !attrs.is_empty() {
                    return reject("lane.source.const_attributes");
                }
                let actual = Source {
                    functions: &functions,
                    result: Ty::Unit,
                    loops: 0,
                }
                .node(value, &mut globals.clone())?;
                if let Some(ty) = ty {
                    equal(actual, scalar(ty)?)?;
                }
                if !matches!(actual, Ty::I64 | Ty::F64)
                    || globals.insert(name.clone(), actual).is_some()
                {
                    return reject("lane.source.const_type");
                }
                Ok(())
            })()
            .map_err(|r| r.located(node.span().start()))?;
        }
    }
    for node in &source.items {
        let checked: Check<()> = match node {
            Node::FnDef(fd, _) => {
                let sig = &functions[&fd.name];
                let mut env = globals.clone();
                let mut names = BTreeSet::new();
                (|| -> Check<()> {
                    for (p, ty) in fd.params.iter().zip(&sig.params) {
                        if !names.insert(&p.name) {
                            return reject("lane.source.duplicate_parameter");
                        }
                        env.insert(p.name.clone(), *ty);
                    }
                    let got = Source {
                        functions: &functions,
                        result: sig.result,
                        loops: 0,
                    }
                    .block(&fd.body, &mut env)?;
                    equal(got, sig.result)
                })()
                .map_err(|r| r.in_function(&fd.name))
            }
            Node::Const { .. } => Ok(()),
            // A plain project import names a module the bridge's resolver
            // already linked into this image; it has no runtime meaning,
            // and the frozen parser skips the `;`-terminated item (the
            // bridge refuses an unterminated one). Calls still bind only to
            // functions in this one image, whose names must be unique.
            // An `as` rename would make call spelling module-relative:
            // refuse it.
            Node::Import { alias: None, .. } => Ok(()),
            Node::Export {
                names, category, ..
            } if category.is_none() && names.iter().all(|n| functions.contains_key(n)) => Ok(()),
            Node::ExternBlock { .. } => reject("lane.extern.unimplemented"),
            _ => reject("lane.source.top_level"),
        };
        checked.map_err(|r| r.located(node.span().start()))?;
    }
    Ok(functions)
}

/// Source-first entry used before lowering. Mandatory again in combined check.
pub fn profile_lane_admits_source(source: &ast::Module) -> Check<()> {
    source_facts(source).map(|_| ())
}

fn get(env: &Values, id: &ValueId) -> Check<Ty> {
    env.get(id)
        .copied()
        .ok_or(LaneProfileRejection::new("lane.ir.untyped_value"))
}
fn put(env: &mut Values, id: ValueId, ty: Ty) -> Check<()> {
    if let Some(old) = env.insert(id, ty) {
        equal(ty, old)?;
    }
    Ok(())
}

struct Ir<'a> {
    functions: &'a Functions,
    result: Ty,
    loops: Vec<BTreeMap<String, Ty>>,
    params: &'a [(String, ValueId)],
}
impl Ir<'_> {
    fn walk(&mut self, body: &[Instr], env: &mut Values) -> Check<()> {
        for instr in body {
            match instr {
                Instr::ConstI64(id, _) => put(env, *id, Ty::I64)?,
                Instr::ConstF64(id, _) => put(env, *id, Ty::F64)?,
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
                    let ty = match conversion_type(name, &args) {
                        Some(ty) => ty,
                        None => call_type(name, &args, self.functions)?,
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
                Instr::ExternFnDecl { .. } => return reject("lane.extern.unimplemented"),
                Instr::ConstArray { .. }
                | Instr::ArrayLoad { .. }
                | Instr::ArrayStore { .. }
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
                | Instr::VecReduceAddI64 { .. } => return reject("lane.ir.unsupported"),
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
pub fn profile_lane_admits(source: &ast::Module, ir: &IRModule) -> Check<()> {
    let functions = source_facts(source)?;
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
                    || !value_types.is_empty()
                {
                    return reject("lane.ir.function_binding");
                }
                let mut env = Values::new();
                for ((_, id), ty) in params.iter().zip(&sig.params) {
                    put(&mut env, *id, *ty)?;
                }
                Ir {
                    functions: &functions,
                    result: sig.result,
                    loops: vec![],
                    params,
                }
                .walk(body, &mut env)?;
                if let Some(ret) = ret_id {
                    equal(get(&env, ret)?, sig.result)?;
                }
                if let Some(meta) = semantic_types {
                    for (id, ty) in meta.values() {
                        use crate::types::{ScalarType, SemanticType};
                        let expected = match ty {
                            SemanticType::Scalar(ScalarType::I64 | ScalarType::Bool) => Ty::I64,
                            SemanticType::Scalar(ScalarType::F64) => Ty::F64,
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
                functions: &functions,
                result: Ty::Unit,
                loops: vec![],
                params: &[],
            }
            .walk(std::slice::from_ref(instr), &mut globals)?;
        }
    }
    if seen.len() != functions.len() {
        return reject("lane.ir.function_binding");
    }
    Ok(())
}
