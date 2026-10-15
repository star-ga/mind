// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Candidate scalar native delivery profile. This does not select a compiler,
//! change the historical frozen predicate, or establish seed qualification.
//! Source facts are mandatory: legacy IR erases scalar widths and annotations.
//! Only the closed i64/f64 scalar surface is admitted. C extern compatibility,
//! unsigned/narrow arithmetic, aggregates and implicit std dependencies refuse.

use crate::ast::{self, Literal, Node, TypeAnn};
use crate::ir::{BinOp, IRModule, Instr, ValueId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneProfileRejection {
    pub construct: &'static str,
}
type Check<T> = Result<T, LaneProfileRejection>;
fn reject<T>(construct: &'static str) -> Check<T> {
    Err(LaneProfileRejection { construct })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    I64,
    F64,
    Unit,
    Discard,
    Never,
}
#[derive(Clone, Debug)]
struct Signature {
    params: Vec<Ty>,
    result: Ty,
}
type Functions = BTreeMap<String, Signature>;
type Values = BTreeMap<ValueId, Ty>;

fn scalar(ann: &TypeAnn) -> Check<Ty> {
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
    intrinsic_signature(name).map(|s| s.params.len())
}

fn intrinsic_signature(name: &str) -> Option<Signature> {
    use Ty::{F64, I64};
    // Reuse canonical i64 ABI rows only for the explicitly supported names.
    // Preserve the canonical DiscardOnly compatibility policy. The delivery
    // emitter can materialize zero for value-position stores; that behavior
    // does not broaden this profile. Positioned read/write is delivery-specific.
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
                crate::intrinsic_contract::IntrinsicResult::DiscardOnly => Ty::Discard,
            },
        });
    }
    let (params, result) = match name {
        "__mind_free" | "__mind_load_i32" | "__mind_conv_i64" | "__mind_conv_i32" => {
            (vec![I64], I64)
        }
        "__mind_realloc" => (vec![I64, I64], I64),
        "__mind_store_i32" => (vec![I64, I64], Ty::Discard),
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

fn equal(actual: Ty, wanted: Ty) -> Check<()> {
    if actual == wanted || actual == Ty::Never {
        Ok(())
    } else {
        reject("lane.type.mismatch")
    }
}
fn unify(a: Ty, b: Ty) -> Check<Ty> {
    if a == Ty::Never {
        return Ok(b);
    }
    if b == Ty::Never || a == b {
        Ok(a)
    } else {
        reject("lane.type.branch_mismatch")
    }
}
fn call_type(name: &str, args: &[Ty], functions: &Functions) -> Check<Ty> {
    let intrinsic = intrinsic_signature(name);
    let sig = functions
        .get(name)
        .or(intrinsic.as_ref())
        .ok_or(LaneProfileRejection {
            construct: "lane.call.unknown",
        })?;
    if args.len() != sig.params.len() {
        return reject("lane.call.arity");
    }
    for (&actual, &expected) in args.iter().zip(&sig.params) {
        equal(actual, expected)?;
    }
    Ok(sig.result)
}
fn binary(op: &BinOp, left: Ty, right: Ty) -> Check<Ty> {
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
fn ast_bin(op: ast::BinOp) -> BinOp {
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
        match node {
            Node::Lit(Literal::Int(_), _) => Ok(Ty::I64),
            Node::Lit(Literal::Float(_), _) => Ok(Ty::F64),
            Node::Lit(Literal::Ident(name), _) => {
                env.get(name).copied().ok_or(LaneProfileRejection {
                    construct: "lane.source.unbound",
                })
            }
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
                let ty = self.node(operand, env)?;
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
                name, ann, value, ..
            } => {
                let ty = self.node(value, env)?;
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
                let old = *env.get(name).ok_or(LaneProfileRejection {
                    construct: "lane.source.unbound",
                })?;
                equal(self.node(value, env)?, old)?;
                Ok(Ty::Unit)
            }
            Node::Call { callee, args, .. } => {
                let types = args
                    .iter()
                    .map(|a| self.node(a, env))
                    .collect::<Check<Vec<_>>>()?;
                let result = call_type(callee, &types, self.functions)?;
                Ok(if result == Ty::Discard {
                    Ty::Unit
                } else {
                    result
                })
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
                let from = self.node(expr, env)?;
                // Narrow source casts retain width semantics in subsequent
                // expressions, but legacy scalar IR erases that information.
                match ty {
                    TypeAnn::ScalarI64 => {
                        equal(from, Ty::I64)?;
                        Ok(Ty::I64)
                    }
                    _ => reject("lane.source.cast"),
                }
            }
            Node::Print { args, .. } => {
                if args.len() != 1 {
                    return reject("lane.source.print_arity");
                }
                equal(self.node(&args[0], env)?, Ty::I64)?;
                Ok(Ty::Unit)
            }
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
            if fd.name.starts_with("__mind_")
                || !fd.type_params.is_empty()
                || fd.reap_threshold.is_some()
                || !fd.attrs.is_empty()
            {
                return reject("lane.source.function_contract");
            }
            let sig = Signature {
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
            };
            if functions.insert(fd.name.clone(), sig).is_some() {
                return reject("lane.source.duplicate_function");
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
        }
    }
    for node in &source.items {
        match node {
            Node::FnDef(fd, _) => {
                let sig = &functions[&fd.name];
                let mut env = globals.clone();
                let mut names = BTreeSet::new();
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
                equal(got, sig.result)?;
            }
            Node::Const { .. } => {}
            Node::Export {
                names, category, ..
            } if category.is_none() && names.iter().all(|n| functions.contains_key(n)) => {}
            Node::ExternBlock { .. } => return reject("lane.extern.unimplemented"),
            _ => return reject("lane.source.top_level"),
        }
    }
    Ok(functions)
}

/// Source-first entry used before lowering. Mandatory again in combined check.
pub fn profile_lane_admits_source(source: &ast::Module) -> Check<()> {
    source_facts(source).map(|_| ())
}

fn get(env: &Values, id: &ValueId) -> Check<Ty> {
    let ty = env.get(id).copied().ok_or(LaneProfileRejection {
        construct: "lane.ir.untyped_value",
    })?;
    if ty == Ty::Discard {
        reject("lane.ir.discard_result_use")
    } else {
        Ok(ty)
    }
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
                    let ty = call_type(name, &args, self.functions)?;
                    put(env, *dst, ty)?;
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
                    put(
                        env,
                        *dst,
                        unify(get(&a, then_result)?, get(&b, else_result)?)?,
                    )?;
                    for (id, left, right) in merges {
                        put(env, *id, unify(get(&a, left)?, get(&b, right)?)?)?;
                    }
                    for (_, id) in branch_bindings {
                        // A binding escaping a branch must be defined on both
                        // paths or be an explicit validated merge/dst.
                        if !env.contains_key(id) {
                            put(env, *id, unify(get(&a, id)?, get(&b, id)?)?)?;
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
                    let expected = self.loops.last().ok_or(LaneProfileRejection {
                        construct: "lane.ir.loop_exit_scope",
                    })?;
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

/// Require the source and its single lowering together. The caller owns their
/// provenance/binding and the exact compiler-capability selection; this function
/// is not a validator for arbitrary independent source/IR pairs or proof data.
pub fn profile_lane_admits(source: &ast::Module, ir: &IRModule) -> Check<()> {
    let functions = source_facts(source)?;
    if !ir.value_types.is_empty() {
        return reject("lane.ir.aggregate_metadata");
    }
    crate::ir::check_ssa_well_formed(ir).map_err(|_| LaneProfileRejection {
        construct: "lane.ir.ssa",
    })?;
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
            let sig = functions.get(name).ok_or(LaneProfileRejection {
                construct: "lane.ir.function_binding",
            })?;
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

#[cfg(test)]
mod tests;
