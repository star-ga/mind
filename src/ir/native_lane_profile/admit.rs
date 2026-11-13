// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Source admission walk for the candidate lane profile (the IR leg is
//! `admit_ir`).

use super::admit_extern;
use super::types::{
    Callees, Check, Functions, LaneProfileRejection, Signature, Ty, ast_bin, binary, call_type,
    equal, reject, unify, value,
};
use super::{alloc_size, arrays, names};
use crate::ast::{self, Literal, Node, TypeAnn};
use std::collections::{BTreeMap, BTreeSet};

struct Source<'a> {
    callees: Callees<'a>,
    result: Ty,
    loops: usize,
    loop_calls: &'a mut BTreeSet<String>,
    /// Proven-positive allocation sizes are required of the user image
    /// (`alloc_size`). A seed body is the shipped blob and is typed, not
    /// re-proved.
    prove_alloc_size: bool,
    /// Names the function assigns: they carry no positivity fact.
    assigned: BTreeSet<String>,
}
impl Source<'_> {
    /// `print(a, b, ..)`: each argument is a string literal, an i64 or a bool
    /// (examples/mindc_mind/nb_print.mind writes the same bytes as the MLIR
    /// runtime). The native code evaluates every argument before its first
    /// write, MLIR prints each one after its own evaluation, so an argument
    /// after the first must have no effect: a literal, a name, `-` of one.
    /// The runtime helper names must not have a user or extern definition.
    fn print_statement(&mut self, args: &[Node], env: &mut BTreeMap<String, Ty>) -> Check<Ty> {
        if !self.callees.print {
            return reject("lane.source.print");
        }
        let helpers = ["printI64", "printNewline", "print_bytes"];
        if helpers
            .iter()
            .any(|n| self.callees.user.contains_key(*n) || self.callees.externs.contains_key(*n))
        {
            return reject("lane.source.print_collision");
        }
        fn effect_free(node: &Node) -> bool {
            match node {
                Node::Lit(Literal::Int(_) | Literal::Str(_) | Literal::Ident(_), _) => true,
                Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => effect_free(inner),
                _ => false,
            }
        }
        for (i, arg) in args.iter().enumerate() {
            let checked = if i > 0 && !effect_free(arg) {
                reject("lane.source.print_order")
            } else if matches!(arg, Node::Lit(Literal::Str(_), _)) {
                Ok(())
            } else {
                match self.node(arg, env)? {
                    Ty::I64 | Ty::Bool => Ok(()),
                    _ => reject("lane.source.print_argument"),
                }
            };
            checked.map_err(|r| r.located(arg.span().start()))?;
        }
        Ok(Ty::Unit)
    }
    fn block(&mut self, nodes: &[Node], env: &mut BTreeMap<String, Ty>) -> Check<Ty> {
        let mut result = Ty::Unit;
        for node in nodes {
            let next = self.node(node, env)?;
            if self.prove_alloc_size {
                alloc_size::learn(node, env, &self.assigned);
            }
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
                .ok_or_else(|| arrays::not_scalar(name, env)),
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
                let source_ty = self.node(operand, env)?;
                if source_ty == Ty::Bool {
                    return reject("lane.bool.width");
                }
                let ty = value(source_ty);
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
                let ty = self.node(operand, env)?;
                if ty == Ty::Bool {
                    return reject("lane.bool.width");
                }
                equal(ty, Ty::I64)?;
                Ok(Ty::I64)
            }
            Node::Let {
                name,
                ann,
                value: init,
                ..
            } => {
                if let Node::ArrayLit { elements, .. } = init.as_ref() {
                    alloc_size::forget(name, env);
                    arrays::bind(name, ann.as_ref(), elements, env)?;
                    return Ok(Ty::Unit);
                }
                let init_ty = self.node(init, env)?;
                let mut ty = if init_ty == Ty::Bool {
                    Ty::Bool
                } else {
                    value(init_ty)
                };
                if let Some(ann) = ann {
                    let annotated = names::scalar_named(ann)?;
                    equal(ty, annotated)?;
                    if annotated == Ty::Bool {
                        ty = Ty::Bool;
                    }
                }
                // A `let _` discard of an i32 result binds no name: no later
                // read finds it, so the narrow value never reaches i64 code.
                if name == "_" && ann.is_none() && ty == Ty::I32 {
                    env.remove(name);
                    arrays::shadow(name, env);
                    alloc_size::forget(name, env);
                    return Ok(Ty::Unit);
                }
                if !matches!(ty, Ty::I64 | Ty::Bool | Ty::F64) {
                    return reject("lane.source.binding_type");
                }
                env.insert(name.clone(), ty);
                arrays::shadow(name, env);
                alloc_size::forget(name, env);
                Ok(Ty::Unit)
            }
            Node::IndexAccess {
                receiver, index, ..
            } => {
                let index = self.node(index, env)?;
                arrays::read(receiver, index, env)
            }
            Node::Assign { name, value, .. } => {
                let old = *env.get(name).ok_or_else(|| arrays::not_scalar(name, env))?;
                equal(self.node(value, env)?, old)?;
                Ok(Ty::Unit)
            }
            Node::Call { callee, args, .. } => {
                let types = args
                    .iter()
                    .map(|a| self.node(a, env))
                    .collect::<Check<Vec<_>>>()?;
                let result = call_type(callee, &types, self.callees)?;
                // Validate the ordinary signature first so wrong types/arity keep
                // their existing diagnostics. A size not proven positive refuses
                // in the user image. The shipped seed blob is typed, not re-proved.
                if self.prove_alloc_size {
                    if let Some(size) = alloc_size::size_operand(callee, args) {
                        if !alloc_size::proven(size, env) {
                            return reject("lane.alloc.size_positive");
                        }
                    }
                }
                if self.loops > 0 {
                    // Native free does not reclaim arena storage. Do not admit
                    // repeated frees (including calls through helper functions)
                    // as evidence that repeated allocations fit in the arena.
                    if callee == "__mind_free" {
                        return reject("lane.alloc.free_in_loop");
                    }
                    if self.callees.user.contains_key(callee)
                        || self.callees.seed.contains_key(callee)
                    {
                        self.loop_calls.insert(callee.clone());
                    }
                }
                Ok(result)
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
                self.loops += 1;
                equal(self.node(cond, env)?, Ty::I64)?;
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
                arrays::shadow(var, &mut local);
                alloc_size::forget(var, &mut local);
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
                // The only float conversions are full-width i64 <-> f64: the
                // marked compiler and MLIR both convert i64 -> f64 with IEEE
                // round-to-nearest-even and f64 -> i64 by SATURATING truncation
                // (NaN -> 0, out of range -> i64::MIN / i64::MAX). An integer
                // cast to i32 / u32 / a pointer yields a value that no binding,
                // operator or return accepts, so it reaches only an extern
                // argument of that exact width or a further cast. See
                // `conversion_type` for the IR side of the same pairs.
                match ty {
                    TypeAnn::ScalarI64
                        if matches!(
                            from,
                            Ty::I64 | Ty::F64 | Ty::I32 | Ty::U32 | Ty::Ptr | Ty::Never
                        ) =>
                    {
                        Ok(Ty::I64)
                    }
                    TypeAnn::ScalarF64 if matches!(from, Ty::I64 | Ty::Never) => Ok(Ty::F64),
                    TypeAnn::ScalarI32
                        if matches!(from, Ty::I64 | Ty::I32 | Ty::U32 | Ty::Ptr | Ty::Never) =>
                    {
                        Ok(Ty::I32)
                    }
                    TypeAnn::ScalarU32
                        if matches!(from, Ty::I64 | Ty::I32 | Ty::U32 | Ty::Ptr | Ty::Never) =>
                    {
                        Ok(Ty::U32)
                    }
                    TypeAnn::RawPtr { .. }
                        if matches!(from, Ty::I64 | Ty::I32 | Ty::U32 | Ty::Ptr | Ty::Never) =>
                    {
                        Ok(Ty::Ptr)
                    }
                    _ => reject("lane.source.cast"),
                }
            }
            // `print` lowers to the `printI64`/`printNewline`/`print_bytes`
            // runtime calls (src/eval/lower.rs, `Node::Print`). Only a compiler
            // with the `nbprint1` mark lowers it natively; other compilers keep
            // the refusal at the source site (`print_statement`).
            Node::Print { args, .. } => self.print_statement(args, env),
            Node::ExternBlock { .. } => reject("lane.extern.nested"),
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
            | Node::SliceRange { .. }
            | Node::IndexAssign { .. }
            | Node::FieldAssign { .. }
            | Node::Match { .. }
            | Node::Try { .. }
            | Node::Ref { .. }
            | Node::Region { .. }
            | Node::Closure(..)
            | Node::TraitDef { .. }
            | Node::ImplBlock { .. } => Err(LaneProfileRejection::new(names::source(node))),
        }
    }
}

/// User functions, the admitted extern tier and every helper reached from a
/// loop (user or seed) of one module.
pub(super) struct Admitted {
    pub(super) user: Functions,
    pub(super) externs: Functions,
    pub(super) loop_calls: BTreeSet<String>,
    /// See `Callees::print`.
    pub(super) print: bool,
}

impl Admitted {
    /// The call-resolution chain of the user image over `seed`.
    pub(super) fn callees<'a>(&'a self, seed: &'a Functions) -> Callees<'a> {
        Callees {
            user: &self.user,
            externs: &self.externs,
            seed,
            print: self.print,
        }
    }
}

/// `#[protection]` with no arguments is a plane marker: it changes no codegen
/// and the native compiler skips it (main.mind attr_is_known_inert, all ten
/// bytes). Every other attribute stays `lane.source.function_contract`.
fn codegen_inert_attr(a: &ast::Attribute) -> bool {
    a.name == "protection" && a.args.is_empty()
}

/// The lane signature of one function definition, user or reached seed.
pub(super) fn signature(fd: &ast::FnDefData) -> Check<Signature> {
    if fd.name.starts_with("__mind_")
        || !fd.type_params.is_empty()
        || fd.reap_threshold.is_some()
        || !fd.attrs.iter().all(codegen_inert_attr)
    {
        return reject("lane.source.function_contract");
    }
    Ok(Signature {
        params: fd
            .params
            .iter()
            .map(|p| names::scalar_named(&p.ty))
            .collect::<Check<_>>()?,
        result: fd
            .ret_type
            .as_ref()
            .map(names::scalar_named)
            .transpose()?
            .unwrap_or(Ty::Unit),
    })
}

/// Walk one function body under `callees`: parameters over `globals`.
///
/// `loops` is the depth already entered by the caller (one, when this body
/// was reached from a loop). `loop_calls` collects helpers reached from there.
/// `prove_alloc_size` is set for the user image only.
pub(super) fn walk_function(
    fd: &ast::FnDefData,
    sig: &Signature,
    globals: &BTreeMap<String, Ty>,
    callees: Callees,
    loop_calls: &mut BTreeSet<String>,
    loops: usize,
    prove_alloc_size: bool,
) -> Check<()> {
    let mut env = globals.clone();
    let mut names = BTreeSet::new();
    for (p, ty) in fd.params.iter().zip(&sig.params) {
        if !names.insert(&p.name) {
            return reject("lane.source.duplicate_parameter");
        }
        env.insert(p.name.clone(), *ty);
        alloc_size::forget(&p.name, &mut env);
    }
    let mut walker = Source {
        callees,
        result: sig.result,
        loops,
        loop_calls,
        prove_alloc_size,
        assigned: alloc_size::assigned_names(&fd.body),
    };
    equal(walker.block(&fd.body, &mut env)?, sig.result)
}

/// User signatures and bodies; `seed` is the reached std seed tier.
///
/// The user image rechecks its own functions here; the seed walker rechecks
/// the rest of `loop_calls`.
/// `print` is the selected compiler's `nbprint1` mark (`Callees::print`).
pub(super) fn source_facts(source: &ast::Module, seed: &Functions, print: bool) -> Check<Admitted> {
    let mut functions = Functions::new();
    for node in &source.items {
        if let Node::FnDef(fd, _) = node {
            let at = node.span().start();
            let sig = signature(fd).map_err(|r| r.in_function(&fd.name).located(at))?;
            if functions.insert(fd.name.clone(), sig).is_some() {
                return Err(LaneProfileRejection::new("lane.source.duplicate_function")
                    .in_function(&fd.name)
                    .located(at));
            }
        }
    }
    let mut loop_calls = BTreeSet::new();
    let externs = admit_extern::collect_externs(&source.items, &functions)?;
    let callees = Callees {
        user: &functions,
        externs: &externs,
        seed,
        print,
    };
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
                    callees,
                    result: Ty::Unit,
                    loops: 0,
                    loop_calls: &mut loop_calls,
                    prove_alloc_size: true,
                    assigned: BTreeSet::new(),
                }
                .node(value, &mut globals.clone())?;
                if let Some(ty) = ty {
                    equal(actual, names::scalar_named(ty)?)?;
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
            Node::FnDef(fd, _) => walk_function(
                fd,
                &functions[&fd.name],
                &globals,
                callees,
                &mut loop_calls,
                0,
                true,
            )
            .map_err(|r| r.in_function(&fd.name)),
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
            } if category.is_none() && names.iter().all(|n| callees.user.contains_key(n)) => Ok(()),
            Node::ExternBlock { .. } => Ok(()),
            _ => reject("lane.source.top_level"),
        };
        checked.map_err(|r| r.located(node.span().start()))?;
    }
    // Keep established source refusals ahead of the entry ABI restriction.
    if functions
        .get("main")
        .is_some_and(|sig| sig.result == Ty::F64)
    {
        return Err(LaneProfileRejection::new("lane.entry.f64_result").in_function("main"));
    }
    // Close the set of helpers reachable from a loop. Rechecking with loop
    // context also covers mutual recursion without recursing in this walker.
    let mut checked = BTreeSet::new();
    while let Some(name) = loop_calls.iter().find(|n| !checked.contains(*n)).cloned() {
        checked.insert(name.clone());
        for node in &source.items {
            if let Node::FnDef(fd, _) = node {
                if fd.name != name {
                    continue;
                }
                walk_function(
                    fd,
                    &functions[&name],
                    &globals,
                    callees,
                    &mut loop_calls,
                    1,
                    true,
                )
                .map_err(|r| r.in_function(&name))?;
            }
        }
    }
    Ok(Admitted {
        user: functions,
        externs,
        loop_calls,
        print,
    })
}

/// Refuse a user `extern "C"` block unless the selected compiler carries the
/// `nbextc01` mark (`extern_fixed`): without it nothing can lower the call.
/// A `bind` clause is refused first, whatever the mark.
pub(super) fn extern_gate(source: &ast::Module, extern_fixed: bool) -> Check<()> {
    // Spec "Extern bindings": route A refuses each `bind` clause by name in
    // 0.10.3, before any other extern rule; the seed never sees the clause.
    let bound =
        |n: &Node| matches!(n, Node::ExternBlock { binding, .. } if *binding != Default::default());
    if source.items.iter().any(bound) {
        return reject("lane.extern.bind_clause");
    }
    let has_extern = source
        .items
        .iter()
        .any(|n| matches!(n, Node::ExternBlock { fns, .. } if !fns.is_empty()));
    if has_extern && !extern_fixed {
        return reject("lane.extern.compiler_unmarked");
    }
    Ok(())
}

/// Source-first entry used before lowering. Mandatory again in combined check.
/// `extern_fixed` is the selected compiler's `nbextc01` mark.
pub fn profile_lane_admits_source(source: &ast::Module, extern_fixed: bool) -> Check<()> {
    extern_gate(source, extern_fixed)?;
    source_facts(source, &Functions::new(), false).map(|_| ())
}
