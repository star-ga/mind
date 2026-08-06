// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Closure desugaring (#267 Phase 1) — the byte-identity-safe, monomorphic form.
//!
//! A `|caps; params| -> R { body }` closure is rewritten, BEFORE IR emit, into
//! ordinary constructs that already round-trip through the mic@3 codec — a
//! synthesized env `struct`, a top-level `fn`, and a `struct`-literal — so there
//! is NO wire-format change and NO `MIC3_VERSION` bump (the same principle as the
//! `for`/`match` desugars). Determinism comes for free: only fns/structs/calls
//! reach `emit_mic3`, and every synthesized name/field is keyed by `span.start()`
//! in SOURCE ORDER (never a hashmap walk), so the emitted bytes are deterministic
//! and cross-substrate identical.
//!
//! # Phase 1 support boundary (fail-closed outside it)
//! - EXACTLY one capture and one param (multi-arity is Phase 2, rejected here);
//! - capture-by-value **i64** (the env field type is fixed to `i64`);
//! - a closure **bound** via `let f = <closure>` at fn-statement position;
//! - **applied** by a DIRECT call `f(args)` in the same fn scope, monomorphized to
//!   a direct call `__cl_{span}(env, args)` — no fn-pointer, no dispatch (#268).
//!
//! A closure body / enclosing statement of ANY node kind is handled: the capture
//! and closure-call rewriters walk an EXHAUSTIVE child-visitor (`node_children_mut`,
//! no `_ => {}`), so a captured ident or a direct `f(x)` inside a `match`/`while`/
//! `Try`/… position is rewritten too (never silently left to resolve to a global).
//!
//! Any other FORM (multi-arity, a closure passed as an argument, returned, stored,
//! or a param that shadows a capture) is REJECTED fail-closed with an E2600
//! diagnostic; and the `lower_expr` terminal arm panics on any `Closure` that still
//! slips through, so an unsupported form can never become a silent miscompile.
//!
//! Deferred to a later slice: the `apply(f, x)`-style indirection (specializing a
//! separate closure-taking fn), multi-/non-i64 captures, and the self-host
//! `main.mind` mirror. `main.mind` has zero closures, so the keystone stays
//! byte-identical.
//! // deferred: apply()-indirection + non-i64/multi capture + self-host mirror —
//! // upgrade path: RFC #267 Phase 2/3 (monomorphize a closure-param fn per env
//! // type; thread the capture DType through instead of assuming i64).

use crate::ast::{Field, FnDefData, Literal, Module, Node, Param, Span, StructLitField, TypeAnn};
use crate::diagnostics::{Diagnostic, Span as DiagSpan};

/// Desugar every supported closure in `module` in place, appending the
/// synthesized env structs + closure fns as new top-level items. Returns any
/// fail-closed diagnostics (an unsupported closure form). A module with no
/// closure is left byte-identical (early no-op — no clone, no allocation).
pub fn desugar_closures(module: &mut Module, source: &str, file: Option<&str>) -> Vec<Diagnostic> {
    if !module.items.iter().any(node_has_closure) {
        return Vec::new();
    }
    let mut synthesized: Vec<Node> = Vec::new();
    let mut diags: Vec<Diagnostic> = Vec::new();
    for item in module.items.iter_mut() {
        if let Node::FnDef(fd, _) = item {
            desugar_fn_body(&mut fd.body, &mut synthesized, &mut diags, source, file);
        }
    }
    // Append synthesized items in encounter (span) order — deterministic, and
    // existing item positions are untouched. Fns resolve globally (one link
    // unit), so definition order does not affect calls.
    module.items.extend(synthesized);
    // Fail-closed backstop: any `Closure` node that survived is an unsupported
    // Phase-1 form. Reject fail-loud rather than let it reach lowering (which
    // would panic) — a clean diagnostic is better UX.
    for item in &module.items {
        if let Some(sp) = find_closure_span(item) {
            diags.push(unsupported_diag(source, file, sp));
        }
    }
    diags
}

/// Desugar the closure bindings + direct closure-calls in one fn body.
fn desugar_fn_body(
    body: &mut [Node],
    synthesized: &mut Vec<Node>,
    diags: &mut Vec<Diagnostic>,
    source: &str,
    file: Option<&str>,
) {
    // let-name -> synthesized `__cl_{span}` fn name, in source order.
    let mut closure_fns: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for stmt in body.iter_mut() {
        // `let f = |caps; params| ...` — the sole Phase-1 binding shape.
        if let Node::Let { name, value, .. } = stmt {
            if let Node::Closure(..) = value.as_ref() {
                match build_closure_items(value, synthesized, source, file) {
                    Ok((fn_name, struct_lit)) => {
                        closure_fns.insert(name.clone(), fn_name);
                        // Write through the existing box (reuses the allocation).
                        **value = struct_lit;
                    }
                    Err(d) => diags.push(d),
                }
                continue;
            }
        }
        // Every other statement: monomorphize any direct call `f(args)` whose
        // callee is a closure-local into `__cl_{span}(f, args)`.
        rewrite_closure_calls(stmt, &closure_fns);
    }
}

/// Build the env `struct` + `__cl_{span}` fn for one closure, push both to
/// `synthesized`, and return `(fn_name, env_struct_literal)` — the literal that
/// replaces the closure expression at its `let` binding.
// `Diagnostic` is large by value, but this is a cold fail-closed error path
// (one allocation only when a closure is rejected), so boxing it is not worth
// the churn at every `?`.
#[allow(clippy::result_large_err)]
fn build_closure_items(
    closure: &Node,
    synthesized: &mut Vec<Node>,
    source: &str,
    file: Option<&str>,
) -> Result<(String, Node), Diagnostic> {
    let (captures, params, ret_type, cbody, span) = match closure {
        Node::Closure(data, span) => (
            &data.captures,
            &data.params,
            &data.ret_type,
            &data.body,
            *span,
        ),
        _ => unreachable!("build_closure_items called on non-closure"),
    };
    // Phase-1 support boundary — EXACTLY one capture and one param. Multi-capture
    // / multi-param is Phase 2; accepting it here would (a) exceed the tested
    // contract and (b) rely on the i64-only env-field assumption for more slots.
    // Fail closed (E2600) rather than silently compile an untested shape.
    if captures.len() != 1 || params.len() != 1 {
        return Err(unsupported_diag(source, file, span));
    }

    let s = span.start();
    let env_type = format!("__clenv_{s}");
    let fn_name = format!("__cl_{s}");
    // Unbindable env value-param name (`;`-free but span-keyed + `self` marker,
    // matching the `__for_i_{span}` / `__match_scrut_{span}` synthesized-name
    // convention a user ident cannot collide with in practice).
    let env_param = format!("__clenv_self_{s}");

    // Adversarial soundness: a param that shadows a capture makes the in-body
    // name ambiguous (which one does `k` mean?) — reject fail-closed.
    for p in params {
        if captures.iter().any(|c| c == &p.name) {
            return Err(unsupported_diag(source, file, span));
        }
    }

    // Env struct fields = captures in SOURCE ORDER, each i64 (Phase 1).
    // deferred: the capture DType is fixed to i64 (Phase-1 boundary). A non-i64
    // capture is NOT rejected here because the bare `|k; ...|` capture syntax
    // carries no type annotation, so the DType is unknown pre-typecheck; the
    // synthesized `__clenv { k: k }` field is i64 and a genuine i64 capture is
    // exact. upgrade path (Phase 2): admit `|k: T; ...|` typed captures, set the
    // env field to T, and reject any T outside the supported set with E2600.
    let fields: Vec<Field> = captures
        .iter()
        .map(|c| Field {
            is_pub: false,
            name: c.clone(),
            ty: TypeAnn::ScalarI64,
            span,
        })
        .collect();
    let struct_def = Node::StructDef {
        is_pub: false,
        name: env_type.clone(),
        fields,
        attrs: Vec::new(),
        span,
    };

    // Fn params: env first, then the closure's own params.
    let mut fn_params: Vec<Param> = Vec::with_capacity(params.len() + 1);
    fn_params.push(Param {
        name: env_param.clone(),
        ty: TypeAnn::Named(env_type.clone()),
        span,
    });
    fn_params.extend(params.iter().cloned());

    // Fn body: closure body with each captured name rewritten to
    // `env.<capture>` (a FieldAccess), and the trailing value wrapped in an
    // explicit `Return`.
    let capture_set: std::collections::BTreeSet<&str> =
        captures.iter().map(|s| s.as_str()).collect();
    let mut fn_body: Vec<Node> = cbody.clone();
    for n in fn_body.iter_mut() {
        rewrite_captures(n, &capture_set, &env_param);
    }
    wrap_last_in_return(&mut fn_body, span);

    let fn_def = Node::FnDef(
        Box::new(FnDefData {
            is_pub: false,
            is_test: false,
            name: fn_name.clone(),
            type_params: Vec::new(),
            params: fn_params,
            ret_type: ret_type.clone(),
            body: fn_body,
            reap_threshold: None,
            attrs: Vec::new(),
        }),
        span,
    );

    synthesized.push(struct_def);
    synthesized.push(fn_def);

    // The env struct literal that replaces the closure at its binding site:
    // `__clenv_{s} { cap: cap, ... }` — each field reads the captured variable
    // from the enclosing scope by name.
    let lit_fields: Vec<StructLitField> = captures
        .iter()
        .map(|c| StructLitField {
            name: c.clone(),
            value: Node::Lit(Literal::Ident(c.clone()), span),
            span,
        })
        .collect();
    let struct_lit = Node::StructLit {
        name: env_type,
        fields: lit_fields,
        span,
    };
    Ok((fn_name, struct_lit))
}

/// Rewrite every `Node::Lit(Ident(name))` where `name` is a capture into
/// `env_param.name` (a FieldAccess), recursing through expression positions.
/// Does NOT descend into a nested `Closure` (Phase 1 has none; a surviving one
/// is caught by the fail-closed scan / lowering panic).
fn rewrite_captures(node: &mut Node, captures: &std::collections::BTreeSet<&str>, env_param: &str) {
    // Replace a bare captured ident with `env.field`.
    if let Node::Lit(Literal::Ident(name), sp) = node {
        if captures.contains(name.as_str()) {
            let field = name.clone();
            let span = *sp;
            *node = Node::FieldAccess {
                receiver: Box::new(Node::Lit(Literal::Ident(env_param.to_string()), span)),
                field,
                span,
            };
            return;
        }
    }
    for child in node_children_mut(node) {
        rewrite_captures(child, captures, env_param);
    }
}

/// Rewrite a direct closure-application `f(args)` (callee is a closure-local)
/// into `__cl_{span}(f, args)`, recursing through expression positions.
fn rewrite_closure_calls(
    node: &mut Node,
    closure_fns: &std::collections::BTreeMap<String, String>,
) {
    if let Node::Call { callee, args, span } = node {
        // Rewrite args first (they may themselves contain closure calls).
        for a in args.iter_mut() {
            rewrite_closure_calls(a, closure_fns);
        }
        if let Some(fn_name) = closure_fns.get(callee) {
            let env_ident = Node::Lit(Literal::Ident(callee.clone()), *span);
            let mut new_args = Vec::with_capacity(args.len() + 1);
            new_args.push(env_ident);
            new_args.append(args);
            *node = Node::Call {
                callee: fn_name.clone(),
                args: new_args,
                span: *span,
            };
        }
        return;
    }
    for child in node_children_mut(node) {
        rewrite_closure_calls(child, closure_fns);
    }
}

/// Wrap the final statement of a body in an explicit `Return` unless it already
/// is one (or the body is empty). MIND fns require an explicit return value;
/// this keeps the synthesized closure fn's semantics identical to the closure
/// body's trailing expression.
fn wrap_last_in_return(body: &mut Vec<Node>, span: Span) {
    let Some(last) = body.pop() else {
        return;
    };
    match last {
        Node::Return { .. } => body.push(last),
        other => body.push(Node::Return {
            value: Some(Box::new(other)),
            span,
        }),
    }
}

/// Mutable iterator over the direct child `Node`s of `node` reachable in
/// expression / statement positions, used by [`rewrite_captures`] and
/// [`rewrite_closure_calls`].
///
/// This match is DELIBERATELY EXHAUSTIVE (no `_ => {}`). A wildcard here was a
/// silent-miscompile: a captured ident or a direct `f(x)` call inside a
/// `match`/`while`/`Try`/… position that the walker skipped was never rewritten,
/// resolving instead to a module global — with no `Closure` node left for the
/// survivor scan or the `lower_expr` panic to catch. Keeping it exhaustive makes
/// rustc FORCE every future `Node` variant to be classified here, so that bug
/// class cannot recur.
///
/// `FnDef` / `Closure` bodies are treated as LEAVES: they open a different scope,
/// so the capture/call rewrite must not descend into them (a nested fn/closure
/// cannot see this closure's captures). The survivor scan reaches their bodies
/// separately in [`find_closure_span`].
fn node_children_mut(node: &mut Node) -> Vec<&mut Node> {
    let mut out: Vec<&mut Node> = Vec::new();
    match node {
        // ── Binary-shaped ────────────────────────────────────────────────
        Node::Binary { left, right, .. }
        | Node::Logical { left, right, .. }
        | Node::Bitwise { left, right, .. } => {
            out.push(left);
            out.push(right);
        }
        // ── Single inner expression ──────────────────────────────────────
        Node::Paren(inner, _)
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. }
        | Node::Ref { inner, .. }
        | Node::As { expr: inner, .. }
        | Node::Try { inner, .. } => out.push(inner),
        // ── Single value-bearing statements ──────────────────────────────
        Node::Let { value, .. }
        | Node::Assign { value, .. }
        | Node::LetTuple { value, .. }
        | Node::Const { value, .. } => out.push(value),
        Node::Return { value: Some(v), .. } => out.push(v),
        Node::Return { value: None, .. } => {}
        // ── Vec-of-Node holders ──────────────────────────────────────────
        Node::Call { args, .. }
        | Node::Print { args, .. }
        | Node::Tuple { elements: args, .. }
        | Node::ArrayLit { elements: args, .. }
        | Node::SetLit { elements: args, .. } => out.extend(args.iter_mut()),
        Node::MethodCall { receiver, args, .. } => {
            out.push(receiver);
            out.extend(args.iter_mut());
        }
        Node::FieldAccess { receiver, .. } => out.push(receiver),
        Node::FieldAssign {
            receiver, value, ..
        } => {
            out.push(receiver);
            out.push(value);
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            out.push(receiver);
            out.push(index);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            out.push(receiver);
            out.push(index);
            out.push(value);
        }
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            out.push(receiver);
            out.push(start);
            out.push(end);
        }
        Node::StructLit { fields, .. } => out.extend(fields.iter_mut().map(|f| &mut f.value)),
        Node::MapLit { entries, .. } => {
            for (k, v) in entries.iter_mut() {
                out.push(k);
                out.push(v);
            }
        }
        Node::Block { stmts, .. } => out.extend(stmts.iter_mut()),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            out.push(cond);
            out.extend(then_branch.iter_mut());
            if let Some(eb) = else_branch {
                out.extend(eb.iter_mut());
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            out.push(start);
            out.push(end);
            out.extend(body.iter_mut());
        }
        Node::ForEach {
            collection, body, ..
        } => {
            out.push(collection);
            out.extend(body.iter_mut());
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            out.push(scrutinee);
            for arm in arms.iter_mut() {
                if let Some(g) = &mut arm.guard {
                    out.push(g);
                }
                out.push(&mut arm.body);
            }
        }
        Node::Assert { cond, .. } => out.push(cond),
        // ── Tensor / autodiff ops carrying inner expressions ─────────────
        Node::CallGrad { loss, .. } => out.push(loss),
        Node::CallTensorSum { x, .. }
        | Node::CallTensorMean { x, .. }
        | Node::CallReshape { x, .. }
        | Node::CallExpandDims { x, .. }
        | Node::CallSqueeze { x, .. }
        | Node::CallTranspose { x, .. }
        | Node::CallIndex { x, .. }
        | Node::CallSlice { x, .. }
        | Node::CallSliceStride { x, .. }
        | Node::CallTensorRelu { x, .. } => out.push(x),
        Node::CallGather { x, idx, .. } => {
            out.push(x);
            out.push(idx);
        }
        Node::CallTensorConv2d { x, w, .. } => {
            out.push(x);
            out.push(w);
        }
        Node::CallDot { a, b, .. } | Node::CallMatMul { a, b, .. } => {
            out.push(a);
            out.push(b);
        }
        Node::TensorMatmul { lhs, rhs, .. } | Node::TensorElemwise { lhs, rhs, .. } => {
            out.push(lhs);
            out.push(rhs);
        }
        // ── std-surface control-flow / region ────────────────────────────
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            out.push(cond);
            out.extend(body.iter_mut());
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => out.extend(body.iter_mut()),
        // ── Leaves / different-scope boundaries (nothing to rewrite) ─────
        Node::Lit(..)
        | Node::FnDef(..)
        | Node::Closure(..)
        | Node::TraitDef { .. }
        | Node::ImplBlock { .. }
        | Node::Import { .. }
        | Node::ExternConst { .. }
        | Node::TypeAlias { .. }
        | Node::Export { .. }
        | Node::StructDef { .. }
        | Node::EnumDef { .. }
        | Node::ExternBlock { .. }
        | Node::CallTensorRand { .. } => {}
        #[cfg(feature = "std-surface")]
        Node::Break { .. } | Node::Continue { .. } => {}
    }
    out
}

/// Recursively test whether `node` contains any `Closure` (used for the
/// module-level no-op fast path and, per-item, the fail-closed survivor scan).
fn node_has_closure(node: &Node) -> bool {
    find_closure_span(node).is_some()
}

/// Return the span of the FIRST `Closure` node found anywhere under `node`
/// (pre-order), or `None`. Recurses through both the mutable-walk child set and
/// the extra bodies that set omits (FnDef/While/closure bodies), so a survivor
/// in any position is detected.
fn find_closure_span(node: &Node) -> Option<Span> {
    if let Node::Closure(_, span) = node {
        return Some(*span);
    }
    // FnDef / closure bodies are not in `node_children` (immutable variant):
    // scan them explicitly.
    match node {
        Node::FnDef(fd, _) => {
            for n in &fd.body {
                if let Some(sp) = find_closure_span(n) {
                    return Some(sp);
                }
            }
        }
        Node::Closure(data, _) => {
            for n in &data.body {
                if let Some(sp) = find_closure_span(n) {
                    return Some(sp);
                }
            }
        }
        _ => {}
    }
    // Mirror `node_children_mut` immutably.
    for child in node_children_ref(node) {
        if let Some(sp) = find_closure_span(child) {
            return Some(sp);
        }
    }
    None
}

/// Immutable twin of [`node_children_mut`] — same EXHAUSTIVE classification (no
/// `_ => {}`), used by the `Closure`-survivor scan / no-op fast path.
pub(crate) fn node_children_ref(node: &Node) -> Vec<&Node> {
    let mut out: Vec<&Node> = Vec::new();
    match node {
        Node::Binary { left, right, .. }
        | Node::Logical { left, right, .. }
        | Node::Bitwise { left, right, .. } => {
            out.push(left);
            out.push(right);
        }
        Node::Paren(inner, _)
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. }
        | Node::Ref { inner, .. }
        | Node::As { expr: inner, .. }
        | Node::Try { inner, .. } => out.push(inner),
        Node::Let { value, .. }
        | Node::Assign { value, .. }
        | Node::LetTuple { value, .. }
        | Node::Const { value, .. } => out.push(value),
        Node::Return { value: Some(v), .. } => out.push(v),
        Node::Return { value: None, .. } => {}
        Node::Call { args, .. }
        | Node::Print { args, .. }
        | Node::Tuple { elements: args, .. }
        | Node::ArrayLit { elements: args, .. }
        | Node::SetLit { elements: args, .. } => out.extend(args.iter()),
        Node::MethodCall { receiver, args, .. } => {
            out.push(receiver);
            out.extend(args.iter());
        }
        Node::FieldAccess { receiver, .. } => out.push(receiver),
        Node::FieldAssign {
            receiver, value, ..
        } => {
            out.push(receiver);
            out.push(value);
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            out.push(receiver);
            out.push(index);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            out.push(receiver);
            out.push(index);
            out.push(value);
        }
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            out.push(receiver);
            out.push(start);
            out.push(end);
        }
        Node::StructLit { fields, .. } => out.extend(fields.iter().map(|f| &f.value)),
        Node::MapLit { entries, .. } => {
            for (k, v) in entries.iter() {
                out.push(k);
                out.push(v);
            }
        }
        Node::Block { stmts, .. } => out.extend(stmts.iter()),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            out.push(cond);
            out.extend(then_branch.iter());
            if let Some(eb) = else_branch {
                out.extend(eb.iter());
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            out.push(start);
            out.push(end);
            out.extend(body.iter());
        }
        Node::ForEach {
            collection, body, ..
        } => {
            out.push(collection);
            out.extend(body.iter());
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            out.push(scrutinee);
            for arm in arms.iter() {
                if let Some(g) = &arm.guard {
                    out.push(g);
                }
                out.push(&arm.body);
            }
        }
        Node::Assert { cond, .. } => out.push(cond),
        Node::CallGrad { loss, .. } => out.push(loss),
        Node::CallTensorSum { x, .. }
        | Node::CallTensorMean { x, .. }
        | Node::CallReshape { x, .. }
        | Node::CallExpandDims { x, .. }
        | Node::CallSqueeze { x, .. }
        | Node::CallTranspose { x, .. }
        | Node::CallIndex { x, .. }
        | Node::CallSlice { x, .. }
        | Node::CallSliceStride { x, .. }
        | Node::CallTensorRelu { x, .. } => out.push(x),
        Node::CallGather { x, idx, .. } => {
            out.push(x);
            out.push(idx);
        }
        Node::CallTensorConv2d { x, w, .. } => {
            out.push(x);
            out.push(w);
        }
        Node::CallDot { a, b, .. } | Node::CallMatMul { a, b, .. } => {
            out.push(a);
            out.push(b);
        }
        Node::TensorMatmul { lhs, rhs, .. } | Node::TensorElemwise { lhs, rhs, .. } => {
            out.push(lhs);
            out.push(rhs);
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            out.push(cond);
            out.extend(body.iter());
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => out.extend(body.iter()),
        Node::Lit(..)
        | Node::FnDef(..)
        | Node::Closure(..)
        | Node::TraitDef { .. }
        | Node::ImplBlock { .. }
        | Node::Import { .. }
        | Node::ExternConst { .. }
        | Node::TypeAlias { .. }
        | Node::Export { .. }
        | Node::StructDef { .. }
        | Node::EnumDef { .. }
        | Node::ExternBlock { .. }
        | Node::CallTensorRand { .. } => {}
        #[cfg(feature = "std-surface")]
        Node::Break { .. } | Node::Continue { .. } => {}
    }
    out
}

/// Fail-closed diagnostic for an unsupported Phase-1 closure form.
fn unsupported_diag(source: &str, file: Option<&str>, span: Span) -> Diagnostic {
    let dspan = DiagSpan::from_offsets(source, span.start(), span.end(), file);
    Diagnostic::error(
        "closure",
        "E2600",
        "unsupported closure form (#267 Phase 1)",
    )
    .with_span(dspan)
    .with_help(
        "Phase 1 supports a single-param closure that captures i64 values, bound \
         via `let f = |caps; x: i64| -> R { body }` and applied by a DIRECT call \
         `f(args)` in the same fn scope. Passing a closure as an argument, \
         returning it, capturing a non-i64, or a param shadowing a capture is not \
         yet supported.",
    )
}
