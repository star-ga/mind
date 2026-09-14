// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Exhaustive AST child walkers shared by closure rewriting and survivor scans.
//!
//! Keeping this classification in a focused module makes adding AST variants
//! visible to both mutable and immutable traversals without growing the closure
//! desugar implementation.

use crate::ast::Node;

/// Mutable iterator over the direct child `Node`s of `node` reachable in
/// expression / statement positions, used by rewrite_captures and
/// rewrite_closure_calls.
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
/// separately in find_closure_span.
pub(crate) fn node_children_mut(node: &mut Node) -> Vec<&mut Node> {
    let mut out: Vec<&mut Node> = Vec::new();
    match node {
        // ── Binary-shaped ────────────────────────────────────────────────
        Node::Binary { left, right, .. } | Node::Logical { left, right, .. } => {
            out.push(left);
            out.push(right);
        }
        #[cfg(feature = "std-surface")]
        Node::Bitwise { left, right, .. } => {
            out.push(left);
            out.push(right);
        }
        // ── Single inner expression ──────────────────────────────────────
        Node::Paren(inner, _)
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. }
        | Node::Ref { inner, .. }
        | Node::Deref { operand: inner, .. }
        | Node::As { expr: inner, .. }
        | Node::Try { inner, .. } => out.push(inner),
        Node::DerefAssign { target, value, .. } => {
            out.push(target);
            out.push(value);
        }
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

/// Immutable twin of [`node_children_mut`] — same EXHAUSTIVE classification (no
/// `_ => {}`), used by the `Closure`-survivor scan / no-op fast path.
pub(crate) fn node_children_ref(node: &Node) -> Vec<&Node> {
    let mut out: Vec<&Node> = Vec::new();
    match node {
        Node::Binary { left, right, .. } | Node::Logical { left, right, .. } => {
            out.push(left);
            out.push(right);
        }
        #[cfg(feature = "std-surface")]
        Node::Bitwise { left, right, .. } => {
            out.push(left);
            out.push(right);
        }
        Node::Paren(inner, _)
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. }
        | Node::Ref { inner, .. }
        | Node::Deref { operand: inner, .. }
        | Node::As { expr: inner, .. }
        | Node::Try { inner, .. } => out.push(inner),
        Node::DerefAssign { target, value, .. } => {
            out.push(target);
            out.push(value);
        }
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
