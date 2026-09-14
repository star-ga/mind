// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

/// Recurse into a node's children looking for statement Vecs that hold a
/// `continue` targeting the enclosing loop, splicing the loop `step` before each
/// (via [`super::inject_step_before_continue`]). The match is EXHAUSTIVE (no silent
/// catch-all) so a future `crate::ast::Node` variant forces a compile error here rather
/// than silently re-opening the infinite-loop hazard. Boundaries that are NOT
/// descended: a nested `while`/`for`/`for-each` BODY (its `continue`s target the
/// inner loop, stepped when that loop is lowered) and a nested `fn` body (a
/// different scope). A nested loop's header/range/collection expression IS
/// descended — a `continue` there still targets THIS loop.
#[cfg(feature = "std-surface")]
pub(super) fn descend_for_continue(node: &mut crate::ast::Node, step: &crate::ast::Node) {
    use crate::ast::Node as N;
    match node {
        // ---- statement-Vec holders: splice-check inside ------------------
        N::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            descend_for_continue(cond, step);
            super::inject_step_before_continue(then_branch, step);
            if let Some(eb) = else_branch {
                super::inject_step_before_continue(eb, step);
            }
        }
        N::Match {
            scrutinee, arms, ..
        } => {
            descend_for_continue(scrutinee, step);
            for arm in arms.iter_mut() {
                super::inject_step_before_continue_arm(&mut arm.body, step);
            }
        }
        N::Region { body, .. } => super::inject_step_before_continue(body, step),
        N::Block { stmts, .. } => super::inject_step_before_continue(stmts, step),
        // ---- nested loops: header IS ours, BODY is a boundary ------------
        N::While { cond, .. } => descend_for_continue(cond, step),
        N::For { start, end, .. } => {
            descend_for_continue(start, step);
            descend_for_continue(end, step);
        }
        N::ForEach { collection, .. } => descend_for_continue(collection, step),
        // ---- pure expression wrappers: recurse into operands -------------
        N::Binary { left, right, .. } | N::Logical { left, right, .. } => {
            descend_for_continue(left, step);
            descend_for_continue(right, step);
        }
        #[cfg(feature = "std-surface")]
        N::Bitwise { left, right, .. } => {
            descend_for_continue(left, step);
            descend_for_continue(right, step);
        }
        N::Paren(inner, _) => descend_for_continue(inner, step),
        N::Neg { operand, .. } | N::Not { operand, .. } | N::BitNot { operand, .. } => {
            descend_for_continue(operand, step)
        }
        N::As { expr, .. } => descend_for_continue(expr, step),
        N::Ref { inner, .. } => descend_for_continue(inner, step),
        N::Deref { operand, .. } => descend_for_continue(operand, step),
        N::DerefAssign { target, value, .. } => {
            descend_for_continue(target, step);
            descend_for_continue(value, step);
        }
        N::Tuple { elements, .. } | N::ArrayLit { elements, .. } | N::SetLit { elements, .. } => {
            for e in elements.iter_mut() {
                descend_for_continue(e, step);
            }
        }
        N::MapLit { entries, .. } => {
            for (k, v) in entries.iter_mut() {
                descend_for_continue(k, step);
                descend_for_continue(v, step);
            }
        }
        N::StructLit { fields, .. } => {
            for f in fields.iter_mut() {
                descend_for_continue(&mut f.value, step);
            }
        }
        N::Call { args, .. } | N::Print { args, .. } => {
            for a in args.iter_mut() {
                descend_for_continue(a, step);
            }
        }
        N::MethodCall { receiver, args, .. } => {
            descend_for_continue(receiver, step);
            for a in args.iter_mut() {
                descend_for_continue(a, step);
            }
        }
        N::FieldAccess { receiver, .. } => descend_for_continue(receiver, step),
        N::IndexAccess {
            receiver, index, ..
        } => {
            descend_for_continue(receiver, step);
            descend_for_continue(index, step);
        }
        N::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            descend_for_continue(receiver, step);
            descend_for_continue(start, step);
            descend_for_continue(end, step);
        }
        N::Let { value, .. }
        | N::LetTuple { value, .. }
        | N::Assign { value, .. }
        | N::Const { value, .. } => descend_for_continue(value, step),
        N::FieldAssign {
            receiver, value, ..
        } => {
            descend_for_continue(receiver, step);
            descend_for_continue(value, step);
        }
        N::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            descend_for_continue(receiver, step);
            descend_for_continue(index, step);
            descend_for_continue(value, step);
        }
        N::Return { value, .. } => {
            if let Some(v) = value {
                descend_for_continue(v, step);
            }
        }
        N::Assert { cond, .. } => descend_for_continue(cond, step),
        // W1.5f: postfix `?` wraps a single operand expression — descend into
        // it (a loop-targeting `continue` may hide behind `f(if c {continue})?`).
        N::Try { inner, .. } => descend_for_continue(inner, step),
        // ---- tensor / autodiff call wrappers (Box<Node> operands) --------
        N::CallGrad { loss, .. } => descend_for_continue(loss, step),
        N::CallTensorSum { x, .. }
        | N::CallTensorMean { x, .. }
        | N::CallReshape { x, .. }
        | N::CallExpandDims { x, .. }
        | N::CallSqueeze { x, .. }
        | N::CallTranspose { x, .. }
        | N::CallIndex { x, .. }
        | N::CallSlice { x, .. }
        | N::CallSliceStride { x, .. }
        | N::CallTensorRelu { x, .. } => descend_for_continue(x, step),
        N::CallGather { x, idx, .. } => {
            descend_for_continue(x, step);
            descend_for_continue(idx, step);
        }
        N::CallDot { a, b, .. } | N::CallMatMul { a, b, .. } => {
            descend_for_continue(a, step);
            descend_for_continue(b, step);
        }
        N::TensorMatmul { lhs, rhs, .. } | N::TensorElemwise { lhs, rhs, .. } => {
            descend_for_continue(lhs, step);
            descend_for_continue(rhs, step);
        }
        N::CallTensorConv2d { x, w, .. } => {
            descend_for_continue(x, step);
            descend_for_continue(w, step);
        }
        // ---- leaves / boundaries: nothing to descend --------------------
        // `Break`/`Continue` (Continue handled by the caller's splice), literals,
        // imports, declarations, a nested `fn` body (different scope), and
        // childless tensor ops carry no loop-targeting `continue`.
        N::Continue { .. }
        | N::Break { .. }
        | N::Lit(..)
        | N::Import { .. }
        | N::FnDef(..)
        // A closure body is a different scope (like a nested `fn`); it is also
        // desugared away before lowering, so this is unreachable in practice.
        | N::Closure { .. }
        // #268: trait/impl are desugared away before lowering (unreachable here).
        | N::TraitDef { .. }
        | N::ImplBlock { .. }
        | N::StructDef { .. }
        | N::EnumDef { .. }
        | N::TypeAlias { .. }
        | N::Export { .. }
        | N::ExternConst { .. }
        | N::ExternBlock { .. }
        | N::CallTensorRand { .. } => {}
    }
}
