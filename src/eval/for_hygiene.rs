// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Preconditions of the range-`for` hygiene gate (the `For` arm of
//! `src/eval/lower.rs`).
//!
//! `for VAR in START..END { BODY }` keeps its byte-neutral desugar
//! (`let VAR = START; while VAR < END { BODY; VAR = VAR + 1 }`) unless that
//! form can diverge from the interpreter, which evaluates `END` once and binds
//! `VAR` fresh each iteration. These are two of the divergence tests.

use crate::ast;

/// Does expression `node` contain a function/method call (or any tensor
/// builtin call) anywhere?
///
/// Used by the range-`for` hygiene gate: the interpreter oracle evaluates the
/// loop's `END` bound EXACTLY ONCE (`eval` For arm, `src/eval/mod.rs`), while
/// the byte-neutral desugar re-lowers `END` into the `while` condition
/// submodule and therefore re-evaluates it EVERY iteration. When `END` is a
/// pure arithmetic expression over consts/idents that are not written in the
/// body, once-vs-per-iteration are observationally identical and the old
/// desugar is kept verbatim. A CALL in `END` breaks that equivalence (a side
/// effect, or a counter/observable that differs between evaluations), so the
/// presence of a call forces the hygienic form that pre-lowers `END` once.
///
/// The match mirrors `lower::ast_reads_ident`'s traversal so a call buried under any
/// expression wrapper is still detected; only genuine leaves / declarations
/// (which cannot host a range-endpoint call) fall through to `false`.
pub(super) fn expr_contains_call(node: &ast::Node) -> bool {
    use ast::Node as N;
    match node {
        // Any call-like node: this is exactly what breaks once-vs-per-iter.
        N::Call { .. }
        | N::MethodCall { .. }
        | N::CallGrad { .. }
        | N::CallTensorSum { .. }
        | N::CallTensorMean { .. }
        | N::CallReshape { .. }
        | N::CallExpandDims { .. }
        | N::CallSqueeze { .. }
        | N::CallTranspose { .. }
        | N::CallIndex { .. }
        | N::CallSlice { .. }
        | N::CallSliceStride { .. }
        | N::CallGather { .. }
        | N::CallDot { .. }
        | N::CallMatMul { .. }
        | N::CallTensorRelu { .. }
        | N::CallTensorRand { .. }
        | N::CallTensorConv2d { .. }
        | N::TensorMatmul { .. }
        | N::TensorElemwise { .. } => true,
        N::Lit(..) => false,
        N::Binary { left, right, .. } | N::Logical { left, right, .. } => {
            expr_contains_call(left) || expr_contains_call(right)
        }
        #[cfg(feature = "std-surface")]
        N::Bitwise { left, right, .. } => expr_contains_call(left) || expr_contains_call(right),
        N::Paren(inner, _)
        | N::Neg { operand: inner, .. }
        | N::Not { operand: inner, .. }
        | N::Ref { inner, .. }
        | N::As { expr: inner, .. } => expr_contains_call(inner),
        N::Tuple { elements, .. } | N::ArrayLit { elements, .. } | N::SetLit { elements, .. } => {
            elements.iter().any(expr_contains_call)
        }
        N::FieldAccess { receiver, .. } => expr_contains_call(receiver),
        N::IndexAccess {
            receiver, index, ..
        } => expr_contains_call(receiver) || expr_contains_call(index),
        N::StructLit { fields, .. } => fields.iter().any(|f| expr_contains_call(&f.value)),
        N::MapLit { entries, .. } => entries
            .iter()
            .any(|(k, v)| expr_contains_call(k) || expr_contains_call(v)),
        // Leaves / declarations / statements that cannot appear as a range
        // endpoint expression carry no relevant call.
        _ => false,
    }
}

/// Does a top-level statement of `body` re-declare `var`, by `let` or by a
/// tuple `let` naming it?
///
/// Such a binding shadows the loop counter for the rest of the iteration, so
/// the desugar's trailing `VAR = VAR + 1` would step the SHADOW and the counter
/// would never advance: the loop never terminates. A `let` in a nested block,
/// branch or loop is scoped to it (the `Block`, `Region`, `If` and `While`
/// arms lower against a cloned env), so only the body's own statements count.
pub(super) fn body_redeclares(body: &[ast::Node], var: &str) -> bool {
    body.iter().any(|stmt| match stmt {
        ast::Node::Let { name, .. } => name == var,
        ast::Node::LetTuple { names, .. } => names.iter().any(|n| n == var),
        _ => false,
    })
}
