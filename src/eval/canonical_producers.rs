// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Adapter from checked source facts to canonical SSA values.
//!
//! This module does not infer types from syntax. The existing type checker
//! records exact facts (or explicit unknown provenance) during the canonical
//! pipeline; this adapter only attaches an exact fact to the `ValueId` that
//! the already-run lowerer returned for the same source occurrence.

use super::LoweringContext;
use crate::type_checker::canonical_facts::CheckedFact;

pub(super) fn record_checked_expr(
    node: &crate::ast::Node,
    value: crate::ir::ValueId,
    context: &mut LoweringContext,
) {
    let Some(fact) = context.canonical_checked_fact(node.span()) else {
        return;
    };
    let CheckedFact::Exact(value_type) = fact else {
        return;
    };
    let Some(semantic_type) = lowered_semantic_type(node, value_type) else {
        return;
    };
    context.canonical_record_producer(node.span(), value, semantic_type);
}

/// The checker intentionally calls an unannotated integer literal `i32`, while
/// the existing scalar lowering emits `ConstI64` (and the pure integer
/// arithmetic lane emits i64 `BinOp`s).  This is an explicit source-to-IR
/// coercion already present in the lowering contract, so preserve it here
/// without re-inferring declarations or expected call types.  A checked i32
/// parameter/identifier remains i32 metadata.
fn lowered_semantic_type(
    node: &crate::ast::Node,
    value_type: crate::types::ValueType,
) -> Option<crate::types::SemanticType> {
    if matches!(value_type, crate::types::ValueType::ScalarI32) && is_i64_integer_lowering(node) {
        return Some(crate::types::SemanticType::Scalar(
            crate::types::ScalarType::I64,
        ));
    }
    semantic_type(value_type)
}

fn is_i64_integer_lowering(node: &crate::ast::Node) -> bool {
    use crate::ast::{BinOp, Literal, Node};
    match node {
        Node::Lit(Literal::Int(_), _) => true,
        Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => is_i64_integer_lowering(inner),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => is_i64_integer_lowering(left) && is_i64_integer_lowering(right),
        _ => false,
    }
}

pub(super) fn semantic_type(
    value_type: crate::types::ValueType,
) -> Option<crate::types::SemanticType> {
    use crate::types::{ScalarType, SemanticType, ValueType};
    let scalar = match value_type {
        ValueType::ScalarI32 => ScalarType::I32,
        ValueType::ScalarI64 => ScalarType::I64,
        ValueType::ScalarF32 => ScalarType::F32,
        ValueType::ScalarF64 => ScalarType::F64,
        ValueType::ScalarBool => ScalarType::Bool,
        // A reference capability has no scalar semantic type (deref-assign D2).
        ValueType::Tensor(_) | ValueType::GradMap(_) | ValueType::Ref { .. } => return None,
    };
    Some(SemanticType::Scalar(scalar))
}
