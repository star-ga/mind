// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Equality of two enum values in the tree evaluator.
//!
//! Unit variants compare by the variant path. A payload compares with it.
//! Any other operator stays unsupported, the same as the catch-all arm.

use super::{EvalError, Value};
use crate::ast::BinOp;

pub(super) fn cmp(op: BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
    let (
        Value::Enum {
            variant: left_variant,
            payload: left_payload,
        },
        Value::Enum {
            variant: right_variant,
            payload: right_payload,
        },
    ) = (left, right)
    else {
        return Err(EvalError::Unsupported);
    };
    match op {
        BinOp::Eq => Ok(Value::Int(
            (left_variant == right_variant && left_payload == right_payload) as i64,
        )),
        BinOp::Ne => Ok(Value::Int(
            (left_variant != right_variant || left_payload != right_payload) as i64,
        )),
        _ => Err(EvalError::Unsupported),
    }
}
