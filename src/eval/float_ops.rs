// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Scalar float binary operators of the tree-walking evaluator.
//!
//! A float comparison is a `bool`: it yields the same `Value::Int(0 | 1)` an
//! integer comparison does, which is what every consumer of a condition
//! (`if`, `while`, `&&`, `||`, `!`, `assert`, match guards) reads. The compiled
//! path lowers it to an `i1` the same way. The evaluator used to return
//! `Value::Float(1.0 | 0.0)`: `&&`/`||` read that as false, `if` as true, and
//! `!` refused it. Comparisons are IEEE 754 ordered comparisons, so each is
//! false when an operand is NaN and `!=` is true; `-0.0 == 0.0`.

use crate::ast::BinOp;

use super::{EvalError, Value};

/// Evaluate `l op r` for two float scalars (an int operand is widened first).
pub(super) fn float_binary(op: BinOp, l: f64, r: f64) -> Result<Value, EvalError> {
    let truth = |b: bool| Ok(Value::Int(b as i64));
    match op {
        BinOp::Add => Ok(Value::Float(l + r)),
        BinOp::Sub => Ok(Value::Float(l - r)),
        BinOp::Mul => Ok(Value::Float(l * r)),
        BinOp::Div | BinOp::Mod if r == 0.0 => Err(EvalError::DivZero),
        BinOp::Div => Ok(Value::Float(l / r)),
        BinOp::Mod => Ok(Value::Float(l % r)),
        BinOp::Lt => truth(l < r),
        BinOp::Le => truth(l <= r),
        BinOp::Gt => truth(l > r),
        BinOp::Ge => truth(l >= r),
        BinOp::Eq => truth(l == r),
        BinOp::Ne => truth(l != r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(v: Result<Value, EvalError>) -> i64 {
        match v {
            Ok(Value::Int(n)) => n,
            other => panic!("expected an Int truth value, got {other:?}"),
        }
    }

    #[test]
    fn comparisons_are_int_truth_values() {
        assert_eq!(int(float_binary(BinOp::Ge, 3.0, 1.0)), 1);
        assert_eq!(int(float_binary(BinOp::Le, 3.0, 1.0)), 0);
        assert_eq!(int(float_binary(BinOp::Lt, 1.0, 3.0)), 1);
        assert_eq!(int(float_binary(BinOp::Gt, 1.0, 3.0)), 0);
        assert_eq!(int(float_binary(BinOp::Eq, -0.0, 0.0)), 1);
        assert_eq!(int(float_binary(BinOp::Ne, 2.5, 2.5)), 0);
    }

    #[test]
    fn nan_comparisons_follow_ieee_754() {
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge, BinOp::Eq] {
            assert_eq!(int(float_binary(op, f64::NAN, 1.0)), 0, "{op:?}");
        }
        assert_eq!(int(float_binary(BinOp::Ne, f64::NAN, f64::NAN)), 1);
    }

    #[test]
    fn arithmetic_stays_float_and_division_by_zero_is_an_error() {
        assert_eq!(
            float_binary(BinOp::Add, 1.5, 2.0).unwrap(),
            Value::Float(3.5)
        );
        assert_eq!(
            float_binary(BinOp::Mod, 7.0, 4.0).unwrap(),
            Value::Float(3.0)
        );
        assert!(matches!(
            float_binary(BinOp::Div, 1.0, 0.0),
            Err(EvalError::DivZero)
        ));
        assert!(matches!(
            float_binary(BinOp::Mod, 1.0, 0.0),
            Err(EvalError::DivZero)
        ));
    }
}
