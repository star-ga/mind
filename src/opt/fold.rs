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

use crate::ast::BinOp;
use crate::ast::Literal;
use crate::ast::Node;
use crate::types::DType;

/// Fold constant integer subtrees bottom-up. Pure function; no env.
pub fn fold(node: &Node) -> Node {
    match node {
        Node::Binary {
            op,
            left,
            right,
            span,
        } => {
            let l = fold(left);
            let r = fold(right);
            if let (Node::Lit(Literal::Int(a), _), Node::Lit(Literal::Int(b), _)) = (&l, &r) {
                let v = match op {
                    // Defined-wrap is the language semantics (== the MLIR artifact), but a
                    // compile-time fold must be EXACT or not happen: fold only when the result
                    // is representable, else leave the subtree unfolded so the runtime wraps it
                    // identically (same bail shape as Div-by-0 below). Never emit a wrapped or
                    // saturated constant — that is a silent const-vs-runtime miscompile.
                    BinOp::Add => match a.checked_add(*b) {
                        Some(v) => v,
                        None => {
                            return Node::Binary {
                                op: *op,
                                left: Box::new(l),
                                right: Box::new(r),
                                span: *span,
                            };
                        }
                    },
                    BinOp::Sub => match a.checked_sub(*b) {
                        Some(v) => v,
                        None => {
                            return Node::Binary {
                                op: *op,
                                left: Box::new(l),
                                right: Box::new(r),
                                span: *span,
                            };
                        }
                    },
                    BinOp::Mul => match a.checked_mul(*b) {
                        Some(v) => v,
                        None => {
                            return Node::Binary {
                                op: *op,
                                left: Box::new(l),
                                right: Box::new(r),
                                span: *span,
                            };
                        }
                    },
                    // checked_div / checked_rem are None for a zero divisor AND for
                    // i64::MIN by -1 (which panics as `a / b`): leave both unfolded.
                    BinOp::Div => match a.checked_div(*b) {
                        Some(v) => v,
                        None => {
                            return Node::Binary {
                                op: *op,
                                left: Box::new(l),
                                right: Box::new(r),
                                span: *span,
                            };
                        }
                    },
                    BinOp::Mod => match a.checked_rem(*b) {
                        Some(v) => v,
                        None => {
                            return Node::Binary {
                                op: *op,
                                left: Box::new(l),
                                right: Box::new(r),
                                span: *span,
                            };
                        }
                    },
                    BinOp::Lt => (a < b) as i64,
                    BinOp::Le => (a <= b) as i64,
                    BinOp::Gt => (a > b) as i64,
                    BinOp::Ge => (a >= b) as i64,
                    BinOp::Eq => (a == b) as i64,
                    BinOp::Ne => (a != b) as i64,
                };
                Node::Lit(Literal::Int(v), *span)
            } else if let (Node::Lit(Literal::Float(a), _), Node::Lit(Literal::Float(b), _)) =
                (&l, &r)
            {
                let v = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Mod => a % b,
                    BinOp::Lt => {
                        if a < b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Le => {
                        if a <= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Gt => {
                        if a > b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Ge => {
                        if a >= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Eq => {
                        if a == b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    BinOp::Ne => {
                        if a != b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
                Node::Lit(Literal::Float(v), *span)
            } else {
                Node::Binary {
                    op: *op,
                    left: Box::new(l),
                    right: Box::new(r),
                    span: *span,
                }
            }
        }
        Node::Paren(inner, span) => {
            let f = fold(inner);
            Node::Paren(Box::new(f), *span)
        }
        other => other.clone(),
    }
}

/// The value of an integer constant expression (spec 3.2): an integer literal,
/// a negation, a parenthesis, or `+ - * /` over such operands, folded bottom-up.
///
/// This is the ONE fold that the array-literal paths read: the fixed-array and
/// const-array lowering and the native-lane fence. A division by zero or an
/// overflow anywhere in the tree gives `None`, so the element stays a runtime
/// value (or a refusal in the fence). It never gives a wrapped constant.
pub fn const_int(node: &Node) -> Option<i64> {
    match node {
        Node::Lit(Literal::Int(value), _) => Some(*value),
        Node::Neg { operand, .. } => const_int(operand)?.checked_neg(),
        Node::Paren(inner, _) => const_int(inner),
        Node::Binary {
            op, left, right, ..
        } => {
            let (a, b) = (const_int(left)?, const_int(right)?);
            match op {
                BinOp::Add => a.checked_add(b),
                BinOp::Sub => a.checked_sub(b),
                BinOp::Mul => a.checked_mul(b),
                // checked_div is None for a zero divisor and for MIN / -1.
                BinOp::Div => a.checked_div(b),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The value of a constant float element: a float literal, its negation or
/// parenthesis, or an integer constant expression ([`const_int`]) widened to
/// f64. `None` for anything else; a caller never reads it as 0.0.
pub fn const_float(node: &Node) -> Option<f64> {
    match node {
        Node::Lit(Literal::Float(value), _) => Some(*value),
        Node::Neg { operand, .. } => const_float(operand).map(|value| -value),
        Node::Paren(inner, _) => const_float(inner),
        _ => const_int(node).map(|value| value as f64),
    }
}

/// Spec 3.3, constant conversion `Q(x)` from binary64 to Q16.16: `r` is
/// `x * 65536` rounded to the nearest integer, ties to even. `None` (the
/// compiler refuses, `Q16ConstantOutOfRange`) for NaN, an infinity, or `r`
/// outside `[-2^31, 2^31 - 1]`. `x * 65536.0` is exact for every finite `x`
/// whose product is finite, and an overflowed product is out of range.
pub fn q16_const(x: f64) -> Option<i32> {
    // NaN and the infinities are outside the range too.
    let r = (x * 65536.0).round_ties_even();
    (-2_147_483_648.0..=2_147_483_647.0)
        .contains(&r)
        .then_some(r as i32)
}

/// Why a dense literal element has no constant bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstElem {
    /// The element does not fold (spec 3.2).
    NonConstant,
    /// A Q16.16 element is NaN, infinite or out of range (spec 3.3).
    Q16OutOfRange,
}

/// Encode one dense-tensor-literal element to the `u64` bit pattern stored in
/// `Instr::ConstDenseTensor` (must mirror `render_dense_elem` in the MLIR
/// backend): f32 → its u32 IEEE bits, f64 → all 64 bits, i32 → two's-complement
/// low-32, i64 as-is, Q16.16 → [`q16_const`] as i32. An error makes the caller
/// refuse by name; it never stores a zero.
pub fn dense_elem_bits(node: &Node, dtype: &DType) -> Result<u64, ConstElem> {
    let int = || const_int(node).ok_or(ConstElem::NonConstant);
    let float = || const_float(node).ok_or(ConstElem::NonConstant);
    Ok(match dtype {
        DType::F32 | DType::F16 | DType::BF16 => (float()? as f32).to_bits() as u64,
        DType::F64 => float()?.to_bits(),
        DType::I32 => ((int()? as i32) as u32) as u64,
        DType::Q16 => (q16_const(float()?).ok_or(ConstElem::Q16OutOfRange)? as u32) as u64,
        DType::I64 => int()? as u64,
    })
}

/// True when the integer `n` survives the IR's binary64 fill exactly. The
/// i128 compare has no saturation: `i64::MAX as f64` is 2^63, not `i64::MAX`.
pub fn f64_holds_int(n: i64) -> bool {
    (n as f64) as i128 == i128::from(n)
}

/// True when a scalar fill `v` of an integer tensor is an integer the element
/// type holds exactly. Other dtypes are always exact here.
pub fn exact_int_fill(v: f64, dtype: &DType) -> bool {
    let bound = match dtype {
        DType::I32 => 2_147_483_648.0,
        DType::I64 => 9_223_372_036_854_775_808.0,
        _ => return true,
    };
    v.fract() == 0.0 && (-bound..bound).contains(&v)
}

#[cfg(test)]
mod overflow_tests {
    use super::*;
    use crate::ast::Span;

    fn int(n: i64) -> Node {
        Node::Lit(Literal::Int(n), Span::new(0, 0))
    }
    fn bin(op: BinOp, a: i64, b: i64) -> Node {
        Node::Binary {
            op,
            left: Box::new(int(a)),
            right: Box::new(int(b)),
            span: Span::new(0, 0),
        }
    }

    // Non-overflowing constants still fold exactly — no behavior change, no canary drift.
    #[test]
    fn nonoverflowing_folds_exactly() {
        assert!(matches!(
            fold(&bin(BinOp::Add, 2, 3)),
            Node::Lit(Literal::Int(5), _)
        ));
        assert!(matches!(
            fold(&bin(BinOp::Sub, 10, 4)),
            Node::Lit(Literal::Int(6), _)
        ));
        assert!(matches!(
            fold(&bin(BinOp::Mul, -6, 7)),
            Node::Lit(Literal::Int(-42), _)
        ));
    }

    // Overflowing const folds are REFUSED (left unfolded) — never a wrapped or saturated
    // constant. Running under `cargo test` (debug) also proves mindc no longer PANICS on an
    // overflowing const expression.
    #[test]
    fn overflowing_add_left_unfolded() {
        assert!(matches!(
            fold(&bin(BinOp::Add, i64::MAX, 1)),
            Node::Binary { .. }
        ));
    }
    #[test]
    fn overflowing_sub_left_unfolded() {
        assert!(matches!(
            fold(&bin(BinOp::Sub, i64::MIN, 1)),
            Node::Binary { .. }
        ));
    }
    #[test]
    fn const_int_folds_the_spec_subset_and_refuses_the_rest() {
        let neg = Node::Neg {
            operand: Box::new(int(7)),
            span: Span::new(0, 0),
        };
        assert_eq!(const_int(&bin(BinOp::Sub, 0, 1)), Some(-1));
        assert_eq!(const_int(&bin(BinOp::Div, -7, 2)), Some(-3));
        assert_eq!(const_int(&neg), Some(-7));
        assert_eq!(
            const_int(&Node::Paren(
                Box::new(bin(BinOp::Mul, 3, 5)),
                Span::new(0, 0)
            )),
            Some(15)
        );
        // Division by zero, overflow and an operator outside spec 3.2 stay unfolded.
        assert_eq!(const_int(&bin(BinOp::Div, 1, 0)), None);
        assert_eq!(const_int(&bin(BinOp::Div, i64::MIN, -1)), None);
        assert_eq!(const_int(&bin(BinOp::Add, i64::MAX, 1)), None);
        assert_eq!(const_int(&bin(BinOp::Mul, i64::MIN, -1)), None);
        assert_eq!(const_int(&bin(BinOp::Mod, 7, 2)), None);
        assert_eq!(const_int(&bin(BinOp::Lt, 1, 2)), None);
    }
    // i64::MIN / -1 and i64::MIN % -1 overflow: `fold` leaves them unfolded
    // (it panicked here before), like a zero divisor.
    #[test]
    fn min_by_minus_one_and_zero_divisor_left_unfolded() {
        for op in [BinOp::Div, BinOp::Mod] {
            assert!(matches!(fold(&bin(op, i64::MIN, -1)), Node::Binary { .. }));
            assert!(matches!(fold(&bin(op, 7, 0)), Node::Binary { .. }));
        }
        assert!(matches!(
            fold(&bin(BinOp::Div, -7, 2)),
            Node::Lit(Literal::Int(-3), _)
        ));
        assert!(matches!(
            fold(&bin(BinOp::Mod, -7, 2)),
            Node::Lit(Literal::Int(-1), _)
        ));
    }
    // Spec 3.3: NaN, infinities and out-of-range r refuse; ties go to even.
    #[test]
    fn q16_const_ties_to_even_and_refuses_non_finite() {
        assert_eq!(q16_const(f64::NAN), None);
        assert_eq!(q16_const(f64::INFINITY), None);
        assert_eq!(q16_const(f64::NEG_INFINITY), None);
        assert_eq!(q16_const(32768.0), None);
        assert_eq!(q16_const(-32768.0), Some(i32::MIN));
        assert_eq!(q16_const(2.0f64.powi(-17)), Some(0));
        assert_eq!(q16_const(5.0 * 2.0f64.powi(-17)), Some(2));
        assert_eq!(q16_const(-0.0), Some(0));
        assert!(exact_int_fill(3.0, &DType::I32) && !exact_int_fill(1.5, &DType::I32));
        assert!(
            !exact_int_fill(f64::NAN, &DType::I64) && !exact_int_fill(2147483648.0, &DType::I32)
        );
        let p53 = 1i64 << 53;
        assert!(
            [p53, p53 - 1, -p53, i64::MIN, 0]
                .into_iter()
                .all(f64_holds_int)
        );
        assert!(
            ![p53 + 1, i64::MAX, i64::MIN + 1, -p53 - 1]
                .into_iter()
                .any(f64_holds_int)
        );
    }
    #[test]
    fn overflowing_mul_left_unfolded() {
        assert!(matches!(
            fold(&bin(BinOp::Mul, i64::MIN, -1)),
            Node::Binary { .. }
        ));
        assert!(matches!(
            fold(&bin(BinOp::Mul, i64::MAX, 2)),
            Node::Binary { .. }
        ));
    }
}
