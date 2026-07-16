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

//! Minimal compile-time i64 const evaluator for the Salov loop-collapse pass
//! (`opt::collapse`, S1).
//!
//! This is deliberately tiny and self-contained: the collapse pass needs its
//! OWN const-bound discovery (`opt::fold` is leaf-only and not pipeline-wired),
//! so this module folds an integer subtree to an `i64` iff every leaf is an
//! integer literal reachable through `+ - * / %`, unary negation, or parens.
//! A non-integer leaf (an identifier, a call, a float) makes the whole subtree
//! non-const and the fold returns `None` — the caller then emits the symbolic
//! branchless closed form instead.
//!
//! ## Ring semantics (the load-bearing invariant)
//! Arithmetic is EXACT in `Z/2^64` using `wrapping_*`, matching the language's
//! defined-wrap i64 semantics (== the emitted MLIR artifact). There is NO
//! saturation and NO float, ever, so a const-folded bound is byte-identical to
//! the value the same expression would produce at runtime.
//!
//! `Div`/`Mod` by a zero literal returns `None` (the loop is left intact / the
//! collapse refuses) rather than trapping at compile time.

use crate::ast::BinOp;
use crate::ast::Literal;
use crate::ast::Node;

/// Fold `node` to an `i64` iff it is a fully-constant integer expression under
/// wrapping (`Z/2^64`) semantics. Returns `None` for any non-constant or
/// division-by-zero subtree.
pub fn eval_const_i64(node: &Node) -> Option<i64> {
    match node {
        Node::Lit(Literal::Int(v), _) => Some(*v),
        Node::Paren(inner, _) => eval_const_i64(inner),
        Node::Neg { operand, .. } => Some(0i64.wrapping_sub(eval_const_i64(operand)?)),
        Node::Binary {
            op, left, right, ..
        } => {
            let a = eval_const_i64(left)?;
            let b = eval_const_i64(right)?;
            match op {
                BinOp::Add => Some(a.wrapping_add(b)),
                BinOp::Sub => Some(a.wrapping_sub(b)),
                BinOp::Mul => Some(a.wrapping_mul(b)),
                // `wrapping_div`/`wrapping_rem` are exact for every operand pair
                // except division by zero (returns None) and i64::MIN / -1 (whose
                // wrapping result is i64::MIN — the defined-wrap answer). Both
                // mirror `arith.divsi`/`arith.remsi` at runtime.
                BinOp::Div => {
                    if b == 0 {
                        None
                    } else {
                        Some(a.wrapping_div(b))
                    }
                }
                BinOp::Mod => {
                    if b == 0 {
                        None
                    } else {
                        Some(a.wrapping_rem(b))
                    }
                }
                // Comparisons/other ops are not part of an affine bound.
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;

    fn int(v: i64) -> Node {
        Node::Lit(Literal::Int(v), Span::new(0, 0))
    }
    fn bin(op: BinOp, l: Node, r: Node) -> Node {
        Node::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn folds_wrapping_arithmetic() {
        // (2 + 3) * 4 = 20
        let e = bin(BinOp::Mul, bin(BinOp::Add, int(2), int(3)), int(4));
        assert_eq!(eval_const_i64(&e), Some(20));
    }

    #[test]
    fn wraps_at_i64_boundary() {
        // i64::MAX + 1 wraps to i64::MIN (defined-wrap, not None).
        let e = bin(BinOp::Add, int(i64::MAX), int(1));
        assert_eq!(eval_const_i64(&e), Some(i64::MIN));
    }

    #[test]
    fn ident_is_not_const() {
        let e = Node::Lit(Literal::Ident("n".into()), Span::new(0, 0));
        assert_eq!(eval_const_i64(&e), None);
    }

    #[test]
    fn div_by_zero_is_none() {
        let e = bin(BinOp::Div, int(10), int(0));
        assert_eq!(eval_const_i64(&e), None);
    }
}
