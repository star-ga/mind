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

use libmind::ast::Node;
use libmind::opt::fold;
use libmind::parser;

#[test]
fn folds_simple_arith() {
    let m = parser::parse("1 + 2 * 3").unwrap();
    let node = &m.items[0];
    let f = fold::fold(node);
    if let Node::Lit(_, _) = f {
        // folded to literal
    } else {
        panic!("not folded");
    }
}

/// `opt::fold` used Rust's `/` and `%`, which PANIC on `i64::MIN / -1` in every
/// build profile — division overflow is not gated by `overflow-checks`. The folder
/// is a `pub` API, so that was a compiler abort on a two-literal expression. It must
/// now fold to the defined two's-complement answer, the same one the native backend,
/// the MLIR backend and the interpreter all give at run time.
///
/// The AST is built by editing a parsed `7 / 3` rather than parsing a
/// `-9223372036854775808` literal, because the parser may spell a negative literal as
/// `Neg(Lit)` — which `fold` never folds, so a source-text test could pass without
/// ever reaching the arm under test.
#[test]
fn folding_int_min_by_neg1_wraps_instead_of_panicking() {
    use libmind::ast::{BinOp, Literal};

    fn with_ints(src_op: &str, a: i64, b: i64) -> Node {
        let m = parser::parse(&format!("7 {src_op} 3")).unwrap();
        let Node::Binary {
            op,
            left,
            right,
            span,
        } = m.items[0].clone()
        else {
            panic!("`7 {src_op} 3` did not parse to a Binary node");
        };
        let lit = |n: i64, old: &Node| match old {
            Node::Lit(Literal::Int(_), s) => Node::Lit(Literal::Int(n), *s),
            other => panic!("expected an int literal operand, got {other:?}"),
        };
        Node::Binary {
            op,
            left: Box::new(lit(a, &left)),
            right: Box::new(lit(b, &right)),
            span,
        }
    }

    for (sym, op, want) in [("/", BinOp::Div, i64::MIN), ("%", BinOp::Mod, 0)] {
        let node = with_ints(sym, i64::MIN, -1);
        // Positive control: the node really is the operator under test, so a pass
        // cannot come from folding some other expression.
        assert!(matches!(&node, Node::Binary { op: o, .. } if *o == op));
        match fold::fold(&node) {
            Node::Lit(Literal::Int(v), _) => {
                assert_eq!(v, want, "i64::MIN {sym} -1 must fold to {want}")
            }
            other => panic!("i64::MIN {sym} -1 did not fold to an int literal: {other:?}"),
        }
    }

    // A zero divisor is still left UNFOLDED for the run-time guard to answer.
    assert!(matches!(
        fold::fold(&with_ints("/", 7, 0)),
        Node::Binary { .. }
    ));
}
