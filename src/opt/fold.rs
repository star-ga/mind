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
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => {
                        if *b == 0 {
                            return Node::Binary {
                                op: *op,
                                left: Box::new(l),
                                right: Box::new(r),
                                span: *span,
                            };
                        } else {
                            a / b
                        }
                    }
                };
                Node::Lit(Literal::Int(v), *span)
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
