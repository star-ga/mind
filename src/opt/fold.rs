use crate::ast::{BinOp, Literal, Node};

/// Fold constant integer subtrees bottom-up. Pure function; no env.
pub fn fold(node: &Node) -> Node {
    match node {
        Node::Binary { op, left, right } => {
            let l = fold(left);
            let r = fold(right);
            if let (Node::Lit(Literal::Int(a)), Node::Lit(Literal::Int(b))) = (&l, &r) {
                let v = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => {
                        if *b == 0 {
                            return Node::Binary {
                                op: op.clone(),
                                left: Box::new(l),
                                right: Box::new(r),
                            };
                        } else {
                            a / b
                        }
                    }
                };
                Node::Lit(Literal::Int(v))
            } else {
                Node::Binary {
                    op: op.clone(),
                    left: Box::new(l),
                    right: Box::new(r),
                }
            }
        }
        Node::Paren(inner) => {
            let f = fold(inner);
            Node::Paren(Box::new(f))
        }
        other => other.clone(),
    }
}
