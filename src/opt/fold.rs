use crate::ast::{BinOp, Literal, Node};

/// Fold constant integer subtrees bottom-up. Pure function; no env.
pub fn fold(node: &Node) -> Node {
    match node {
        Node::Binary { op, left, right, span } => {
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
                Node::Binary { op: *op, left: Box::new(l), right: Box::new(r), span: *span }
            }
        }
        Node::Paren(inner, span) => {
            let f = fold(inner);
            Node::Paren(Box::new(f), *span)
        }
        other => other.clone(),
    }
}
