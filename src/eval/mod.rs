use crate::ast::{BinOp, Literal, Module, Node};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("unsupported node")]
    Unsupported,
    #[error("division by zero")]
    DivZero,
}

pub fn eval_int(node: &Node) -> Result<i64, EvalError> {
    match node {
        Node::Lit(Literal::Int(value)) => Ok(*value),
        Node::Paren(inner) => eval_int(inner),
        Node::Binary { op, left, right } => {
            let left = eval_int(left)?;
            let right = eval_int(right)?;
            match op {
                BinOp::Add => Ok(left + right),
                BinOp::Sub => Ok(left - right),
                BinOp::Mul => Ok(left * right),
                BinOp::Div => {
                    if right == 0 {
                        Err(EvalError::DivZero)
                    } else {
                        Ok(left / right)
                    }
                }
            }
        }
        _ => Err(EvalError::Unsupported),
    }
}

pub fn eval_first_expr(module: &Module) -> Result<i64, EvalError> {
    module.items.first().map(eval_int).unwrap_or(Ok(0))
}
