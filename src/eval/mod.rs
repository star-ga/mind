use std::collections::HashMap;

use crate::ast::{BinOp, Literal, Module, Node};
use crate::diagnostics::Diagnostic;
use crate::types::ValueType;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("unsupported node")]
    Unsupported,
    #[error("division by zero")]
    DivZero,
    #[error("unknown variable: {0}")]
    UnknownVar(String),
    #[error("type error")]
    TypeError(Vec<Diagnostic>),
}

fn eval_expr(node: &Node, env: &HashMap<String, i64>) -> Result<i64, EvalError> {
    match node {
        Node::Lit(Literal::Int(n)) => Ok(*n),
        Node::Lit(Literal::Ident(name)) => {
            env.get(name).copied().ok_or_else(|| EvalError::UnknownVar(name.clone()))
        }
        Node::Paren(inner) => eval_expr(inner, env),
        Node::Binary { op, left, right } => {
            let l = eval_expr(left, env)?;
            let r = eval_expr(right, env)?;
            Ok(match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => {
                    if r == 0 {
                        return Err(EvalError::DivZero);
                    } else {
                        l / r
                    }
                }
            })
        }
        _ => Err(EvalError::Unsupported),
    }
}

pub fn eval_module_with_env(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
) -> Result<i64, EvalError> {
    if let Some(src) = src_for_types {
        let mut tenv: HashMap<String, ValueType> = HashMap::new();
        for (name, _value) in env.iter() {
            tenv.insert(name.clone(), ValueType::ScalarI32);
        }
        let diags = crate::type_checker::check_module_types(m, src, &tenv);
        if !diags.is_empty() {
            return Err(EvalError::TypeError(diags));
        }
    }

    let mut last = 0_i64;
    for item in &m.items {
        match item {
            Node::Let { name, value } | Node::Assign { name, value } => {
                let v = eval_expr(value, env)?;
                env.insert(name.clone(), v);
                last = v;
            }
            _ => {
                last = eval_expr(item, env)?;
            }
        }
    }
    Ok(last)
}

pub fn eval_module(m: &Module) -> Result<i64, EvalError> {
    let mut env: HashMap<String, i64> = HashMap::new();
    eval_module_with_env(m, &mut env, None)
}

pub fn eval_first_expr(m: &Module) -> Result<i64, EvalError> {
    eval_module(m)
}
