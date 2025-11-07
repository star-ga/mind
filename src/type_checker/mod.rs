use std::collections::HashMap;

use crate::ast::{Literal, Module, Node};
use crate::diagnostics::{Diagnostic, Location};
use crate::types::ValueType;

#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("unknown identifier `{0}`")]
    UnknownIdent(String),
    #[error("incompatible types in binary op: left={0:?}, right={1:?}")]
    BadBinop(ValueType, ValueType),
}

pub type TypeEnv = HashMap<String, ValueType>;

fn infer_expr(node: &Node, env: &TypeEnv) -> Result<ValueType, TypeError> {
    match node {
        Node::Lit(Literal::Int(_)) => Ok(ValueType::ScalarI32),
        Node::Lit(Literal::Ident(name)) => {
            env.get(name).cloned().ok_or_else(|| TypeError::UnknownIdent(name.clone()))
        }
        Node::Paren(inner) => infer_expr(inner, env),
        Node::Binary { left, right, .. } => {
            let lt = infer_expr(left, env)?;
            let rt = infer_expr(right, env)?;
            match (lt, rt) {
                (ValueType::ScalarI32, ValueType::ScalarI32) => Ok(ValueType::ScalarI32),
                (l, r) => Err(TypeError::BadBinop(l, r)),
            }
        }
        Node::Let { value, .. } | Node::Assign { value, .. } => infer_expr(value, env),
    }
}

pub fn check_module_types(module: &Module, src: &str, env: &TypeEnv) -> Vec<Diagnostic> {
    let mut errors = Vec::new();
    let mut tenv = env.clone();

    for item in &module.items {
        match item {
            Node::Let { name, value } | Node::Assign { name, value } => {
                match infer_expr(value, &tenv) {
                    Ok(t) => {
                        tenv.insert(name.clone(), t);
                    }
                    Err(err) => errors.push(pretty_whole_input(src, err)),
                }
            }
            other => {
                if let Err(err) = infer_expr(other, &tenv) {
                    errors.push(pretty_whole_input(src, err));
                }
            }
        }
    }

    errors
}

fn pretty_whole_input(src: &str, err: TypeError) -> Diagnostic {
    Diagnostic {
        message: err.to_string(),
        span: 0..src.len(),
        start: Location { line: 1, col: 1 },
        end: Location { line: 1, col: 1 },
    }
}
