use std::collections::HashMap;

use crate::ast::{Literal, Module, Node};
use crate::diagnostics::{Diagnostic as Pretty, Location, Span};
use crate::types::ValueType;

#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("unknown identifier `{0}`")]
    UnknownIdent(String),
    #[error("incompatible types in binary operation")]
    BadBinop,
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
                _ => Err(TypeError::BadBinop),
            }
        }
        Node::Let { value, .. } | Node::Assign { value, .. } => infer_expr(value, env),
    }
}

/// Walk statements; extend env on let/assign; return pretty diags for any errors.
pub fn check_module_types(module: &Module, src: &str, env: &TypeEnv) -> Vec<Pretty> {
    let mut errs = Vec::new();
    let mut tenv = env.clone();

    for item in &module.items {
        match item {
            Node::Let { name, value } | Node::Assign { name, value } => {
                match infer_expr(value, &tenv) {
                    Ok(t) => {
                        tenv.insert(name.clone(), t);
                    }
                    Err(e) => errs.push(pretty_whole_input(src, e)),
                }
            }
            other => {
                if let Err(e) = infer_expr(other, &tenv) {
                    errs.push(pretty_whole_input(src, e));
                }
            }
        }
    }

    errs
}

fn pretty_whole_input(src: &str, err: TypeError) -> Pretty {
    let span: Span = 0..src.len();
    let start = Location { line: 1, col: 1 };
    Pretty { message: err.to_string(), span, start: start.clone(), end: start }
}
