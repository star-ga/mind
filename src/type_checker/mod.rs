use std::collections::HashMap;

use crate::ast::{Literal, Module, Node};
use crate::diagnostics::{Diagnostic as Pretty, Location, Span};
use crate::types::{DType, ShapeDim, TensorType, ValueType};

#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("unknown identifier `{0}`")]
    UnknownIdent(String),
    #[error("incompatible types in binary operation")]
    BadBinop,
    #[error("{0}")]
    Msg(String),
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

fn dtype_from_str(s: &str) -> Option<DType> {
    match s {
        "i32" => Some(DType::I32),
        "f32" => Some(DType::F32),
        _ => None,
    }
}

fn shape_from_dims(dims: &[String]) -> Vec<ShapeDim> {
    dims.iter()
        .map(|d| {
            if let Ok(n) = d.parse::<usize>() {
                ShapeDim::Known(n)
            } else {
                ShapeDim::Sym(Box::leak(d.clone().into_boxed_str()))
            }
        })
        .collect()
}

fn valuetype_from_ann(ann: &crate::ast::TypeAnn) -> Option<ValueType> {
    match ann {
        crate::ast::TypeAnn::ScalarI32 => Some(ValueType::ScalarI32),
        crate::ast::TypeAnn::Tensor { dtype, dims } => {
            let dt = dtype_from_str(dtype)?;
            let shape = shape_from_dims(dims);
            Some(ValueType::Tensor(TensorType::new(dt, shape)))
        }
    }
}

/// Walk statements; extend env on let/assign; return pretty diags for any errors.
pub fn check_module_types(module: &Module, src: &str, env: &TypeEnv) -> Vec<Pretty> {
    let mut errs = Vec::new();
    let mut tenv = env.clone();

    for item in &module.items {
        match item {
            Node::Let { name, ann, value } => match ann {
                Some(annotation) => match valuetype_from_ann(annotation) {
                    Some(vt_ann) => {
                        match infer_expr(value, &tenv) {
                            Ok(vt) => {
                                if vt_ann != vt {
                                    errs.push(pretty_whole_input(
                                            src,
                                            TypeError::Msg(format!(
                                                "type mismatch for `{}`: annotation {:?} vs inferred {:?}",
                                                name, vt_ann, vt
                                            )),
                                        ));
                                }
                            }
                            Err(e) => errs.push(pretty_whole_input(src, e)),
                        }
                        tenv.insert(name.clone(), vt_ann);
                    }
                    None => errs.push(pretty_whole_input(
                        src,
                        TypeError::Msg(format!("unsupported annotation for `{}`", name)),
                    )),
                },
                None => match infer_expr(value, &tenv) {
                    Ok(vt) => {
                        tenv.insert(name.clone(), vt);
                    }
                    Err(e) => errs.push(pretty_whole_input(src, e)),
                },
            },
            Node::Assign { name, value } => {
                let rhs = infer_expr(value, &tenv);
                match (tenv.get(name).cloned(), rhs) {
                    (Some(vt_lhs), Ok(vt_rhs)) => {
                        if vt_lhs != vt_rhs {
                            errs.push(pretty_whole_input(
                                src,
                                TypeError::Msg(format!(
                                    "cannot assign `{}`: expected {:?} but found {:?}",
                                    name, vt_lhs, vt_rhs
                                )),
                            ));
                        }
                    }
                    (None, Ok(vt_rhs)) => {
                        tenv.insert(name.clone(), vt_rhs);
                    }
                    (_, Err(e)) => errs.push(pretty_whole_input(src, e)),
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
