use std::collections::HashMap;

use crate::ast::{BinOp, Literal, Module, Node};
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

fn dtype_name(dtype: &DType) -> &'static str {
    match dtype {
        DType::I32 => "i32",
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
    }
}

fn format_shape(shape: &[ShapeDim]) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(sym) => sym.to_string(),
        })
        .collect();
    format!("({})", dims.join(","))
}

fn describe_tensor(tensor: &TensorType) -> String {
    format!("Tensor[{}, {}]", dtype_name(&tensor.dtype), format_shape(&tensor.shape))
}

fn describe_value_type(v: &ValueType) -> String {
    match v {
        ValueType::ScalarI32 => "Scalar[i32]".to_string(),
        ValueType::Tensor(tensor) => describe_tensor(tensor),
    }
}

fn binop_display(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
    }
}

fn promote_scalar_to(dtype: DType) -> Option<ValueType> {
    match dtype {
        DType::F32 => Some(ValueType::ScalarI32),
        DType::I32 => Some(ValueType::ScalarI32),
        _ => None,
    }
}

fn combine_dtypes(lhs: &ValueType, rhs: &ValueType) -> Option<DType> {
    match (lhs, rhs) {
        (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
            if tl.dtype == tr.dtype {
                Some(tl.dtype.clone())
            } else {
                None
            }
        }
        (ValueType::Tensor(t), ValueType::ScalarI32)
        | (ValueType::ScalarI32, ValueType::Tensor(t)) => {
            if t.dtype == DType::F32 {
                promote_scalar_to(t.dtype.clone()).map(|_| DType::F32)
            } else {
                None
            }
        }
        (ValueType::ScalarI32, ValueType::ScalarI32) => None,
    }
}

fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
    let mut out = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;

    while i >= 0 || j >= 0 {
        let da = if i >= 0 { a[i as usize].clone() } else { ShapeDim::Known(1) };
        let db = if j >= 0 { b[j as usize].clone() } else { ShapeDim::Known(1) };

        let dim = match (da, db) {
            (ShapeDim::Known(x), ShapeDim::Known(y)) => {
                if x == y {
                    ShapeDim::Known(x)
                } else if x == 1 {
                    ShapeDim::Known(y)
                } else if y == 1 {
                    ShapeDim::Known(x)
                } else {
                    return None;
                }
            }
            (ShapeDim::Sym(s1), ShapeDim::Sym(s2)) => {
                if s1 == s2 {
                    ShapeDim::Sym(s1)
                } else {
                    return None;
                }
            }
            (ShapeDim::Sym(sym), ShapeDim::Known(n)) | (ShapeDim::Known(n), ShapeDim::Sym(sym)) => {
                if n == 1 {
                    ShapeDim::Sym(sym)
                } else {
                    return None;
                }
            }
        };

        out.push(dim);
        i -= 1;
        j -= 1;
    }

    out.reverse();
    Some(out)
}

fn infer_expr(node: &Node, env: &TypeEnv) -> Result<ValueType, TypeError> {
    match node {
        Node::Lit(Literal::Int(_)) => Ok(ValueType::ScalarI32),
        Node::Lit(Literal::Ident(name)) => {
            env.get(name).cloned().ok_or_else(|| TypeError::UnknownIdent(name.clone()))
        }
        Node::Paren(inner) => infer_expr(inner, env),
        Node::Binary { op, left, right } => {
            let lt = infer_expr(left, env)?;
            let rt = infer_expr(right, env)?;
            if matches!((&lt, &rt), (ValueType::ScalarI32, ValueType::ScalarI32)) {
                return Ok(ValueType::ScalarI32);
            }

            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        if let Some(shape) = broadcast_shapes(&tl.shape, &tr.shape) {
                            Ok(ValueType::Tensor(TensorType::new(dtype, shape)))
                        } else {
                            Err(TypeError::Msg(format!(
                                "cannot broadcast shapes {} and {} for `{}`",
                                format_shape(&tl.shape),
                                format_shape(&tr.shape),
                                binop_display(op)
                            )))
                        }
                    } else {
                        Err(TypeError::Msg(format!(
                            "dtype mismatch for `{}`: left {} vs right {}",
                            binop_display(op),
                            describe_tensor(tl),
                            describe_tensor(tr)
                        )))
                    }
                }
                (ValueType::Tensor(t), ValueType::ScalarI32)
                | (ValueType::ScalarI32, ValueType::Tensor(t)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        Ok(ValueType::Tensor(TensorType::new(dtype, t.shape.clone())))
                    } else {
                        let dtype_str = dtype_name(&t.dtype);
                        let message = match promote_scalar_to(t.dtype.clone()) {
                            Some(_) => format!(
                                "cannot apply `{}`: scalar promotion to tensor dtype `{}` is not supported",
                                binop_display(op), dtype_str
                            ),
                            None => format!(
                                "cannot apply `{}`: tensor dtype `{}` does not support scalar operands",
                                binop_display(op), dtype_str
                            ),
                        };
                        Err(TypeError::Msg(message))
                    }
                }
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
                                            "type mismatch for `{}`: annotation {} vs inferred {}",
                                            name,
                                            describe_value_type(&vt_ann),
                                            describe_value_type(&vt)
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
                                    "cannot assign `{}`: expected {} but found {}",
                                    name,
                                    describe_value_type(&vt_lhs),
                                    describe_value_type(&vt_rhs)
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
