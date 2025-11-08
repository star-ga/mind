use std::collections::HashMap;

use crate::ast::{BinOp, Literal, Module, Node, Span as AstSpan};
use crate::diagnostics::{Diagnostic as Pretty, Location};
use crate::types::{DType, ShapeDim, TensorType, ValueType};

#[derive(Debug)]
pub struct TypeErrSpan {
    pub msg: String,
    pub span: AstSpan,
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

fn infer_expr(node: &Node, env: &TypeEnv) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    match node {
        Node::Lit(Literal::Int(_), span) => Ok((ValueType::ScalarI32, *span)),
        Node::Lit(Literal::Ident(name), span) => {
            env.get(name).cloned().map(|t| (t, *span)).ok_or_else(|| TypeErrSpan {
                msg: format!("unknown identifier `{name}`"),
                span: *span,
            })
        }
        Node::Paren(inner, span) => {
            let (ty, _) = infer_expr(inner, env)?;
            Ok((ty, *span))
        }
        Node::Binary { op, left, right, span } => {
            let (lt, _) = infer_expr(left, env)?;
            let (rt, _) = infer_expr(right, env)?;
            if matches!((&lt, &rt), (ValueType::ScalarI32, ValueType::ScalarI32)) {
                return Ok((ValueType::ScalarI32, *span));
            }

            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        if let Some(shape) = broadcast_shapes(&tl.shape, &tr.shape) {
                            Ok((ValueType::Tensor(TensorType::new(dtype, shape)), *span))
                        } else {
                            Err(TypeErrSpan {
                                msg: format!(
                                    "cannot broadcast shapes {} and {} for `{}`",
                                    format_shape(&tl.shape),
                                    format_shape(&tr.shape),
                                    binop_display(op)
                                ),
                                span: *span,
                            })
                        }
                    } else {
                        Err(TypeErrSpan {
                            msg: format!(
                                "dtype mismatch for `{}`: left {} vs right {}",
                                binop_display(op),
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        })
                    }
                }
                (ValueType::Tensor(t), ValueType::ScalarI32)
                | (ValueType::ScalarI32, ValueType::Tensor(t)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        Ok((ValueType::Tensor(TensorType::new(dtype, t.shape.clone())), *span))
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
                        Err(TypeErrSpan { msg: message, span: *span })
                    }
                }
                _ => Err(TypeErrSpan {
                    msg: "incompatible types in binary operation".to_string(),
                    span: *span,
                }),
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
            Node::Let { name, ann, value, span } => match ann {
                Some(annotation) => match valuetype_from_ann(annotation) {
                    Some(vt_ann) => {
                        match infer_expr(value, &tenv) {
                            Ok((vt, _)) => {
                                let allow_scalar_fill = matches!(
                                    (&vt_ann, &vt),
                                    (ValueType::Tensor(_), ValueType::ScalarI32)
                                );
                                if vt_ann != vt && !allow_scalar_fill {
                                    errs.push(diag_from_span(
                                        src,
                                        format!(
                                            "type mismatch for `{}`: annotation {} vs inferred {}",
                                            name,
                                            describe_value_type(&vt_ann),
                                            describe_value_type(&vt)
                                        ),
                                        value.span(),
                                    ));
                                }
                            }
                            Err(e) => errs.push(diag_from_type_err(src, e)),
                        }
                        tenv.insert(name.clone(), vt_ann);
                    }
                    None => errs.push(diag_from_span(
                        src,
                        format!("unsupported annotation for `{}`", name),
                        *span,
                    )),
                },
                None => match infer_expr(value, &tenv) {
                    Ok((vt, _)) => {
                        tenv.insert(name.clone(), vt);
                    }
                    Err(e) => errs.push(diag_from_type_err(src, e)),
                },
            },
            Node::Assign { name, value, .. } => {
                let rhs = infer_expr(value, &tenv);
                match (tenv.get(name).cloned(), rhs) {
                    (Some(vt_lhs), Ok((vt_rhs, _))) => {
                        if vt_lhs != vt_rhs {
                            errs.push(diag_from_span(
                                src,
                                format!(
                                    "cannot assign `{}`: expected {} but found {}",
                                    name,
                                    describe_value_type(&vt_lhs),
                                    describe_value_type(&vt_rhs)
                                ),
                                value.span(),
                            ));
                        }
                    }
                    (None, Ok((vt_rhs, _))) => {
                        tenv.insert(name.clone(), vt_rhs);
                    }
                    (_, Err(e)) => errs.push(diag_from_type_err(src, e)),
                }
            }
            other => {
                if let Err(e) = infer_expr(other, &tenv) {
                    errs.push(diag_from_type_err(src, e));
                }
            }
        }
    }

    errs
}

fn location_at(src: &str, offset: usize) -> Location {
    let mut line = 1usize;
    let mut col = 1usize;
    let mut count = 0usize;
    for ch in src.chars() {
        if count >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
        count += ch.len_utf8();
    }
    Location { line, col }
}

fn diag_from_span(src: &str, msg: String, span: AstSpan) -> Pretty {
    let range = span.start()..span.end();
    let start = location_at(src, span.start());
    let end = location_at(src, span.end());
    Pretty { message: msg, span: range, start, end }
}

fn diag_from_type_err(src: &str, err: TypeErrSpan) -> Pretty {
    diag_from_span(src, err.msg, err.span)
}
