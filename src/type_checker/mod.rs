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
        ValueType::GradMap(entries) => {
            let mut parts = Vec::new();
            for (name, tensor) in entries {
                parts.push(format!("{}: {}", name, describe_tensor(tensor)));
            }
            format!("GradMap{{{}}}", parts.join(", "))
        }
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
        (ValueType::GradMap(_), _) | (_, ValueType::GradMap(_)) => None,
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
        Node::Tuple { span, .. } => Ok((ValueType::ScalarI32, *span)),
        Node::Call { callee, args, span } => infer_call(callee, args, *span, env),
        Node::CallGrad { loss, wrt, span } => infer_grad(loss, wrt, *span, env),
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

fn infer_grad(
    loss: &Node,
    wrt: &[String],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    let (loss_ty, _) = infer_expr(loss, env)?;
    match loss_ty {
        ValueType::Tensor(ref t) => {
            if !t.shape.is_empty() {
                return Err(TypeErrSpan {
                    msg: "`grad` expects a scalar loss with shape ()".to_string(),
                    span: loss.span(),
                });
            }
        }
        _ => {
            return Err(TypeErrSpan {
                msg: "`grad` requires the loss to be a tensor expression".to_string(),
                span: loss.span(),
            });
        }
    };

    let mut entries = Vec::new();
    for name in wrt {
        match env.get(name) {
            Some(ValueType::Tensor(t)) => entries.push((name.clone(), t.clone())),
            Some(_) => {
                return Err(TypeErrSpan {
                    msg: format!("`{}` is not a tensor variable", name),
                    span,
                });
            }
            None => {
                return Err(TypeErrSpan {
                    msg: format!("unknown tensor `{}` in `wrt`", name),
                    span,
                });
            }
        }
    }

    Ok((ValueType::GradMap(entries), span))
}

fn infer_call(
    callee: &str,
    args: &[Node],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    match callee {
        "tensor.zeros" | "tensor.ones" => {
            if args.len() != 2 {
                return Err(TypeErrSpan {
                    msg: format!("`{callee}` expects (dtype, shape) arguments"),
                    span,
                });
            }
            let dtype = infer_dtype_arg(&args[0])?;
            let shape = infer_shape_arg(&args[1])?;
            Ok((ValueType::Tensor(TensorType::new(dtype, shape)), span))
        }
        "tensor.shape" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.shape` expects a single tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(_) => Ok((ValueType::ScalarI32, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.shape` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        "tensor.sum" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.sum` expects a single tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    Ok((ValueType::Tensor(TensorType::new(tensor.dtype, Vec::new())), span))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.sum` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        "tensor.dtype" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.dtype` expects a single tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(_) => Ok((ValueType::ScalarI32, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.dtype` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        "tensor.print" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.print` expects a single argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            if matches!(arg_ty, ValueType::Tensor(_) | ValueType::ScalarI32) {
                Ok((arg_ty, span))
            } else {
                Err(TypeErrSpan {
                    msg: "`tensor.print` requires a tensor or scalar argument".to_string(),
                    span,
                })
            }
        }
        #[cfg(feature = "cpu-buffers")]
        "tensor.materialize" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.materialize` expects a tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty.clone() {
                ValueType::Tensor(_) => Ok((arg_ty, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.materialize` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        #[cfg(feature = "cpu-buffers")]
        "tensor.is_materialized" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.is_materialized` expects a tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(_) => Ok((ValueType::ScalarI32, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.is_materialized` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        #[cfg(feature = "cpu-buffers")]
        "tensor.sample" => {
            if args.len() != 2 {
                return Err(TypeErrSpan {
                    msg: "`tensor.sample` expects (tensor, count) arguments".to_string(),
                    span,
                });
            }
            let (tensor_ty, _) = infer_expr(&args[0], env)?;
            let (count_ty, _) = infer_expr(&args[1], env)?;
            match (tensor_ty, count_ty) {
                (ValueType::Tensor(tensor), ValueType::ScalarI32) => Ok((
                    ValueType::Tensor(TensorType::new(
                        tensor.dtype,
                        vec![ShapeDim::Sym("_sample")],
                    )),
                    span,
                )),
                (ValueType::Tensor(_), _) => Err(TypeErrSpan {
                    msg: "`tensor.sample` requires the second argument to be an integer"
                        .to_string(),
                    span,
                }),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.sample` requires a tensor and an integer argument".to_string(),
                    span,
                }),
            }
        }
        _ => Err(TypeErrSpan { msg: format!("unsupported call to `{callee}`"), span }),
    }
}

fn infer_dtype_arg(node: &Node) -> Result<DType, TypeErrSpan> {
    match node {
        Node::Lit(Literal::Ident(name), span) => DType::from_str(name)
            .ok_or(TypeErrSpan { msg: format!("unknown dtype `{name}`"), span: *span }),
        _ => Err(TypeErrSpan { msg: "expected dtype identifier".to_string(), span: node.span() }),
    }
}

fn infer_shape_arg(node: &Node) -> Result<Vec<ShapeDim>, TypeErrSpan> {
    match node {
        Node::Tuple { .. } | Node::Paren(..) | Node::Lit(..) => infer_shape_node(node),
        _ => infer_shape_node(node),
    }
}

fn infer_shape_node(node: &Node) -> Result<Vec<ShapeDim>, TypeErrSpan> {
    match node {
        Node::Tuple { elements, .. } => {
            let mut dims = Vec::new();
            for el in elements {
                dims.extend(infer_shape_node(el)?);
            }
            Ok(dims)
        }
        Node::Paren(inner, _) => infer_shape_node(inner),
        Node::Lit(Literal::Int(n), span) => {
            if *n < 0 {
                Err(TypeErrSpan {
                    msg: "shape dimensions must be non-negative".to_string(),
                    span: *span,
                })
            } else {
                Ok(vec![ShapeDim::Known(*n as usize)])
            }
        }
        Node::Lit(Literal::Ident(name), _span) => Ok(vec![ShapeDim::Sym(leak_symbol(name))]),
        _ => Err(TypeErrSpan { msg: "unsupported shape literal".to_string(), span: node.span() }),
    }
}

fn leak_symbol(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
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
