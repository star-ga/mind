// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::collections::BTreeSet;
use std::collections::HashMap;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::ast::BinOp;

use crate::ast::Literal;

use crate::ast::Module;

use crate::ast::Node;

use crate::ast::Span as AstSpan;

use crate::diagnostics::{Diagnostic as Pretty, Severity, Span};

use crate::linalg;
use crate::shapes::engine;
use crate::types::ConvPadding;
use crate::types::DType;
use crate::types::ShapeDim;
use crate::types::TensorType;
use crate::types::ValueType;

#[derive(Debug)]
pub struct TypeErrSpan {
    pub msg: String,
    pub span: AstSpan,
}

pub type TypeEnv = HashMap<String, ValueType>;

const TYPE_ERR_CODE: &str = "E2001";
const SHAPE_BROADCAST_CODE: &str = "E2101";
const SHAPE_RANK_CODE: &str = "E2102";
const SHAPE_INNER_DIM_CODE: &str = "E2103";

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

fn format_usize_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({})", dims.join(","))
}

fn describe_tensor(tensor: &TensorType) -> String {
    format!(
        "Tensor[{}, {}]",
        dtype_name(&tensor.dtype),
        format_shape(&tensor.shape)
    )
}

fn describe_value_type(v: &ValueType) -> String {
    match v {
        ValueType::ScalarI32 => "Scalar[i32]".to_string(),
        ValueType::ScalarI64 => "Scalar[i64]".to_string(),
        ValueType::ScalarF32 => "Scalar[f32]".to_string(),
        ValueType::ScalarF64 => "Scalar[f64]".to_string(),
        ValueType::ScalarBool => "Scalar[bool]".to_string(),
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

fn dim_display(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
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

fn shape_op_for_binop(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "tensor.add",
        BinOp::Sub => "tensor.sub",
        BinOp::Mul => "tensor.mul",
        BinOp::Div => "tensor.div",
    }
}

fn concrete_shape(shape: &[ShapeDim]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            ShapeDim::Known(n) => out.push(*n),
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(out)
}

fn shape_from_usize(shape: &[usize]) -> Vec<ShapeDim> {
    shape.iter().copied().map(ShapeDim::Known).collect()
}

fn shape_engine_error(op_display: &str, err: engine::ShapeError, span: AstSpan) -> TypeErrSpan {
    match err.kind {
        engine::ShapeErrorKind::UnknownOp => TypeErrSpan {
            msg: format!("shape rule not defined for `{op_display}`"),
            span,
        },
        engine::ShapeErrorKind::RankMismatch {
            expected,
            actual_lhs,
            actual_rhs,
        } => {
            let expected_display = if op_display.contains("matmul") && actual_rhs.is_some() {
                format!(
                    "matmul inner dimension mismatch (lhs.shape[1]={} vs rhs.shape[0]={})",
                    actual_lhs.get(1).copied().unwrap_or(0),
                    actual_rhs
                        .as_ref()
                        .and_then(|rhs| rhs.first().copied())
                        .unwrap_or(0)
                )
            } else {
                expected
            };
            let rhs_str = actual_rhs
                .as_ref()
                .map(|rhs| format!(", rhs={}", format_usize_shape(rhs)))
                .unwrap_or_default();
            TypeErrSpan {
                msg: format!(
                    "rank mismatch for `{op_display}`: expected {expected_display}, got lhs={}{}",
                    format_usize_shape(&actual_lhs),
                    rhs_str
                ),
                span,
            }
        }
        engine::ShapeErrorKind::BroadcastError { lhs, rhs } => TypeErrSpan {
            msg: format!(
                "cannot broadcast shapes {} and {} for `{op_display}`",
                format_usize_shape(&lhs),
                format_usize_shape(&rhs)
            ),
            span,
        },
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
        | (ValueType::ScalarI32, ValueType::Tensor(t))
        | (ValueType::Tensor(t), ValueType::ScalarF32)
        | (ValueType::ScalarF32, ValueType::Tensor(t)) => {
            if t.dtype == DType::F32 {
                promote_scalar_to(t.dtype.clone()).map(|_| DType::F32)
            } else {
                None
            }
        }
        (ValueType::ScalarI32, ValueType::ScalarI32) => None,
        (ValueType::ScalarF32, ValueType::ScalarF32) => Some(DType::F32),
        (ValueType::ScalarBool, ValueType::ScalarBool) => None,
        (ValueType::GradMap(_), _) | (_, ValueType::GradMap(_)) => None,
        _ => None, // Other scalar combinations
    }
}

fn linalg_type_err(op: &str, span: AstSpan, msg: String) -> TypeErrSpan {
    TypeErrSpan {
        msg: format!("`{op}`: {msg}"),
        span,
    }
}

fn normalize_axis(axis: i32, rank: usize, span: AstSpan, op: &str) -> Result<usize, TypeErrSpan> {
    let rank_i32 = rank as i32;
    let idx = if axis < 0 { rank_i32 + axis } else { axis };
    if idx < 0 || idx >= rank_i32 {
        Err(TypeErrSpan {
            msg: format!("axis {axis} out of range for `{op}` (rank {rank})"),
            span,
        })
    } else {
        Ok(idx as usize)
    }
}

fn dim_len(dim: &ShapeDim) -> Option<usize> {
    match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    }
}

fn slice_len(start: i32, end: i32) -> Option<usize> {
    if start < 0 || end < start {
        None
    } else {
        Some((end - start) as usize)
    }
}

fn slice_len_with_step(len: Option<usize>, start: i32, end: i32, step: i32) -> Option<usize> {
    if step == 0 {
        return None;
    }
    let len = len?;
    let len_i = len as i64;
    let step_i = step as i64;

    let mut start_i = start as i64;
    let mut end_i = end as i64;

    if step_i > 0 {
        if start_i < 0 {
            start_i += len_i;
        }
        if start_i < 0 {
            start_i = 0;
        }
        if start_i > len_i {
            start_i = len_i;
        }

        if end_i < 0 {
            end_i += len_i;
        }
        if end_i < 0 {
            end_i = 0;
        }
        if end_i > len_i {
            end_i = len_i;
        }

        if start_i >= end_i {
            Some(0)
        } else {
            let diff = end_i - start_i;
            Some(((diff + step_i.abs() - 1) / step_i.abs()) as usize)
        }
    } else {
        if len == 0 {
            return Some(0);
        }

        if start_i < 0 {
            start_i += len_i;
        }
        if start_i < -1 {
            start_i = -1;
        }
        if start_i >= len_i {
            start_i = len_i - 1;
        }

        if end_i < 0 {
            end_i += len_i;
        }
        if end_i < -1 {
            end_i = -1;
        }
        if end_i >= len_i {
            end_i = len_i - 1;
        }

        if start_i <= end_i {
            Some(0)
        } else {
            let diff = start_i - end_i;
            Some(((diff + (-step_i) - 1) / (-step_i)) as usize)
        }
    }
}

fn conv_channels_compatible(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(sa), ShapeDim::Sym(sb)) => sa == sb,
        _ => true,
    }
}

fn conv_output_dim(
    input: &ShapeDim,
    kernel: Option<&ShapeDim>,
    stride: usize,
    padding: ConvPadding,
    span: AstSpan,
    axis: &str,
) -> Result<ShapeDim, TypeErrSpan> {
    let input_known = dim_len(input);
    let kernel_known = kernel.and_then(dim_len);
    let result = match padding {
        ConvPadding::Valid => linalg::conv_output_dim_valid(input_known, kernel_known, stride),
        ConvPadding::Same => linalg::conv_output_dim_same(input_known, stride),
    };
    match result {
        Ok(Some(v)) => Ok(ShapeDim::Known(v)),
        Ok(None) => Ok(ShapeDim::Sym(fresh_symbol(&format!("_conv_{axis}")))),
        Err(msg) => Err(TypeErrSpan {
            msg: format!("`tensor.conv2d`: {msg} ({axis})"),
            span,
        }),
    }
}

fn normalize_axes_list(
    axes: &[i32],
    rank: usize,
    span: AstSpan,
    op: &str,
) -> Result<Vec<usize>, TypeErrSpan> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut normalized = Vec::new();
    for &axis in axes {
        let idx = normalize_axis(axis, rank, span, op)?;
        if !seen.insert(idx) {
            return Err(TypeErrSpan {
                msg: format!("duplicate axis {axis} in `{op}`"),
                span,
            });
        }
        normalized.push(idx);
    }
    normalized.sort_unstable();
    Ok(normalized)
}

fn normalize_reduce_axes(
    axes: &[i32],
    rank: usize,
    span: AstSpan,
    op: &str,
) -> Result<Vec<usize>, TypeErrSpan> {
    if axes.is_empty() {
        return Ok((0..rank).collect());
    }
    normalize_axes_list(axes, rank, span, op)
}

fn reduce_shape(shape: &[ShapeDim], axes: &[usize], keepdims: bool) -> Vec<ShapeDim> {
    if keepdims {
        let mut out = shape.to_vec();
        for &axis in axes {
            if axis < out.len() {
                out[axis] = ShapeDim::Known(1);
            }
        }
        out
    } else {
        let axis_set: BTreeSet<usize> = axes.iter().cloned().collect();
        let mut out = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if !axis_set.contains(&idx) {
                out.push(dim.clone());
            }
        }
        out
    }
}

fn known_product(shape: &[ShapeDim]) -> Option<usize> {
    let mut total = 1usize;
    for dim in shape {
        match dim {
            ShapeDim::Known(n) => {
                total = total.checked_mul(*n)?;
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(total)
}

fn normalize_expand_axis(axis: i32, rank: usize, span: AstSpan) -> Result<usize, TypeErrSpan> {
    let extended = rank + 1;
    let idx = if axis < 0 {
        (extended as i32) + axis
    } else {
        axis
    };
    if idx < 0 || idx > extended as i32 - 1 {
        Err(TypeErrSpan {
            msg: format!("axis {axis} out of range for `tensor.expand_dims` (rank {rank})"),
            span,
        })
    } else {
        Ok(idx as usize)
    }
}

fn compute_squeeze_axes(
    shape: &[ShapeDim],
    axes: &[i32],
    span: AstSpan,
) -> Result<Vec<usize>, TypeErrSpan> {
    if axes.is_empty() {
        let mut remove = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if matches!(dim, ShapeDim::Known(1)) {
                remove.push(idx);
            }
        }
        return Ok(remove);
    }
    let normalized = normalize_axes_list(axes, shape.len(), span, "tensor.squeeze")?;
    for &axis in &normalized {
        match shape.get(axis) {
            Some(ShapeDim::Known(1)) => {}
            Some(_) => {
                return Err(TypeErrSpan {
                    msg: format!("cannot squeeze axis {axis}: dimension is not 1"),
                    span,
                });
            }
            None => {
                return Err(TypeErrSpan {
                    msg: format!("axis {axis} out of range for `tensor.squeeze`"),
                    span,
                });
            }
        }
    }
    Ok(normalized)
}

fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
    let mut out = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;

    while i >= 0 || j >= 0 {
        let da = if i >= 0 {
            a[i as usize].clone()
        } else {
            ShapeDim::Known(1)
        };
        let db = if j >= 0 {
            b[j as usize].clone()
        } else {
            ShapeDim::Known(1)
        };

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
        Node::Lit(Literal::Ident(name), span) => env
            .get(name)
            .cloned()
            .map(|t| (t, *span))
            .ok_or_else(|| TypeErrSpan {
                msg: format!("unknown identifier `{name}`"),
                span: *span,
            }),
        Node::Paren(inner, span) => {
            let (ty, _) = infer_expr(inner, env)?;
            Ok((ty, *span))
        }
        Node::Tuple { elements, span } => {
            // Infer type from the last element (matches lowering semantics).
            if let Some(last) = elements.last() {
                let (ty, _) = infer_expr(last, env)?;
                Ok((ty, *span))
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::Call { callee, args, span } => infer_call(callee, args, *span, env),
        Node::CallGrad { loss, wrt, span } => infer_grad(loss, wrt, *span, env),
        Node::CallTensorSum {
            x,
            axes,
            keepdims,
            span,
        } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axes_norm =
                        normalize_reduce_axes(axes, tensor.shape.len(), *span, "tensor.sum")?;
                    let shape = reduce_shape(&tensor.shape, &axes_norm, *keepdims);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.sum` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallTensorMean {
            x,
            axes,
            keepdims,
            span,
        } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axes_norm =
                        normalize_reduce_axes(axes, tensor.shape.len(), *span, "tensor.mean")?;
                    let shape = reduce_shape(&tensor.shape, &axes_norm, *keepdims);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.mean` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallReshape { x, dims, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let new_shape = shape_from_dims(dims);
                    if new_shape.len() != tensor.shape.len() {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.reshape` expects {} dimensions but got {}",
                                tensor.shape.len(),
                                new_shape.len()
                            ),
                            span: *span,
                        });
                    }
                    if let (Some(old), Some(new)) =
                        (known_product(&tensor.shape), known_product(&new_shape))
                    {
                        if old != new {
                            return Err(TypeErrSpan {
                                msg: format!(
                                    "`tensor.reshape` element count mismatch: {old} vs {new}"
                                ),
                                span: *span,
                            });
                        }
                    }
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, new_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.reshape` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallExpandDims { x, axis, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let rank = tensor.shape.len();
                    let axis = normalize_expand_axis(*axis, rank, *span)?;
                    let mut shape = tensor.shape;
                    shape.insert(axis, ShapeDim::Known(1));
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.expand_dims` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallSqueeze { x, axes, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axes_to_remove = compute_squeeze_axes(&tensor.shape, axes, *span)?;
                    let axis_set: BTreeSet<usize> = axes_to_remove.iter().cloned().collect();
                    let mut shape = Vec::new();
                    for (idx, dim) in tensor.shape.iter().enumerate() {
                        if !axis_set.contains(&idx) {
                            shape.push(dim.clone());
                        }
                    }
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.squeeze` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallTranspose { x, axes, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let rank = tensor.shape.len();
                    let perm = if let Some(list) = axes {
                        linalg::normalize_permutation(list, rank)
                            .map_err(|msg| linalg_type_err("tensor.transpose", *span, msg))?
                    } else {
                        linalg::default_transpose(rank)
                    };
                    if perm.len() != rank {
                        return Err(linalg_type_err(
                            "tensor.transpose",
                            *span,
                            format!("expected {} axes but got {}", rank, perm.len()),
                        ));
                    }
                    let shape = linalg::permute_shape(&tensor.shape, &perm);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.transpose` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallIndex { x, axis, i, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    if tensor.shape.is_empty() {
                        return Err(TypeErrSpan {
                            msg: "`tensor.index` requires a tensor with rank >= 1".to_string(),
                            span: *span,
                        });
                    }
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.index")?;
                    if let Some(len) = dim_len(&tensor.shape[axis_norm]) {
                        if *i < 0 || (*i as usize) >= len {
                            return Err(TypeErrSpan {
                                msg: format!(
                                    "`tensor.index`: index {i} out of bounds for axis {axis_norm} (len {len})"
                                ),
                                span: *span,
                            });
                        }
                    }
                    let mut shape = tensor.shape.clone();
                    shape.remove(axis_norm);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.index` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallSlice {
            x,
            axis,
            start,
            end,
            span,
        } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    if *start < 0 || *end < *start {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.slice` requires 0 <= start <= end (got start={start}, end={end})"
                            ),
                            span: *span,
                        });
                    }
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.slice")?;
                    if let Some(len) = dim_len(&tensor.shape[axis_norm]) {
                        if *end as usize > len {
                            return Err(TypeErrSpan {
                                msg: format!(
                                    "`tensor.slice`: end {end} out of bounds for axis {axis_norm} (len {len})"
                                ),
                                span: *span,
                            });
                        }
                    }
                    let new_dim = match (dim_len(&tensor.shape[axis_norm]), slice_len(*start, *end))
                    {
                        (Some(_), Some(len)) => ShapeDim::Known(len),
                        _ => ShapeDim::Sym(fresh_symbol("_slice")),
                    };
                    let mut shape = tensor.shape.clone();
                    shape[axis_norm] = new_dim;
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.slice` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallSliceStride {
            x,
            axis,
            start,
            end,
            step,
            span,
        } => {
            if *step == 0 {
                return Err(TypeErrSpan {
                    msg: "`tensor.slice_stride` requires step != 0".to_string(),
                    span: *span,
                });
            }
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.slice_stride")?;
                    let dim = tensor.shape[axis_norm].clone();
                    let new_dim = if let Some(len) = dim_len(&dim) {
                        let Some(result_len) = slice_len_with_step(Some(len), *start, *end, *step)
                        else {
                            return Err(TypeErrSpan {
                                msg: "`tensor.slice_stride` bounds are invalid for the axis"
                                    .to_string(),
                                span: *span,
                            });
                        };
                        ShapeDim::Known(result_len)
                    } else if (*step > 0 && *start >= *end) || (*step < 0 && *start <= *end) {
                        ShapeDim::Known(0)
                    } else {
                        ShapeDim::Sym(fresh_symbol("_slice_stride"))
                    };
                    let mut shape = tensor.shape.clone();
                    shape[axis_norm] = new_dim;
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.slice_stride` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallGather { x, axis, idx, span } => {
            let (x_ty, _) = infer_expr(x, env)?;
            let (idx_ty, idx_span) = infer_expr(idx, env)?;
            match (x_ty, idx_ty) {
                (ValueType::Tensor(tensor), ValueType::Tensor(idx_tensor)) => {
                    if idx_tensor.dtype != DType::I32 {
                        return Err(TypeErrSpan {
                            msg: "`tensor.gather` requires `idx` to be an i32 tensor".to_string(),
                            span: idx.span(),
                        });
                    }
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.gather")?;
                    let mut shape = Vec::new();
                    shape.extend_from_slice(&tensor.shape[..axis_norm]);
                    shape.extend(idx_tensor.shape.iter().cloned());
                    if axis_norm < tensor.shape.len() {
                        shape.extend_from_slice(&tensor.shape[axis_norm + 1..]);
                    }
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                (ValueType::Tensor(_), _) => Err(TypeErrSpan {
                    msg: "`tensor.gather` requires `idx` to be a tensor".to_string(),
                    span: idx_span,
                }),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.gather` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallDot { a, b, span } => {
            let (lt, _) = infer_expr(a, env)?;
            let (rt, _) = infer_expr(b, env)?;
            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if tl.dtype != tr.dtype {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.dot` dtype mismatch: left {} vs right {}",
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        });
                    }
                    let info = linalg::compute_matmul_shape_info(&tl.shape, &tr.shape)
                        .map_err(|msg| linalg_type_err("tensor.dot", *span, msg))?;
                    Ok((
                        ValueType::Tensor(TensorType::new(tl.dtype.clone(), info.result_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.dot` requires tensor arguments".to_string(),
                    span: *span,
                }),
            }
        }
        Node::CallMatMul { a, b, span } => {
            let (lt, _) = infer_expr(a, env)?;
            let (rt, _) = infer_expr(b, env)?;
            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if tl.dtype != tr.dtype {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.matmul` dtype mismatch: left {} vs right {}",
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        });
                    }
                    if let (Some(lhs), Some(rhs)) =
                        (concrete_shape(&tl.shape), concrete_shape(&tr.shape))
                    {
                        match engine::infer_output_shape("tensor.matmul", &[&lhs, &rhs]) {
                            Ok(out) => {
                                return Ok((
                                    ValueType::Tensor(TensorType::new(
                                        tl.dtype.clone(),
                                        shape_from_usize(&out),
                                    )),
                                    *span,
                                ))
                            }
                            Err(e) => return Err(shape_engine_error("tensor.matmul", e, *span)),
                        }
                    }
                    let info = linalg::compute_matmul_shape_info(&tl.shape, &tr.shape)
                        .map_err(|msg| linalg_type_err("tensor.matmul", *span, msg))?;
                    Ok((
                        ValueType::Tensor(TensorType::new(tl.dtype.clone(), info.result_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.matmul` requires tensor arguments".to_string(),
                    span: *span,
                }),
            }
        }
        Node::CallTensorRelu { x, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => Ok((ValueType::Tensor(tensor), *span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.relu` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallTensorConv2d {
            x,
            w,
            stride_h,
            stride_w,
            padding,
            span,
        } => {
            if *stride_h == 0 || *stride_w == 0 {
                return Err(TypeErrSpan {
                    msg: "`tensor.conv2d`: strides must be positive".to_string(),
                    span: *span,
                });
            }

            let (x_ty, _) = infer_expr(x, env)?;
            let (w_ty, _) = infer_expr(w, env)?;
            let (x_tensor, w_tensor) = match (x_ty, w_ty) {
                (ValueType::Tensor(a), ValueType::Tensor(b)) => (a, b),
                (ValueType::Tensor(_), other) => {
                    return Err(TypeErrSpan {
                        msg: format!(
                            "`tensor.conv2d`: expected tensor weights but found {}",
                            describe_value_type(&other)
                        ),
                        span: w.span(),
                    });
                }
                (other, _) => {
                    return Err(TypeErrSpan {
                        msg: format!(
                            "`tensor.conv2d`: expected tensor input but found {}",
                            describe_value_type(&other)
                        ),
                        span: x.span(),
                    });
                }
            };

            if x_tensor.shape.len() != 4 {
                return Err(TypeErrSpan {
                    msg: "`tensor.conv2d` expects input layout NHWC (rank 4)".to_string(),
                    span: x.span(),
                });
            }
            if w_tensor.shape.len() != 4 {
                return Err(TypeErrSpan {
                    msg: "`tensor.conv2d` expects filter layout HWIO (rank 4)".to_string(),
                    span: w.span(),
                });
            }

            let in_channels = &x_tensor.shape[3];
            let kernel_channels = &w_tensor.shape[2];
            if !conv_channels_compatible(in_channels, kernel_channels) {
                return Err(TypeErrSpan {
                    msg: format!(
                        "`tensor.conv2d`: channel mismatch {} vs {}",
                        dim_display(in_channels),
                        dim_display(kernel_channels)
                    ),
                    span: *span,
                });
            }

            if let Some(kh) = dim_len(&w_tensor.shape[0]) {
                if kh == 0 {
                    return Err(TypeErrSpan {
                        msg: "`tensor.conv2d`: kernel height must be positive".to_string(),
                        span: w.span(),
                    });
                }
            }
            if let Some(kw) = dim_len(&w_tensor.shape[1]) {
                if kw == 0 {
                    return Err(TypeErrSpan {
                        msg: "`tensor.conv2d`: kernel width must be positive".to_string(),
                        span: w.span(),
                    });
                }
            }

            let dtype = if x_tensor.dtype == w_tensor.dtype {
                x_tensor.dtype.clone()
            } else if matches!(x_tensor.dtype, DType::F32) || matches!(w_tensor.dtype, DType::F32) {
                DType::F32
            } else {
                return Err(TypeErrSpan {
                    msg: format!(
                        "`tensor.conv2d`: incompatible dtypes {} and {}",
                        dtype_name(&x_tensor.dtype),
                        dtype_name(&w_tensor.dtype)
                    ),
                    span: *span,
                });
            };

            let out_h = conv_output_dim(
                &x_tensor.shape[1],
                Some(&w_tensor.shape[0]),
                *stride_h,
                *padding,
                *span,
                "h",
            )?;
            let out_w = conv_output_dim(
                &x_tensor.shape[2],
                Some(&w_tensor.shape[1]),
                *stride_w,
                *padding,
                *span,
                "w",
            )?;

            let out_shape = vec![
                x_tensor.shape[0].clone(),
                out_h,
                out_w,
                w_tensor.shape[3].clone(),
            ];

            Ok((ValueType::Tensor(TensorType::new(dtype, out_shape)), *span))
        }
        Node::Binary {
            op,
            left,
            right,
            span,
        } => {
            let (lt, _) = infer_expr(left, env)?;
            let (rt, _) = infer_expr(right, env)?;
            if matches!((&lt, &rt), (ValueType::ScalarI32, ValueType::ScalarI32)) {
                return Ok((ValueType::ScalarI32, *span));
            }

            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        if let (Some(lhs), Some(rhs)) =
                            (concrete_shape(&tl.shape), concrete_shape(&tr.shape))
                        {
                            match engine::infer_output_shape(shape_op_for_binop(op), &[&lhs, &rhs])
                            {
                                Ok(out) => {
                                    return Ok((
                                        ValueType::Tensor(TensorType::new(
                                            dtype,
                                            shape_from_usize(&out),
                                        )),
                                        *span,
                                    ))
                                }
                                Err(e) => {
                                    return Err(shape_engine_error(binop_display(op), e, *span))
                                }
                            }
                        }

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
                        Ok((
                            ValueType::Tensor(TensorType::new(dtype, t.shape.clone())),
                            *span,
                        ))
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
                        Err(TypeErrSpan {
                            msg: message,
                            span: *span,
                        })
                    }
                }
                _ => Err(TypeErrSpan {
                    msg: "incompatible types in binary operation".to_string(),
                    span: *span,
                }),
            }
        }
        Node::Let { value, .. } | Node::Assign { value, .. } => infer_expr(value, env),
        // Function definitions don't have a value type in expression context
        Node::FnDef { span, .. } => Ok((ValueType::ScalarI32, *span)), // Placeholder
        Node::Return { value, span } => {
            if let Some(v) = value {
                infer_expr(v, env)
            } else {
                Ok((ValueType::ScalarI32, *span)) // Void return
            }
        }
        Node::Block { stmts, span } => {
            if let Some(last) = stmts.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::If {
            then_branch, span, ..
        } => {
            if let Some(last) = then_branch.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        // Import statements don't have a value type; they're module-level declarations
        Node::Import { span, .. } => Ok((ValueType::ScalarI32, *span)),
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
                ValueType::Tensor(tensor) => Ok((
                    ValueType::Tensor(TensorType::new(tensor.dtype, Vec::new())),
                    span,
                )),
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
        _ => Err(TypeErrSpan {
            msg: format!("unsupported call to `{callee}`"),
            span,
        }),
    }
}

fn infer_dtype_arg(node: &Node) -> Result<DType, TypeErrSpan> {
    match node {
        Node::Lit(Literal::Ident(name), span) => name.parse().map_err(|_| TypeErrSpan {
            msg: format!("unknown dtype `{name}`"),
            span: *span,
        }),
        _ => Err(TypeErrSpan {
            msg: "expected dtype identifier".to_string(),
            span: node.span(),
        }),
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
        _ => Err(TypeErrSpan {
            msg: "unsupported shape literal".to_string(),
            span: node.span(),
        }),
    }
}

fn leak_symbol(name: &str) -> &'static str {
    crate::types::intern::intern_str(name)
}

fn fresh_symbol(prefix: &str) -> &'static str {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::types::intern::intern_str(&format!("{prefix}{id}"))
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
        crate::ast::TypeAnn::ScalarI64 => Some(ValueType::ScalarI64),
        crate::ast::TypeAnn::ScalarF32 => Some(ValueType::ScalarF32),
        crate::ast::TypeAnn::ScalarF64 => Some(ValueType::ScalarF64),
        crate::ast::TypeAnn::ScalarBool => Some(ValueType::ScalarBool),
        crate::ast::TypeAnn::Tensor { dtype, dims }
        | crate::ast::TypeAnn::DiffTensor { dtype, dims } => {
            let dt = dtype_from_str(dtype)?;
            let shape = shape_from_dims(dims);
            Some(ValueType::Tensor(TensorType::new(dt, shape)))
        }
    }
}

/// Walk statements; extend env on let/assign; return pretty diags for any errors.
pub fn check_module_types(module: &Module, src: &str, env: &TypeEnv) -> Vec<Pretty> {
    check_module_types_in_file(module, src, None, env)
}

pub fn check_module_types_in_file(
    module: &Module,
    src: &str,
    file: Option<&str>,
    env: &TypeEnv,
) -> Vec<Pretty> {
    let mut errs = Vec::new();
    let mut tenv = env.clone();

    for item in &module.items {
        match item {
            Node::Let {
                name,
                ann,
                value,
                span,
            } => match ann {
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
                                        file,
                                        format!(
                                            "type mismatch for `{}`: annotation {} vs inferred {}",
                                            name,
                                            describe_value_type(&vt_ann),
                                            describe_value_type(&vt)
                                        ),
                                        value.span(),
                                        TYPE_ERR_CODE,
                                    ));
                                }
                            }
                            Err(e) => errs.push(diag_from_type_err(src, file, e)),
                        }
                        tenv.insert(name.clone(), vt_ann);
                    }
                    None => errs.push(diag_from_span(
                        src,
                        file,
                        format!("unsupported annotation for `{}`", name),
                        *span,
                        TYPE_ERR_CODE,
                    )),
                },
                None => match infer_expr(value, &tenv) {
                    Ok((vt, _)) => {
                        tenv.insert(name.clone(), vt);
                    }
                    Err(e) => errs.push(diag_from_type_err(src, file, e)),
                },
            },
            Node::Assign { name, value, .. } => {
                let rhs = infer_expr(value, &tenv);
                match (tenv.get(name).cloned(), rhs) {
                    (Some(vt_lhs), Ok((vt_rhs, _))) => {
                        if vt_lhs != vt_rhs {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "cannot assign `{}`: expected {} but found {}",
                                    name,
                                    describe_value_type(&vt_lhs),
                                    describe_value_type(&vt_rhs)
                                ),
                                value.span(),
                                TYPE_ERR_CODE,
                            ));
                        }
                    }
                    (None, Ok((vt_rhs, _))) => {
                        tenv.insert(name.clone(), vt_rhs);
                    }
                    (_, Err(e)) => errs.push(diag_from_type_err(src, file, e)),
                }
            }
            // Import statements are handled at module level; skip type checking
            Node::Import { .. } => {}
            other => {
                if let Err(e) = infer_expr(other, &tenv) {
                    errs.push(diag_from_type_err(src, file, e));
                }
            }
        }
    }

    errs
}

fn diag_from_span(
    src: &str,
    file: Option<&str>,
    msg: String,
    span: AstSpan,
    code: &'static str,
) -> Pretty {
    let span = Span::from_offsets(src, span.start(), span.end(), file);
    Pretty {
        phase: "type-check",
        code,
        severity: Severity::Error,
        message: msg,
        span: Some(span),
        notes: Vec::new(),
        help: None,
    }
}

fn diag_from_type_err(src: &str, file: Option<&str>, err: TypeErrSpan) -> Pretty {
    let code = classify_error_code(&err.msg);
    diag_from_span(src, file, err.msg, err.span, code)
}

fn classify_error_code(msg: &str) -> &'static str {
    if msg.contains("inner") && msg.contains("dimension") {
        SHAPE_INNER_DIM_CODE
    } else if msg.contains("broadcast") {
        SHAPE_BROADCAST_CODE
    } else if msg.contains("rank mismatch") {
        SHAPE_RANK_CODE
    } else {
        TYPE_ERR_CODE
    }
}
