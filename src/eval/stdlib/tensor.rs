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

use crate::ast::Literal;

use crate::ast::Node;

use crate::eval::autodiff::TensorEnvEntry;
use crate::eval::eval_value_expr_mode;
use crate::eval::format_value_human;
use crate::eval::EvalError;
use crate::eval::ExecMode;
use crate::eval::TensorVal;
use crate::eval::Value;

#[cfg(feature = "cpu-buffers")]
use crate::eval::materialize_filled;
#[cfg(feature = "cpu-buffers")]
use crate::eval::num_elems;
#[cfg(feature = "cpu-buffers")]
use crate::eval::MATERIALIZE_MAX;

use crate::linalg;
use crate::linalg::MatMulShapeInfo;

use crate::types::ConvPadding;
use crate::types::DType;
use crate::types::ShapeDim;

#[cfg(feature = "cpu-buffers")]
use crate::eval::value::Buffer;

pub fn dispatch(
    callee: &str,
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    match callee {
        "tensor.zeros" => construct(args, env, tensor_env, mode.clone(), 0.0),
        "tensor.ones" => construct(args, env, tensor_env, mode.clone(), 1.0),
        "tensor.shape" => tensor_shape(args, env, tensor_env, mode.clone()),
        "tensor.dtype" => tensor_dtype(args, env, tensor_env, mode.clone()),
        "tensor.sum" => tensor_sum(args, env, tensor_env, mode.clone()),
        "tensor.print" => tensor_print(args, env, tensor_env, mode.clone()),
        #[cfg(feature = "cpu-buffers")]
        "tensor.materialize" => tensor_materialize(args, env, tensor_env, mode.clone()),
        #[cfg(feature = "cpu-buffers")]
        "tensor.sample" => tensor_sample(args, env, tensor_env, mode.clone()),
        #[cfg(feature = "cpu-buffers")]
        "tensor.is_materialized" => tensor_is_materialized(args, env, tensor_env, mode.clone()),
        "tensor.relu" => tensor_relu(args, env, tensor_env, mode.clone()),
        _ => Err(EvalError::Unsupported),
    }
}

fn tensor_relu(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    match value {
        Value::Tensor(t) => {
            let result = relu_tensor(t, mode.clone())?;
            Ok(Value::Tensor(result))
        }
        _ => Err(EvalError::Unsupported),
    }
}

fn construct(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
    fill: f64,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unsupported);
    }
    let dtype = parse_dtype(&args[0])?;
    let shape = parse_shape(&args[1], env, tensor_env, mode.clone())?;
    Ok(Value::Tensor(TensorVal::new(dtype, shape, Some(fill))))
}

fn tensor_shape(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    if let Value::Tensor(t) = value {
        let mut items = Vec::with_capacity(t.shape.len());
        for dim in &t.shape {
            match dim {
                ShapeDim::Known(n) => items.push(Value::Int(*n as i64)),
                ShapeDim::Sym(sym) => items.push(Value::Str(sym.to_string())),
            }
        }
        Ok(Value::Tuple(items))
    } else {
        Err(EvalError::Unsupported)
    }
}

fn tensor_dtype(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    if let Value::Tensor(t) = value {
        Ok(Value::Str(t.dtype.as_str().to_string()))
    } else {
        Err(EvalError::Unsupported)
    }
}

fn tensor_sum(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    if let Value::Tensor(t) = value {
        let result = sum_tensor_preview(&t, &[], false)?;
        #[cfg(feature = "cpu-buffers")]
        {
            let mut result_clone = result.clone();
            if result_clone.fill.is_some() {
                crate::eval::materialize_filled(&mut result_clone);
                return Ok(Value::Tensor(result_clone));
            }
        }
        Ok(Value::Tensor(result))
    } else {
        Err(EvalError::Unsupported)
    }
}

fn tensor_print(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    println!("{}", format_value_human(&value));
    Ok(value)
}

#[cfg(feature = "cpu-buffers")]
fn tensor_materialize(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    if let Value::Tensor(mut t) = value {
        materialize_filled(&mut t);
        Ok(Value::Tensor(t))
    } else {
        Err(EvalError::Unsupported)
    }
}

#[cfg(feature = "cpu-buffers")]
fn tensor_is_materialized(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    if let Value::Tensor(t) = value {
        Ok(Value::Int(if t.buf.is_some() { 1 } else { 0 }))
    } else {
        Err(EvalError::Unsupported)
    }
}

#[cfg(feature = "cpu-buffers")]
fn tensor_sample(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr_mode(&args[0], env, tensor_env, mode.clone())?;
    let count = eval_value_expr_mode(&args[1], env, tensor_env, mode.clone())?;
    let requested = match count {
        Value::Int(n) => {
            if n < 0 {
                0
            } else {
                n as usize
            }
        }
        _ => return Err(EvalError::Unsupported),
    };

    if let Value::Tensor(mut t) = value {
        if t.buf.is_none() {
            materialize_filled(&mut t);
        }
        if t.buf.is_none() {
            return Err(EvalError::Unsupported);
        }

        let total = match num_elems(&t.shape) {
            Some(n) => n,
            None => match &t.buf {
                Some(Buffer::I32(values)) => values.len(),
                Some(Buffer::F32(values)) => values.len(),
                None => 0,
            },
        };
        let limit = requested.min(total).min(MATERIALIZE_MAX);
        let mut sample_tensor = TensorVal::new(t.dtype.clone(), vec![ShapeDim::Known(limit)], None);
        match &t.buf {
            Some(Buffer::I32(values)) => {
                let mut out = Vec::with_capacity(limit);
                for &v in values.iter().take(limit) {
                    out.push(v);
                }
                sample_tensor.buf = Some(Buffer::I32(out));
            }
            Some(Buffer::F32(values)) => {
                let mut out = Vec::with_capacity(limit);
                for &v in values.iter().take(limit) {
                    out.push(v);
                }
                sample_tensor.buf = Some(Buffer::F32(out));
            }
            _ => return Err(EvalError::Unsupported),
        }
        Ok(Value::Tensor(sample_tensor))
    } else {
        Err(EvalError::Unsupported)
    }
}

fn parse_dtype(node: &Node) -> Result<DType, EvalError> {
    match node {
        Node::Lit(Literal::Ident(name), _) => name.parse().map_err(|_| EvalError::Unsupported),
        _ => Err(EvalError::Unsupported),
    }
}

fn parse_shape(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Vec<ShapeDim>, EvalError> {
    match node {
        Node::Tuple { elements, .. } => {
            let mut dims = Vec::with_capacity(elements.len());
            for el in elements {
                dims.push(parse_shape_dim(el, env, tensor_env, mode.clone())?);
            }
            Ok(dims)
        }
        Node::Paren(inner, _) => parse_shape(inner, env, tensor_env, mode.clone()),
        Node::Lit(Literal::Ident(name), _) => {
            if let Some(value) = env.get(name) {
                shape_from_value(value)
            } else {
                Ok(vec![ShapeDim::Sym(leak_symbol(name))])
            }
        }
        Node::Lit(Literal::Int(_), _) => {
            Ok(vec![parse_shape_dim(node, env, tensor_env, mode.clone())?])
        }
        _ => {
            let value = eval_value_expr_mode(node, env, tensor_env, mode.clone())?;
            shape_from_value(&value)
        }
    }
}

fn parse_shape_dim(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<ShapeDim, EvalError> {
    match node {
        Node::Lit(Literal::Int(n), _) => {
            if *n < 0 {
                return Err(EvalError::Unsupported);
            }
            Ok(ShapeDim::Known(*n as usize))
        }
        Node::Lit(Literal::Ident(name), _) => {
            if let Some(value) = env.get(name) {
                shape_dim_from_value(value)
            } else {
                Ok(ShapeDim::Sym(leak_symbol(name)))
            }
        }
        _ => {
            let value = eval_value_expr_mode(node, env, tensor_env, mode.clone())?;
            shape_dim_from_value(&value)
        }
    }
}

#[allow(dead_code)]
fn known_num_elems(shape: &[ShapeDim]) -> Option<usize> {
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

fn shape_from_value(value: &Value) -> Result<Vec<ShapeDim>, EvalError> {
    match value {
        Value::Tuple(items) => {
            let mut dims = Vec::with_capacity(items.len());
            for item in items {
                dims.push(shape_dim_from_value(item)?);
            }
            Ok(dims)
        }
        _ => Ok(vec![shape_dim_from_value(value)?]),
    }
}

fn shape_dim_from_value(value: &Value) -> Result<ShapeDim, EvalError> {
    match value {
        Value::Int(n) => {
            if *n < 0 {
                return Err(EvalError::Unsupported);
            }
            Ok(ShapeDim::Known(*n as usize))
        }
        Value::Str(sym) => Ok(ShapeDim::Sym(leak_symbol(sym))),
        Value::Tensor(t) => {
            if t.shape.len() == 1 {
                match t.shape[0] {
                    ShapeDim::Known(n) => Ok(ShapeDim::Known(n)),
                    ShapeDim::Sym(sym) => Ok(ShapeDim::Sym(sym)),
                }
            } else {
                Err(EvalError::Unsupported)
            }
        }
        Value::Tuple(_) => Err(EvalError::Unsupported),
        Value::GradMap(_) => Err(EvalError::Unsupported),
    }
}

fn leak_symbol(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}

fn fresh_symbol(prefix: &str) -> &'static str {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    Box::leak(format!("{prefix}{id}").into_boxed_str())
}

fn normalize_axis(axis: i32, rank: usize) -> Result<usize, EvalError> {
    let rank_i32 = rank as i32;
    let idx = if axis < 0 { rank_i32 + axis } else { axis };
    if idx < 0 || idx >= rank_i32 {
        Err(EvalError::Unsupported)
    } else {
        Ok(idx as usize)
    }
}

fn normalize_axes_list(axes: &[i32], rank: usize) -> Result<Vec<usize>, EvalError> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut normalized = Vec::new();
    for &axis in axes {
        let idx = normalize_axis(axis, rank)?;
        if !seen.insert(idx) {
            return Err(EvalError::Unsupported);
        }
        normalized.push(idx);
    }
    normalized.sort_unstable();
    Ok(normalized)
}

fn normalize_reduce_axes(axes: &[i32], rank: usize) -> Result<Vec<usize>, EvalError> {
    if axes.is_empty() {
        return Ok((0..rank).collect());
    }
    normalize_axes_list(axes, rank)
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

fn product_of_axes(shape: &[ShapeDim], axes: &[usize]) -> Option<usize> {
    if axes.is_empty() {
        return Some(1);
    }
    let mut total = 1usize;
    for &axis in axes {
        match shape.get(axis)? {
            ShapeDim::Known(n) => {
                total = total.checked_mul(*n)?;
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(total)
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

fn normalize_expand_axis(axis: i32, rank: usize) -> Result<usize, EvalError> {
    let extended = rank + 1;
    let idx = if axis < 0 {
        (extended as i32) + axis
    } else {
        axis
    };
    if idx < 0 || idx > extended as i32 - 1 {
        Err(EvalError::Unsupported)
    } else {
        Ok(idx as usize)
    }
}

fn compute_squeeze_axes(shape: &[ShapeDim], axes: &[i32]) -> Result<Vec<usize>, EvalError> {
    if axes.is_empty() {
        let mut remove = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if matches!(dim, ShapeDim::Known(1)) {
                remove.push(idx);
            }
        }
        return Ok(remove);
    }
    let normalized = normalize_axes_list(axes, shape.len())?;
    for &axis in &normalized {
        match shape.get(axis) {
            Some(ShapeDim::Known(1)) => {}
            _ => return Err(EvalError::Unsupported),
        }
    }
    Ok(normalized)
}

fn dims_from_strings(dims: &[String]) -> Vec<ShapeDim> {
    dims.iter()
        .map(|d| {
            if let Ok(n) = d.parse::<usize>() {
                ShapeDim::Known(n)
            } else {
                ShapeDim::Sym(leak_symbol(d))
            }
        })
        .collect()
}

fn relu_fill(fill: Option<f64>) -> Option<f64> {
    fill.map(|f| if f < 0.0 { 0.0 } else { f })
}

pub(crate) fn relu_tensor_preview(tensor: &TensorVal) -> Result<TensorVal, EvalError> {
    let mut result = TensorVal::new(tensor.dtype.clone(), tensor.shape.clone(), tensor.fill);
    result.fill = relu_fill(tensor.fill);
    Ok(result)
}

#[allow(unused_mut)]
pub(crate) fn relu_tensor(mut tensor: TensorVal, mode: ExecMode) -> Result<TensorVal, EvalError> {
    let preview = relu_tensor_preview(&tensor)?;

    #[cfg(feature = "cpu-buffers")]
    {
        if matches!(mode, ExecMode::CpuExec) {
            materialize_filled(&mut tensor);
            #[cfg(feature = "cpu-exec")]
            {
                if tensor.dtype == DType::F32 && tensor.as_f32().is_some() {
                    match crate::exec::cpu::exec_relu(&tensor) {
                        Ok(out) => return Ok(out),
                        Err(err) => {
                            let mapped = crate::eval::exec_error_to_eval(err);
                            if matches!(mapped, EvalError::DivZero) {
                                return Err(mapped);
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = mode;
    Ok(preview)
}

fn conv_dtype(dtype_x: &DType, dtype_w: &DType) -> Option<DType> {
    if dtype_x == dtype_w {
        Some(dtype_x.clone())
    } else if matches!(dtype_x, DType::F32) || matches!(dtype_w, DType::F32) {
        Some(DType::F32)
    } else {
        None
    }
}

fn conv_channels_match(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(sa), ShapeDim::Sym(sb)) => sa == sb,
        _ => true,
    }
}

fn conv_output_dim_eval(
    input: &ShapeDim,
    kernel: Option<&ShapeDim>,
    stride: usize,
    padding: ConvPadding,
) -> Result<ShapeDim, EvalError> {
    let input_known = match input {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    };
    let kernel_known = kernel.and_then(|dim| match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    });
    let res = match padding {
        ConvPadding::Valid => linalg::conv_output_dim_valid(input_known, kernel_known, stride),
        ConvPadding::Same => linalg::conv_output_dim_same(input_known, stride),
    };
    match res {
        Ok(Some(v)) => Ok(ShapeDim::Known(v)),
        Ok(None) => Ok(ShapeDim::Sym(fresh_symbol("_conv"))),
        Err(msg) => Err(EvalError::UnsupportedMsg(format!("`tensor.conv2d`: {msg}"))),
    }
}

pub(crate) fn conv2d_tensor_preview(
    x: &TensorVal,
    w: &TensorVal,
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> Result<TensorVal, EvalError> {
    if stride_h == 0 || stride_w == 0 {
        return Err(EvalError::UnsupportedMsg(
            "`tensor.conv2d`: strides must be positive".to_string(),
        ));
    }

    if x.shape.len() != 4 {
        return Err(EvalError::UnsupportedMsg(
            "`tensor.conv2d` expects input layout NHWC (rank 4)".to_string(),
        ));
    }
    if w.shape.len() != 4 {
        return Err(EvalError::UnsupportedMsg(
            "`tensor.conv2d` expects filter layout HWIO (rank 4)".to_string(),
        ));
    }

    if !conv_channels_match(&x.shape[3], &w.shape[2]) {
        return Err(EvalError::UnsupportedMsg(format!(
            "`tensor.conv2d`: channel mismatch {} vs {}",
            dim_str(&x.shape[3]),
            dim_str(&w.shape[2])
        )));
    }

    if matches!(w.shape[0], ShapeDim::Known(0)) {
        return Err(EvalError::UnsupportedMsg(
            "`tensor.conv2d`: kernel height must be positive".to_string(),
        ));
    }
    if matches!(w.shape[1], ShapeDim::Known(0)) {
        return Err(EvalError::UnsupportedMsg(
            "`tensor.conv2d`: kernel width must be positive".to_string(),
        ));
    }

    let dtype = conv_dtype(&x.dtype, &w.dtype).ok_or_else(|| {
        EvalError::UnsupportedMsg(format!(
            "`tensor.conv2d`: incompatible dtypes {} and {}",
            x.dtype.as_str(),
            w.dtype.as_str()
        ))
    })?;

    let out_h = conv_output_dim_eval(&x.shape[1], Some(&w.shape[0]), stride_h, padding)?;
    let out_w = conv_output_dim_eval(&x.shape[2], Some(&w.shape[1]), stride_w, padding)?;

    let out_shape = vec![x.shape[0].clone(), out_h, out_w, w.shape[3].clone()];

    Ok(TensorVal::new(dtype, out_shape, None))
}

fn dim_str(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

#[allow(unused_mut)]
pub(crate) fn conv2d_tensor(
    mut x: TensorVal,
    mut w: TensorVal,
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
    mode: ExecMode,
) -> Result<TensorVal, EvalError> {
    let preview = conv2d_tensor_preview(&x, &w, stride_h, stride_w, padding)?;

    #[cfg(feature = "cpu-buffers")]
    {
        if matches!(mode, ExecMode::CpuExec) {
            materialize_filled(&mut x);
            materialize_filled(&mut w);
            #[cfg(all(feature = "cpu-exec", feature = "cpu-conv"))]
            {
                if x.dtype == DType::F32
                    && w.dtype == DType::F32
                    && x.as_f32().is_some()
                    && w.as_f32().is_some()
                {
                    match crate::exec::conv::exec_conv2d(&x, &w, stride_h, stride_w, padding) {
                        Ok(t) => return Ok(t),
                        Err(err) => {
                            return Err(crate::eval::exec_error_to_eval(err));
                        }
                    }
                }
            }
            #[cfg(all(feature = "cpu-exec", not(feature = "cpu-conv")))]
            {
                return Err(EvalError::UnsupportedMsg(
                    "`tensor.conv2d` execution requires enabling the `cpu-conv` feature"
                        .to_string(),
                ));
            }
        }
    }

    let _ = mode;
    Ok(preview)
}

pub(crate) fn sum_tensor_preview(
    tensor: &TensorVal,
    axes: &[i32],
    keepdims: bool,
) -> Result<TensorVal, EvalError> {
    let axes_norm = normalize_reduce_axes(axes, tensor.shape.len())?;
    let shape = reduce_shape(&tensor.shape, &axes_norm, keepdims);
    let mut result = TensorVal::new(tensor.dtype.clone(), shape, None);
    if let Some(fill) = tensor.fill {
        if let Some(count) = product_of_axes(&tensor.shape, &axes_norm) {
            result.fill = Some(fill * count as f64);
        }
    }
    Ok(result)
}

pub(crate) fn mean_tensor_preview(
    tensor: &TensorVal,
    axes: &[i32],
    keepdims: bool,
) -> Result<TensorVal, EvalError> {
    let axes_norm = normalize_reduce_axes(axes, tensor.shape.len())?;
    let shape = reduce_shape(&tensor.shape, &axes_norm, keepdims);
    let mut result = TensorVal::new(tensor.dtype.clone(), shape, tensor.fill);
    result.fill = tensor.fill;
    Ok(result)
}

pub(crate) fn reshape_tensor_preview(
    tensor: &TensorVal,
    dims: &[String],
) -> Result<TensorVal, EvalError> {
    let new_shape = dims_from_strings(dims);
    if new_shape.len() != tensor.shape.len() {
        return Err(EvalError::Unsupported);
    }
    if let (Some(old), Some(new)) = (known_product(&tensor.shape), known_product(&new_shape)) {
        if old != new {
            return Err(EvalError::Unsupported);
        }
    }
    Ok(TensorVal::new(tensor.dtype.clone(), new_shape, tensor.fill))
}

pub(crate) fn expand_dims_tensor_preview(
    tensor: &TensorVal,
    axis: i32,
) -> Result<TensorVal, EvalError> {
    let rank = tensor.shape.len();
    let axis = normalize_expand_axis(axis, rank)?;
    let mut shape = tensor.shape.clone();
    shape.insert(axis, ShapeDim::Known(1));
    Ok(TensorVal::new(tensor.dtype.clone(), shape, tensor.fill))
}

pub(crate) fn squeeze_tensor_preview(
    tensor: &TensorVal,
    axes: &[i32],
) -> Result<TensorVal, EvalError> {
    let axes_to_remove = compute_squeeze_axes(&tensor.shape, axes)?;
    let axis_set: BTreeSet<usize> = axes_to_remove.iter().cloned().collect();
    let mut shape = Vec::new();
    for (idx, dim) in tensor.shape.iter().enumerate() {
        if !axis_set.contains(&idx) {
            shape.push(dim.clone());
        }
    }
    Ok(TensorVal::new(tensor.dtype.clone(), shape, tensor.fill))
}

fn matmul_shape_info(a: &TensorVal, b: &TensorVal, op: &str) -> Result<MatMulShapeInfo, EvalError> {
    if a.dtype != b.dtype {
        return Err(EvalError::UnsupportedMsg(format!(
            "`{}` dtype mismatch: left {:?} vs right {:?}",
            op, a.dtype, b.dtype
        )));
    }
    linalg::compute_matmul_shape_info(&a.shape, &b.shape)
        .map_err(|msg| EvalError::UnsupportedMsg(format!("`{op}`: {msg}")))
}

pub(crate) fn transpose_tensor_preview(
    tensor: &TensorVal,
    axes: Option<&[i32]>,
) -> Result<(TensorVal, Vec<usize>), EvalError> {
    let rank = tensor.shape.len();
    let perm = if let Some(spec) = axes {
        linalg::normalize_permutation(spec, rank)
            .map_err(|msg| EvalError::UnsupportedMsg(format!("`tensor.transpose`: {msg}")))?
    } else {
        linalg::default_transpose(rank)
    };
    let shape = linalg::permute_shape(&tensor.shape, &perm);
    Ok((
        TensorVal::new(tensor.dtype.clone(), shape, tensor.fill),
        perm,
    ))
}

pub(crate) fn index_tensor_preview(
    tensor: &TensorVal,
    axis: i32,
    i: i32,
) -> Result<TensorVal, EvalError> {
    let rank = tensor.shape.len();
    if rank == 0 {
        return Err(EvalError::Unsupported);
    }
    let axis = normalize_axis(axis, rank)?;
    if let ShapeDim::Known(n) = tensor.shape[axis] {
        if i < 0 || (i as usize) >= n {
            return Err(EvalError::OutOfBounds);
        }
    }
    let mut shape = tensor.shape.clone();
    shape.remove(axis);
    Ok(TensorVal::new(tensor.dtype.clone(), shape, tensor.fill))
}

pub(crate) fn slice_tensor_preview(
    tensor: &TensorVal,
    axis: i32,
    start: i32,
    end: i32,
) -> Result<TensorVal, EvalError> {
    if start < 0 || end < start {
        return Err(EvalError::Unsupported);
    }
    let rank = tensor.shape.len();
    let axis = normalize_axis(axis, rank)?;
    let mut shape = tensor.shape.clone();
    let new_dim = match tensor.shape[axis].clone() {
        ShapeDim::Known(n) => {
            if end as usize > n {
                return Err(EvalError::OutOfBounds);
            }
            ShapeDim::Known((end - start) as usize)
        }
        ShapeDim::Sym(sym) => ShapeDim::Sym(sym),
    };
    shape[axis] = new_dim;
    Ok(TensorVal::new(tensor.dtype.clone(), shape, tensor.fill))
}

pub(crate) fn slice_stride_tensor_preview(
    tensor: &TensorVal,
    axis: i32,
    start: i32,
    end: i32,
    step: i32,
) -> Result<TensorVal, EvalError> {
    if step == 0 {
        return Err(EvalError::Unsupported);
    }
    let rank = tensor.shape.len();
    let axis = normalize_axis(axis, rank)?;
    let mut shape = tensor.shape.clone();
    let new_dim = match tensor.shape[axis].clone() {
        ShapeDim::Known(n) => {
            let Some(len) = slice_len_with_step(Some(n), start, end, step) else {
                return Err(EvalError::Unsupported);
            };
            ShapeDim::Known(len)
        }
        ShapeDim::Sym(_) => {
            if (step > 0 && start >= end) || (step < 0 && start <= end) {
                ShapeDim::Known(0)
            } else {
                ShapeDim::Sym(fresh_symbol("_slice_stride"))
            }
        }
    };
    shape[axis] = new_dim;
    let fill = if tensor.fill.is_some() {
        tensor.fill
    } else {
        None
    };
    Ok(TensorVal::new(tensor.dtype.clone(), shape, fill))
}

pub(crate) fn gather_tensor_preview(
    tensor: &TensorVal,
    axis: i32,
    idx: &TensorVal,
) -> Result<TensorVal, EvalError> {
    if idx.dtype != DType::I32 {
        return Err(EvalError::Unsupported);
    }
    let axis = normalize_axis(axis, tensor.shape.len())?;
    let mut shape = Vec::new();
    shape.extend_from_slice(&tensor.shape[..axis]);
    shape.extend(idx.shape.iter().cloned());
    if axis < tensor.shape.len() {
        shape.extend_from_slice(&tensor.shape[axis + 1..]);
    }
    Ok(TensorVal::new(tensor.dtype.clone(), shape, tensor.fill))
}

fn matmul_fill(a: &TensorVal, b: &TensorVal, info: &MatMulShapeInfo) -> Option<f64> {
    match (a.fill, b.fill, linalg::known_dim_value(&info.k_dim)) {
        (Some(fa), Some(fb), Some(k)) => Some(fa * fb * k as f64),
        _ => None,
    }
}

pub(crate) fn matmul_tensor_preview_with_info(
    a: &TensorVal,
    b: &TensorVal,
) -> Result<(TensorVal, MatMulShapeInfo), EvalError> {
    let info = matmul_shape_info(a, b, "tensor.matmul")?;
    let fill = matmul_fill(a, b, &info);
    let result = TensorVal::new(a.dtype.clone(), info.result_shape.clone(), fill);
    Ok((result, info))
}

pub(crate) fn matmul_tensor_preview(a: &TensorVal, b: &TensorVal) -> Result<TensorVal, EvalError> {
    matmul_tensor_preview_with_info(a, b).map(|(t, _)| t)
}

pub(crate) fn dot_tensor_preview_with_info(
    a: &TensorVal,
    b: &TensorVal,
) -> Result<(TensorVal, MatMulShapeInfo), EvalError> {
    let info = matmul_shape_info(a, b, "tensor.dot")?;
    let fill = matmul_fill(a, b, &info);
    let result = TensorVal::new(a.dtype.clone(), info.result_shape.clone(), fill);
    Ok((result, info))
}

pub(crate) fn dot_tensor_preview(a: &TensorVal, b: &TensorVal) -> Result<TensorVal, EvalError> {
    dot_tensor_preview_with_info(a, b).map(|(t, _)| t)
}

// ---------------------------------------------------------------------------
// Helpers for CpuExec dispatch (Option-returning wrappers)
// ---------------------------------------------------------------------------

/// Convert dim strings to usize, returning None if any dim is symbolic.
#[cfg(feature = "cpu-exec")]
pub(crate) fn dims_from_strings_usize(dims: &[String]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(dims.len());
    for d in dims {
        match d.parse::<usize>() {
            Ok(n) => out.push(n),
            Err(_) => return None,
        }
    }
    Some(out)
}

/// Normalize expand_dims axis, returning None on error.
#[cfg(feature = "cpu-exec")]
pub(crate) fn normalize_expand_axis_usize(axis: i32, rank: usize) -> Option<usize> {
    normalize_expand_axis(axis, rank).ok()
}

/// Normalize axis for indexing, returning None on error.
#[cfg(feature = "cpu-exec")]
pub(crate) fn normalize_axis_usize(axis: i32, rank: usize) -> Option<usize> {
    normalize_axis(axis, rank).ok()
}

/// Compute squeeze axes as usize, returning None on error.
#[cfg(feature = "cpu-exec")]
pub(crate) fn normalize_squeeze_axes_usize(shape: &[ShapeDim], axes: &[i32]) -> Option<Vec<usize>> {
    compute_squeeze_axes(shape, axes).ok()
}

/// Compute transpose permutation as usize vec, returning None on error.
#[cfg(feature = "cpu-exec")]
pub(crate) fn compute_transpose_perm(
    shape: &[ShapeDim],
    axes: Option<&[i32]>,
) -> Option<Vec<usize>> {
    let rank = shape.len();
    let perm = if let Some(spec) = axes {
        crate::linalg::normalize_permutation(spec, rank).ok()?
    } else {
        crate::linalg::default_transpose(rank)
    };
    Some(perm)
}
