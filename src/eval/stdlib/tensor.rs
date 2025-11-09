use std::collections::{BTreeSet, HashMap};

use crate::ast::{Literal, Node};
use crate::eval::autodiff::TensorEnvEntry;
use crate::eval::{eval_value_expr, format_value_human, EvalError, TensorVal, Value};
#[cfg(feature = "cpu-buffers")]
use crate::eval::{materialize_filled, num_elems, MATERIALIZE_MAX};
use crate::types::{DType, ShapeDim};

#[cfg(feature = "cpu-buffers")]
use crate::eval::value::Buffer;

pub fn dispatch(
    callee: &str,
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
) -> Result<Value, EvalError> {
    match callee {
        "tensor.zeros" => construct(args, env, tensor_env, 0.0),
        "tensor.ones" => construct(args, env, tensor_env, 1.0),
        "tensor.shape" => tensor_shape(args, env, tensor_env),
        "tensor.dtype" => tensor_dtype(args, env, tensor_env),
        "tensor.sum" => tensor_sum(args, env, tensor_env),
        "tensor.print" => tensor_print(args, env, tensor_env),
        #[cfg(feature = "cpu-buffers")]
        "tensor.materialize" => tensor_materialize(args, env, tensor_env),
        #[cfg(feature = "cpu-buffers")]
        "tensor.sample" => tensor_sample(args, env, tensor_env),
        #[cfg(feature = "cpu-buffers")]
        "tensor.is_materialized" => tensor_is_materialized(args, env, tensor_env),
        _ => Err(EvalError::Unsupported),
    }
}

fn construct(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    fill: f64,
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unsupported);
    }
    let dtype = parse_dtype(&args[0])?;
    let shape = parse_shape(&args[1], env, tensor_env)?;
    Ok(Value::Tensor(TensorVal::new(dtype, shape, Some(fill))))
}

fn tensor_shape(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
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
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
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
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
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
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
    println!("{}", format_value_human(&value));
    Ok(value)
}

#[cfg(feature = "cpu-buffers")]
fn tensor_materialize(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
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
) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
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
) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env, tensor_env)?;
    let count = eval_value_expr(&args[1], env, tensor_env)?;
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
        Node::Lit(Literal::Ident(name), _) => DType::from_str(name).ok_or(EvalError::Unsupported),
        _ => Err(EvalError::Unsupported),
    }
}

fn parse_shape(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
) -> Result<Vec<ShapeDim>, EvalError> {
    match node {
        Node::Tuple { elements, .. } => {
            let mut dims = Vec::with_capacity(elements.len());
            for el in elements {
                dims.push(parse_shape_dim(el, env, tensor_env)?);
            }
            Ok(dims)
        }
        Node::Paren(inner, _) => parse_shape(inner, env, tensor_env),
        Node::Lit(Literal::Ident(name), _) => {
            if let Some(value) = env.get(name) {
                shape_from_value(value)
            } else {
                Ok(vec![ShapeDim::Sym(leak_symbol(name))])
            }
        }
        Node::Lit(Literal::Int(_), _) => Ok(vec![parse_shape_dim(node, env, tensor_env)?]),
        _ => {
            let value = eval_value_expr(node, env, tensor_env)?;
            shape_from_value(&value)
        }
    }
}

fn parse_shape_dim(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
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
            let value = eval_value_expr(node, env, tensor_env)?;
            shape_dim_from_value(&value)
        }
    }
}

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

fn normalize_expand_axis(axis: i32, rank: usize) -> Result<usize, EvalError> {
    let extended = rank + 1;
    let idx = if axis < 0 { (extended as i32) + axis } else { axis };
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
