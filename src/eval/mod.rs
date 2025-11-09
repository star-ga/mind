use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::ast::{BinOp, Literal, Module, Node, TypeAnn};
use crate::eval::autodiff::TensorEnvEntry;
use crate::types::{DType, ShapeDim, ValueType};

#[cfg(feature = "cpu-buffers")]
use value::Buffer;

pub mod autodiff;
pub mod value;

pub use value::{format_value_human, TensorVal, Value, VarId};

#[cfg(feature = "cpu-buffers")]
pub(crate) fn num_elems(shape: &[ShapeDim]) -> Option<usize> {
    let mut n: usize = 1;
    for d in shape {
        match d {
            ShapeDim::Known(k) => {
                n = n.saturating_mul(*k);
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(n)
}

#[cfg(feature = "cpu-buffers")]
pub(crate) const MATERIALIZE_MAX: usize = 1_024;

#[cfg(feature = "cpu-buffers")]
pub(crate) fn materialize_filled(t: &mut TensorVal) {
    if t.buf.is_some() {
        return;
    }
    let fill = match t.fill {
        Some(f) => f,
        None => return,
    };
    if let Some(ne) = num_elems(&t.shape) {
        if ne <= MATERIALIZE_MAX {
            match t.dtype {
                DType::I32 => {
                    let v = fill as i32;
                    t.buf = Some(Buffer::I32(vec![v; ne]));
                }
                DType::F32 => {
                    let v = fill as f32;
                    t.buf = Some(Buffer::F32(vec![v; ne]));
                }
                _ => {}
            }
        }
    }
}

mod stdlib;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("unsupported operation")]
    Unsupported,
    #[error("unsupported: {0}")]
    UnsupportedMsg(String),
    #[error("division by zero")]
    DivZero,
    #[error("unknown variable: {0}")]
    UnknownVar(String),
    #[error("type error: {0}")]
    TypeError(String),
    #[error("out of bounds")]
    OutOfBounds,
}

pub fn eval_module_value_with_env(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
) -> Result<Value, EvalError> {
    if let Some(src) = src_for_types {
        let mut tenv: HashMap<String, ValueType> = HashMap::new();
        for (name, _value) in env.iter() {
            tenv.insert(name.clone(), ValueType::ScalarI32);
        }
        let diags = crate::type_checker::check_module_types(m, src, &tenv);
        if !diags.is_empty() {
            let msg = diags.into_iter().map(|diag| diag.message).collect::<Vec<_>>().join("; ");
            return Err(EvalError::TypeError(msg));
        }
    }

    let mut venv: HashMap<String, Value> =
        env.iter().map(|(name, value)| (name.clone(), Value::Int(*value))).collect();
    let mut tensor_env: HashMap<String, TensorEnvEntry> = HashMap::new();

    let mut last = Value::Int(0_i64);
    for item in &m.items {
        match item {
            Node::Let { name, ann, value, .. } => {
                let rhs = eval_value_expr(value, &venv, &tensor_env)?;
                let stored = match ann {
                    Some(TypeAnn::Tensor { dtype, dims }) => {
                        let (dtype, shape) = parse_tensor_ann(dtype, dims)?;
                        let fill = match rhs {
                            Value::Int(n) => Some(n as f64),
                            Value::Tensor(ref t) => t.fill,
                            _ => None,
                        };
                        Value::Tensor(TensorVal::new(dtype, shape, fill))
                    }
                    Some(TypeAnn::ScalarI32) | None => rhs,
                };
                if let Value::Int(n) = stored {
                    env.insert(name.clone(), n);
                    venv.insert(name.clone(), Value::Int(n));
                    last = Value::Int(n);
                } else {
                    venv.insert(name.clone(), stored.clone());
                    last = stored;
                }
                match venv.get(name) {
                    Some(Value::Tensor(tensor)) => {
                        let expr = match ann {
                            Some(TypeAnn::Tensor { .. }) => None,
                            _ => Some((**value).clone()),
                        };
                        tensor_env
                            .insert(name.clone(), TensorEnvEntry { value: tensor.clone(), expr });
                    }
                    _ => {
                        tensor_env.remove(name);
                    }
                }
            }
            Node::Assign { name, value, .. } => {
                let rhs = eval_value_expr(value, &venv, &tensor_env)?;
                if let Value::Int(n) = rhs {
                    env.insert(name.clone(), n);
                    venv.insert(name.clone(), Value::Int(n));
                    last = Value::Int(n);
                } else {
                    venv.insert(name.clone(), rhs.clone());
                    last = rhs;
                }
                match venv.get(name) {
                    Some(Value::Tensor(tensor)) => {
                        tensor_env.insert(
                            name.clone(),
                            TensorEnvEntry { value: tensor.clone(), expr: Some((**value).clone()) },
                        );
                    }
                    _ => {
                        tensor_env.remove(name);
                    }
                }
            }
            _ => {
                last = eval_value_expr(item, &venv, &tensor_env)?;
            }
        }
    }
    Ok(last)
}

pub fn eval_module_with_env(
    m: &Module,
    env: &mut HashMap<String, i64>,
    src_for_types: Option<&str>,
) -> Result<i64, EvalError> {
    match eval_module_value_with_env(m, env, src_for_types)? {
        Value::Int(n) => Ok(n),
        _ => Err(EvalError::Unsupported),
    }
}

pub fn eval_module(m: &Module) -> Result<i64, EvalError> {
    let mut env: HashMap<String, i64> = HashMap::new();
    eval_module_with_env(m, &mut env, None)
}

pub fn eval_first_expr(m: &Module) -> Result<i64, EvalError> {
    eval_module(m)
}

pub(crate) fn eval_value_expr(
    node: &Node,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
) -> Result<Value, EvalError> {
    match node {
        Node::Lit(Literal::Int(n), _) => Ok(Value::Int(*n)),
        Node::Lit(Literal::Ident(name), _) => {
            env.get(name).cloned().ok_or_else(|| EvalError::UnknownVar(name.clone()))
        }
        Node::Paren(inner, _) => eval_value_expr(inner, env, tensor_env),
        Node::Tuple { elements, .. } => {
            let mut items = Vec::with_capacity(elements.len());
            for item in elements {
                items.push(eval_value_expr(item, env, tensor_env)?);
            }
            Ok(Value::Tuple(items))
        }
        Node::Call { callee, args, .. } => stdlib::tensor::dispatch(callee, args, env, tensor_env),
        Node::CallTensorSum { x, axes, keepdims, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::sum_tensor_preview(&t, axes, *keepdims)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallTensorMean { x, axes, keepdims, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::mean_tensor_preview(&t, axes, *keepdims)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallReshape { x, dims, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::reshape_tensor_preview(&t, dims)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallExpandDims { x, axis, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::expand_dims_tensor_preview(&t, *axis)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallSqueeze { x, axes, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::squeeze_tensor_preview(&t, axes)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallTranspose { x, axes, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let axes_ref = axes.as_ref().map(|v| v.as_slice());
                    let (result, _) = stdlib::tensor::transpose_tensor_preview(&t, axes_ref)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallIndex { x, axis, i, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::index_tensor_preview(&t, *axis, *i)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallSlice { x, axis, start, end, .. } => {
            let value = eval_value_expr(x, env, tensor_env)?;
            match value {
                Value::Tensor(t) => {
                    let result = stdlib::tensor::slice_tensor_preview(&t, *axis, *start, *end)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallDot { a, b, .. } => {
            let left = eval_value_expr(a, env, tensor_env)?;
            let right = eval_value_expr(b, env, tensor_env)?;
            match (left, right) {
                (Value::Tensor(tl), Value::Tensor(tr)) => {
                    let result = stdlib::tensor::dot_tensor_preview(&tl, &tr)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallMatMul { a, b, .. } => {
            let left = eval_value_expr(a, env, tensor_env)?;
            let right = eval_value_expr(b, env, tensor_env)?;
            match (left, right) {
                (Value::Tensor(tl), Value::Tensor(tr)) => {
                    let result = stdlib::tensor::matmul_tensor_preview(&tl, &tr)?;
                    Ok(Value::Tensor(result))
                }
                _ => Err(EvalError::Unsupported),
            }
        }
        Node::CallGrad { loss, wrt, .. } => eval_grad_map(loss, env, tensor_env, wrt),
        Node::Binary { op, left, right, .. } => {
            let lv = eval_value_expr(left, env, tensor_env)?;
            let rv = eval_value_expr(right, env, tensor_env)?;
            apply_binary(*op, lv, rv)
        }
        Node::Let { value, .. } | Node::Assign { value, .. } => {
            eval_value_expr(value, env, tensor_env)
        }
    }
}

pub fn eval_grad_map(
    loss_expr: &Node,
    _env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    wrt: &[String],
) -> Result<Value, EvalError> {
    let mut tenv: HashMap<String, TensorEnvEntry> = HashMap::new();
    for (name, entry) in tensor_env {
        tenv.insert(name.clone(), entry.clone());
    }

    let mut expanding = BTreeSet::new();
    let (loss_id, tape, vars_all) =
        crate::eval::autodiff::build_graph_loss(loss_expr, &tenv, &mut expanding)
            .map_err(EvalError::UnsupportedMsg)?;

    if !tape.node_shape(loss_id).is_empty() {
        return Err(EvalError::UnsupportedMsg(
            "grad() expects the loss expression to have shape ()".to_string(),
        ));
    }

    let requested: BTreeMap<String, crate::eval::autodiff::NodeId> =
        vars_all.into_iter().filter(|(name, _)| wrt.contains(name)).collect();

    let mut grads = crate::eval::autodiff::backprop_to_vars(loss_id, &tape, &requested);
    for name in wrt {
        if !grads.contains_key(name) {
            if let Some(entry) = tenv.get(name) {
                grads.insert(
                    name.clone(),
                    TensorVal::new(entry.value.dtype.clone(), entry.value.shape.clone(), Some(0.0)),
                );
            }
        }
    }

    let mut out = BTreeMap::new();
    for name in wrt {
        if let Some(tensor) = grads.get(name) {
            out.insert(VarId(name.clone()), tensor.clone());
        }
    }

    Ok(Value::GradMap(out))
}

fn apply_binary(op: BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => apply_int_op(op, l, r).map(Value::Int),
        (Value::Tensor(t), Value::Int(s)) => apply_tensor_scalar(op, t, s as f64, true),
        (Value::Int(s), Value::Tensor(t)) => apply_tensor_scalar(op, t, s as f64, false),
        (Value::Tensor(a), Value::Tensor(b)) => apply_tensor_tensor(op, a, b),
        _ => Err(EvalError::Unsupported),
    }
}

fn apply_int_op(op: BinOp, left: i64, right: i64) -> Result<i64, EvalError> {
    Ok(match op {
        BinOp::Add => left + right,
        BinOp::Sub => left - right,
        BinOp::Mul => left * right,
        BinOp::Div => {
            if right == 0 {
                return Err(EvalError::DivZero);
            }
            left / right
        }
    })
}

fn apply_tensor_scalar(
    op: BinOp,
    tensor: TensorVal,
    scalar: f64,
    tensor_on_left: bool,
) -> Result<Value, EvalError> {
    if matches!(op, BinOp::Div) && tensor_on_left && scalar == 0.0 {
        return Err(EvalError::DivZero);
    }

    #[cfg(feature = "cpu-buffers")]
    let tensor_buf = tensor.buf.clone();

    let dtype = tensor.dtype;
    let shape = tensor.shape;
    let fill = tensor.fill;

    let result_fill = match fill {
        Some(f) => {
            if matches!(op, BinOp::Div) && !tensor_on_left && f == 0.0 {
                return Err(EvalError::DivZero);
            }
            Some(match op {
                BinOp::Add => f + scalar,
                BinOp::Sub => {
                    if tensor_on_left {
                        f - scalar
                    } else {
                        scalar - f
                    }
                }
                BinOp::Mul => f * scalar,
                BinOp::Div => {
                    if tensor_on_left {
                        f / scalar
                    } else {
                        scalar / f
                    }
                }
            })
        }
        None => None,
    };

    #[cfg_attr(not(feature = "cpu-buffers"), allow(unused_mut))]
    let mut result = TensorVal::new(dtype.clone(), shape, result_fill);

    #[cfg(feature = "cpu-buffers")]
    {
        if let Some(buf) = tensor_buf.as_ref() {
            match (buf, &dtype) {
                (Buffer::I32(values), DType::I32) => {
                    if matches!(op, BinOp::Div) && !tensor_on_left {
                        if values.iter().any(|&v| v == 0) {
                            return Err(EvalError::DivZero);
                        }
                    }
                    let scalar_i32 = scalar as i32;
                    let mut out = Vec::with_capacity(values.len());
                    for &v in values {
                        let computed = match op {
                            BinOp::Add => v + scalar_i32,
                            BinOp::Sub => {
                                if tensor_on_left {
                                    v - scalar_i32
                                } else {
                                    scalar_i32 - v
                                }
                            }
                            BinOp::Mul => v * scalar_i32,
                            BinOp::Div => {
                                if tensor_on_left {
                                    v / scalar_i32
                                } else {
                                    scalar_i32 / v
                                }
                            }
                        };
                        out.push(computed);
                    }
                    result.buf = Some(Buffer::I32(out));
                }
                (Buffer::F32(values), DType::F32) => {
                    if matches!(op, BinOp::Div) && !tensor_on_left {
                        if values.iter().any(|&v| v == 0.0) {
                            return Err(EvalError::DivZero);
                        }
                    }
                    let scalar_f32 = scalar as f32;
                    let mut out = Vec::with_capacity(values.len());
                    for &v in values {
                        let computed = match op {
                            BinOp::Add => v + scalar_f32,
                            BinOp::Sub => {
                                if tensor_on_left {
                                    v - scalar_f32
                                } else {
                                    scalar_f32 - v
                                }
                            }
                            BinOp::Mul => v * scalar_f32,
                            BinOp::Div => {
                                if tensor_on_left {
                                    v / scalar_f32
                                } else {
                                    scalar_f32 / v
                                }
                            }
                        };
                        out.push(computed);
                    }
                    result.buf = Some(Buffer::F32(out));
                }
                _ => {}
            }
        } else if result.fill.is_some() {
            materialize_filled(&mut result);
        }
    }

    Ok(Value::Tensor(result))
}

fn apply_tensor_tensor(op: BinOp, left: TensorVal, right: TensorVal) -> Result<Value, EvalError> {
    if left.dtype != right.dtype {
        return Err(EvalError::Unsupported);
    }

    let shape = broadcast_shapes(&left.shape, &right.shape).ok_or(EvalError::Unsupported)?;

    #[cfg(feature = "cpu-buffers")]
    let left_buf = left.buf.clone();
    #[cfg(feature = "cpu-buffers")]
    let right_buf = right.buf.clone();

    let left_fill = left.fill;
    let right_fill = right.fill;
    let dtype = left.dtype.clone();

    if matches!(op, BinOp::Div) {
        if let Some(fill) = right_fill {
            if fill == 0.0 {
                return Err(EvalError::DivZero);
            }
        }
        #[cfg(feature = "cpu-buffers")]
        if let Some(buf) = right_buf.as_ref() {
            match buf {
                Buffer::I32(values) => {
                    if values.iter().any(|&v| v == 0) {
                        return Err(EvalError::DivZero);
                    }
                }
                Buffer::F32(values) => {
                    if values.iter().any(|&v| v == 0.0) {
                        return Err(EvalError::DivZero);
                    }
                }
            }
        }
    }

    let fill = match (left_fill, right_fill) {
        (Some(a), Some(b)) => Some(match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
        }),
        _ => None,
    };

    #[cfg_attr(not(feature = "cpu-buffers"), allow(unused_mut))]
    let mut result = TensorVal::new(dtype, shape, fill);

    #[cfg(feature = "cpu-buffers")]
    {
        if let (Some(lb), Some(rb)) = (left_buf.as_ref(), right_buf.as_ref()) {
            if let Some(ne) = num_elems(&result.shape) {
                match (lb, rb) {
                    (Buffer::I32(lv), Buffer::I32(rv)) if lv.len() == ne && rv.len() == ne => {
                        let mut out = Vec::with_capacity(ne);
                        for i in 0..ne {
                            let computed = match op {
                                BinOp::Add => lv[i] + rv[i],
                                BinOp::Sub => lv[i] - rv[i],
                                BinOp::Mul => lv[i] * rv[i],
                                BinOp::Div => lv[i] / rv[i],
                            };
                            out.push(computed);
                        }
                        result.buf = Some(Buffer::I32(out));
                    }
                    (Buffer::F32(lv), Buffer::F32(rv)) if lv.len() == ne && rv.len() == ne => {
                        let mut out = Vec::with_capacity(ne);
                        for i in 0..ne {
                            let computed = match op {
                                BinOp::Add => lv[i] + rv[i],
                                BinOp::Sub => lv[i] - rv[i],
                                BinOp::Mul => lv[i] * rv[i],
                                BinOp::Div => lv[i] / rv[i],
                            };
                            out.push(computed);
                        }
                        result.buf = Some(Buffer::F32(out));
                    }
                    _ => {}
                }
            }
        }

        if result.buf.is_none()
            && result.fill.is_some()
            && left_fill.is_some()
            && right_fill.is_some()
        {
            materialize_filled(&mut result);
        }
    }

    Ok(Value::Tensor(result))
}

pub(crate) fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
    let mut result = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;
    while i >= 0 || j >= 0 {
        let da = if i >= 0 { &a[i as usize] } else { &ShapeDim::Known(1) };
        let db = if j >= 0 { &b[j as usize] } else { &ShapeDim::Known(1) };
        let dim = match (da, db) {
            (ShapeDim::Known(x), ShapeDim::Known(y)) if x == y => ShapeDim::Known(*x),
            (ShapeDim::Known(1), ShapeDim::Known(y)) => ShapeDim::Known(*y),
            (ShapeDim::Known(x), ShapeDim::Known(1)) => ShapeDim::Known(*x),
            (ShapeDim::Sym(s1), ShapeDim::Sym(s2)) if s1 == s2 => ShapeDim::Sym(s1),
            (ShapeDim::Sym(sym), ShapeDim::Known(1)) | (ShapeDim::Known(1), ShapeDim::Sym(sym)) => {
                ShapeDim::Sym(sym)
            }
            _ => return None,
        };
        result.push(dim);
        i -= 1;
        j -= 1;
    }
    result.reverse();
    Some(result)
}

fn parse_tensor_ann(dtype: &str, dims: &[String]) -> Result<(DType, Vec<ShapeDim>), EvalError> {
    let dtype = DType::from_str(dtype).ok_or(EvalError::Unsupported)?;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        if let Ok(n) = dim.parse::<usize>() {
            shape.push(ShapeDim::Known(n));
        } else {
            let leaked: &'static str = Box::leak(dim.clone().into_boxed_str());
            shape.push(ShapeDim::Sym(leaked));
        }
    }
    Ok((dtype, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    #[test]
    fn eval_tensor_add_scalar_preview() {
        let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![ShapeDim::Known(2), ShapeDim::Known(3)]);
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.fill, Some(1.0));
            }
            _ => panic!("expected tensor"),
        }
    }
}
