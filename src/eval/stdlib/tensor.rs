use std::collections::HashMap;

use crate::ast::{Literal, Node};
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
) -> Result<Value, EvalError> {
    match callee {
        "tensor.zeros" => construct(args, env, 0.0),
        "tensor.ones" => construct(args, env, 1.0),
        "tensor.shape" => tensor_shape(args, env),
        "tensor.dtype" => tensor_dtype(args, env),
        "tensor.sum" => tensor_sum(args, env),
        "tensor.print" => tensor_print(args, env),
        #[cfg(feature = "cpu-buffers")]
        "tensor.materialize" => tensor_materialize(args, env),
        #[cfg(feature = "cpu-buffers")]
        "tensor.sample" => tensor_sample(args, env),
        #[cfg(feature = "cpu-buffers")]
        "tensor.is_materialized" => tensor_is_materialized(args, env),
        _ => Err(EvalError::Unsupported),
    }
}

fn construct(args: &[Node], env: &HashMap<String, Value>, fill: f64) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unsupported);
    }
    let dtype = parse_dtype(&args[0])?;
    let shape = parse_shape(&args[1], env)?;
    Ok(Value::Tensor(TensorVal::new(dtype, shape, Some(fill))))
}

fn tensor_shape(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
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

fn tensor_dtype(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
    if let Value::Tensor(t) = value {
        Ok(Value::Str(t.dtype.as_str().to_string()))
    } else {
        Err(EvalError::Unsupported)
    }
}

fn tensor_sum(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
    if let Value::Tensor(t) = value {
        let mut result = TensorVal::new(t.dtype.clone(), Vec::new(), None);
        if let Some(fill) = t.fill {
            if let Some(count) = known_num_elems(&t.shape) {
                result.fill = Some(fill * count as f64);
            }
        }
        #[cfg(feature = "cpu-buffers")]
        {
            if result.fill.is_some() {
                crate::eval::materialize_filled(&mut result);
            }
        }
        Ok(Value::Tensor(result))
    } else {
        Err(EvalError::Unsupported)
    }
}

fn tensor_print(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
    println!("{}", format_value_human(&value));
    Ok(value)
}

#[cfg(feature = "cpu-buffers")]
fn tensor_materialize(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
    if let Value::Tensor(mut t) = value {
        materialize_filled(&mut t);
        Ok(Value::Tensor(t))
    } else {
        Err(EvalError::Unsupported)
    }
}

#[cfg(feature = "cpu-buffers")]
fn tensor_is_materialized(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 1 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
    if let Value::Tensor(t) = value {
        Ok(Value::Int(if t.buf.is_some() { 1 } else { 0 }))
    } else {
        Err(EvalError::Unsupported)
    }
}

#[cfg(feature = "cpu-buffers")]
fn tensor_sample(args: &[Node], env: &HashMap<String, Value>) -> Result<Value, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::Unsupported);
    }
    let value = eval_value_expr(&args[0], env)?;
    let count = eval_value_expr(&args[1], env)?;
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

fn parse_shape(node: &Node, env: &HashMap<String, Value>) -> Result<Vec<ShapeDim>, EvalError> {
    match node {
        Node::Tuple { elements, .. } => {
            let mut dims = Vec::with_capacity(elements.len());
            for el in elements {
                dims.push(parse_shape_dim(el, env)?);
            }
            Ok(dims)
        }
        Node::Paren(inner, _) => parse_shape(inner, env),
        Node::Lit(Literal::Ident(name), _) => {
            if let Some(value) = env.get(name) {
                shape_from_value(value)
            } else {
                Ok(vec![ShapeDim::Sym(leak_symbol(name))])
            }
        }
        Node::Lit(Literal::Int(_), _) => Ok(vec![parse_shape_dim(node, env)?]),
        _ => {
            let value = eval_value_expr(node, env)?;
            shape_from_value(&value)
        }
    }
}

fn parse_shape_dim(node: &Node, env: &HashMap<String, Value>) -> Result<ShapeDim, EvalError> {
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
            let value = eval_value_expr(node, env)?;
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
