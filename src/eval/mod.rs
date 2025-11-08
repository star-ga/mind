use std::collections::HashMap;

use crate::ast::{BinOp, Literal, Module, Node, TypeAnn};
use crate::types::{DType, ShapeDim, ValueType};

pub mod value;

pub use value::{format_value_human, TensorVal, Value};

mod stdlib;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("unsupported operation")]
    Unsupported,
    #[error("division by zero")]
    DivZero,
    #[error("unknown variable: {0}")]
    UnknownVar(String),
    #[error("type error: {0}")]
    TypeError(String),
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

    let mut last = Value::Int(0_i64);
    for item in &m.items {
        match item {
            Node::Let { name, ann, value, .. } => {
                let rhs = eval_value_expr(value, &venv)?;
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
            }
            Node::Assign { name, value, .. } => {
                let rhs = eval_value_expr(value, &venv)?;
                if let Value::Int(n) = rhs {
                    env.insert(name.clone(), n);
                    venv.insert(name.clone(), Value::Int(n));
                    last = Value::Int(n);
                } else {
                    venv.insert(name.clone(), rhs.clone());
                    last = rhs;
                }
            }
            _ => {
                last = eval_value_expr(item, &venv)?;
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
) -> Result<Value, EvalError> {
    match node {
        Node::Lit(Literal::Int(n), _) => Ok(Value::Int(*n)),
        Node::Lit(Literal::Ident(name), _) => {
            env.get(name).cloned().ok_or_else(|| EvalError::UnknownVar(name.clone()))
        }
        Node::Paren(inner, _) => eval_value_expr(inner, env),
        Node::Tuple { elements, .. } => {
            let mut items = Vec::with_capacity(elements.len());
            for item in elements {
                items.push(eval_value_expr(item, env)?);
            }
            Ok(Value::Tuple(items))
        }
        Node::Call { callee, args, .. } => stdlib::tensor::dispatch(callee, args, env),
        Node::Binary { op, left, right, .. } => {
            let lv = eval_value_expr(left, env)?;
            let rv = eval_value_expr(right, env)?;
            apply_binary(*op, lv, rv)
        }
        Node::Let { value, .. } | Node::Assign { value, .. } => eval_value_expr(value, env),
    }
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

    let TensorVal { dtype, shape, fill } = tensor;

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

    Ok(Value::Tensor(TensorVal::new(dtype, shape, result_fill)))
}

fn apply_tensor_tensor(op: BinOp, left: TensorVal, right: TensorVal) -> Result<Value, EvalError> {
    if left.dtype != right.dtype {
        return Err(EvalError::Unsupported);
    }

    let shape = broadcast_shapes(&left.shape, &right.shape).ok_or(EvalError::Unsupported)?;

    if matches!(op, BinOp::Div) {
        if let Some(fill) = right.fill {
            if fill == 0.0 {
                return Err(EvalError::DivZero);
            }
        }
    }

    let fill = match (left.fill, right.fill) {
        (Some(a), Some(b)) => Some(match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
        }),
        _ => None,
    };

    Ok(Value::Tensor(TensorVal::new(left.dtype, shape, fill)))
}

fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
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
