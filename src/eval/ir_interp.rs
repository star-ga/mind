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

use std::collections::HashMap;

use crate::eval::value::TensorVal;
use crate::eval::value::Value;
use crate::ir::BinOp;
use crate::ir::IRModule;
use crate::ir::Instr;
use crate::ir::ValueId;
use crate::types::ShapeDim;

pub fn eval_ir(ir: &IRModule) -> Value {
    let mut vals: HashMap<ValueId, Value> = HashMap::new();
    let mut last = Value::Int(0);

    for instr in &ir.instrs {
        match instr {
            Instr::ConstI64(id, n) => {
                vals.insert(*id, Value::Int(*n));
            }
            Instr::ConstTensor(id, dtype, shape, fill) => {
                vals.insert(
                    *id,
                    Value::Tensor(TensorVal::new(dtype.clone(), shape.clone(), *fill)),
                );
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                let l = vals.get(lhs).cloned().unwrap_or(Value::Int(0));
                let r = vals.get(rhs).cloned().unwrap_or(Value::Int(0));
                let v = eval_binop(*op, l, r);
                vals.insert(*dst, v.clone());
                last = v;
            }
            Instr::Sum { dst, src, .. } => {
                let input = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let out = match input {
                    Value::Tensor(t) => Value::Tensor(TensorVal::new(t.dtype, vec![], t.fill)),
                    other => other,
                };
                vals.insert(*dst, out.clone());
                last = out;
            }
            Instr::Mean { dst, src, .. } => {
                let input = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let out = match input {
                    Value::Tensor(t) => Value::Tensor(TensorVal::new(t.dtype, vec![], t.fill)),
                    other => other,
                };
                vals.insert(*dst, out.clone());
                last = out;
            }
            Instr::Reshape {
                dst,
                src,
                new_shape,
            } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let reshaped = match value {
                    Value::Tensor(t) => {
                        Value::Tensor(TensorVal::new(t.dtype, new_shape.clone(), t.fill))
                    }
                    other => other,
                };
                vals.insert(*dst, reshaped.clone());
                last = reshaped;
            }
            Instr::ExpandDims { dst, src, axis } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let expanded = match value {
                    Value::Tensor(t) => {
                        let mut shape = t.shape.clone();
                        let axis = (*axis).clamp(0, shape.len() as i64) as usize;
                        shape.insert(axis, ShapeDim::Known(1));
                        Value::Tensor(TensorVal::new(t.dtype, shape, t.fill))
                    }
                    other => other,
                };
                vals.insert(*dst, expanded.clone());
                last = expanded;
            }
            Instr::Squeeze { dst, src, axes } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let squeezed = match value {
                    Value::Tensor(t) => {
                        let mut shape = Vec::new();
                        for (i, dim) in t.shape.iter().enumerate() {
                            if axes.iter().any(|axis| *axis as usize == i) {
                                continue;
                            }
                            shape.push(dim.clone());
                        }
                        Value::Tensor(TensorVal::new(t.dtype, shape, t.fill))
                    }
                    other => other,
                };
                vals.insert(*dst, squeezed.clone());
                last = squeezed;
            }
            Instr::Transpose { dst, src, .. } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                vals.insert(*dst, value.clone());
                last = value;
            }
            Instr::Dot { dst, a, b } => {
                let lhs = vals.get(a).cloned().unwrap_or(Value::Int(0));
                let rhs = vals.get(b).cloned().unwrap_or(Value::Int(0));
                let v = match (lhs, rhs) {
                    (Value::Tensor(at), Value::Tensor(bt)) => {
                        let fill = match (at.fill, bt.fill) {
                            (Some(x), Some(y)) => Some(x * y),
                            _ => None,
                        };
                        Value::Tensor(TensorVal::new(at.dtype, vec![], fill))
                    }
                    _ => Value::Int(0),
                };
                vals.insert(*dst, v.clone());
                last = v;
            }
            Instr::MatMul { dst, a, b } => {
                let lhs = vals.get(a).cloned().unwrap_or(Value::Int(0));
                let rhs = vals.get(b).cloned().unwrap_or(Value::Int(0));
                let v = match (lhs, rhs) {
                    (Value::Tensor(at), Value::Tensor(bt)) => {
                        let shape = broadcast_matmul_shape(&at.shape, &bt.shape);
                        let fill = match (at.fill, bt.fill) {
                            (Some(x), Some(y)) => Some(x * y),
                            _ => None,
                        };
                        Value::Tensor(TensorVal::new(at.dtype, shape, fill))
                    }
                    _ => Value::Int(0),
                };
                vals.insert(*dst, v.clone());
                last = v;
            }
            Instr::Conv2d { dst, input, .. } => {
                let value = vals.get(input).cloned().unwrap_or(Value::Int(0));
                vals.insert(*dst, value.clone());
                last = value;
            }
            Instr::Conv2dGradInput {
                dst, input_shape, ..
            } => {
                // Create a tensor with the input shape
                let shape: Vec<ShapeDim> = input_shape.iter().map(|d| ShapeDim::Known(*d)).collect();
                let v = Value::Tensor(TensorVal::new(crate::types::DType::F32, shape, None));
                vals.insert(*dst, v.clone());
                last = v;
            }
            Instr::Conv2dGradFilter {
                dst, filter_shape, ..
            } => {
                // Create a tensor with the filter shape
                let shape: Vec<ShapeDim> = filter_shape.iter().map(|d| ShapeDim::Known(*d)).collect();
                let v = Value::Tensor(TensorVal::new(crate::types::DType::F32, shape, None));
                vals.insert(*dst, v.clone());
                last = v;
            }
            Instr::Index { dst, src, .. } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                vals.insert(*dst, value.clone());
                last = value;
            }
            Instr::Slice { dst, src, .. } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                vals.insert(*dst, value.clone());
                last = value;
            }
            Instr::Gather { dst, src, .. } => {
                let value = vals.get(src).cloned().unwrap_or(Value::Int(0));
                vals.insert(*dst, value.clone());
                last = value;
            }
            Instr::Output(id) => {
                if let Some(v) = vals.get(id).cloned() {
                    last = v;
                }
            }
        }
    }

    last
}

fn eval_binop(op: BinOp, left: Value, right: Value) -> Value {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Value::Int(match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
        }),
        (Value::Tensor(t), Value::Int(s)) => tensor_scalar(op, t, s as f64, true),
        (Value::Int(s), Value::Tensor(t)) => tensor_scalar(op, t, s as f64, false),
        (Value::Tensor(a), Value::Tensor(b)) => tensor_tensor(op, a, b),
        (other, _) => other,
    }
}

fn tensor_scalar(op: BinOp, tensor: TensorVal, scalar: f64, tensor_left: bool) -> Value {
    let dtype = tensor.dtype;
    let shape = tensor.shape;
    let fill = tensor.fill.map(|f| match op {
        BinOp::Add => f + scalar,
        BinOp::Sub => {
            if tensor_left {
                f - scalar
            } else {
                scalar - f
            }
        }
        BinOp::Mul => f * scalar,
        BinOp::Div => {
            if tensor_left {
                f / scalar
            } else {
                scalar / f
            }
        }
    });
    Value::Tensor(TensorVal::new(dtype, shape, fill))
}

fn tensor_tensor(op: BinOp, a: TensorVal, b: TensorVal) -> Value {
    let dtype = a.dtype;
    let shape = a.shape;
    let fill = match (a.fill, b.fill) {
        (Some(x), Some(y)) => Some(match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
        }),
        _ => None,
    };
    Value::Tensor(TensorVal::new(dtype, shape, fill))
}

fn broadcast_matmul_shape(a: &[ShapeDim], b: &[ShapeDim]) -> Vec<ShapeDim> {
    let mut out = Vec::new();
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}
