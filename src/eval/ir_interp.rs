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
            Instr::Relu { dst, src } => {
                let input = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let out = match input {
                    // ReLU preserves shape; the constant-fill preview clamps at 0.
                    Value::Tensor(t) => {
                        Value::Tensor(TensorVal::new(t.dtype, t.shape, t.fill.map(|f| f.max(0.0))))
                    }
                    other => other,
                };
                vals.insert(*dst, out.clone());
                last = out;
            }
            Instr::ReluGrad { dst, grad, src } => {
                // Backward ReLU preview: dx = grad * step(src). Shape-preserving
                // (dx matches grad/src). On the constant-fill model the gate is
                // src.fill > 0; otherwise the gradient is zeroed.
                let grad_val = vals.get(grad).cloned().unwrap_or(Value::Int(0));
                let src_val = vals.get(src).cloned().unwrap_or(Value::Int(0));
                let out = match (grad_val, src_val) {
                    (Value::Tensor(g), Value::Tensor(s)) => {
                        let gated = match (g.fill, s.fill) {
                            (Some(gf), Some(sf)) => Some(if sf > 0.0 { gf } else { 0.0 }),
                            _ => None,
                        };
                        Value::Tensor(TensorVal::new(g.dtype, g.shape, gated))
                    }
                    (other, _) => other,
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
                let shape: Vec<ShapeDim> =
                    input_shape.iter().map(|d| ShapeDim::Known(*d)).collect();
                let v = Value::Tensor(TensorVal::new(crate::types::DType::F32, shape, None));
                vals.insert(*dst, v.clone());
                last = v;
            }
            Instr::Conv2dGradFilter {
                dst, filter_shape, ..
            } => {
                // Create a tensor with the filter shape
                let shape: Vec<ShapeDim> =
                    filter_shape.iter().map(|d| ShapeDim::Known(*d)).collect();
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
            // The variants below are NOT modelled by this preview evaluator.
            // They are listed explicitly rather than swallowed by a `_ => {}`
            // wildcard so that adding an `Instr` variant fails to COMPILE here
            // instead of being silently ignored at runtime — a wildcard is what
            // let a control-flow program report the last *handled* value
            // (`Int(0)`) as if it were the program's result. The conformance
            // suite no longer uses this function as its runtime-value oracle
            // (see `conformance::run_value_oracle`); it remains the cheap
            // constant-fill preview printed by `mindc <file>`.
            Instr::ConstF64(..)
            | Instr::ConstDenseTensor { .. }
            | Instr::SparseAttr { .. }
            | Instr::FnDef { .. }
            | Instr::Call { .. }
            | Instr::Return { .. }
            | Instr::Param { .. } => {}
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { .. }
            | Instr::ArrayLoad { .. }
            | Instr::ArrayStore { .. }
            | Instr::While { .. }
            | Instr::Break { .. }
            | Instr::Continue { .. }
            | Instr::If { .. }
            | Instr::VecLoad { .. }
            | Instr::VecLoadI32 { .. }
            | Instr::VecStore { .. }
            | Instr::VecFma { .. }
            | Instr::VecMulAddQ16 { .. }
            | Instr::VecReduceAdd { .. }
            | Instr::VecReduceAddI64 { .. }
            | Instr::Region { .. }
            | Instr::ExternFnDecl { .. } => {}
        }
    }

    last
}

fn eval_binop(op: BinOp, left: Value, right: Value) -> Value {
    match (left, right) {
        // SIGNED i64 integer arms match `eval::apply_int_op` — this is the
        // conformance oracle (`src/conformance.rs`, and the `--- Result ---` path
        // of `mindc <file>`), so any arm that disagrees with the tree-walking
        // evaluator or with the emitted artifact turns the oracle into a second
        // opinion instead of a witness. Before these arms wrapped, a literal
        // `i64::MIN / -1` or a constant-folded `x / 0` PANICKED the compiler here.
        // deferred: this oracle is type-blind — it has no u64 notion, so for a
        // `ScalarU64` operand it answers the SIGNED `/ % < >>` where the artifact
        // emits `divui`/`remui`/`ult`/`shrui` (e.g. u64::MAX / 2 -> 0 here,
        // 2^63-1 in the artifact), and it masks shifts to 63 where the i32 narrow
        // arm masks to 31. Upgrade path: carry the per-ValueId scalar dtype into
        // `eval_binop` (task #313's dtype threading) and dispatch like
        // `apply_int_op_u64`.
        //   * `+ − ×`: MIND integer overflow is defined two's-complement
        //     wraparound (== `arith.addi`, no nsw/nuw). Bare `a + b` PANICS in a
        //     debug build and wraps in release — the oracle must not depend on
        //     the profile it was compiled with.
        //   * `/ %`: total and deterministic — `x/0 == 0`, `x%0 == 0`, and
        //     `i64::MIN / -1 == i64::MIN` (`% -1 == 0`), matching the native
        //     `nb_div_guarded` zero-guard and the MLIR `div_zero_guard`. Bare
        //     `a / b` here trapped on a zero divisor and panicked on the
        //     INT_MIN/-1 pair, both of which the artifact answers with a value.
        (Value::Int(a), Value::Int(b)) => Value::Int(match op {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => {
                if b == 0 {
                    0
                } else {
                    a.wrapping_div(b)
                }
            }
            BinOp::Mod => {
                if b == 0 {
                    0
                } else {
                    a.wrapping_rem(b)
                }
            }
            BinOp::Lt => (a < b) as i64,
            BinOp::Le => (a <= b) as i64,
            BinOp::Gt => (a > b) as i64,
            BinOp::Ge => (a >= b) as i64,
            BinOp::Eq => (a == b) as i64,
            BinOp::Ne => (a != b) as i64,
            // Phase 6.5 Stage 1a — bitwise ops interpreted on i64.
            #[cfg(feature = "std-surface")]
            BinOp::BitAnd => a & b,
            #[cfg(feature = "std-surface")]
            BinOp::BitOr => a | b,
            #[cfg(feature = "std-surface")]
            BinOp::BitXor => a ^ b,
            #[cfg(feature = "std-surface")]
            BinOp::Shl => a.wrapping_shl(b as u32),
            // `wrapping_shr` masks the shift amount to 0..63 and shifts
            // ARITHMETICALLY, matching the sibling `wrapping_shl` above and the
            // signed i64 artifact (`andi rhs, 63` + `shrsi`; x86 `sar rax,cl`).
            // A u64 operand would need a LOGICAL shift — see the deferred note at
            // the top of this match. Bare `a >> b` panicked in a debug build for
            // `b >= 64`.
            #[cfg(feature = "std-surface")]
            BinOp::Shr => a.wrapping_shr(b as u32),
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
        BinOp::Mod => {
            if tensor_left {
                f % scalar
            } else {
                scalar % f
            }
        }
        BinOp::Lt => {
            if (tensor_left && f < scalar) || (!tensor_left && scalar < f) {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Le => {
            if (tensor_left && f <= scalar) || (!tensor_left && scalar <= f) {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Gt => {
            if (tensor_left && f > scalar) || (!tensor_left && scalar > f) {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Ge => {
            if (tensor_left && f >= scalar) || (!tensor_left && scalar >= f) {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Eq => {
            if f == scalar {
                1.0
            } else {
                0.0
            }
        }
        BinOp::Ne => {
            if f != scalar {
                1.0
            } else {
                0.0
            }
        }
        // Bitwise ops on tensors: fall through to 0.0 (not meaningful for
        // floating-point data; gated so default build is byte-identical).
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => 0.0,
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
            BinOp::Mod => x % y,
            BinOp::Lt => {
                if x < y {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Le => {
                if x <= y {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Gt => {
                if x > y {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Ge => {
                if x >= y {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Eq => {
                if x == y {
                    1.0
                } else {
                    0.0
                }
            }
            BinOp::Ne => {
                if x != y {
                    1.0
                } else {
                    0.0
                }
            }
            // Bitwise ops not meaningful on floating-point tensors.
            #[cfg(feature = "std-surface")]
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => 0.0,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate `a <op> b` through the IR oracle exactly as `conformance.rs`
    /// does — two constants and one `BinOp`, read back from `eval_ir`'s result.
    fn ir_int_binop(op: BinOp, a: i64, b: i64) -> i64 {
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::ConstI64(ValueId(0), a),
            Instr::ConstI64(ValueId(1), b),
            Instr::BinOp {
                dst: ValueId(2),
                op,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ];
        match eval_ir(&m) {
            Value::Int(n) => n,
            other => panic!("expected Int from the IR oracle, got {other:?}"),
        }
    }

    /// `eval_ir` is the CONFORMANCE ORACLE (`src/conformance.rs`). An oracle
    /// that disagrees with the artifact it is asked to judge is worse than no
    /// oracle, so these are the same edge values the native backend is pinned
    /// against in `examples/mindc_mind/div_shift_cmp_edge_smoke.py` and that
    /// `eval::apply_int_op` now answers identically.
    ///
    /// Every case below either trapped (`a / b` on a zero divisor) or panicked
    /// (`INT_MIN / -1`, and `+ − ×` overflow in a debug build) before the
    /// wrapping arms landed — so this test is mutation-sensitive against a
    /// revert in either direction.
    #[test]
    fn ir_oracle_integer_arms_match_apply_int_op_and_the_artifact() {
        // Total division: a value, never a trap.
        assert_eq!(ir_int_binop(BinOp::Div, 7, 0), 0, "7/0 == 0");
        assert_eq!(ir_int_binop(BinOp::Mod, 7, 0), 0, "7%0 == 0");
        assert_eq!(
            ir_int_binop(BinOp::Div, i64::MIN, -1),
            i64::MIN,
            "INT_MIN/-1 == INT_MIN (defined wrap, no panic)"
        );
        assert_eq!(ir_int_binop(BinOp::Mod, i64::MIN, -1), 0, "INT_MIN%-1 == 0");

        // Signed truncation / remainder sign — unchanged behaviour, pinned so a
        // future "simplification" to unsigned ops is caught here too.
        assert_eq!(ir_int_binop(BinOp::Div, -17, 5), -3, "-17/5 == -3");
        assert_eq!(ir_int_binop(BinOp::Mod, -17, 5), -2, "-17%5 == -2");

        // Defined two's-complement wraparound on `+ − ×` (== `arith.addi`, no
        // nsw/nuw). A debug build panicked on each of these before.
        assert_eq!(
            ir_int_binop(BinOp::Add, i64::MAX, 1),
            i64::MIN,
            "MAX+1 wraps to MIN"
        );
        assert_eq!(
            ir_int_binop(BinOp::Sub, i64::MIN, 1),
            i64::MAX,
            "MIN-1 wraps to MAX"
        );
        assert_eq!(
            ir_int_binop(BinOp::Mul, i64::MAX, 2),
            -2,
            "MAX*2 wraps to -2"
        );
    }

    /// `>>` is ARITHMETIC (sign-extending) and its shift amount is masked to
    /// 0..63 — the `sar rax,cl` the artifact runs. A bare `a >> b` panicked for
    /// `b >= 64` in a debug build.
    #[cfg(feature = "std-surface")]
    #[test]
    fn ir_oracle_shift_right_is_arithmetic_and_masks_the_amount() {
        assert_eq!(ir_int_binop(BinOp::Shr, -256, 2), -64, "-256>>2 == -64 (sar)");
        assert_eq!(ir_int_binop(BinOp::Shr, 100, 3), 12, "100>>3 == 12");
        assert_eq!(
            ir_int_binop(BinOp::Shr, 1, 64),
            1,
            "shift amount masks to 0 (CL & 63), matching the sibling `Shl` arm"
        );
    }
}
