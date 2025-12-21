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

use crate::ir::{BinOp, Instr, ValueId};

use super::engine::GradientOps;
use super::engine::{as_invalid, AutodiffError};

pub(super) fn apply_rule(
    ops: &mut impl GradientOps,
    instr: &Instr,
    upstream: ValueId,
) -> Result<(), AutodiffError> {
    match instr {
        Instr::ConstI64(..) | Instr::ConstTensor(..) => Ok(()),
        Instr::BinOp { op, lhs, rhs, .. } => {
            match op {
                BinOp::Add => {
                    ops.add_grad(*lhs, upstream);
                    ops.add_grad(*rhs, upstream);
                }
                BinOp::Sub => {
                    ops.add_grad(*lhs, upstream);
                    let neg_one = ops.add_const_i64(-1);
                    let rhs_contrib = ops.add_binop(BinOp::Mul, upstream, neg_one);
                    ops.add_grad(*rhs, rhs_contrib);
                }
                BinOp::Mul => {
                    let dlhs = ops.add_binop(BinOp::Mul, upstream, *rhs);
                    let drhs = ops.add_binop(BinOp::Mul, upstream, *lhs);
                    ops.add_grad(*lhs, dlhs);
                    ops.add_grad(*rhs, drhs);
                }
                BinOp::Div => {
                    return Err(AutodiffError::UnsupportedOp { op: "div" });
                }
            }
            Ok(())
        }
        Instr::MatMul { a, b, .. } => {
            let bt = ops.add_transpose(*b, vec![1, 0]);
            let at = ops.add_transpose(*a, vec![1, 0]);
            let da = ops.add_matmul(upstream, bt);
            let db = ops.add_matmul(at, upstream);
            ops.add_grad(*a, da);
            ops.add_grad(*b, db);
            Ok(())
        }
        Instr::Conv2d { .. } => Err(AutodiffError::UnsupportedOp { op: "conv2d" }),
        Instr::Conv2dGradInput { .. } => Err(AutodiffError::UnsupportedOp {
            op: "conv2d_grad_input",
        }),
        Instr::Conv2dGradFilter { .. } => Err(AutodiffError::UnsupportedOp {
            op: "conv2d_grad_filter",
        }),
        Instr::Dot { a, b, .. } => {
            let da = ops.add_binop(BinOp::Mul, upstream, *b);
            let db = ops.add_binop(BinOp::Mul, upstream, *a);
            ops.add_grad(*a, da);
            ops.add_grad(*b, db);
            Ok(())
        }
        Instr::Transpose { src, perm, .. } => {
            if perm.is_empty() {
                return Err(as_invalid("transpose requires permutation for autodiff"));
            }
            let mut inverse = vec![0; perm.len()];
            for (idx, &p) in perm.iter().enumerate() {
                let p = p as usize;
                if p >= perm.len() {
                    return Err(as_invalid("invalid transpose permutation"));
                }
                inverse[p] = idx as i64;
            }
            let back = ops.add_transpose(upstream, inverse);
            ops.add_grad(*src, back);
            Ok(())
        }
        Instr::Mean {
            src,
            axes,
            keepdims,
            ..
        } => {
            if axes.is_empty() {
                return Err(as_invalid("mean requires explicit axes for autodiff"));
            }
            let count = ops.add_const_i64(i64::try_from(axes.len()).unwrap_or(0).max(1));
            let scaled = ops.add_binop(BinOp::Div, upstream, count);
            ops.add_grad(*src, scaled);
            if *keepdims {
                // Do not attempt to broadcast; assume shapes already match.
            }
            Ok(())
        }
        Instr::Sum { src, .. }
        | Instr::Reshape { src, .. }
        | Instr::ExpandDims { src, .. }
        | Instr::Squeeze { src, .. }
        | Instr::Index { src, .. }
        | Instr::Slice { src, .. }
        | Instr::Gather { src, .. } => {
            ops.add_grad(*src, upstream);
            Ok(())
        }
        Instr::Output(_) => Ok(()),
    }
}
