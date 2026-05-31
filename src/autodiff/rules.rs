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
use super::engine::{AutodiffError, as_invalid};

pub(super) fn apply_rule(
    ops: &mut impl GradientOps,
    instr: &Instr,
    upstream: ValueId,
) -> Result<(), AutodiffError> {
    match instr {
        // Constant leaves: no input to propagate a gradient into. ConstF64 is
        // included because `x * 2.0`-style programs give a float literal an
        // upstream gradient, and dropping it here is correct (it has no operands).
        Instr::ConstI64(..) | Instr::ConstF64(..) | Instr::ConstTensor(..) => Ok(()),
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
                    // c = a / b.  dc/da = 1/b      -> da = upstream / b.
                    //             dc/db = -a / b^2  -> db = -(upstream * a) / (b * b).
                    // Quotient-rule backward built entirely from existing binops
                    // (Div/Mul) + a scalar -1, so the gradient IR lowers through
                    // the same deterministic, Q16.16-safe paths as the forward op.
                    let dlhs = ops.add_binop(BinOp::Div, upstream, *rhs);
                    ops.add_grad(*lhs, dlhs);

                    let num = ops.add_binop(BinOp::Mul, upstream, *lhs);
                    let denom = ops.add_binop(BinOp::Mul, *rhs, *rhs);
                    let quot = ops.add_binop(BinOp::Div, num, denom);
                    let neg_one = ops.add_const_i64(-1);
                    let drhs = ops.add_binop(BinOp::Mul, quot, neg_one);
                    ops.add_grad(*rhs, drhs);
                }
                BinOp::Mod => {
                    return Err(AutodiffError::UnsupportedOp { op: "mod" });
                }
                // Comparison ops are non-differentiable (gradient is zero)
                BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                    // No gradient contribution from comparison operations
                }
                // Bitwise / shift ops are integer bit-manipulation, not
                // differentiable. Fail loudly (like Div/Mod) rather than emit a
                // silent zero gradient — a bitwise op in an autodiff graph is
                // almost certainly an error, and a silent zero would mask it
                // (the silent-wrong-training trap). These BinOp variants are
                // `std-surface`-gated, so the arm carries the same cfg — without
                // it the variants don't exist and the match is already complete.
                #[cfg(feature = "std-surface")]
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    return Err(AutodiffError::UnsupportedOp {
                        op: "bitwise/shift",
                    });
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
        Instr::Conv2d {
            input,
            filter,
            stride_h,
            stride_w,
            padding,
            ..
        } => {
            // Conv2d backward reuses the existing Conv2dGradInput / Conv2dGradFilter
            // ops. Both need a static 4-D shape (input NHWC / filter HWIO) to size
            // their output; if the shape is dynamic we fail loud rather than emit a
            // conv backward against a guessed shape (silent-wrong training).
            let input_shape =
                ops.static_shape4(*input)
                    .ok_or_else(|| AutodiffError::UnsupportedShape {
                        reason: "conv2d input shape is not statically known (NHWC)".to_string(),
                    })?;
            let filter_shape =
                ops.static_shape4(*filter)
                    .ok_or_else(|| AutodiffError::UnsupportedShape {
                        reason: "conv2d filter shape is not statically known (HWIO)".to_string(),
                    })?;
            let dinput = ops.add_conv2d_grad_input(
                upstream,
                *filter,
                input_shape,
                *stride_h,
                *stride_w,
                *padding,
            );
            ops.add_grad(*input, dinput);
            let dfilter = ops.add_conv2d_grad_filter(
                *input,
                upstream,
                filter_shape,
                *stride_h,
                *stride_w,
                *padding,
            );
            ops.add_grad(*filter, dfilter);
            Ok(())
        }
        Instr::Conv2dGradInput { .. } => Err(AutodiffError::UnsupportedOp {
            op: "conv2d_grad_input",
        }),
        Instr::Conv2dGradFilter { .. } => Err(AutodiffError::UnsupportedOp {
            op: "conv2d_grad_filter",
        }),
        // ReLU backward: dx = grad * step(src), emitted as a dedicated
        // `ReluGrad` op so the masking is a single deterministic elementwise
        // map (cmpf + select) rather than a float comparison routed through an
        // integer-only `cmpi`. `upstream` is the incoming gradient; `src` is the
        // original ReLU input. The step gate is non-differentiable, so no
        // gradient flows back into the mask — only into `src`.
        Instr::Relu { src, .. } => {
            let dsrc = ops.add_relu_grad(upstream, *src);
            ops.add_grad(*src, dsrc);
            Ok(())
        }
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
        // No autodiff rule for this instruction. Fail loud rather than return a
        // silent zero gradient — the wedge forbids silently-wrong training.
        // Differentiation covers the Core-v1 tensor ops (see docs/autodiff.md);
        // function/control-flow/std-surface ops are intentionally not handled.
        _ => Err(as_invalid(
            "no autodiff rule for this IR instruction; differentiation supports the \
             Core-v1 tensor ops only (function, control-flow, and std-surface ops are \
             not differentiable — see docs/autodiff.md)",
        )),
    }
}
