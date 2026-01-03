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

use std::fmt::Write;

use crate::ir::{BinOp, IRModule, Instr, ValueId};
use crate::types::ConvPadding;
use crate::types::{DType, ShapeDim};

/// Format an [`IRModule`] into a stable, human-readable string.
pub fn format_ir_module(module: &IRModule) -> String {
    let mut out = String::new();
    writeln!(&mut out, "module {{").expect("write to string cannot fail");
    for instr in &module.instrs {
        format_instr(instr, &mut out);
    }
    writeln!(&mut out, "}}  // next_id = {}", module.next_id).expect("write to string cannot fail");
    out
}

fn format_instr(instr: &Instr, out: &mut String) {
    match instr {
        Instr::ConstI64(id, value) => {
            writeln!(out, "  {} = const.i64 {}", value_name(*id), value).unwrap();
        }
        Instr::ConstTensor(id, dtype, shape, fill) => {
            writeln!(
                out,
                "  {} = const.tensor {} {:?} fill={:?}",
                value_name(*id),
                format_dtype(dtype),
                format_shape(shape),
                fill
            )
            .unwrap();
        }
        Instr::BinOp { dst, op, lhs, rhs } => {
            writeln!(
                out,
                "  {} = {} {}, {}",
                value_name(*dst),
                format_binop(*op),
                value_name(*lhs),
                value_name(*rhs)
            )
            .unwrap();
        }
        Instr::Sum {
            dst,
            src,
            axes,
            keepdims,
        } => {
            writeln!(
                out,
                "  {} = sum {} axes={:?} keepdims={}",
                value_name(*dst),
                value_name(*src),
                axes,
                keepdims
            )
            .unwrap();
        }
        Instr::Mean {
            dst,
            src,
            axes,
            keepdims,
        } => {
            writeln!(
                out,
                "  {} = mean {} axes={:?} keepdims={}",
                value_name(*dst),
                value_name(*src),
                axes,
                keepdims
            )
            .unwrap();
        }
        Instr::Reshape {
            dst,
            src,
            new_shape,
        } => {
            writeln!(
                out,
                "  {} = reshape {} {:?}",
                value_name(*dst),
                value_name(*src),
                format_shape(new_shape)
            )
            .unwrap();
        }
        Instr::ExpandDims { dst, src, axis } => {
            writeln!(
                out,
                "  {} = expand_dims {} axis={}",
                value_name(*dst),
                value_name(*src),
                axis
            )
            .unwrap();
        }
        Instr::Squeeze { dst, src, axes } => {
            writeln!(
                out,
                "  {} = squeeze {} axes={:?}",
                value_name(*dst),
                value_name(*src),
                axes
            )
            .unwrap();
        }
        Instr::Transpose { dst, src, perm } => {
            writeln!(
                out,
                "  {} = transpose {} perm={:?}",
                value_name(*dst),
                value_name(*src),
                perm
            )
            .unwrap();
        }
        Instr::Dot { dst, a, b } => {
            writeln!(
                out,
                "  {} = dot {}, {}",
                value_name(*dst),
                value_name(*a),
                value_name(*b)
            )
            .unwrap();
        }
        Instr::MatMul { dst, a, b } => {
            writeln!(
                out,
                "  {} = matmul {}, {}",
                value_name(*dst),
                value_name(*a),
                value_name(*b)
            )
            .unwrap();
        }
        Instr::Conv2d {
            dst,
            input,
            filter,
            stride_h,
            stride_w,
            padding,
        } => {
            writeln!(
                out,
                "  {} = conv2d {} {} strides=({}, {}) padding={}",
                value_name(*dst),
                value_name(*input),
                value_name(*filter),
                stride_h,
                stride_w,
                format_padding(*padding)
            )
            .unwrap();
        }
        Instr::Conv2dGradInput {
            dst,
            dy,
            filter,
            input_shape,
            stride_h,
            stride_w,
            padding,
        } => {
            writeln!(
                out,
                "  {} = conv2d_grad_input {} {} input_shape={:?} strides=({}, {}) padding={}",
                value_name(*dst),
                value_name(*dy),
                value_name(*filter),
                input_shape,
                stride_h,
                stride_w,
                format_padding(*padding)
            )
            .unwrap();
        }
        Instr::Conv2dGradFilter {
            dst,
            input,
            dy,
            filter_shape,
            stride_h,
            stride_w,
            padding,
        } => {
            writeln!(
                out,
                "  {} = conv2d_grad_filter {} {} filter_shape={:?} strides=({}, {}) padding={}",
                value_name(*dst),
                value_name(*input),
                value_name(*dy),
                filter_shape,
                stride_h,
                stride_w,
                format_padding(*padding)
            )
            .unwrap();
        }
        Instr::Index { dst, src, indices } => {
            writeln!(
                out,
                "  {} = index {} {:?}",
                value_name(*dst),
                value_name(*src),
                indices
            )
            .unwrap();
        }
        Instr::Slice { dst, src, dims } => {
            writeln!(
                out,
                "  {} = slice {} {:?}",
                value_name(*dst),
                value_name(*src),
                dims
            )
            .unwrap();
        }
        Instr::Gather {
            dst,
            src,
            indices,
            axis,
        } => {
            writeln!(
                out,
                "  {} = gather {} indices={} axis={}",
                value_name(*dst),
                value_name(*src),
                value_name(*indices),
                axis
            )
            .unwrap();
        }
        Instr::Output(id) => {
            writeln!(out, "  output {}", value_name(*id)).unwrap();
        }
        _ => {}
    }
}

fn value_name(id: ValueId) -> String {
    format!("%{}", id.0)
}

fn format_binop(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
    }
}

fn format_dtype(dtype: &DType) -> String {
    format!("{:?}", dtype)
}

fn format_shape(shape: &[ShapeDim]) -> Vec<String> {
    shape
        .iter()
        .map(|dim| match dim {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(sym) => sym.to_string(),
        })
        .collect()
}

fn format_padding(padding: ConvPadding) -> &'static str {
    padding.as_str()
}
