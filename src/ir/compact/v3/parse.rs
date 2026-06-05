// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! MIC@3 binary parser — deserializes bytes into [`IRModule`].

use std::io::{Cursor, Read};

use crate::ir::{IRModule, IndexSpec, Instr, SliceSpec, ValueId};
use crate::types::ShapeDim;
use crate::types::intern::intern_str;

use super::emit::{
    OP_ARRAY_LOAD, OP_BINOP, OP_BREAK, OP_CALL, OP_CONST_ARRAY, OP_CONST_F64, OP_CONST_I64,
    OP_CONST_TENSOR, OP_CONTINUE, OP_CONV2D, OP_CONV2D_GRAD_FILTER, OP_CONV2D_GRAD_INPUT, OP_DOT,
    OP_EXPAND_DIMS, OP_EXTERN_FN_DECL, OP_FN_DEF, OP_GATHER, OP_IF, OP_INDEX, OP_MATMUL, OP_MEAN,
    OP_OUTPUT, OP_PARAM, OP_REGION, OP_RELU, OP_RELU_GRAD, OP_RESHAPE, OP_RETURN, OP_SLICE,
    OP_SPARSE_ATTR, OP_SQUEEZE, OP_SUM, OP_TRANSPOSE, OP_VEC_FMA, OP_VEC_LOAD, OP_VEC_LOAD_I32,
    OP_VEC_MUL_ADD_Q16, OP_VEC_REDUCE_ADD, OP_VEC_REDUCE_ADD_I64, OP_VEC_STORE, OP_WHILE,
    byte_to_binop, byte_to_dtype, byte_to_padding, byte_to_sparse_layout,
};
use super::{MIC3_MAGIC, MIC3_VERSION};
use crate::ir::compact::v2::{uleb128_read, zigzag_decode};

// ─── Error type ──────────────────────────────────────────────────────────────

/// Error produced by [`parse_mic3`].
#[derive(Debug, Clone)]
pub struct Mic3Error {
    pub message: String,
}

impl std::fmt::Display for Mic3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mic3: {}", self.message)
    }
}

impl std::error::Error for Mic3Error {}

impl From<std::io::Error> for Mic3Error {
    fn from(e: std::io::Error) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

macro_rules! err {
    ($($t:tt)*) => {
        Mic3Error { message: format!($($t)*) }
    };
}

// ─── DoS hardening limits ─────────────────────────────────────────────────────

/// Maximum accepted mic@3 input size (bytes). Mirrors the mic@1 / v2 parser
/// `MAX_INPUT_SIZE` policy so the binary serialization is bounded the same way
/// the textual ones are. Inputs larger than this are rejected up front, before
/// any allocation, so an untrusted artifact cannot drive unbounded memory use.
pub const MAX_MIC3_INPUT: usize = 10 * 1024 * 1024;

/// Maximum recursion depth for the nested instruction / type-annotation
/// decoders. Mirrors the v2 parser's `MAX_NESTING` discipline. A few hundred
/// frames comfortably covers any legitimate deeply-nested control flow while
/// failing long before native stack exhaustion (~8–12k frames).
const MAX_MIC3_DEPTH: usize = 256;

/// Bound an untrusted element count against the total input length so that a
/// tiny crafted header cannot request a huge `Vec::with_capacity`. Every wire
/// element occupies at least one byte, so a valid count never exceeds the input
/// length — this clamp is therefore a no-op for well-formed input (byte-identity
/// preserved) and only caps adversarial counts. The reader still validates the
/// real element bytes as it pushes, so under-reservation is harmless.
#[inline]
fn bounded_cap(n: usize, limit: usize) -> usize {
    n.min(limit)
}

// ─── Low-level helpers ────────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8, Mic3Error> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, Mic3Error> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_uleb<R: Read>(r: &mut R) -> Result<u64, Mic3Error> {
    uleb128_read(r).map_err(Into::into)
}

fn read_vid<R: Read>(r: &mut R) -> Result<ValueId, Mic3Error> {
    Ok(ValueId(read_uleb(r)? as usize))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, Mic3Error> {
    Ok(zigzag_decode(read_uleb(r)?))
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool, Mic3Error> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R, strings: &[String]) -> Result<String, Mic3Error> {
    let idx = read_uleb(r)? as usize;
    strings.get(idx).cloned().ok_or_else(|| {
        err!(
            "string index {} out of bounds (table size {})",
            idx,
            strings.len()
        )
    })
}

fn read_opt_vid<R: Read>(r: &mut R) -> Result<Option<ValueId>, Mic3Error> {
    match read_u8(r)? {
        0 => Ok(None),
        1 => Ok(Some(read_vid(r)?)),
        tag => Err(err!("invalid opt-vid tag: {}", tag)),
    }
}

fn read_opt_f64<R: Read>(r: &mut R) -> Result<Option<f64>, Mic3Error> {
    match read_u8(r)? {
        0 => Ok(None),
        1 => {
            let bits = read_u64_le(r)?;
            Ok(Some(f64::from_bits(bits)))
        }
        tag => Err(err!("invalid opt-f64 tag: {}", tag)),
    }
}

#[cfg(feature = "std-surface")]
fn read_opt_string<R: Read>(r: &mut R, strings: &[String]) -> Result<Option<String>, Mic3Error> {
    match read_u8(r)? {
        0 => Ok(None),
        1 => Ok(Some(read_string(r, strings)?)),
        tag => Err(err!("invalid opt-string tag: {}", tag)),
    }
}

fn read_opt_i64<R: Read>(r: &mut R) -> Result<Option<i64>, Mic3Error> {
    match read_u8(r)? {
        0 => Ok(None),
        1 => Ok(Some(read_i64(r)?)),
        tag => Err(err!("invalid opt-i64 tag: {}", tag)),
    }
}

fn read_i64_vec<R: Read>(r: &mut R, limit: usize) -> Result<Vec<i64>, Mic3Error> {
    let n = read_uleb(r)? as usize;
    let mut v = Vec::with_capacity(bounded_cap(n, limit));
    for _ in 0..n {
        v.push(read_i64(r)?);
    }
    Ok(v)
}

fn read_vid_vec<R: Read>(r: &mut R, limit: usize) -> Result<Vec<ValueId>, Mic3Error> {
    let n = read_uleb(r)? as usize;
    let mut v = Vec::with_capacity(bounded_cap(n, limit));
    for _ in 0..n {
        v.push(read_vid(r)?);
    }
    Ok(v)
}

fn read_named_vids<R: Read>(
    r: &mut R,
    strings: &[String],
    limit: usize,
) -> Result<Vec<(String, ValueId)>, Mic3Error> {
    let n = read_uleb(r)? as usize;
    let mut v = Vec::with_capacity(bounded_cap(n, limit));
    for _ in 0..n {
        let name = read_string(r, strings)?;
        let id = read_vid(r)?;
        v.push((name, id));
    }
    Ok(v)
}

fn read_shape_dim<R: Read>(r: &mut R, strings: &[String]) -> Result<ShapeDim, Mic3Error> {
    match read_u8(r)? {
        0 => {
            let n = read_uleb(r)? as usize;
            Ok(ShapeDim::Known(n))
        }
        1 => {
            let s = read_string(r, strings)?;
            Ok(ShapeDim::Sym(intern_str(&s)))
        }
        tag => Err(err!("unknown shape-dim tag: {}", tag)),
    }
}

fn read_shape<R: Read>(
    r: &mut R,
    strings: &[String],
    limit: usize,
) -> Result<Vec<ShapeDim>, Mic3Error> {
    let n = read_uleb(r)? as usize;
    let mut dims = Vec::with_capacity(bounded_cap(n, limit));
    for _ in 0..n {
        dims.push(read_shape_dim(r, strings)?);
    }
    Ok(dims)
}

// ─── TypeAnn decode (std-surface) ─────────────────────────────────────────────

#[cfg(feature = "std-surface")]
fn read_type_ann<R: Read>(
    r: &mut R,
    strings: &[String],
    depth: usize,
    limit: usize,
) -> Result<crate::ast::TypeAnn, Mic3Error> {
    use super::emit::byte_to_sparse_layout;
    use crate::ast::TypeAnn;

    if depth >= MAX_MIC3_DEPTH {
        return Err(err!(
            "type-annotation nesting depth exceeds limit {}",
            MAX_MIC3_DEPTH
        ));
    }
    let depth = depth + 1;

    let tag = read_u8(r)?;
    match tag {
        0x01 => Ok(TypeAnn::ScalarI32),
        0x02 => Ok(TypeAnn::ScalarI64),
        0x03 => Ok(TypeAnn::ScalarF32),
        0x04 => Ok(TypeAnn::ScalarF64),
        0x05 => Ok(TypeAnn::ScalarBool),
        0x06 => Ok(TypeAnn::ScalarU32),
        0x07 | 0x08 => {
            let dtype = read_string(r, strings)?;
            let nd = read_uleb(r)? as usize;
            let mut dims = Vec::with_capacity(bounded_cap(nd, limit));
            for _ in 0..nd {
                dims.push(read_string(r, strings)?);
            }
            if tag == 0x07 {
                Ok(TypeAnn::Tensor { dtype, dims })
            } else {
                Ok(TypeAnn::DiffTensor { dtype, dims })
            }
        }
        0x09 => {
            let name = read_string(r, strings)?;
            Ok(TypeAnn::Named(name))
        }
        0x0A => {
            let mutable = read_bool(r)?;
            let element = read_type_ann(r, strings, depth, limit)?;
            Ok(TypeAnn::Slice {
                mutable,
                element: Box::new(element),
            })
        }
        0x0B => {
            let length = read_uleb(r)? as u32;
            let element = read_type_ann(r, strings, depth, limit)?;
            Ok(TypeAnn::Array {
                element: Box::new(element),
                length,
            })
        }
        0x0C => {
            let mutable = read_bool(r)?;
            let target = read_type_ann(r, strings, depth, limit)?;
            Ok(TypeAnn::Ref {
                mutable,
                target: Box::new(target),
            })
        }
        0x0D => {
            let name = read_string(r, strings)?;
            let na = read_uleb(r)? as usize;
            let mut args = Vec::with_capacity(bounded_cap(na, limit));
            for _ in 0..na {
                args.push(read_type_ann(r, strings, depth, limit)?);
            }
            Ok(TypeAnn::Generic { name, args })
        }
        0x0E => {
            let ne = read_uleb(r)? as usize;
            let mut elements = Vec::with_capacity(bounded_cap(ne, limit));
            for _ in 0..ne {
                elements.push(read_type_ann(r, strings, depth, limit)?);
            }
            Ok(TypeAnn::Tuple { elements })
        }
        0x0F => {
            let layout_byte = read_u8(r)?;
            let layout = byte_to_sparse_layout(layout_byte)
                .ok_or_else(|| err!("unknown sparse layout byte: {}", layout_byte))?;
            let element = read_type_ann(r, strings, depth, limit)?;
            let ns = read_uleb(r)? as usize;
            let mut shape = Vec::with_capacity(bounded_cap(ns, limit));
            for _ in 0..ns {
                shape.push(read_shape_dim(r, strings)?);
            }
            Ok(TypeAnn::SparseTensor {
                layout,
                element: Box::new(element),
                shape,
            })
        }
        0x10 => {
            let mutable = read_bool(r)?;
            let pointee = read_type_ann(r, strings, depth, limit)?;
            Ok(TypeAnn::RawPtr {
                mutable,
                pointee: Box::new(pointee),
            })
        }
        0x11 => {
            let np = read_uleb(r)? as usize;
            let mut params = Vec::with_capacity(bounded_cap(np, limit));
            for _ in 0..np {
                params.push(read_type_ann(r, strings, depth, limit)?);
            }
            let has_ret = read_u8(r)? != 0;
            let ret = if has_ret {
                Some(Box::new(read_type_ann(r, strings, depth, limit)?))
            } else {
                None
            };
            Ok(TypeAnn::FnPtr { params, ret })
        }
        _ => Err(err!("unknown TypeAnn tag: {}", tag)),
    }
}

// ─── Instruction decoder ──────────────────────────────────────────────────────

fn decode_instr<R: Read>(
    r: &mut R,
    strings: &[String],
    depth: usize,
    limit: usize,
) -> Result<Instr, Mic3Error> {
    if depth >= MAX_MIC3_DEPTH {
        return Err(err!(
            "instruction nesting depth exceeds limit {}",
            MAX_MIC3_DEPTH
        ));
    }
    let depth = depth + 1;

    let op = read_u8(r)?;
    match op {
        OP_CONST_I64 => {
            let dst = read_vid(r)?;
            let v = read_i64(r)?;
            Ok(Instr::ConstI64(dst, v))
        }
        OP_CONST_F64 => {
            let dst = read_vid(r)?;
            let bits = read_u64_le(r)?;
            Ok(Instr::ConstF64(dst, f64::from_bits(bits)))
        }
        OP_CONST_TENSOR => {
            let dst = read_vid(r)?;
            let dtype_byte = read_u8(r)?;
            let dtype = byte_to_dtype(dtype_byte)
                .ok_or_else(|| err!("unknown dtype byte: {}", dtype_byte))?;
            let shape = read_shape(r, strings, limit)?;
            let fill = read_opt_f64(r)?;
            Ok(Instr::ConstTensor(dst, dtype, shape, fill))
        }
        OP_BINOP => {
            let dst = read_vid(r)?;
            let op_byte = read_u8(r)?;
            let op =
                byte_to_binop(op_byte).ok_or_else(|| err!("unknown binop byte: {}", op_byte))?;
            let lhs = read_vid(r)?;
            let rhs = read_vid(r)?;
            Ok(Instr::BinOp { dst, op, lhs, rhs })
        }
        OP_SUM => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let axes = read_i64_vec(r, limit)?;
            let keepdims = read_bool(r)?;
            Ok(Instr::Sum {
                dst,
                src,
                axes,
                keepdims,
            })
        }
        OP_MEAN => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let axes = read_i64_vec(r, limit)?;
            let keepdims = read_bool(r)?;
            Ok(Instr::Mean {
                dst,
                src,
                axes,
                keepdims,
            })
        }
        OP_RELU => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            Ok(Instr::Relu { dst, src })
        }
        OP_RELU_GRAD => {
            let dst = read_vid(r)?;
            let grad = read_vid(r)?;
            let src = read_vid(r)?;
            Ok(Instr::ReluGrad { dst, grad, src })
        }
        OP_RESHAPE => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let new_shape = read_shape(r, strings, limit)?;
            Ok(Instr::Reshape {
                dst,
                src,
                new_shape,
            })
        }
        OP_EXPAND_DIMS => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let axis = read_i64(r)?;
            Ok(Instr::ExpandDims { dst, src, axis })
        }
        OP_SQUEEZE => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let axes = read_i64_vec(r, limit)?;
            Ok(Instr::Squeeze { dst, src, axes })
        }
        OP_TRANSPOSE => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let perm = read_i64_vec(r, limit)?;
            Ok(Instr::Transpose { dst, src, perm })
        }
        OP_DOT => {
            let dst = read_vid(r)?;
            let a = read_vid(r)?;
            let b = read_vid(r)?;
            Ok(Instr::Dot { dst, a, b })
        }
        OP_MATMUL => {
            let dst = read_vid(r)?;
            let a = read_vid(r)?;
            let b = read_vid(r)?;
            Ok(Instr::MatMul { dst, a, b })
        }
        OP_CONV2D => {
            let dst = read_vid(r)?;
            let input = read_vid(r)?;
            let filter = read_vid(r)?;
            let stride_h = read_uleb(r)? as usize;
            let stride_w = read_uleb(r)? as usize;
            let pb = read_u8(r)?;
            let padding =
                byte_to_padding(pb).ok_or_else(|| err!("unknown padding byte: {}", pb))?;
            Ok(Instr::Conv2d {
                dst,
                input,
                filter,
                stride_h,
                stride_w,
                padding,
            })
        }
        OP_CONV2D_GRAD_INPUT => {
            let dst = read_vid(r)?;
            let dy = read_vid(r)?;
            let filter = read_vid(r)?;
            let mut input_shape = [0usize; 4];
            for s in &mut input_shape {
                *s = read_uleb(r)? as usize;
            }
            let stride_h = read_uleb(r)? as usize;
            let stride_w = read_uleb(r)? as usize;
            let pb = read_u8(r)?;
            let padding =
                byte_to_padding(pb).ok_or_else(|| err!("unknown padding byte: {}", pb))?;
            Ok(Instr::Conv2dGradInput {
                dst,
                dy,
                filter,
                input_shape,
                stride_h,
                stride_w,
                padding,
            })
        }
        OP_CONV2D_GRAD_FILTER => {
            let dst = read_vid(r)?;
            let input = read_vid(r)?;
            let dy = read_vid(r)?;
            let mut filter_shape = [0usize; 4];
            for s in &mut filter_shape {
                *s = read_uleb(r)? as usize;
            }
            let stride_h = read_uleb(r)? as usize;
            let stride_w = read_uleb(r)? as usize;
            let pb = read_u8(r)?;
            let padding =
                byte_to_padding(pb).ok_or_else(|| err!("unknown padding byte: {}", pb))?;
            Ok(Instr::Conv2dGradFilter {
                dst,
                input,
                dy,
                filter_shape,
                stride_h,
                stride_w,
                padding,
            })
        }
        OP_INDEX => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let n = read_uleb(r)? as usize;
            let mut indices = Vec::with_capacity(bounded_cap(n, limit));
            for _ in 0..n {
                let axis = read_i64(r)?;
                let index = read_i64(r)?;
                indices.push(IndexSpec { axis, index });
            }
            Ok(Instr::Index { dst, src, indices })
        }
        OP_SLICE => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let n = read_uleb(r)? as usize;
            let mut dims = Vec::with_capacity(bounded_cap(n, limit));
            for _ in 0..n {
                let axis = read_i64(r)?;
                let start = read_i64(r)?;
                let end = read_opt_i64(r)?;
                let stride = read_i64(r)?;
                dims.push(SliceSpec {
                    axis,
                    start,
                    end,
                    stride,
                });
            }
            Ok(Instr::Slice { dst, src, dims })
        }
        OP_GATHER => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let indices = read_vid(r)?;
            let axis = read_i64(r)?;
            Ok(Instr::Gather {
                dst,
                src,
                indices,
                axis,
            })
        }
        OP_OUTPUT => {
            let id = read_vid(r)?;
            Ok(Instr::Output(id))
        }
        OP_SPARSE_ATTR => {
            let src = read_vid(r)?;
            let dst = read_vid(r)?;
            let lb = read_u8(r)?;
            let layout = byte_to_sparse_layout(lb)
                .ok_or_else(|| err!("unknown sparse layout byte: {}", lb))?;
            Ok(Instr::SparseAttr { src, dst, layout })
        }
        OP_FN_DEF => {
            let name = read_string(r, strings)?;
            let params = read_named_vids(r, strings, limit)?;
            let ret_id = read_opt_vid(r)?;
            let reap_threshold = read_opt_f64(r)?;
            let body_len = read_uleb(r)? as usize;
            let mut body = Vec::with_capacity(body_len);
            for _ in 0..body_len {
                body.push(decode_instr(r, strings, depth, limit)?);
            }
            Ok(Instr::FnDef {
                name,
                params,
                ret_id,
                body,
                reap_threshold,
            })
        }
        OP_CALL => {
            let dst = read_vid(r)?;
            let name = read_string(r, strings)?;
            let args = read_vid_vec(r, limit)?;
            Ok(Instr::Call { dst, name, args })
        }
        OP_RETURN => {
            let value = read_opt_vid(r)?;
            Ok(Instr::Return { value })
        }
        OP_PARAM => {
            let dst = read_vid(r)?;
            let name = read_string(r, strings)?;
            let index = read_uleb(r)? as usize;
            Ok(Instr::Param { dst, name, index })
        }
        #[cfg(feature = "std-surface")]
        OP_CONST_ARRAY => {
            let dst = read_vid(r)?;
            let name = read_opt_string(r, strings)?;
            let n = read_uleb(r)? as usize;
            let mut values = Vec::with_capacity(bounded_cap(n, limit));
            for _ in 0..n {
                values.push(zigzag_decode(read_uleb(r)?));
            }
            Ok(Instr::ConstArray { dst, name, values })
        }
        #[cfg(feature = "std-surface")]
        OP_ARRAY_LOAD => {
            let dst = read_vid(r)?;
            let base = read_vid(r)?;
            let index = read_vid(r)?;
            Ok(Instr::ArrayLoad { dst, base, index })
        }
        #[cfg(feature = "std-surface")]
        OP_WHILE => {
            let cond_id = read_vid(r)?;
            let cond_len = read_uleb(r)? as usize;
            let mut cond_instrs = Vec::with_capacity(cond_len);
            for _ in 0..cond_len {
                cond_instrs.push(decode_instr(r, strings, depth, limit)?);
            }
            let body_len = read_uleb(r)? as usize;
            let mut body = Vec::with_capacity(body_len);
            for _ in 0..body_len {
                body.push(decode_instr(r, strings, depth, limit)?);
            }
            let live_vars = read_named_vids(r, strings, limit)?;
            let init_ids = read_vid_vec(r, limit)?;
            Ok(Instr::While {
                cond_id,
                cond_instrs,
                body,
                live_vars,
                init_ids,
                // F2: lowering-internal, not in the wire format. Reconstructed
                // during AST->IR lowering; default empty on parse.
                exit_ids: Vec::new(),
            })
        }
        #[cfg(feature = "std-surface")]
        OP_IF => {
            let cond_id = read_vid(r)?;
            let cond_len = read_uleb(r)? as usize;
            let mut cond_instrs = Vec::with_capacity(cond_len);
            for _ in 0..cond_len {
                cond_instrs.push(decode_instr(r, strings, depth, limit)?);
            }
            let then_len = read_uleb(r)? as usize;
            let mut then_instrs = Vec::with_capacity(then_len);
            for _ in 0..then_len {
                then_instrs.push(decode_instr(r, strings, depth, limit)?);
            }
            let then_result = read_vid(r)?;
            let else_len = read_uleb(r)? as usize;
            let mut else_instrs = Vec::with_capacity(else_len);
            for _ in 0..else_len {
                else_instrs.push(decode_instr(r, strings, depth, limit)?);
            }
            let else_result = read_vid(r)?;
            let dst = read_vid(r)?;
            let branch_bindings = read_named_vids(r, strings, limit)?;
            Ok(Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                dst,
                branch_bindings,
                // F2: lowering-internal, not in the wire format. Reconstructed
                // during AST->IR lowering; default empty on parse.
                merges: Vec::new(),
            })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_LOAD => {
            let dst = read_vid(r)?;
            let base = read_vid(r)?;
            let offset = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecLoad {
                dst,
                base,
                offset,
                lanes,
            })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_FMA => {
            let dst = read_vid(r)?;
            let a = read_vid(r)?;
            let b = read_vid(r)?;
            let acc = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecFma {
                dst,
                a,
                b,
                acc,
                lanes,
            })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_REDUCE_ADD => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecReduceAdd { dst, src, lanes })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_STORE => {
            let src = read_vid(r)?;
            let base = read_vid(r)?;
            let offset = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecStore {
                src,
                base,
                offset,
                lanes,
            })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_LOAD_I32 => {
            let dst = read_vid(r)?;
            let base = read_vid(r)?;
            let offset = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecLoadI32 {
                dst,
                base,
                offset,
                lanes,
            })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_MUL_ADD_Q16 => {
            let dst = read_vid(r)?;
            let a = read_vid(r)?;
            let b = read_vid(r)?;
            let acc = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecMulAddQ16 {
                dst,
                a,
                b,
                acc,
                lanes,
            })
        }
        #[cfg(feature = "std-surface")]
        OP_VEC_REDUCE_ADD_I64 => {
            let dst = read_vid(r)?;
            let src = read_vid(r)?;
            let lanes = read_uleb(r)? as usize;
            Ok(Instr::VecReduceAddI64 { dst, src, lanes })
        }
        #[cfg(feature = "std-surface")]
        OP_REGION => {
            let body_len = read_uleb(r)? as usize;
            let mut body = Vec::with_capacity(body_len);
            for _ in 0..body_len {
                body.push(decode_instr(r, strings, depth, limit)?);
            }
            let result = read_vid(r)?;
            let enter_id = read_vid(r)?;
            let exit_id = read_vid(r)?;
            let alloc_ids = read_vid_vec(r, limit)?;
            Ok(Instr::Region {
                body,
                result,
                enter_id,
                exit_id,
                alloc_ids,
            })
        }
        #[cfg(feature = "std-surface")]
        OP_EXTERN_FN_DECL => {
            use super::emit::byte_to_callconv;
            let name = read_string(r, strings)?;
            let np = read_uleb(r)? as usize;
            let mut param_types = Vec::with_capacity(np);
            for _ in 0..np {
                param_types.push(read_string(r, strings)?);
            }
            let ret_type = read_opt_string(r, strings)?;
            let is_varargs = read_bool(r)?;
            let nh = read_uleb(r)? as usize;
            let mut vararg_hints = Vec::with_capacity(nh);
            for _ in 0..nh {
                vararg_hints.push(read_string(r, strings)?);
            }
            let cb = read_u8(r)?;
            let callconv =
                byte_to_callconv(cb).ok_or_else(|| err!("unknown callconv byte: {}", cb))?;
            Ok(Instr::ExternFnDecl {
                name,
                param_types,
                ret_type,
                is_varargs,
                vararg_hints,
                callconv,
            })
        }
        // break / continue carry the loop-control `live` snapshot
        // (`name -> ValueId`), serialized identically to `live_vars`.
        #[cfg(feature = "std-surface")]
        OP_BREAK => Ok(Instr::Break {
            live: read_named_vids(r, strings, limit)?,
        }),
        #[cfg(feature = "std-surface")]
        OP_CONTINUE => Ok(Instr::Continue {
            live: read_named_vids(r, strings, limit)?,
        }),
        // Non-std-surface opcodes that are only valid with the feature get this
        // on the non-feature build: they should never appear in practice.
        #[cfg(not(feature = "std-surface"))]
        OP_CONST_ARRAY
        | OP_ARRAY_LOAD
        | OP_WHILE
        | OP_IF
        | OP_VEC_LOAD
        | OP_VEC_FMA
        | OP_VEC_REDUCE_ADD
        | OP_VEC_STORE
        | OP_VEC_LOAD_I32
        | OP_VEC_MUL_ADD_Q16
        | OP_VEC_REDUCE_ADD_I64
        | OP_REGION
        | OP_BREAK
        | OP_CONTINUE
        | OP_EXTERN_FN_DECL => Err(err!(
            "opcode 0x{:02X} requires std-surface feature (not enabled in this build)",
            op
        )),
        _ => Err(err!("unknown opcode: 0x{:02X}", op)),
    }
}

// ─── Top-level entry point ────────────────────────────────────────────────────

/// Parse MIC@3 binary bytes into an [`IRModule`].
///
/// Returns [`Mic3Error`] if the bytes are malformed or the magic / version do
/// not match. No other format is accepted — detect `MIC3` magic before calling.
pub fn parse_mic3(data: &[u8]) -> Result<IRModule, Mic3Error> {
    // DoS guard: reject oversized input up front, before any allocation, so an
    // untrusted artifact cannot drive unbounded work. Mirrors the mic@1 / v2
    // parser `MAX_INPUT_SIZE` policy.
    if data.len() > MAX_MIC3_INPUT {
        return Err(err!(
            "mic@3 input too large: {} bytes (max {})",
            data.len(),
            MAX_MIC3_INPUT
        ));
    }
    // Every wire element occupies at least one byte; the input length is a hard
    // ceiling on any untrusted element count, used to bound pre-allocation.
    let limit = data.len();
    let mut r = Cursor::new(data);

    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MIC3_MAGIC {
        return Err(err!(
            "invalid MIC3 magic: expected {:?}, got {:?}",
            MIC3_MAGIC,
            magic
        ));
    }

    // Version
    let version = read_u8(&mut r)?;
    if version != MIC3_VERSION {
        return Err(err!(
            "unsupported MIC3 version: expected 0x{:02X}, got 0x{:02X}",
            MIC3_VERSION,
            version
        ));
    }

    // String table
    let n_strings = read_uleb(&mut r)? as usize;
    let mut strings: Vec<String> = Vec::with_capacity(bounded_cap(n_strings, limit));
    for _ in 0..n_strings {
        let len = read_uleb(&mut r)? as usize;
        if len > limit {
            return Err(err!(
                "string-table entry length {} exceeds input size {}",
                len,
                limit
            ));
        }
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)?;
        let s = String::from_utf8(buf).map_err(|_| err!("invalid UTF-8 in string table"))?;
        strings.push(s);
    }

    // next_id
    let next_id = read_uleb(&mut r)? as usize;

    // Exports
    let n_exports = read_uleb(&mut r)? as usize;
    let mut exports = std::collections::HashSet::new();
    for _ in 0..n_exports {
        let s = read_string(&mut r, &strings)?;
        exports.insert(s);
    }

    // Instructions
    let n_instrs = read_uleb(&mut r)? as usize;
    let mut instrs = Vec::with_capacity(bounded_cap(n_instrs, limit));
    for _ in 0..n_instrs {
        instrs.push(decode_instr(&mut r, &strings, 0, limit)?);
    }

    #[allow(unused_mut)]
    let mut module = IRModule {
        instrs,
        next_id,
        exports,
        #[cfg(feature = "std-surface")]
        struct_defs: std::collections::BTreeMap::new(),
        #[cfg(feature = "std-surface")]
        const_array_defs: std::collections::BTreeMap::new(),
        #[cfg(feature = "std-surface")]
        repr_c_structs: std::collections::BTreeMap::new(),
        // The enum-discriminant table is a lowering-only side-table; it is
        // never serialised into mic@3, so the parse path leaves it empty (no
        // wire-format change, no version bump).
        #[cfg(feature = "std-surface")]
        enum_variant_tags: std::collections::BTreeMap::new(),
    };

    // std-surface registries
    #[cfg(feature = "std-surface")]
    {
        // struct_defs
        let n_sd = read_uleb(&mut r)? as usize;
        for _ in 0..n_sd {
            let name = read_string(&mut r, &strings)?;
            let nf = read_uleb(&mut r)? as usize;
            let mut fields = Vec::with_capacity(bounded_cap(nf, limit));
            for _ in 0..nf {
                fields.push(read_string(&mut r, &strings)?);
            }
            module.struct_defs.insert(name, fields);
        }

        // const_array_defs
        let n_cad = read_uleb(&mut r)? as usize;
        for _ in 0..n_cad {
            let name = read_string(&mut r, &strings)?;
            let nv = read_uleb(&mut r)? as usize;
            let mut vals = Vec::with_capacity(bounded_cap(nv, limit));
            for _ in 0..nv {
                vals.push(zigzag_decode(read_uleb(&mut r)?));
            }
            module.const_array_defs.insert(name, vals);
        }

        // repr_c_structs
        let n_rcs = read_uleb(&mut r)? as usize;
        for _ in 0..n_rcs {
            let name = read_string(&mut r, &strings)?;
            let nf = read_uleb(&mut r)? as usize;
            let mut fields = Vec::with_capacity(bounded_cap(nf, limit));
            for _ in 0..nf {
                fields.push(read_type_ann(&mut r, &strings, 0, limit)?);
            }
            module.repr_c_structs.insert(name, fields);
        }
    }

    Ok(module)
}
