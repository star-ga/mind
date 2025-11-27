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

//! Minimal CPU executor for TensorVal with materialized buffers (f32 only for v1).

use crate::eval::value::TensorVal;
use crate::exec::simd_chunks_mut;
use crate::types::DType;
use crate::types::ShapeDim;

#[derive(Debug)]
pub enum ExecError {
    Unsupported(String),
    Shape(String),
    Type(String),
    Math(String),
}

type R<T> = Result<T, ExecError>;

#[inline]
fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn shape_usize(shape: &[ShapeDim]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            ShapeDim::Known(n) => out.push(*n),
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(out)
}

fn ensure_f32(t: &TensorVal) -> R<()> {
    match t.dtype {
        DType::F32 => Ok(()),
        _ => Err(ExecError::Type(
            "only f32 tensors supported in cpu-exec".into(),
        )),
    }
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> R<Vec<usize>> {
    let mut out = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;
    while i >= 0 || j >= 0 {
        let da = if i >= 0 { a[i as usize] } else { 1 };
        let db = if j >= 0 { b[j as usize] } else { 1 };
        let dim = if da == db || da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(ExecError::Shape(format!(
                "cannot broadcast shapes {:?} and {:?}",
                a, b
            )));
        };
        out.push(dim);
        i -= 1;
        j -= 1;
    }
    out.reverse();
    Ok(out)
}

fn stride_for(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        stride[i] = acc;
        acc *= shape[i];
    }
    stride
}

fn index_broadcast(idx: &[usize], shape: &[usize], stride: &[usize]) -> usize {
    let offset_base = idx.len().saturating_sub(shape.len());
    let mut offset = 0usize;
    for i in 0..shape.len() {
        let dim = shape[i];
        let take = if dim == 1 { 0 } else { idx[offset_base + i] };
        offset += take * stride[i];
    }
    offset
}

fn elementwise_binop_f32(
    op: fn(f32, f32) -> f32,
    lhs: (&[f32], &[usize]),
    rhs: (&[f32], &[usize]),
) -> R<(Vec<f32>, Vec<usize>)> {
    let (ldata, lshape) = lhs;
    let (rdata, rshape) = rhs;
    let out_shape = broadcast_shapes(lshape, rshape)?;
    let out_numel = numel(&out_shape);
    let lstride = stride_for(lshape);
    let rstride = stride_for(rshape);
    let mut out = vec![0f32; out_numel];
    if out_shape.is_empty() {
        let li = index_broadcast(&[], lshape, &lstride);
        let ri = index_broadcast(&[], rshape, &rstride);
        out[0] = op(ldata[li], rdata[ri]);
        return Ok((out, out_shape));
    }
    let mut idx = vec![0usize; out_shape.len()];
    for linear in 0..out_numel {
        let mut t = linear;
        for ax in (0..out_shape.len()).rev() {
            let dim = out_shape[ax];
            idx[ax] = t % dim;
            t /= dim;
        }
        let li = index_broadcast(&idx, lshape, &lstride);
        let ri = index_broadcast(&idx, rshape, &rstride);
        out[linear] = op(ldata[li], rdata[ri]);
    }
    Ok((out, out_shape))
}

fn elementwise_unary_f32<F>(op: F, data: &[f32]) -> Vec<f32>
where
    F: Fn(f32) -> f32,
{
    let mut out = data.to_vec();
    for chunk in simd_chunks_mut(&mut out) {
        for v in chunk {
            *v = op(*v);
        }
    }
    out
}

fn reduce_sum(data: &[f32]) -> f32 {
    data.iter().copied().sum()
}

pub fn exec_add(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    ensure_f32(lhs)?;
    ensure_f32(rhs)?;
    let lshape = shape_usize(&lhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let rshape = shape_usize(&rhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let ldata = lhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let rdata = rhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
    let (out, shape) = elementwise_binop_f32(|a, b| a + b, (ldata, &lshape), (rdata, &rshape))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_sub(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    ensure_f32(lhs)?;
    ensure_f32(rhs)?;
    let lshape = shape_usize(&lhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let rshape = shape_usize(&rhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let ldata = lhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let rdata = rhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
    let (out, shape) = elementwise_binop_f32(|a, b| a - b, (ldata, &lshape), (rdata, &rshape))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_mul(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    ensure_f32(lhs)?;
    ensure_f32(rhs)?;
    let lshape = shape_usize(&lhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let rshape = shape_usize(&rhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let ldata = lhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let rdata = rhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
    let (out, shape) = elementwise_binop_f32(|a, b| a * b, (ldata, &lshape), (rdata, &rshape))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_div(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    ensure_f32(lhs)?;
    ensure_f32(rhs)?;
    let lshape = shape_usize(&lhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let rshape = shape_usize(&rhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let ldata = lhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let rdata = rhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
    if rdata.iter().any(|&v| v == 0.0) {
        return Err(ExecError::Math("division by zero".into()));
    }
    let (out, shape) = elementwise_binop_f32(|a, b| a / b, (ldata, &lshape), (rdata, &rshape))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_add_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    ensure_f32(t)?;
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let out = elementwise_unary_f32(|v| v + scalar, data);
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_sub_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    ensure_f32(t)?;
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let out = elementwise_unary_f32(|v| v - scalar, data);
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_scalar_sub(scalar: f32, t: &TensorVal) -> R<TensorVal> {
    ensure_f32(t)?;
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let out = elementwise_unary_f32(|v| scalar - v, data);
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_mul_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    ensure_f32(t)?;
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let out = elementwise_unary_f32(|v| v * scalar, data);
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_div_scalar(t: &TensorVal, scalar: f32, tensor_on_left: bool) -> R<TensorVal> {
    ensure_f32(t)?;
    if !tensor_on_left && data_has_zero(t)? {
        return Err(ExecError::Math("division by zero".into()));
    }
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let out = if tensor_on_left {
        elementwise_unary_f32(|v| v / scalar, data)
    } else {
        elementwise_unary_f32(|v| scalar / v, data)
    };
    Ok(TensorVal::from_materialized_f32(shape, out))
}

fn data_has_zero(t: &TensorVal) -> R<bool> {
    ensure_f32(t)?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    Ok(data.iter().any(|&v| v == 0.0))
}

pub fn exec_sum_all(t: &TensorVal) -> R<TensorVal> {
    ensure_f32(t)?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let sum = reduce_sum(data);
    Ok(TensorVal::from_materialized_f32(vec![], vec![sum]))
}

pub fn exec_mean_all(t: &TensorVal) -> R<TensorVal> {
    ensure_f32(t)?;
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    if shape.iter().any(|&d| d == 0) {
        return Err(ExecError::Math("mean over zero elements".into()));
    }
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let sum = reduce_sum(data);
    let count = numel(&shape) as f32;
    Ok(TensorVal::from_materialized_f32(vec![], vec![sum / count]))
}

pub fn relu_inplace(buf: &mut [f32]) {
    for chunk in simd_chunks_mut(buf) {
        for v in chunk {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }
}

pub fn exec_relu(t: &TensorVal) -> R<TensorVal> {
    ensure_f32(t)?;
    let shape = shape_usize(&t.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let data = t
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("tensor not materialized".into()))?;
    let mut out = data.to_vec();
    relu_inplace(&mut out);
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_matmul(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    ensure_f32(lhs)?;
    ensure_f32(rhs)?;
    let lshape = shape_usize(&lhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let rshape = shape_usize(&rhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    if lshape.len() != 2 || rshape.len() != 2 {
        return Err(ExecError::Shape("matmul expects 2D tensors".into()));
    }
    let (m, k1) = (lshape[0], lshape[1]);
    let (k2, n) = (rshape[0], rshape[1]);
    if k1 != k2 {
        return Err(ExecError::Shape("inner dims mismatch".into()));
    }
    let ldata = lhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let rdata = rhs
        .as_f32()
        .ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for k in 0..k1 {
                acc += ldata[i * k1 + k] * rdata[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(TensorVal::from_materialized_f32(vec![m, n], out))
}

pub fn exec_dot(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    ensure_f32(lhs)?;
    ensure_f32(rhs)?;
    let lshape = shape_usize(&lhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    let rshape = shape_usize(&rhs.shape).ok_or_else(|| ExecError::Shape("symbolic dims".into()))?;
    match (lshape.len(), rshape.len()) {
        (1, 1) => {
            if lshape[0] != rshape[0] {
                return Err(ExecError::Shape("dot expects equal length vectors".into()));
            }
            let ldata = lhs
                .as_f32()
                .ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
            let rdata = rhs
                .as_f32()
                .ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
            let sum = ldata.iter().zip(rdata.iter()).map(|(a, b)| a * b).sum();
            Ok(TensorVal::from_materialized_f32(vec![], vec![sum]))
        }
        (2, 2) => exec_matmul(lhs, rhs),
        _ => Err(ExecError::Shape("dot expects 1D or 2D tensors".into())),
    }
}
