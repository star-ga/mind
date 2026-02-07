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

//! Open-core CPU reference interpreter.
//!
//! Naive, unoptimized implementations for the public MIND compiler.
//! These provide correct results for learning, prototyping, and small workloads.
//!
//! For production performance (SIMD, tiled matmul, GPU backends), see:
//! https://mindlang.dev/enterprise

use crate::eval::value::TensorVal;
use crate::types::ShapeDim;

#[derive(Debug)]
pub enum ExecError {
    Unsupported(String),
    Shape(String),
    #[allow(dead_code)]
    Type(String),
    #[allow(dead_code)]
    Math(String),
}

type R<T> = Result<T, ExecError>;

fn shape_usize(shape: &[ShapeDim]) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => Some(*n),
            _ => None,
        })
        .collect()
}

fn get_f32<'a>(t: &'a TensorVal) -> Option<&'a [f32]> {
    t.as_f32()
}

// ---------------------------------------------------------------------------
// Elementwise binary ops (tensor x tensor)
// ---------------------------------------------------------------------------

fn elementwise_bin(lhs: &TensorVal, rhs: &TensorVal, f: fn(f32, f32) -> f32) -> R<TensorVal> {
    let a = get_f32(lhs).ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let b = get_f32(rhs).ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;
    if a.len() != b.len() {
        return Err(ExecError::Shape(format!(
            "elementwise length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(f(a[i], b[i]));
    }
    let shape = lhs
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_add(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    elementwise_bin(lhs, rhs, |a, b| a + b)
}

pub fn exec_sub(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    elementwise_bin(lhs, rhs, |a, b| a - b)
}

pub fn exec_mul(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    elementwise_bin(lhs, rhs, |a, b| a * b)
}

pub fn exec_div(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    elementwise_bin(lhs, rhs, |a, b| a / b)
}

// ---------------------------------------------------------------------------
// Scalar broadcast ops
// ---------------------------------------------------------------------------

fn scalar_op(t: &TensorVal, s: f32, f: fn(f32, f32) -> f32) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let mut out = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        out.push(f(data[i], s));
    }
    let shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

pub fn exec_add_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    scalar_op(t, scalar, |a, b| a + b)
}

pub fn exec_sub_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    scalar_op(t, scalar, |a, b| a - b)
}

pub fn exec_scalar_sub(scalar: f32, t: &TensorVal) -> R<TensorVal> {
    scalar_op(t, scalar, |a, b| b - a)
}

pub fn exec_mul_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    scalar_op(t, scalar, |a, b| a * b)
}

pub fn exec_div_scalar(t: &TensorVal, scalar: f32, tensor_on_left: bool) -> R<TensorVal> {
    if tensor_on_left {
        scalar_op(t, scalar, |a, b| a / b)
    } else {
        scalar_op(t, scalar, |a, b| b / a)
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

pub fn exec_sum_all(t: &TensorVal) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let mut total: f32 = 0.0;
    for i in 0..data.len() {
        total += data[i];
    }
    Ok(TensorVal::from_materialized_f32(vec![], vec![total]))
}

pub fn exec_mean_all(t: &TensorVal) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    if data.is_empty() {
        return Err(ExecError::Shape("mean of empty tensor".into()));
    }
    let mut total: f32 = 0.0;
    for i in 0..data.len() {
        total += data[i];
    }
    let mean = total / data.len() as f32;
    Ok(TensorVal::from_materialized_f32(vec![], vec![mean]))
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

pub fn relu_inplace(buf: &mut [f32]) {
    for i in 0..buf.len() {
        if buf[i] < 0.0 {
            buf[i] = 0.0;
        }
    }
}

pub fn exec_relu(t: &TensorVal) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let mut out = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        out.push(if data[i] > 0.0 { data[i] } else { 0.0 });
    }
    let shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    Ok(TensorVal::from_materialized_f32(shape, out))
}

// ---------------------------------------------------------------------------
// Matrix multiply — naive triple loop, O(M*N*K)
// ---------------------------------------------------------------------------

pub fn exec_matmul(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    let a = get_f32(lhs).ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let b = get_f32(rhs).ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;

    let lshape = shape_usize(&lhs.shape)
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    let rshape = shape_usize(&rhs.shape)
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;

    if lshape.len() != 2 || rshape.len() != 2 {
        return Err(ExecError::Shape("matmul requires 2D tensors".into()));
    }

    let (m, k1) = (lshape[0], lshape[1]);
    let (k2, n) = (rshape[0], rshape[1]);

    if k1 != k2 {
        return Err(ExecError::Shape(format!(
            "matmul inner dimension mismatch: {} vs {}",
            k1, k2
        )));
    }

    let k = k1;
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    Ok(TensorVal::from_materialized_f32(vec![m, n], out))
}

// ---------------------------------------------------------------------------
// Shape ops — metadata-only (buffer unchanged)
// ---------------------------------------------------------------------------

/// Reshape: same data, new shape metadata.
pub fn exec_reshape(t: &TensorVal, new_shape: Vec<usize>) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let new_numel: usize = new_shape.iter().product();
    if data.len() != new_numel {
        return Err(ExecError::Shape(format!(
            "reshape: element count mismatch {} vs {}",
            data.len(),
            new_numel
        )));
    }
    Ok(TensorVal::from_materialized_f32(new_shape, data.to_vec()))
}

/// Expand dims: insert a size-1 dimension.
pub fn exec_expand_dims(t: &TensorVal, axis: usize) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let mut shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    if axis > shape.len() {
        return Err(ExecError::Shape(format!(
            "expand_dims: axis {} out of bounds for rank {}",
            axis,
            shape.len()
        )));
    }
    shape.insert(axis, 1);
    Ok(TensorVal::from_materialized_f32(shape, data.to_vec()))
}

/// Squeeze: remove size-1 dimensions.
pub fn exec_squeeze(t: &TensorVal, axes: &[usize]) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let old_shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    let mut shape = Vec::new();
    for (i, &dim) in old_shape.iter().enumerate() {
        if axes.is_empty() {
            // Squeeze all size-1 dims
            if dim != 1 {
                shape.push(dim);
            }
        } else if !axes.contains(&i) {
            shape.push(dim);
        }
    }
    Ok(TensorVal::from_materialized_f32(shape, data.to_vec()))
}

// ---------------------------------------------------------------------------
// Shape ops — data-moving
// ---------------------------------------------------------------------------

/// Transpose: permute buffer data according to axes.
pub fn exec_transpose(t: &TensorVal, perm: &[usize]) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let old_shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    let rank = old_shape.len();
    if perm.len() != rank {
        return Err(ExecError::Shape(format!(
            "transpose: perm length {} != rank {}",
            perm.len(),
            rank
        )));
    }

    // Compute new shape
    let new_shape: Vec<usize> = perm.iter().map(|&p| old_shape[p]).collect();
    let numel: usize = old_shape.iter().product();
    let mut out = vec![0.0f32; numel];

    // Compute strides for old shape
    let mut old_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }

    // Compute strides for new shape
    let mut new_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    // Permute each element
    for flat_idx in 0..numel {
        // Convert flat index to multi-dimensional index in old shape
        let mut remaining = flat_idx;
        let mut old_idx = vec![0usize; rank];
        for d in 0..rank {
            old_idx[d] = remaining / old_strides[d];
            remaining %= old_strides[d];
        }

        // Permuted index in new shape
        let mut new_flat = 0;
        for d in 0..rank {
            new_flat += old_idx[perm[d]] * new_strides[d];
        }

        out[new_flat] = data[flat_idx];
    }

    Ok(TensorVal::from_materialized_f32(new_shape, out))
}

/// Index: extract a subtensor by removing one axis at a fixed index.
pub fn exec_index(t: &TensorVal, axis: usize, i: usize) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    if axis >= shape.len() {
        return Err(ExecError::Shape("index: axis out of bounds".into()));
    }
    if i >= shape[axis] {
        return Err(ExecError::Shape("index: index out of bounds".into()));
    }

    // Compute stride for the axis
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let axis_size = shape[axis];

    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        let base = o * axis_size * inner + i * inner;
        out.extend_from_slice(&data[base..base + inner]);
    }

    let mut new_shape = shape;
    new_shape.remove(axis);
    Ok(TensorVal::from_materialized_f32(new_shape, out))
}

/// Slice: extract a range along one axis.
pub fn exec_slice(t: &TensorVal, axis: usize, start: usize, end: usize) -> R<TensorVal> {
    let data = get_f32(t).ok_or_else(|| ExecError::Unsupported("not materialized".into()))?;
    let shape = t
        .shape_as_usize()
        .ok_or_else(|| ExecError::Shape("dynamic shape".into()))?;
    if axis >= shape.len() {
        return Err(ExecError::Shape("slice: axis out of bounds".into()));
    }
    let dim = shape[axis];
    let end = end.min(dim);
    if start > end {
        return Err(ExecError::Shape("slice: start > end".into()));
    }

    let slice_len = end - start;
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let axis_size = shape[axis];

    let mut out = Vec::with_capacity(outer * slice_len * inner);
    for o in 0..outer {
        for s in start..end {
            let base = o * axis_size * inner + s * inner;
            out.extend_from_slice(&data[base..base + inner]);
        }
    }

    let mut new_shape = shape;
    new_shape[axis] = slice_len;
    Ok(TensorVal::from_materialized_f32(new_shape, out))
}

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

pub fn exec_dot(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    let a = get_f32(lhs).ok_or_else(|| ExecError::Unsupported("lhs not materialized".into()))?;
    let b = get_f32(rhs).ok_or_else(|| ExecError::Unsupported("rhs not materialized".into()))?;

    if a.len() != b.len() {
        return Err(ExecError::Shape(format!(
            "dot product length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    Ok(TensorVal::from_materialized_f32(vec![], vec![sum]))
}
