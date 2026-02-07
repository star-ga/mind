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
