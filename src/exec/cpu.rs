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

//! CPU execution backend.
//!
//! Provides f32 tensor compute for element-wise, reduction, and matmul operations.
//! Operates on materialized buffers in `TensorVal`.

use crate::eval::value::{Buffer, TensorVal};
use crate::types::ShapeDim;

#[derive(Debug)]
pub enum ExecError {
    Unsupported(String),
    Shape(String),
    Type(String),
    Math(String),
}

type R<T> = Result<T, ExecError>;

/// Get the flat f32 buffer from a tensor, or return an error.
fn get_f32(t: &TensorVal) -> R<&[f32]> {
    t.as_f32()
        .ok_or_else(|| ExecError::Type("expected materialized F32 tensor".into()))
}


// ---------------------------------------------------------------------------
// Element-wise binary ops
// ---------------------------------------------------------------------------

fn binary_op(
    lhs: &TensorVal,
    rhs: &TensorVal,
    f: fn(f32, f32) -> f32,
) -> R<TensorVal> {
    let a = get_f32(lhs)?;
    let b = get_f32(rhs)?;
    if a.len() != b.len() {
        return Err(ExecError::Shape(format!(
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let out: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    Ok(TensorVal {
        dtype: lhs.dtype.clone(),
        shape: lhs.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

pub fn exec_add(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    binary_op(lhs, rhs, |a, b| a + b)
}

pub fn exec_sub(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    binary_op(lhs, rhs, |a, b| a - b)
}

pub fn exec_mul(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    binary_op(lhs, rhs, |a, b| a * b)
}

pub fn exec_div(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    binary_op(lhs, rhs, |a, b| {
        if b == 0.0 {
            f32::NAN
        } else {
            a / b
        }
    })
}

// ---------------------------------------------------------------------------
// Scalar broadcast ops
// ---------------------------------------------------------------------------

pub fn exec_add_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    let a = get_f32(t)?;
    let out: Vec<f32> = a.iter().map(|&x| x + scalar).collect();
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

pub fn exec_sub_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    let a = get_f32(t)?;
    let out: Vec<f32> = a.iter().map(|&x| x - scalar).collect();
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

pub fn exec_scalar_sub(scalar: f32, t: &TensorVal) -> R<TensorVal> {
    let a = get_f32(t)?;
    let out: Vec<f32> = a.iter().map(|&x| scalar - x).collect();
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

pub fn exec_mul_scalar(t: &TensorVal, scalar: f32) -> R<TensorVal> {
    let a = get_f32(t)?;
    let out: Vec<f32> = a.iter().map(|&x| x * scalar).collect();
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

pub fn exec_div_scalar(t: &TensorVal, scalar: f32, tensor_on_left: bool) -> R<TensorVal> {
    let a = get_f32(t)?;
    let out: Vec<f32> = if tensor_on_left {
        a.iter().map(|&x| if scalar == 0.0 { f32::NAN } else { x / scalar }).collect()
    } else {
        a.iter().map(|&x| if x == 0.0 { f32::NAN } else { scalar / x }).collect()
    };
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

pub fn exec_sum_all(t: &TensorVal) -> R<TensorVal> {
    let a = get_f32(t)?;
    let sum: f32 = a.iter().copied().sum();
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: vec![],
        fill: None,
        buf: Some(Buffer::F32(vec![sum])),
    })
}

pub fn exec_mean_all(t: &TensorVal) -> R<TensorVal> {
    let a = get_f32(t)?;
    let n = a.len() as f32;
    let mean = if n > 0.0 {
        a.iter().copied().sum::<f32>() / n
    } else {
        0.0
    };
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: vec![],
        fill: None,
        buf: Some(Buffer::F32(vec![mean])),
    })
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

pub fn relu_inplace(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

pub fn exec_relu(t: &TensorVal) -> R<TensorVal> {
    let a = get_f32(t)?;
    let out: Vec<f32> = a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
    Ok(TensorVal {
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

// ---------------------------------------------------------------------------
// MatMul: (M,K) x (K,N) -> (M,N)
// ---------------------------------------------------------------------------

pub fn exec_matmul(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    let a = get_f32(lhs)?;
    let b = get_f32(rhs)?;

    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err(ExecError::Shape("matmul requires 2D tensors".into()));
    }

    let m = match lhs.shape[0] {
        ShapeDim::Known(n) => n,
        _ => return Err(ExecError::Shape("symbolic dim in matmul".into())),
    };
    let k = match lhs.shape[1] {
        ShapeDim::Known(n) => n,
        _ => return Err(ExecError::Shape("symbolic dim in matmul".into())),
    };
    let k2 = match rhs.shape[0] {
        ShapeDim::Known(n) => n,
        _ => return Err(ExecError::Shape("symbolic dim in matmul".into())),
    };
    let n = match rhs.shape[1] {
        ShapeDim::Known(nn) => nn,
        _ => return Err(ExecError::Shape("symbolic dim in matmul".into())),
    };

    if k != k2 {
        return Err(ExecError::Shape(format!(
            "matmul inner dims mismatch: {} vs {}",
            k, k2
        )));
    }

    let mut out = vec![0.0f32; m * n];
    // Standard ikj loop order for better cache locality on row-major B
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            for j in 0..n {
                out[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }

    Ok(TensorVal {
        dtype: lhs.dtype.clone(),
        shape: vec![ShapeDim::Known(m), ShapeDim::Known(n)],
        fill: None,
        buf: Some(Buffer::F32(out)),
    })
}

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

pub fn exec_dot(lhs: &TensorVal, rhs: &TensorVal) -> R<TensorVal> {
    let a = get_f32(lhs)?;
    let b = get_f32(rhs)?;
    if a.len() != b.len() {
        return Err(ExecError::Shape(format!(
            "dot length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    Ok(TensorVal {
        dtype: lhs.dtype.clone(),
        shape: vec![],
        fill: None,
        buf: Some(Buffer::F32(vec![dot])),
    })
}
