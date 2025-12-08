// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shared tensor shape helpers for the MIND compiler.

use std::collections::BTreeSet;

use crate::linalg;
use crate::types::{ConvPadding, ShapeDim};

/// Error type returned by the shape helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    RankMismatch {
        expected: usize,
        found: usize,
    },
    AxisOutOfRange {
        axis: i32,
        rank: usize,
    },
    DuplicateAxis {
        axis: i32,
    },
    BroadcastIncompatible {
        lhs: Vec<ShapeDim>,
        rhs: Vec<ShapeDim>,
    },
    SizeMismatch {
        expected: usize,
        found: usize,
    },
    ElementCountMismatch {
        lhs: usize,
        rhs: usize,
    },
    InvalidPermutation {
        expected: usize,
        found: usize,
    },
    InvalidSliceBounds,
    InvalidStride,
    ConvChannelMismatch {
        input: ShapeDim,
        filter: ShapeDim,
    },
    ConvInputRank(usize),
    ConvFilterRank(usize),
    ConvKernelEmpty,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::RankMismatch { expected, found } => {
                write!(f, "expected rank {} but found {}", expected, found)
            }
            ShapeError::AxisOutOfRange { axis, rank } => {
                write!(f, "axis {axis} out of range for rank {rank}")
            }
            ShapeError::DuplicateAxis { axis } => {
                write!(f, "duplicate axis {axis}")
            }
            ShapeError::BroadcastIncompatible { lhs, rhs } => {
                write!(f, "cannot broadcast shapes {:?} and {:?}", lhs, rhs)
            }
            ShapeError::SizeMismatch { expected, found } => {
                write!(f, "expected {expected} dimensions but found {found}")
            }
            ShapeError::ElementCountMismatch { lhs, rhs } => {
                write!(f, "element count mismatch: {lhs} vs {rhs}")
            }
            ShapeError::InvalidPermutation { expected, found } => {
                write!(
                    f,
                    "expected permutation of length {expected} but found {found}"
                )
            }
            ShapeError::InvalidSliceBounds => {
                write!(f, "invalid slice bounds")
            }
            ShapeError::InvalidStride => write!(f, "stride must be non-zero"),
            ShapeError::ConvChannelMismatch { input, filter } => write!(
                f,
                "channel mismatch between input {input:?} and filter {filter:?}"
            ),
            ShapeError::ConvInputRank(rank) => {
                write!(f, "conv2d expects input rank 4 (NHWC) but found {rank}")
            }
            ShapeError::ConvFilterRank(rank) => {
                write!(f, "conv2d expects filter rank 4 (HWCF) but found {rank}")
            }
            ShapeError::ConvKernelEmpty => write!(f, "conv2d kernel dimensions must be positive"),
        }
    }
}

fn fresh_symbol(prefix: &str) -> &'static str {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    Box::leak(format!("{prefix}{id}").into_boxed_str())
}

fn dim_len(dim: &ShapeDim) -> Option<usize> {
    match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    }
}

fn normalize_axis(axis: i32, rank: usize) -> Result<usize, ShapeError> {
    let rank_i = rank as i32;
    let idx = if axis < 0 { rank_i + axis } else { axis };
    if idx < 0 || idx >= rank_i {
        Err(ShapeError::AxisOutOfRange { axis, rank })
    } else {
        Ok(idx as usize)
    }
}

fn normalize_axes_list(axes: &[i32], rank: usize) -> Result<Vec<usize>, ShapeError> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for &axis in axes {
        let idx = normalize_axis(axis, rank)?;
        if !seen.insert(idx) {
            return Err(ShapeError::DuplicateAxis { axis });
        }
        normalized.push(idx);
    }
    normalized.sort_unstable();
    Ok(normalized)
}

pub fn broadcast_shapes(lhs: &[ShapeDim], rhs: &[ShapeDim]) -> Result<Vec<ShapeDim>, ShapeError> {
    let mut out = Vec::new();
    let mut i = lhs.len() as isize - 1;
    let mut j = rhs.len() as isize - 1;

    while i >= 0 || j >= 0 {
        let da = if i >= 0 {
            lhs[i as usize].clone()
        } else {
            ShapeDim::Known(1)
        };
        let db = if j >= 0 {
            rhs[j as usize].clone()
        } else {
            ShapeDim::Known(1)
        };

        let dim = match (da, db) {
            (ShapeDim::Known(x), ShapeDim::Known(y)) => {
                if x == y {
                    ShapeDim::Known(x)
                } else if x == 1 {
                    ShapeDim::Known(y)
                } else if y == 1 {
                    ShapeDim::Known(x)
                } else {
                    return Err(ShapeError::BroadcastIncompatible {
                        lhs: lhs.to_vec(),
                        rhs: rhs.to_vec(),
                    });
                }
            }
            (ShapeDim::Sym(s1), ShapeDim::Sym(s2)) => {
                if s1 == s2 {
                    ShapeDim::Sym(s1)
                } else {
                    return Err(ShapeError::BroadcastIncompatible {
                        lhs: lhs.to_vec(),
                        rhs: rhs.to_vec(),
                    });
                }
            }
            (ShapeDim::Sym(sym), ShapeDim::Known(n)) | (ShapeDim::Known(n), ShapeDim::Sym(sym)) => {
                if n == 1 {
                    ShapeDim::Sym(sym)
                } else {
                    return Err(ShapeError::BroadcastIncompatible {
                        lhs: lhs.to_vec(),
                        rhs: rhs.to_vec(),
                    });
                }
            }
        };

        out.push(dim);
        i -= 1;
        j -= 1;
    }

    out.reverse();
    Ok(out)
}

pub fn reduce_shape(
    input: &[ShapeDim],
    axes: &[i32],
    keepdims: bool,
) -> Result<Vec<ShapeDim>, ShapeError> {
    let axes = if axes.is_empty() {
        (0..input.len() as i32).collect::<Vec<_>>()
    } else {
        axes.to_vec()
    };
    let normalized = normalize_axes_list(&axes, input.len())?;
    if keepdims {
        let mut out = input.to_vec();
        for &axis in &normalized {
            out[axis] = ShapeDim::Known(1);
        }
        Ok(out)
    } else {
        let axis_set: BTreeSet<usize> = normalized.into_iter().collect();
        let mut out = Vec::new();
        for (idx, dim) in input.iter().enumerate() {
            if !axis_set.contains(&idx) {
                out.push(dim.clone());
            }
        }
        Ok(out)
    }
}

fn known_product(shape: &[ShapeDim]) -> Option<usize> {
    let mut total = 1usize;
    for dim in shape {
        match dim {
            ShapeDim::Known(n) => {
                total = total.checked_mul(*n)?;
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(total)
}

pub fn reshape_shape(
    input: &[ShapeDim],
    new_dims: &[ShapeDim],
) -> Result<Vec<ShapeDim>, ShapeError> {
    if input.len() != new_dims.len() {
        return Err(ShapeError::SizeMismatch {
            expected: input.len(),
            found: new_dims.len(),
        });
    }
    if let (Some(old), Some(new)) = (known_product(input), known_product(new_dims)) {
        if old != new {
            return Err(ShapeError::ElementCountMismatch { lhs: old, rhs: new });
        }
    }
    Ok(new_dims.to_vec())
}

pub fn transpose_shape(input: &[ShapeDim], perm: &[usize]) -> Result<Vec<ShapeDim>, ShapeError> {
    if perm.len() != input.len() {
        return Err(ShapeError::InvalidPermutation {
            expected: input.len(),
            found: perm.len(),
        });
    }
    if let Some(max) = perm.iter().copied().max() {
        if max >= input.len() {
            return Err(ShapeError::AxisOutOfRange {
                axis: max as i32,
                rank: input.len(),
            });
        }
    }
    let mut seen = BTreeSet::new();
    for &axis in perm {
        if !seen.insert(axis) {
            return Err(ShapeError::DuplicateAxis { axis: axis as i32 });
        }
    }
    Ok(linalg::permute_shape(input, perm))
}

pub fn expand_dims_shape(input: &[ShapeDim], axis: i32) -> Result<Vec<ShapeDim>, ShapeError> {
    let rank = input.len();
    let extended = rank + 1;
    let idx = if axis < 0 {
        (extended as i32) + axis
    } else {
        axis
    };
    if idx < 0 || idx >= extended as i32 {
        return Err(ShapeError::AxisOutOfRange { axis, rank });
    }
    let mut shape = input.to_vec();
    shape.insert(idx as usize, ShapeDim::Known(1));
    Ok(shape)
}

pub fn squeeze_shape(input: &[ShapeDim], axes: &[i32]) -> Result<Vec<ShapeDim>, ShapeError> {
    let axes_to_remove = if axes.is_empty() {
        input
            .iter()
            .enumerate()
            .filter_map(|(idx, dim)| matches!(dim, ShapeDim::Known(1)).then_some(idx))
            .collect::<Vec<_>>()
    } else {
        normalize_axes_list(axes, input.len())?
    };
    let mut axis_set = BTreeSet::new();
    for &axis in &axes_to_remove {
        if !matches!(input.get(axis), Some(ShapeDim::Known(1))) {
            return Err(ShapeError::SizeMismatch {
                expected: 1,
                found: axis,
            });
        }
        axis_set.insert(axis);
    }
    let mut out = Vec::new();
    for (idx, dim) in input.iter().enumerate() {
        if !axis_set.contains(&idx) {
            out.push(dim.clone());
        }
    }
    Ok(out)
}

pub fn index_shape(input: &[ShapeDim], axis: i32) -> Result<Vec<ShapeDim>, ShapeError> {
    if input.is_empty() {
        return Err(ShapeError::RankMismatch {
            expected: 1,
            found: 0,
        });
    }
    let axis = normalize_axis(axis, input.len())?;
    let mut shape = input.to_vec();
    shape.remove(axis);
    Ok(shape)
}

fn slice_len(start: i32, end: i32) -> Option<usize> {
    if start < 0 || end < start {
        None
    } else {
        Some((end - start) as usize)
    }
}

fn slice_len_with_step(len: Option<usize>, start: i32, end: i32, step: i32) -> Option<usize> {
    if step == 0 {
        return None;
    }
    let len = len?;
    let len_i = len as i64;
    let step_i = step as i64;

    let mut start_i = start as i64;
    let mut end_i = end as i64;

    if step_i > 0 {
        if start_i < 0 {
            start_i += len_i;
        }
        if start_i < 0 {
            start_i = 0;
        }
        if start_i > len_i {
            start_i = len_i;
        }

        if end_i < 0 {
            end_i += len_i;
        }
        if end_i < 0 {
            end_i = 0;
        }
        if end_i > len_i {
            end_i = len_i;
        }

        if start_i >= end_i {
            Some(0)
        } else {
            let diff = end_i - start_i;
            Some(((diff + step_i.abs() - 1) / step_i.abs()) as usize)
        }
    } else {
        if len == 0 {
            return Some(0);
        }

        if start_i < 0 {
            start_i += len_i;
        }
        if start_i < -1 {
            start_i = -1;
        }
        if start_i >= len_i {
            start_i = len_i - 1;
        }

        if end_i < 0 {
            end_i += len_i;
        }
        if end_i < -1 {
            end_i = -1;
        }
        if end_i >= len_i {
            end_i = len_i - 1;
        }

        if start_i <= end_i {
            Some(0)
        } else {
            let diff = start_i - end_i;
            Some(((diff + (-step_i) - 1) / (-step_i)) as usize)
        }
    }
}

pub fn slice_shape(
    input: &[ShapeDim],
    axis: i32,
    start: i32,
    end: i32,
) -> Result<Vec<ShapeDim>, ShapeError> {
    if start < 0 || end < start {
        return Err(ShapeError::InvalidSliceBounds);
    }
    let axis = normalize_axis(axis, input.len())?;
    let new_dim = match (dim_len(&input[axis]), slice_len(start, end)) {
        (Some(_), Some(len)) => ShapeDim::Known(len),
        _ => ShapeDim::Sym(fresh_symbol("_slice")),
    };
    let mut shape = input.to_vec();
    shape[axis] = new_dim;
    Ok(shape)
}

pub fn slice_stride_shape(
    input: &[ShapeDim],
    axis: i32,
    start: i32,
    end: i32,
    step: i32,
) -> Result<Vec<ShapeDim>, ShapeError> {
    if step == 0 {
        return Err(ShapeError::InvalidStride);
    }
    let axis = normalize_axis(axis, input.len())?;
    let dim = input[axis].clone();
    let new_dim = if let Some(len) = dim_len(&dim) {
        let Some(result_len) = slice_len_with_step(Some(len), start, end, step) else {
            return Err(ShapeError::InvalidSliceBounds);
        };
        ShapeDim::Known(result_len)
    } else if (step > 0 && start >= end) || (step < 0 && start <= end) {
        ShapeDim::Known(0)
    } else {
        ShapeDim::Sym(fresh_symbol("_slice_stride"))
    };
    let mut shape = input.to_vec();
    shape[axis] = new_dim;
    Ok(shape)
}

pub fn gather_shape(
    input: &[ShapeDim],
    axis: i32,
    idx_shape: &[ShapeDim],
) -> Result<Vec<ShapeDim>, ShapeError> {
    let axis = normalize_axis(axis, input.len())?;
    let mut shape = Vec::new();
    shape.extend_from_slice(&input[..axis]);
    shape.extend(idx_shape.iter().cloned());
    if axis < input.len() {
        shape.extend_from_slice(&input[axis + 1..]);
    }
    Ok(shape)
}

pub fn conv2d_shape(
    input: &[ShapeDim],
    filter: &[ShapeDim],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> Result<(ShapeDim, ShapeDim), ShapeError> {
    if input.len() != 4 {
        return Err(ShapeError::ConvInputRank(input.len()));
    }
    if filter.len() != 4 {
        return Err(ShapeError::ConvFilterRank(filter.len()));
    }
    let in_channels = &input[3];
    let kernel_channels = &filter[2];
    match (in_channels, kernel_channels) {
        (ShapeDim::Known(a), ShapeDim::Known(b)) if a != b => {
            return Err(ShapeError::ConvChannelMismatch {
                input: in_channels.clone(),
                filter: kernel_channels.clone(),
            })
        }
        (ShapeDim::Sym(a), ShapeDim::Sym(b)) if a != b => {
            return Err(ShapeError::ConvChannelMismatch {
                input: in_channels.clone(),
                filter: kernel_channels.clone(),
            })
        }
        (ShapeDim::Known(0), _) | (_, ShapeDim::Known(0)) => {
            return Err(ShapeError::ConvKernelEmpty)
        }
        _ => {}
    }

    let compute_hw = |input_dim: &ShapeDim, kernel_dim: &ShapeDim, stride: usize| {
        match linalg::conv_output_dim_valid(dim_len(input_dim), dim_len(kernel_dim), stride) {
            Ok(Some(v)) => Ok(ShapeDim::Known(v)),
            Ok(None) => Ok(ShapeDim::Sym(fresh_symbol("_conv"))),
            Err(_) => Err(ShapeError::ConvKernelEmpty),
        }
    };

    let mut out_h = compute_hw(&input[1], &filter[0], stride_h)?;
    let mut out_w = compute_hw(&input[2], &filter[1], stride_w)?;

    if let ConvPadding::Same = padding {
        out_h = linalg::conv_output_dim_same(dim_len(&input[1]), stride_h)
            .map(|opt| {
                opt.map(ShapeDim::Known)
                    .unwrap_or_else(|| ShapeDim::Sym(fresh_symbol("_conv_h")))
            })
            .map_err(|_| ShapeError::ConvKernelEmpty)?;
        out_w = linalg::conv_output_dim_same(dim_len(&input[2]), stride_w)
            .map(|opt| {
                opt.map(ShapeDim::Known)
                    .unwrap_or_else(|| ShapeDim::Sym(fresh_symbol("_conv_w")))
            })
            .map_err(|_| ShapeError::ConvKernelEmpty)?;
    }

    Ok((out_h, out_w))
}
