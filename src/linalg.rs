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

use crate::types::ShapeDim;

#[derive(Debug, Clone)]
pub struct MatMulShapeInfo {
    pub a_shape: Vec<ShapeDim>,
    pub b_shape: Vec<ShapeDim>,
    pub broadcast_shape: Vec<ShapeDim>,
    pub result_shape: Vec<ShapeDim>,
    pub a_was_vec: bool,
    pub b_was_vec: bool,
    pub m_dim: ShapeDim,
    pub n_dim: ShapeDim,
    pub k_dim: ShapeDim,
}

fn format_dim(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

fn dims_equal(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(sa), ShapeDim::Sym(sb)) => sa == sb,
        _ => false,
    }
}

fn merge_dims(a: ShapeDim, b: ShapeDim) -> Result<ShapeDim, String> {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => {
            if x == y {
                Ok(ShapeDim::Known(x))
            } else if x == 1 {
                Ok(ShapeDim::Known(y))
            } else if y == 1 {
                Ok(ShapeDim::Known(x))
            } else {
                Err(format!("cannot broadcast dimensions {x} and {y}"))
            }
        }
        (ShapeDim::Sym(sa), ShapeDim::Sym(sb)) => {
            if sa == sb {
                Ok(ShapeDim::Sym(sa))
            } else {
                Err(format!(
                    "cannot broadcast symbolic dimensions {sa} and {sb}"
                ))
            }
        }
        (ShapeDim::Sym(sym), ShapeDim::Known(1)) | (ShapeDim::Known(1), ShapeDim::Sym(sym)) => {
            Ok(ShapeDim::Sym(sym))
        }
        (ShapeDim::Sym(sym), ShapeDim::Known(other))
        | (ShapeDim::Known(other), ShapeDim::Sym(sym)) => {
            if other == 1 {
                Ok(ShapeDim::Sym(sym))
            } else {
                Err(format!(
                    "cannot broadcast dimension {} with symbolic {}",
                    other, sym
                ))
            }
        }
    }
}

pub fn broadcast_leading(lhs: &[ShapeDim], rhs: &[ShapeDim]) -> Result<Vec<ShapeDim>, String> {
    let mut result = Vec::new();
    let mut i = lhs.len() as isize - 1;
    let mut j = rhs.len() as isize - 1;
    while i >= 0 || j >= 0 {
        let ld = if i >= 0 {
            lhs[i as usize].clone()
        } else {
            ShapeDim::Known(1)
        };
        let rd = if j >= 0 {
            rhs[j as usize].clone()
        } else {
            ShapeDim::Known(1)
        };
        let dim = merge_dims(ld, rd)?;
        result.push(dim);
        i -= 1;
        j -= 1;
    }
    result.reverse();
    Ok(result)
}

pub fn normalize_axis(axis: i32, rank: usize) -> Result<usize, String> {
    let rank_i32 = rank as i32;
    let idx = if axis < 0 { rank_i32 + axis } else { axis };
    if idx < 0 || idx >= rank_i32 {
        Err(format!("axis {axis} out of range for rank {rank}"))
    } else {
        Ok(idx as usize)
    }
}

pub fn normalize_permutation(axes: &[i32], rank: usize) -> Result<Vec<usize>, String> {
    if axes.len() != rank {
        return Err(format!(
            "permutation length {} does not match rank {}",
            axes.len(),
            rank
        ));
    }
    let mut seen = vec![false; rank];
    let mut perm = Vec::with_capacity(rank);
    for &axis in axes {
        let idx = normalize_axis(axis, rank)?;
        if seen[idx] {
            return Err(format!("duplicate axis {axis} in permutation"));
        }
        seen[idx] = true;
        perm.push(idx);
    }
    Ok(perm)
}

pub fn default_transpose(rank: usize) -> Vec<usize> {
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.reverse();
    axes
}

pub fn permute_shape(shape: &[ShapeDim], perm: &[usize]) -> Vec<ShapeDim> {
    perm.iter().map(|&idx| shape[idx].clone()).collect()
}

pub fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (idx, &axis) in perm.iter().enumerate() {
        inv[axis] = idx;
    }
    inv
}

pub fn known_dim_value(dim: &ShapeDim) -> Option<usize> {
    match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    }
}

pub fn conv_output_dim_valid(
    input: Option<usize>,
    kernel: Option<usize>,
    stride: usize,
) -> Result<Option<usize>, String> {
    if stride == 0 {
        return Err("stride must be positive".to_string());
    }
    if let Some(k) = kernel {
        if k == 0 {
            return Err("kernel size must be positive".to_string());
        }
    }
    match (input, kernel) {
        (Some(h), Some(k)) => {
            if h < k {
                return Err(format!("kernel size {k} exceeds input size {h}"));
            }
            let out = (h - k) / stride + 1;
            Ok(Some(out))
        }
        _ => Ok(None),
    }
}

pub fn conv_output_dim_same(input: Option<usize>, stride: usize) -> Result<Option<usize>, String> {
    if stride == 0 {
        return Err("stride must be positive".to_string());
    }
    match input {
        Some(h) => {
            if h == 0 {
                return Err("input spatial dimension must be positive".to_string());
            }
            let out = h.div_ceil(stride);
            Ok(Some(out))
        }
        None => Ok(None),
    }
}

pub fn compute_matmul_shape_info(
    a_shape: &[ShapeDim],
    b_shape: &[ShapeDim],
) -> Result<MatMulShapeInfo, String> {
    if a_shape.is_empty() || b_shape.is_empty() {
        // Allow rank-1 tensors but not scalars
        return Err("`tensor.matmul` requires tensors with rank >= 1".to_string());
    }

    let mut a_was_vec = false;
    let mut b_was_vec = false;

    let a_adj = if a_shape.len() == 1 {
        a_was_vec = true;
        vec![ShapeDim::Known(1), a_shape[0].clone()]
    } else {
        a_shape.to_vec()
    };

    let b_adj = if b_shape.len() == 1 {
        b_was_vec = true;
        let mut dims = b_shape.to_vec();
        dims.push(ShapeDim::Known(1));
        dims
    } else {
        b_shape.to_vec()
    };

    if a_adj.len() < 2 || b_adj.len() < 2 {
        return Err("`tensor.matmul` requires tensors with rank >= 1".to_string());
    }

    let a_leading = &a_adj[..a_adj.len() - 2];
    let b_leading = &b_adj[..b_adj.len() - 2];
    let broadcast_shape = broadcast_leading(a_leading, b_leading)?;
    let m_dim = a_adj[a_adj.len() - 2].clone();
    let k_left = a_adj[a_adj.len() - 1].clone();
    let k_right = b_adj[b_adj.len() - 2].clone();
    let n_dim = b_adj[b_adj.len() - 1].clone();

    if !dims_equal(&k_left, &k_right) {
        return Err(format!(
            "contraction dimension mismatch: {} vs {}",
            format_dim(&k_left),
            format_dim(&k_right)
        ));
    }

    let mut a_shape_broadcast = broadcast_shape.clone();
    a_shape_broadcast.push(m_dim.clone());
    a_shape_broadcast.push(k_left.clone());

    let mut b_shape_broadcast = broadcast_shape.clone();
    b_shape_broadcast.push(k_right.clone());
    b_shape_broadcast.push(n_dim.clone());

    let mut result_shape = broadcast_shape.clone();
    if !a_was_vec {
        result_shape.push(m_dim.clone());
    }
    if !b_was_vec {
        result_shape.push(n_dim.clone());
    }

    Ok(MatMulShapeInfo {
        a_shape: a_shape_broadcast,
        b_shape: b_shape_broadcast,
        broadcast_shape,
        result_shape,
        a_was_vec,
        b_was_vec,
        m_dim,
        n_dim,
        k_dim: k_left,
    })
}
