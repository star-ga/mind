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

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use crate::ast;
#[cfg(feature = "cpu-buffers")]
use crate::eval::conv2d_grad;
use crate::eval::TensorVal;
use crate::linalg;
use crate::linalg::MatMulShapeInfo;

#[derive(Clone)]
pub struct TensorEnvEntry {
    pub value: TensorVal,
    pub expr: Option<ast::Node>,
}
use crate::types::ConvPadding;
use crate::types::DType;
use crate::types::ShapeDim;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

#[derive(Debug, Clone)]
enum Op {
    LeafVar,
    ConstInt,
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    ReduceSum {
        x: NodeId,
        axes: Vec<usize>,
        keepdims: bool,
        reduced_elems: Option<usize>,
    },
    ReduceMean {
        x: NodeId,
        axes: Vec<usize>,
        keepdims: bool,
        reduced_elems: Option<usize>,
    },
    Reshape {
        x: NodeId,
    },
    ExpandDims {
        x: NodeId,
        axis: usize,
    },
    Squeeze {
        x: NodeId,
        axes: Vec<usize>,
    },
    Transpose {
        x: NodeId,
        axes: Vec<usize>,
    },
    Index {
        x: NodeId,
        axis: usize,
        i: i32,
    },
    Slice {
        x: NodeId,
        axis: usize,
        start: i32,
        end: i32,
    },
    SliceStride {
        x: NodeId,
        axis: usize,
        start: i32,
        end: i32,
        step: i32,
        in_shape: Vec<ShapeDim>,
    },
    Gather {
        x: NodeId,
        idx: NodeId,
        axis: usize,
        in_shape: Vec<ShapeDim>,
    },
    Dot {
        a: NodeId,
        b: NodeId,
        info: MatMulShapeInfo,
    },
    MatMul {
        a: NodeId,
        b: NodeId,
        info: MatMulShapeInfo,
    },
    Relu {
        x: NodeId,
    },
    #[allow(dead_code)]
    Conv2d {
        x: NodeId,
        w: NodeId,
        stride_h: usize,
        stride_w: usize,
        padding: ConvPadding,
    },
}

#[derive(Debug, Clone)]
struct NodeInfo {
    op: Op,
    dtype: DType,
    shape: Vec<ShapeDim>,
    fill: Option<f64>,
}

pub struct Tape {
    nodes: Vec<NodeInfo>,
}

impl Tape {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push(&mut self, info: NodeInfo) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(info);
        id
    }

    pub(crate) fn node_shape(&self, id: NodeId) -> &[ShapeDim] {
        &self.nodes[id.0].shape
    }
}

fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
    crate::eval::broadcast_shapes(a, b)
}

fn normalize_axis(axis: i32, rank: usize) -> Result<usize, String> {
    let rank_i32 = rank as i32;
    let idx = if axis < 0 { rank_i32 + axis } else { axis };
    if idx < 0 || idx >= rank_i32 {
        Err(format!("axis {axis} out of range (rank {rank})"))
    } else {
        Ok(idx as usize)
    }
}

fn normalize_axes_list(axes: &[i32], rank: usize) -> Result<Vec<usize>, String> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut normalized = Vec::new();
    for &axis in axes {
        let idx = normalize_axis(axis, rank)?;
        if !seen.insert(idx) {
            return Err(format!("duplicate axis {axis}"));
        }
        normalized.push(idx);
    }
    normalized.sort_unstable();
    Ok(normalized)
}

fn normalize_reduce_axes(axes: &[i32], rank: usize) -> Result<Vec<usize>, String> {
    if axes.is_empty() {
        return Ok((0..rank).collect());
    }
    normalize_axes_list(axes, rank)
}

fn reduce_shape(shape: &[ShapeDim], axes: &[usize], keepdims: bool) -> Vec<ShapeDim> {
    if keepdims {
        let mut out = shape.to_vec();
        for &axis in axes {
            if axis < out.len() {
                out[axis] = ShapeDim::Known(1);
            }
        }
        out
    } else {
        let axis_set: BTreeSet<usize> = axes.iter().cloned().collect();
        let mut out = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if !axis_set.contains(&idx) {
                out.push(dim.clone());
            }
        }
        out
    }
}

fn product_of_axes(shape: &[ShapeDim], axes: &[usize]) -> Option<usize> {
    if axes.is_empty() {
        return Some(1);
    }
    let mut total = 1usize;
    for &axis in axes {
        match shape.get(axis)? {
            ShapeDim::Known(n) => {
                total = total.checked_mul(*n)?;
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(total)
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

fn normalize_expand_axis(axis: i32, rank: usize) -> Result<usize, String> {
    let extended = rank + 1;
    let idx = if axis < 0 {
        (extended as i32) + axis
    } else {
        axis
    };
    if idx < 0 || idx > extended as i32 - 1 {
        Err(format!(
            "axis {axis} out of range for expand_dims (rank {rank})"
        ))
    } else {
        Ok(idx as usize)
    }
}

fn compute_squeeze_axes(shape: &[ShapeDim], axes: &[i32]) -> Result<Vec<usize>, String> {
    if axes.is_empty() {
        let mut remove = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if matches!(dim, ShapeDim::Known(1)) {
                remove.push(idx);
            }
        }
        return Ok(remove);
    }
    let normalized = normalize_axes_list(axes, shape.len())?;
    for &axis in &normalized {
        match shape.get(axis) {
            Some(ShapeDim::Known(1)) => {}
            Some(_) => {
                return Err(format!("cannot squeeze axis {axis}: dimension is not 1"));
            }
            None => return Err(format!("axis {axis} out of range for squeeze")),
        }
    }
    Ok(normalized)
}

fn dims_from_strings(dims: &[String]) -> Vec<ShapeDim> {
    dims.iter()
        .map(|d| {
            if let Ok(n) = d.parse::<usize>() {
                ShapeDim::Known(n)
            } else {
                ShapeDim::Sym(Box::leak(d.clone().into_boxed_str()))
            }
        })
        .collect()
}

fn fresh_symbol(prefix: &str) -> &'static str {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    Box::leak(format!("{prefix}{id}").into_boxed_str())
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

fn conv_channels_compatible(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(sa), ShapeDim::Sym(sb)) => sa == sb,
        _ => true,
    }
}

fn dim_display(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

fn conv_output_dim_autodiff(
    input: &ShapeDim,
    kernel: Option<&ShapeDim>,
    stride: usize,
    padding: ConvPadding,
) -> Result<ShapeDim, String> {
    let input_known = match input {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    };
    let kernel_known = kernel.and_then(|dim| match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    });
    let res = match padding {
        ConvPadding::Valid => linalg::conv_output_dim_valid(input_known, kernel_known, stride),
        ConvPadding::Same => linalg::conv_output_dim_same(input_known, stride),
    };
    match res {
        Ok(Some(v)) => Ok(ShapeDim::Known(v)),
        Ok(None) => Ok(ShapeDim::Sym(fresh_symbol("_conv"))),
        Err(msg) => Err(msg),
    }
}

fn expand_reduced_axes(mut grad: TensorVal, axes: &[usize], target_rank: usize) -> TensorVal {
    if axes.is_empty() {
        return grad;
    }
    let axis_set: BTreeSet<usize> = axes.iter().cloned().collect();
    let mut shape = Vec::with_capacity(target_rank);
    let mut src_idx = 0usize;
    for axis in 0..target_rank {
        if axis_set.contains(&axis) {
            shape.push(ShapeDim::Known(1));
        } else if src_idx < grad.shape.len() {
            shape.push(grad.shape[src_idx].clone());
            src_idx += 1;
        } else {
            shape.push(ShapeDim::Known(1));
        }
    }
    grad.shape = shape;
    grad
}

fn can_broadcast_to(from: &[ShapeDim], to: &[ShapeDim]) -> bool {
    if let Some(result) = broadcast_shapes(from, to) {
        result == to
    } else {
        false
    }
}

fn broadcast_to_shape(mut grad: TensorVal, target_shape: &[ShapeDim]) -> TensorVal {
    let can = can_broadcast_to(&grad.shape, target_shape);
    grad.shape = target_shape.to_vec();
    if !can {
        grad.fill = None;
    }
    grad
}

pub fn build_graph_loss(
    expr: &ast::Node,
    tenv: &HashMap<String, TensorEnvEntry>,
    expanding: &mut BTreeSet<String>,
) -> Result<(NodeId, Tape, BTreeMap<String, NodeId>), String> {
    let mut tape = Tape::new();
    let mut vars: BTreeMap<String, NodeId> = BTreeMap::new();
    let mut var_nodes: HashMap<String, NodeId> = HashMap::new();

    fn rec(
        node: &ast::Node,
        tenv: &HashMap<String, TensorEnvEntry>,
        tape: &mut Tape,
        vars: &mut BTreeMap<String, NodeId>,
        var_nodes: &mut HashMap<String, NodeId>,
        expanding: &mut BTreeSet<String>,
    ) -> Result<NodeId, String> {
        use ast::Literal;
        match node {
            ast::Node::Lit(Literal::Int(k), _) => Ok(tape.push(NodeInfo {
                op: Op::ConstInt,
                dtype: DType::F32,
                shape: Vec::new(),
                fill: Some(*k as f64),
            })),
            ast::Node::Lit(Literal::Ident(name), _) => {
                if let Some(existing) = var_nodes.get(name) {
                    vars.entry(name.clone()).or_insert(*existing);
                    return Ok(*existing);
                }
                let entry = tenv
                    .get(name)
                    .ok_or_else(|| format!("unknown tensor variable `{name}`"))?;
                if let Some(expr) = &entry.expr {
                    if !expanding.insert(name.clone()) {
                        return Err(format!("cyclic tensor alias `{name}`"));
                    }
                    let result = rec(expr, tenv, tape, vars, var_nodes, expanding);
                    expanding.remove(name);
                    let id = result?;
                    vars.insert(name.clone(), id);
                    var_nodes.insert(name.clone(), id);
                    Ok(id)
                } else {
                    let tensor = &entry.value;
                    let id = tape.push(NodeInfo {
                        op: Op::LeafVar,
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        fill: tensor.fill,
                    });
                    vars.insert(name.clone(), id);
                    var_nodes.insert(name.clone(), id);
                    Ok(id)
                }
            }
            ast::Node::Paren(inner, _) => rec(inner, tenv, tape, vars, var_nodes, expanding),
            ast::Node::Binary {
                op, left, right, ..
            } => {
                let l = rec(left, tenv, tape, vars, var_nodes, expanding)?;
                let r = rec(right, tenv, tape, vars, var_nodes, expanding)?;
                let lhs = &tape.nodes[l.0];
                let rhs = &tape.nodes[r.0];
                if lhs.dtype != rhs.dtype {
                    return Err("dtype mismatch in autodiff".to_string());
                }
                let shape = broadcast_shapes(&lhs.shape, &rhs.shape)
                    .ok_or_else(|| "broadcast failure in autodiff".to_string())?;
                let fill = match (lhs.fill, rhs.fill, op) {
                    (Some(a), Some(b), ast::BinOp::Add) => Some(a + b),
                    (Some(a), Some(b), ast::BinOp::Sub) => Some(a - b),
                    (Some(a), Some(b), ast::BinOp::Mul) => Some(a * b),
                    (Some(a), Some(b), ast::BinOp::Div) => {
                        if b == 0.0 {
                            None
                        } else {
                            Some(a / b)
                        }
                    }
                    _ => None,
                };
                let info = match op {
                    ast::BinOp::Add => NodeInfo {
                        op: Op::Add(l, r),
                        dtype: lhs.dtype.clone(),
                        shape,
                        fill,
                    },
                    ast::BinOp::Sub => NodeInfo {
                        op: Op::Sub(l, r),
                        dtype: lhs.dtype.clone(),
                        shape,
                        fill,
                    },
                    ast::BinOp::Mul => NodeInfo {
                        op: Op::Mul(l, r),
                        dtype: lhs.dtype.clone(),
                        shape,
                        fill,
                    },
                    ast::BinOp::Div => NodeInfo {
                        op: Op::Div(l, r),
                        dtype: lhs.dtype.clone(),
                        shape,
                        fill,
                    },
                };
                Ok(tape.push(info))
            }
            ast::Node::Call { callee, args, .. } => {
                if callee == "tensor.sum" && args.len() == 1 {
                    let child = rec(&args[0], tenv, tape, vars, var_nodes, expanding)?;
                    let child_info = &tape.nodes[child.0];
                    let axes = normalize_reduce_axes(&[], child_info.shape.len())?;
                    let reduced = product_of_axes(&child_info.shape, &axes);
                    let shape = reduce_shape(&child_info.shape, &axes, false);
                    let fill = match (child_info.fill, reduced) {
                        (Some(f), Some(n)) => Some(f * n as f64),
                        _ => None,
                    };
                    let info = NodeInfo {
                        op: Op::ReduceSum {
                            x: child,
                            axes,
                            keepdims: false,
                            reduced_elems: reduced,
                        },
                        dtype: child_info.dtype.clone(),
                        shape,
                        fill,
                    };
                    Ok(tape.push(info))
                } else {
                    Err("unsupported call in autodiff".to_string())
                }
            }
            ast::Node::CallTensorRelu { x, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let fill = child_info.fill.map(|f| if f < 0.0 { 0.0 } else { f });
                let info = NodeInfo {
                    op: Op::Relu { x: child },
                    dtype: child_info.dtype.clone(),
                    shape: child_info.shape.clone(),
                    fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallTensorConv2d {
                x,
                w,
                stride_h,
                stride_w,
                padding,
                ..
            } => {
                if *stride_h == 0 || *stride_w == 0 {
                    return Err("`tensor.conv2d`: strides must be positive".to_string());
                }
                let left = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let right = rec(w, tenv, tape, vars, var_nodes, expanding)?;
                let left_info = &tape.nodes[left.0];
                let right_info = &tape.nodes[right.0];
                if left_info.shape.len() != 4 {
                    return Err("`tensor.conv2d` expects input layout NHWC".to_string());
                }
                if right_info.shape.len() != 4 {
                    return Err("`tensor.conv2d` expects filter layout HWIO".to_string());
                }
                if !conv_channels_compatible(&left_info.shape[3], &right_info.shape[2]) {
                    return Err(format!(
                        "`tensor.conv2d`: channel mismatch {} vs {}",
                        dim_display(&left_info.shape[3]),
                        dim_display(&right_info.shape[2])
                    ));
                }
                let dtype = if left_info.dtype == right_info.dtype {
                    left_info.dtype.clone()
                } else if matches!(left_info.dtype, DType::F32)
                    || matches!(right_info.dtype, DType::F32)
                {
                    DType::F32
                } else {
                    return Err("`tensor.conv2d`: incompatible dtypes".to_string());
                };

                let out_h = conv_output_dim_autodiff(
                    &left_info.shape[1],
                    Some(&right_info.shape[0]),
                    *stride_h,
                    *padding,
                )?;
                let out_w = conv_output_dim_autodiff(
                    &left_info.shape[2],
                    Some(&right_info.shape[1]),
                    *stride_w,
                    *padding,
                )?;

                let shape = vec![
                    left_info.shape[0].clone(),
                    out_h,
                    out_w,
                    right_info.shape[3].clone(),
                ];

                let info = NodeInfo {
                    op: Op::Conv2d {
                        x: left,
                        w: right,
                        stride_h: *stride_h,
                        stride_w: *stride_w,
                        padding: *padding,
                    },
                    dtype,
                    shape,
                    fill: None,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallTensorSum {
                x, axes, keepdims, ..
            } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let axes_norm = normalize_reduce_axes(axes, child_info.shape.len())?;
                let reduced = product_of_axes(&child_info.shape, &axes_norm);
                let shape = reduce_shape(&child_info.shape, &axes_norm, *keepdims);
                let fill = match (child_info.fill, reduced) {
                    (Some(f), Some(n)) => Some(f * n as f64),
                    (Some(f), None) if axes_norm.is_empty() => Some(f),
                    (Some(f), None) if child_info.shape.is_empty() => Some(f),
                    _ => None,
                };
                let info = NodeInfo {
                    op: Op::ReduceSum {
                        x: child,
                        axes: axes_norm.clone(),
                        keepdims: *keepdims,
                        reduced_elems: reduced,
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallTensorMean {
                x, axes, keepdims, ..
            } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let axes_norm = normalize_reduce_axes(axes, child_info.shape.len())?;
                let reduced = product_of_axes(&child_info.shape, &axes_norm);
                let shape = reduce_shape(&child_info.shape, &axes_norm, *keepdims);
                let fill = child_info.fill;
                let info = NodeInfo {
                    op: Op::ReduceMean {
                        x: child,
                        axes: axes_norm.clone(),
                        keepdims: *keepdims,
                        reduced_elems: reduced,
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallReshape { x, dims, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let new_shape = dims_from_strings(dims);
                if new_shape.len() != child_info.shape.len() {
                    return Err("reshape rank mismatch".to_string());
                }
                if let (Some(old), Some(new)) =
                    (known_product(&child_info.shape), known_product(&new_shape))
                {
                    if old != new {
                        return Err("reshape element count mismatch".to_string());
                    }
                }
                let info = NodeInfo {
                    op: Op::Reshape { x: child },
                    dtype: child_info.dtype.clone(),
                    shape: new_shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallExpandDims { x, axis, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let axis_norm = normalize_expand_axis(*axis, child_info.shape.len())?;
                let mut shape = child_info.shape.clone();
                shape.insert(axis_norm, ShapeDim::Known(1));
                let info = NodeInfo {
                    op: Op::ExpandDims {
                        x: child,
                        axis: axis_norm,
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallSqueeze { x, axes, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let axes_to_remove = compute_squeeze_axes(&child_info.shape, axes)?;
                let axis_set: BTreeSet<usize> = axes_to_remove.iter().cloned().collect();
                let mut shape = Vec::new();
                for (idx, dim) in child_info.shape.iter().enumerate() {
                    if !axis_set.contains(&idx) {
                        shape.push(dim.clone());
                    }
                }
                let info = NodeInfo {
                    op: Op::Squeeze {
                        x: child,
                        axes: axes_to_remove.clone(),
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallTranspose { x, axes, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let rank = child_info.shape.len();
                let perm = if let Some(spec) = axes {
                    linalg::normalize_permutation(spec, rank)
                        .map_err(|msg| format!("tensor.transpose: {msg}"))?
                } else {
                    linalg::default_transpose(rank)
                };
                let shape = linalg::permute_shape(&child_info.shape, &perm);
                let info = NodeInfo {
                    op: Op::Transpose {
                        x: child,
                        axes: perm.clone(),
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallIndex { x, axis, i, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                if child_info.shape.is_empty() {
                    return Err("tensor.index: rank must be >= 1".to_string());
                }
                let axis_norm = normalize_axis(*axis, child_info.shape.len())?;
                if let ShapeDim::Known(n) = child_info.shape[axis_norm] {
                    if *i < 0 || (*i as usize) >= n {
                        return Err("tensor.index: index out of bounds".to_string());
                    }
                }
                let mut shape = child_info.shape.clone();
                shape.remove(axis_norm);
                let info = NodeInfo {
                    op: Op::Index {
                        x: child,
                        axis: axis_norm,
                        i: *i,
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallSlice {
                x,
                axis,
                start,
                end,
                ..
            } => {
                if *start < 0 || *end < *start {
                    return Err("tensor.slice: expected 0 <= start <= end".to_string());
                }
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let axis_norm = normalize_axis(*axis, child_info.shape.len())?;
                let new_dim = match child_info.shape[axis_norm].clone() {
                    ShapeDim::Known(n) => {
                        if *end as usize > n {
                            return Err("tensor.slice: end out of bounds".to_string());
                        }
                        ShapeDim::Known((*end - *start) as usize)
                    }
                    ShapeDim::Sym(sym) => ShapeDim::Sym(sym),
                };
                let mut shape = child_info.shape.clone();
                shape[axis_norm] = new_dim;
                let info = NodeInfo {
                    op: Op::Slice {
                        x: child,
                        axis: axis_norm,
                        start: *start,
                        end: *end,
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallSliceStride {
                x,
                axis,
                start,
                end,
                step,
                ..
            } => {
                if *step == 0 {
                    return Err("tensor.slice_stride: step must be non-zero".to_string());
                }
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let axis_norm = normalize_axis(*axis, child_info.shape.len())?;
                let new_dim = if let Some(len) = slice_len_with_step(
                    match child_info.shape[axis_norm] {
                        ShapeDim::Known(n) => Some(n),
                        ShapeDim::Sym(_) => None,
                    },
                    *start,
                    *end,
                    *step,
                ) {
                    ShapeDim::Known(len)
                } else if matches!(child_info.shape[axis_norm], ShapeDim::Known(_)) {
                    return Err("tensor.slice_stride: invalid bounds".to_string());
                } else if (*step > 0 && *start >= *end) || (*step < 0 && *start <= *end) {
                    ShapeDim::Known(0)
                } else {
                    ShapeDim::Sym(fresh_symbol("_slice_stride"))
                };
                let mut shape = child_info.shape.clone();
                shape[axis_norm] = new_dim;
                let info = NodeInfo {
                    op: Op::SliceStride {
                        x: child,
                        axis: axis_norm,
                        start: *start,
                        end: *end,
                        step: *step,
                        in_shape: child_info.shape.clone(),
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallGather { x, axis, idx, .. } => {
                let child = rec(x, tenv, tape, vars, var_nodes, expanding)?;
                let idx_id = rec(idx, tenv, tape, vars, var_nodes, expanding)?;
                let child_info = &tape.nodes[child.0];
                let idx_info = &tape.nodes[idx_id.0];
                if idx_info.dtype != DType::I32 {
                    return Err("tensor.gather: idx must be i32 tensor".to_string());
                }
                let axis_norm = normalize_axis(*axis, child_info.shape.len())?;
                let mut shape = Vec::new();
                shape.extend_from_slice(&child_info.shape[..axis_norm]);
                shape.extend(idx_info.shape.iter().cloned());
                if axis_norm < child_info.shape.len() {
                    shape.extend_from_slice(&child_info.shape[axis_norm + 1..]);
                }
                let info = NodeInfo {
                    op: Op::Gather {
                        x: child,
                        idx: idx_id,
                        axis: axis_norm,
                        in_shape: child_info.shape.clone(),
                    },
                    dtype: child_info.dtype.clone(),
                    shape,
                    fill: child_info.fill,
                };
                Ok(tape.push(info))
            }
            ast::Node::CallDot { a, b, .. } => {
                let left = rec(a, tenv, tape, vars, var_nodes, expanding)?;
                let right = rec(b, tenv, tape, vars, var_nodes, expanding)?;
                let lhs = &tape.nodes[left.0];
                let rhs = &tape.nodes[right.0];
                if lhs.dtype != rhs.dtype {
                    return Err("tensor.dot dtype mismatch".to_string());
                }
                let info = linalg::compute_matmul_shape_info(&lhs.shape, &rhs.shape)
                    .map_err(|msg| format!("tensor.dot: {msg}"))?;
                let fill = match (lhs.fill, rhs.fill, linalg::known_dim_value(&info.k_dim)) {
                    (Some(a), Some(b), Some(k)) => Some(a * b * k as f64),
                    _ => None,
                };
                let node_info = NodeInfo {
                    op: Op::Dot {
                        a: left,
                        b: right,
                        info: info.clone(),
                    },
                    dtype: lhs.dtype.clone(),
                    shape: info.result_shape.clone(),
                    fill,
                };
                Ok(tape.push(node_info))
            }
            ast::Node::CallMatMul { a, b, .. } => {
                let left = rec(a, tenv, tape, vars, var_nodes, expanding)?;
                let right = rec(b, tenv, tape, vars, var_nodes, expanding)?;
                let lhs = &tape.nodes[left.0];
                let rhs = &tape.nodes[right.0];
                if lhs.dtype != rhs.dtype {
                    return Err("tensor.matmul dtype mismatch".to_string());
                }
                let info = linalg::compute_matmul_shape_info(&lhs.shape, &rhs.shape)
                    .map_err(|msg| format!("tensor.matmul: {msg}"))?;
                let fill = match (lhs.fill, rhs.fill, linalg::known_dim_value(&info.k_dim)) {
                    (Some(a), Some(b), Some(k)) => Some(a * b * k as f64),
                    _ => None,
                };
                let node_info = NodeInfo {
                    op: Op::MatMul {
                        a: left,
                        b: right,
                        info: info.clone(),
                    },
                    dtype: lhs.dtype.clone(),
                    shape: info.result_shape.clone(),
                    fill,
                };
                Ok(tape.push(node_info))
            }
            _ => Err("unsupported node in autodiff".to_string()),
        }
    }

    let loss = rec(expr, tenv, &mut tape, &mut vars, &mut var_nodes, expanding)?;
    Ok((loss, tape, vars))
}

pub fn backprop_to_vars(
    loss: NodeId,
    tape: &Tape,
    vars: &BTreeMap<String, NodeId>,
) -> BTreeMap<String, TensorVal> {
    backprop_to_vars_with_tenv(loss, tape, vars, &HashMap::new())
}

/// Backprop with access to tensor environment for real data computation.
pub fn backprop_to_vars_with_tenv(
    loss: NodeId,
    tape: &Tape,
    vars: &BTreeMap<String, NodeId>,
    tenv: &HashMap<String, TensorEnvEntry>,
) -> BTreeMap<String, TensorVal> {
    // Create inverse mapping from NodeId to variable name for data lookup
    let node_to_name: HashMap<NodeId, &str> = vars
        .iter()
        .map(|(name, id)| (*id, name.as_str()))
        .collect();
    let mut adj: HashMap<NodeId, TensorVal> = HashMap::new();
    if let Some(loss_node) = tape.nodes.get(loss.0) {
        adj.insert(
            loss,
            TensorVal::new(loss_node.dtype.clone(), loss_node.shape.clone(), Some(1.0)),
        );
    }

    for idx in (0..tape.nodes.len()).rev() {
        let nid = NodeId(idx);
        let Some(grad) = adj.get(&nid).cloned() else {
            continue;
        };
        let node = &tape.nodes[idx];
        match &node.op {
            Op::Add(l, r) => {
                push_grad(&mut adj, tape, *l, &grad);
                push_grad(&mut adj, tape, *r, &grad);
            }
            Op::Sub(l, r) => {
                push_grad(&mut adj, tape, *l, &grad);
                push_grad_neg(&mut adj, tape, *r, &grad);
            }
            Op::Mul(l, r) => {
                let right_fill = tape.nodes[r.0].fill;
                let left_fill = tape.nodes[l.0].fill;
                push_grad_scaled(&mut adj, tape, *l, &grad, right_fill);
                push_grad_scaled(&mut adj, tape, *r, &grad, left_fill);
            }
            Op::Div(l, r) => {
                let right_fill = tape.nodes[r.0].fill;
                let left_fill = tape.nodes[l.0].fill;
                let scale_left =
                    right_fill.and_then(|v| if v == 0.0 { None } else { Some(1.0 / v) });
                let scale_right = match (left_fill, right_fill) {
                    (Some(x), Some(rf)) if rf != 0.0 => Some(-x / (rf * rf)),
                    _ => None,
                };
                push_grad_scaled(&mut adj, tape, *l, &grad, scale_left);
                push_grad_scaled(&mut adj, tape, *r, &grad, scale_right);
            }
            Op::ReduceSum {
                x,
                axes,
                keepdims,
                reduced_elems,
            } => {
                let _ = reduced_elems;
                let mut expanded = grad.clone();
                if !keepdims {
                    expanded = expand_reduced_axes(expanded, axes, tape.nodes[x.0].shape.len());
                }
                let broadcasted = broadcast_to_shape(expanded, &tape.nodes[x.0].shape);
                accumulate_grad(&mut adj, *x, broadcasted);
            }
            Op::ReduceMean {
                x,
                axes,
                keepdims,
                reduced_elems,
            } => {
                let mut adjusted = grad.clone();
                match reduced_elems {
                    Some(n) if *n > 0 => {
                        if let Some(fill) = adjusted.fill {
                            adjusted.fill = Some(fill / *n as f64);
                        }
                    }
                    Some(_) => {}
                    None => {
                        if adjusted.fill.is_some() {
                            adjusted.fill = None;
                        }
                    }
                }
                if !keepdims {
                    adjusted = expand_reduced_axes(adjusted, axes, tape.nodes[x.0].shape.len());
                }
                let broadcasted = broadcast_to_shape(adjusted, &tape.nodes[x.0].shape);
                accumulate_grad(&mut adj, *x, broadcasted);
            }
            Op::Reshape { x } => {
                let mut reshaped = grad.clone();
                reshaped.shape = tape.nodes[x.0].shape.clone();
                accumulate_grad(&mut adj, *x, reshaped);
            }
            Op::ExpandDims { x, axis } => {
                let _ = axis;
                let mut reshaped = grad.clone();
                reshaped.shape = tape.nodes[x.0].shape.clone();
                accumulate_grad(&mut adj, *x, reshaped);
            }
            Op::Squeeze { x, axes } => {
                let _ = axes;
                let mut reshaped = grad.clone();
                reshaped.shape = tape.nodes[x.0].shape.clone();
                accumulate_grad(&mut adj, *x, reshaped);
            }
            Op::Transpose { x, axes } => {
                let inv = linalg::invert_permutation(axes);
                let transposed = transpose_tensorval(&grad, &inv);
                accumulate_grad(&mut adj, *x, transposed);
            }
            Op::Index { x, axis, i } => {
                let child_info = &tape.nodes[x.0];
                let _ = (axis, i);
                let scattered =
                    TensorVal::new(child_info.dtype.clone(), child_info.shape.clone(), None);
                accumulate_grad(&mut adj, *x, scattered);
            }
            Op::Slice {
                x,
                axis,
                start,
                end,
            } => {
                let child_info = &tape.nodes[x.0];
                let mut fill = None;
                if let (Some(gfill), Some(_orig_fill)) = (grad.fill, child_info.fill) {
                    if let ShapeDim::Known(len) = child_info.shape[*axis] {
                        if *start == 0 && *end == len as i32 {
                            fill = Some(gfill);
                        }
                    }
                }
                let back = TensorVal::new(child_info.dtype.clone(), child_info.shape.clone(), fill);
                accumulate_grad(&mut adj, *x, back);
            }
            Op::SliceStride {
                x,
                in_shape,
                axis,
                start,
                end,
                step,
            } => {
                let _ = (axis, start, end, step);
                let child_info = &tape.nodes[x.0];
                let back = TensorVal::new(child_info.dtype.clone(), in_shape.clone(), None);
                accumulate_grad(&mut adj, *x, back);
            }
            Op::Gather {
                x,
                in_shape,
                idx,
                axis,
            } => {
                let _ = (idx, axis);
                let child_info = &tape.nodes[x.0];
                let back = TensorVal::new(child_info.dtype.clone(), in_shape.clone(), None);
                accumulate_grad(&mut adj, *x, back);
            }
            Op::Dot { a, b, info } | Op::MatMul { a, b, info } => {
                backprop_matmul_op(&mut adj, tape, &grad, *a, *b, info);
            }
            Op::Relu { x } => {
                let child_info = &tape.nodes[x.0];
                let mut adjusted = grad.clone();
                match child_info.fill {
                    Some(f) if f > 0.0 => {}
                    Some(_) => {
                        adjusted.fill = Some(0.0);
                    }
                    None => {
                        adjusted.fill = None;
                    }
                }
                accumulate_grad(&mut adj, *x, adjusted);
            }
            Op::Conv2d {
                x,
                w,
                stride_h,
                stride_w,
                padding,
            } => {
                let x_info = &tape.nodes[x.0];
                let w_info = &tape.nodes[w.0];

                // Try to compute real gradients if buffer data is available
                #[cfg(feature = "cpu-buffers")]
                {
                    let computed = try_compute_conv2d_grad(
                        &grad,
                        *x,
                        *w,
                        x_info,
                        w_info,
                        *stride_h,
                        *stride_w,
                        *padding,
                        &node_to_name,
                        tenv,
                    );
                    if let Some((dx, dw)) = computed {
                        accumulate_grad(&mut adj, *x, dx);
                        accumulate_grad(&mut adj, *w, dw);
                        continue;
                    }
                }

                // Fallback to shape-only gradients
                #[cfg(not(feature = "cpu-buffers"))]
                let _ = (&node_to_name, tenv, stride_h, stride_w, padding);

                accumulate_grad(
                    &mut adj,
                    *x,
                    TensorVal::new(x_info.dtype.clone(), x_info.shape.clone(), None),
                );
                accumulate_grad(
                    &mut adj,
                    *w,
                    TensorVal::new(w_info.dtype.clone(), w_info.shape.clone(), None),
                );
            }
            Op::LeafVar | Op::ConstInt => {}
        }
    }

    let mut out = BTreeMap::new();
    for (name, id) in vars {
        if let Some(g) = adj.get(id) {
            eprintln!("grad {:?} fill {:?}", name, g.fill);
            out.insert(name.clone(), g.clone());
        }
    }
    out
}

fn push_grad(
    adj: &mut HashMap<NodeId, TensorVal>,
    tape: &Tape,
    target: NodeId,
    upstream: &TensorVal,
) {
    let adjusted = adjust_for_broadcast(upstream, &tape.nodes[target.0]);
    accumulate_grad(adj, target, adjusted);
}

fn push_grad_neg(
    adj: &mut HashMap<NodeId, TensorVal>,
    tape: &Tape,
    target: NodeId,
    upstream: &TensorVal,
) {
    push_grad_scaled(adj, tape, target, upstream, Some(-1.0));
}

fn push_grad_scaled(
    adj: &mut HashMap<NodeId, TensorVal>,
    tape: &Tape,
    target: NodeId,
    upstream: &TensorVal,
    scale: Option<f64>,
) {
    let mut scaled = upstream.clone();
    match scale {
        Some(s) => {
            if let Some(f) = scaled.fill {
                scaled.fill = Some(f * s);
            } else if s == 0.0 {
                scaled.fill = Some(0.0);
            }
        }
        None => {
            scaled.fill = None;
        }
    }
    push_grad(adj, tape, target, &scaled);
}

fn adjust_for_broadcast(upstream: &TensorVal, target: &NodeInfo) -> TensorVal {
    if upstream.shape == target.shape {
        return TensorVal::new(target.dtype.clone(), target.shape.clone(), upstream.fill);
    }

    let factor = reduction_factor(&upstream.shape, &target.shape);
    let fill = match (upstream.fill, factor) {
        (Some(f), Some(fac)) => Some(f * fac),
        (Some(_), None) => None,
        (None, _) => None,
    };

    TensorVal::new(target.dtype.clone(), target.shape.clone(), fill)
}

fn reduction_factor(output: &[ShapeDim], target: &[ShapeDim]) -> Option<f64> {
    let mut factor = 1.0f64;
    let mut i = output.len() as isize - 1;
    let mut j = target.len() as isize - 1;
    while i >= 0 || j >= 0 {
        let od = if i >= 0 {
            &output[i as usize]
        } else {
            &ShapeDim::Known(1)
        };
        let td = if j >= 0 {
            &target[j as usize]
        } else {
            &ShapeDim::Known(1)
        };
        match (od, td) {
            (ShapeDim::Known(o), ShapeDim::Known(t)) => {
                if o == t {
                    // nothing
                } else if *t == 1 {
                    factor *= *o as f64;
                } else {
                    return None;
                }
            }
            (ShapeDim::Known(o), ShapeDim::Sym(_)) => {
                if *o != 1 {
                    return None;
                }
            }
            (ShapeDim::Sym(_), ShapeDim::Known(_)) => return None,
            (ShapeDim::Sym(os), ShapeDim::Sym(ts)) => {
                if os != ts {
                    return None;
                }
            }
        }
        i -= 1;
        j -= 1;
    }
    Some(factor)
}

fn accumulate_grad(adj: &mut HashMap<NodeId, TensorVal>, target: NodeId, incoming: TensorVal) {
    adj.entry(target)
        .and_modify(|existing| {
            existing.fill = match (existing.fill, incoming.fill) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
        })
        .or_insert(incoming);
}

fn transpose_tensorval(tensor: &TensorVal, perm: &[usize]) -> TensorVal {
    let shape = linalg::permute_shape(&tensor.shape, perm);
    TensorVal::new(tensor.dtype.clone(), shape, tensor.fill)
}

fn reshape_to_target(mut tensor: TensorVal, target_shape: &[ShapeDim]) -> TensorVal {
    if tensor.shape == target_shape {
        return tensor;
    }
    let mut can_keep_fill = true;
    let mut src_idx = tensor.shape.len() as isize - 1;
    let mut tgt_idx = target_shape.len() as isize - 1;
    while tgt_idx >= 0 {
        if src_idx < 0 {
            can_keep_fill = false;
            break;
        }
        let src_dim = &tensor.shape[src_idx as usize];
        let tgt_dim = &target_shape[tgt_idx as usize];
        if src_dim == tgt_dim {
            src_idx -= 1;
            tgt_idx -= 1;
        } else if matches!(src_dim, ShapeDim::Known(1)) {
            src_idx -= 1;
        } else {
            can_keep_fill = false;
            break;
        }
    }
    if tgt_idx >= 0 {
        can_keep_fill = false;
    }
    while src_idx >= 0 {
        if !matches!(tensor.shape[src_idx as usize], ShapeDim::Known(1)) {
            can_keep_fill = false;
            break;
        }
        src_idx -= 1;
    }
    tensor.shape = target_shape.to_vec();
    if !can_keep_fill {
        tensor.fill = None;
    }
    tensor
}

fn expand_grad_for_matmul(grad: &TensorVal, info: &MatMulShapeInfo) -> TensorVal {
    let mut shape = info.broadcast_shape.clone();
    if info.a_was_vec {
        shape.push(ShapeDim::Known(1));
    } else {
        shape.push(info.m_dim.clone());
    }
    if info.b_was_vec {
        shape.push(ShapeDim::Known(1));
    } else {
        shape.push(info.n_dim.clone());
    }
    TensorVal::new(grad.dtype.clone(), shape, grad.fill)
}

fn matmul_preview_simple(lhs: &TensorVal, rhs: &TensorVal) -> Option<(TensorVal, MatMulShapeInfo)> {
    if lhs.dtype != rhs.dtype {
        return None;
    }
    let info = linalg::compute_matmul_shape_info(&lhs.shape, &rhs.shape).ok()?;
    let fill = match (lhs.fill, rhs.fill, linalg::known_dim_value(&info.k_dim)) {
        (Some(a), Some(b), Some(k)) => Some(a * b * k as f64),
        _ => None,
    };
    let result = TensorVal::new(lhs.dtype.clone(), info.result_shape.clone(), fill);
    Some((result, info))
}

fn swap_last_two(rank: usize) -> Vec<usize> {
    let mut axes: Vec<usize> = (0..rank).collect();
    if rank >= 2 {
        axes.swap(rank - 1, rank - 2);
    }
    axes
}

fn backprop_matmul_op(
    adj: &mut HashMap<NodeId, TensorVal>,
    tape: &Tape,
    upstream: &TensorVal,
    a: NodeId,
    b: NodeId,
    info: &MatMulShapeInfo,
) {
    let grad_expanded = expand_grad_for_matmul(upstream, info);

    let b_info = &tape.nodes[b.0];
    let b_tensor = TensorVal::new(b_info.dtype.clone(), info.b_shape.clone(), b_info.fill);
    let b_t = transpose_tensorval(&b_tensor, &swap_last_two(info.b_shape.len()));
    if let Some((mut grad_a, _)) = matmul_preview_simple(&grad_expanded, &b_t) {
        grad_a.fill = grad_expanded.fill;
        let aligned = reshape_to_target(grad_a, &tape.nodes[a.0].shape);
        accumulate_grad(adj, a, aligned);
    } else {
        let zero = TensorVal::new(
            tape.nodes[a.0].dtype.clone(),
            tape.nodes[a.0].shape.clone(),
            None,
        );
        accumulate_grad(adj, a, zero);
    }

    let a_info = &tape.nodes[a.0];
    let a_tensor = TensorVal::new(a_info.dtype.clone(), info.a_shape.clone(), a_info.fill);
    let a_t = transpose_tensorval(&a_tensor, &swap_last_two(info.a_shape.len()));
    if let Some((mut grad_b, _)) = matmul_preview_simple(&a_t, &grad_expanded) {
        grad_b.fill = grad_expanded.fill;
        let aligned = reshape_to_target(grad_b, &tape.nodes[b.0].shape);
        accumulate_grad(adj, b, aligned);
    } else {
        let zero = TensorVal::new(
            tape.nodes[b.0].dtype.clone(),
            tape.nodes[b.0].shape.clone(),
            None,
        );
        accumulate_grad(adj, b, zero);
    }
}

#[allow(dead_code)]
fn known_num_elems(shape: &[ShapeDim]) -> Option<usize> {
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

/// Try to compute real Conv2d gradients when buffer data is available.
///
/// Returns Some((dx, dw)) if computation was successful, None otherwise.
#[cfg(feature = "cpu-buffers")]
#[allow(clippy::too_many_arguments)]
fn try_compute_conv2d_grad(
    upstream_grad: &TensorVal,
    x_id: NodeId,
    w_id: NodeId,
    x_info: &NodeInfo,
    w_info: &NodeInfo,
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
    node_to_name: &HashMap<NodeId, &str>,
    tenv: &HashMap<String, TensorEnvEntry>,
) -> Option<(TensorVal, TensorVal)> {
    use crate::eval::value::Buffer;

    // Check that we have f32 dtype
    if !matches!(x_info.dtype, DType::F32) || !matches!(w_info.dtype, DType::F32) {
        return None;
    }

    // Get variable names for x and w
    let x_name = node_to_name.get(&x_id)?;
    let w_name = node_to_name.get(&w_id)?;

    // Look up tensors in environment
    let x_entry = tenv.get(*x_name)?;
    let w_entry = tenv.get(*w_name)?;

    // Get buffer data
    let x_buf = x_entry.value.buf.as_ref()?;
    let w_buf = w_entry.value.buf.as_ref()?;

    let x_data = match x_buf {
        Buffer::F32(data) => data.as_slice(),
        _ => return None,
    };
    let w_data = match w_buf {
        Buffer::F32(data) => data.as_slice(),
        _ => return None,
    };

    // Get shapes as [usize; 4]
    let x_shape = shape_to_array4(&x_info.shape)?;
    let w_shape = shape_to_array4(&w_info.shape)?;

    // Get or materialize upstream gradient dy
    let dy_shape = shape_to_array4(&upstream_grad.shape)?;
    let dy_data: Vec<f32> = match &upstream_grad.buf {
        Some(Buffer::F32(data)) => data.clone(),
        _ => {
            // If upstream is fill-based, materialize it
            let fill = upstream_grad.fill? as f32;
            let n = dy_shape.iter().product();
            vec![fill; n]
        }
    };

    // Compute gradients
    let (dx_data, dw_data) = conv2d_grad::conv2d_vjp_nhwc_hwio_f32(
        x_data, x_shape, w_data, w_shape, &dy_data, dy_shape, stride_h, stride_w, padding,
    );

    // Build result TensorVals with buffer data
    let mut dx = TensorVal::new(DType::F32, x_info.shape.clone(), None);
    dx.buf = Some(Buffer::F32(dx_data));

    let mut dw = TensorVal::new(DType::F32, w_info.shape.clone(), None);
    dw.buf = Some(Buffer::F32(dw_data));

    Some((dx, dw))
}

/// Convert ShapeDim slice to [usize; 4] array if all dimensions are known.
#[cfg(feature = "cpu-buffers")]
fn shape_to_array4(shape: &[ShapeDim]) -> Option<[usize; 4]> {
    if shape.len() != 4 {
        return None;
    }
    let mut arr = [0usize; 4];
    for (i, dim) in shape.iter().enumerate() {
        match dim {
            ShapeDim::Known(n) => arr[i] = *n,
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(arr)
}
