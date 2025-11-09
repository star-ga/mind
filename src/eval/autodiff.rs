use std::collections::{BTreeMap, HashMap};

use crate::ast;
use crate::eval::TensorVal;
use crate::types::{DType, ShapeDim};

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
    Sum(NodeId),
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

pub fn build_graph_loss(
    expr: &ast::Node,
    tenv: &HashMap<String, TensorVal>,
) -> Result<(NodeId, Tape, BTreeMap<String, NodeId>), String> {
    let mut tape = Tape::new();
    let mut vars: BTreeMap<String, NodeId> = BTreeMap::new();
    let mut var_nodes: HashMap<String, NodeId> = HashMap::new();

    fn rec(
        node: &ast::Node,
        tenv: &HashMap<String, TensorVal>,
        tape: &mut Tape,
        vars: &mut BTreeMap<String, NodeId>,
        var_nodes: &mut HashMap<String, NodeId>,
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
                    return Ok(*existing);
                }
                if let Some(t) = tenv.get(name) {
                    let id = tape.push(NodeInfo {
                        op: Op::LeafVar,
                        dtype: t.dtype.clone(),
                        shape: t.shape.clone(),
                        fill: t.fill,
                    });
                    vars.insert(name.clone(), id);
                    var_nodes.insert(name.clone(), id);
                    Ok(id)
                } else {
                    Err(format!("unknown tensor variable `{name}`"))
                }
            }
            ast::Node::Paren(inner, _) => rec(inner, tenv, tape, vars, var_nodes),
            ast::Node::Binary { op, left, right, .. } => {
                let l = rec(left, tenv, tape, vars, var_nodes)?;
                let r = rec(right, tenv, tape, vars, var_nodes)?;
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
                    ast::BinOp::Add => {
                        NodeInfo { op: Op::Add(l, r), dtype: lhs.dtype.clone(), shape, fill }
                    }
                    ast::BinOp::Sub => {
                        NodeInfo { op: Op::Sub(l, r), dtype: lhs.dtype.clone(), shape, fill }
                    }
                    ast::BinOp::Mul => {
                        NodeInfo { op: Op::Mul(l, r), dtype: lhs.dtype.clone(), shape, fill }
                    }
                    ast::BinOp::Div => {
                        NodeInfo { op: Op::Div(l, r), dtype: lhs.dtype.clone(), shape, fill }
                    }
                };
                Ok(tape.push(info))
            }
            ast::Node::Call { callee, args, .. } => {
                if callee == "tensor.sum" && args.len() == 1 {
                    let child = rec(&args[0], tenv, tape, vars, var_nodes)?;
                    let child_info = &tape.nodes[child.0];
                    let fill = match child_info.fill {
                        Some(f) => known_num_elems(&child_info.shape).map(|n| f * n as f64),
                        None => None,
                    };
                    let info = NodeInfo {
                        op: Op::Sum(child),
                        dtype: child_info.dtype.clone(),
                        shape: Vec::new(),
                        fill,
                    };
                    Ok(tape.push(info))
                } else {
                    Err("unsupported call in autodiff".to_string())
                }
            }
            _ => Err("unsupported node in autodiff".to_string()),
        }
    }

    let loss = rec(expr, tenv, &mut tape, &mut vars, &mut var_nodes)?;
    Ok((loss, tape, vars))
}

pub fn backprop_to_vars(
    loss: NodeId,
    tape: &Tape,
    vars: &BTreeMap<String, NodeId>,
) -> BTreeMap<String, TensorVal> {
    let mut adj: HashMap<NodeId, TensorVal> = HashMap::new();
    if let Some(loss_node) = tape.nodes.get(loss.0) {
        adj.insert(
            loss,
            TensorVal::new(loss_node.dtype.clone(), loss_node.shape.clone(), Some(1.0)),
        );
    }

    for idx in (0..tape.nodes.len()).rev() {
        let nid = NodeId(idx);
        let Some(grad) = adj.get(&nid).cloned() else { continue };
        let node = &tape.nodes[idx];
        match node.op {
            Op::Add(l, r) => {
                push_grad(&mut adj, tape, l, &grad);
                push_grad(&mut adj, tape, r, &grad);
            }
            Op::Sub(l, r) => {
                push_grad(&mut adj, tape, l, &grad);
                push_grad_neg(&mut adj, tape, r, &grad);
            }
            Op::Mul(l, r) => {
                let right_fill = tape.nodes[r.0].fill;
                let left_fill = tape.nodes[l.0].fill;
                push_grad_scaled(&mut adj, tape, l, &grad, right_fill);
                push_grad_scaled(&mut adj, tape, r, &grad, left_fill);
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
                push_grad_scaled(&mut adj, tape, l, &grad, scale_left);
                push_grad_scaled(&mut adj, tape, r, &grad, scale_right);
            }
            Op::Sum(child) => {
                push_grad_sum(&mut adj, tape, child, &grad);
            }
            Op::LeafVar | Op::ConstInt => {}
        }
    }

    let mut out = BTreeMap::new();
    for (name, id) in vars {
        if let Some(g) = adj.get(id) {
            out.insert(name.clone(), g.clone());
        } else if let Some(node) = tape.nodes.get(id.0) {
            out.insert(
                name.clone(),
                TensorVal::new(node.dtype.clone(), node.shape.clone(), Some(0.0)),
            );
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

fn push_grad_sum(
    adj: &mut HashMap<NodeId, TensorVal>,
    tape: &Tape,
    target: NodeId,
    upstream: &TensorVal,
) {
    let target_node = &tape.nodes[target.0];
    let grad = TensorVal::new(target_node.dtype.clone(), target_node.shape.clone(), upstream.fill);
    accumulate_grad(adj, target, grad);
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
        let od = if i >= 0 { &output[i as usize] } else { &ShapeDim::Known(1) };
        let td = if j >= 0 { &target[j as usize] } else { &ShapeDim::Known(1) };
        match (od, td) {
            (ShapeDim::Known(o), ShapeDim::Known(t)) => {
                if o == t {
                    // nothing
                } else if *t == 1 {
                    factor *= *o as f64;
                } else if *o == 1 {
                    return None;
                } else {
                    return None;
                }
            }
            (ShapeDim::Known(o), ShapeDim::Sym(_)) => {
                if *o != 1 {
                    return None;
                }
            }
            (ShapeDim::Sym(_), ShapeDim::Known(t)) => {
                if *t == 1 {
                    return None;
                } else {
                    return None;
                }
            }
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
            if let (Some(a), Some(b)) = (existing.fill, incoming.fill) {
                existing.fill = Some(a + b);
            } else if incoming.fill.is_some() {
                existing.fill = None;
            } else {
                existing.fill = None;
            }
        })
        .or_insert(incoming);
}

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
