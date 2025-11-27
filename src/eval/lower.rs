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

use std::collections::HashMap;

use crate::ast;
use crate::ast::Literal;
use crate::ast::TypeAnn;

use crate::ir::BinOp;
use crate::ir::IRModule;
use crate::ir::IndexSpec;
use crate::ir::Instr;
use crate::ir::SliceSpec;
use crate::ir::ValueId;
use crate::types::DType;
use crate::types::ShapeDim;

pub fn lower_to_ir(module: &ast::Module) -> IRModule {
    let mut ir = IRModule::new();
    let mut env: HashMap<String, ValueId> = HashMap::new();

    for item in &module.items {
        match item {
            ast::Node::Let {
                name, ann, value, ..
            } => {
                let id = match ann {
                    Some(TypeAnn::Tensor { dtype, dims }) => {
                        lower_tensor_binding(&mut ir, value, dtype, dims, &env)
                    }
                    _ => lower_expr(value, &mut ir, &env),
                };
                env.insert(name.clone(), id);
                ir.instrs.push(Instr::Output(id));
            }
            ast::Node::Assign { name, value, .. } => {
                let id = lower_expr(value, &mut ir, &env);
                env.insert(name.clone(), id);
                ir.instrs.push(Instr::Output(id));
            }
            other => {
                let id = lower_expr(other, &mut ir, &env);
                ir.instrs.push(Instr::Output(id));
            }
        }
    }

    ir
}

fn lower_tensor_binding(
    ir: &mut IRModule,
    value: &ast::Node,
    dtype: &str,
    dims: &[String],
    env: &HashMap<String, ValueId>,
) -> ValueId {
    if let Some((dtype, shape)) = parse_tensor_ann(dtype, dims) {
        match value {
            ast::Node::Lit(Literal::Int(n), _) => {
                let id = ir.fresh();
                ir.instrs
                    .push(Instr::ConstTensor(id, dtype, shape, Some(*n as f64)));
                return id;
            }
            ast::Node::Lit(Literal::Ident(name), _) => {
                if let Some(id) = env.get(name) {
                    return *id;
                }
            }
            _ => {}
        }
    }

    lower_expr(value, ir, env)
}

fn lower_expr(node: &ast::Node, ir: &mut IRModule, env: &HashMap<String, ValueId>) -> ValueId {
    match node {
        ast::Node::Lit(Literal::Int(n), _) => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, *n));
            id
        }
        ast::Node::Lit(Literal::Ident(name), _) => env.get(name).copied().unwrap_or_else(|| {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }),
        ast::Node::Binary {
            op, left, right, ..
        } => {
            let lhs = lower_expr(left, ir, env);
            let rhs = lower_expr(right, ir, env);
            let dst = ir.fresh();
            let op = match op {
                ast::BinOp::Add => BinOp::Add,
                ast::BinOp::Sub => BinOp::Sub,
                ast::BinOp::Mul => BinOp::Mul,
                ast::BinOp::Div => BinOp::Div,
            };
            ir.instrs.push(Instr::BinOp { dst, op, lhs, rhs });
            dst
        }
        ast::Node::CallTensorSum {
            x, axes, keepdims, ..
        } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Sum {
                dst,
                src,
                axes,
                keepdims: *keepdims,
            });
            dst
        }
        ast::Node::CallTensorMean {
            x, axes, keepdims, ..
        } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Mean {
                dst,
                src,
                axes,
                keepdims: *keepdims,
            });
            dst
        }
        ast::Node::CallReshape { x, dims, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let new_shape = dims.iter().map(|dim| parse_dim(dim)).collect();
            ir.instrs.push(Instr::Reshape {
                dst,
                src,
                new_shape,
            });
            dst
        }
        ast::Node::CallExpandDims { x, axis, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::ExpandDims {
                dst,
                src,
                axis: *axis as i64,
            });
            dst
        }
        ast::Node::CallSqueeze { x, axes, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Squeeze { dst, src, axes });
            dst
        }
        ast::Node::CallTranspose { x, axes, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let perm = axes
                .as_ref()
                .map(|axes| axes.iter().map(|a| *a as i64).collect())
                .unwrap_or_default();
            ir.instrs.push(Instr::Transpose { dst, src, perm });
            dst
        }
        ast::Node::CallIndex { x, axis, i, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let indices = vec![IndexSpec {
                axis: (*axis).max(0) as i64,
                index: (*i).max(0) as i64,
            }];
            ir.instrs.push(Instr::Index { dst, src, indices });
            dst
        }
        ast::Node::CallMatMul { a, b, .. } => {
            let lhs = lower_expr(a, ir, env);
            let rhs = lower_expr(b, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::MatMul {
                dst,
                a: lhs,
                b: rhs,
            });
            dst
        }
        ast::Node::CallDot { a, b, .. } => {
            let lhs = lower_expr(a, ir, env);
            let rhs = lower_expr(b, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Dot {
                dst,
                a: lhs,
                b: rhs,
            });
            dst
        }
        ast::Node::CallSlice {
            x,
            axis,
            start,
            end,
            ..
        } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let dims = vec![SliceSpec {
                axis: (*axis).max(0) as i64,
                start: (*start).max(0) as i64,
                end: Some((*end).max(0) as i64),
                stride: 1,
            }];
            ir.instrs.push(Instr::Slice { dst, src, dims });
            dst
        }
        ast::Node::CallSliceStride {
            x,
            axis,
            start,
            end,
            step,
            ..
        } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let dims = vec![SliceSpec {
                axis: (*axis).max(0) as i64,
                start: (*start).max(0) as i64,
                end: Some((*end).max(0) as i64),
                stride: (*step).max(1) as i64,
            }];
            ir.instrs.push(Instr::Slice { dst, src, dims });
            dst
        }
        ast::Node::CallGather { x, axis, idx, .. } => {
            let src = lower_expr(x, ir, env);
            let indices = lower_expr(idx, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Gather {
                dst,
                src,
                indices,
                axis: (*axis).max(0) as i64,
            });
            dst
        }
        ast::Node::Paren(inner, _) => lower_expr(inner, ir, env),
        ast::Node::Tuple { elements, .. } => {
            let mut last = None;
            for element in elements {
                last = Some(lower_expr(element, ir, env));
            }
            last.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        _ => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
    }
}

fn parse_tensor_ann(dtype: &str, dims: &[String]) -> Option<(DType, Vec<ShapeDim>)> {
    let dtype = dtype.parse().ok()?;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        shape.push(parse_dim(dim));
    }
    Some((dtype, shape))
}

fn parse_dim(dim: &str) -> ShapeDim {
    if let Ok(n) = dim.parse::<usize>() {
        ShapeDim::Known(n)
    } else {
        let leaked: &'static str = Box::leak(dim.to_owned().into_boxed_str());
        ShapeDim::Sym(leaked)
    }
}
