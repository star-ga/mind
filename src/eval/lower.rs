use std::collections::HashMap;

use crate::ast::{self, Literal, TypeAnn};
use crate::ir::{BinOp, IRModule, Instr, ValueId};
use crate::types::{DType, ShapeDim};

pub fn lower_to_ir(module: &ast::Module) -> IRModule {
    let mut ir = IRModule::new();
    let mut env: HashMap<String, ValueId> = HashMap::new();

    for item in &module.items {
        match item {
            ast::Node::Let { name, ann, value, .. } => {
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
                ir.instrs.push(Instr::ConstTensor(id, dtype, shape, Some(*n as f64)));
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
        ast::Node::Binary { op, left, right, .. } => {
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
        ast::Node::CallTensorSum { x, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Sum { dst, src });
            dst
        }
        ast::Node::CallReshape { x, dims, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            let new_shape = dims.iter().map(parse_dim).collect();
            ir.instrs.push(Instr::Reshape { dst, src, new_shape });
            dst
        }
        ast::Node::CallMatMul { a, b, .. } => {
            let lhs = lower_expr(a, ir, env);
            let rhs = lower_expr(b, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::MatMul { dst, a: lhs, b: rhs });
            dst
        }
        ast::Node::CallSlice { x, axis, start, end, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Slice {
                dst,
                src,
                axis: (*axis).max(0) as usize,
                start: (*start).max(0) as usize,
                end: (*end).max(0) as usize,
                stride: 1,
            });
            dst
        }
        ast::Node::CallSliceStride { x, axis, start, end, step, .. } => {
            let src = lower_expr(x, ir, env);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Slice {
                dst,
                src,
                axis: (*axis).max(0) as usize,
                start: (*start).max(0) as usize,
                end: (*end).max(0) as usize,
                stride: (*step).max(1) as usize,
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
    let dtype = DType::from_str(dtype)?;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        shape.push(parse_dim(dim));
    }
    Some((dtype, shape))
}

fn parse_dim(dim: &String) -> ShapeDim {
    if let Ok(n) = dim.parse::<usize>() {
        ShapeDim::Known(n)
    } else {
        let leaked: &'static str = Box::leak(dim.clone().into_boxed_str());
        ShapeDim::Sym(leaked)
    }
}
