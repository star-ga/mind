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
            ast::Node::Export { names, .. } => {
                // RFC 0002, deliverable 1: lower the parsed `export { ... }`
                // block into `IRModule.exports`. Verification that named
                // functions exist lands in deliverable 2 under
                // `feature = "ffi-c-user"` together with the codegen pass.
                ir.exports.extend(names.iter().cloned());
            }
            // RFC 0005 P0e Step 1 — record the struct's field-name order in
            // the schema registry so a later `StructLit` can reorder
            // literal fields into canonical order before emitting stores.
            // The placeholder `Output(ConstI64(0))` is preserved to keep
            // the IR-shape contract that downstream consumers (verifier,
            // canonicaliser, MLIR emitter) rely on for declaration-only
            // modules — a struct declaration is still a no-op at the
            // value level, the side-table is pure metadata.
            #[cfg(feature = "std-surface")]
            ast::Node::StructDef { name, fields, .. } => {
                let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                ir.struct_defs.insert(name.clone(), field_names);
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
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
            ast::Node::Lit(Literal::Float(f), _) => {
                let id = ir.fresh();
                ir.instrs
                    .push(Instr::ConstTensor(id, dtype, shape, Some(*f)));
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
        ast::Node::Lit(Literal::Float(f), _) => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstF64(id, *f));
            id
        }
        ast::Node::Lit(Literal::Str(_), _) => {
            // Strings don't have IR representation yet; emit placeholder
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Lit(Literal::Ident(name), _) => env.get(name).copied().unwrap_or_else(|| {
            #[cfg(debug_assertions)]
            eprintln!("[WARN] lower_expr: undefined identifier `{name}`, defaulting to 0");
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
                ast::BinOp::Mod => BinOp::Mod,
                ast::BinOp::Lt => BinOp::Lt,
                ast::BinOp::Le => BinOp::Le,
                ast::BinOp::Gt => BinOp::Gt,
                ast::BinOp::Ge => BinOp::Ge,
                ast::BinOp::Eq => BinOp::Eq,
                ast::BinOp::Ne => BinOp::Ne,
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
        ast::Node::CallTensorRand { shape, .. } => {
            let dst = ir.fresh();
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            ir.instrs.push(Instr::ConstTensor(
                dst,
                crate::types::DType::F32,
                dims.iter()
                    .map(|s| crate::types::ShapeDim::Known(s.parse().unwrap()))
                    .collect(),
                None, // None = random fill, forces GPU materialization
            ));
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
        ast::Node::FnDef {
            name,
            params,
            body,
            reap_threshold,
            ..
        } => {
            // Lower function definition
            let mut fn_ir = IRModule::new();
            let mut fn_env = env.clone();

            // Create parameters
            let mut param_pairs = Vec::new();
            for (idx, param) in params.iter().enumerate() {
                let param_id = fn_ir.fresh();
                fn_ir.instrs.push(Instr::Param {
                    dst: param_id,
                    name: param.name.clone(),
                    index: idx,
                });
                fn_env.insert(param.name.clone(), param_id);
                param_pairs.push((param.name.clone(), param_id));
            }

            // Lower function body
            let mut ret_id = None;
            for stmt in body {
                match stmt {
                    ast::Node::Return { value, .. } => {
                        if let Some(val) = value {
                            ret_id = Some(lower_expr(val, &mut fn_ir, &fn_env));
                        }
                        fn_ir.instrs.push(Instr::Return { value: ret_id });
                    }
                    ast::Node::Let {
                        name, ann, value, ..
                    } => {
                        let id = match ann {
                            Some(TypeAnn::Tensor { dtype, dims })
                            | Some(TypeAnn::DiffTensor { dtype, dims }) => {
                                lower_tensor_binding(&mut fn_ir, value, dtype, dims, &fn_env)
                            }
                            _ => lower_expr(value, &mut fn_ir, &fn_env),
                        };
                        fn_env.insert(name.clone(), id);
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let id = lower_expr(value, &mut fn_ir, &fn_env);
                        fn_env.insert(name.clone(), id);
                    }
                    other => {
                        let id = lower_expr(other, &mut fn_ir, &fn_env);
                        ret_id = Some(id);
                    }
                }
            }

            // Add function definition to IR, propagating the REAP threshold
            // from the AST attribute if present.
            ir.instrs.push(Instr::FnDef {
                name: name.clone(),
                params: param_pairs,
                ret_id,
                body: fn_ir.instrs,
                reap_threshold: *reap_threshold,
            });

            // Function definitions don't produce a value
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Return { value, .. } => {
            let ret_val = value.as_ref().map(|v| lower_expr(v, ir, env));
            ir.instrs.push(Instr::Return { value: ret_val });
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Block { stmts, .. } => {
            let mut last_id = None;
            for stmt in stmts {
                last_id = Some(lower_expr(stmt, ir, env));
            }
            last_id.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        ast::Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            // Lower condition
            let _cond_id = lower_expr(cond, ir, env);
            // For now, just lower the then branch (control flow needs more work)
            let mut last_id = None;
            for stmt in then_branch {
                last_id = Some(lower_expr(stmt, ir, env));
            }
            if let Some(else_stmts) = else_branch {
                for stmt in else_stmts {
                    last_id = Some(lower_expr(stmt, ir, env));
                }
            }
            last_id.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        ast::Node::Call { callee, args, .. } => {
            let arg_ids: Vec<ValueId> = args.iter().map(|a| lower_expr(a, ir, env)).collect();
            let dst = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst,
                name: callee.clone(),
                args: arg_ids,
            });
            dst
        }
        // Phase 10.7: `match scrutinee { arms }` — lower to sequential
        // arm evaluation in v1 (chain-of-if-else semantics). The last
        // arm's value is the result; the runtime interprets patterns.
        ast::Node::Match {
            scrutinee, arms, ..
        } => {
            let _scrut_id = lower_expr(scrutinee, ir, env);
            let mut last_id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(last_id, 0));
            for arm in arms {
                last_id = lower_expr(&arm.body, ir, env);
            }
            last_id
        }
        // Phase 10.7: `&expr` / `&mut expr` — no-op metadata wrapper in
        // v1. The inner expression lowers directly; the ref tag is only
        // meaningful to the type-checker.
        ast::Node::Ref { inner, .. } => lower_expr(inner, ir, env),
        // RFC 0005 P0e Step 1 — `Foo { f1: v1, f2: v2, ... }` lowers to a
        // heap record. Layout = one `i64` slot per field, packed at
        // 8-byte stride. The struct value is the `i64` base address from
        // `__mind_alloc`; field reads are deferred to P0f (FieldAccess
        // needs the receiver's struct name threaded through env first).
        //
        //   addr = __mind_alloc(8 * N)
        //   __mind_store_i64(addr + 0,        v_for_field_0)
        //   __mind_store_i64(addr + 8,        v_for_field_1)
        //   ...
        //   addr            ← the struct's value
        //
        // Field order is canonical (from `StructDef`) — literals can
        // appear out of order and we reorder here. Unknown struct names
        // (no matching `StructDef` was lowered) fall through to literal
        // order so a forward-reference doesn't lose data.
        #[cfg(feature = "std-surface")]
        ast::Node::StructLit { name, fields, .. } => {
            // Canonical field order, if the schema is known.
            let canonical = ir.struct_defs.get(name).cloned();
            let order: Vec<&ast::StructLitField> = match canonical {
                Some(names) => names
                    .iter()
                    .filter_map(|fname| fields.iter().find(|f| &f.name == fname))
                    .collect(),
                None => fields.iter().collect(),
            };
            let n = order.len() as i64;

            // bytes = 8 * n  — emit two consts + a Mul rather than a
            // precomputed literal so the IR matches what a future
            // arbitrary-N codegen path will produce.
            let eight = ir.fresh();
            ir.instrs.push(Instr::ConstI64(eight, 8));
            let count = ir.fresh();
            ir.instrs.push(Instr::ConstI64(count, n));
            let bytes = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: bytes,
                op: BinOp::Mul,
                lhs: eight,
                rhs: count,
            });

            // addr = __mind_alloc(bytes)
            let addr = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: addr,
                name: "__mind_alloc".to_string(),
                args: vec![bytes],
            });

            // Per-field store at offset 8*i.
            for (i, f) in order.iter().enumerate() {
                let value = lower_expr(&f.value, ir, env);
                let field_addr = if i == 0 {
                    addr
                } else {
                    let offset = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(offset, (i as i64) * 8));
                    let sum = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: sum,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: offset,
                    });
                    sum
                };
                let store_ret = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: store_ret,
                    name: "__mind_store_i64".to_string(),
                    args: vec![field_addr, value],
                });
            }

            addr
        }
        _ => {
            #[cfg(debug_assertions)]
            eprintln!("[WARN] lower_expr: unhandled AST node kind, defaulting to 0");
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
        ShapeDim::Sym(crate::types::intern::intern_str(dim))
    }
}
