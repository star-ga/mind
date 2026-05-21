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
    // RFC 0005 P0f Step 1 — track `let x = Foo { ... }` so a later
    // `x.field` can resolve `Foo`'s canonical field-name order from
    // `ir.struct_defs` and emit the correct heap-record load offset.
    // Stays empty in non-std-surface builds; the FieldAccess arm and
    // the Let-side insert below are gated identically so the
    // side-table is dead-code-eliminated. `mut` is unused without the
    // feature, so silence the unused-mut lint instead of duplicating
    // the binding under a second cfg.
    #[allow(unused_mut)]
    let mut struct_env: HashMap<String, String> = HashMap::new();
    // RFC 0005 P0f Step 2 — module-wide side-table that maps every
    // `FieldAccess` span to its receiver's struct-type name. Built by
    // a single AST pre-pass so the FieldAccess arm in `lower_expr` can
    // resolve chained access (`a.b.c`), function-return receivers
    // (`foo().x`), and struct-typed parameters even when struct_env
    // doesn't have a direct Ident binding for the receiver. Multi-LLM
    // consensus on 2026-05-18 (grok-4.3 / glm-5.1 / mistral-large,
    // 3/3 unanimous) picked this "type-checker annotation" approach
    // over a post-lowering IR rewrite. The builder lives in
    // src/eval/struct_resolver.rs; in non-feature builds the table is
    // empty and never queried.
    #[cfg(feature = "std-surface")]
    let receiver_types_owned: HashMap<crate::ast::Span, String> =
        crate::eval::struct_resolver::build_field_access_types(module);
    #[cfg(not(feature = "std-surface"))]
    let receiver_types_owned: HashMap<crate::ast::Span, String> = HashMap::new();
    let receiver_types: &HashMap<crate::ast::Span, String> = &receiver_types_owned;

    // RFC 0010 Phase B fix: two-pass repr_c collection.
    // Pass 0: collect ALL #[repr(C)] struct field types before processing any
    // ExternBlock nodes.  A single top-to-bottom pass caused a declaration-order
    // hazard: structs defined after the extern block were not in repr_c_structs
    // when the ExternBlock lowering ran, causing silent fallback to "i64" for
    // mixed-type structs that should classify to MEMORY (!llvm.ptr).
    #[cfg(feature = "std-surface")]
    for item in &module.items {
        if let ast::Node::StructDef { name, fields, attrs, .. } = item {
            if attrs.iter().any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C")) {
                let field_types: Vec<crate::ast::TypeAnn> =
                    fields.iter().map(|f| f.ty.clone()).collect();
                ir.repr_c_structs.insert(name.clone(), field_types);
            }
        }
    }

    for item in &module.items {
        match item {
            ast::Node::Let {
                name, ann, value, ..
            } => {
                let id = match ann {
                    Some(TypeAnn::Tensor { dtype, dims }) => lower_tensor_binding(
                        &mut ir,
                        value,
                        dtype,
                        dims,
                        &env,
                        &struct_env,
                        receiver_types,
                    ),
                    _ => lower_expr(value, &mut ir, &env, &struct_env, receiver_types),
                };
                env.insert(name.clone(), id);
                // P0f Step 1: if the RHS is a StructLit, record the var→type
                // binding so a later FieldAccess on this name resolves the
                // correct offset out of `ir.struct_defs`.
                #[cfg(feature = "std-surface")]
                if let ast::Node::StructLit {
                    name: struct_name, ..
                } = value.as_ref()
                {
                    struct_env.insert(name.clone(), struct_name.clone());
                }
                ir.instrs.push(Instr::Output(id));
            }
            ast::Node::Assign { name, value, .. } => {
                let id = lower_expr(value, &mut ir, &env, &struct_env, receiver_types);
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
            ast::Node::StructDef { name, fields, attrs, .. } => {
                let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                ir.struct_defs.insert(name.clone(), field_names);
                // RFC 0010 Phase B: if the struct carries `#[repr(C)]`, register
                // its field types in `repr_c_structs` so extern_type_to_mlir can
                // classify Named types that appear in `extern "C"` signatures.
                let is_repr_c = attrs.iter().any(|a| {
                    a.name == "repr" && a.args.iter().any(|arg| arg == "C")
                });
                if is_repr_c {
                    let field_types: Vec<crate::ast::TypeAnn> =
                        fields.iter().map(|f| f.ty.clone()).collect();
                    ir.repr_c_structs.insert(name.clone(), field_types);
                }
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                ir.instrs.push(Instr::Output(id));
            }
            // RFC 0005 Phase 6.2b Gap 2 — module-level `const NAME: [i64; N] = [...]`.
            // Lowers to a named ConstArray IR node and also registers the
            // element data in `ir.const_array_defs` so that fn bodies (which
            // use a fresh SSA namespace) can re-emit the blob on demand.
            #[cfg(feature = "std-surface")]
            ast::Node::Const {
                name,
                ty: Some(TypeAnn::Array { .. }),
                value,
                ..
            } => {
                let values = extract_array_lit_values(value);
                ir.const_array_defs.insert(name.clone(), values.clone());
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstArray {
                    dst: id,
                    name: Some(name.clone()),
                    values,
                });
                env.insert(name.clone(), id);
                ir.instrs.push(Instr::Output(id));
            }
            other => {
                let id = lower_expr(other, &mut ir, &env, &struct_env, receiver_types);
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
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
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
            // Negated literal tensor fill (`let t: f32[4] = -1.0`). Without
            // this, the negative fill value fell through to `lower_expr` and
            // lost its tensor shape; fold the sign into the fill scalar.
            ast::Node::Neg { operand, .. } => match operand.as_ref() {
                ast::Node::Lit(Literal::Int(n), _) => {
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstTensor(
                        id,
                        dtype,
                        shape,
                        Some(n.wrapping_neg() as f64),
                    ));
                    return id;
                }
                ast::Node::Lit(Literal::Float(f), _) => {
                    let id = ir.fresh();
                    ir.instrs
                        .push(Instr::ConstTensor(id, dtype, shape, Some(-*f)));
                    return id;
                }
                _ => {}
            },
            _ => {}
        }
    }

    lower_expr(value, ir, env, struct_env, receiver_types)
}

fn lower_expr(
    node: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    // RFC 0005 P0f Step 1 — per-fn binding from variable name to its
    // struct-type name. Populated at Let sites whose RHS is a
    // `StructLit`; consumed by the FieldAccess read-path arm below
    // to look up the canonical field-name list from `ir.struct_defs`
    // and emit `__mind_load_i64` at the correct 8-byte offset.
    struct_env: &HashMap<String, String>,
    // RFC 0005 P0f Step 2 — module-wide side-table keyed on each
    // FieldAccess span, mapping to the receiver's struct-type name.
    // Built once per `lower_to_ir` call by `struct_resolver`. Lets
    // the FieldAccess arm resolve chained access (`a.b.c`), fn
    // returns (`foo().x`), and struct-typed parameters that Step 1
    // can't see via a direct `Ident` lookup.
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> ValueId {
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
        // Unary negation `-expr`. Without this arm a bare negative literal
        // (`-65536`) — or any unary minus — fell through to the catch-all
        // `_ =>` and was silently lowered to `const.i64 0`. `-N` must be
        // identical to `(0 - N)` for every i64 N. Literal operands fold to
        // a single negated constant; runtime operands lower as `0 - operand`
        // so the type-driven IR→MLIR path picks `arith.subi`/`arith.subf`
        // exactly as the binary-subtraction source form already does.
        ast::Node::Neg { operand, .. } => match operand.as_ref() {
            ast::Node::Lit(Literal::Int(n), _) => {
                let id = ir.fresh();
                // `wrapping_neg` keeps INT64_MIN well-defined: `-INT64_MIN`
                // wraps back to INT64_MIN, matching two's-complement
                // `0 - INT64_MIN` via `arith.subi`.
                ir.instrs.push(Instr::ConstI64(id, n.wrapping_neg()));
                id
            }
            ast::Node::Lit(Literal::Float(f), _) => {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstF64(id, -*f));
                id
            }
            _ => {
                let zero = ir.fresh();
                ir.instrs.push(Instr::ConstI64(zero, 0));
                let rhs = lower_expr(operand, ir, env, struct_env, receiver_types);
                let dst = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst,
                    op: BinOp::Sub,
                    lhs: zero,
                    rhs,
                });
                dst
            }
        },
        ast::Node::Lit(Literal::Str(_), _) => {
            // Strings don't have IR representation yet; emit placeholder
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Lit(Literal::Ident(name), _) => {
            // Fast path: SSA binding from env (params, let-bindings).
            if let Some(id) = env.get(name).copied() {
                return id;
            }
            // Phase 6.2b Gap 2: const-array identifier — re-emit the
            // ConstArray blob into the current IR (fn body or module level)
            // so the ArrayLoad that follows has a valid base in this
            // IR's SSA namespace.
            #[cfg(feature = "std-surface")]
            if let Some(values) = ir.const_array_defs.get(name).cloned() {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstArray {
                    dst: id,
                    name: Some(name.clone()),
                    values,
                });
                return id;
            }
            // Undefined — emit placeholder.
            #[cfg(debug_assertions)]
            eprintln!("[WARN] lower_expr: undefined identifier `{name}`, defaulting to 0");
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Binary {
            op, left, right, ..
        } => {
            let lhs = lower_expr(left, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(right, ir, env, struct_env, receiver_types);
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
        // Phase 6.5 Stage 1a — bitwise binary operators.
        // `ast::Node::Bitwise` is kept separate from `Node::Binary` by design
        // (see ast/mod.rs comments). Map each BitOp to its IR BinOp variant.
        // Gated to `std-surface`.
        #[cfg(feature = "std-surface")]
        ast::Node::Bitwise {
            op, left, right, ..
        } => {
            let lhs = lower_expr(left, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(right, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let ir_op = match op {
                ast::BitOp::And => BinOp::BitAnd,
                ast::BitOp::Or => BinOp::BitOr,
                ast::BitOp::Xor => BinOp::BitXor,
                ast::BitOp::Shl => BinOp::Shl,
                ast::BitOp::Shr => BinOp::Shr,
            };
            ir.instrs.push(Instr::BinOp {
                dst,
                op: ir_op,
                lhs,
                rhs,
            });
            dst
        }
        ast::Node::CallTensorSum {
            x, axes, keepdims, ..
        } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
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
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
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
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
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
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::ExpandDims {
                dst,
                src,
                axis: *axis as i64,
            });
            dst
        }
        ast::Node::CallSqueeze { x, axes, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let axes = axes.iter().map(|a| *a as i64).collect();
            ir.instrs.push(Instr::Squeeze { dst, src, axes });
            dst
        }
        ast::Node::CallTranspose { x, axes, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let perm = axes
                .as_ref()
                .map(|axes| axes.iter().map(|a| *a as i64).collect())
                .unwrap_or_default();
            ir.instrs.push(Instr::Transpose { dst, src, perm });
            dst
        }
        ast::Node::CallIndex { x, axis, i, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let indices = vec![IndexSpec {
                axis: (*axis).max(0) as i64,
                index: (*i).max(0) as i64,
            }];
            ir.instrs.push(Instr::Index { dst, src, indices });
            dst
        }
        ast::Node::CallMatMul { a, b, .. } => {
            let lhs = lower_expr(a, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(b, ir, env, struct_env, receiver_types);
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
            let lhs = lower_expr(a, ir, env, struct_env, receiver_types);
            let rhs = lower_expr(b, ir, env, struct_env, receiver_types);
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
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
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
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
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
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let indices = lower_expr(idx, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Gather {
                dst,
                src,
                indices,
                axis: (*axis).max(0) as i64,
            });
            dst
        }
        ast::Node::Paren(inner, _) => lower_expr(inner, ir, env, struct_env, receiver_types),
        ast::Node::Tuple { elements, .. } => {
            let mut last = None;
            for element in elements {
                last = Some(lower_expr(element, ir, env, struct_env, receiver_types));
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
            // RFC 0005 P0f Step 1 — the FieldAccess read-path resolves
            // a field offset via `fn_ir.struct_defs[T]`; without
            // inheriting the parent module's schema registry, every
            // struct used inside a fn body would silently fall through
            // to the placeholder. Schema is metadata only — cloning
            // does not duplicate any IR instructions and is gated to
            // std-surface so non-feature builds incur zero cost.
            #[cfg(feature = "std-surface")]
            {
                fn_ir.struct_defs = ir.struct_defs.clone();
                // Phase 6.2b Gap 2: inherit const-array data so that
                // fn bodies can re-emit ConstArray nodes on demand.
                fn_ir.const_array_defs = ir.const_array_defs.clone();
            }
            // Build fn_env from env, but do NOT carry over const-array
            // SSA ids from the outer module — those ids are only valid in
            // the outer ir's SSA namespace.  Const-array identifiers will
            // be re-resolved in the Ident arm below via const_array_defs.
            let mut fn_env: HashMap<String, ValueId> = env
                .iter()
                .filter(|(name, _)| {
                    #[cfg(feature = "std-surface")]
                    {
                        !ir.const_array_defs.contains_key(*name)
                    }
                    #[cfg(not(feature = "std-surface"))]
                    {
                        // `name` is only consulted under `std-surface`
                        // (const-array shadowing); touch it here so the
                        // binding isn't flagged unused in the default build.
                        let _ = name;
                        true
                    }
                })
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            // RFC 0005 P0f Step 1 — fresh per-fn struct binding map.
            // Inherits outer module-scope bindings (so an outer
            // `let cfg = Config { ... }` is visible to inner field
            // reads) but additions inside this fn body do not leak
            // back out to siblings or to module scope. `mut` is
            // unused without std-surface; silence the lint here too.
            #[allow(unused_mut)]
            let mut fn_struct_env = struct_env.clone();

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
                            ret_id = Some(lower_expr(
                                val,
                                &mut fn_ir,
                                &fn_env,
                                &fn_struct_env,
                                receiver_types,
                            ));
                        }
                        fn_ir.instrs.push(Instr::Return { value: ret_id });
                    }
                    ast::Node::Let {
                        name, ann, value, ..
                    } => {
                        let id = match ann {
                            Some(TypeAnn::Tensor { dtype, dims })
                            | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                                &mut fn_ir,
                                value,
                                dtype,
                                dims,
                                &fn_env,
                                &fn_struct_env,
                                receiver_types,
                            ),
                            _ => lower_expr(
                                value,
                                &mut fn_ir,
                                &fn_env,
                                &fn_struct_env,
                                receiver_types,
                            ),
                        };
                        fn_env.insert(name.clone(), id);
                        // P0f Step 1: track fn-scoped var→struct binding for
                        // FieldAccess inside this fn body.
                        #[cfg(feature = "std-surface")]
                        if let ast::Node::StructLit {
                            name: struct_name, ..
                        } = value.as_ref()
                        {
                            fn_struct_env.insert(name.clone(), struct_name.clone());
                        }
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let id =
                            lower_expr(value, &mut fn_ir, &fn_env, &fn_struct_env, receiver_types);
                        fn_env.insert(name.clone(), id);
                    }
                    other => {
                        let id =
                            lower_expr(other, &mut fn_ir, &fn_env, &fn_struct_env, receiver_types);
                        ret_id = Some(id);
                        // Gap C: if the emitted statement was an `Instr::If`,
                        // thread its branch_bindings back into `fn_env` so
                        // subsequent statements in this fn body can reference
                        // let bindings declared inside either branch.
                        #[cfg(feature = "std-surface")]
                        if let Some(Instr::If {
                            branch_bindings, ..
                        }) = fn_ir.instrs.last()
                        {
                            for (bname, bid) in branch_bindings.clone() {
                                fn_env.insert(bname, bid);
                            }
                        }
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
            let ret_val = value
                .as_ref()
                .map(|v| lower_expr(v, ir, env, struct_env, receiver_types));
            ir.instrs.push(Instr::Return { value: ret_val });
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        ast::Node::Block { stmts, .. } => {
            let mut last_id = None;
            for stmt in stmts {
                last_id = Some(lower_expr(stmt, ir, env, struct_env, receiver_types));
            }
            last_id.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        // Phase 6.5 Stage 1a — `if cond { then } else { else }` lowering.
        //
        // The condition, then-branch, and else-branch each lower into separate
        // sub-IRModule scratch buffers so that `Instr::Return` nodes inside a
        // branch do not appear as mid-block terminators in the parent flat
        // instruction stream. The MLIR lowerer converts `Instr::If` into an
        // `scf.if` or a `cf.cond_br`+basic-block structure, placing each
        // branch's instructions in its own MLIR basic block.
        //
        // Gap C: `let` bindings produced inside either branch are collected in
        // `branch_bindings` and re-inserted into the outer `env` after the
        // `Instr::If` is emitted so subsequent statements in the same scope can
        // reference them. This replicates the pattern `Instr::While` uses for
        // `live_vars`.
        //
        // Gated to `std-surface`.
        #[cfg(feature = "std-surface")]
        ast::Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            // ── 1. Lower the condition into a scratch sub-module ──────────────
            //
            // Sub-modules must inherit `struct_defs` and `const_array_defs`
            // from the parent IR so that `FieldAccess` and const-array
            // references inside branch conditions and bodies resolve correctly.
            //
            // Chain `next_id`: cond_ir starts at ir.next_id, then_ir starts
            // at cond_ir.next_id, else_ir starts at then_ir.next_id. This
            // ensures all ValueIds across all three sub-modules are globally
            // unique and disjoint from the parent scope's ids (especially fn
            // parameters which occupy the lowest ids).
            let mut cond_ir = sub_ir_from(ir);
            let cond_env = env.clone();
            let cond_id = lower_expr(cond, &mut cond_ir, &cond_env, struct_env, receiver_types);

            // ── 2. Lower the then-branch into a scratch sub-module ────────────
            //      Starts from cond_ir's highest id.
            let mut then_ir = sub_ir_from_after(&cond_ir, ir);
            let mut then_env = env.clone();
            let mut branch_bindings: Vec<(String, ValueId)> = Vec::new();
            let mut then_result = then_ir.fresh();
            then_ir.instrs.push(Instr::ConstI64(then_result, 0));
            for stmt in then_branch {
                match stmt {
                    ast::Node::Return { value, .. } => {
                        let ret_val = value.as_ref().map(|v| {
                            lower_expr(v, &mut then_ir, &then_env, struct_env, receiver_types)
                        });
                        then_ir.instrs.push(Instr::Return { value: ret_val });
                        if let Some(rv) = ret_val {
                            then_result = rv;
                        }
                    }
                    ast::Node::Let {
                        name, ann, value, ..
                    } => {
                        let id = match ann {
                            Some(TypeAnn::Tensor { dtype, dims })
                            | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                                &mut then_ir,
                                value,
                                dtype,
                                dims,
                                &then_env,
                                struct_env,
                                receiver_types,
                            ),
                            _ => lower_expr(
                                value,
                                &mut then_ir,
                                &then_env,
                                struct_env,
                                receiver_types,
                            ),
                        };
                        then_env.insert(name.clone(), id);
                        // Gap C: record binding so it is visible after the if.
                        if let Some(pos) = branch_bindings.iter().position(|(n, _)| n == name) {
                            branch_bindings[pos].1 = id;
                        } else {
                            branch_bindings.push((name.clone(), id));
                        }
                        then_result = id;
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let id =
                            lower_expr(value, &mut then_ir, &then_env, struct_env, receiver_types);
                        then_env.insert(name.clone(), id);
                        then_result = id;
                    }
                    other => {
                        then_result =
                            lower_expr(other, &mut then_ir, &then_env, struct_env, receiver_types);
                    }
                }
            }

            // ── 3. Lower the else-branch (or synthesise a unit zero) ──────────
            //      Starts from then_ir's highest id.
            let mut else_ir = sub_ir_from_after(&then_ir, ir);
            let mut else_env = env.clone();
            let mut else_result = else_ir.fresh();
            else_ir.instrs.push(Instr::ConstI64(else_result, 0));
            if let Some(else_stmts) = else_branch {
                for stmt in else_stmts {
                    match stmt {
                        ast::Node::Return { value, .. } => {
                            let ret_val = value.as_ref().map(|v| {
                                lower_expr(v, &mut else_ir, &else_env, struct_env, receiver_types)
                            });
                            else_ir.instrs.push(Instr::Return { value: ret_val });
                            if let Some(rv) = ret_val {
                                else_result = rv;
                            }
                        }
                        ast::Node::Let {
                            name, ann, value, ..
                        } => {
                            let id = match ann {
                                Some(TypeAnn::Tensor { dtype, dims })
                                | Some(TypeAnn::DiffTensor { dtype, dims }) => {
                                    lower_tensor_binding(
                                        &mut else_ir,
                                        value,
                                        dtype,
                                        dims,
                                        &else_env,
                                        struct_env,
                                        receiver_types,
                                    )
                                }
                                _ => lower_expr(
                                    value,
                                    &mut else_ir,
                                    &else_env,
                                    struct_env,
                                    receiver_types,
                                ),
                            };
                            else_env.insert(name.clone(), id);
                            if let Some(pos) = branch_bindings.iter().position(|(n, _)| n == name) {
                                branch_bindings[pos].1 = id;
                            } else {
                                branch_bindings.push((name.clone(), id));
                            }
                            else_result = id;
                        }
                        ast::Node::Assign { name, value, .. } => {
                            let id = lower_expr(
                                value,
                                &mut else_ir,
                                &else_env,
                                struct_env,
                                receiver_types,
                            );
                            else_env.insert(name.clone(), id);
                            else_result = id;
                        }
                        other => {
                            else_result = lower_expr(
                                other,
                                &mut else_ir,
                                &else_env,
                                struct_env,
                                receiver_types,
                            );
                        }
                    }
                }
            }

            // ── 4. Emit Instr::If into the parent IR stream ───────────────────
            // Advance the parent's SSA counter to after all sub-module ids.
            ir.next_id = ir.next_id.max(else_ir.next_id);
            let dst = ir.fresh();
            ir.instrs.push(Instr::If {
                cond_id,
                cond_instrs: cond_ir.instrs,
                then_instrs: then_ir.instrs,
                then_result,
                else_instrs: else_ir.instrs,
                else_result,
                dst,
                branch_bindings,
            });

            // Gap C: branch_bindings are stored on the Instr::If node so
            // callers that own a mutable env (e.g. the fn-body loop below)
            // can thread them back after the if.  `lower_expr` takes `env`
            // as a shared reference and cannot mutate the outer scope here.

            dst
        }
        // Non-gated fallback for `ast::Node::If` when `std-surface` is off.
        // Retains the old sequential-flatten behaviour so the default build
        // compiles and the existing `if_expr` tests continue to pass.
        #[cfg(not(feature = "std-surface"))]
        ast::Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let _cond_id = lower_expr(cond, ir, env, struct_env, receiver_types);
            let mut last_id = None;
            for stmt in then_branch {
                last_id = Some(lower_expr(stmt, ir, env, struct_env, receiver_types));
            }
            if let Some(else_stmts) = else_branch {
                for stmt in else_stmts {
                    last_id = Some(lower_expr(stmt, ir, env, struct_env, receiver_types));
                }
            }
            last_id.unwrap_or_else(|| {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                id
            })
        }
        ast::Node::Call { callee, args, .. } => {
            let arg_ids: Vec<ValueId> = args
                .iter()
                .map(|a| lower_expr(a, ir, env, struct_env, receiver_types))
                .collect();
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
            let _scrut_id = lower_expr(scrutinee, ir, env, struct_env, receiver_types);
            let mut last_id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(last_id, 0));
            for arm in arms {
                last_id = lower_expr(&arm.body, ir, env, struct_env, receiver_types);
            }
            last_id
        }
        // Phase 10.7: `&expr` / `&mut expr` — no-op metadata wrapper in
        // v1. The inner expression lowers directly; the ref tag is only
        // meaningful to the type-checker.
        ast::Node::Ref { inner, .. } => lower_expr(inner, ir, env, struct_env, receiver_types),
        // RFC 0005 Gap 1: `while cond { body }` lowering.
        //
        // The condition and body each lower into their own sub-modules so
        // the MLIR stage can place them in separate basic blocks (header and
        // body blocks, respectively).  Mutable variables that are written in
        // the body are collected as `live_vars` and threaded as block
        // arguments in the MLIR lowering.
        //
        // Gated to `std-surface` — default builds never reach this arm.
        #[cfg(feature = "std-surface")]
        ast::Node::While { cond, body, .. } => {
            // Lower the condition expression into a scratch sub-module to
            // capture the instructions that produce it without polluting the
            // parent IR stream.  The resulting ValueIds are local to the
            // sub-module; MLIR lowering re-emits them verbatim in the header
            // block so the numbering is stable.
            let mut cond_ir = IRModule::new();
            // Seed the condition sub-module's env with the current bindings
            // so identifiers in the condition (e.g. `i`, `n`) resolve.
            let cond_env = env.clone();
            let cond_id = lower_expr(cond, &mut cond_ir, &cond_env, struct_env, receiver_types);

            // Lower the body into a scratch sub-module.  Track every Assign
            // target — those are the variables that are live across the
            // back-edge and must become block arguments in MLIR.
            let mut body_ir = IRModule::new();
            let mut body_env = env.clone();
            let mut mutated: Vec<(String, ValueId)> = Vec::new();

            for stmt in body {
                match stmt {
                    ast::Node::Assign { name, value, .. } => {
                        let new_id =
                            lower_expr(value, &mut body_ir, &body_env, struct_env, receiver_types);
                        body_env.insert(name.clone(), new_id);
                        // Record the variable and its post-body value.
                        if let Some(pos) = mutated.iter().position(|(n, _)| n == name) {
                            mutated[pos].1 = new_id;
                        } else {
                            mutated.push((name.clone(), new_id));
                        }
                    }
                    other => {
                        lower_expr(other, &mut body_ir, &body_env, struct_env, receiver_types);
                    }
                }
            }

            // After the loop, the parent env sees the post-body bindings of
            // mutated variables so that code after the while statement uses
            // the correct SSA ids.  Gap 1 full implementation will thread
            // them via block arguments; the `env.insert` is deferred until
            // `lower_expr` accepts `&mut env` (a Gap 1 follow-on).

            ir.instrs.push(Instr::While {
                cond_id,
                cond_instrs: cond_ir.instrs,
                body: body_ir.instrs,
                live_vars: mutated,
            });

            // `while` is a statement; produce a unit i64 placeholder.
            let unit = ir.fresh();
            ir.instrs.push(Instr::ConstI64(unit, 0));
            unit
        }
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
                let value = lower_expr(&f.value, ir, env, struct_env, receiver_types);
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
        // RFC 0005 P0f — `receiver.field` reads from the heap record
        // produced by P0e StructLit lowering.
        //
        //   offset      = index_of(field, struct_defs[T]) * 8
        //   field_addr  = addr + offset    (or addr itself when offset == 0)
        //   result      = __mind_load_i64(field_addr)
        //
        // Step 1 — fast path: receiver is a plain `Ident` bound to a
        // `StructLit` via `Let` in this (or an enclosing) scope. The
        // receiver's struct name lives in `struct_env[var_name]`;
        // we look it up without re-lowering the receiver.
        //
        // Step 2 — general path: receiver type is precomputed by the
        // pre-pass in `src/eval/struct_resolver.rs` and stored in the
        // `receiver_types` side-table keyed on this FieldAccess's
        // span. Covers chained access (`a.b.c`), function-return
        // receivers (`foo().x`), and struct-typed parameters. We
        // lower the receiver expression for its base-address value
        // and then add the field's 8-byte offset as before.
        //
        // Unresolved receivers fall through to a `ConstI64(0)`
        // placeholder so the IR shape is stable and older modules
        // still compile.
        #[cfg(feature = "std-surface")]
        ast::Node::FieldAccess {
            receiver,
            field,
            span,
        } => {
            // ── Step 1: cheap Ident-bound lookup ─────────────────────
            let step1 = match receiver.as_ref() {
                ast::Node::Lit(Literal::Ident(var_name), _) => {
                    struct_env.get(var_name).and_then(|struct_name| {
                        ir.struct_defs
                            .get(struct_name)
                            .and_then(|fields| fields.iter().position(|f| f == field))
                            .map(|idx| (Some(var_name.clone()), idx))
                    })
                }
                _ => None,
            };
            // ── Step 2: side-table fallback (general path) ───────────
            // Only consulted when Step 1 fast-path failed.
            let step2 = if step1.is_none() {
                receiver_types.get(span).and_then(|struct_name| {
                    ir.struct_defs
                        .get(struct_name)
                        .and_then(|fields| fields.iter().position(|f| f == field))
                        .map(|idx| (None::<String>, idx))
                })
            } else {
                None
            };

            let resolved = step1.or(step2);

            match resolved {
                Some((var_name_opt, idx)) => {
                    // Step 1 path can take addr from env without re-lowering.
                    // Step 2 path must lower the receiver expression to
                    // get its base address (it may be a Call, FieldAccess,
                    // or anything else that evaluates to an i64 heap addr).
                    let addr = match var_name_opt {
                        Some(var_name) => match env.get(&var_name) {
                            Some(id) => *id,
                            None => {
                                let id = ir.fresh();
                                ir.instrs.push(Instr::ConstI64(id, 0));
                                return id;
                            }
                        },
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                    };
                    let field_addr = if idx == 0 {
                        addr
                    } else {
                        let offset = ir.fresh();
                        ir.instrs.push(Instr::ConstI64(offset, (idx as i64) * 8));
                        let sum = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sum,
                            op: BinOp::Add,
                            lhs: addr,
                            rhs: offset,
                        });
                        sum
                    };
                    let result = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: result,
                        name: "__mind_load_i64".to_string(),
                        args: vec![field_addr],
                    });
                    result
                }
                None => {
                    // Receiver type still unresolvable even after the
                    // side-table — emit placeholder so the module
                    // produces a stable IR shape. Step 3 will lift the
                    // remaining cases (heap-allocated fields of struct
                    // type, generics) when std.vec needs them.
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(id, 0));
                    id
                }
            }
        }
        // RFC 0005 Phase 6.2b Gap 2 — anonymous array literal `[v0, v1, …]`
        // in expression position.  Elements are extracted iteratively
        // (not by recursing once per element) so a 4,096-entry literal
        // does not grow the Rust call-stack linearly.
        #[cfg(feature = "std-surface")]
        ast::Node::ArrayLit { elements, .. } => {
            let values: Vec<i64> = elements
                .iter()
                .map(|e| extract_const_i64(e).unwrap_or(0))
                .collect();
            let dst = ir.fresh();
            ir.instrs.push(Instr::ConstArray {
                dst,
                name: None,
                values,
            });
            dst
        }
        // RFC 0005 Phase 6.2b Gap 2 — `receiver[index]`.  When the receiver
        // resolves to a ConstArray base address, this emits `ArrayLoad`.
        #[cfg(feature = "std-surface")]
        ast::Node::IndexAccess {
            receiver, index, ..
        } => {
            let base = lower_expr(receiver, ir, env, struct_env, receiver_types);
            let index_id = lower_expr(index, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::ArrayLoad {
                dst,
                base,
                index: index_id,
            });
            dst
        }
        // RFC 0005 Phase 6.2b Gap 2 — `receiver[index] = value` on arrays.
        // Const arrays are read-only in this phase; emit a placeholder to
        // keep the IR shape stable.
        #[cfg(feature = "std-surface")]
        ast::Node::IndexAssign { .. } => {
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
        }
        // RFC 0010 Phase A — `extern "C" { fn decls }` block.
        //
        // Emits one `Instr::ExternFnDecl` per declared symbol so the MLIR
        // lowerer knows to emit `llvm.func @name(...)` declarations and to
        // use `llvm.call` (not `func.call`) for calls to those names.
        //
        // Phase A lowers all integer/pointer parameter types to `i64` and
        // f64 to `f64`; raw pointer types lower to `i64` (opaque address
        // under the Option-C ABI convention already in use across the
        // std-surface runtime bridge).
        //
        // Gated to `std-surface` — default builds never construct this.
        #[cfg(feature = "std-surface")]
        ast::Node::ExternBlock { fns, callconv, .. } => {
            // RFC 0010 Phase B/C: use the repr_c_structs registry (populated by
            // any preceding StructDef nodes with `#[repr(C)]`) to emit correct
            // ABI-classified types for struct-valued parameters.
            // Phase C: dispatch to Win64 or SysV classifier based on callconv.
            let repr_c_snapshot = ir.repr_c_structs.clone();
            let effective_callconv = resolve_callconv(*callconv);
            for efn in fns {
                let param_types: Vec<String> = efn
                    .params
                    .iter()
                    .flat_map(|p| extern_type_to_mlir_multi_for(
                        &p.ty, &repr_c_snapshot, effective_callconv,
                    ))
                    .collect();
                let ret_type = efn.ret_type.as_ref().map(|t| {
                    // Return types: structs >8B returned via hidden pointer;
                    // use first ABI slot as the declared return type (single register).
                    extern_type_to_mlir_multi_for(t, &repr_c_snapshot, effective_callconv)
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| "i64".to_string())
                });
                ir.instrs.push(Instr::ExternFnDecl {
                    name: efn.name.clone(),
                    param_types,
                    ret_type,
                    is_varargs: efn.is_varargs,
                    vararg_hints: Vec::new(),
                    callconv: effective_callconv,
                });
            }
            let id = ir.fresh();
            ir.instrs.push(Instr::ConstI64(id, 0));
            id
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

/// Extract a compile-time i64 value from a literal expression node.
/// Returns `None` for non-literal (runtime) expressions.
#[cfg(feature = "std-surface")]
fn extract_const_i64(node: &ast::Node) -> Option<i64> {
    match node {
        ast::Node::Lit(Literal::Int(n), _) => Some(*n),
        ast::Node::Neg { operand, .. } => extract_const_i64(operand).map(|v| -v),
        _ => None,
    }
}

/// Extract the element value list from an `ArrayLit` node iteratively.
/// Non-literal elements default to 0.  Returns an empty Vec for non-ArrayLit RHS.
#[cfg(feature = "std-surface")]
fn extract_array_lit_values(node: &ast::Node) -> Vec<i64> {
    match node {
        ast::Node::ArrayLit { elements, .. } => {
            let mut out = Vec::with_capacity(elements.len());
            for elem in elements {
                out.push(extract_const_i64(elem).unwrap_or(0));
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Create a sub-IRModule for branch/body lowering that inherits the metadata
/// tables from `parent` (struct schemas, const-array data) and starts its SSA
/// counter at `parent.next_id`.
///
/// Starting at the parent's current `next_id` ensures that every ValueId
/// allocated inside the sub-module is disjoint from all ValueIds already
/// visible in the enclosing function scope (parameters, outer lets, etc.).
/// Without this, a constant emitted in a condition sub-IR (e.g.
/// `ConstI64(ValueId(0), 32)`) would collide with the function's first
/// parameter (`%0: i64`) when both are serialised into the same MLIR block.
///
/// Used by `Instr::If` lowering and any future control-flow arms that lower
/// branches into separate scratch IRModules.
#[cfg(feature = "std-surface")]
fn sub_ir_from(parent: &IRModule) -> IRModule {
    let mut m = IRModule::new();
    m.next_id = parent.next_id;
    m.struct_defs = parent.struct_defs.clone();
    m.const_array_defs = parent.const_array_defs.clone();
    m
}

/// Like `sub_ir_from`, but chains the SSA counter from `prev` (the previously
/// built sub-module) so that each successive sub-module's ids are disjoint from
/// all predecessors.  Metadata is still copied from `meta_src` (the original
/// parent scope).
#[cfg(feature = "std-surface")]
fn sub_ir_from_after(prev: &IRModule, meta_src: &IRModule) -> IRModule {
    let mut m = IRModule::new();
    m.next_id = prev.next_id;
    m.struct_defs = meta_src.struct_defs.clone();
    m.const_array_defs = meta_src.const_array_defs.clone();
    m
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

/// RFC 0010 Phase A/B — map an `extern "C"` parameter/return `TypeAnn` to
/// the MLIR type string(s) used in `llvm.func` declarations and `llvm.call`
/// ops.
///
/// Returns a `Vec<String>` of MLIR type tokens because a single MIND type
/// can expand to multiple MLIR types under the SysV x86_64 ABI (e.g. a
/// 16-byte all-integer `#[repr(C)]` struct expands to two `i64` parameters).
/// For all non-struct types the Vec always has exactly one element.
///
/// `repr_c` is the `repr_c_structs` registry from `IRModule` — a map from
/// struct name to field types. Pass an empty map when no struct types are
/// expected (Phase A callers).
#[cfg(feature = "std-surface")]
pub(crate) fn extern_type_to_mlir_multi(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarF32 => vec!["f32".to_string()],
        TypeAnn::ScalarF64 => vec!["f64".to_string()],
        TypeAnn::RawPtr { .. } => vec!["!llvm.ptr".to_string()],
        // RFC 0010 Phase B: callback function pointer -> opaque !llvm.ptr.
        TypeAnn::FnPtr { .. } => vec!["!llvm.ptr".to_string()],
        TypeAnn::Named(name) => {
            match name.as_str() {
                "f32" => return vec!["f32".to_string()],
                "f64" => return vec!["f64".to_string()],
                "i8" | "i16" | "i32" | "u8" | "u16" | "u32" | "bool" => {
                    return vec!["i64".to_string()]
                }
                "i64" | "u64" | "usize" | "isize" => return vec!["i64".to_string()],
                _ => {}
            }
            // Check for repr(C) struct — apply SysV classification.
            if let Some(fields) = repr_c.get(name.as_str()) {
                sysv_classify_struct(fields, repr_c)
            } else {
                vec!["i64".to_string()]
            }
        }
        // Built-in scalar integer/bool types all lower to i64.
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarBool
        | TypeAnn::ScalarU32 => vec!["i64".to_string()],
        // Fallback: any aggregate that slipped past the type-checker becomes i64.
        _ => vec!["i64".to_string()],
    }
}

/// RFC 0010 Phase A compatibility shim — single-type version of
/// `extern_type_to_mlir_multi`. Used by Phase A callers. Struct types are
/// passed by pointer (!llvm.ptr) when the registry is empty.
#[cfg(feature = "std-surface")]
#[allow(dead_code)]
pub(crate) fn extern_type_to_mlir(ty: &crate::ast::TypeAnn) -> String {
    let empty = std::collections::BTreeMap::new();
    extern_type_to_mlir_multi(ty, &empty)
        .into_iter()
        .next()
        .unwrap_or_else(|| "i64".to_string())
}

/// RFC 0010 Phase B — System V AMD64 ABI struct field classification.
/// Classifies a scalar type into Integer or Float class and returns its byte size.
/// Returns `(None, 0)` for types that cannot be classified (nested aggregates, etc.).
#[cfg(feature = "std-surface")]
pub fn classify_scalar_field(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> (Option<SysVClass>, usize) {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarI32 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            (Some(SysVClass::Integer), 4)
        }
        TypeAnn::ScalarI64 => (Some(SysVClass::Integer), 8),
        TypeAnn::ScalarF32 => (Some(SysVClass::Float), 4),
        TypeAnn::ScalarF64 => (Some(SysVClass::Float), 8),
        TypeAnn::RawPtr { .. } | TypeAnn::FnPtr { .. } => (Some(SysVClass::Integer), 8),
        TypeAnn::Named(name) => match name.as_str() {
            "i8" | "u8" => (Some(SysVClass::Integer), 1),
            "i16" | "u16" => (Some(SysVClass::Integer), 2),
            "i32" | "u32" | "bool" => (Some(SysVClass::Integer), 4),
            "i64" | "u64" | "usize" | "isize" => (Some(SysVClass::Integer), 8),
            "f32" => (Some(SysVClass::Float), 4),
            "f64" => (Some(SysVClass::Float), 8),
            _other => {
                // Nested repr(C) struct or unknown — Phase B defers to MEMORY class.
                let _ = repr_c; // used in future phases
                (None, 0)
            }
        },
        _ => (None, 0),
    }
}

/// RFC 0010 Phase B — SysV AMD64 struct parameter class.
/// Used by `sysv_classify_struct` and exposed for tests.
#[cfg(feature = "std-surface")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysVClass {
    /// Integer/pointer fields — passed in general-purpose registers.
    Integer,
    /// Floating-point fields — passed in XMM registers.
    Float,
    /// Aggregate too large or mixed — passed via pointer (caller allocates).
    Memory,
}

/// RFC 0010 Phase B — SysV AMD64 ABI struct-passing classification.
///
/// Given the field types of a `#[repr(C)]` struct (up to 4 fields, all Copy),
/// returns the list of MLIR type strings that represent how the struct is
/// passed in a function call under the SysV AMD64 ABI:
///
/// - All-integer/pointer, total ≤ 8 B → `["i64"]` (one eightbyte)
/// - All-integer/pointer, total ≤ 16 B → `["i64", "i64"]` (two eightbytes)
/// - All-float, total ≤ 8 B → single float type
/// - All-float, total ≤ 16 B → two float types
/// - Mixed int+float or > 16 B → `["!llvm.ptr"]` (MEMORY class)
///
/// This is a pure function with no I/O; it is `pub` so Phase B tests can
/// invoke it directly to verify the classification logic.
#[cfg(feature = "std-surface")]
pub fn sysv_classify_struct(
    fields: &[crate::ast::TypeAnn],
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    if fields.is_empty() {
        return vec!["i64".to_string()];
    }
    if fields.len() > 4 {
        return vec!["!llvm.ptr".to_string()];
    }

    let mut total_bytes: usize = 0;
    let mut classes: Vec<SysVClass> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();

    for field_ty in fields {
        match classify_scalar_field(field_ty, repr_c) {
            (Some(cls), sz) => {
                total_bytes += sz;
                classes.push(cls);
                sizes.push(sz);
            }
            (None, _) => return vec!["!llvm.ptr".to_string()],
        }
    }

    if total_bytes > 16 {
        return vec!["!llvm.ptr".to_string()];
    }

    let all_integer = classes.iter().all(|c| *c == SysVClass::Integer);
    let all_float = classes.iter().all(|c| *c == SysVClass::Float);

    if all_integer {
        if total_bytes <= 8 {
            vec!["i64".to_string()]
        } else {
            vec!["i64".to_string(), "i64".to_string()]
        }
    } else if all_float {
        // Determine the dominant float type per eightbyte slot (0..8, 8..16).
        // f64 (8 bytes) dominates f32 (4 bytes) within a slot.
        // Walk fields in order, tracking byte offset to assign each to a slot.
        let mut slot0_has_f64 = false;
        let mut slot1_has_f64 = false;
        let mut byte_off: usize = 0;
        for &sz in &sizes {
            if byte_off < 8 {
                if sz == 8 {
                    slot0_has_f64 = true;
                }
            } else {
                if sz == 8 {
                    slot1_has_f64 = true;
                }
            }
            byte_off += sz;
        }
        let first_slot = if slot0_has_f64 { "f64" } else { "f32" };
        if total_bytes <= 8 {
            vec![first_slot.to_string()]
        } else {
            let second_slot = if slot1_has_f64 { "f64" } else { "f32" };
            vec![first_slot.to_string(), second_slot.to_string()]
        }
    } else {
        // Mixed integer + float -> MEMORY class.
        vec!["!llvm.ptr".to_string()]
    }
}

/// RFC 0010 Phase C — Win64 struct parameter class.
/// Used by `win64_classify_struct` and exposed for tests.
#[cfg(feature = "std-surface")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Win64Class {
    /// Struct fits in one general-purpose register (size ∈ {1, 2, 4, 8}).
    Register,
    /// Struct passed by pointer (caller-allocated; size not in {1,2,4,8}).
    Memory,
}

/// RFC 0010 Phase C — Microsoft x64 ABI struct-passing classification.
///
/// Microsoft x64 ABI §4 (struct/union passing rules):
/// - Structs of size exactly 1, 2, 4, or 8 bytes: passed by value in one
///   general-purpose register as the matching integer type (i8, i16, i32, i64).
/// - All other sizes: passed by pointer (caller allocates on the stack).
///   Sizes 3, 5, 6, 7 technically "round up" but since there is no canonical
///   way to represent a 3-byte integer in LLVM IR, we classify them as MEMORY
///   (the caller passes a pointer to the aligned copy, which is the safe and
///   correct ABI implementation).
///
/// This function returns the Vec<String> of MLIR type tokens, matching the
/// calling convention of `sysv_classify_struct`.
#[cfg(feature = "std-surface")]
pub fn win64_classify_struct(
    fields: &[crate::ast::TypeAnn],
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    if fields.is_empty() {
        return vec!["i64".to_string()];
    }

    // Compute total byte size using the same scalar classifier as SysV.
    let mut total_bytes: usize = 0;
    for field_ty in fields {
        match classify_scalar_field(field_ty, repr_c) {
            (Some(_), sz) => total_bytes += sz,
            (None, _) => return vec!["!llvm.ptr".to_string()],
        }
    }

    // Win64: pass by value only for sizes {1, 2, 4, 8}.
    match total_bytes {
        1 => vec!["i8".to_string()],
        2 => vec!["i16".to_string()],
        4 => vec!["i32".to_string()],
        8 => vec!["i64".to_string()],
        _ => vec!["!llvm.ptr".to_string()],
    }
}

/// RFC 0010 Phase C — resolve `CallConv::C` to the platform-default ABI.
///
/// On Linux/macOS x86_64: resolves to `CallConv::SysV`.
/// On Windows x86_64: resolves to `CallConv::Win64`.
/// `CallConv::Aapcs` is passed through (Phase D will handle it; Phase C
/// callers fall back to SysV for the MLIR emission).
#[cfg(feature = "std-surface")]
pub(crate) fn resolve_callconv(cc: crate::ast::CallConv) -> crate::ast::CallConv {
    use crate::ast::CallConv;
    match cc {
        CallConv::C => {
            if cfg!(target_os = "windows") {
                CallConv::Win64
            } else {
                CallConv::SysV
            }
        }
        other => other,
    }
}

/// RFC 0010 Phase C — ABI-aware type classifier dispatcher.
///
/// Routes to `extern_type_to_mlir_multi` (SysV) or
/// `extern_type_to_mlir_multi_win64` (Win64) based on the resolved callconv.
/// `CallConv::Aapcs` is not yet implemented (Phase D); it falls back to SysV
/// with a runtime note so callers can test the dispatch path today.
#[cfg(feature = "std-surface")]
pub(crate) fn extern_type_to_mlir_multi_for(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
    callconv: crate::ast::CallConv,
) -> Vec<String> {
    use crate::ast::CallConv;
    match callconv {
        CallConv::Win64 => extern_type_to_mlir_multi_win64(ty, repr_c),
        CallConv::SysV | CallConv::C => extern_type_to_mlir_multi(ty, repr_c),
        CallConv::Aapcs => {
            // Phase D deferred. Fall back to SysV for now.
            extern_type_to_mlir_multi(ty, repr_c)
        }
    }
}

/// RFC 0010 Phase C — Win64 variant of `extern_type_to_mlir_multi`.
///
/// Maps a MIND `TypeAnn` to the MLIR LLVM type string(s) using the
/// Microsoft x64 ABI struct-passing rules instead of SysV.
///
/// For non-struct types the result is identical to `extern_type_to_mlir_multi`
/// (scalars, pointers, function pointers all have the same representation
/// under both ABIs on x86_64). The difference appears only for `#[repr(C)]`
/// struct types: Win64 passes them by value when they are exactly {1,2,4,8}
/// bytes, and by pointer otherwise.
#[cfg(feature = "std-surface")]
pub fn extern_type_to_mlir_multi_win64(
    ty: &crate::ast::TypeAnn,
    repr_c: &std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
) -> Vec<String> {
    use crate::ast::TypeAnn;
    match ty {
        TypeAnn::ScalarF32 => vec!["f32".to_string()],
        TypeAnn::ScalarF64 => vec!["f64".to_string()],
        TypeAnn::RawPtr { .. } => vec!["!llvm.ptr".to_string()],
        TypeAnn::FnPtr { .. } => vec!["!llvm.ptr".to_string()],
        TypeAnn::Named(name) => {
            match name.as_str() {
                "f32" => return vec!["f32".to_string()],
                "f64" => return vec!["f64".to_string()],
                "i8" | "u8" => return vec!["i8".to_string()],
                "i16" | "u16" => return vec!["i16".to_string()],
                "i32" | "u32" | "bool" => return vec!["i32".to_string()],
                "i64" | "u64" | "usize" | "isize" => return vec!["i64".to_string()],
                _ => {}
            }
            // Check for repr(C) struct — apply Win64 classification.
            if let Some(fields) = repr_c.get(name.as_str()) {
                win64_classify_struct(fields, repr_c)
            } else {
                vec!["i64".to_string()]
            }
        }
        TypeAnn::ScalarI32 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            vec!["i32".to_string()]
        }
        TypeAnn::ScalarI64 => vec!["i64".to_string()],
        _ => vec!["i64".to_string()],
    }
}
