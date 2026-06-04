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
use crate::ast::TensorElemOp;
use crate::ast::TypeAnn;

use crate::ir::BinOp;
use crate::ir::IRModule;
use crate::ir::IndexSpec;
use crate::ir::Instr;
use crate::ir::SliceSpec;
use crate::ir::ValueId;
use crate::types::DType;
use crate::types::ShapeDim;

// ---------------------------------------------------------------------------
// Codegen monomorphization (bounded slice) — generic fns reach the IR backend.
//
// A generic `fn id<T>(x: T) -> T { x }` is a TEMPLATE: it is never emitted as
// an `Instr::FnDef` itself. Instead, every concrete call site (`id(5)`) records
// a monomorphization request keyed on the deterministic mangled instance name
// (`id$i64`); after the module body is lowered, each distinct instance is
// emitted once, in sorted-by-mangled-name order, so `emit_mic3`/`trace_hash`
// stay byte-identical across runs (no HashMap iteration, no clocks, no rng).
//
// WEDGE: non-generic code is untouched — a function with empty `type_params`
// flows through the existing FnDef path verbatim, and a call whose callee is
// not a registered generic flows through the existing Call path verbatim. The
// state below lives in a `thread_local!` scoped to a single `lower_to_ir` call,
// so no existing `lower_expr` signature changes (and the non-generic lowering
// is literally the same code).
//
// Bounded to: a single type parameter, scalar concrete arg types inferred from
// a literal call argument (`Int -> i64`, `Float -> f64`), identity/min shape.
// Multi-param / nested generics / trait bounds are a later slice.
// ---------------------------------------------------------------------------

/// One pending monomorphization: the generic fn name plus the concrete type
/// substituted for its single type parameter, both already reflected in the
/// mangled instance name used as the map key.
#[derive(Clone)]
struct MonoRequest {
    /// Source-level generic fn name (e.g. `id`).
    generic_name: String,
    /// Concrete type bound to the (single) type parameter.
    concrete: TypeAnn,
}

#[derive(Default)]
struct MonoCtx {
    /// Generic fn templates by source name (only fns with non-empty
    /// `type_params`). Read-only after the pre-pass.
    templates: std::collections::BTreeMap<String, ast::Node>,
    /// Requested instances keyed on mangled name (e.g. `id$i64`). `BTreeMap`
    /// so draining is deterministic.
    requests: std::collections::BTreeMap<String, MonoRequest>,
}

thread_local! {
    static MONO: std::cell::RefCell<MonoCtx> = std::cell::RefCell::new(MonoCtx::default());
}

/// Short, deterministic mangle suffix for a concrete scalar type.
/// Returns `None` for shapes outside the bounded slice (the call then lowers
/// through the ordinary, non-monomorphized path — no behavior change).
fn mangle_suffix(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        TypeAnn::ScalarI64 => Some("i64"),
        TypeAnn::ScalarI32 => Some("i32"),
        TypeAnn::ScalarF64 => Some("f64"),
        TypeAnn::ScalarF32 => Some("f32"),
        TypeAnn::ScalarBool => Some("bool"),
        _ => None,
    }
}

/// Infer the concrete scalar type of a call argument from its literal form.
/// Bounded slice: only literal `Int`/`Float` operands are recognised; anything
/// else returns `None` so the call falls back to the ordinary path.
fn infer_concrete_arg_type(arg: &ast::Node) -> Option<TypeAnn> {
    match arg {
        ast::Node::Lit(Literal::Int(_), _) => Some(TypeAnn::ScalarI64),
        ast::Node::Lit(Literal::Float(_), _) => Some(TypeAnn::ScalarF64),
        ast::Node::Neg { operand, .. } => match operand.as_ref() {
            ast::Node::Lit(Literal::Int(_), _) => Some(TypeAnn::ScalarI64),
            ast::Node::Lit(Literal::Float(_), _) => Some(TypeAnn::ScalarF64),
            _ => None,
        },
        _ => None,
    }
}

/// If `callee` names a registered generic and the call's single argument has an
/// inferable concrete scalar type, register the instance and return its mangled
/// name. Returns `None` (caller keeps the original name) when the callee is not
/// generic or the shape is outside the bounded slice.
fn try_register_mono_instance(callee: &str, args: &[ast::Node]) -> Option<String> {
    MONO.with(|cell| {
        let mut ctx = cell.borrow_mut();
        if !ctx.templates.contains_key(callee) {
            return None;
        }
        // Bounded slice: exactly one argument bound to the single type param.
        if args.len() != 1 {
            return None;
        }
        let concrete = infer_concrete_arg_type(&args[0])?;
        let suffix = mangle_suffix(&concrete)?;
        let mangled = format!("{callee}${suffix}");
        ctx.requests
            .entry(mangled.clone())
            .or_insert_with(|| MonoRequest {
                generic_name: callee.to_string(),
                concrete: concrete.clone(),
            });
        Some(mangled)
    })
}

/// Synthesize the concrete (non-generic) `FnDef` AST for one instance:
/// clone the template, rename it to `mangled`, drop `type_params`, and rewrite
/// every parameter typed by the (single) type parameter to the concrete type.
fn instantiate_template(
    template: &ast::Node,
    mangled: &str,
    concrete: &TypeAnn,
) -> Option<ast::Node> {
    if let ast::Node::FnDef {
        is_pub,
        is_test,
        type_params,
        params,
        ret_type,
        body,
        reap_threshold,
        attrs,
        span,
        ..
    } = template
    {
        // Bounded slice: a single type parameter.
        let tp = type_params.first()?.clone();
        let new_params: Vec<ast::Param> = params
            .iter()
            .map(|p| {
                let ty = if matches!(&p.ty, TypeAnn::Named(n) if *n == tp) {
                    concrete.clone()
                } else {
                    p.ty.clone()
                };
                ast::Param {
                    name: p.name.clone(),
                    ty,
                    span: p.span,
                }
            })
            .collect();
        let new_ret = ret_type.as_ref().map(|r| {
            if matches!(r, TypeAnn::Named(n) if *n == tp) {
                concrete.clone()
            } else {
                r.clone()
            }
        });
        Some(ast::Node::FnDef {
            is_pub: *is_pub,
            is_test: *is_test,
            name: mangled.to_string(),
            type_params: Vec::new(),
            params: new_params,
            ret_type: new_ret,
            body: body.clone(),
            reap_threshold: *reap_threshold,
            attrs: attrs.clone(),
            span: *span,
        })
    } else {
        None
    }
}

pub fn lower_to_ir(module: &ast::Module) -> IRModule {
    let mut ir = IRModule::new();
    // Codegen monomorphization pre-pass — collect generic fn TEMPLATES (those
    // with a non-empty `type_params`) so call sites can route to a concrete
    // instance. The thread-local is reset at entry so a prior `lower_to_ir`
    // call on this thread can never leak templates/requests into this one
    // (which would perturb `emit_mic3`/`trace_hash`). Functions with empty
    // `type_params` are never registered, so the non-generic path is untouched.
    MONO.with(|cell| {
        let mut ctx = cell.borrow_mut();
        *ctx = MonoCtx::default();
        for item in &module.items {
            if let ast::Node::FnDef {
                name, type_params, ..
            } = item
            {
                if !type_params.is_empty() {
                    ctx.templates.insert(name.clone(), item.clone());
                }
            }
        }
    });
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
        if let ast::Node::StructDef {
            name,
            fields,
            attrs,
            ..
        } = item
        {
            if attrs
                .iter()
                .any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C"))
            {
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
            ast::Node::StructDef {
                name,
                fields,
                attrs,
                ..
            } => {
                let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                ir.struct_defs.insert(name.clone(), field_names);
                // RFC 0010 Phase B: if the struct carries `#[repr(C)]`, register
                // its field types in `repr_c_structs` so extern_type_to_mlir can
                // classify Named types that appear in `extern "C"` signatures.
                let is_repr_c = attrs
                    .iter()
                    .any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C"));
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
            // "finish MIND" Step 2 — record each enum variant's ordinal i64
            // tag (`0, 1, 2, …` in declaration order) under its fully-qualified
            // path (`"Mode::On"`) so that (a) the `Lit(Ident)` arm can lower a
            // variant used as a value to its tag and (b) `desugar_match_to_if`
            // can compare a scrutinee against a fieldless variant's tag. An
            // enum declaration is a no-op at the value level, so the
            // `ConstI64(0)`/`Output` placeholder is preserved for the
            // declaration-only IR-shape contract (mirrors `StructDef`).
            #[cfg(feature = "std-surface")]
            ast::Node::EnumDef { name, variants, .. } => {
                for (ordinal, variant) in variants.iter().enumerate() {
                    let path = format!("{name}::{}", variant.name);
                    ir.enum_variant_tags.insert(path, ordinal as i64);
                }
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
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
        #[cfg(feature = "std-surface")]
        ast::Node::Lit(Literal::Str(s), _) => {
            // Materialize the literal into a real `String { addr, len, cap }`
            // value, reusing the StructLit machinery verbatim:
            //
            //   addr = __mind_alloc(n)               // n = UTF-8 byte length
            //   __mind_store_i8(addr + i, byte_i)    // for each UTF-8 byte
            //   rec  = __mind_alloc(24)              // 3-field i64 String record
            //   __mind_store_i64(rec + 0,  addr)     // field 0: addr
            //   __mind_store_i64(rec + 8,  n)        // field 1: len
            //   __mind_store_i64(rec + 16, n)        // field 2: cap
            //   rec                                  // ← the String value
            //
            // The literal then IS a normal String value that string_len /
            // string_slice_from / string_eq operate on with zero further
            // change. No new Instr / MLIR codegen / ABI surface — every
            // intrinsic here is already used by std (the same __mind_store_i8
            // that string_push_byte calls at std/string.mind:99).
            let bytes = s.as_bytes();
            let n = bytes.len() as i64;

            // n_const = number of UTF-8 bytes
            let n_const = ir.fresh();
            ir.instrs.push(Instr::ConstI64(n_const, n));

            // addr = __mind_alloc(n)  — backing buffer for the bytes
            let addr = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: addr,
                name: "__mind_alloc".to_string(),
                args: vec![n_const],
            });

            // __mind_store_i8(addr + i, byte_i) for each UTF-8 byte.
            for (i, &b) in bytes.iter().enumerate() {
                let byte_addr = if i == 0 {
                    addr
                } else {
                    let offset = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(offset, i as i64));
                    let sum = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: sum,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: offset,
                    });
                    sum
                };
                let byte_val = ir.fresh();
                ir.instrs.push(Instr::ConstI64(byte_val, b as i64));
                let store_ret = ir.fresh();
                ir.instrs.push(Instr::Call {
                    dst: store_ret,
                    name: "__mind_store_i8".to_string(),
                    args: vec![byte_addr, byte_val],
                });
            }

            // Build the 3-field i64 String record (24 bytes), exactly as the
            // StructLit arm does: rec = __mind_alloc(24); store addr/len/cap.
            let rec_bytes = ir.fresh();
            ir.instrs.push(Instr::ConstI64(rec_bytes, 24));
            let rec = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: rec,
                name: "__mind_alloc".to_string(),
                args: vec![rec_bytes],
            });

            // field 0 (addr) at offset 0
            let store0 = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: store0,
                name: "__mind_store_i64".to_string(),
                args: vec![rec, addr],
            });

            // field 1 (len) at offset 8
            let off8 = ir.fresh();
            ir.instrs.push(Instr::ConstI64(off8, 8));
            let rec8 = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: rec8,
                op: BinOp::Add,
                lhs: rec,
                rhs: off8,
            });
            let len_val = ir.fresh();
            ir.instrs.push(Instr::ConstI64(len_val, n));
            let store1 = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: store1,
                name: "__mind_store_i64".to_string(),
                args: vec![rec8, len_val],
            });

            // field 2 (cap) at offset 16
            let off16 = ir.fresh();
            ir.instrs.push(Instr::ConstI64(off16, 16));
            let rec16 = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: rec16,
                op: BinOp::Add,
                lhs: rec,
                rhs: off16,
            });
            let cap_val = ir.fresh();
            ir.instrs.push(Instr::ConstI64(cap_val, n));
            let store2 = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst: store2,
                name: "__mind_store_i64".to_string(),
                args: vec![rec16, cap_val],
            });

            rec
        }
        #[cfg(not(feature = "std-surface"))]
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
            // "finish MIND" Step 2 (Defect B fix): a fully-qualified enum
            // variant used as a VALUE (e.g. `Mode::On`) lowers to its ordinal
            // i64 discriminant tag instead of falling through to the
            // placeholder `const.i64 0`. The path string (`"Mode::On"`) is the
            // exact key the parser accumulates for a `Type::Variant`
            // identifier, matching the `enum_variant_tags` registry built when
            // the `EnumDef` was lowered.
            #[cfg(feature = "std-surface")]
            if let Some(tag) = ir.enum_variant_tags.get(name).copied() {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, tag));
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
        ast::Node::CallTensorRelu { x, .. } => {
            let src = lower_expr(x, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::Relu { dst, src });
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
        // RFC 0012 Phase B — `A @ B` matmul operator.
        //
        // DESUGAR POINT (single, well-defined): `A @ B` lowers here to
        // `Instr::MatMul { a, b }` — the same IR node that `CallMatMul`
        // (the explicit `tensor.matmul(A, B)` form) produces.  This
        // guarantees byte-identical IR text between `A @ B` and
        // `tensor.matmul(A, B)`.
        //
        // MLIR-level byte-identity with `matmul_rmajor_f32_v` (the RFC
        // 0012 §7.2 gate-matrix target) requires threading shape dims
        // (M, K) through from the type-checker to emit the correct
        // `Instr::Call` args — deferred to Phase B.2.  At the IR text
        // level (`format_ir_module`) both forms emit `matmul %A, %B`,
        // which is byte-identical and sufficient for the Phase B gate
        // as implemented in this test suite.
        ast::Node::TensorMatmul { lhs, rhs, .. } => {
            let a = lower_expr(lhs, ir, env, struct_env, receiver_types);
            let b = lower_expr(rhs, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            ir.instrs.push(Instr::MatMul { dst, a, b });
            dst
        }
        // RFC 0012 Phase B — elementwise `.+ .- .* ./` operators.
        //
        // DESUGAR POINT (single, well-defined): desugars to `Instr::BinOp`
        // — the same IR node that `Node::Binary` (scalar `+`, `-`, `*`, `/`)
        // produces for tensor operands.  The IR-level representation is
        // identical: both forms emit `add %L, %R` (or sub/mul/div).
        ast::Node::TensorElemwise { op, lhs, rhs, .. } => {
            let l = lower_expr(lhs, ir, env, struct_env, receiver_types);
            let r = lower_expr(rhs, ir, env, struct_env, receiver_types);
            let dst = ir.fresh();
            let ir_op = match op {
                TensorElemOp::Add => BinOp::Add,
                TensorElemOp::Sub => BinOp::Sub,
                TensorElemOp::Mul => BinOp::Mul,
                TensorElemOp::Div => BinOp::Div,
            };
            ir.instrs.push(Instr::BinOp {
                dst,
                op: ir_op,
                lhs: l,
                rhs: r,
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
            type_params,
            params,
            body,
            reap_threshold,
            ..
        } => {
            // Codegen monomorphization: a generic fn is a TEMPLATE, never
            // emitted as an `Instr::FnDef` here. It was registered in the
            // pre-pass; concrete instances are emitted after the module body
            // is lowered (sorted by mangled name). Like every other declaration
            // arm, a template produces no value — return the unit placeholder.
            if !type_params.is_empty() {
                let id = ir.fresh();
                ir.instrs.push(Instr::ConstI64(id, 0));
                return id;
            }
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
                // "finish MIND" Step 2: inherit the enum-discriminant table so
                // a variant referenced inside a fn body (as a value or a match
                // arm) resolves to its tag rather than the placeholder 0.
                fn_ir.enum_variant_tags = ir.enum_variant_tags.clone();
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

            // Lower function body.
            //
            // `Return` is unique to fn scope and handled inline.
            // `Let` / `Assign` / expression stmts share the same
            // Let→tensor-binding + Assign→bind + expr pattern that is
            // extracted in `lower_stmt_seq` (used by `Node::Region`).
            // FnDef-specific extras — P0f struct-env tracking and Gap-C
            // branch-binding propagation — are layered on top after each
            // stmt is lowered.
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
                        // RFC 0005 Gap 1: if the emitted statement was an
                        // `Instr::While`, thread live_vars back into `fn_env`
                        // so code after the loop uses the post-loop SSA ids.
                        // The While emitter appends a trailing ConstI64(unit,0)
                        // after the While instr itself, so we check the
                        // second-to-last instruction for the While node.
                        #[cfg(feature = "std-surface")]
                        {
                            let n = fn_ir.instrs.len();
                            if n >= 2 {
                                if let Instr::While {
                                    live_vars,
                                    exit_ids,
                                    ..
                                } = &fn_ir.instrs[n - 2]
                                {
                                    // F2: rebind to the loop EXIT id (dominating
                                    // ^while_after block arg), not the
                                    // body-internal post_id.
                                    for (k, (vname, _post)) in live_vars.iter().enumerate() {
                                        let exit =
                                            exit_ids.get(k).copied().unwrap_or(live_vars[k].1);
                                        fn_env.insert(vname.clone(), exit);
                                    }
                                }
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
            // F2: names this branch writes — outer-var Assigns, branch-local
            // Lets, and any outer var rebound by a NESTED region (loop/if).
            // The union of then/else writes becomes the merge phi set, and each
            // merged var's per-branch value is taken from this env (dominating
            // at the branch's exit).
            let mut then_writes: Vec<String> = Vec::new();
            let record_then_write = |name: &str, writes: &mut Vec<String>| {
                if !writes.iter().any(|n| n == name) {
                    writes.push(name.to_owned());
                }
            };
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
                        record_then_write(name, &mut then_writes);
                        then_result = id;
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let id =
                            lower_expr(value, &mut then_ir, &then_env, struct_env, receiver_types);
                        then_env.insert(name.clone(), id);
                        record_then_write(name, &mut then_writes);
                        then_result = id;
                    }
                    other => {
                        then_result =
                            lower_expr(other, &mut then_ir, &then_env, struct_env, receiver_types);
                        // F2: thread a nested region's EXIT/merge ids upward so
                        // an outer var mutated inside it is visible (and
                        // dominating) at this branch's exit.
                        for (nm, eid) in last_region_exit_rebindings(&then_ir.instrs) {
                            then_env.insert(nm.clone(), eid);
                            record_then_write(&nm, &mut then_writes);
                        }
                    }
                }
            }

            // ── 3. Lower the else-branch (or synthesise a unit zero) ──────────
            //      Starts from then_ir's highest id.
            let mut else_ir = sub_ir_from_after(&then_ir, ir);
            let mut else_env = env.clone();
            let mut else_writes: Vec<String> = Vec::new();
            let record_else_write = |name: &str, writes: &mut Vec<String>| {
                if !writes.iter().any(|n| n == name) {
                    writes.push(name.to_owned());
                }
            };
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
                            record_else_write(name, &mut else_writes);
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
                            record_else_write(name, &mut else_writes);
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
                            for (nm, eid) in last_region_exit_rebindings(&else_ir.instrs) {
                                else_env.insert(nm.clone(), eid);
                                record_else_write(&nm, &mut else_writes);
                            }
                        }
                    }
                }
            }

            // ── 4. Build the merge phi set ────────────────────────────────────
            //
            // F2 dominance fix. For every variable written in EITHER branch,
            // allocate a fresh merge id (declared as an `^if_after` block arg)
            // and record, per branch, the value of that variable at the branch
            // EXIT (`then_env`/`else_env`). These per-branch values dominate the
            // branch's `cf.br ^if_after` because they are either the incoming
            // value, a top-level branch value, or a nested region's exit id
            // (threaded above) — never a raw value defined in a deeper branch.
            //
            // A branch that does not write the variable passes its incoming
            // value (`env[name]`). If the variable does not exist in the outer
            // env either (a branch-local `let`), that branch synthesises a unit
            // 0 inside its own block so both edges still pass a dominating
            // value of matching type.
            //
            // `branch_bindings[i].1` is set to the merge id so post-if code and
            // upward threading (`region_exit_rebindings`) pick up the
            // dominating merge value, never a branch-internal id.
            ir.next_id = ir.next_id.max(else_ir.next_id);
            let mut merged_names: Vec<String> = Vec::new();
            for n in then_writes.iter().chain(else_writes.iter()) {
                if !merged_names.iter().any(|m| m == n) {
                    merged_names.push(n.clone());
                }
            }
            // A branch that ends in `return` does not fall through to
            // `^if_after`; its `cf.br` is omitted and it must not pass a merge
            // value (and must not get a dead const pushed after its terminator).
            let then_falls_through = !matches!(then_ir.instrs.last(), Some(Instr::Return { .. }));
            let else_falls_through = !matches!(else_ir.instrs.last(), Some(Instr::Return { .. }));
            let mut branch_bindings: Vec<(String, ValueId)> = Vec::new();
            let mut merges: Vec<(ValueId, ValueId, ValueId)> = Vec::new();
            for name in &merged_names {
                // then-edge value (only meaningful if then falls through).
                let then_val = if then_falls_through {
                    match then_env.get(name) {
                        Some(&id) => id,
                        None => {
                            let z = ir.fresh();
                            then_ir.instrs.push(Instr::ConstI64(z, 0));
                            z
                        }
                    }
                } else {
                    // No then-edge; reuse the else value as the placeholder so
                    // the tuple is well-formed (the then `cf.br` is not emitted).
                    *else_env.get(name).unwrap_or(&ValueId(usize::MAX))
                };
                // else-edge value (only meaningful if else falls through).
                let else_val = if else_falls_through {
                    match else_env.get(name) {
                        Some(&id) => id,
                        None => {
                            let z = ir.fresh();
                            else_ir.instrs.push(Instr::ConstI64(z, 0));
                            z
                        }
                    }
                } else {
                    *then_env.get(name).unwrap_or(&ValueId(usize::MAX))
                };
                let merge_id = ir.fresh();
                merges.push((merge_id, then_val, else_val));
                branch_bindings.push((name.clone(), merge_id));
            }

            // ── 5. Emit Instr::If into the parent IR stream ───────────────────
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
                merges,
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
            // "finish MIND" Step 5 — payload-carrying enum constructor.
            // When the callee resolves to an enum variant in
            // `enum_variant_tags` AND the call has at least one argument
            // (e.g. `Opt::Some(42)`), the variant is a payload-carrying
            // constructor, NOT a runtime function call. Build the same
            // 2-field heap record the StructLit arm builds —
            // `[tag @ +0, payload @ +8]` — so the matching
            // `Opt::Some(v) => …` arm (desugared below) can load the tag
            // from `scrutinee + 0` and the bound payload from
            // `scrutinee + 8`. A FIELDLESS variant is never a `Call`
            // (it parses to `Lit(Ident)` and lowers to its bare tag in the
            // `Lit(Ident)` arm above), so this only fires for tuple
            // variants. The record shape is identical to StructLit's
            // (`__mind_alloc` + `__mind_store_i64` at 8-byte offsets), so
            // no new Instr/intrinsic/ABI is introduced.
            #[cfg(feature = "std-surface")]
            if !args.is_empty() {
                if let Some(tag) = ir.enum_variant_tags.get(callee).copied() {
                    // Lower the single payload argument first (mirrors the
                    // StructLit per-field lowering order). v1 tuple variants
                    // carry exactly one payload slot; additional args are a
                    // type error elsewhere and ignored here.
                    let payload = lower_expr(&args[0], ir, env, struct_env, receiver_types);

                    // bytes = 8 * 2 — two i64 slots: tag + payload. Emit
                    // const*const*Mul to mirror StructLit's heap-record build.
                    let eight = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(eight, 8));
                    let count = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(count, 2));
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

                    // __mind_store_i64(addr + 0, tag)
                    let tag_id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(tag_id, tag));
                    let store_tag = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: store_tag,
                        name: "__mind_store_i64".to_string(),
                        args: vec![addr, tag_id],
                    });

                    // payload_addr = addr + 8
                    let offset = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(offset, 8));
                    let payload_addr = ir.fresh();
                    ir.instrs.push(Instr::BinOp {
                        dst: payload_addr,
                        op: BinOp::Add,
                        lhs: addr,
                        rhs: offset,
                    });

                    // __mind_store_i64(payload_addr, payload)
                    let store_payload = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: store_payload,
                        name: "__mind_store_i64".to_string(),
                        args: vec![payload_addr, payload],
                    });

                    return addr;
                }
            }
            let arg_ids: Vec<ValueId> = args
                .iter()
                .map(|a| lower_expr(a, ir, env, struct_env, receiver_types))
                .collect();
            // Codegen monomorphization: if the callee is a registered generic
            // and the concrete arg type is inferable, route this call to the
            // mangled instance (`id$i64`) and queue that instance for emission.
            // A non-generic callee (or a shape outside the bounded slice) keeps
            // the original name, so non-generic call lowering is byte-identical.
            let name = try_register_mono_instance(callee, args).unwrap_or_else(|| callee.clone());
            let dst = ir.fresh();
            ir.instrs.push(Instr::Call {
                dst,
                name,
                args: arg_ids,
            });
            dst
        }
        // Phase 10.7 / "finish MIND" Step 1: `match scrutinee { arms }` —
        // DESUGAR to a right-nested chain of `Instr::If`. Each integer/bool
        // (`Literal::Int`) arm becomes `if scrutinee == <lit> { body } else
        // { rest }`; a `Wildcard` (`_`) or bare `Ident` arm becomes the
        // terminal `else` (an `Ident` first binds the scrutinee under that
        // name). The desugar is purely at the AST level and recurses into the
        // existing `ast::Node::If` lowering, so it reuses the keystone-
        // protected `exit_ids`/`merges`/`region_exit_rebindings` machinery
        // untouched. Enum-discriminant and payload-binding patterns are a
        // later step and fall back to the old sequential lowering.
        #[cfg(feature = "std-surface")]
        ast::Node::Match {
            scrutinee, arms, ..
        } => {
            match desugar_match_to_if(scrutinee, arms, &ir.enum_variant_tags) {
                Some(if_node) => lower_expr(&if_node, ir, env, struct_env, receiver_types),
                None => {
                    // Unsupported pattern kind (enum variant / non-int
                    // literal) — preserve the prior sequential behaviour so
                    // those matches are not regressed by this step.
                    let _scrut_id = lower_expr(scrutinee, ir, env, struct_env, receiver_types);
                    let mut last_id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(last_id, 0));
                    for arm in arms {
                        last_id = lower_expr(&arm.body, ir, env, struct_env, receiver_types);
                    }
                    last_id
                }
            }
        }
        // Non-gated fallback: default builds have no branching `If` lowering,
        // so retain the sequential-flatten behaviour.
        #[cfg(not(feature = "std-surface"))]
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
        // A cast `<expr> as <ty>` is value-preserving in the std-surface i64
        // ABI: scalars and raw pointers are all carried as i64 SSA values, so
        // the target type is purely a type-checker concern. Lower the operand
        // transparently (mirrors `Ref` / `Paren`). Without this arm the cast
        // fell through to the catch-all and was silently lowered to
        // `const.i64 0`, which dropped the operand entirely — e.g.
        // `memset(sa as *mut u8, 0, 16)` lost `sa`, then the FFI bridge in the
        // MLIR backend `inttoptr`-ed a zero, producing a NULL-pointer memset
        // and an `!llvm.ptr` vs `i64` mlir-opt type error.
        ast::Node::As { expr, .. } => lower_expr(expr, ir, env, struct_env, receiver_types),
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
            //
            // Use sub_ir_from so the condition sub-module's ValueIds start
            // above the parent's current next_id.  Without this, a constant
            // emitted in the condition (e.g. ConstI64(ValueId(0), 16)) would
            // collide with the function's first parameter (%0: i64) when both
            // are serialised into the same MLIR func.func body — the same
            // fix already applied to Instr::If (see sub_ir_from comment).
            #[cfg(feature = "std-surface")]
            let mut cond_ir = sub_ir_from(ir);
            #[cfg(not(feature = "std-surface"))]
            let mut cond_ir = IRModule::new();
            // Seed the condition sub-module's env with the current bindings
            // so identifiers in the condition (e.g. `i`, `n`) resolve.
            let cond_env = env.clone();
            let cond_id = lower_expr(cond, &mut cond_ir, &cond_env, struct_env, receiver_types);

            // Lower the body into a scratch sub-module.  Track every Assign
            // target — those are the variables that are live across the
            // back-edge and must become block arguments in MLIR.
            //
            // Chain from cond_ir so body ValueIds are disjoint from both
            // parent scope and condition scope (mirrors sub_ir_from_after
            // in the Instr::If path).
            #[cfg(feature = "std-surface")]
            let mut body_ir = sub_ir_from_after(&cond_ir, ir);
            #[cfg(not(feature = "std-surface"))]
            let mut body_ir = IRModule::new();
            let mut body_env = env.clone();
            let mut mutated: Vec<(String, ValueId)> = Vec::new();
            // Pre-loop ValueId for each mutated variable (parallel to mutated).
            // Captures the ValueId from env BEFORE the while loop so the MLIR
            // emitter can produce `cf.br ^while_header(init_0, init_1, ...)`.
            let mut init_ids: Vec<ValueId> = Vec::new();

            // Record that `name` is loop-carried with post-body value `new_id`.
            // The first time the loop sees a variable mutated, capture its
            // pre-loop init id from `body_env` (parallel to `mutated`).
            fn record_loop_mut(
                name: &str,
                new_id: ValueId,
                mutated: &mut Vec<(String, ValueId)>,
                init_ids: &mut Vec<ValueId>,
                pre_init: Option<ValueId>,
            ) {
                if let Some(pos) = mutated.iter().position(|(n, _)| n == name) {
                    mutated[pos].1 = new_id;
                } else {
                    init_ids.push(pre_init.unwrap_or(ValueId(usize::MAX)));
                    mutated.push((name.to_owned(), new_id));
                }
            }

            for stmt in body {
                match stmt {
                    ast::Node::Let { name, value, .. } => {
                        // `let` inside the loop body introduces a new SSA binding
                        // scoped to the body.  Emit the RHS and update body_env so
                        // subsequent body statements can reference the binding.
                        // These are NOT live_vars (they don't survive across the
                        // back-edge) unless a later Assign overwrites them.
                        let new_id =
                            lower_expr(value, &mut body_ir, &body_env, struct_env, receiver_types);
                        body_env.insert(name.clone(), new_id);
                    }
                    ast::Node::Assign { name, value, .. } => {
                        let pre_init = body_env.get(name.as_str()).copied();
                        let new_id =
                            lower_expr(value, &mut body_ir, &body_env, struct_env, receiver_types);
                        body_env.insert(name.clone(), new_id);
                        record_loop_mut(name, new_id, &mut mutated, &mut init_ids, pre_init);
                    }
                    other => {
                        lower_expr(other, &mut body_ir, &body_env, struct_env, receiver_types);
                        // F2: a nested region (if/while) inside the loop body may
                        // mutate an OUTER (loop-carried) variable. Thread the
                        // nested region's EXIT/merge id into body_env AND record
                        // the variable as loop-carried, so the back-edge passes a
                        // dominating value and the header re-feeds it next
                        // iteration. Without this, mutations buried in a nested
                        // branch (e.g. `while c { if p { x = 0 } }`) are invisible
                        // to the loop and it never makes progress.
                        {
                            for (nm, eid) in last_region_exit_rebindings(&body_ir.instrs) {
                                // Update body_env for any variable the nested
                                // region modified that is visible in the current
                                // outer loop body scope (including `let mut`
                                // bindings declared earlier in this body that are
                                // NOT pre-loop vars, e.g. `let mut min = i`).
                                // This ensures reads AFTER the inner loop within
                                // the same outer-loop iteration see the updated id.
                                //
                                // Only call record_loop_mut (make it loop-carried
                                // across the back-edge) for variables that exist in
                                // the pre-loop env (`env`) — genuine outer vars.
                                // Body-local `let` bindings are re-initialised each
                                // iteration and must not cross the back-edge.
                                if body_env.contains_key(&nm) {
                                    let pre_init = body_env.get(nm.as_str()).copied();
                                    body_env.insert(nm.clone(), eid);
                                    if env.contains_key(&nm) {
                                        record_loop_mut(
                                            &nm,
                                            eid,
                                            &mut mutated,
                                            &mut init_ids,
                                            pre_init,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Note: we cannot update the parent `env` here because `lower_expr`
            // takes `env: &HashMap` (immutable).  The FnDef body loop detects
            // the emitted `Instr::While` and propagates `live_vars` back to its
            // own mutable `fn_env`, mirroring the `branch_bindings` pattern used
            // for `Instr::If`.  See the `other =>` arm in the FnDef lowering.

            // Advance the parent next_id past all IDs used in cond and body
            // so subsequent instructions in the parent fn body stay disjoint.
            #[cfg(feature = "std-surface")]
            {
                ir.next_id = ir.next_id.max(cond_ir.next_id).max(body_ir.next_id);
            }

            // F2 region-scoped exit env: one fresh SSA id per loop-carried var.
            // `^while_after_N` declares these as block args (fed by the
            // header→after cond_br edge with header args, which dominate), and
            // code AFTER the loop is rebound to these exit ids instead of the
            // body-internal `post_id`s — guaranteeing dominance for every
            // post-loop reference, at every nesting level.
            #[cfg(feature = "std-surface")]
            let exit_ids: Vec<ValueId> = mutated.iter().map(|_| ir.fresh()).collect();
            #[cfg(not(feature = "std-surface"))]
            let exit_ids: Vec<ValueId> = Vec::new();

            ir.instrs.push(Instr::While {
                cond_id,
                cond_instrs: cond_ir.instrs,
                body: body_ir.instrs,
                live_vars: mutated,
                init_ids,
                exit_ids,
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
        // RFC 0005 P0g — `receiver.field = value` writes IN PLACE to the
        // heap record produced by P0e StructLit lowering. Exact inverse of
        // the FieldAccess read arm above: the same Step-1 (struct_env Ident
        // lookup) + Step-2 (receiver_types side-table) resolver computes the
        // field index, the same idx==0 fast path computes the field address,
        // and the RHS is lowered to an SSA value — but we emit a
        // `__mind_store_i64(field_addr, value)` instead of a load.
        //
        //   offset      = index_of(field, struct_defs[T]) * 8
        //   field_addr  = base + offset      (or base itself when offset == 0)
        //   __mind_store_i64(field_addr, value)
        //
        // The base address is the SAME allocation already bound in `env`
        // (Step 1) or produced by re-lowering the receiver (Step 2), so this
        // is a pure in-place mutation — no new struct-value SSA id is created
        // or threaded, and no exit_ids/merges/region rebinding changes are
        // needed. Both flat fields (`s.f = v`) and nested struct-typed field
        // writes (`o.inner.v = x`) are supported: the Step-2 side-table now
        // resolves a `FieldAccess` receiver (struct_resolver chains through
        // the inner field's declared type), so `lower_expr(receiver, …)`
        // re-lowers `o.inner` to the inner record's base address and the
        // store targets `base + idx*8` exactly like the flat case.
        // Unresolved receivers fall through to a `ConstI64(0)` placeholder,
        // matching the read arm, so older modules still compile.
        #[cfg(feature = "std-surface")]
        ast::Node::FieldAssign {
            receiver,
            field,
            value,
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
                    // Step 1 takes the base addr straight from env (no
                    // re-lowering); Step 2 must lower the receiver to get it.
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
                    let rhs = lower_expr(value, ir, env, struct_env, receiver_types);
                    let store_ret = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst: store_ret,
                        name: "__mind_store_i64".to_string(),
                        args: vec![field_addr, rhs],
                    });
                    // A field assignment is a statement; the store's return
                    // (unit) id is the value this expression yields.
                    store_ret
                }
                None => {
                    // Receiver type unresolvable — emit a stable placeholder,
                    // mirroring the read arm, so older modules still compile.
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
                    .flat_map(|p| {
                        extern_type_to_mlir_multi_for(&p.ty, &repr_c_snapshot, effective_callconv)
                    })
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
        // RFC 0010 Phase J-A — `region { ... }` block lowering.
        //
        // Strategy:
        //   1. Lower the body statements into a scratch sub-IRModule so that
        //      alloc ids are collected in a fresh SSA namespace.
        //   2. Walk the sub-module's instructions to record every SSA id that
        //      was produced by a `__mind_alloc` call (region-interior allocs).
        //   3. Perform the escape check: if the body's result value (last SSA
        //      id) is in `alloc_ids`, emit a `safety::region_escape` diagnostic
        //      and continue (we don't abort — diagnostics are advisory at the
        //      IR level; the runtime will safely free the ptr at region exit).
        //   4. Emit `Instr::Region { body, result, alloc_ids }` into the
        //      parent IR. The MLIR backend emits the enter/track/exit calls.
        //
        // Gated to `std-surface`.
        #[cfg(feature = "std-surface")]
        ast::Node::Region { body, .. } => {
            let mut body_ir = sub_ir_from(ir);
            let mut body_env = env.clone();
            let mut alloc_ids: Vec<crate::ir::ValueId> = Vec::new();

            // Lower the body using the shared statement-sequence helper.
            // It handles Let / Assign / expression statements and appends
            // every __mind_alloc call id to `alloc_ids` so the runtime
            // track-calls and the type-checker escape check can both act on
            // the same information.
            let last_id = lower_stmt_seq(
                body,
                &mut body_ir,
                &mut body_env,
                struct_env,
                receiver_types,
                Some(&mut alloc_ids),
            );

            // Determine the result value (last expression in body).
            let result = last_id.unwrap_or_else(|| {
                let id = body_ir.fresh();
                body_ir.instrs.push(Instr::ConstI64(id, 0));
                id
            });

            // The escape check (safety::region_escape) is now performed at
            // the type-checker level (Node::Region arm in
            // check_module_types_in_file) so that it flows through the
            // structured diagnostic surface (--reporter json / lsp) and is
            // consistent with the Phase A/B pattern.  No eprintln here.

            // Advance the parent IR's next_id past everything allocated in
            // the body sub-module so all SSA ids remain globally unique.
            ir.next_id = body_ir.next_id;

            // Allocate unique SSA ids for the enter/exit call results.
            // These must be globally unique (from the parent IR's counter)
            // so that nested regions do not emit duplicate MLIR value names.
            let enter_id = ir.fresh();
            let exit_id = ir.fresh();

            ir.instrs.push(Instr::Region {
                body: body_ir.instrs,
                result,
                enter_id,
                exit_id,
                alloc_ids,
            });

            result
        }
        // RFC 0005 method brick — a `recv.method(args)` call. Two cases,
        // both keyed off the receiver's resolved struct type `T`:
        //
        //   1. ZERO-ARG FIELD ACCESSOR — a zero-arg method whose name matches
        //      a field of `T` is a field read (`s.len()` on a String == `s.len`).
        //      We emit the identical `__mind_load_i64(base + idx*8)` the
        //      `FieldAccess` arm emits. No new IR, no new ABI. (Already shipped.)
        //
        //   2. UFCS DESUGAR — any other resolved method (`v.push(x)`,
        //      `s.push_byte(b)`) is sugar for the free function
        //      `{lowercase(T)}_{method}(recv, args…)` that std declares by
        //      convention (struct `Vec` -> `vec_push`, struct `String` ->
        //      `string_push_byte`). We lower the receiver + each arg and emit a
        //      plain `Instr::Call` with the receiver threaded as the first
        //      argument — the SAME machinery the `Node::Call` arm uses for a
        //      direct free-function call. If the named function does not exist
        //      it fails LOUD at link time (an undefined `func.call` symbol), it
        //      never silently returns 0. This is purely additive desugar onto
        //      existing `Instr::Call`; no new IR/intrinsic/ABI.
        //
        // The receiver's struct type is resolved exactly like the `FieldAccess`
        // arm: Step 1 is the `struct_env` Ident fast-path, Step 2 is the
        // `receiver_types` side-table keyed on this MethodCall's span (populated
        // by the struct_resolver pre-pass). When the receiver type cannot be
        // resolved AT ALL, a method-with-args call would otherwise be a silent
        // const-0 miscompile — so we emit a clear diagnostic and a poison value
        // (panics in debug, returns a loud sentinel call in release) rather than
        // const-0. Method calls are absent from the keystone and all of std, so
        // the keystone artifact is unaffected.
        #[cfg(feature = "std-surface")]
        ast::Node::MethodCall {
            receiver,
            method,
            args,
            span,
        } => {
            // Resolve the receiver's struct type name `T` and, for the cheap
            // Ident path, the bound variable name so we can reuse its SSA id.
            let (struct_name, var_name_opt): (Option<String>, Option<String>) =
                match receiver.as_ref() {
                    ast::Node::Lit(Literal::Ident(var_name), _) => match struct_env.get(var_name) {
                        Some(t) => (Some(t.clone()), Some(var_name.clone())),
                        None => (receiver_types.get(span).cloned(), None),
                    },
                    _ => (receiver_types.get(span).cloned(), None),
                };

            // Case 1 — zero-arg accessor whose name is a field of `T`.
            let field_idx = if args.is_empty() {
                struct_name.as_ref().and_then(|t| {
                    ir.struct_defs
                        .get(t)
                        .and_then(|fields| fields.iter().position(|f| f == method))
                })
            } else {
                None
            };

            if let Some(idx) = field_idx {
                let addr = match &var_name_opt {
                    Some(var_name) => match env.get(var_name) {
                        Some(id) => *id,
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
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
                return result;
            }

            // Case 2 — UFCS desugar to `{lowercase(T)}_{method}(recv, args…)`.
            match &struct_name {
                Some(t) => {
                    let recv_id = match &var_name_opt {
                        Some(var_name) => match env.get(var_name) {
                            Some(id) => *id,
                            None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                        },
                        None => lower_expr(receiver, ir, env, struct_env, receiver_types),
                    };
                    let mut call_args = vec![recv_id];
                    for a in args {
                        call_args.push(lower_expr(a, ir, env, struct_env, receiver_types));
                    }
                    let fn_name = format!("{}_{}", t.to_lowercase(), method);
                    let dst = ir.fresh();
                    ir.instrs.push(Instr::Call {
                        dst,
                        name: fn_name,
                        args: call_args,
                    });
                    dst
                }
                None => {
                    // Receiver type unresolved. A zero-arg unresolved call could
                    // be a never-defined accessor; a with-args unresolved call is
                    // a guaranteed silent miscompile under the old const-0
                    // fallthrough. Fail LOUD — never emit a silent const-0 at a
                    // method-with-args call site (#306 fail-closed philosophy).
                    if !args.is_empty() {
                        panic!(
                            "method call `{}.{}(...)` could not be resolved: the \
                             receiver's struct type is unknown, so it cannot desugar \
                             to a `<type>_{}` free function. Annotate the receiver's \
                             type or use the free-function form directly. (Emitting a \
                             const-0 placeholder here would be a silent miscompile.)",
                            describe_receiver(receiver),
                            method,
                            method,
                        );
                    }
                    // Zero-arg, unresolved type, not a known field — preserve the
                    // historical const-0 placeholder (it returns the receiver's
                    // identity for opaque accessors; unchanged behaviour).
                    let id = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(id, 0));
                    id
                }
            }
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

/// Best-effort textual description of a method-call receiver, used only to
/// build a clear diagnostic when a `recv.method(args)` call cannot resolve its
/// receiver's struct type (so it cannot UFCS-desugar to a `<type>_method` free
/// function). Falls back to a generic placeholder for non-Ident receivers.
#[cfg(feature = "std-surface")]
fn describe_receiver(receiver: &ast::Node) -> String {
    match receiver {
        ast::Node::Lit(Literal::Ident(name), _) => name.clone(),
        _ => "<expr>".to_string(),
    }
}

/// Collect all SSA ids produced by `__mind_alloc` calls in `instrs` into
/// `out`. Called after lowering each region body statement so that alloc
/// sites introduced by nested calls (vec_new, struct literals, etc.) are
/// also recorded.
///
/// Only looks one level deep — the Phase J-A escape check is conservative
/// (flags direct-return of an alloc result; aliasing through fields is
/// Phase J-B).
#[cfg(feature = "std-surface")]
fn collect_alloc_ids(instrs: &[Instr], out: &mut Vec<crate::ir::ValueId>) {
    for instr in instrs {
        if let Instr::Call { dst, name, .. } = instr {
            if name == "__mind_alloc" && !out.contains(dst) {
                out.push(*dst);
            }
        }
    }
}

/// "finish MIND" Step 1/2: desugar a `match` expression into a right-nested
/// chain of `ast::Node::If` so the existing branching `If` lowering executes
/// the match (instead of evaluating every arm unconditionally).
///
/// Supported: integer/bool arms (`Pattern::Literal(Literal::Int)`, which is
/// also how `true`/`false` patterns parse) and — Step 2 — FIELDLESS
/// enum-variant arms (`Pattern::EnumVariant` with no payload), which compare
/// the scrutinee against the variant's ordinal discriminant tag looked up in
/// `enum_tags`. Both forms may be followed by a single terminal catch-all
/// (`Wildcard` or bare `Ident`); an `Ident` catch-all binds the scrutinee
/// under that name before its arm body via a synthetic `let`.
///
/// Returns `None` for any unsupported pattern kind (payload-binding variants
/// such as `Some(v)`, string/float literals, or an enum-variant path absent
/// from `enum_tags`) so the caller can fall back to the prior behaviour; those
/// are handled by later steps.
///
/// The result is a pure AST rewrite: it constructs standard `Node::If` /
/// `Node::Binary(Eq)` nodes and is lowered through the unchanged
/// `ast::Node::If` arm, so none of the dominance/merge machinery is touched.
#[cfg(feature = "std-surface")]
fn desugar_match_to_if(
    scrutinee: &ast::Node,
    arms: &[ast::MatchArm],
    enum_tags: &std::collections::BTreeMap<String, i64>,
) -> Option<ast::Node> {
    let span = scrutinee.span();
    // An arm body written with braces (`1 => { x = 100 }`) parses to a single
    // `Node::Block` wrapping the arm's statements, whereas a parsed `if { … }`
    // produces a FLAT `Vec<Node>` of statements. The If lowering only treats
    // top-level `Assign`/`Let` nodes as branch writes that merge into the
    // post-`if` scope (its `then_writes`/`else_writes` → `branch_bindings`);
    // a `Block` falls through to the expression arm and its enclosing-scope
    // mutations are dropped at the merge. So splice a braced body's statements
    // directly into the branch list to mirror the parsed-`if` shape exactly.
    let flatten_body = |body: ast::Node| -> Vec<ast::Node> {
        match body {
            ast::Node::Block { stmts, .. } => stmts,
            other => vec![other],
        }
    };
    // Split the arms into the leading test arms and the terminal else.
    // Only the FINAL arm may be a catch-all (`_` / bare ident); a catch-all
    // in a non-final position would shadow the rest — leave such (malformed)
    // matches to the fallback path.
    let mut else_branch: Option<Vec<ast::Node>> = None;
    let mut last_idx = arms.len();
    if let Some(last) = arms.last() {
        match &last.pattern {
            ast::Pattern::Wildcard => {
                else_branch = Some(flatten_body(last.body.clone()));
                last_idx = arms.len() - 1;
            }
            ast::Pattern::Ident(name) => {
                // Bind the scrutinee under `name` before the arm body.
                let bind = ast::Node::Let {
                    name: name.clone(),
                    mutable: false,
                    ann: None,
                    value: Box::new(scrutinee.clone()),
                    span,
                };
                let mut stmts = vec![bind];
                stmts.extend(flatten_body(last.body.clone()));
                else_branch = Some(stmts);
                last_idx = arms.len() - 1;
            }
            _ => {}
        }
    }

    // "finish MIND" Step 5 — does this match scrutinise a PAYLOAD-carrying
    // enum value? If ANY arm binds a payload (`Some(v)` — a non-empty
    // `EnumVariant.args`), the scrutinee is a 2-field heap record
    // `[tag @ +0, payload @ +8]` (built by the enum-constructor path in the
    // `Node::Call` arm), NOT a bare discriminant. In that case every
    // enum-variant comparison must test the LOADED tag
    // (`__mind_load_i64(scrutinee + 0)`) rather than the scrutinee value
    // itself, and a payload-binding arm prepends a synthetic
    // `let <ident> = __mind_load_i64(scrutinee + 8)` so the payload binds —
    // exactly the synthetic-let shape the terminal `Ident` catch-all above
    // already uses. Pure fieldless (C-like) enums keep comparing the bare
    // scrutinee value, so this never perturbs Step-2 fieldless matches.
    let scrutinee_carries_payload = arms
        .iter()
        .any(|a| matches!(&a.pattern, ast::Pattern::EnumVariant { args, .. } if !args.is_empty()));

    // Build the comparison LHS node: for a payload-carrying scrutinee this is
    // `__mind_load_i64(scrutinee + 0)` (mirrors the FieldAccess +0 fast path,
    // which passes the base address with no offset); otherwise the bare
    // scrutinee.
    let load_i64 = |addr: ast::Node| ast::Node::Call {
        callee: "__mind_load_i64".to_string(),
        args: vec![addr],
        span,
    };
    let cmp_lhs: ast::Node = if scrutinee_carries_payload {
        load_i64(scrutinee.clone())
    } else {
        scrutinee.clone()
    };

    // Each remaining arm becomes a `<cmp_lhs> == <rhs>` comparison. Build the
    // RHS literal node per pattern kind: an integer/bool literal compares
    // against itself; an enum variant (fieldless OR payload-carrying)
    // compares against its ordinal discriminant tag. Any other pattern
    // (unknown variant path, non-int literal) bails the whole match to the
    // fallback.
    let test_arms = &arms[..last_idx];
    let mut rhs_nodes: Vec<ast::Node> = Vec::with_capacity(test_arms.len());
    for arm in test_arms {
        let rhs = match &arm.pattern {
            ast::Pattern::Literal(Literal::Int(_)) => {
                let lit = match &arm.pattern {
                    ast::Pattern::Literal(l) => l.clone(),
                    _ => unreachable!(),
                };
                ast::Node::Lit(lit, span)
            }
            // Step 2/5: enum-discriminant arm. Fieldless variants compare the
            // bare scrutinee against the tag; payload variants compare the
            // loaded tag and bind their payload (handled below when the arm
            // body is assembled).
            ast::Pattern::EnumVariant { path, .. } => {
                let tag = enum_tags.get(path).copied()?;
                ast::Node::Lit(Literal::Int(tag), span)
            }
            _ => return None,
        };
        rhs_nodes.push(rhs);
    }
    // Need at least one test arm to form a branch; a lone catch-all is
    // already handled fine by the fallback (and has no comparison to make).
    if test_arms.is_empty() {
        return None;
    }

    // Build the chain from the tail backwards so the first arm ends up
    // outermost. `else_stmts` is the body of the current innermost `else`:
    // the terminal catch-all arm initially, then each enclosing `If`.
    let mut else_stmts: Option<Vec<ast::Node>> = else_branch;
    for (arm, rhs) in test_arms.iter().zip(rhs_nodes.iter()).rev() {
        let cond = ast::Node::Binary {
            op: ast::BinOp::Eq,
            left: Box::new(cmp_lhs.clone()),
            right: Box::new(rhs.clone()),
            span,
        };
        // Step 5: for a payload-binding arm (`Some(v) => …`), PREPEND a
        // synthetic `let v = __mind_load_i64(scrutinee + 8)` to the arm body
        // so the bound identifier resolves to the record's payload slot. Only
        // a single-`Ident` sub-pattern is supported (v1 tuple variants carry
        // one payload); anything else bails the whole match to the fallback.
        let then_branch: Vec<ast::Node> = match &arm.pattern {
            ast::Pattern::EnumVariant { args, .. } if !args.is_empty() => {
                let bound = match args.as_slice() {
                    [ast::Pattern::Ident(name)] => name.clone(),
                    _ => return None,
                };
                // payload_addr = scrutinee + 8
                let offset = ast::Node::Lit(Literal::Int(8), span);
                let payload_addr = ast::Node::Binary {
                    op: ast::BinOp::Add,
                    left: Box::new(scrutinee.clone()),
                    right: Box::new(offset),
                    span,
                };
                let bind = ast::Node::Let {
                    name: bound,
                    mutable: false,
                    ann: None,
                    value: Box::new(load_i64(payload_addr)),
                    span,
                };
                let mut stmts = vec![bind];
                stmts.extend(flatten_body(arm.body.clone()));
                stmts
            }
            _ => flatten_body(arm.body.clone()),
        };
        let if_node = ast::Node::If {
            cond: Box::new(cond),
            then_branch,
            else_branch: else_stmts.take(),
            span,
        };
        else_stmts = Some(vec![if_node]);
    }
    // `else_stmts` now holds the single outermost `If` (test_arms is
    // non-empty, so exactly one node).
    else_stmts.and_then(|mut v| v.pop())
}

/// Lower a sequence of `Let` / `Assign` / expression body statements into
/// `ir`, updating `env` with new name→id bindings.
///
/// Returns the `ValueId` of the last statement, or `None` when `stmts` is
/// empty.  Callers that need a unit value (region, fn body) synthesise a
/// `ConstI64(0)` when `None` is returned.
///
/// When `alloc_ids` is `Some`, every `__mind_alloc` call id produced during
/// lowering is appended to it (used by `Node::Region` to build the escape-
/// check set passed to `Instr::Region`).
///
/// This is the shared body-lowering core used by both `Node::Region` and
/// `Node::FnDef`.  The `FnDef` arm adds `Return`-statement handling and
/// `std-surface`-gated struct-env / branch-binding tracking on top.
#[cfg(feature = "std-surface")]
fn lower_stmt_seq(
    stmts: &[ast::Node],
    ir: &mut IRModule,
    env: &mut HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
    mut alloc_ids: Option<&mut Vec<ValueId>>,
) -> Option<ValueId> {
    let mut last_id: Option<ValueId> = None;
    for stmt in stmts {
        let id = match stmt {
            ast::Node::Let {
                name, ann, value, ..
            } => {
                let id = match ann {
                    Some(TypeAnn::Tensor { dtype, dims })
                    | Some(TypeAnn::DiffTensor { dtype, dims }) => lower_tensor_binding(
                        ir,
                        value,
                        dtype,
                        dims,
                        env,
                        struct_env,
                        receiver_types,
                    ),
                    _ => lower_expr(value, ir, env, struct_env, receiver_types),
                };
                env.insert(name.clone(), id);
                id
            }
            ast::Node::Assign { name, value, .. } => {
                let id = lower_expr(value, ir, env, struct_env, receiver_types);
                env.insert(name.clone(), id);
                id
            }
            other => lower_expr(other, ir, env, struct_env, receiver_types),
        };
        if let Some(ref mut out) = alloc_ids {
            collect_alloc_ids(&ir.instrs, out);
        }
        last_id = Some(id);
    }
    last_id
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
    m.enum_variant_tags = parent.enum_variant_tags.clone();
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
    m.enum_variant_tags = meta_src.enum_variant_tags.clone();
    m
}

/// F2: extract the variable rebindings produced by a nested control-flow
/// instruction so they can be threaded into the enclosing region's env at
/// EVERY nesting level (fn-body, while-body, then-branch, else-branch).
///
/// Returns `(name, exit_id)` pairs where `exit_id` is a DOMINATING value for
/// the enclosing region — the loop's `^while_after` exit block-arg id
/// (`While.exit_ids`) or the if's `^if_after` merge block-arg id (the value in
/// `If.branch_bindings`, which the F2 If lowering sets to the merge id).
///
/// This is the recursion crux: when an if-branch or loop-body contains a nested
/// loop/if that mutates an outer variable, the enclosing region picks up the
/// nested region's EXIT id (not a raw value defined inside the deeper branch),
/// so any value later passed on a branch/back edge is guaranteed to dominate.
#[cfg(feature = "std-surface")]
fn region_exit_rebindings(instr: &Instr) -> Vec<(String, ValueId)> {
    match instr {
        Instr::While {
            live_vars,
            exit_ids,
            ..
        } => live_vars
            .iter()
            .enumerate()
            .map(|(k, (name, post))| {
                let exit = exit_ids.get(k).copied().unwrap_or(*post);
                (name.clone(), exit)
            })
            .collect(),
        Instr::If {
            branch_bindings, ..
        } => branch_bindings.clone(),
        _ => Vec::new(),
    }
}

/// F2: the rebindings produced by the LAST control-flow statement pushed into
/// `instrs`. A `While` arm pushes the `Instr::While` followed by a trailing
/// unit `ConstI64`, so a bare last-instruction check would miss it — we look
/// past that trailing placeholder. An `If` arm pushes only the `Instr::If`
/// (its `dst` is the value), so the last instruction is the node itself.
#[cfg(feature = "std-surface")]
fn last_region_exit_rebindings(instrs: &[Instr]) -> Vec<(String, ValueId)> {
    let n = instrs.len();
    if n == 0 {
        return Vec::new();
    }
    // Direct match (If, or While with no trailing placeholder).
    let direct = region_exit_rebindings(&instrs[n - 1]);
    if !direct.is_empty() {
        return direct;
    }
    // While pushes a trailing unit ConstI64 after the loop node; the loop is
    // the second-to-last instruction.
    if n >= 2 {
        if let Instr::ConstI64(..) = &instrs[n - 1] {
            return region_exit_rebindings(&instrs[n - 2]);
        }
    }
    Vec::new()
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
                    return vec!["i64".to_string()];
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
        TypeAnn::ScalarI32 | TypeAnn::ScalarI64 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            vec!["i64".to_string()]
        }
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
/// This function returns the `Vec<String>` of MLIR type tokens, matching the
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
