// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Struct field read/write lowering kept outside the large expression matcher.
//! The helpers use the canonical fixed-array metadata and transient lowering
//! context owned by [`super`]; they do not introduce an independent resolver.

use super::fixed_array_struct::{
    fixed_array_cell_bits_ty, fixed_array_cell_supported_in, fixed_array_cell_type,
    load_helper_for_width, lower_struct_field_value, refuse_invalid_field_receiver,
    refuse_unrepresentable_field, store_fixed_array_field, store_helper_for_width,
    struct_field_type, struct_field_width, struct_layout, unrepresentable_field,
};
use super::{
    HashMap, LoweringContext, MAP_SENTINEL, MAP_STR_SENTINEL, SET_SENTINEL, SET_STR_SENTINEL,
    lower_expr, mask_narrow_let, receiver_collection_sentinel, receiver_struct_type,
    resolve_numeric_field_receiver_ty, struct_key_for,
};
use crate::ast::{self, Literal, TypeAnn};
use crate::eval::materialization::MaterializationRefusal;
use crate::eval::slice_abi::is_vec_handle_sentinel;
use crate::ir::{BinOp, IRModule, Instr, ValueId};
use crate::project::std_struct_schema::bundled_struct_schema;
use crate::types::{DType, ShapeDim};

#[cfg(feature = "std-surface")]
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_field_access(
    receiver: &ast::Node,
    field: &str,
    span: &ast::Span,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<ast::Span, String>,
    context: &mut LoweringContext,
) -> ValueId {
    // Finding 6: numeric TUPLE index `t.N` (parser now emits it as a
    // FieldAccess whose `field` is an all-digit string). Resolve the
    // receiver's DECLARED `(T, U, …)` tuple annotation from
    // `NARROW_LOCALS`, bounds-check N, and emit the same
    // `__mind_load_i64(base + 8*N)` shape LetTuple destructuring uses —
    // then re-materialise the element at its declared type
    // (`mask_narrow_let`), so a `u64` element compares UNSIGNED and a
    // narrow element keeps its width. Any MISS — non-ident receiver,
    // unannotated/non-tuple binding, out-of-range N — FAILS LOUDLY:
    // falling through to struct resolution would reach the const-0 /
    // ArrayLoad placeholder paths, a guaranteed silent miscompile.
    if !field.is_empty() && field.bytes().all(|b| b.is_ascii_digit()) {
        let idx: usize = field
            .parse()
            .unwrap_or_else(|_| panic!("tuple index `.{field}` is out of range for a tuple type"));
        // Resolve the receiver's tuple element types RECURSIVELY, so a
        // chained numeric index (`t.0.1` on a nested-tuple-typed `t`)
        // resolves the inner tuple's element type — not only a plain
        // `Ident` receiver. The address chain for the outer `.N` is built
        // by `lower_expr(receiver, …)` below, which recurses through this
        // same arm for the inner `.M`.
        let elements: Vec<TypeAnn> = resolve_numeric_field_receiver_ty(receiver)
            .and_then(|t| match t {
                TypeAnn::Tuple { elements } => Some(elements),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!(
                    "tuple index `.{field}` requires a receiver with a declared \
                     tuple annotation — either a local `let t: (T, U) = …`, or a \
                     nested tuple element such as `t.0` where element 0 is itself \
                     a tuple; annotate the receiver's tuple type. (Falling through \
                     to struct-field resolution here would be a silent miscompile.)"
                )
            });
        if idx >= elements.len() {
            panic!(
                "tuple index `.{field}` is out of bounds for a {}-element \
                 tuple",
                elements.len()
            );
        }
        let addr = lower_expr(receiver, ir, env, struct_env, receiver_types, context);
        let elem_addr = if idx == 0 {
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
        let loaded = ir.fresh();
        ir.instrs.push(Instr::legacy_call(
            loaded,
            "__mind_load_i64".to_string(),
            vec![elem_addr],
        ));
        return mask_narrow_let(ir, &Some(elements[idx].clone()), loaded);
    }
    // `array<T>` length: `arr.len` / `arr.length` (no parens) on a
    // vec-sentinel receiver lowers to the std.vec `vec_len` free
    // function. mind-flow writes `.length`; std.vec exposes `vec_len`,
    // so the name is normalised here. The sentinel is only set for
    // `array<T>`-annotated bindings/params, so non-array `.len`/`.length`
    // field reads are unaffected and the keystone never hits this path.
    #[cfg(feature = "std-surface")]
    if field == "len" || field == "length" {
        // `arr.len`/`.length` → vec_len; `m.len`/`s.len` → map_len.
        // mind-flow writes `.length`; std exposes `vec_len`/`map_len`, so
        // the name is normalised. Resolves an Ident-bound collection AND a
        // struct-FIELD collection (`a.ids.len`) via the unified resolver,
        // so non-collection field reads are unaffected.
        #[cfg(feature = "std-surface")]
        {
            let len_fn =
                match receiver_collection_sentinel(receiver, ir, struct_env, receiver_types) {
                    Some(s) if is_vec_handle_sentinel(s) => Some("vec_len"),
                    Some(MAP_SENTINEL) | Some(MAP_STR_SENTINEL) => Some("map_len"),
                    Some(SET_SENTINEL) | Some(SET_STR_SENTINEL) => Some("map_len"),
                    _ => None,
                };
            if let Some(len_fn) = len_fn {
                let recv_id = lower_expr(receiver, ir, env, struct_env, receiver_types, context);
                let dst = ir.fresh();
                ir.instrs
                    .push(Instr::legacy_call(dst, len_fn.to_string(), vec![recv_id]));
                return dst;
            }
        }
    }
    let resolved =
        resolve_field(receiver, field, span, ir, struct_env, receiver_types).or_else(|| {
            let refusal = unrepresentable_field("struct field read", span);
            resolve_string_field(
                receiver,
                field,
                refusal,
                ir,
                struct_env,
                receiver_types,
                context,
            )
        });

    match resolved {
        Some((var_name_opt, idx, struct_name)) => {
            // Width-aware field offset/load. `(offset, width, signed)`;
            // an all-i64 struct yields `(idx*8, 8, _)` so the emitted IR
            // is byte-identical to the legacy path. Falls back to the
            // legacy `idx*8` / i64 when the layout is unknown.
            let fld = struct_layout(ir, &struct_name).and_then(|(l, _, _)| l.get(idx).copied());
            let (offset, width, signed) = fld.unwrap_or(((idx as i64) * 8, 8, true));
            // Step 1 path can take addr from env without re-lowering.
            // Step 2 path must lower the receiver expression to
            // get its base address (it may be a Call, FieldAccess,
            // or anything else that evaluates to an i64 heap addr).
            let addr = match var_name_opt {
                Some(var_name) => match env.get(&var_name) {
                    Some(id) => *id,
                    None => refuse_invalid_field_receiver(ir, context, "struct field read", span),
                },
                None => lower_expr(receiver, ir, env, struct_env, receiver_types, context),
            };
            let field_addr = if offset == 0 {
                addr
            } else {
                let off = ir.fresh();
                ir.instrs.push(Instr::ConstI64(off, offset));
                let sum = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst: sum,
                    op: BinOp::Add,
                    lhs: addr,
                    rhs: off,
                });
                sum
            };
            if let Some((element, length)) = struct_field_type(ir, &struct_name, idx)
                .and_then(fixed_array_cell_type)
                .map(|(element, length)| (element.clone(), length))
                .filter(|(element, _)| fixed_array_cell_supported_in(element, ir))
            {
                if !context.charge_fixed_field_copyout(length, fixed_array_cell_bits_ty(&element)) {
                    return ir.fresh();
                }
                let aggregate = ir.fresh();
                if fixed_array_cell_bits_ty(&element) {
                    // f64 cells are kept as raw bits in the record;
                    // start with a typed f64 aggregate so ArrayStore
                    // receives an f64 scalar after the inverse bitcast.
                    ir.instrs.push(Instr::ConstDenseTensor {
                        dst: aggregate,
                        dtype: DType::F64,
                        shape: vec![ShapeDim::Known(length as usize)],
                        data: vec![0; length as usize],
                    });
                } else {
                    ir.instrs.push(Instr::ConstArray {
                        dst: aggregate,
                        name: None,
                        values: vec![0; length as usize],
                    });
                }
                let mut current = aggregate;
                for position in 0..length {
                    let cell_addr = if position == 0 {
                        field_addr
                    } else {
                        let cell_offset = ir.fresh();
                        ir.instrs
                            .push(Instr::ConstI64(cell_offset, i64::from(position) * 8));
                        let sum = ir.fresh();
                        ir.instrs.push(Instr::BinOp {
                            dst: sum,
                            op: BinOp::Add,
                            lhs: field_addr,
                            rhs: cell_offset,
                        });
                        sum
                    };
                    let bits = ir.fresh();
                    ir.instrs.push(Instr::legacy_call(
                        bits,
                        load_helper_for_width(struct_field_width(&element).0).to_string(),
                        vec![cell_addr],
                    ));
                    let item = if fixed_array_cell_bits_ty(&element) {
                        let decoded = ir.fresh();
                        ir.instrs.push(Instr::legacy_call(
                            decoded,
                            "__mind_bits_to_f64".to_string(),
                            vec![bits],
                        ));
                        decoded
                    } else {
                        bits
                    };
                    let item = mask_narrow_let(ir, &Some(element.clone()), item);
                    let index = ir.fresh();
                    ir.instrs.push(Instr::ConstI64(index, i64::from(position)));
                    let next = ir.fresh();
                    ir.instrs.push(Instr::ArrayStore {
                        dst: next,
                        base: current,
                        index,
                        value: item,
                    });
                    current = next;
                }
                return current;
            }
            let loaded = ir.fresh();
            ir.instrs.push(Instr::legacy_call(
                loaded,
                load_helper_for_width(width).to_string(),
                vec![field_addr],
            ));
            // The typed load zero-extends; a SIGNED narrow field needs a
            // sign-extend (shl then arithmetic shr by 64-bits). i64 and
            // unsigned/bool fields use the zero-extended value directly,
            // so this is byte-identical for the all-i64 path.
            if signed && width < 8 {
                let shift = 64 - width * 8;
                let sh = ir.fresh();
                ir.instrs.push(Instr::ConstI64(sh, shift));
                let shl = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst: shl,
                    op: BinOp::Shl,
                    lhs: loaded,
                    rhs: sh,
                });
                let sar = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst: sar,
                    op: BinOp::Shr,
                    lhs: shl,
                    rhs: sh,
                });
                sar
            } else {
                loaded
            }
        }
        None => refuse_unrepresentable_field(ir, context, "struct field read", span),
    }
}

#[cfg(feature = "std-surface")]
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_field_assign(
    receiver: &ast::Node,
    field: &str,
    value: &ast::Node,
    span: &ast::Span,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<ast::Span, String>,
    context: &mut LoweringContext,
) -> ValueId {
    let resolved =
        resolve_field(receiver, field, span, ir, struct_env, receiver_types).or_else(|| {
            let refusal = unrepresentable_field("struct field assignment", span);
            resolve_string_field(
                receiver,
                field,
                refusal,
                ir,
                struct_env,
                receiver_types,
                context,
            )
        });

    match resolved {
        Some((var_name_opt, idx, struct_name)) => {
            // Width-aware field offset/store; all-i64 yields (idx*8, 8)
            // so the IR is byte-identical to the legacy path.
            let fld = struct_layout(ir, &struct_name).and_then(|(l, _, _)| l.get(idx).copied());
            let (offset, width, _signed) = fld.unwrap_or(((idx as i64) * 8, 8, true));
            // Step 1 takes the base addr straight from env (no
            // re-lowering); Step 2 must lower the receiver to get it.
            let addr = match var_name_opt {
                Some(var_name) => match env.get(&var_name) {
                    Some(id) => *id,
                    None => {
                        refuse_invalid_field_receiver(ir, context, "struct field assignment", span)
                    }
                },
                None => lower_expr(receiver, ir, env, struct_env, receiver_types, context),
            };
            let field_addr = if offset == 0 {
                addr
            } else {
                let off = ir.fresh();
                ir.instrs.push(Instr::ConstI64(off, offset));
                let sum = ir.fresh();
                ir.instrs.push(Instr::BinOp {
                    dst: sum,
                    op: BinOp::Add,
                    lhs: addr,
                    rhs: off,
                });
                sum
            };
            if let Some((element, length)) = struct_field_type(ir, &struct_name, idx)
                .and_then(fixed_array_cell_type)
                .map(|(element, length)| (element.clone(), length))
                .filter(|(element, _)| fixed_array_cell_supported_in(element, ir))
            {
                let rhs = lower_struct_field_value(
                    &struct_name,
                    field,
                    value,
                    ir,
                    env,
                    struct_env,
                    receiver_types,
                    context,
                );
                store_fixed_array_field(&element, length, field_addr, rhs, ir, context);
                if context.failed() {
                    return ir.fresh();
                }
                let unit = ir.fresh();
                ir.instrs.push(Instr::ConstI64(unit, 0));
                return unit;
            }
            let rhs = lower_expr(value, ir, env, struct_env, receiver_types, context);
            let store_ret = ir.fresh();
            ir.instrs.push(Instr::legacy_call(
                store_ret,
                store_helper_for_width(width).to_string(),
                vec![field_addr, rhs],
            ));
            // A field assignment is a statement; the store's return
            // (unit) id is the value this expression yields.
            store_ret
        }
        None => refuse_unrepresentable_field(ir, context, "struct field assignment", span),
    }
}

/// The std string type: its struct name, and the `struct_env` sentinel of a
/// string-typed binding.
const STD_STRING: &str = "String";

/// A resolved field: the receiver's binding when it is a plain identifier, the
/// field's index, and the schema key the field resolved against.
type FieldSlot = (Option<String>, usize, String);

/// Resolve `field` against the schema of the receiver's declared struct type.
#[cfg(feature = "std-surface")]
fn resolve_field(
    receiver: &ast::Node,
    field: &str,
    span: &ast::Span,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<ast::Span, String>,
) -> Option<FieldSlot> {
    // ── Step 1: cheap Ident-bound lookup ─────────────────────
    // `struct_key_for` normalizes a module-qualified receiver type
    // (`contract.DispatchResult`) to the bare key its schema is
    // registered under, so a cross-module struct's field offset
    // resolves (else the receiver falls to a null-address placeholder
    // and any collection field read segfaults). Identity for same-file
    // names — the keystone is byte-identical.
    let step1 = match receiver {
        ast::Node::Lit(Literal::Ident(var_name), _) => {
            struct_env.get(var_name).and_then(|struct_name| {
                let key = struct_key_for(ir, struct_name);
                ir.struct_defs
                    .get(key)
                    .and_then(|fields| fields.iter().position(|f| f == field))
                    .map(|idx| (Some(var_name.clone()), idx, key.to_string()))
            })
        }
        _ => None,
    };
    // ── Step 2: side-table fallback (general path) ───────────
    // Only consulted when Step 1 fast-path failed.
    step1.or_else(|| {
        receiver_types.get(span).and_then(|struct_name| {
            let key = struct_key_for(ir, struct_name);
            ir.struct_defs
                .get(key)
                .and_then(|fields| fields.iter().position(|f| f == field))
                .map(|idx| (None::<String>, idx, key.to_string()))
        })
    })
}

/// Resolve `field` on a `String` value against the bundled `std/string.mind`
/// schema, the layout every compiler-built string has. A project build
/// installs that schema; a single translation unit has no project registry, so
/// unless the file declares `String` itself the schema is registered here, and
/// `refusal` (what this access reported without it) is kept for
/// [`refuse_shadowed_std_string`]. A registered `String` with another layout
/// is the program's own, and a string value is never read through it.
#[cfg(feature = "std-surface")]
fn resolve_string_field(
    receiver: &ast::Node,
    field: &str,
    refusal: MaterializationRefusal,
    ir: &mut IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<ast::Span, String>,
    context: &mut LoweringContext,
) -> Option<FieldSlot> {
    if !receiver_is_string(receiver, ir, struct_env, receiver_types) {
        return None;
    }
    let (names, types) = bundled_struct_schema(STD_STRING)?;
    let idx = names.iter().position(|f| f == field)?;
    match ir.struct_defs.get(STD_STRING) {
        Some(registered) => {
            if registered != names || ir.struct_field_types.get(STD_STRING) != Some(types) {
                return None;
            }
        }
        None => {
            ir.struct_defs.insert(STD_STRING.to_string(), names.clone());
            ir.struct_field_types
                .insert(STD_STRING.to_string(), types.clone());
            context.std_string_seed.get_or_insert(refusal);
        }
    }
    let binding = match receiver {
        ast::Node::Lit(Literal::Ident(var_name), _) => Some(var_name.clone()),
        _ => None,
    };
    Some((binding, idx, STD_STRING.to_string()))
}

/// A module that declares its own `String` after a field access lowered
/// against the bundled schema, or inside a `module { … }` block (which this
/// lowering never registers), does not use that layout: restore the refusal
/// the access had before the schema was registered.
#[cfg(feature = "std-surface")]
pub(super) fn refuse_shadowed_std_string(items: &[ast::Node], context: &mut LoweringContext) {
    fn declares(items: &[ast::Node]) -> bool {
        items.iter().any(|item| match item {
            ast::Node::StructDef { name, .. }
            | ast::Node::EnumDef { name, .. }
            | ast::Node::TypeAlias { name, .. } => name == STD_STRING,
            ast::Node::Block { stmts, .. } => declares(stmts),
            _ => false,
        })
    }
    if let Some(refusal) = context.std_string_seed.take() {
        if declares(items) {
            context.refusal.get_or_insert(refusal);
        }
    }
}

/// True when a receiver is the std `String` type — an Ident bound to a string
/// (`struct_env` sentinel `"String"`, set for string lets/params and for-each
/// elements over `array<string>`), a struct-FIELD whose declared type is
/// `string`, or a typed function call returning `string`. A string receiver's
/// methods route to the `string_<method>` std free functions
/// (`.split`→string_split, `.trim`→string_trim, etc.), and its fields resolve
/// against the `String` schema (`resolve_string_field`).
#[cfg(feature = "std-surface")]
pub(super) fn receiver_is_string(
    receiver: &ast::Node,
    ir: &IRModule,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
) -> bool {
    match receiver {
        // String literals are materialized by the `Literal::Str` lowering arm
        // into the same three-field String record as every other string value.
        // Classify them here so a direct `"text".len()` / `.split(...)` call
        // takes the same std UFCS path as an identifier-bound String.
        ast::Node::Lit(Literal::Str(_), _) => true,
        ast::Node::Lit(Literal::Ident(v), _) => struct_env
            .get(v)
            .map(|s| s == "String" || s == "string")
            .unwrap_or(false),
        ast::Node::FieldAccess {
            receiver: base,
            field,
            span,
        } => {
            let sname = receiver_types
                .get(span)
                .cloned()
                .or_else(|| receiver_struct_type(base, ir, struct_env));
            sname
                .and_then(|sn| {
                    let idx = ir.struct_defs.get(&sn)?.iter().position(|f| f == field)?;
                    ir.struct_field_types.get(&sn)?.get(idx).cloned()
                })
                .map(|ty| matches!(ty, TypeAnn::Named(n) if n == "string" || n == "String"))
                .unwrap_or(false)
        }
        ast::Node::Call { callee, .. } => ir
            .fn_signatures
            .get(callee)
            .and_then(|(_, ret)| ret.clone())
            .or_else(|| crate::ir::with_global_enums(|g| g.fn_returns.get(callee).cloned()))
            .is_some_and(|ty| matches!(ty, TypeAnn::Named(n) if n == "string" || n == "String")),
        _ => false,
    }
}
