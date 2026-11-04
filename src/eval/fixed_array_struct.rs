// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Fixed-array struct-field ABI layout, materialization, and scalar access.
//!
//! This focused module keeps length-driven field logic out of the large
//! expression lowerer while retaining the same private lowering context.

use super::{
    HashMap, fixed_array, is_array_surface_ty, lower_array_surface_lit, lower_expr,
    receiver_struct_type, struct_key_for,
};
use crate::ast::{self, Literal, TypeAnn};
use crate::ir::{BinOp, IRModule, Instr, ValueId};

/// The byte width and signedness of a struct field type for the canonical
/// width-aware struct ABI. Returns `(width_bytes, signed)`:
///   * `i64`/`u64`/struct-handle (`Named` non-narrow)/pointer  → 8, signed
///   * `i32`/`u32`                                             → 4
///   * `i16`/`u16`                                             → 2
///   * `i8`/`u8`/`bool`                                        → 1
///
/// `signed` is true only for the signed integer scalars (`i64`/`i32`/`i16`/`i8`);
/// `u*`/`bool`/handles are unsigned (zero-extended on load). Any field that is
/// not a recognised scalar (a nested struct handle, a `Vec`/`String`/`Map`
/// handle, etc.) is an i64-wide handle.
#[cfg(feature = "std-surface")]
pub(super) fn struct_field_width(ty: &TypeAnn) -> (i64, bool) {
    match ty {
        // Fixed arrays are stored inline as one canonical 8-byte cell per
        // element.  The aggregate itself is an SSA tensor; the record owns a
        // value-copy of its scalar cells, so a field read can reconstruct a
        // fresh tensor without treating an i64 record slot as a tensor.
        TypeAnn::Array { length, .. } => (8 * i64::from((*length).max(1)), false),
        TypeAnn::ScalarI64 => (8, true),
        TypeAnn::ScalarI32 => (4, true),
        TypeAnn::ScalarU32 => (4, false),
        TypeAnn::ScalarBool => (1, false),
        TypeAnn::Named(n) => match n.as_str() {
            "i8" => (1, true),
            "u8" => (1, false),
            "i16" => (2, true),
            "u16" => (2, false),
            "i32" => (4, true),
            "u32" => (4, false),
            "i64" => (8, true),
            // u64 and every other Named type (nested struct / Vec / String /
            // Map handle, type alias) is an i64-wide value.
            _ => (8, false),
        },
        // Floats are handled by the existing loud lowering error, not here;
        // anything else is treated as an i64-wide handle (8 bytes).
        _ => (8, false),
    }
}

#[cfg(feature = "std-surface")]
pub(super) fn struct_field_type<'a>(
    ir: &'a IRModule,
    struct_name: &str,
    idx: usize,
) -> Option<&'a TypeAnn> {
    ir.struct_field_types.get(struct_name)?.get(idx)
}

#[cfg(feature = "std-surface")]
pub(super) fn fixed_array_cell_type(ty: &TypeAnn) -> Option<(&TypeAnn, u32)> {
    match ty {
        TypeAnn::Array { element, length } => Some((element.as_ref(), *length)),
        _ => None,
    }
}

#[cfg(feature = "std-surface")]
pub(super) fn fixed_array_cell_bits_ty(ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::ScalarF64 => true,
        TypeAnn::Named(name) => name == "f64",
        _ => false,
    }
}

#[cfg(feature = "std-surface")]
pub(super) fn fixed_array_cell_supported(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::ScalarI64 | TypeAnn::ScalarF64)
        || matches!(ty, TypeAnn::Named(name) if matches!(
            name.as_str(),
            "i8" | "u8" | "i16" | "u16"
        ))
}

#[cfg(feature = "std-surface")]
pub(super) fn fixed_array_cell_supported_in(ty: &TypeAnn, ir: &IRModule) -> bool {
    let _ = ir;
    fixed_array_cell_supported(ty)
        || matches!(ty, TypeAnn::Named(name) if matches!(name.as_str(), "i64" | "f64"))
}

/// Return whether a resolved struct field is a fixed array whose element ABI
/// is outside the currently runnable cell representation. Shared by the
/// lowering fallback and runnable-artifact gate so both use the resolved IR
/// schema rather than independently guessing from declarations.
pub(crate) fn fixed_array_field_unsupported(ir: &IRModule, struct_name: &str, field: &str) -> bool {
    let Some(fields) = ir.struct_defs.get(struct_name) else {
        return false;
    };
    let Some(idx) = fields.iter().position(|name| name == field) else {
        return false;
    };
    let Some(TypeAnn::Array { element, .. }) = ir
        .struct_field_types
        .get(struct_name)
        .and_then(|types| types.get(idx))
    else {
        return false;
    };
    !fixed_array_cell_supported_in(element, ir)
}

/// Stop lowering after a field operation reaches a representation that the
/// inline scalar-cell ABI cannot describe. The runnable ABI gate normally
/// reports this after lowering, but nested array-of-struct accesses can leave
/// the receiver unresolved before that pass. Use the terminal structured
/// refusal path so callers get an E6009 diagnostic and no partial artifact,
/// rather than the historical unresolved-receiver panic or fabricated zero.
#[cfg(feature = "std-surface")]
pub(super) fn refuse_unrepresentable_field(
    ir: &mut IRModule,
    context: &mut super::LoweringContext,
    operation: &'static str,
    span: &ast::Span,
) -> ValueId {
    // Preserve the first terminal failure even if an enclosing expression
    // visits another field before the invocation returns its refusal.
    context.refusal.get_or_insert(
        crate::eval::materialization::MaterializationRefusal::UnsupportedLoweringOperation {
            operation,
            start: span.start(),
            end: span.end(),
        },
    );
    ir.fresh()
}

#[cfg(feature = "std-surface")]
pub(super) fn refuse_invalid_field_receiver(
    ir: &mut IRModule,
    context: &mut super::LoweringContext,
    operation: &'static str,
    span: &ast::Span,
) -> ValueId {
    // Preserve the first terminal failure even if an enclosing expression
    // visits another field before the invocation returns its refusal.
    context.refusal.get_or_insert(
        crate::eval::materialization::MaterializationRefusal::InvalidLoweringReceiver {
            operation,
            start: span.start(),
            end: span.end(),
        },
    );
    ir.fresh()
}

#[cfg(feature = "std-surface")]
pub(super) fn struct_field_alignment(ty: &TypeAnn) -> i64 {
    if matches!(ty, TypeAnn::Array { .. }) {
        8
    } else {
        struct_field_width(ty).0
    }
}

/// The `__mind_store_i{N}` intrinsic name for a field byte width.
#[cfg(feature = "std-surface")]
pub(super) fn store_helper_for_width(width: i64) -> &'static str {
    match width {
        1 => "__mind_store_i8",
        2 => "__mind_store_i16",
        4 => "__mind_store_i32",
        _ => "__mind_store_i64",
    }
}

/// The `__mind_load_i{N}` intrinsic name for a field byte width.
#[cfg(feature = "std-surface")]
pub(super) fn load_helper_for_width(width: i64) -> &'static str {
    match width {
        1 => "__mind_load_i8",
        2 => "__mind_load_i16",
        4 => "__mind_load_i32",
        _ => "__mind_load_i64",
    }
}

/// Canonical per-field layout for a struct: `(offset, width_bytes, signed)` in
/// declaration order, plus the total allocation size. Offsets are a pure
/// function of the declared field widths (scalar fields use their width;
/// fixed-array fields use 8-byte cell alignment), so the layout is identical on every
/// substrate — no host `sizeof`/`alignof`, no target-dependent padding. Returns
/// `None` when the field-type side-table has no entry for `name` (an unknown /
/// forward-referenced struct), so callers fall back to the legacy 8-byte-stride
/// path. `all_i64` is true when every field is 8 bytes wide AND tightly packed
/// at `8*i` — the case where the legacy `__mind_alloc(8*n)` + `store_i64` IR is
/// byte-identical and must be preserved verbatim.
/// One field's resolved placement within a struct: `(byte_offset, width_bytes,
/// signed)`. Offsets are self-aligned and substrate-independent (see
/// `struct_layout`).
#[cfg(feature = "std-surface")]
pub(super) type FieldPlacement = (i64, i64, bool);

/// A struct's fully-resolved layout: each field's placement in declaration
/// order, the total allocation size in bytes, and `all_i64` (every field is an
/// 8-byte tightly-packed slot — the legacy byte-identical `__mind_alloc(8*n)`
/// path).
#[cfg(feature = "std-surface")]
pub(super) type StructLayout = (Vec<FieldPlacement>, i64, bool);

#[cfg(feature = "std-surface")]
pub(super) fn struct_layout(ir: &IRModule, name: &str) -> Option<StructLayout> {
    let field_types = ir.struct_field_types.get(name)?;
    let mut layout = Vec::with_capacity(field_types.len());
    let mut running: i64 = 0;
    let mut all_i64 = true;
    for ty in field_types {
        let (w, signed) = struct_field_width(ty);
        // Scalar fields retain their width alignment. Inline aggregate cells
        // are each 8-byte aligned, while `w` remains their total size.
        let alignment = struct_field_alignment(ty);
        let offset = (running + (alignment - 1)) / alignment * alignment;
        // Even a one-element fixed array occupies one 8-byte cell, but it is
        // still an aggregate and must use the element-wise path below rather
        // than the legacy scalar `all_i64` store.
        if matches!(ty, TypeAnn::Array { .. }) || w != 8 || offset != (layout.len() as i64) * 8 {
            all_i64 = false;
        }
        layout.push((offset, w, signed));
        running = offset + w;
    }
    Some((layout, running, all_i64))
}

/// Lower a `StructLit` FIELD value. A field declared `array<T>` whose literal is
/// `[..]` must lower onto the std.vec heap runtime (a registered i64 handle), not
/// the generic `ArrayLit` const-array/tensor path — whose result is a non-i64
/// aggregate the field's `__mind_store_i64` cannot accept (it surfaces as the
/// "non-i64 argument to call" aggregate-ABI error). Every other field value
/// lowers normally.
#[cfg(feature = "std-surface")]
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_struct_field_value(
    struct_name: &str,
    field_name: &str,
    value: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
    context: &mut super::LoweringContext,
) -> ValueId {
    let field_ty = ir
        .struct_defs
        .get(struct_name)
        .and_then(|names| names.iter().position(|n| n == field_name))
        .and_then(|idx| struct_field_type(ir, struct_name, idx));
    if let Some((element, length)) = field_ty
        .and_then(fixed_array_cell_type)
        .map(|(element, length)| (element.clone(), length))
    {
        if fixed_array_cell_supported_in(&element, ir) {
            return fixed_array::lower_binding(
                &element,
                length,
                value,
                ir,
                |node, inner_ir, context| {
                    lower_expr(node, inner_ir, env, struct_env, receiver_types, context)
                },
                context,
            );
        }
    }
    if let ast::Node::ArrayLit { elements, .. } = value {
        let is_arr_field = ir
            .struct_defs
            .get(struct_name)
            .and_then(|names| names.iter().position(|n| n == field_name))
            .and_then(|idx| {
                ir.struct_field_types
                    .get(struct_name)
                    .and_then(|ts| ts.get(idx))
            })
            .map(is_array_surface_ty)
            .unwrap_or(false);
        if is_arr_field {
            return lower_array_surface_lit(elements, ir, env, struct_env, receiver_types, context);
        }
    }
    lower_expr(value, ir, env, struct_env, receiver_types, context)
}

/// Store a fixed-array SSA aggregate into the inline cells owned by a struct
/// record.  A tensor is never passed to a scalar memory intrinsic: each
/// element is extracted from the typed aggregate, optionally converted to its
/// canonical f64 bit representation, and stored in one 8-byte cell. Narrow
/// integer cells use their declared-width memory helper; the cell stride
/// remains eight bytes for ABI compatibility.
#[cfg(feature = "std-surface")]
pub(super) fn store_fixed_array_field(
    element: &TypeAnn,
    length: u32,
    field_addr: ValueId,
    value: ValueId,
    ir: &mut IRModule,
    context: &mut super::LoweringContext,
) {
    if context.failed() {
        return;
    }
    if !fixed_array_cell_supported_in(element, ir) {
        return;
    }
    let f64_bits = matches!(element, TypeAnn::ScalarF64)
        || matches!(element, TypeAnn::Named(name) if name == "f64");
    if !context.charge_fixed_field_store(length, f64_bits) {
        return;
    }
    for position in 0..length {
        let index = ir.fresh();
        ir.instrs.push(Instr::ConstI64(index, i64::from(position)));
        let item = ir.fresh();
        ir.instrs.push(Instr::ArrayLoad {
            dst: item,
            base: value,
            index,
        });
        let stored = if f64_bits {
            let bits = ir.fresh();
            ir.instrs.push(Instr::legacy_call(
                bits,
                "__mind_f64_to_bits".to_string(),
                vec![item],
            ));
            bits
        } else {
            item
        };
        let element_addr = if position == 0 {
            field_addr
        } else {
            let offset = ir.fresh();
            ir.instrs
                .push(Instr::ConstI64(offset, i64::from(position) * 8));
            let sum = ir.fresh();
            ir.instrs.push(Instr::BinOp {
                dst: sum,
                op: BinOp::Add,
                lhs: field_addr,
                rhs: offset,
            });
            sum
        };
        let unit = ir.fresh();
        ir.instrs.push(Instr::legacy_call(
            unit,
            store_helper_for_width(struct_field_width(element).0).to_string(),
            vec![element_addr, stored],
        ));
    }
}

/// Lower `s.xs[i] = value` for an inline fixed-array field.  The field read
/// first creates a value aggregate, then the updated aggregate is copied back
/// into the owning record cells.  This keeps field-index assignment's value
/// semantics explicit and avoids the scalar `lower_expr(IndexAssign)` refusal.
#[cfg(feature = "std-surface")]
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_fixed_array_field_index_assign(
    receiver: &ast::Node,
    index: &ast::Node,
    value: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
    context: &mut super::LoweringContext,
) -> Option<ValueId> {
    if context.failed() {
        return Some(ir.fresh());
    }
    let ast::Node::FieldAccess {
        receiver: owner,
        field,
        span,
    } = receiver
    else {
        return None;
    };
    let struct_name = receiver_types
        .get(span)
        .cloned()
        .or_else(|| receiver_struct_type(owner, ir, struct_env))?;
    let struct_key = struct_key_for(ir, &struct_name).to_string();
    let idx = ir
        .struct_defs
        .get(&struct_key)?
        .iter()
        .position(|f| f == field)?;
    let (element, length) = struct_field_type(ir, &struct_key, idx)
        .and_then(fixed_array_cell_type)
        .map(|(element, length)| (element.clone(), length))?;
    if !fixed_array_cell_supported_in(&element, ir) {
        return None;
    }
    let owner_id = lower_expr(owner, ir, env, struct_env, receiver_types, context);
    if context.failed() {
        return Some(ir.fresh());
    }
    let (offset, _, _) = struct_layout(ir, &struct_key)
        .and_then(|(layout, _, _)| layout.get(idx).copied())
        .unwrap_or(((idx as i64) * 8, 8, true));
    let field_addr = if offset == 0 {
        owner_id
    } else {
        let off = ir.fresh();
        ir.instrs.push(Instr::ConstI64(off, offset));
        let sum = ir.fresh();
        ir.instrs.push(Instr::BinOp {
            dst: sum,
            op: BinOp::Add,
            lhs: owner_id,
            rhs: off,
        });
        sum
    };
    let index_id = lower_expr(index, ir, env, struct_env, receiver_types, context);
    if context.failed() {
        return Some(ir.fresh());
    }
    let value_id = lower_expr(value, ir, env, struct_env, receiver_types, context);
    if context.failed() {
        return Some(ir.fresh());
    }
    let length_id = ir.fresh();
    ir.instrs
        .push(Instr::ConstI64(length_id, i64::from(length)));
    let checked_index = ir.fresh();
    ir.instrs.push(Instr::legacy_call(
        checked_index,
        "__mind_oob_check".to_string(),
        vec![index_id, length_id],
    ));
    let byte_width = ir.fresh();
    ir.instrs.push(Instr::ConstI64(byte_width, 8));
    let byte_offset = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: byte_offset,
        op: BinOp::Mul,
        lhs: checked_index,
        rhs: byte_width,
    });
    let element_addr = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: element_addr,
        op: BinOp::Add,
        lhs: field_addr,
        rhs: byte_offset,
    });
    let stored = if fixed_array_cell_bits_ty(&element) {
        let bits = ir.fresh();
        ir.instrs.push(Instr::legacy_call(
            bits,
            "__mind_f64_to_bits".to_string(),
            vec![value_id],
        ));
        bits
    } else {
        value_id
    };
    let store = ir.fresh();
    ir.instrs.push(Instr::legacy_call(
        store,
        store_helper_for_width(struct_field_width(&element).0).to_string(),
        vec![element_addr, stored],
    ));
    let unit = ir.fresh();
    ir.instrs.push(Instr::ConstI64(unit, 0));
    Some(unit)
}

/// Lower `s.xs[i]` when `xs` is an inline fixed-array struct field. A field
/// read used to reconstruct the whole aggregate before applying `ArrayLoad`,
/// making the cost proportional to the declared length and giving the MLIR
/// backend an i64 record handle where it expected a tensor. Indexed reads are
/// scalar ABI operations: evaluate the owner and index once, check the index,
/// then load exactly one eight-byte cell. Whole-field reads continue through
/// the value-copy path in the `FieldAccess` arm above.
#[cfg(feature = "std-surface")]
pub(super) fn lower_fixed_array_field_index_access(
    receiver: &ast::Node,
    index: &ast::Node,
    ir: &mut IRModule,
    env: &HashMap<String, ValueId>,
    struct_env: &HashMap<String, String>,
    receiver_types: &HashMap<crate::ast::Span, String>,
    context: &mut super::LoweringContext,
) -> Option<ValueId> {
    if context.failed() {
        return Some(ir.fresh());
    }
    let ast::Node::FieldAccess {
        receiver: owner,
        field,
        span,
    } = receiver
    else {
        return None;
    };

    // Keep this resolution in lockstep with the ordinary FieldAccess reader
    // and the indexed-write helper. The side table is authoritative for
    // chained/call receivers; struct_env is the cheap plain-ident path.
    let step1 = match owner.as_ref() {
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
    let step2 = if step1.is_none() {
        receiver_types.get(span).and_then(|struct_name| {
            let key = struct_key_for(ir, struct_name);
            ir.struct_defs
                .get(key)
                .and_then(|fields| fields.iter().position(|f| f == field))
                .map(|idx| (None::<String>, idx, key.to_string()))
        })
    } else {
        None
    };
    let (owner_name, idx, struct_name) = step1.or(step2)?;
    let (element, length) = struct_field_type(ir, &struct_name, idx)
        .and_then(fixed_array_cell_type)
        .map(|(element, length)| (element.clone(), length))?;
    if !fixed_array_cell_supported_in(&element, ir) {
        return None;
    }

    // Lower the receiver before the index, and never lower either expression
    // a second time. The plain-ident path reuses its already-bound address;
    // chained receivers are lowered once through the normal FieldAccess path.
    let owner_id = match owner_name {
        Some(name) => {
            let Some(owner_id) = env.get(&name).copied() else {
                return Some(refuse_invalid_field_receiver(
                    ir,
                    context,
                    "indexed struct field read",
                    span,
                ));
            };
            owner_id
        }
        None => lower_expr(owner, ir, env, struct_env, receiver_types, context),
    };
    if context.failed() {
        return Some(ir.fresh());
    }
    let (offset, _, _) = struct_layout(ir, &struct_name)
        .and_then(|(layout, _, _)| layout.get(idx).copied())
        .unwrap_or(((idx as i64) * 8, 8, true));
    let field_addr = if offset == 0 {
        owner_id
    } else {
        let field_offset = ir.fresh();
        ir.instrs.push(Instr::ConstI64(field_offset, offset));
        let addr = ir.fresh();
        ir.instrs.push(Instr::BinOp {
            dst: addr,
            op: BinOp::Add,
            lhs: owner_id,
            rhs: field_offset,
        });
        addr
    };
    let index_id = lower_expr(index, ir, env, struct_env, receiver_types, context);
    if context.failed() {
        return Some(ir.fresh());
    }
    let length_id = ir.fresh();
    ir.instrs
        .push(Instr::ConstI64(length_id, i64::from(length)));
    let checked_index = ir.fresh();
    ir.instrs.push(Instr::legacy_call(
        checked_index,
        "__mind_oob_check".to_string(),
        vec![index_id, length_id],
    ));
    let byte_width = ir.fresh();
    ir.instrs.push(Instr::ConstI64(byte_width, 8));
    let byte_offset = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: byte_offset,
        op: BinOp::Mul,
        lhs: checked_index,
        rhs: byte_width,
    });
    let element_addr = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: element_addr,
        op: BinOp::Add,
        lhs: field_addr,
        rhs: byte_offset,
    });
    let loaded = ir.fresh();
    ir.instrs.push(Instr::legacy_call(
        loaded,
        load_helper_for_width(struct_field_width(&element).0).to_string(),
        vec![element_addr],
    ));
    if fixed_array_cell_bits_ty(&element) {
        let decoded = ir.fresh();
        ir.instrs.push(Instr::legacy_call(
            decoded,
            "__mind_bits_to_f64".to_string(),
            vec![loaded],
        ));
        Some(decoded)
    } else {
        Some(super::mask_narrow_let(ir, &Some(element), loaded))
    }
}
