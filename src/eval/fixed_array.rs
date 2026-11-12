// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Typed construction for float bindings: fixed arrays whose elements are
//! computed at runtime, and a scalar binding declared `f32` of an `f64` value.

use crate::ast::{Node, TypeAnn};
use crate::ir::{BinOp, IRModule, Instr, ValueId};
use crate::types::{DType, ShapeDim};

fn const_i64(node: &Node) -> Option<i64> {
    crate::opt::fold::const_int(node)
}

fn const_f64(node: &Node) -> Option<f64> {
    crate::opt::fold::const_float(node)
}

/// The `ConstArray` values of a named constant array literal; `None` when the
/// initializer is not an array literal or an element is not a constant.
pub(super) fn const_array_values(value: &Node) -> Option<Vec<i64>> {
    let Node::ArrayLit { elements, .. } = value else {
        return None;
    };
    elements.iter().map(const_i64).collect()
}

fn lower_dense(
    element: &TypeAnn,
    length: u32,
    value: &Node,
    ir: &mut IRModule,
    context: &mut super::LoweringContext,
) -> Option<ValueId> {
    let dtype = match element {
        TypeAnn::ScalarF64 => DType::F64,
        TypeAnn::ScalarF32 => DType::F32,
        TypeAnn::Named(name) if name == "f64" => DType::F64,
        TypeAnn::Named(name) if name == "f32" => DType::F32,
        _ => return None,
    };
    let Node::ArrayLit { elements, .. } = value else {
        return None;
    };
    if elements.len() != length as usize || elements.iter().any(|item| const_f64(item).is_none()) {
        return None;
    }
    if !context.charge_fixed_literal(length, false) {
        return None;
    }
    let data = elements
        .iter()
        .map(|item| match dtype {
            DType::F32 => (const_f64(item).unwrap() as f32).to_bits() as u64,
            DType::F64 => const_f64(item).unwrap().to_bits(),
            _ => unreachable!("fixed float arrays use only f32/f64"),
        })
        .collect();
    let dst = ir.fresh();
    ir.instrs.push(Instr::ConstDenseTensor {
        dst,
        dtype,
        shape: vec![ShapeDim::Known(length as usize)],
        data,
    });
    Some(dst)
}

fn is_i64_abi_element(element: &TypeAnn, ir: &IRModule) -> bool {
    match element {
        TypeAnn::ScalarI64 | TypeAnn::ScalarBool => true,
        TypeAnn::Named(name) => {
            matches!(
                name.as_str(),
                "i64" | "u64" | "i32" | "u32" | "i16" | "u16" | "i8" | "u8" | "bool"
            ) || ir.struct_defs.contains_key(name)
                || name
                    .rsplit('.')
                    .next()
                    .is_some_and(|bare| ir.struct_defs.contains_key(bare))
        }
        _ => false,
    }
}

/// True when `value` is a fixed-array literal that cannot use `ConstArray`'s
/// literal-only payload and can be represented by the i64 aggregate ABI.
pub(crate) fn needs_runtime_construction(
    element: &TypeAnn,
    length: u32,
    value: &Node,
    ir: &IRModule,
) -> bool {
    let Node::ArrayLit { elements, .. } = value else {
        return false;
    };
    elements.len() == length as usize
        && is_i64_abi_element(element, ir)
        && elements.iter().any(|item| const_i64(item).is_none())
}

/// Build `[T; N]` from a zero scaffold followed by ordered `ArrayStore`s.
///
/// `ConstArray` establishes the fixed `tensor<Nxi64>` representation. Each
/// source element is then lowered exactly once, in source order, and inserted
/// into a fresh aggregate incarnation. This is the same value-semantic path as
/// ordinary fixed-array assignment and keeps struct elements as their opaque
/// i64 record handles. Constant scalar arrays return `None` so their historical
/// compact `ConstArray` bytes remain unchanged. Float arrays return `None` and
/// stay on the typed dense path; this helper never coerces them to i64.
pub(crate) fn lower_runtime_construction(
    element: &TypeAnn,
    length: u32,
    value: &Node,
    ir: &mut IRModule,
    mut lower_element: impl FnMut(&Node, &mut IRModule, &mut super::LoweringContext) -> ValueId,
    context: &mut super::LoweringContext,
) -> Option<ValueId> {
    if !needs_runtime_construction(element, length, value, ir) {
        return None;
    }
    let Node::ArrayLit { elements, .. } = value else {
        return None;
    };
    if !context.charge_fixed_literal(length, true) {
        return None;
    }

    let mut aggregate = ir.fresh();
    ir.instrs.push(Instr::ConstArray {
        dst: aggregate,
        name: None,
        values: vec![0; elements.len()],
    });
    for (position, item) in elements.iter().enumerate() {
        let item_id = lower_element(item, ir, context);
        if context.failed() {
            return None;
        }
        let index = ir.fresh();
        ir.instrs.push(Instr::ConstI64(index, position as i64));
        let next = ir.fresh();
        ir.instrs.push(Instr::ArrayStore {
            dst: next,
            base: aggregate,
            index,
            value: item_id,
        });
        aggregate = next;
    }
    Some(aggregate)
}

/// Lower a typed `[T; N]` binding through its representation-specific path.
///
/// Constant float literals retain the typed dense representation. Runtime i64
/// ABI values use ordered aggregate insertion. Everything else delegates to the
/// ordinary expression lowerer, which preserves existing behavior and errors.
pub(crate) fn lower_binding(
    element: &TypeAnn,
    length: u32,
    value: &Node,
    ir: &mut IRModule,
    mut lower_node: impl FnMut(&Node, &mut IRModule, &mut super::LoweringContext) -> ValueId,
    context: &mut super::LoweringContext,
) -> ValueId {
    if let Some(dense) = lower_dense(element, length, value, ir, context) {
        return dense;
    }
    if let Some(runtime) =
        lower_runtime_construction(element, length, value, ir, &mut lower_node, context)
    {
        return runtime;
    }
    lower_node(value, ir, context)
}

/// A binding declared `f32` (a `let`, an assignment) takes an `f64` value
/// through `__mind_conv_f32` (`arith.truncf`, IEEE round to nearest even),
/// the conversion `x as f32` emits: the declared width governs the binding, as
/// `let a: i8 = 300` wraps. Only a value proven `f64` (`is_f64`) converts. Any
/// other value, and any other declared type, returns `val` unchanged
/// (byte-identical); an unproven `f64` value stays a loud `mlir-opt` refusal.
pub(super) fn f32_literal(ir: &mut IRModule, ty: &TypeAnn, val: ValueId) -> ValueId {
    if super::scalar_float_cast_width(ty) != Some(32) || !is_f64(ir, val) {
        return val;
    }
    let conv = ir.fresh();
    let name = "__mind_conv_f32".to_string();
    ir.instrs.push(Instr::legacy_call(conv, name, vec![val]));
    conv
}

/// True when `id` is proven `f64`: an `f64` constant, a `__mind_conv_f64` call,
/// a call of a fn declared `-> f64`, a parameter declared `f64`, or `+ - * / %`
/// with an `f64` operand (both operands of a float op have one type).
fn is_f64(ir: &IRModule, id: ValueId) -> bool {
    let def = ir
        .instrs
        .iter()
        .rev()
        .find(|i| crate::ir::instruction_dst(i) == Some(id));
    let declared_f64 =
        |ty: Option<&TypeAnn>| ty.and_then(super::scalar_float_cast_width) == Some(64);
    match def {
        Some(Instr::ConstF64(..)) => true,
        Some(Instr::Call { name, .. }) => {
            name == "__mind_conv_f64"
                || declared_f64(ir.fn_signatures.get(name).and_then(|(_, r)| r.as_ref()))
        }
        // Params are recorded by name at fn entry (`enter_narrow_scope`); a
        // shadowing `let` replaces or removes it, so a stale id is never proven.
        Some(Instr::Param { name, .. }) => declared_f64(
            super::NARROW_LOCALS
                .with(|n| n.borrow().get(name).cloned())
                .as_ref(),
        ),
        Some(Instr::BinOp { op, lhs, rhs, .. }) => {
            matches!(
                op,
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
            ) && (is_f64(ir, *lhs) || is_f64(ir, *rhs))
        }
        _ => false,
    }
}
