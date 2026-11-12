// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Narrow `extern "C"` arguments, and the callee of a call through an address.
//!
//! Each narrow parameter is declared at its C width. Signed names use
//! `llvm.signext`. Unsigned names and `bool` use `llvm.zeroext`. `bool`
//! is `i1`. The call passes the signless width. An `f64` argument to an
//! `f32` parameter is converted with `arith.truncf`. A `bind(.address)` call
//! has no declaration to carry those attributes, so it extends each narrow
//! argument to 32 bits itself (spec "Extern bindings").

use super::{LoweringContext, ValueKind};
use crate::ir::ValueId;

/// The primitive row of a narrow integer token; `None` for any other token.
pub(super) fn narrow(token: &str) -> Option<crate::types::value::ScalarDesc> {
    crate::types::value::prim_scalar(token).filter(|d| d.is_narrow_int())
}

pub(super) fn call_ty(token: &str) -> &str {
    match narrow(token) {
        Some(d) if d.is_bool => "i1",
        Some(d) => d.signless(),
        None => token,
    }
}

pub(super) fn decl_param(token: &str) -> String {
    let ty = call_ty(token);
    let attr = narrow(token).map(|d| {
        if d.signed {
            "llvm.signext"
        } else {
            "llvm.zeroext"
        }
    });
    match attr {
        Some(attr) => format!("{ty} {{{attr}}}"),
        None => ty.to_string(),
    }
}

/// Record the value kind of an extern call result. An `f32`/`f64` return keeps
/// its float kind, so a use reads it at its real type; every other return (a
/// widened narrow integer, a pointer address, `i64`) is in the i64 slot.
pub(super) fn record_ret(ctx: &mut LoweringContext, dst: ValueId, call_ret_ty: &str) {
    let kind = match call_ret_ty {
        "f32" => ValueKind::ScalarF32,
        "f64" => ValueKind::ScalarF64,
        _ => ValueKind::ScalarI64,
    };
    ctx.values.insert(dst, kind);
}

fn phys_ty(ctx: &LoweringContext, id: ValueId) -> &'static str {
    if ctx.i1_values.contains(&id) {
        return "i1";
    }
    match ctx.values.get(&id) {
        Some(ValueKind::ScalarI32 | ValueKind::ScalarU32) => "i32",
        Some(ValueKind::ScalarF32) => "f32",
        Some(ValueKind::ScalarF64) => "f64",
        _ => "i64",
    }
}

fn bits(ty: &str) -> Option<u32> {
    match ty {
        "i1" => Some(1),
        "i8" => Some(8),
        "i16" => Some(16),
        "i32" => Some(32),
        "i64" => Some(64),
        _ => None,
    }
}

/// The declared type of each argument position of an extern call: the
/// declared parameter, else (a varargs position) the hint at its offset, else
/// `i64`. RFC 0010 Phase C / R-03: an `f32` in a varargs position is promoted
/// to `f64` (C11 6.5.2.2p6 default argument promotions).
pub(super) fn call_param_types(
    declared: &[String],
    hints: &[String],
    n_args: usize,
) -> Vec<String> {
    (0..n_args)
        .map(|i| match declared.get(i) {
            Some(ty) => ty.clone(),
            None => match hints.get(i - declared.len()).map(String::as_str) {
                Some("f32") => "f64".to_string(),
                Some(hint) => hint.to_string(),
                None => "i64".to_string(),
            },
        })
        .collect()
}

pub(super) fn adapt_call_args(
    ctx: &mut LoweringContext,
    dst: ValueId,
    args: &[ValueId],
    declared: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut refs = Vec::with_capacity(args.len());
    let mut types = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        let token = declared.get(i).map(String::as_str).unwrap_or("i64");
        let target = call_ty(token);
        types.push(target.to_string());
        if target == "!llvm.ptr" {
            let name = format!("%ptrarg_{}_{i}", dst.0);
            ctx.emit_line(&format!(
                "    {name} = llvm.inttoptr %{} : i64 to !llvm.ptr",
                arg.0
            ));
            refs.push(name);
            continue;
        }
        let phys = phys_ty(ctx, *arg);
        if phys == target {
            refs.push(format!("%{}", arg.0));
            continue;
        }
        let widened = match (bits(phys), bits(target)) {
            // The declared width governs: an f64 into an `f32` slot is `as f32`.
            _ if (phys, target) == ("f64", "f32") => Some("arith.truncf"),
            (Some(from), Some(to)) if from > to => Some("arith.trunci"),
            (Some(from), Some(to)) if from < to => Some(widen_op(ctx, *arg)),
            _ => None,
        };
        let Some(op) = widened else {
            refs.push(format!("%{}", arg.0));
            continue;
        };
        let name = format!("%extarg_{}_{i}", dst.0);
        ctx.emit_line(&format!(
            "    {name} = {op} %{} : {phys} to {target}",
            arg.0
        ));
        refs.push(name);
    }
    (refs, types)
}

fn widen_op(ctx: &LoweringContext, arg: ValueId) -> &'static str {
    if ctx.i1_values.contains(&arg) {
        return "arith.extui";
    }
    match ctx.values.get(&arg) {
        Some(ValueKind::ScalarU32 | ValueKind::ScalarBool) => "arith.extui",
        _ => "arith.extsi",
    }
}

/// The callee text, the function-type prefix and the argument operands of an
/// extern call. A base or library binding calls `@name`. A `bind(.address)`
/// call takes the code address from `args[0]`: if it is 0 the process exits
/// with status 78 and writes nothing (a call through a null pointer is
/// undefined in LLVM, so MIND gives it one result), else it calls through it.
pub(super) fn callee<'a>(
    ctx: &mut LoweringContext,
    dst: ValueId,
    name: &str,
    binding: &crate::ast::ExternBinding,
    args: &'a [ValueId],
) -> (String, &'static str, &'a [ValueId]) {
    let (crate::ast::ExternBinding::Address, Some((addr, rest))) = (binding, args.split_first())
    else {
        return (format!("@{name}"), "", args);
    };
    let (zero, null, fp) = (
        format!("%fpz_{}", dst.0),
        format!("%fpn_{}", dst.0),
        format!("%fp_{}", dst.0),
    );
    ctx.emit_line(&format!("    {zero} = arith.constant 0 : i64"));
    ctx.emit_line(&format!(
        "    {null} = arith.cmpi eq, %{}, {zero} : i64",
        addr.0
    ));
    ctx.emit_null_call_guard(&null);
    ctx.emit_line(&format!(
        "    {fp} = llvm.inttoptr %{} : i64 to !llvm.ptr",
        addr.0
    ));
    (fp, "!llvm.ptr, ", rest)
}

/// Extend each narrow argument of a call through an address to 32 bits by its
/// declared row: `i8`/`i16` sign-extend, `u8`/`u16`/`bool` zero-extend. A
/// direct call gets the same effect from `llvm.signext`/`llvm.zeroext` on its
/// declaration (`decl_param`); an indirect call has no declaration.
pub(super) fn widen_for_indirect(
    ctx: &mut LoweringContext,
    dst: ValueId,
    refs: &mut [String],
    types: &mut [String],
    declared: &[String],
) {
    for (i, token) in declared.iter().enumerate().take(refs.len()) {
        let op = match token.as_str() {
            "i8" | "i16" => "arith.extsi",
            "u8" | "u16" | "bool" => "arith.extui",
            _ => continue,
        };
        let wide = format!("%indarg_{}_{i}", dst.0);
        ctx.emit_line(&format!(
            "    {wide} = {op} {} : {} to i32",
            refs[i], types[i]
        ));
        refs[i] = wide;
        types[i] = "i32".to_string();
    }
}
