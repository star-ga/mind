// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Narrow `extern "C"` arguments.
//!
//! Each narrow parameter is declared at its C width. Signed names use
//! `llvm.signext`. Unsigned names and `bool` use `llvm.zeroext`. `bool`
//! is `i1`. The call passes the signless width.

use super::{LoweringContext, ValueKind};
use crate::ir::ValueId;

pub(super) fn call_ty(token: &str) -> &str {
    match token {
        "i8" | "u8" => "i8",
        "i16" | "u16" => "i16",
        "i32" | "u32" => "i32",
        "bool" => "i1",
        other => other,
    }
}

pub(super) fn decl_param(token: &str) -> String {
    let ty = call_ty(token);
    let attr = match token {
        "i8" | "i16" | "i32" => Some("llvm.signext"),
        "u8" | "u16" | "u32" | "bool" => Some("llvm.zeroext"),
        _ => None,
    };
    match attr {
        Some(attr) => format!("{ty} {{{attr}}}"),
        None => ty.to_string(),
    }
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
