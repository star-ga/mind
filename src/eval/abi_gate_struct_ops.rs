// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Operation-scoped runnable capability checks for fixed-array struct fields.

use crate::ast::{Module, Node, Span as AstSpan};
use crate::diagnostics::{Diagnostic, Span};

const PHASE: &str = "lower";
const HELP: &str = "the shipped backend lowers only the shipped scalar-cell ABI; this construct is not yet \
     lowerable to a runnable artifact (RUNS burndown). Run it with the `mind` interpreter, or keep \
     the compiled path to the supported scalar-cell subset.";

fn mk(src: &str, file: Option<&str>, span: AstSpan, code: &'static str, msg: String) -> Diagnostic {
    Diagnostic::error(PHASE, code, msg)
        .with_span(Span::from_offsets(src, span.start(), span.end(), file))
        .with_help(HELP)
}

/// Fixed arrays remain valid source-level types and are available through the
/// inspection/IR surfaces. The inline struct-field ABI materializes i64/f64 and
/// the existing i8/u8/i16/u16 cells, however, so a runnable artifact must
/// refuse an actual construction/access/update of any other fixed-array field
/// before MLIR/code emission. Declaration-only fields are deliberately ignored:
/// they have no lowering operation and retain the base125 runnable behavior.
#[cfg(feature = "std-surface")]
pub(super) fn check_fixed_struct_array_operations(
    module: &Module,
    ir: &crate::ir::IRModule,
    src: &str,
    file: Option<&str>,
    out: &mut Vec<Diagnostic>,
) {
    let receiver_types = crate::eval::struct_resolver::build_field_access_types(module);
    let mut seen = std::collections::HashSet::new();

    #[allow(clippy::too_many_arguments)]
    fn report(
        owner: &str,
        field: &str,
        span: AstSpan,
        ir: &crate::ir::IRModule,
        src: &str,
        file: Option<&str>,
        out: &mut Vec<Diagnostic>,
        seen: &mut std::collections::HashSet<AstSpan>,
    ) {
        if !crate::eval::lower::fixed_array_struct::fixed_array_field_unsupported(ir, owner, field)
            || !seen.insert(span)
        {
            return;
        }
        out.push(mk(
            src,
            file,
            span,
            "lower::fixed_struct_array_cell",
            format!(
                "fixed struct-array field `{owner}.{field}` is not lowerable to a runnable artifact: only i64/f64 and i8/u8/i16/u16 scalar cells are supported"
            ),
        ));
    }

    fn walk(
        node: &Node,
        ir: &crate::ir::IRModule,
        receiver_types: &crate::eval::struct_resolver::FieldAccessTypes,
        src: &str,
        file: Option<&str>,
        out: &mut Vec<Diagnostic>,
        seen: &mut std::collections::HashSet<AstSpan>,
    ) {
        use Node as N;
        match node {
            N::FnDef(fd, _) => fd
                .body
                .iter()
                .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen)),
            N::ImplBlock { methods, .. } => methods
                .iter()
                .for_each(|method| walk(method, ir, receiver_types, src, file, out, seen)),
            N::Closure(data, _) => data
                .body
                .iter()
                .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen)),
            N::StructLit { name, fields, span } => {
                for field in fields {
                    report(name, &field.name, *span, ir, src, file, out, seen);
                    walk(&field.value, ir, receiver_types, src, file, out, seen);
                }
            }
            N::FieldAccess {
                receiver,
                field,
                span,
            } => {
                if let Some(owner) = receiver_types.get(span) {
                    report(owner, field, *span, ir, src, file, out, seen);
                }
                walk(receiver, ir, receiver_types, src, file, out, seen);
            }
            N::FieldAssign {
                receiver,
                field,
                value,
                span,
            } => {
                if let Some(owner) = receiver_types.get(span) {
                    report(owner, field, *span, ir, src, file, out, seen);
                }
                walk(receiver, ir, receiver_types, src, file, out, seen);
                walk(value, ir, receiver_types, src, file, out, seen);
            }
            N::Let { value, .. } | N::Const { value, .. } => {
                walk(value, ir, receiver_types, src, file, out, seen)
            }
            N::LetTuple { value, .. } => walk(value, ir, receiver_types, src, file, out, seen),
            N::Assign { value, .. }
            | N::Return {
                value: Some(value), ..
            } => walk(value, ir, receiver_types, src, file, out, seen),
            N::Call { args, .. } => args
                .iter()
                .for_each(|a| walk(a, ir, receiver_types, src, file, out, seen)),
            N::CallGrad { loss, .. }
            | N::CallTensorSum { x: loss, .. }
            | N::CallTensorMean { x: loss, .. }
            | N::CallReshape { x: loss, .. }
            | N::CallExpandDims { x: loss, .. }
            | N::CallSqueeze { x: loss, .. }
            | N::CallTranspose { x: loss, .. }
            | N::CallIndex { x: loss, .. }
            | N::CallSlice { x: loss, .. }
            | N::CallSliceStride { x: loss, .. }
            | N::CallTensorRelu { x: loss, .. }
            | N::CallGather { x: loss, .. } => walk(loss, ir, receiver_types, src, file, out, seen),
            N::CallTensorConv2d { x, w, .. } => {
                walk(x, ir, receiver_types, src, file, out, seen);
                walk(w, ir, receiver_types, src, file, out, seen);
            }
            N::CallDot { a, b, .. }
            | N::CallMatMul { a, b, .. }
            | N::TensorMatmul { lhs: a, rhs: b, .. }
            | N::TensorElemwise { lhs: a, rhs: b, .. } => {
                walk(a, ir, receiver_types, src, file, out, seen);
                walk(b, ir, receiver_types, src, file, out, seen);
            }
            N::MethodCall { receiver, args, .. } => {
                walk(receiver, ir, receiver_types, src, file, out, seen);
                args.iter()
                    .for_each(|a| walk(a, ir, receiver_types, src, file, out, seen));
            }
            N::IndexAccess {
                receiver, index, ..
            } => {
                walk(receiver, ir, receiver_types, src, file, out, seen);
                walk(index, ir, receiver_types, src, file, out, seen);
            }
            N::IndexAssign {
                receiver,
                index,
                value,
                ..
            } => {
                walk(receiver, ir, receiver_types, src, file, out, seen);
                walk(index, ir, receiver_types, src, file, out, seen);
                walk(value, ir, receiver_types, src, file, out, seen);
            }
            N::Tuple { elements, .. }
            | N::ArrayLit { elements, .. }
            | N::SetLit { elements, .. } => {
                elements
                    .iter()
                    .for_each(|e| walk(e, ir, receiver_types, src, file, out, seen));
            }
            N::MapLit { entries, .. } => {
                for (key, value) in entries {
                    walk(key, ir, receiver_types, src, file, out, seen);
                    walk(value, ir, receiver_types, src, file, out, seen);
                }
            }
            N::Block { stmts, .. } => stmts
                .iter()
                .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen)),
            N::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                walk(cond, ir, receiver_types, src, file, out, seen);
                then_branch
                    .iter()
                    .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen));
                if let Some(branch) = else_branch {
                    branch
                        .iter()
                        .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen));
                }
            }
            N::For {
                start, end, body, ..
            } => {
                walk(start, ir, receiver_types, src, file, out, seen);
                walk(end, ir, receiver_types, src, file, out, seen);
                body.iter()
                    .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen));
            }
            N::ForEach {
                collection, body, ..
            } => {
                walk(collection, ir, receiver_types, src, file, out, seen);
                body.iter()
                    .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen));
            }
            N::Paren(inner, _)
            | N::Neg { operand: inner, .. }
            | N::Not { operand: inner, .. }
            | N::BitNot { operand: inner, .. }
            | N::Ref { inner, .. }
            | N::Try { inner, .. } => walk(inner, ir, receiver_types, src, file, out, seen),
            N::As { expr, .. } => walk(expr, ir, receiver_types, src, file, out, seen),
            N::SliceRange {
                receiver,
                start,
                end,
                ..
            } => {
                walk(receiver, ir, receiver_types, src, file, out, seen);
                walk(start, ir, receiver_types, src, file, out, seen);
                walk(end, ir, receiver_types, src, file, out, seen);
            }
            N::Print { args, .. } => args
                .iter()
                .for_each(|a| walk(a, ir, receiver_types, src, file, out, seen)),
            N::Assert { cond, .. } => walk(cond, ir, receiver_types, src, file, out, seen),
            N::Binary { left, right, .. } | N::Logical { left, right, .. } => {
                walk(left, ir, receiver_types, src, file, out, seen);
                walk(right, ir, receiver_types, src, file, out, seen);
            }
            #[cfg(feature = "std-surface")]
            N::Bitwise { left, right, .. } => {
                walk(left, ir, receiver_types, src, file, out, seen);
                walk(right, ir, receiver_types, src, file, out, seen);
            }
            N::Match {
                scrutinee, arms, ..
            } => {
                walk(scrutinee, ir, receiver_types, src, file, out, seen);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        walk(guard, ir, receiver_types, src, file, out, seen);
                    }
                    walk(&arm.body, ir, receiver_types, src, file, out, seen);
                }
            }
            #[cfg(feature = "std-surface")]
            N::While { cond, body, .. } => {
                walk(cond, ir, receiver_types, src, file, out, seen);
                body.iter()
                    .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen));
            }
            #[cfg(feature = "std-surface")]
            N::Region { body, .. } => body
                .iter()
                .for_each(|s| walk(s, ir, receiver_types, src, file, out, seen)),
            _ => {}
        }
    }

    for item in &module.items {
        walk(item, ir, &receiver_types, src, file, out, &mut seen);
    }
}
