// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

#[cfg(feature = "std-surface")]
use crate::eval::materialization::MaterializationRefusal;

pub(super) fn first_deref_span(node: &crate::ast::Node, admitted: bool) -> Option<(usize, usize)> {
    use crate::ast::Node as N;
    match node {
        // Deref-assign: when `admitted` is false (the direct lowering API, which
        // has NOT run D3 semantic admission) EVERY `*p` / `*p = v` is refused —
        // fail closed. When `admitted` is true (the type-checked pipeline), a
        // BARE `*p` / `*p = v` (operand/target a plain identifier) flows to the
        // D4 load/store arms (deref_check already validated owner/capability/
        // region/call-context); only a NON-bare deref (`*x.y`, `*(expr)`, `**p`)
        // stays structurally unsupported.
        N::Deref { operand, span } => {
            if !admitted || !matches!(operand.as_ref(), N::Lit(crate::ast::Literal::Ident(_), _)) {
                return Some((span.start(), span.end()));
            }
        }
        N::DerefAssign {
            target,
            value,
            span,
        } => {
            if !admitted || !matches!(target.as_ref(), N::Lit(crate::ast::Literal::Ident(_), _)) {
                return Some((span.start(), span.end()));
            }
            // Admitted bare target: still scan the assigned value for a nested
            // unsupported deref.
            if let Some(hit) = first_deref_span(value, admitted) {
                return Some(hit);
            }
        }
        N::FnDef(fd, _) => {
            for s in &fd.body {
                if let Some(hit) = first_deref_span(s, admitted) {
                    return Some(hit);
                }
            }
        }
        N::Closure(cd, _) => {
            for s in &cd.body {
                if let Some(hit) = first_deref_span(s, admitted) {
                    return Some(hit);
                }
            }
        }
        _ => {
            for child in crate::eval::closures::node_children_ref(node) {
                if let Some(hit) = first_deref_span(child, admitted) {
                    return Some(hit);
                }
            }
        }
    }
    None
}

#[cfg(feature = "std-surface")]
pub(super) fn lower_mut_field_ref(
    inner: &crate::ast::Node,
    ir: &mut super::IRModule,
    env: &super::HashMap<String, super::ValueId>,
    struct_env: &super::HashMap<String, String>,
    receiver_types: &super::HashMap<crate::ast::Span, String>,
    context: &mut super::LoweringContext,
) -> super::ValueId {
    let crate::ast::Node::FieldAccess {
        receiver,
        field,
        span,
    } = inner
    else {
        unreachable!("guarded by the caller pattern")
    };
    match super::field_access::lower_field_cell_addr(
        receiver,
        field,
        span,
        ir,
        env,
        struct_env,
        receiver_types,
        context,
    ) {
        Some(cell) => cell,
        None => {
            context.refusal = Some(MaterializationRefusal::UnsupportedLoweringOperation {
                operation: "`&mut r.f`: the field owner/layout could not be resolved to a declared cell offset (deref-assign field-first subset)",
                start: span.start(),
                end: span.end(),
            });
            ir.fresh()
        }
    }
}

pub(super) fn lower_deref(
    operand: &crate::ast::Node,
    ir: &mut super::IRModule,
    env: &super::HashMap<String, super::ValueId>,
    struct_env: &super::HashMap<String, String>,
    receiver_types: &super::HashMap<crate::ast::Span, String>,
    context: &mut super::LoweringContext,
) -> super::ValueId {
    let p = super::lower_expr(operand, ir, env, struct_env, receiver_types, context);
    let dst = ir.fresh();
    ir.instrs.push(super::Instr::legacy_call(
        dst,
        "__mind_load_i64".to_string(),
        vec![p],
    ));
    dst
}

pub(super) fn lower_deref_assign(
    target: &crate::ast::Node,
    value: &crate::ast::Node,
    ir: &mut super::IRModule,
    env: &super::HashMap<String, super::ValueId>,
    struct_env: &super::HashMap<String, String>,
    receiver_types: &super::HashMap<crate::ast::Span, String>,
    context: &mut super::LoweringContext,
) -> super::ValueId {
    let p = super::lower_expr(target, ir, env, struct_env, receiver_types, context);
    let v = super::lower_expr(value, ir, env, struct_env, receiver_types, context);
    let dst = ir.fresh();
    ir.instrs.push(super::Instr::legacy_call(
        dst,
        "__mind_store_i64".to_string(),
        vec![p, v],
    ));
    dst
}
