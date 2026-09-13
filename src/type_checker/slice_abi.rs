// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Provenance and capability checks for the Option-C slice ABI.
//! A dynamic `array<T>` is an opaque i64 handle to the public `std.vec`
//! `[addr, len, cap]` record. Slice parameters use that same physical handle,
//! but only a value proven to have that layout may cross the call boundary.
//! This pass runs during type checking, before AST-to-IR lowering, so an invalid
//! scalar/map/opaque handle produces a structured diagnostic rather than a
//! lowering panic or a silent zero.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::ast::{Node, Span, TypeAnn};
use crate::diagnostics::Diagnostic;

use super::{
    ClassCtx, confident_scalar_class, diag_from_span, intra_lookup_fn, scalar_class_of_ann,
};

mod borrow_flow;
#[cfg(feature = "std-surface")]
mod deref_check;
mod owner_assignment;
mod provenance;
pub(super) use borrow_flow::type_contains_collection_owner;
use borrow_flow::{
    LoopControl, borrowed_kind, check_call, check_loop_flow, check_while, merge_flow,
    reject_borrowed_control, remove_pattern_bindings, stmt_guarantees_return, type_contains_slice,
    value_contains_borrow,
};
pub(super) use borrow_flow::{check_fn, check_struct_fields};
#[cfg(feature = "std-surface")]
pub(super) use deref_check::check_fn as deref_check_fn;
#[cfg(feature = "std-surface")]
pub(super) use deref_check::fn_param_sigs;
pub(super) use owner_assignment::{StructFieldTypes, struct_field_types};
use provenance::{compatible_source, expr_kind, expr_type, same_type};

pub(super) const SLICE_ARG_ABI_CODE: &str = "E2032";
pub(super) const SLICE_CAPABILITY_CODE: &str = "E2033";
pub(super) const COLLECTION_OWNER_ASSIGN_CODE: &str = "E2034";
pub(super) const NARROW_ELEMENT_ASSIGN_CODE: &str = "E2036";

#[derive(Clone, Debug, PartialEq, Eq)]
enum HandleKind {
    Array(TypeAnn),
    Slice { mutable: bool, element: TypeAnn },
}

type BindingId = (usize, usize);

impl HandleKind {
    fn element(&self) -> &TypeAnn {
        match self {
            Self::Array(element) | Self::Slice { element, .. } => element,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Env {
    handles: BTreeMap<String, HandleKind>,
    classes: ClassCtx,
    types: BTreeMap<String, TypeAnn>,
    declared: BTreeMap<String, TypeAnn>,
    bindings: BTreeMap<String, BindingId>,
    binding_handles: BTreeMap<BindingId, HandleKind>,
    binding_borrows: BTreeSet<BindingId>,
    may_borrow: BTreeSet<String>,
    struct_fields: Rc<StructFieldTypes>,
}

fn declared_kind(ty: &TypeAnn) -> Option<HandleKind> {
    match ty {
        TypeAnn::Generic { name, args } if name == "array" => {
            Some(HandleKind::Array(args.first()?.clone()))
        }
        TypeAnn::Slice { mutable, element } => Some(HandleKind::Slice {
            mutable: *mutable,
            element: (**element).clone(),
        }),
        _ => None,
    }
}

fn vec_element_uses_i64_abi(ty: &TypeAnn) -> bool {
    !type_contains_slice(ty)
        && !matches!(
            ty,
            TypeAnn::ScalarF32
                | TypeAnn::ScalarF64
                | TypeAnn::Tensor { .. }
                | TypeAnn::DiffTensor { .. }
                | TypeAnn::Array { .. }
        )
}

fn call_signature(callee: &str) -> Option<(Vec<TypeAnn>, Option<TypeAnn>)> {
    if let Some(sig) = intra_lookup_fn(callee) {
        return Some((sig.param_types, sig.ret_type));
    }
    #[cfg(feature = "cross-module-imports")]
    if let Some(sig) = super::cm_lookup_fn(callee) {
        return Some((sig.param_types, sig.ret_type));
    }
    None
}

fn report(
    errs: &mut Vec<Diagnostic>,
    src: &str,
    file: Option<&str>,
    message: String,
    span: Span,
    code: &'static str,
) {
    errs.push(diag_from_span(src, file, message, span, code));
}

fn check_expr(
    node: &Node,
    env: &Env,
    expected_return: Option<&TypeAnn>,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    match node {
        Node::Call { callee, args, .. } => {
            check_call(callee, args, env, src, file, errs);
            for arg in args {
                check_expr(arg, env, expected_return, src, file, errs);
            }
        }
        Node::MethodCall {
            receiver,
            method,
            args,
            span,
        } => {
            if borrowed_kind(receiver, env).is_none() && value_contains_borrow(receiver, env) {
                report(
                    errs,
                    src,
                    file,
                    format!(
                        "method `{method}` has a derived borrowed receiver with no preserved capability"
                    ),
                    *span,
                    SLICE_CAPABILITY_CODE,
                );
            }
            if let Some(HandleKind::Slice { mutable, .. }) = expr_kind(receiver, env) {
                let allowed = matches!(method.as_str(), "get" | "len" | "length")
                    || (mutable && method == "set");
                if !allowed {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "method `{method}` is not available on a {}slice; slices cannot grow, free, or expose owner capacity",
                            if mutable { "mutable " } else { "read-only " },
                        ),
                        *span,
                        SLICE_CAPABILITY_CODE,
                    );
                }
            }
            for arg in args {
                if value_contains_borrow(arg, env) {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "method `{method}` cannot store or consume a borrowed slice argument"
                        ),
                        arg.span(),
                        SLICE_CAPABILITY_CODE,
                    );
                }
            }
            check_expr(receiver, env, expected_return, src, file, errs);
            for arg in args {
                check_expr(arg, env, expected_return, src, file, errs);
            }
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            span,
        } => {
            let target = provenance::indexed_type(receiver, env);
            if let Some(target) = target.as_ref() {
                if provenance::narrow_integer_rejects_value(target, value, env) {
                    report(
                        errs,
                        src,
                        file,
                        "a narrow integer array element cannot store a proven non-integer or opaque-handle value; use an explicit numeric conversion"
                            .to_string(),
                        value.span(),
                        NARROW_ELEMENT_ASSIGN_CODE,
                    );
                }
            }
            owner_assignment::check(
                target,
                value,
                "an indexed collection-owner slot",
                env,
                src,
                file,
                errs,
            );
            if borrowed_kind(receiver, env).is_none() && value_contains_borrow(receiver, env) {
                report(
                    errs,
                    src,
                    file,
                    "index assignment has a derived borrowed receiver with no preserved capability"
                        .to_string(),
                    *span,
                    SLICE_CAPABILITY_CODE,
                );
            }
            if matches!(
                expr_kind(receiver, env),
                Some(HandleKind::Slice { mutable: false, .. })
            ) {
                report(
                    errs,
                    src,
                    file,
                    "cannot assign through a read-only slice; declare the parameter `&mut [T]`"
                        .to_string(),
                    *span,
                    SLICE_CAPABILITY_CODE,
                );
            }
            for operand in [index.as_ref(), value.as_ref()] {
                if value_contains_borrow(operand, env) {
                    report(
                        errs,
                        src,
                        file,
                        "a borrowed slice cannot be used as an index or stored as a slice element"
                            .to_string(),
                        operand.span(),
                        SLICE_CAPABILITY_CODE,
                    );
                }
            }
            check_expr(receiver, env, expected_return, src, file, errs);
            check_expr(index, env, expected_return, src, file, errs);
            check_expr(value, env, expected_return, src, file, errs);
        }
        Node::FieldAssign {
            receiver,
            field,
            value,
            span,
        } => {
            if value_contains_borrow(receiver, env) || value_contains_borrow(value, env) {
                report(
                    errs,
                    src,
                    file,
                    "borrowed slice handle used outside the supported read/alias/call capability surface"
                        .to_string(),
                    *span,
                    SLICE_CAPABILITY_CODE,
                );
            }
            owner_assignment::check(
                provenance::field_type(receiver, field, env),
                value,
                &format!("collection-owner field `{field}`"),
                env,
                src,
                file,
                errs,
            );
            check_expr(receiver, env, expected_return, src, file, errs);
            check_expr(value, env, expected_return, src, file, errs);
        }
        Node::IndexAccess {
            receiver, index, ..
        } => {
            if borrowed_kind(receiver, env).is_none() && value_contains_borrow(receiver, env) {
                report(
                    errs,
                    src,
                    file,
                    "derived borrowed slice receiver has no capability-preserving representation"
                        .to_string(),
                    receiver.span(),
                    SLICE_CAPABILITY_CODE,
                );
            }
            if value_contains_borrow(index, env) {
                report(
                    errs,
                    src,
                    file,
                    "borrowed slice handle cannot be used as an index".to_string(),
                    index.span(),
                    SLICE_CAPABILITY_CODE,
                );
            }
            check_expr(receiver, env, expected_return, src, file, errs);
            check_expr(index, env, expected_return, src, file, errs);
        }
        Node::FieldAccess {
            receiver,
            field,
            span,
        } if borrowed_kind(receiver, env).is_some() => {
            if !matches!(field.as_str(), "len" | "length") {
                report(
                    errs,
                    src,
                    file,
                    format!("slice field `{field}` is outside the supported read-only surface"),
                    *span,
                    SLICE_CAPABILITY_CODE,
                );
            }
            check_expr(receiver, env, expected_return, src, file, errs);
        }
        Node::Paren(inner, _) => check_expr(inner, env, expected_return, src, file, errs),
        Node::ArrayLit { .. }
        | Node::Tuple { .. }
        | Node::StructLit { .. }
        | Node::MapLit { .. }
        | Node::SetLit { .. }
            if value_contains_borrow(node, env) =>
        {
            report(
                errs,
                src,
                file,
                "borrowed slice handles cannot be stored inside aggregate values".to_string(),
                node.span(),
                SLICE_CAPABILITY_CODE,
            );
            super::nerve_walk::for_each_child(node, &mut |child| {
                check_expr(child, env, expected_return, src, file, errs)
            });
        }
        Node::Block { stmts, .. } => {
            let mut inner = env.clone();
            check_stmts(stmts, &mut inner, expected_return, src, file, errs);
        }
        Node::Return {
            value: Some(value),
            span,
        } => {
            if let Some(target) = expected_return.and_then(declared_kind) {
                if !compatible_source(&target, value, env) {
                    let borrowed_into_array = matches!(&target, HandleKind::Array(_))
                        && value_contains_borrow(value, env);
                    report(
                        errs,
                        src,
                        file,
                        if borrowed_into_array {
                            "an `array<T>` return cannot take ownership of a borrowed slice handle"
                                .to_string()
                        } else {
                            "slice/array return value is not a proven compatible Vec-layout handle"
                                .to_string()
                        },
                        *span,
                        if borrowed_into_array {
                            SLICE_CAPABILITY_CODE
                        } else {
                            SLICE_ARG_ABI_CODE
                        },
                    );
                }
            } else if value_contains_borrow(value, env) {
                report(
                    errs,
                    src,
                    file,
                    "a borrowed slice handle cannot escape through a non-slice function return"
                        .to_string(),
                    *span,
                    SLICE_CAPABILITY_CODE,
                );
            }
            check_expr(value, env, expected_return, src, file, errs);
        }
        Node::Return { value: None, span } if expected_return.and_then(declared_kind).is_some() => {
            report(
                errs,
                src,
                file,
                "slice/array return requires a proven compatible Vec-layout value".to_string(),
                *span,
                SLICE_ARG_ABI_CODE,
            )
        }
        _ => {
            let mut direct_borrow_use = false;
            super::nerve_walk::for_each_child(node, &mut |child| {
                direct_borrow_use |= value_contains_borrow(child, env);
            });
            if direct_borrow_use {
                report(
                    errs,
                    src,
                    file,
                    "borrowed slice handle used outside the supported read/alias/call capability surface"
                        .to_string(),
                    node.span(),
                    SLICE_CAPABILITY_CODE,
                );
            }
            super::nerve_walk::for_each_child(node, &mut |child| {
                check_expr(child, env, expected_return, src, file, errs)
            });
        }
    }
}

fn check_stmts(
    stmts: &[Node],
    env: &mut Env,
    expected_return: Option<&TypeAnn>,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    for stmt in stmts {
        match stmt {
            Node::Let {
                name,
                ann,
                value,
                span,
                ..
            } => {
                check_expr(value, env, expected_return, src, file, errs);
                let borrowed = value_contains_borrow(value, env);
                if let Some(ann) = ann {
                    env.declared.insert(name.clone(), ann.clone());
                } else {
                    env.declared.remove(name);
                }
                if ann.as_ref().is_some_and(|ty| {
                    !matches!(ty, TypeAnn::Slice { .. }) && type_contains_slice(ty)
                }) {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "binding `{name}` has an aggregate type containing a borrowed slice, which the current ABI cannot preserve"
                        ),
                        value.span(),
                        SLICE_CAPABILITY_CODE,
                    );
                }
                if borrowed && ann.as_ref().is_some_and(|ty| declared_kind(ty).is_none()) {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "binding `{name}` cannot erase a borrowed slice capability into a non-slice type"
                        ),
                        value.span(),
                        SLICE_CAPABILITY_CODE,
                    );
                }
                if borrowed && borrowed_kind(value, env).is_none() {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "binding `{name}` cannot preserve a borrowed slice hidden inside an expression"
                        ),
                        value.span(),
                        SLICE_CAPABILITY_CODE,
                    );
                }
                let declared = ann.as_ref().and_then(declared_kind);
                let kind = match declared {
                    Some(HandleKind::Slice { .. }) => {
                        report(
                            errs,
                            src,
                            file,
                            format!(
                                "slice binding `{name}` is not supported by the current Option-C lowering; keep the value as `array<T>` or pass it directly to a slice parameter"
                            ),
                            value.span(),
                            SLICE_ARG_ABI_CODE,
                        );
                        None
                    }
                    Some(target) if compatible_source(&target, value, env) => Some(target),
                    Some(HandleKind::Array(_)) => {
                        if value_contains_borrow(value, env) {
                            report(
                                errs,
                                src,
                                file,
                                format!(
                                    "array binding `{name}` cannot take ownership of a borrowed slice handle"
                                ),
                                value.span(),
                                SLICE_CAPABILITY_CODE,
                            );
                        }
                        None
                    }
                    None => expr_kind(value, env),
                };
                if let Some(kind) = kind.clone() {
                    env.handles.insert(name.clone(), kind);
                } else {
                    env.handles.remove(name);
                }
                env.bindings
                    .insert(name.clone(), (span.start(), span.end()));
                borrow_flow::update_current_flow(env, name, borrowed);
                if let Some(class) = ann
                    .as_ref()
                    .and_then(scalar_class_of_ann)
                    .or_else(|| confident_scalar_class(value, &env.classes))
                {
                    env.classes.classes.insert(name.clone(), class);
                } else {
                    env.classes.classes.remove(name);
                }
                let inferred = expr_type(value, env);
                let proven = match (ann, inferred) {
                    (Some(ann), Some(inferred)) if same_type(ann, &inferred) => Some(ann.clone()),
                    (Some(ann), _) if declared_kind(ann).is_some() && kind.is_some() => {
                        Some(ann.clone())
                    }
                    (Some(ann), _)
                        if provenance::is_collection_owner_type(ann)
                            && provenance::compatible_collection_owner_source(ann, value, env) =>
                    {
                        Some(ann.clone())
                    }
                    (None, inferred) => inferred,
                    _ => None,
                };
                if let Some(ty) = proven {
                    env.types.insert(name.clone(), ty);
                } else {
                    env.types.remove(name);
                }
            }
            Node::Assign {
                name, value, span, ..
            } => {
                check_expr(value, env, expected_return, src, file, errs);
                let borrowed = value_contains_borrow(value, env);
                let source = expr_kind(value, env);
                if borrowed
                    && env
                        .declared
                        .get(name)
                        .is_some_and(|ty| declared_kind(ty).is_none())
                {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "assignment to `{name}` cannot erase a borrowed slice capability into a non-slice type"
                        ),
                        *span,
                        SLICE_CAPABILITY_CODE,
                    );
                }
                if borrowed && !matches!(source, Some(HandleKind::Slice { .. })) {
                    report(
                        errs,
                        src,
                        file,
                        format!(
                            "assignment to `{name}` cannot preserve a borrowed slice hidden inside an expression"
                        ),
                        *span,
                        SLICE_CAPABILITY_CODE,
                    );
                }
                if let Some(target) = env.handles.get(name).cloned() {
                    if !compatible_source(&target, value, env) {
                        let borrowed_into_array =
                            matches!(&target, HandleKind::Array(_)) && borrowed;
                        report(
                            errs,
                            src,
                            file,
                            if borrowed_into_array {
                                format!(
                                    "array binding `{name}` cannot take ownership of a borrowed slice handle"
                                )
                            } else {
                                format!(
                                    "assignment to slice/array binding `{name}` is not a proven compatible Vec-layout handle"
                                )
                            },
                            *span,
                            if borrowed_into_array {
                                SLICE_CAPABILITY_CODE
                            } else {
                                SLICE_ARG_ABI_CODE
                            },
                        );
                    }
                    // Preserve the binding's declared/inferred capability. In
                    // particular, assigning an owned array handle to a slice
                    // variable must not turn that variable into an owner.
                    env.handles.insert(name.clone(), target);
                } else if let Some(kind) = source {
                    env.handles.insert(name.clone(), kind);
                }
                borrow_flow::update_current_flow(env, name, borrowed);
                if let Some(class) = confident_scalar_class(value, &env.classes) {
                    env.classes.classes.insert(name.clone(), class);
                } else {
                    env.classes.classes.remove(name);
                }
                let inferred = expr_type(value, env);
                let target_ty = env.types.get(name).cloned();
                match (target_ty.as_ref(), inferred) {
                    (Some(target), Some(source)) if same_type(target, &source) => {}
                    (None, Some(source)) => {
                        env.types.insert(name.clone(), source);
                    }
                    _ => {
                        env.types.remove(name);
                    }
                }
            }
            Node::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                reject_borrowed_control(cond, env, "a condition", src, file, errs);
                check_expr(cond, env, expected_return, src, file, errs);
                let mut left = env.clone();
                check_stmts(then_branch, &mut left, expected_return, src, file, errs);
                let mut right = env.clone();
                if let Some(branch) = else_branch {
                    check_stmts(branch, &mut right, expected_return, src, file, errs);
                }
                let mut flows = Vec::new();
                if !then_branch.iter().any(stmt_guarantees_return) {
                    flows.push(left.clone());
                }
                if !else_branch
                    .as_ref()
                    .is_some_and(|branch| branch.iter().any(stmt_guarantees_return))
                {
                    flows.push(right.clone());
                }
                if !flows.is_empty() {
                    merge_flow(env, &flows);
                }
                env.classes.classes.retain(|name, class| {
                    flows
                        .iter()
                        .all(|branch| branch.classes.classes.get(name) == Some(class))
                });
                env.types.retain(|name, ty| {
                    flows
                        .iter()
                        .all(|branch| branch.types.get(name) == Some(ty))
                });
            }
            Node::Block { stmts, .. } => {
                let mut inner = env.clone();
                check_stmts(stmts, &mut inner, expected_return, src, file, errs);
                merge_flow(env, &[inner.clone()]);
                env.handles
                    .retain(|name, kind| inner.handles.get(name) == Some(kind));
                env.classes
                    .classes
                    .retain(|name, class| inner.classes.classes.get(name) == Some(class));
                env.types
                    .retain(|name, ty| inner.types.get(name) == Some(ty));
            }
            Node::For {
                var,
                start,
                end,
                body,
                span,
                ..
            } => {
                reject_borrowed_control(start, env, "a loop bound", src, file, errs);
                reject_borrowed_control(end, env, "a loop bound", src, file, errs);
                check_expr(start, env, expected_return, src, file, errs);
                check_expr(end, env, expected_return, src, file, errs);
                check_loop_flow(
                    body,
                    LoopControl::Binding(var, *span),
                    env,
                    expected_return,
                    src,
                    file,
                    errs,
                );
            }
            Node::ForEach {
                var,
                collection,
                body,
                span,
                ..
            } => {
                reject_borrowed_control(collection, env, "a for-each source", src, file, errs);
                check_expr(collection, env, expected_return, src, file, errs);
                check_loop_flow(
                    body,
                    LoopControl::Binding(var, *span),
                    env,
                    expected_return,
                    src,
                    file,
                    errs,
                );
            }
            #[cfg(feature = "std-surface")]
            Node::While { cond, body, .. } => {
                check_while(cond, body, env, expected_return, src, file, errs);
            }
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => {
                let mut inner = env.clone();
                check_stmts(body, &mut inner, expected_return, src, file, errs);
                merge_flow(env, &[inner.clone()]);
                env.handles
                    .retain(|name, kind| inner.handles.get(name) == Some(kind));
                env.classes
                    .classes
                    .retain(|name, class| inner.classes.classes.get(name) == Some(class));
                env.types
                    .retain(|name, ty| inner.types.get(name) == Some(ty));
            }
            Node::Match {
                scrutinee, arms, ..
            } => {
                reject_borrowed_control(scrutinee, env, "a match scrutinee", src, file, errs);
                check_expr(scrutinee, env, expected_return, src, file, errs);
                let entry = env.clone();
                let mut branches = vec![entry.clone()];
                for arm in arms {
                    let mut arm_env = entry.clone();
                    remove_pattern_bindings(&arm.pattern, &mut arm_env, arm.body.span());
                    if let Some(guard) = &arm.guard {
                        reject_borrowed_control(guard, &arm_env, "a match guard", src, file, errs);
                        check_expr(guard, &arm_env, expected_return, src, file, errs);
                    }
                    check_stmts(
                        std::slice::from_ref(&arm.body),
                        &mut arm_env,
                        expected_return,
                        src,
                        file,
                        errs,
                    );
                    if !stmt_guarantees_return(&arm.body) {
                        branches.push(arm_env);
                    }
                }
                merge_flow(env, &branches);
            }
            _ => check_expr(stmt, env, expected_return, src, file, errs),
        }
        if stmt_guarantees_return(stmt) {
            break;
        }
    }
}
