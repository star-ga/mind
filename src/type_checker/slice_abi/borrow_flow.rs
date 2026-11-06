// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Borrowed-handle flow boundaries for the Option-C slice ABI.

use crate::ast::{FnDefData, Node, Pattern, Span, TypeAnn};
use crate::diagnostics::Diagnostic;

use super::aliases::type_needs_collection_flow;
use super::{
    Env, HandleKind, SLICE_ARG_ABI_CODE, SLICE_CAPABILITY_CODE, call_signature, canonical,
    check_stmts, compatible_source, declared_kind, expr_kind, report, scalar_class_of_ann,
    vec_element_uses_i64_abi,
};

pub(super) fn type_contains_slice(ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::Slice { .. } => true,
        TypeAnn::Array { element, .. }
        | TypeAnn::Ref {
            target: element, ..
        }
        | TypeAnn::SparseTensor { element, .. }
        | TypeAnn::RawPtr {
            pointee: element, ..
        } => type_contains_slice(element),
        TypeAnn::Generic { args, .. } => args.iter().any(type_contains_slice),
        TypeAnn::Tuple { elements } => elements.iter().any(type_contains_slice),
        TypeAnn::FnPtr { params, ret } => {
            params.iter().any(type_contains_slice)
                || ret.as_deref().is_some_and(type_contains_slice)
        }
        _ => false,
    }
}

pub(super) fn borrowed_kind(node: &Node, env: &Env) -> Option<HandleKind> {
    expr_kind(node, env).filter(|kind| matches!(&kind, HandleKind::Slice { .. }))
}

/// Whether the value being formed retains a borrowed handle. Calls and reads
/// are boundaries: their result is borrowed only when their declared result is
/// a slice, while a checked slice argument may safely produce an ordinary i64.
pub(super) fn value_contains_borrow(node: &Node, env: &Env) -> bool {
    if matches!(node, Node::Lit(crate::ast::Literal::Ident(name), _) if env.may_borrow.contains(name))
    {
        return true;
    }
    if borrowed_kind(node, env).is_some() {
        return true;
    }
    match node {
        Node::Call { .. } | Node::MethodCall { .. } | Node::IndexAccess { .. } => false,
        Node::FieldAccess { field, .. } if matches!(field.as_str(), "len" | "length") => false,
        Node::Paren(inner, _) => value_contains_borrow(inner, env),
        _ => {
            let mut found = false;
            super::super::nerve_walk::for_each_child(node, &mut |child| {
                found |= value_contains_borrow(child, env)
            });
            found
        }
    }
}

/// Snapshot the lexical name's current facts under its stable binding ID.
/// Old IDs intentionally remain: an outer assignment followed by a local
/// same-name declaration must still reach the enclosing scope's join.
pub(super) fn sync_current_binding(env: &mut Env, name: &str) {
    let Some(binding) = env.bindings.get(name).copied() else {
        return;
    };
    if let Some(kind) = env.handles.get(name).cloned() {
        env.binding_handles.insert(binding, kind);
    } else {
        env.binding_handles.remove(&binding);
    }
    if env.may_borrow.contains(name) {
        env.binding_borrows.insert(binding);
    } else {
        env.binding_borrows.remove(&binding);
    }
}

pub(super) fn update_current_flow(env: &mut Env, name: &str, borrowed: bool) {
    if borrowed {
        env.may_borrow.insert(name.to_string());
    } else {
        env.may_borrow.remove(name);
    }
    sync_current_binding(env, name);
}

/// Join capability flow for alternative control-flow successors. Exact handle
/// facts survive only when every successor agrees; possible borrowing is a
/// union. Each successor is queried by the entry binding's stable ID, so a
/// later same-name declaration cannot hide an earlier write to the outer value.
pub(super) fn merge_flow(env: &mut Env, branches: &[Env]) {
    let entry = env.clone();
    let mut handles = std::collections::BTreeMap::new();
    let mut may_borrow = std::collections::BTreeSet::new();
    let mut binding_handles = entry.binding_handles.clone();
    let mut binding_borrows = entry.binding_borrows.clone();
    for (name, key) in &entry.bindings {
        let state = |branch: &Env| {
            (
                branch.binding_handles.get(key).cloned(),
                branch.binding_borrows.contains(key),
            )
        };
        if let Some((first_kind, first_may)) = branches.first().map(state) {
            binding_handles.remove(key);
            if branches
                .iter()
                .skip(1)
                .all(|branch| state(branch).0.as_ref() == first_kind.as_ref())
            {
                if let Some(kind) = first_kind {
                    binding_handles.insert(*key, kind.clone());
                    handles.insert(name.clone(), kind);
                }
            }
            binding_borrows.remove(key);
            if first_may || branches.iter().skip(1).any(|branch| state(branch).1) {
                binding_borrows.insert(*key);
                may_borrow.insert(name.clone());
            }
        }
    }
    env.handles = handles;
    env.may_borrow = may_borrow;
    env.binding_handles = binding_handles;
    env.binding_borrows = binding_borrows;
}

fn shadow_binding(env: &mut Env, name: &str, span: Span) {
    env.bindings
        .insert(name.to_string(), (span.start(), span.end()));
    env.handles.remove(name);
    env.may_borrow.remove(name);
    env.types.remove(name);
    env.declared.remove(name);
    env.classes.classes.remove(name);
    sync_current_binding(env, name);
}

#[derive(Default)]
struct LoopPaths {
    fallthrough: Vec<Env>,
    breaks: Vec<Env>,
    continues: Vec<Env>,
}

fn restore_scope(entry: &Env, inner: Env) -> Env {
    let mut outer = entry.clone();
    merge_flow(&mut outer, std::slice::from_ref(&inner));
    outer
        .classes
        .classes
        .retain(|name, class| inner.classes.classes.get(name) == Some(class));
    outer
        .types
        .retain(|name, ty| inner.types.get(name) == Some(ty));
    outer
}

fn restore_paths(entry: &Env, paths: LoopPaths) -> LoopPaths {
    LoopPaths {
        fallthrough: paths
            .fallthrough
            .into_iter()
            .map(|state| restore_scope(entry, state))
            .collect(),
        breaks: paths
            .breaks
            .into_iter()
            .map(|state| restore_scope(entry, state))
            .collect(),
        continues: paths
            .continues
            .into_iter()
            .map(|state| restore_scope(entry, state))
            .collect(),
    }
}

fn append_paths(target: &mut LoopPaths, mut source: LoopPaths) {
    target.fallthrough.append(&mut source.fallthrough);
    target.breaks.append(&mut source.breaks);
    target.continues.append(&mut source.continues);
}

/// Collapse alternatives that reach the same control-flow target. Borrowing
/// is a may-property, while exact handle, scalar-class, and inferred-type facts
/// survive only when every predecessor agrees. Keeping one joined state per
/// target prevents sequential branches inside a loop from enumerating 2^N
/// paths without mixing fallthrough, break, and continue transfers.
fn join_target_states(states: Vec<Env>) -> Vec<Env> {
    let Some(mut joined) = states.first().cloned() else {
        return Vec::new();
    };
    if states.len() == 1 {
        return states;
    }
    merge_flow(&mut joined, &states);
    joined.classes.classes.retain(|name, class| {
        states
            .iter()
            .all(|state| state.classes.classes.get(name) == Some(class))
    });
    joined
        .types
        .retain(|name, ty| states.iter().all(|state| state.types.get(name) == Some(ty)));
    vec![joined]
}

/// Transfer one loop body's states while keeping `break` exits separate from
/// `continue` backedges. Statements after either transfer are unreachable and
/// therefore cannot clear a borrowed fact established before the transfer.
fn analyze_loop_body(
    stmts: &[Node],
    initial: Env,
    expected_return: Option<&TypeAnn>,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) -> LoopPaths {
    let mut active = vec![initial];
    let mut result = LoopPaths::default();
    for stmt in stmts {
        let mut next = Vec::new();
        for mut state in active {
            match stmt {
                Node::Break { .. } => result.breaks.push(state),
                Node::Continue { .. } => result.continues.push(state),
                Node::Return { .. } => {
                    check_stmts(
                        std::slice::from_ref(stmt),
                        &mut state,
                        expected_return,
                        src,
                        file,
                        errs,
                    );
                }
                Node::If {
                    cond,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    reject_borrowed_control(cond, &state, "a condition", src, file, errs);
                    super::check_expr(cond, &state, expected_return, src, file, errs);
                    let left = restore_paths(
                        &state,
                        analyze_loop_body(
                            then_branch,
                            state.clone(),
                            expected_return,
                            src,
                            file,
                            errs,
                        ),
                    );
                    let right = if let Some(branch) = else_branch {
                        restore_paths(
                            &state,
                            analyze_loop_body(
                                branch,
                                state.clone(),
                                expected_return,
                                src,
                                file,
                                errs,
                            ),
                        )
                    } else {
                        LoopPaths {
                            fallthrough: vec![state],
                            ..LoopPaths::default()
                        }
                    };
                    let mut branches = LoopPaths::default();
                    append_paths(&mut branches, left);
                    append_paths(&mut branches, right);
                    next.append(&mut branches.fallthrough);
                    result.breaks.append(&mut branches.breaks);
                    result.continues.append(&mut branches.continues);
                }
                Node::Block { stmts, .. } | Node::Region { body: stmts, .. } => {
                    let mut paths = restore_paths(
                        &state,
                        analyze_loop_body(stmts, state.clone(), expected_return, src, file, errs),
                    );
                    next.append(&mut paths.fallthrough);
                    result.breaks.append(&mut paths.breaks);
                    result.continues.append(&mut paths.continues);
                }
                Node::Match {
                    scrutinee, arms, ..
                } => {
                    reject_borrowed_control(
                        scrutinee,
                        &state,
                        "a match scrutinee",
                        src,
                        file,
                        errs,
                    );
                    super::check_expr(scrutinee, &state, expected_return, src, file, errs);
                    let mut paths = LoopPaths::default();
                    let mut exhaustive = false;
                    for arm in arms {
                        let mut arm_state = state.clone();
                        remove_pattern_bindings(&arm.pattern, &mut arm_state, arm.body.span());
                        if let Some(guard) = &arm.guard {
                            reject_borrowed_control(
                                guard,
                                &arm_state,
                                "a match guard",
                                src,
                                file,
                                errs,
                            );
                            super::check_expr(guard, &arm_state, expected_return, src, file, errs);
                        }
                        append_paths(
                            &mut paths,
                            restore_paths(
                                &state,
                                analyze_loop_body(
                                    std::slice::from_ref(&arm.body),
                                    arm_state,
                                    expected_return,
                                    src,
                                    file,
                                    errs,
                                ),
                            ),
                        );
                        if arm.guard.is_none()
                            && matches!(arm.pattern, Pattern::Ident(_) | Pattern::Wildcard)
                        {
                            exhaustive = true;
                            break;
                        }
                    }
                    if !exhaustive {
                        paths.fallthrough.push(state);
                    }
                    next.append(&mut paths.fallthrough);
                    result.breaks.append(&mut paths.breaks);
                    result.continues.append(&mut paths.continues);
                }
                _ => {
                    check_stmts(
                        std::slice::from_ref(stmt),
                        &mut state,
                        expected_return,
                        src,
                        file,
                        errs,
                    );
                    next.push(state);
                }
            }
        }
        active = join_target_states(next);
        result.breaks = join_target_states(result.breaks);
        result.continues = join_target_states(result.continues);
        if active.is_empty() {
            break;
        }
    }
    result.fallthrough = active;
    result
}

/// A loop introduces a binding or re-evaluates a condition at its header.
#[derive(Clone, Copy)]
pub(super) enum LoopControl<'a> {
    Binding(&'a str, Span),
    Condition(&'a Node),
}

/// Analyze a loop at the least fixed point of its entry and backedge borrow
/// facts, then validate once against that stabilized header. This catches a
/// value borrowed late in iteration N and consumed early in iteration N+1.
pub(super) fn check_loop_flow(
    body: &[Node],
    control: LoopControl<'_>,
    env: &mut Env,
    expected_return: Option<&TypeAnn>,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    let entry = env.clone();
    let mut header = entry.clone();
    loop {
        let mut body_entry = header.clone();
        if let LoopControl::Binding(name, span) = control {
            shadow_binding(&mut body_entry, name, span);
        }
        let mut ignored = Vec::new();
        let paths = analyze_loop_body(body, body_entry, expected_return, src, file, &mut ignored);
        let backedges: Vec<_> = paths
            .fallthrough
            .into_iter()
            .chain(paths.continues)
            .collect();
        let mut next = entry.clone();
        let mut header_inputs = vec![entry.clone()];
        header_inputs.extend(backedges);
        merge_flow(&mut next, &header_inputs);
        if next.handles == header.handles && next.may_borrow == header.may_borrow {
            header = next;
            break;
        }
        header = next;
    }
    if let LoopControl::Condition(condition) = control {
        reject_borrowed_control(condition, &header, "a while condition", src, file, errs);
        super::check_expr(condition, &header, expected_return, src, file, errs);
    }
    let mut body_entry = header;
    if let LoopControl::Binding(name, span) = control {
        shadow_binding(&mut body_entry, name, span);
    }
    let paths = analyze_loop_body(body, body_entry, expected_return, src, file, errs);
    let mut exits = vec![entry];
    exits.extend(paths.fallthrough);
    exits.extend(paths.continues);
    exits.extend(paths.breaks);
    merge_flow(env, &exits);
}

pub(super) fn check_while(
    condition: &Node,
    body: &[Node],
    env: &mut Env,
    expected_return: Option<&TypeAnn>,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    check_loop_flow(
        body,
        LoopControl::Condition(condition),
        env,
        expected_return,
        src,
        file,
        errs,
    );
}

pub(super) fn check_call(
    callee: &str,
    args: &[Node],
    env: &Env,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    let Some((params, _)) = call_signature(callee) else {
        for arg in args {
            if value_contains_borrow(arg, env) {
                report(
                    errs,
                    src,
                    file,
                    format!(
                        "borrowed slice value cannot be passed to unresolved function `{callee}`"
                    ),
                    arg.span(),
                    SLICE_CAPABILITY_CODE,
                );
            }
        }
        return;
    };
    for (index, arg) in args.iter().enumerate() {
        let Some(TypeAnn::Slice { mutable, element }) = params.get(index) else {
            if value_contains_borrow(arg, env) {
                report(
                    errs,
                    src,
                    file,
                    format!(
                        "function `{callee}` parameter {index} is not a slice and cannot receive a borrowed slice handle"
                    ),
                    arg.span(),
                    SLICE_CAPABILITY_CODE,
                );
            }
            continue;
        };
        let target_element = element.as_ref();
        let target = HandleKind::Slice {
            mutable: *mutable,
            element: target_element.clone(),
        };
        if !vec_element_uses_i64_abi(target_element) || !compatible_source(&target, arg, env) {
            report(
                errs,
                src,
                file,
                format!(
                    "function `{callee}` parameter {index} is declared as a {}slice, but the argument is not a proven compatible `array<T>`/slice Vec-layout handle; pass an `array<T>` with the same element type{}",
                    if *mutable { "mutable " } else { "" },
                    if *mutable { " or a mutable slice" } else { "" },
                ),
                arg.span(),
                SLICE_ARG_ABI_CODE,
            );
        }
    }
}

pub(super) fn reject_borrowed_tail(
    stmts: &[Node],
    env: &Env,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    match stmts.last() {
        Some(Node::Return { .. }) | None => {}
        Some(Node::Block { stmts, .. }) => reject_borrowed_tail(stmts, env, src, file, errs),
        Some(Node::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        }) => {
            reject_borrowed_tail(then_branch, env, src, file, errs);
            reject_borrowed_tail(else_branch, env, src, file, errs);
        }
        Some(value) if value_contains_borrow(value, env) => report(
            errs,
            src,
            file,
            "a borrowed slice handle cannot escape as an implicit non-slice function result"
                .to_string(),
            value.span(),
            SLICE_CAPABILITY_CODE,
        ),
        Some(_) => {}
    }
}

pub(crate) fn check_struct_fields(
    name: &str,
    fields: &[crate::ast::Field],
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    for field in fields {
        if type_contains_slice(&field.ty) || super::slice_through_alias(&field.ty) {
            report(
                errs,
                src,
                file,
                format!(
                    "struct `{name}` field `{}` stores a borrowed slice, which the current Option-C lowering cannot preserve with its read-only/mutable capability",
                    field.name
                ),
                field.span,
                SLICE_CAPABILITY_CODE,
            );
        }
    }
}

/// Whether executing `node` necessarily leaves the function through an
/// explicit return. This is deliberately conservative for loops: their bodies
/// may execute zero times, so a return inside one does not prove the function's
/// declared handle result.
pub(super) fn stmt_guarantees_return(node: &Node) -> bool {
    match node {
        Node::Return { .. } => true,
        Node::Block { stmts, .. } => stmts.iter().any(stmt_guarantees_return),
        Node::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => {
            then_branch.iter().any(stmt_guarantees_return)
                && else_branch.iter().any(stmt_guarantees_return)
        }
        _ => false,
    }
}

/// Prove that every fallthrough path produces the declared Vec-layout result.
/// Explicit returns are checked individually by `check_expr`; this closes the
/// separate hole where a declaration such as `-> &[i64]` authorized callers
/// even though its body fell through with a scalar or no value.
fn result_paths_proven(stmts: &[Node], target: &HandleKind, env: &Env) -> bool {
    if stmts.iter().any(stmt_guarantees_return) {
        return true;
    }
    match stmts.last() {
        Some(Node::Block { stmts, .. }) => result_paths_proven(stmts, target, env),
        Some(Node::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        }) => {
            result_paths_proven(then_branch, target, env)
                && result_paths_proven(else_branch, target, env)
        }
        Some(value) => compatible_source(target, value, env),
        None => false,
    }
}

pub(crate) fn type_contains_collection_owner(ty: &TypeAnn) -> bool {
    super::provenance::is_collection_owner_type(ty)
        || match ty {
            TypeAnn::Array { element, .. }
            | TypeAnn::Slice { element, .. }
            | TypeAnn::Ref {
                target: element, ..
            } => type_contains_collection_owner(element),
            TypeAnn::Tuple { elements } | TypeAnn::Generic { args: elements, .. } => {
                elements.iter().any(type_contains_collection_owner)
            }
            _ => false,
        }
}

/// Skip scalar-only bodies unless a collection boundary or indexed assignment
/// needs this pass, keeping its fixed-point analysis off unrelated functions.
fn function_needs_collection_flow(fd: &FnDefData, has_collection_owner_field: bool) -> bool {
    if fd
        .params
        .iter()
        .any(|param| type_needs_collection_flow(&param.ty))
        || fd.ret_type.as_ref().is_some_and(type_needs_collection_flow)
    {
        return true;
    }
    fd.body.iter().any(|stmt| {
        super::super::nerve_walk::any(stmt, &|node| match node {
            Node::Let { ann: Some(ty), .. } => type_needs_collection_flow(ty),
            Node::Call { callee, .. } => call_signature(callee).is_some_and(|(params, ret)| {
                params.iter().any(type_needs_collection_flow)
                    || ret.as_ref().is_some_and(type_needs_collection_flow)
            }),
            Node::IndexAssign { .. } => true,
            Node::FieldAssign { .. } => has_collection_owner_field,
            _ => false,
        })
    })
}

pub(crate) fn check_fn(
    fd: &FnDefData,
    struct_fields: &std::rc::Rc<super::StructFieldTypes>,
    has_collection_owner_field: bool,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    if !function_needs_collection_flow(fd, has_collection_owner_field) {
        return;
    }
    let mut env = Env {
        struct_fields: std::rc::Rc::clone(struct_fields),
        ..Env::default()
    };
    super::aliases::refuse_slice_aliases_in_signature(fd, src, file, errs);
    for param in &fd.params {
        let param_ty = canonical(&param.ty);
        if !matches!(param_ty, TypeAnn::Slice { .. }) && type_contains_slice(&param_ty) {
            report(
                errs,
                src,
                file,
                format!(
                    "parameter `{}` stores a borrowed slice inside an aggregate type, which the current ABI cannot preserve",
                    param.name
                ),
                param.span,
                SLICE_CAPABILITY_CODE,
            );
        }
        if let Some(kind) = declared_kind(&param_ty) {
            if matches!(&kind, HandleKind::Slice { .. }) {
                env.may_borrow.insert(param.name.clone());
            }
            env.handles.insert(param.name.clone(), kind);
        }
        if let Some(class) = scalar_class_of_ann(&param_ty) {
            env.classes.classes.insert(param.name.clone(), class);
        }
        env.types.insert(param.name.clone(), param_ty.clone());
        env.declared.insert(param.name.clone(), param_ty);
        env.bindings
            .insert(param.name.clone(), (param.span.start(), param.span.end()));
        sync_current_binding(&mut env, &param.name);
    }
    let ret_type = fd.ret_type.as_ref().map(canonical);
    if ret_type
        .as_ref()
        .is_some_and(|ty| !matches!(ty, TypeAnn::Slice { .. }) && type_contains_slice(ty))
    {
        let span = fd
            .body
            .last()
            .map(Node::span)
            .or_else(|| fd.params.first().map(|p| p.span))
            .unwrap_or_else(|| Span::new(0, 0));
        report(
            errs,
            src,
            file,
            "function return type stores a borrowed slice inside an aggregate, which the current ABI cannot preserve"
                .to_string(),
            span,
            SLICE_CAPABILITY_CODE,
        );
    }
    let expected_return = ret_type.as_ref();
    check_stmts(&fd.body, &mut env, expected_return, src, file, errs);
    if let Some(target) = expected_return.and_then(declared_kind).as_ref() {
        if !result_paths_proven(&fd.body, target, &env) {
            let span = fd
                .body
                .last()
                .map(Node::span)
                .or_else(|| fd.params.first().map(|p| p.span))
                .unwrap_or_else(|| Span::new(0, 0));
            report(
                errs,
                src,
                file,
                "slice/array function result is not proven on every path to be a compatible Vec-layout handle"
                    .to_string(),
                span,
                SLICE_ARG_ABI_CODE,
            );
        }
    }
    if expected_return.and_then(declared_kind).is_none() {
        reject_borrowed_tail(&fd.body, &env, src, file, errs);
    }
}

pub(super) fn reject_borrowed_control(
    node: &Node,
    env: &Env,
    context: &str,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Diagnostic>,
) {
    if value_contains_borrow(node, env) {
        report(
            errs,
            src,
            file,
            format!("borrowed slice handle cannot be used as {context}"),
            node.span(),
            SLICE_CAPABILITY_CODE,
        );
    }
}

pub(super) fn remove_pattern_bindings(pattern: &Pattern, env: &mut Env, span: Span) {
    match pattern {
        Pattern::Ident(name) => {
            env.bindings
                .insert(name.clone(), (span.start(), span.end()));
            env.handles.remove(name);
            env.may_borrow.remove(name);
            env.types.remove(name);
            env.declared.remove(name);
            env.classes.classes.remove(name);
            sync_current_binding(env, name);
        }
        Pattern::EnumVariant { args, .. } | Pattern::Tuple(args) => {
            for pattern in args {
                remove_pattern_bindings(pattern, env, span);
            }
        }
        Pattern::EnumStruct { fields, .. } => {
            for (_, pattern) in fields {
                remove_pattern_bindings(pattern, env, span);
            }
        }
        Pattern::Literal(_) | Pattern::Wildcard => {}
    }
}
