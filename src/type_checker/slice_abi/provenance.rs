// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Proof of the Vec-layout owner or borrowed capability carried by a value.

use super::aliases::int_width_of_ann;
use super::{
    Env, HandleKind, call_signature, canonical, confident_scalar_class, declared_kind,
    scalar_class_of_ann, value_contains_borrow,
};
use crate::ast::{BinOp, Literal, Node, TypeAnn};

pub(super) fn expr_kind(node: &Node, env: &Env) -> Option<HandleKind> {
    match node {
        Node::ArrayLit { .. } => None,
        Node::Lit(Literal::Ident(name), _) => env.handles.get(name).cloned(),
        Node::Paren(inner, _) => expr_kind(inner, env),
        // `check_fn` validates a declared collection result before it authorizes callers.
        Node::Call { callee, .. } => call_signature(callee)
            .and_then(|(_, ret)| ret)
            .as_ref()
            .and_then(declared_kind),
        // Only Vec::push returns a replacement owner handle. In particular,
        // Vec::set returns a scalar status and must never acquire this proof.
        Node::MethodCall {
            receiver,
            method,
            args,
            ..
        } if method == "push" && args.len() == 1 => {
            expr_kind(receiver, env).filter(|kind| matches!(kind, HandleKind::Array(_)))
        }
        Node::FieldAccess {
            receiver, field, ..
        } => field_type(receiver, field, env)
            .as_ref()
            .and_then(declared_kind),
        Node::IndexAccess { receiver, .. } => {
            indexed_type(receiver, env).as_ref().and_then(declared_kind)
        }
        _ => None,
    }
}

/// Scalar class of an array-literal element. A call reports its canonical
/// declared result class, an imported callee's included; an element or field
/// read reports its declared type's class; a name no binding shadows reports
/// the class of the module constant it names.
fn element_class(element: &Node, env: &Env) -> Option<super::super::ScalarClass> {
    if let Some(class) = confident_scalar_class(element, &env.classes) {
        return Some(class);
    }
    match element {
        Node::Call { callee, .. } => call_signature(callee)?
            .1
            .as_ref()
            .and_then(scalar_class_of_ann),
        Node::Lit(Literal::Ident(name), _) if !env.bindings.contains_key(name) => {
            super::aliases::const_class(name)
        }
        Node::IndexAccess { .. } | Node::FieldAccess { .. } => expr_type(element, env)
            .as_ref()
            .and_then(scalar_class_of_ann),
        Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => element_class(inner, env),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => {
            let class = element_class(left, env)?;
            (element_class(right, env) == Some(class)).then_some(class)
        }
        _ => None,
    }
}

/// Proven integer bit-width of an array-literal element: the declared width
/// of a constant, a binding, an element or field read, a call result, or an
/// explicit `as` target, combined over an arithmetic expression. A plain
/// literal element declares no width of its own, so it stays `None` — it
/// already follows the slice element's width — and `None` never refuses.
fn element_int_width(element: &Node, env: &Env) -> Option<u32> {
    match element {
        Node::Lit(Literal::Int(_), _) => None,
        Node::Lit(Literal::Ident(name), _) if !env.bindings.contains_key(name) => {
            super::aliases::const_int_width(name)
        }
        Node::Lit(Literal::Ident(name), _) => env
            .declared
            .get(name)
            .or_else(|| env.types.get(name))
            .and_then(int_width_of_ann),
        Node::As { ty, .. } => int_width_of_ann(&canonical(ty)),
        Node::Call { callee, .. } => call_signature(callee)?
            .1
            .as_ref()
            .and_then(int_width_of_ann),
        Node::IndexAccess { .. } | Node::FieldAccess { .. } => {
            expr_type(element, env).as_ref().and_then(int_width_of_ann)
        }
        Node::Paren(inner, _) | Node::Neg { operand: inner, .. } => element_int_width(inner, env),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => match (element_int_width(left, env), element_int_width(right, env)) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(one), None) | (None, Some(one)) => Some(one),
            (None, None) => None,
        },
        _ => None,
    }
}

fn compatible_array_literal(node: &Node, target_element: &TypeAnn, env: &Env) -> bool {
    let Node::ArrayLit { elements, .. } = node else {
        return false;
    };
    if elements.is_empty() {
        return true;
    }
    if let Some(target_class) = scalar_class_of_ann(target_element) {
        // A literal element proven wider than the slice's integer element type
        // silently truncates in the native build, so it is refused like a
        // wider handle source already is by the element-type comparison.
        let target_width = int_width_of_ann(target_element);
        return elements.iter().all(|element| {
            element_class(element, env) == Some(target_class)
                && target_width.is_none_or(|width| {
                    element_int_width(element, env)
                        .is_none_or(|element_width| element_width <= width)
                })
        });
    }
    elements.iter().all(|element| {
        expr_type(element, env)
            .as_ref()
            .is_some_and(|actual| same_type(actual, target_element))
    })
}

pub(super) fn expr_type(node: &Node, env: &Env) -> Option<TypeAnn> {
    match node {
        Node::Lit(Literal::Ident(name), _) => env.types.get(name).cloned(),
        Node::Lit(Literal::Str(_), _) => Some(TypeAnn::Named("string".to_string())),
        Node::StructLit { name, .. } => Some(canonical(&TypeAnn::Named(name.clone()))),
        Node::Paren(inner, _) => expr_type(inner, env),
        Node::Call { callee, .. } => call_signature(callee).and_then(|(_, ret)| ret),
        Node::FieldAccess {
            receiver, field, ..
        } => field_type(receiver, field, env),
        Node::IndexAccess { receiver, .. } => indexed_type(receiver, env),
        Node::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            let receiver_ty = expr_type(receiver, env)?;
            match (
                collection_owner_name(&receiver_ty),
                method.as_str(),
                args.len(),
            ) {
                (Some("array"), "push", 1)
                | (Some("map"), "insert", 2)
                | (Some("set"), "add" | "insert", 1) => Some(receiver_ty),
                _ => None,
            }
        }
        _ => None,
    }
}

/// True only when a narrow integer destination and the right-hand side have a
/// provably incompatible representation. Unknown expressions remain admitted:
/// this guards against storing an opaque i64 handle's low byte without
/// pretending that the loose type model can classify every expression.
pub(super) fn narrow_integer_rejects_value(target: &TypeAnn, value: &Node, env: &Env) -> bool {
    let narrow = matches!(target, TypeAnn::Named(name) if matches!(name.as_str(), "i8" | "u8" | "i16" | "u16"));
    if !narrow {
        return false;
    }
    if let Some(actual_class) = confident_scalar_class(value, &env.classes) {
        return scalar_class_of_ann(target)
            .is_some_and(|target_class| target_class != actual_class);
    }
    let Some(actual) = expr_type(value, env) else {
        return matches!(
            value,
            Node::ArrayLit { .. }
                | Node::Tuple { .. }
                | Node::StructLit { .. }
                | Node::MapLit { .. }
                | Node::SetLit { .. }
                | Node::Ref { .. }
        );
    };
    if let Some(actual_class) = scalar_class_of_ann(&actual) {
        return scalar_class_of_ann(target)
            .is_some_and(|target_class| target_class != actual_class);
    }
    match actual {
        TypeAnn::Named(name) => {
            matches!(name.as_str(), "string" | "String")
                || env
                    .struct_fields
                    .keys()
                    .any(|(known_owner, _)| known_owner == &name)
        }
        TypeAnn::Generic { .. }
        | TypeAnn::Array { .. }
        | TypeAnn::Slice { .. }
        | TypeAnn::Ref { .. }
        | TypeAnn::Tuple { .. }
        | TypeAnn::Tensor { .. }
        | TypeAnn::DiffTensor { .. }
        | TypeAnn::SparseTensor { .. }
        | TypeAnn::RawPtr { .. }
        | TypeAnn::FnPtr { .. } => true,
        _ => false,
    }
}

pub(super) fn field_type(base: &Node, field: &str, env: &Env) -> Option<TypeAnn> {
    let base_ty = expr_type(base, env)?;
    let struct_name = match &base_ty {
        TypeAnn::Named(name) => name.as_str(),
        TypeAnn::Ref { target, .. } => match target.as_ref() {
            TypeAnn::Named(name) => name.as_str(),
            _ => return None,
        },
        _ => return None,
    };
    let mut keys = vec![struct_name.to_string()];
    if struct_name.contains('.') && !struct_name.starts_with("crate.") {
        keys.push(format!("crate.{struct_name}"));
    }
    keys.into_iter().find_map(|key| {
        let ty = env.struct_fields.get(&(key.clone(), field.to_string()))?;
        Some(read_field_type(ty, &key))
    })
}

/// A local struct is keyed bare. A project struct is keyed by its source key
/// (`crate.defs.Bank`), and its field types are written in that owner's scope.
#[cfg(feature = "cross-module-imports")]
fn read_field_type(ty: &TypeAnn, struct_key: &str) -> TypeAnn {
    match struct_key.rsplit_once('.') {
        Some((owner, _)) => super::canonical_in_owner(ty, owner),
        None => canonical(ty),
    }
}

#[cfg(not(feature = "cross-module-imports"))]
fn read_field_type(ty: &TypeAnn, _struct_key: &str) -> TypeAnn {
    canonical(ty)
}

pub(super) fn indexed_type(receiver: &Node, env: &Env) -> Option<TypeAnn> {
    indexed_type_from_ann(&expr_type(receiver, env)?)
}

pub(super) fn indexed_type_from_ann(receiver: &TypeAnn) -> Option<TypeAnn> {
    match receiver {
        TypeAnn::Generic { name, args } if name == "array" => args.first().cloned(),
        TypeAnn::Array { element, .. } | TypeAnn::Slice { element, .. } => {
            Some((**element).clone())
        }
        _ => None,
    }
}

pub(super) fn collection_owner_name(ty: &TypeAnn) -> Option<&str> {
    match ty {
        TypeAnn::Generic { name, .. } if matches!(name.as_str(), "array" | "map" | "set") => {
            Some(name)
        }
        _ => None,
    }
}

pub(super) fn is_collection_owner_type(ty: &TypeAnn) -> bool {
    collection_owner_name(ty).is_some()
}

fn compatible_value(value: &Node, target: &TypeAnn, env: &Env) -> bool {
    if is_collection_owner_type(target) {
        return compatible_collection_owner_source(target, value, env);
    }
    if let Some(target_class) = scalar_class_of_ann(target) {
        return confident_scalar_class(value, &env.classes) == Some(target_class);
    }
    expr_type(value, env)
        .as_ref()
        .is_some_and(|actual| same_type(actual, target))
}

pub(super) fn compatible_collection_owner_source(
    target: &TypeAnn,
    value: &Node,
    env: &Env,
) -> bool {
    if let Node::Paren(inner, _) = value {
        return compatible_collection_owner_source(target, inner, env);
    }
    match (target, value) {
        (TypeAnn::Generic { name, args }, Node::ArrayLit { elements, .. })
            if name == "array" && args.len() == 1 =>
        {
            elements
                .iter()
                .all(|element| compatible_value(element, &args[0], env))
        }
        (TypeAnn::Generic { name, args }, Node::MapLit { entries, .. })
            if name == "map" && args.len() == 2 =>
        {
            entries.iter().all(|(key, value)| {
                compatible_value(key, &args[0], env) && compatible_value(value, &args[1], env)
            })
        }
        (TypeAnn::Generic { name, args }, Node::SetLit { elements, .. })
            if name == "set" && args.len() == 1 =>
        {
            elements
                .iter()
                .all(|element| compatible_value(element, &args[0], env))
        }
        _ => expr_type(value, env)
            .as_ref()
            .is_some_and(|actual| same_type(actual, target)),
    }
}

pub(super) fn same_type(left: &TypeAnn, right: &TypeAnn) -> bool {
    if left == right {
        return true;
    }
    match (left, right) {
        (TypeAnn::Named(a), TypeAnn::Named(b)) => {
            matches!(a.as_str(), "string" | "String") && matches!(b.as_str(), "string" | "String")
        }
        (
            TypeAnn::Generic {
                name: left_name,
                args: left_args,
            },
            TypeAnn::Generic {
                name: right_name,
                args: right_args,
            },
        ) => {
            left_name == right_name
                && left_args.len() == right_args.len()
                && left_args
                    .iter()
                    .zip(right_args)
                    .all(|(left, right)| same_type(left, right))
        }
        (
            TypeAnn::Array {
                element: left_element,
                length: left_length,
            },
            TypeAnn::Array {
                element: right_element,
                length: right_length,
            },
        ) => left_length == right_length && same_type(left_element, right_element),
        (
            TypeAnn::Slice {
                mutable: left_mutable,
                element: left_element,
            },
            TypeAnn::Slice {
                mutable: right_mutable,
                element: right_element,
            },
        ) => left_mutable == right_mutable && same_type(left_element, right_element),
        (
            TypeAnn::Ref {
                mutable: left_mutable,
                target: left_target,
            },
            TypeAnn::Ref {
                mutable: right_mutable,
                target: right_target,
            },
        ) => left_mutable == right_mutable && same_type(left_target, right_target),
        (
            TypeAnn::Tuple {
                elements: left_elements,
            },
            TypeAnn::Tuple {
                elements: right_elements,
            },
        ) => {
            left_elements.len() == right_elements.len()
                && left_elements
                    .iter()
                    .zip(right_elements)
                    .all(|(left, right)| same_type(left, right))
        }
        _ => false,
    }
}

pub(super) fn compatible_source(target: &HandleKind, value: &Node, env: &Env) -> bool {
    compatible_array_literal(value, target.element(), env)
        || expr_kind(value, env).as_ref().is_some_and(|source| {
            same_type(source.element(), target.element())
                && match target {
                    HandleKind::Array(_) => matches!(source, HandleKind::Array(_)),
                    HandleKind::Slice { mutable: false, .. } => true,
                    HandleKind::Slice { mutable: true, .. } => matches!(
                        source,
                        HandleKind::Array(_) | HandleKind::Slice { mutable: true, .. }
                    ),
                }
        })
        // Existing public raw-handle interop explicitly retypes an opaque i64
        // identifier/call as an owned array. Borrow flow is checked first, so a
        // slice can never reach this compatibility lane by erasing its type.
        || (matches!(target, HandleKind::Array(_))
            && !value_contains_borrow(value, env)
            && matches!(value, Node::Lit(Literal::Ident(_), _) | Node::Call { .. }))
}
