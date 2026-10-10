// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
//! Source facts lost by scalar IR lowering (pointer casts and declared widths).
use super::FrozenProfileRejection;
use super::scalar::{Scalar, declared};
use crate::ast::{Literal, Module, Node, TypeAnn};
use std::collections::BTreeMap;

pub(crate) fn admit(module: &Module) -> Result<(), FrozenProfileRejection> {
    let mut error = crate::eval::narrow_scan::module_mentions_unsupported_native_integer(module)
        .then_some("type.non_i64_integer");
    for item in &module.items {
        check_casts(item, &mut error);
    }
    error.map_or(Ok(()), |construct| {
        Err(FrozenProfileRejection { construct })
    })
}

fn expr_type(node: &Node, env: &BTreeMap<String, Option<Scalar>>) -> Option<Scalar> {
    match node {
        Node::Lit(Literal::Int(_), _) => Some(Scalar::Signed),
        Node::Lit(Literal::Float(_), _) => Some(Scalar::Float),
        Node::Lit(Literal::Ident(n), _) => env.get(n).copied().flatten(),
        Node::Paren(n, _) | Node::Neg { operand: n, .. } => expr_type(n, env),
        Node::Binary { left, right, .. } => {
            let a = expr_type(left, env)?;
            (expr_type(right, env) == Some(a)).then_some(a)
        }
        Node::As { ty, .. } => declared(ty),
        Node::Call { callee, .. } => super::scalar::conversion(callee).or_else(|| {
            crate::ir::frozen_intrinsics::frozen_native_intrinsic(callee)
                .filter(|r| r.result == crate::ir::frozen_intrinsics::FrozenResult::Value)
                .map(|_| Scalar::Signed)
        }),
        _ => None,
    }
}

fn check_casts(node: &Node, error: &mut Option<&'static str>) {
    let mut env = BTreeMap::new();
    let mut annotations = BTreeMap::new();
    // Refuse ambiguous/shadowed bindings conservatively rather than borrowing
    // a same-named variable's type from another scope.
    crate::type_checker::nerve_walk::walk(node, &mut |n| {
        if let Node::FnDef(f, _) = n {
            for p in &f.params {
                env.entry(p.name.clone())
                    .and_modify(|t| *t = None)
                    .or_insert(declared(&p.ty));
            }
        }
        if let Node::Let {
            name, ann, value, ..
        } = n
        {
            if let Some(t) = ann.as_ref().and_then(declared) {
                annotations.insert(name.clone(), Some(t));
            }
            let inferred = expr_type(value, &env);
            let ty = match ann {
                Some(t) => declared(t).filter(|t| {
                    Some(*t) == inferred
                        || (*t == Scalar::Unsigned && inferred == Some(Scalar::Signed))
                }),
                None => inferred,
            };
            env.entry(name.clone())
                .and_modify(|t| *t = None)
                .or_insert(ty);
        }
        if let Node::Assign { name, value, .. } = n {
            let ty = expr_type(value, &env);
            if env.get(name) != Some(&ty) {
                env.insert(name.clone(), None);
            }
        }
    });
    let mut typed = env.clone();
    typed.extend(annotations);
    crate::type_checker::nerve_walk::walk(node, &mut |n| {
        let binary = match n {
            Node::Binary {
                op: crate::ast::BinOp::Div,
                left,
                right,
                ..
            } => Some((left, right, "binop.div_operand_type")),
            Node::Binary {
                op: crate::ast::BinOp::Mod,
                left,
                right,
                ..
            } => Some((left, right, "binop.mod_operand_type")),
            #[cfg(feature = "std-surface")]
            Node::Bitwise {
                op: crate::ast::BitOp::Shl,
                left,
                right,
                ..
            } => Some((left, right, "binop.shl_operand_type")),
            #[cfg(feature = "std-surface")]
            Node::Bitwise {
                op: crate::ast::BitOp::Shr,
                left,
                right,
                ..
            } => Some((left, right, "binop.shr_operand_type")),
            _ => None,
        };
        if let Some((a, b, label)) = binary {
            for expr in [a, b] {
                if expr_type(expr, &typed) == Some(Scalar::Float) {
                    error.get_or_insert(label);
                }
            }
        }
        if let Node::As {
            expr,
            ty: TypeAnn::RawPtr { .. },
            ..
        } = n
        {
            if !matches!(
                expr_type(expr, &env),
                Some(Scalar::Signed | Scalar::Unsigned)
            ) {
                error.get_or_insert("cast.pointer_operand_type");
            }
        }
    });
}
