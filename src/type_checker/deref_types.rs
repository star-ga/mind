// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Type inference helpers for the D4 dereference capability boundary.

use crate::ast::Node;
use crate::ast::TypeAnn;
use crate::types::ValueType;

// Use parent checker types through the private module boundary
#[cfg(not(feature = "std-surface"))]
use super::Pretty;
use super::{TypeEnv, TypeErrSpan};

type Infer = fn(&Node, &TypeEnv) -> Result<(ValueType, crate::ast::Span), TypeErrSpan>;

pub(super) fn value_type_for_ref(mutable: bool, target: &TypeAnn) -> ValueType {
    ValueType::Ref {
        mutable,
        target: match target {
            TypeAnn::Named(name) => name.clone(),
            _ => String::new(),
        },
    }
}

pub(super) fn reference_type(
    inner: &Node,
    mutable: bool,
    span: crate::ast::Span,
    env: &TypeEnv,
    infer: Infer,
) -> Result<(ValueType, crate::ast::Span), TypeErrSpan> {
    infer(inner, env)?;
    Ok((
        ValueType::Ref {
            mutable,
            target: String::new(),
        },
        span,
    ))
}

pub(super) fn infer_deref(
    node: &Node,
    env: &TypeEnv,
    infer: Infer,
) -> Result<(ValueType, crate::ast::Span), TypeErrSpan> {
    #[cfg(not(feature = "std-surface"))]
    let _ = (env, infer);
    #[cfg(feature = "std-surface")]
    {
        return match node {
            Node::Deref { operand, span } => match infer(operand, env) {
                Ok((ValueType::Ref { .. }, _)) => Ok((ValueType::ScalarI64, *span)),
                _ => Err(TypeErrSpan { msg: "dereference `*expr` requires a reference operand (a `&mut <struct>` parameter in the deref-assign field-first subset).".into(), span: *span }),
            },
            Node::DerefAssign { target, value, span } => {
                let _ = infer(value, env);
                match infer(target, env) {
                    Ok((ValueType::Ref { .. }, _)) => Ok((ValueType::ScalarI64, *span)),
                    _ => Err(TypeErrSpan { msg: "assignment through a dereference `*p = value` requires a `&mut` reference target.".into(), span: *span }),
                }
            }
            _ => unreachable!("D4 inference helper called for a non-deref node"),
        };
    }
    #[cfg(not(feature = "std-surface"))]
    {
        let span = match node {
            Node::Deref { span, .. } | Node::DerefAssign { span, .. } => *span,
            _ => unreachable!("D4 inference helper called for a non-deref node"),
        };
        let msg = if matches!(node, Node::Deref { .. }) {
            "dereference `*expr` is not supported: the reference/place ABI requires the std-surface feature."
        } else {
            "assignment through a dereference `*p = value` is not supported: the mutable-place ABI requires the std-surface feature."
        };
        Err(TypeErrSpan {
            msg: msg.into(),
            span,
        })
    }
}

#[cfg(not(feature = "std-surface"))]
pub(super) fn scan_module_derefs(
    module: &crate::ast::Module,
    src: &str,
    file: Option<&str>,
) -> Vec<Pretty> {
    fn scan(node: &Node, src: &str, file: Option<&str>, out: &mut Vec<Pretty>) {
        match node {
            Node::Deref { span, .. } => out.push(super::diag_from_span(src, file, "dereference `*expr` is not supported yet: the reference/place ABI is unimplemented (deref-assign is under architecture review). It is a parse/format-only construct today.".into(), *span, "E2028")),
            Node::DerefAssign { span, .. } => out.push(super::diag_from_span(src, file, "assignment through a dereference `*p = value` is not supported yet: the mutable-place ABI is unimplemented (deref-assign is under architecture review). Record-identity place replacement is not a field copy.".into(), *span, "E2029")),
            _ => {}
        }
        super::nerve_walk::for_each_child(node, &mut |child| scan(child, src, file, out));
    }
    let mut out = Vec::new();
    for item in &module.items {
        scan(item, src, file, &mut out);
    }
    out
}
