// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The `.name` postfix that is not a call and not `.T`.
//!
//! An imported `mod.CONST` collapses to the bare member. A following
//! `Enum.Variant` collapses to the `Enum::Variant` ident when the enum is
//! visible from this module and that unit variant is in the project registry.
//! Anything else stays a field access, so a local named like an enum that this
//! module does not import keeps its field.

use super::P;
use super::eval_imports::EvalImportRefKind;
use crate::ast::{Literal, Node, Span};

impl P<'_> {
    pub(super) fn dotted_field(&mut self, node: Node, method: String) -> Node {
        if let Node::Lit(Literal::Ident(recv_name), _) = &node {
            let recv_name = recv_name.clone();
            if self.imports.iter().any(|s| s == &recv_name) {
                let span = Span::new(node.span_start(), self.pos);
                if self.eval_import_refs.is_some() {
                    self.record_eval_import_ref(
                        span,
                        vec![recv_name.clone()],
                        &method,
                        EvalImportRefKind::Value,
                    );
                }
                self.record_qualifier(span, &recv_name);
                return Node::Lit(Literal::Ident(method), span);
            }
            if self.is_enum_name(&recv_name)
                && let Some(key) = self.enum_scope.unit_variant_key(&recv_name, &method)
            {
                let span = Span::new(node.span_start(), self.pos);
                return Node::Lit(Literal::Ident(key), span);
            }
        }
        let span = Span::new(node.span_start(), self.pos);
        Node::FieldAccess {
            receiver: Box::new(node),
            field: method,
            span,
        }
    }
}
