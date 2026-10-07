// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `::` path expressions.
//!
//! A `Type::Variant` path keeps the full path as the callee or identifier; the
//! type checker resolves it. A path whose head is an imported module (`m::f(x)`,
//! `m::K`) is the same reference as `m.f(x)` / `m.K` and desugars the same way:
//! to the bare member, with the qualifier recorded for the formatter. Kept as
//! `m::f`, the callee reached MLIR as the symbol `@m::f`, which `mlir-opt`
//! rejects, and the callee's signature (its `f64` parameters) was never found.

use super::{P, ParseError};
use crate::ast::{Literal, Node, Span};

impl P<'_> {
    pub(super) fn parse_path_expr(
        &mut self,
        ident: String,
        start: usize,
    ) -> Result<Node, ParseError> {
        let member = self.imported_module_member(&ident);
        if self.at(b'(') {
            let mut node = self.parse_generic_call(ident, start)?;
            self.capture_path_call(&node);
            if let (Some((qualifier, member)), Node::Call { callee, span, .. }) =
                (member, &mut node)
            {
                *callee = member;
                self.record_path_qualifier(*span, &qualifier);
            }
            Ok(node)
        } else if self.at(b'{') && self.struct_lit_body_ahead() {
            self.parse_struct_literal(ident, start)
        } else {
            let span = Span::new(start, self.pos);
            self.capture_path_value(&ident, span);
            match member {
                Some((qualifier, member)) => {
                    self.record_path_qualifier(span, &qualifier);
                    Ok(Node::Lit(Literal::Ident(member), span))
                }
                None => Ok(Node::Lit(Literal::Ident(ident), span)),
            }
        }
    }

    /// `(qualifier, member)` when `ident` is `qualifier::member` and `qualifier`
    /// names an imported module (`m`, or a dotted import path written `a::b`).
    fn imported_module_member(&self, ident: &str) -> Option<(String, String)> {
        let (qualifier, member) = ident.rsplit_once("::")?;
        let plain = !member.is_empty()
            && member
                .bytes()
                .all(|b| b == b'_' || b.is_ascii_alphanumeric());
        let dotted = qualifier.replace("::", ".");
        let imported =
            self.imports.iter().any(|m| m == qualifier) || self.import_paths.contains(&dotted);
        (plain && imported).then(|| (qualifier.to_string(), member.to_string()))
    }

    fn record_path_qualifier(&mut self, span: Span, qualifier: &str) {
        self.record_qualifier(span, qualifier);
        self.spelling.path_qualifiers.push(span);
    }
}
