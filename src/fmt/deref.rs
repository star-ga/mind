// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Canonical formatting helpers for D4 dereference nodes.

use crate::ast::Node;

pub(super) fn emit_deref(p: &mut super::Printer, operand: &Node) {
    p.push("*");
    super::emit_expr(p, operand);
}

pub(super) fn emit_assign(p: &mut super::Printer, target: &Node, value: &Node) {
    p.push("*");
    super::emit_expr(p, target);
    p.push(" = ");
    super::emit_expr(p, value);
}
