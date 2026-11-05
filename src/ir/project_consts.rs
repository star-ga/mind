// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Read access to the whole-project cross-module const table for checkers.

/// The whole-project cross-module value of const `name`, ignoring the
/// lowering pass's module table (which is empty while a file is checked).
pub fn project_const_value(name: &str) -> Option<crate::ast::Node> {
    super::PROJECT_CONSTS.with(|c| c.borrow().get(name).cloned())
}
