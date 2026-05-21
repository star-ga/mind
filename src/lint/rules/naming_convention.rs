// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Rule `lint::naming_convention` — identifier naming conventions.
//!
//! | Declaration kind | Expected convention |
//! |---|---|
//! | `fn` name | `lower_snake_case` |
//! | `let` binding name | `lower_snake_case` |
//! | `struct` name | `UpperCamelCase` |
//! | `enum` name | `UpperCamelCase` |
//! | `type` alias name | `UpperCamelCase` |
//! | `const` name | `SCREAMING_SNAKE_CASE` |
//!
//! Exemptions (no diagnostic emitted):
//! - Single-character names (including `_` by itself).
//! - Names with a leading `_` prefix are allowed for any convention
//!   (they are typically intentionally unused, e.g. `_result`).
//! - `fn` names starting with `__` (intrinsic/FFI convention in MIND stdlib).

use crate::ast::Node;
use crate::lint::rule::{LintCtx, LintRule};
use crate::lint::{Diagnostic, SourceSpan};
use crate::project::RuleSeverity;

/// Lint rule: identifier naming conventions.
pub struct NamingConvention;

impl LintRule for NamingConvention {
    fn id(&self) -> &'static str {
        "lint::naming_convention"
    }

    fn default_severity(&self) -> RuleSeverity {
        RuleSeverity::Warn
    }

    fn description(&self) -> &'static str {
        "identifier naming convention: fn/let → lower_snake_case, struct/enum/type → UpperCamelCase, const → SCREAMING_SNAKE_CASE"
    }

    fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        for item in &ctx.module.items {
            check_node(item, ctx, &mut diagnostics);
        }
        diagnostics
    }
}

fn check_node(node: &Node, ctx: &LintCtx<'_>, out: &mut Vec<Diagnostic>) {
    match node {
        Node::FnDef { name, body, span, .. } => {
            if !is_exempt(name) && !name.starts_with("__") && !is_lower_snake(name) {
                emit(
                    out,
                    ctx,
                    name,
                    "lower_snake_case",
                    span.start(),
                    span.end(),
                );
            }
            for stmt in body {
                check_node(stmt, ctx, out);
            }
        }
        Node::StructDef { name, span, .. } => {
            if !is_exempt(name) && !is_upper_camel(name) {
                emit(out, ctx, name, "UpperCamelCase", span.start(), span.end());
            }
        }
        Node::EnumDef { name, span, .. } => {
            if !is_exempt(name) && !is_upper_camel(name) {
                emit(out, ctx, name, "UpperCamelCase", span.start(), span.end());
            }
        }
        Node::TypeAlias { name, span, .. } => {
            if !is_exempt(name) && !is_upper_camel(name) {
                emit(out, ctx, name, "UpperCamelCase", span.start(), span.end());
            }
        }
        Node::Const { name, span, .. } => {
            if !is_exempt(name) && !is_screaming_snake(name) {
                emit(
                    out,
                    ctx,
                    name,
                    "SCREAMING_SNAKE_CASE",
                    span.start(),
                    span.end(),
                );
            }
        }
        Node::Let { name, value, span, .. } => {
            if !is_exempt(name) && !is_lower_snake(name) {
                emit(
                    out,
                    ctx,
                    name,
                    "lower_snake_case",
                    span.start(),
                    span.end(),
                );
            }
            check_node(value, ctx, out);
        }
        // Recurse into compound nodes.
        Node::Block { stmts, .. } => {
            for s in stmts { check_node(s, ctx, out); }
        }
        Node::If { cond, then_branch, else_branch, .. } => {
            check_node(cond, ctx, out);
            for s in then_branch { check_node(s, ctx, out); }
            if let Some(eb) = else_branch {
                for s in eb { check_node(s, ctx, out); }
            }
        }
        Node::For { start, end, body, .. } => {
            check_node(start, ctx, out);
            check_node(end, ctx, out);
            for s in body { check_node(s, ctx, out); }
        }
        Node::Return { value, .. } => {
            if let Some(v) = value { check_node(v, ctx, out); }
        }
        Node::Assign { value, .. } => { check_node(value, ctx, out); }
        Node::Binary { left, right, .. } => {
            check_node(left, ctx, out);
            check_node(right, ctx, out);
        }
        Node::Bitwise { left, right, .. } => {
            check_node(left, ctx, out);
            check_node(right, ctx, out);
        }
        Node::Logical { left, right, .. } => {
            check_node(left, ctx, out);
            check_node(right, ctx, out);
        }
        Node::Paren(inner, _) => { check_node(inner, ctx, out); }
        Node::Neg { operand, .. } => { check_node(operand, ctx, out); }
        Node::As { expr, .. } => { check_node(expr, ctx, out); }
        Node::Match { scrutinee, arms, .. } => {
            check_node(scrutinee, ctx, out);
            for arm in arms { check_node(&arm.body, ctx, out); }
        }
        Node::Ref { inner, .. } => { check_node(inner, ctx, out); }
        // No naming obligations on remaining node kinds at top-level walk.
        _ => {}
    }
}

fn emit(
    out: &mut Vec<Diagnostic>,
    ctx: &LintCtx<'_>,
    name: &str,
    expected_case: &str,
    start: usize,
    end: usize,
) {
    out.push(Diagnostic {
        rule_id: "lint::naming_convention".to_string(),
        severity: RuleSeverity::Warn,
        message: format!(
            "identifier `{name}` should be {expected_case}"
        ),
        file: ctx.file.to_path_buf(),
        span: SourceSpan { start, end },
        help: Some(format!("rename `{name}` to follow {expected_case}")),
    });
}

/// Exempt single-character names and names that begin with `_`.
fn is_exempt(name: &str) -> bool {
    name.len() <= 1 || name.starts_with('_')
}

/// `lower_snake_case`: all chars are lowercase ASCII letters, digits, or `_`.
/// The name must not start with an uppercase letter.
fn is_lower_snake(name: &str) -> bool {
    if name.is_empty() {
        return true;
    }
    // No uppercase letters.
    !name.chars().any(|c| c.is_ascii_uppercase())
}

/// `UpperCamelCase`: starts with an uppercase letter; no underscores
/// (except a leading `_` which is handled by `is_exempt`).
fn is_upper_camel(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() => {}
        _ => return false,
    }
    // No underscores in the body (excluding trailing ones, which are rare but okay).
    !name.contains('_')
}

/// `SCREAMING_SNAKE_CASE`: all uppercase letters, digits, and `_`.
fn is_screaming_snake(name: &str) -> bool {
    name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}
