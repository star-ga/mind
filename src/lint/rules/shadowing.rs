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

//! Rule `lint::shadowing` — detect `let x` that shadows a previous `let x`
//! binding in the **same scope**.
//!
//! Shadowing is legal MIND, but a second `let x` in the same function body
//! or block is often an unintentional re-bind.  This rule flags it so the
//! author can either rename the variable or add a `#[allow(lint::shadowing)]`
//! annotation to document intent.
//!
//! ## Scope model
//!
//! "Same scope" means the same flat list of statements at one nesting level:
//! - The top-level body of a `fn` definition.
//! - A `Block` expression body (`{ stmts }`).
//! - The `then` or `else` branch of an `if` expression.
//! - The body of a `for` loop.
//!
//! A `let x` introduced in an outer scope and a `let x` introduced in a
//! nested scope (e.g. an `if` body inside a `fn`) are in **different scopes**
//! and do NOT trigger this rule.  Only two `let x` at the **same** nesting
//! level do.

use std::collections::HashMap;

use crate::ast::Node;
use crate::lint::rule::{LintCtx, LintRule};
use crate::lint::{Diagnostic, SourceSpan};
use crate::project::RuleSeverity;

/// Lint rule: same-scope `let` shadowing.
pub struct Shadowing;

impl LintRule for Shadowing {
    fn id(&self) -> &'static str {
        "lint::shadowing"
    }

    fn default_severity(&self) -> RuleSeverity {
        RuleSeverity::Warn
    }

    fn description(&self) -> &'static str {
        "let binding shadows a previous binding in the same scope — rename if unintentional"
    }

    fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        for item in &ctx.module.items {
            check_node(item, ctx, &mut diagnostics);
        }
        diagnostics
    }
}

/// Walk a node, checking for shadowing inside function/block scopes.
fn check_node(node: &Node, ctx: &LintCtx<'_>, out: &mut Vec<Diagnostic>) {
    match node {
        Node::FnDef { body, .. } => {
            check_scope(body, ctx, out);
            // No need to recurse into body here: check_scope recurses into
            // nested blocks for the sub-scopes already.
        }
        Node::Block { stmts, .. } => {
            check_scope(stmts, ctx, out);
        }
        // For other compound nodes, just recurse to find nested fn/block scopes.
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            check_node(cond, ctx, out);
            check_scope(then_branch, ctx, out);
            if let Some(eb) = else_branch {
                check_scope(eb, ctx, out);
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            check_node(start, ctx, out);
            check_node(end, ctx, out);
            check_scope(body, ctx, out);
        }
        Node::Let { value, .. } => {
            check_node(value, ctx, out);
        }
        Node::Assign { value, .. } => {
            check_node(value, ctx, out);
        }
        Node::Return { value: Some(v), .. } => {
            check_node(v, ctx, out);
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            check_node(scrutinee, ctx, out);
            for arm in arms {
                check_node(&arm.body, ctx, out);
            }
        }
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
        Node::Paren(inner, _) => {
            check_node(inner, ctx, out);
        }
        Node::Neg { operand, .. } => {
            check_node(operand, ctx, out);
        }
        Node::As { expr, .. } => {
            check_node(expr, ctx, out);
        }
        Node::Call { args, .. } => {
            for a in args {
                check_node(a, ctx, out);
            }
        }
        Node::MethodCall { receiver, args, .. } => {
            check_node(receiver, ctx, out);
            for a in args {
                check_node(a, ctx, out);
            }
        }
        Node::FieldAccess { receiver, .. } => {
            check_node(receiver, ctx, out);
        }
        Node::Ref { inner, .. } => {
            check_node(inner, ctx, out);
        }
        // Leaf / declaration nodes without sub-scopes.
        _ => {}
    }
}

/// Check a flat list of statements for same-scope `let` shadowing.
///
/// Tracks names introduced by `Let` nodes in declaration order; flags any
/// subsequent `Let` whose name matches a previously-introduced name.
///
/// Recurse into nested scopes **after** building the shadowing map so that
/// inner scopes are independent.
fn check_scope(stmts: &[Node], ctx: &LintCtx<'_>, out: &mut Vec<Diagnostic>) {
    // Map: name → (first_decl_start, first_decl_end)
    let mut seen: HashMap<&str, (usize, usize)> = HashMap::new();

    for stmt in stmts {
        if let Node::Let {
            name, value, span, ..
        } = stmt
        {
            if let Some(&(first_start, first_end)) = seen.get(name.as_str()) {
                // Second (or later) `let name` in the same scope — shadowing.
                let _ = (first_start, first_end); // first decl location noted for message
                out.push(Diagnostic {
                    rule_id: "lint::shadowing".to_string(),
                    severity: RuleSeverity::Warn,
                    message: format!(
                        "`let {name}` shadows previous binding in the same scope — rename if unintentional"
                    ),
                    file: ctx.file.to_path_buf(),
                    span: SourceSpan {
                        start: span.start(),
                        end: span.end(),
                    },
                    help: Some(format!(
                        "rename `{name}` or use `#[allow(lint::shadowing, reason = \"intentional\")]`"
                    )),
                    auto_fix: None,
                });
            } else {
                seen.insert(name.as_str(), (span.start(), span.end()));
            }
            // Recurse into the value expression for nested scopes.
            check_node(value, ctx, out);
        } else {
            // Non-let statement — recurse for any nested fn/block scopes.
            check_node(stmt, ctx, out);
        }
    }
}
