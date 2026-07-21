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

//! Rule `lint::q16_overflow` — detect bare `i32 * i32` without `>>16` narrowing.
//!
//! The Q16.16 fixed-point invariant requires that `i32 × i32` products be
//! widened to `i64`, shifted right by 16, and narrowed back to `i32`:
//! `((a as i64 * b as i64) >> 16) as i32`.  A bare `a * b` where the result
//! is consumed into an explicitly `i32`-typed binding without an intervening
//! `>> 16` shift is the canonical Q16.16 overflow bug.
//!
//! ## Detection strategy (syntactic, conservative)
//!
//! Full type inference is not available at the lint layer.  This rule uses a
//! targeted heuristic anchored on **the consuming `Let` annotation**:
//!
//! 1. Walk every `Let { ann: Some(ScalarI32), value }` node.
//! 2. Check whether `value` contains a `BinOp::Mul` that is not wrapped in a
//!    `>> 16` shift before reaching `i32`.
//! 3. Only flag `Mul` nodes that are:
//!    - A direct child of the `i32` `Let`, OR
//!    - Reached through casts/parens without an intervening `>> 16`.
//!
//! Also flag `BinOp::Mul` inside a `Return` in a function whose declared
//! return type is `i32`.
//!
//! Conservative: only flags when there is strong evidence the result is `i32`.
//! False negatives (misses) are acceptable; false positives are not.
//!
//! Per-target escalation (`warn` on most targets, `error` on `[mindcraft.cpu]`)
//! is deferred to Phase 5 when per-target severity plumbing lands.

use crate::ast::{BinOp, BitOp, Node, TypeAnn};
use crate::lint::rule::{LintCtx, LintRule};
use crate::lint::{Diagnostic, SourceSpan};
use crate::project::RuleSeverity;

/// Lint rule: bare `i32 * i32` without `>> 16` narrowing (Q16.16 overflow).
pub struct Q16Overflow;

impl LintRule for Q16Overflow {
    fn id(&self) -> &'static str {
        "lint::q16_overflow"
    }

    fn default_severity(&self) -> RuleSeverity {
        // Per-target escalation to Error for [mindcraft.cpu] is deferred to
        // Phase 5 (per-target severity plumbing not yet wired).
        RuleSeverity::Warn
    }

    fn description(&self) -> &'static str {
        "bare i32 * i32 without >>16 narrowing — potential Q16.16 overflow"
    }

    fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        for item in &ctx.module.items {
            scan_top(item, false, ctx, &mut diagnostics);
        }
        diagnostics
    }
}

/// Returns `true` if `node` is an integer literal `16`.
fn is_lit_16(node: &Node) -> bool {
    matches!(node, Node::Lit(crate::ast::Literal::Int(16), _))
}

/// Scan a top-level module item, looking for fn bodies to descend into.
fn scan_top(node: &Node, ret_is_i32: bool, ctx: &LintCtx<'_>, out: &mut Vec<Diagnostic>) {
    match node {
        Node::FnDef(fd, _) => {
            let (ret_type, body) = (&fd.ret_type, &fd.body);
            let fn_ret_i32 = matches!(ret_type, Some(TypeAnn::ScalarI32));
            for stmt in body {
                scan_stmt(stmt, fn_ret_i32, ctx, out);
            }
        }
        _ => scan_stmt(node, ret_is_i32, ctx, out),
    }
}

/// Scan a statement for Q16.16 overflow patterns.
///
/// `fn_ret_i32` is true when the enclosing function returns `i32`, so a
/// `Return` with an unshifted `Mul` can be flagged.
fn scan_stmt(node: &Node, fn_ret_i32: bool, ctx: &LintCtx<'_>, out: &mut Vec<Diagnostic>) {
    match node {
        // `let x: i32 = <expr>` — check if <expr> contains an unshifted Mul.
        Node::Let {
            ann: Some(TypeAnn::ScalarI32),
            value,
            ..
        } => {
            if let Some(mul_span) = find_unshifted_mul(value) {
                out.push(make_diag(ctx, mul_span.0, mul_span.1));
            }
            // Recurse in case value contains nested fn defs or blocks.
            scan_expr(value, fn_ret_i32, ctx, out);
        }
        // Propagate into blocks and control flow.
        Node::FnDef(fd, _) => {
            let (ret_type, body) = (&fd.ret_type, &fd.body);
            let inner_ret_i32 = matches!(ret_type, Some(TypeAnn::ScalarI32));
            for s in body {
                scan_stmt(s, inner_ret_i32, ctx, out);
            }
        }
        Node::Block { stmts, .. } => {
            for s in stmts {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            scan_expr(cond, fn_ret_i32, ctx, out);
            for s in then_branch {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    scan_stmt(s, fn_ret_i32, ctx, out);
                }
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            scan_expr(start, fn_ret_i32, ctx, out);
            scan_expr(end, fn_ret_i32, ctx, out);
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        Node::ForEach {
            collection, body, ..
        } => {
            scan_expr(collection, fn_ret_i32, ctx, out);
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            scan_expr(cond, fn_ret_i32, ctx, out);
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        Node::Return { value, .. } if fn_ret_i32 => {
            if let Some(v) = value {
                if let Some(mul_span) = find_unshifted_mul(v) {
                    out.push(make_diag(ctx, mul_span.0, mul_span.1));
                }
                scan_expr(v, fn_ret_i32, ctx, out);
            }
        }
        // For non-i32-annotated lets and other stmts, recurse into sub-expressions
        // to handle nested fn bodies and blocks.
        Node::Let { value, .. } => scan_expr(value, fn_ret_i32, ctx, out),
        Node::Assign { value, .. } => scan_expr(value, fn_ret_i32, ctx, out),
        Node::Return { value, .. } => {
            if let Some(v) = value {
                scan_expr(v, fn_ret_i32, ctx, out);
            }
        }
        _ => scan_expr(node, fn_ret_i32, ctx, out),
    }
}

/// Light scan of an expression for nested fn-def or block scopes.
fn scan_expr(node: &Node, fn_ret_i32: bool, ctx: &LintCtx<'_>, out: &mut Vec<Diagnostic>) {
    match node {
        Node::FnDef(fd, _) => {
            let (ret_type, body) = (&fd.ret_type, &fd.body);
            let inner = matches!(ret_type, Some(TypeAnn::ScalarI32));
            for s in body {
                scan_stmt(s, inner, ctx, out);
            }
        }
        Node::Block { stmts, .. } => {
            for s in stmts {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            scan_expr(cond, fn_ret_i32, ctx, out);
            for s in then_branch {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    scan_stmt(s, fn_ret_i32, ctx, out);
                }
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            scan_expr(start, fn_ret_i32, ctx, out);
            scan_expr(end, fn_ret_i32, ctx, out);
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        Node::ForEach {
            collection, body, ..
        } => {
            scan_expr(collection, fn_ret_i32, ctx, out);
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            scan_expr(cond, fn_ret_i32, ctx, out);
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => {
            for s in body {
                scan_stmt(s, fn_ret_i32, ctx, out);
            }
        }
        _ => {}
    }
}

/// Look for a `BinOp::Mul` inside `expr` that is not protected by a `>> 16`
/// shift at or above it.  Returns the span of the Mul if found.
///
/// The walk descends through:
/// - `Paren(inner)` — transparent
/// - `As { expr, .. }` — transparent (the cast does not shift)
/// - `Bitwise { op: Shr, right: Lit(16) }` — the left child IS shifted; stop.
///
/// Returns `Some((start, end))` for the first unshifted Mul found.
fn find_unshifted_mul(expr: &Node) -> Option<(usize, usize)> {
    find_mul_inner(expr, false)
}

fn find_mul_inner(expr: &Node, under_shr16: bool) -> Option<(usize, usize)> {
    match expr {
        Node::Binary {
            op: BinOp::Mul,
            span,
            left,
            right,
        } => {
            if under_shr16 {
                // This Mul is safely shifted — but its children might not be.
                // For simplicity: once we're under a shift, the product is safe.
                return None;
            }
            // Found an unshifted Mul.
            let _ = (left, right); // don't recurse into children of the flagged Mul
            Some((span.start(), span.end()))
        }
        Node::Bitwise {
            op: BitOp::Shr,
            left,
            right,
            ..
        } if is_lit_16(right) => {
            // Left child is under a >>16 shift — any Mul there is safe.
            // The right child is Lit(16) which can never contain a Mul.
            find_mul_inner(left, true)
        }
        Node::Paren(inner, _) => find_mul_inner(inner, under_shr16),
        Node::As { expr, .. } => find_mul_inner(expr, under_shr16),
        Node::Binary { left, right, .. } => {
            find_mul_inner(left, under_shr16).or_else(|| find_mul_inner(right, under_shr16))
        }
        Node::Bitwise { left, right, .. } => {
            find_mul_inner(left, under_shr16).or_else(|| find_mul_inner(right, under_shr16))
        }
        _ => None,
    }
}

fn make_diag(ctx: &LintCtx<'_>, start: usize, end: usize) -> Diagnostic {
    Diagnostic {
        rule_id: "lint::q16_overflow".to_string(),
        severity: RuleSeverity::Warn,
        message: format!(
            "bare multiplication at byte {start} result assigned to i32 without >>16 narrowing"
        ),
        file: ctx.file.to_path_buf(),
        span: SourceSpan { start, end },
        help: Some(
            "Q16.16 multiplications must shift-right by 16 to narrow the \
             i64 product back to i32: `((a as i64 * b as i64) >> 16) as i32`"
                .to_string(),
        ),
        auto_fix: None,
    }
}
