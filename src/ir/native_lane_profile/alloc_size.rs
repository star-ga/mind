// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Positivity proof for `__mind_alloc` / `__mind_realloc` sizes in the user
//! image (`lane.alloc.size_positive`).
//!
//! The marked compiler bumps its arena by any size it is given, so a size of
//! zero returns a live pointer and a negative size moves the cursor back,
//! where the C runtime and the evaluator return 0 for both. A size is admitted
//! only when it is proven to be at least 1:
//!
//! * a positive integer literal (optionally parenthesised), or
//! * a name of type i64 that the function never assigns, after a statement
//!   `if <name> <= 0 { ... return/break/continue; ... }` with no `else`. The
//!   tests `x <= 0`, `x < 1`, `0 >= x` and `1 > x` are recognised. Control
//!   reaches the next statement only when the test was false, so `x >= 1`
//!   holds there and in every nested block, until a `let`/`for` binding of
//!   the same name hides it.
//!
//! No arithmetic is proven (`n * 8` can wrap), so every other size refuses.
//! The native bridge routes such a size through a guard
//! (`src/build/native_bridge/alloc_guard.rs`): `__mind_alloc` returns 0 for
//! `n <= 0`, and `__mind_realloc` does too except `realloc(NULL, 0)`, which
//! returns a 24-byte block. The guard's own call is this proof.

use super::types::Ty;
use crate::ast::{BinOp, Literal, Node};
use std::collections::{BTreeMap, BTreeSet};

/// Env key of a positivity fact. `+` cannot begin an identifier, so the key
/// never collides with a binding (`arrays` uses `[]` the same way).
fn key(name: &str) -> String {
    format!("+{name}")
}

/// The size operand of an allocation intrinsic call, if `callee` is one.
pub(super) fn size_operand<'a>(callee: &str, args: &'a [Node]) -> Option<&'a Node> {
    match callee {
        "__mind_alloc" => args.first(),
        "__mind_realloc" => args.get(1),
        _ => None,
    }
}

/// A positive integer literal, optionally parenthesised.
pub(crate) fn positive_literal(node: &Node) -> bool {
    match node {
        Node::Lit(Literal::Int(n), _) => *n > 0,
        Node::Paren(inner, _) => positive_literal(inner),
        _ => false,
    }
}

/// `size` is proven to be at least 1 under `env`.
pub(super) fn proven(size: &Node, env: &BTreeMap<String, Ty>) -> bool {
    match size {
        Node::Paren(inner, _) => proven(inner, env),
        Node::Lit(Literal::Ident(name), _) => {
            env.get(name) == Some(&Ty::I64) && env.contains_key(&key(name))
        }
        _ => positive_literal(size),
    }
}

/// A binding of `name` hides any fact about an outer binding of that name.
pub(super) fn forget(name: &str, env: &mut BTreeMap<String, Ty>) {
    env.remove(&key(name));
}

/// The name a test `cond` proves non-positive when true.
fn non_positive_test(cond: &Node) -> Option<&str> {
    let Node::Binary {
        op, left, right, ..
    } = cond
    else {
        return match cond {
            Node::Paren(inner, _) => non_positive_test(inner),
            _ => None,
        };
    };
    fn ident(n: &Node) -> Option<&str> {
        match n {
            Node::Lit(Literal::Ident(name), _) => Some(name.as_str()),
            _ => None,
        }
    }
    fn int(n: &Node) -> Option<i64> {
        match n {
            Node::Lit(Literal::Int(v), _) => Some(*v),
            _ => None,
        }
    }
    match (op, int(left), int(right)) {
        (BinOp::Le, None, Some(0)) | (BinOp::Lt, None, Some(1)) => ident(left),
        (BinOp::Ge, Some(0), None) | (BinOp::Gt, Some(1), None) => ident(right),
        _ => None,
    }
}

/// After the statement `stmt` has been admitted, record the fact it proves.
pub(super) fn learn(stmt: &Node, env: &mut BTreeMap<String, Ty>, assigned: &BTreeSet<String>) {
    let Node::If {
        cond,
        then_branch,
        else_branch: None,
        ..
    } = stmt
    else {
        return;
    };
    let exits = then_branch.iter().any(|s| {
        matches!(
            s,
            Node::Return { .. } | Node::Break { .. } | Node::Continue { .. }
        )
    });
    if let Some(name) = non_positive_test(cond).filter(|_| exits) {
        if env.get(name) == Some(&Ty::I64) && !assigned.contains(name) {
            env.insert(key(name), Ty::I64);
        }
    }
}

/// Every name assigned anywhere in `nodes`. The walk covers every construct
/// the source walker admits; any other construct refuses the function, so an
/// assignment inside it never reaches a proof.
pub(super) fn assigned_names(nodes: &[Node]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    all(nodes, &mut out);
    out
}

fn all(nodes: &[Node], out: &mut BTreeSet<String>) {
    for node in nodes {
        collect(node, out);
    }
}

fn collect(node: &Node, out: &mut BTreeSet<String>) {
    match node {
        Node::Assign { name, value, .. } => {
            collect(value, out);
            out.insert(name.clone());
        }
        Node::Paren(a, _)
        | Node::Neg { operand: a, .. }
        | Node::Not { operand: a, .. }
        | Node::BitNot { operand: a, .. }
        | Node::Let { value: a, .. }
        | Node::As { expr: a, .. }
        | Node::Return { value: Some(a), .. } => collect(a, out),
        Node::Binary { left, right, .. }
        | Node::Bitwise { left, right, .. }
        | Node::Logical { left, right, .. }
        | Node::IndexAccess {
            receiver: left,
            index: right,
            ..
        } => {
            collect(left, out);
            collect(right, out);
        }
        Node::Call { args, .. } => all(args, out),
        Node::Block { stmts, .. } => all(stmts, out),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect(cond, out);
            all(then_branch, out);
            if let Some(b) = else_branch {
                all(b, out);
            }
        }
        Node::While { cond, body, .. } => {
            collect(cond, out);
            all(body, out);
        }
        Node::For {
            start, end, body, ..
        } => {
            collect(start, out);
            collect(end, out);
            all(body, out);
        }
        _ => {}
    }
}
