// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! E2005 for a wrong-arity `__mind_*` intrinsic call, in EVERY expression
//! position.
//!
//! A call to a registered `__mind_*` intrinsic carries exactly the intrinsic's
//! declared arity — the i64 ABI of RFC 0005. The rule used to be enforced only
//! inside `check_std_surface_intrinsic`, which runs where `infer_expr` reaches
//! the call: `let` initializers, `return` values, `assert` conditions, the last
//! statement of an `if`/`for` body, and the arguments of a call whose checker
//! re-infers them. Several positions are unreachable there — `while` bodies and
//! conditions (RFC 0005 Gap 1: `infer_expr` types the loop without visiting
//! either), `else if` branches, the arguments of an intra-module or loose-ABI
//! call (deliberately never re-inferred — see `check_intra_fn_call`), and every
//! non-final statement of a body — so a wrong-arity call sat in one of them
//! until the IR verifier refused it at build time (E3001) and `mindc check`
//! passed it.
//!
//! This pass owns the rule outright. It walks every node the AST has — the
//! exhaustive `node_children_ref` classification plus the function, closure and
//! impl-method bodies that walk treats as scope leaves — and reports E2005 at
//! each intrinsic call site whose argument count differs from the declared
//! arity, wherever the call sits. `check_std_surface_intrinsic` still validates
//! the argument types on the `infer_expr` path; the arity refusal itself has
//! exactly one source.

use super::CALL_ARITY_CODE;
use super::diag_from_span;
use super::std_surface_intrinsic_arity;
use crate::ast::{Module, Node};
use crate::diagnostics::Diagnostic as Pretty;

/// Report E2005 for every wrong-arity registered intrinsic call reachable in
/// `module`, at any depth and in any expression position.
pub(super) fn check(module: &Module, src: &str, file: Option<&str>, errs: &mut Vec<Pretty>) {
    for item in &module.items {
        check_node(item, src, file, errs);
    }
}

fn check_node(node: &Node, src: &str, file: Option<&str>, errs: &mut Vec<Pretty>) {
    if let Node::Call { callee, args, span } = node
        && let Some(arity) = std_surface_intrinsic_arity(callee)
        && args.len() != arity
    {
        errs.push(diag_from_span(
            src,
            file,
            format!(
                "intrinsic `{callee}` expects {arity} i64 argument(s) (i64 ABI — RFC 0005); got {}",
                args.len()
            ),
            *span,
            CALL_ARITY_CODE,
        ));
    }
    // `node_children_ref`'s exhaustive classification stops at `FnDef` /
    // `Closure` / `ImplBlock` (they open a scope, so the closure rewriter must
    // not descend). This walk is scope-blind — a wrong-arity intrinsic call in
    // a body is still a call site — so descend into those bodies explicitly.
    // `TraitDef` carries signatures only (no body) and is a leaf for both.
    match node {
        Node::FnDef(fd, _) => {
            for stmt in &fd.body {
                check_node(stmt, src, file, errs);
            }
        }
        Node::Closure(data, _) => {
            for stmt in &data.body {
                check_node(stmt, src, file, errs);
            }
        }
        Node::ImplBlock { methods, .. } => {
            for method in methods {
                check_node(method, src, file, errs);
            }
        }
        _ => {}
    }
    for child in crate::eval::closures::node_children_ref(node) {
        check_node(child, src, file, errs);
    }
}
