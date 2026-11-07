// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! E2005 for a wrong-arity `__mind_*` intrinsic call, in EVERY expression
//! position.
//!
//! A call to a registered `__mind_*` intrinsic carries exactly the intrinsic's
//! declared arity — the i64 ABI of RFC 0005. The rule used to be enforced only
//! inside `check_std_surface_intrinsic`, which runs only where `infer_expr`
//! reaches the call, and `infer_expr` skips whole positions: it types an `if`
//! by the last statement of its then-branch and a `for` by the last statement
//! of its body, types a `while` without visiting its condition or body (RFC
//! 0005 Gap 1), and never re-infers the arguments of an intra-module or
//! loose-ABI call (see `check_intra_fn_call`). A wrong-arity call in a skipped
//! position passed `mindc check` and was refused only at build time, by the IR
//! verifier (E3001).
//!
//! This pass owns the rule outright. It walks every node the AST has — the
//! exhaustive `node_children_ref` classification plus the function, closure and
//! impl-method bodies that walk treats as scope leaves — and reports E2005 at
//! each intrinsic call site whose argument count differs from the declared
//! arity, wherever the call sits. `check_std_surface_intrinsic` still validates
//! the argument types on the `infer_expr` path; the arity refusal itself has
//! exactly one source. `mindc check` desugars closures and impl methods into
//! functions before this pass runs; the closure and impl arms below serve the
//! callers that type-check a module as parsed (`build::native_scope`).

use super::AstSpan;
use super::CALL_ARITY_CODE;
use super::TypeEnv;
use super::TypeErrSpan;
use super::ValueType;
use super::describe_value_type;
use super::diag_from_span;
use super::infer_expr;
use crate::ast::{Module, Node};
use crate::diagnostics::Diagnostic as Pretty;

// RFC 0005 Phase 1 + 1.5 — pure-MIND standard surface intrinsics.
//
// The table itself moved to `crate::intrinsics` so that ONE row carries a
// primitive's name, its i64 arity, AND whether calling it can make the program's
// result vary between runs. It used to be split: the arity here, the determinism
// verdict in two hand-maintained parallel arrays (this module's and
// `ir::evidence`'s) that drifted identically and let the registered
// `__mind_nerve_rt_*` clock / entropy / stdin surface be attested
// `deterministic`. See the `crate::intrinsics` module docs.
pub(super) fn std_surface_intrinsic_arity(name: &str) -> Option<usize> {
    crate::intrinsics::std_surface_intrinsic_arity(name)
}

pub(super) fn check_std_surface_intrinsic(
    callee: &str,
    args: &[Node],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    let arity = std_surface_intrinsic_arity(callee).ok_or_else(|| TypeErrSpan {
        msg: format!("unsupported call to `{callee}`"),
        span,
    })?;
    if args.len() != arity {
        // The arity rule is `intrinsic_call_arity`'s (E2005 at EVERY expression
        // position); this function only runs where `infer_expr` reaches the
        // call. Type the call as its i64 result and let that pass refuse it —
        // one source for one rule.
        return Ok((ValueType::ScalarI64, span));
    }
    for (i, a) in args.iter().enumerate() {
        let (ty, _) = infer_expr(a, env)?;
        // Integer literals come in as `ScalarI32` (line 562) but lower
        // to `ConstI64` in IR — accept both; reject anything else with
        // a clear i64-ABI message that points to RFC 0005 phase 2+.
        if !matches!(ty, ValueType::ScalarI32 | ValueType::ScalarI64) {
            return Err(TypeErrSpan {
                msg: format!(
                    "argument {i} to intrinsic `{callee}` must be i64 (got {}); the std-surface intrinsics use the i64 ABI — aggregate types are RFC 0005 phase 2+",
                    describe_value_type(&ty)
                ),
                span: a.span(),
            });
        }
    }
    Ok((ValueType::ScalarI64, span))
}

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
