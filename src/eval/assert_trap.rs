// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The runtime trap call a failing `assert` lowers to.

use crate::ast::{Literal, Node, Span};

/// Build `__mind_assert_fail(msg, start, end)` for an `assert` at `span`.
///
/// The runtime intrinsic writes one diagnostic line to stderr and then
/// `abort()`s deterministically (the same `abort()` path the region/genref
/// runtime already relies on — see `runtime-support/mind_intrinsics.c`). It is
/// auto-declared as a `func.func private @__mind_assert_fail(i64, i64, i64) ->
/// i64` extern via the standard `extern_calls` collection, so no new IR
/// variant, MLIR arm, or wire-format change is introduced.
///
/// Arguments: the message as a String record (or 0 when the assert has no
/// message) and the assert's source span. The call sits only inside the
/// failing branch, so the message String is materialized only there and a
/// passing assert costs nothing extra. Every argument is a compile-time
/// constant or the literal's bytes: no pointer bits reach the output and the
/// diagnostic is identical across substrates.
pub(super) fn trap_call(msg: Option<&str>, span: Span) -> Node {
    let message = match msg {
        Some(text) => Node::Lit(Literal::Str(text.to_string()), span),
        None => Node::Lit(Literal::Int(0), span),
    };
    Node::Call {
        callee: "__mind_assert_fail".to_string(),
        args: vec![
            message,
            Node::Lit(Literal::Int(span.start() as i64), span),
            Node::Lit(Literal::Int(span.end() as i64), span),
        ],
        span,
    }
}
