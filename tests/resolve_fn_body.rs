// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Soundness gate for the single-file undefined-reference pass
//! (`src/type_checker/resolve.rs`). Locks the fix that closed the #23
//! fail-open `call_resolvable` stub: a genuinely-undefined call must be
//! reported (E2003), while every legitimate symbol source — params, lets,
//! module fns, the `__mind_*` / `tensor.*` namespaces, the bare math/tensor
//! builtins, and shape variables — must resolve cleanly.
//!
//! Run: `cargo test --features std-surface --test resolve_fn_body`

#![cfg(feature = "std-surface")]

use libmind::parser::parse;
use libmind::type_checker::check_module_types_in_file;

fn diagnostics(src: &str) -> Vec<String> {
    let module = parse(src).expect("parse");
    check_module_types_in_file(&module, src, Some("t.mind"), &Default::default())
        .iter()
        .map(|e| format!("{e:?}"))
        .collect()
}

fn has_e2003(errs: &[String]) -> bool {
    errs.iter().any(|e| e.contains("E2003"))
}
fn has_e2002(errs: &[String]) -> bool {
    errs.iter().any(|e| e.contains("E2002"))
}

#[test]
fn undefined_call_is_reported_e2003() {
    // The fail-open hole: this used to type-check rc=0 and build a binary.
    let errs = diagnostics("fn f() -> i64 {\n    let x = totally_undefined_fn(5);\n    x\n}\n");
    assert!(
        errs.iter()
            .any(|e| e.contains("E2003") && e.contains("totally_undefined_fn")),
        "an undefined call MUST flag E2003 (fail-open closed); got {errs:?}"
    );
}

#[test]
fn params_and_lets_resolve() {
    let errs = diagnostics("fn h(a: i64) -> i64 {\n    let b = a;\n    b\n}\n");
    assert!(
        !has_e2003(&errs) && !has_e2002(&errs),
        "params + lets must resolve; got {errs:?}"
    );
}

#[test]
fn bare_tensor_builtin_resolves() {
    // Bare-name tensor/autodiff builtins (matmul/relu/...) are first-class —
    // they must NOT be false-flagged as undefined.
    let errs = diagnostics(
        "fn g(a: tensor<f32[2, 2]>, b: tensor<f32[2, 2]>) -> tensor<f32[2, 2]> {\n    matmul(a, b)\n}\n",
    );
    assert!(
        !has_e2003(&errs),
        "a bare tensor builtin must resolve, not E2003; got {errs:?}"
    );
}

#[test]
fn shape_variable_resolves() {
    // A symbolic tensor dimension used as a value must resolve (no E2002).
    let errs = diagnostics(
        "fn r(x: tensor<f32[batch, 8]>) -> tensor<f32[batch, 4]> {\n    reshape(x, [batch, 4])\n}\n",
    );
    assert!(
        !has_e2002(&errs),
        "a shape variable used as a value must resolve; got {errs:?}"
    );
}

#[test]
fn generic_type_param_resolves_in_body() {
    // A generic fn's own type parameter referenced as a BARE value in the body
    // must NOT false-positive E2002 — the type params (`<T>`) were not bound in
    // the body scope before this fix, so `let y = T` reported `unknown
    // identifier T`. (A qualified `T::default()` already resolved via the `::`
    // short-circuit; the bare reference is the case the fix actually closes.)
    let errs = diagnostics("fn id<T>(x: T) -> T {\n    let y = T\n    x\n}\n");
    assert!(
        !has_e2002(&errs),
        "a generic fn's type param must resolve in its body (no E2002); got {errs:?}"
    );
}

#[test]
fn named_alias_shape_param_resolves() {
    // A bare named type used as a shape parameter (`x: N`) whose name `N` is
    // also read in the body must resolve: `collect_shape_vars` now recurses
    // through `TypeAnn::Named`, pre-binding `N`. Before the fix the body read
    // of `N` false-positived E2002 (the param's `Named` annotation was a no-op).
    let errs = diagnostics("fn f(x: N) -> i64 {\n    let c = N\n    c\n}\n");
    assert!(
        !has_e2002(&errs),
        "a named-alias shape param must resolve when read in the body; got {errs:?}"
    );
}
