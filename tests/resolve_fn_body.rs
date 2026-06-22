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
use std::collections::HashMap;

fn diagnostics(src: &str) -> Vec<String> {
    let module = parse(src).expect("parse");
    check_module_types_in_file(&module, src, Some("t.mind"), &HashMap::new())
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
fn undefined_arg_in_tensor_builtin_is_reported_e2002() {
    // Finding #2 (missed diagnostic): the resolver used to skip the value
    // arguments of EVERY `tensor.*` builtin, so an undefined identifier inside
    // one was silently accepted. Only the dtype-literal constructors may skip
    // their args; everything else must be walked.
    let errs = diagnostics(
        "fn f() -> i64 {\n    let x: Tensor[f32,(2,2)] = 1\n    let y = tensor.sum(x, undefined_var)\n    0\n}\n",
    );
    assert!(
        errs.iter()
            .any(|e| e.contains("E2002") && e.contains("undefined_var")),
        "an undefined ident inside a `tensor.*` call arg MUST flag E2002; got {errs:?}"
    );
}

#[test]
fn dtype_literal_builtin_args_are_not_false_flagged() {
    // The fix must NOT over-correct: `tensor.zeros(f32, (3, 4))` carries a bare
    // dtype literal `f32` (not a variable) plus a shape tuple — neither may
    // raise E2002.
    let errs = diagnostics(
        "fn g() -> i64 {\n    let z = tensor.zeros(f32, (3, 4))\n    0\n}\n",
    );
    assert!(
        !has_e2002(&errs),
        "a dtype-literal builtin's `f32`/shape args must not be flagged; got {errs:?}"
    );
    let errs2 = diagnostics(
        "fn g2() -> i64 {\n    let z = tensor.ones(f64, (2, 2))\n    0\n}\n",
    );
    assert!(
        !has_e2002(&errs2),
        "`tensor.ones(f64, ..)` dtype/shape args must not be flagged; got {errs2:?}"
    );
}
