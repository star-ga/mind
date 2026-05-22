// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0012 Phase C.1 — function-annotation checks.
//!
//! `[deterministic]`, `[target(...)]`, and `[q16]` are recorded inert on the
//! AST by Phase C.0 and enforced here. The checks are purely additive — only
//! functions that opt in are checked, so un-annotated code never regresses.
//!
//! MIND attribute surface syntax is `[attr]` (not `#[attr]`), matching the
//! existing `[test]` / `[protection]` / `[reap_threshold(..)]` attributes.

use libmind::pipeline::{compile_source_with_name, CompileError, CompileOptions};
use libmind::runtime::types::BackendTarget;

fn opts() -> CompileOptions {
    CompileOptions {
        func: None,
        enable_autodiff: false,
        target: BackendTarget::Cpu,
        ..Default::default()
    }
}

/// Assert the program fails type-check with a diagnostic carrying `code`.
fn expect_code(src: &str, code: &str) {
    let err = compile_source_with_name(src, Some("annot.mind"), &opts())
        .expect_err("program should fail annotation validation");
    match err {
        CompileError::TypeError(diags) => {
            let codes: Vec<&str> = diags.iter().map(|d| d.code).collect();
            assert!(
                codes.contains(&code),
                "expected diagnostic code {code}, saw {codes:?}"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

/// Assert the program compiles cleanly (no type errors).
fn expect_ok(src: &str) {
    compile_source_with_name(src, Some("annot.mind"), &opts())
        .expect("program should compile without annotation errors");
}

// ── [target(...)] name validity ──────────────────────────────────────

#[test]
fn valid_target_compiles() {
    expect_ok("[target(cpu)]\nfn f() -> i64 { 0 }\n");
}

#[test]
fn valid_target_q16_compiles() {
    expect_ok("[target(q16)]\nfn f() -> i64 { 0 }\n");
}

#[test]
fn unknown_target_reports_code() {
    // `cebras` is a typo for `cerebras`.
    expect_code(
        "[target(cebras)]\nfn f() -> i64 { 0 }\n",
        "determinism::unknown_target",
    );
}

#[test]
fn target_without_name_reports_code() {
    expect_code(
        "[target()]\nfn f() -> i64 { 0 }\n",
        "determinism::unknown_target",
    );
}

// ── [deterministic] call-graph ───────────────────────────────────────

#[test]
fn deterministic_calling_deterministic_compiles() {
    expect_ok(
        "[deterministic]\nfn helper() -> i64 { 1 }\n\
         [deterministic]\nfn f() -> i64 { helper() }\n",
    );
}

#[test]
fn deterministic_calling_plain_reports_code() {
    expect_code(
        "fn helper() -> i64 { 1 }\n\
         [deterministic]\nfn f() -> i64 { helper() }\n",
        "determinism::nondeterministic_in_deterministic",
    );
}

// ── [q16] dtype contract ─────────────────────────────────────────────

#[test]
fn q16_with_q16_tensor_compiles() {
    expect_ok("[q16]\nfn f(x: Tensor[q16,(4)]) -> i64 { 0 }\n");
}

#[test]
fn q16_with_f32_param_reports_code() {
    expect_code(
        "[q16]\nfn f(x: Tensor[f32,(4)]) -> i64 { 0 }\n",
        "determinism::float_in_q16_fn",
    );
}

#[test]
fn q16_implies_deterministic_call_check() {
    // `[q16]` ⇒ `[deterministic]`, so calling a plain fn is still flagged.
    expect_code(
        "fn helper() -> i64 { 1 }\n\
         [q16]\nfn f() -> i64 { helper() }\n",
        "determinism::nondeterministic_in_deterministic",
    );
}

// ── Additivity: un-annotated code is never touched ───────────────────

#[test]
fn unannotated_calls_compile() {
    // No annotations anywhere → the Phase C.1 pass must add zero diagnostics.
    expect_ok(
        "fn helper() -> i64 { 1 }\n\
         fn f() -> i64 { helper() }\n",
    );
}
