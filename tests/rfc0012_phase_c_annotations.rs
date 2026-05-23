// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0012 Phase C.1 — function-annotation checks.
//!
//! `#[deterministic]`, `#[target(...)]`, and `#[q16]` are recorded inert on the
//! AST by Phase C.0 and enforced here. The checks are purely additive — only
//! functions that opt in are checked, so un-annotated code never regresses.
//!
//! MIND attribute surface syntax is Rust-style `#[attr]` (RFC 0012 §5; the
//! `#` disambiguates attributes from the `@` operator and bare `[` arrays).

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

// ── #[target(...)] name validity ──────────────────────────────────────

#[test]
fn valid_target_compiles() {
    expect_ok("#[target(cpu)]\nfn f() -> i64 { 0 }\n");
}

#[test]
fn valid_target_q16_compiles() {
    expect_ok("#[target(q16)]\nfn f() -> i64 { 0 }\n");
}

#[test]
fn unknown_target_reports_code() {
    // `cebras` is a typo for `cerebras`.
    expect_code(
        "#[target(cebras)]\nfn f() -> i64 { 0 }\n",
        "determinism::unknown_target",
    );
}

#[test]
fn target_without_name_reports_code() {
    expect_code(
        "#[target()]\nfn f() -> i64 { 0 }\n",
        "determinism::unknown_target",
    );
}

// ── #[deterministic] call-graph ───────────────────────────────────────

#[test]
fn deterministic_calling_deterministic_compiles() {
    expect_ok(
        "#[deterministic]\nfn helper() -> i64 { 1 }\n\
         #[deterministic]\nfn f() -> i64 { helper() }\n",
    );
}

#[test]
fn deterministic_calling_plain_reports_code() {
    expect_code(
        "fn helper() -> i64 { 1 }\n\
         #[deterministic]\nfn f() -> i64 { helper() }\n",
        "determinism::nondeterministic_in_deterministic",
    );
}

// ── #[q16] dtype contract ─────────────────────────────────────────────

#[test]
fn q16_with_q16_tensor_compiles() {
    expect_ok("#[q16]\nfn f(x: Tensor[q16,(4)]) -> i64 { 0 }\n");
}

#[test]
fn q16_with_f32_param_reports_code() {
    expect_code(
        "#[q16]\nfn f(x: Tensor[f32,(4)]) -> i64 { 0 }\n",
        "determinism::float_in_q16_fn",
    );
}

#[test]
fn q16_implies_deterministic_call_check() {
    // `#[q16]` ⇒ `#[deterministic]`, so calling a plain fn is still flagged.
    expect_code(
        "fn helper() -> i64 { 1 }\n\
         #[q16]\nfn f() -> i64 { helper() }\n",
        "determinism::nondeterministic_in_deterministic",
    );
}

// ── Phase C.2: implicit determinism of external (std/intrinsic) calls ─

/// Diagnostic codes from compiling `src` (empty on success). Tolerates other
/// diagnostics so we can assert presence/absence of a specific code on a
/// program that calls undefined external symbols.
fn codes(src: &str) -> Vec<&'static str> {
    match compile_source_with_name(src, Some("annot.mind"), &opts()) {
        Ok(_) => Vec::new(),
        Err(CompileError::TypeError(diags)) => diags.iter().map(|d| d.code).collect(),
        Err(_) => vec!["<non-type-error>"],
    }
}

#[test]
fn deterministic_calling_q16_blas_is_ok() {
    // `dot_q16` is a Q16.16 byte-identical std.blas fn → implicitly
    // deterministic, so a `#[deterministic]` caller must NOT be flagged.
    let c = codes("#[deterministic]\nfn f() -> i64 { dot_q16(0, 0, 0) }\n");
    assert!(
        !c.contains(&"determinism::nondeterministic_in_deterministic"),
        "q16 std.blas call must be implicitly deterministic; saw {c:?}"
    );
}

#[test]
fn deterministic_calling_f32_blas_reports_code() {
    // `dot_f32` reorders its reduction under SIMD → not deterministic.
    let c = codes("#[deterministic]\nfn f() -> i64 { dot_f32(0, 0, 0) }\n");
    assert!(
        c.contains(&"determinism::nondeterministic_in_deterministic"),
        "f32 std.blas call must be flagged in a #[deterministic] fn; saw {c:?}"
    );
}

#[test]
fn deterministic_calling_mind_intrinsic_is_ok() {
    // Integer intrinsics (`__mind_*`) are deterministic by construction.
    let c = codes("#[deterministic]\nfn f() -> i64 { __mind_load_i64(0) }\n");
    assert!(
        !c.contains(&"determinism::nondeterministic_in_deterministic"),
        "integer intrinsic must be implicitly deterministic; saw {c:?}"
    );
}

#[test]
fn deterministic_calling_unknown_external_is_not_flagged() {
    // An external callee with no dtype-suffix signal is unknown → conservative
    // (NOT flagged), so no false positive.
    let c = codes("#[deterministic]\nfn f() -> i64 { mystery_helper(0) }\n");
    assert!(
        !c.contains(&"determinism::nondeterministic_in_deterministic"),
        "unknown external call must not be flagged (no false positive); saw {c:?}"
    );
}

// ── Phase C.2: checks reach functions inside module { } blocks ───────

#[test]
fn deterministic_check_descends_into_module_block() {
    // A `#[deterministic]` fn nested in a `module { }` block calling a plain
    // fn (also nested) must be flagged — the pass descends into module blocks.
    let c = codes(
        "module m {\n#[deterministic]\nfn f() -> i64 { g() }\nfn g() -> i64 { 0 }\n}\n",
    );
    assert!(
        c.contains(&"determinism::nondeterministic_in_deterministic"),
        "determinism check must reach fns inside module blocks; saw {c:?}"
    );
}

#[test]
fn module_block_deterministic_calling_deterministic_is_ok() {
    let c = codes(
        "module m {\n#[deterministic]\nfn helper() -> i64 { 1 }\n\
         #[deterministic]\nfn f() -> i64 { helper() }\n}\n",
    );
    assert!(
        !c.contains(&"determinism::nondeterministic_in_deterministic"),
        "both annotated in a module block → no flag; saw {c:?}"
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
