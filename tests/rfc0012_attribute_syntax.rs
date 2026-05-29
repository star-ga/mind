// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0012 §5 — attribute surface syntax.
//!
//! MIND has exactly ONE attribute form: Rust-style `#[name]`. The `#` is
//! required — it disambiguates an attribute from the `@` matmul operator and
//! from a bare `[` array literal. Bare `[name]` is NOT an attribute. This
//! suite locks in:
//!   1. `#[name]` parses and the Phase C.1 checks fire,
//!   2. a bare `[name]` at item position is NOT treated as an attribute,
//!   3. a lone `#` is not an attribute,
//!   4. the formatter emits `#[name]` and round-trips function attributes
//!      (regression guard for the prior `emit_fn_def` attribute-drop bug).

use libmind::fmt::format_source;
use libmind::pipeline::{CompileError, CompileOptions, compile_source_with_name};
use libmind::project::MindcraftFormatConfig;
use libmind::runtime::types::BackendTarget;

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default()).expect("source should format")
}

fn opts() -> CompileOptions {
    CompileOptions {
        func: None,
        enable_autodiff: false,
        target: BackendTarget::Cpu,
        ..Default::default()
    }
}

/// All diagnostic codes from compiling `src` (empty if it compiled).
fn codes(src: &str) -> Vec<&'static str> {
    match compile_source_with_name(src, Some("a.mind"), &opts()) {
        Ok(_) => Vec::new(),
        Err(CompileError::TypeError(diags)) => diags.iter().map(|d| d.code).collect(),
        Err(_) => vec!["<non-type-error>"],
    }
}

// ── #[name] is the canonical (and only) attribute form ───────────────

#[test]
fn hash_attribute_parses_and_checks_fire() {
    // `#[deterministic]` calling a plain fn must trip the C.1 determinism check.
    assert!(
        codes("#[deterministic]\nfn f() -> i64 { g() }\nfn g() -> i64 { 0 }\n")
            .contains(&"determinism::nondeterministic_in_deterministic")
    );
}

#[test]
fn hash_target_unknown_reports_code() {
    assert!(
        codes("#[target(cebras)]\nfn f() -> i64 { 0 }\n").contains(&"determinism::unknown_target")
    );
}

#[test]
fn hash_q16_float_param_reports_code() {
    assert!(
        codes("#[q16]\nfn f(x: Tensor[f32,(4)]) -> i64 { 0 }\n")
            .contains(&"determinism::float_in_q16_fn")
    );
}

// ── bare [name] is NOT an attribute (hard-cut to one form) ───────────

#[test]
fn bare_bracket_is_not_an_attribute() {
    // With only `#[name]` recognized, a bare `[target(cebras)]` is never
    // parsed as an attribute, so the annotation checks must NOT fire on it.
    let c = codes("[target(cebras)]\nfn f() -> i64 { 0 }\n");
    assert!(
        !c.contains(&"determinism::unknown_target"),
        "bare [target] must not be treated as an attribute; saw {c:?}"
    );
}

#[test]
fn bare_q16_does_not_apply_q16_check() {
    let c = codes("[q16]\nfn f(x: Tensor[f32,(4)]) -> i64 { 0 }\n");
    assert!(
        !c.contains(&"determinism::float_in_q16_fn"),
        "bare [q16] must not be treated as an attribute; saw {c:?}"
    );
}

#[test]
fn lone_hash_is_not_an_attribute() {
    // A `#` not immediately followed by `[` must error, not silently parse.
    let r = compile_source_with_name("#\nfn f() -> i64 { 0 }\n", Some("a.mind"), &opts());
    assert!(r.is_err(), "a lone `#` must not parse");
}

// ── formatter emits canonical #[name] ────────────────────────────────

#[test]
fn formatter_emits_hash_attribute() {
    let out = fmt("#[target(cpu)]\nfn f() -> i64 {\n    0\n}\n");
    assert!(out.contains("#[target(cpu)]"), "got:\n{out}");
}

#[test]
fn formatter_does_not_erase_function_attributes() {
    // Regression guard: emit_fn_def previously dropped attributes entirely,
    // so `mindc fmt` silently erased `#[deterministic]` on functions.
    let out = fmt("#[deterministic]\nfn f() -> i64 {\n    0\n}\n");
    assert!(
        out.contains("#[deterministic]"),
        "fn attribute must survive fmt; got:\n{out}"
    );
}

#[test]
fn formatter_hash_attribute_is_idempotent() {
    let once = fmt("#[target(cerebras)]\nfn f() -> i64 {\n    0\n}\n");
    let twice = fmt(&once);
    assert_eq!(once, twice, "formatting #[..] must be idempotent");
    assert!(once.contains("#[target(cerebras)]"));
}

#[test]
fn formatter_emits_struct_attribute() {
    let out = fmt("#[repr(C)]\nstruct S {\n    x: i64,\n}\n");
    assert!(out.contains("#[repr(C)]"), "struct attr; got:\n{out}");
}
