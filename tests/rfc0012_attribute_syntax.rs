// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0012 — attribute surface syntax resolution.
//!
//! MIND attributes may be written Rust-style `#[name]` (CANONICAL) or bare
//! `[name]` (legacy, still accepted). The parser accepts both; `mindc fmt`
//! normalizes to the canonical `#[name]` form. This suite locks in:
//!   1. the parser accepts both forms,
//!   2. the formatter normalizes bare `[name]` → `#[name]`,
//!   3. function attributes are NOT erased by the formatter (regression guard
//!      for the prior `emit_fn_def` attribute-drop bug),
//!   4. `#[name]` output is idempotent.

use libmind::fmt::format_source;
use libmind::pipeline::{compile_source_with_name, CompileError, CompileOptions};
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

// ── parser accepts both forms ────────────────────────────────────────

#[test]
fn parser_accepts_hash_style_attribute() {
    // Rust-style `#[deterministic]` must parse and the C.1 determinism check
    // must fire exactly as it does for the bare form.
    let err = compile_source_with_name(
        "#[deterministic]\nfn f() -> i64 { g() }\nfn g() -> i64 { 0 }\n",
        Some("a.mind"),
        &opts(),
    )
    .expect_err("deterministic fn calling a plain fn should fail");
    match err {
        CompileError::TypeError(diags) => {
            let codes: Vec<&str> = diags.iter().map(|d| d.code).collect();
            assert!(
                codes.contains(&"determinism::nondeterministic_in_deterministic"),
                "saw {codes:?}"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn parser_accepts_bare_style_attribute() {
    // The legacy bare `[target(cpu)]` form still parses and validates.
    compile_source_with_name("[target(cpu)]\nfn f() -> i64 { 0 }\n", Some("a.mind"), &opts())
        .expect("bare [target(cpu)] should still compile");
}

#[test]
fn lone_hash_is_not_an_attribute() {
    // A `#` not immediately followed by `[` must NOT be consumed as an
    // attribute sigil (it would otherwise corrupt the following item).
    // `#` is not a valid item/expression start, so this must error in parse —
    // NOT silently swallow the `#` and accept the fn.
    let r = compile_source_with_name("#\nfn f() -> i64 { 0 }\n", Some("a.mind"), &opts());
    assert!(r.is_err(), "a lone `#` token must not parse as an attribute");
}

// ── formatter normalizes to canonical #[name] ────────────────────────

#[test]
fn formatter_normalizes_bare_to_hash() {
    let out = fmt("[target(cpu)]\nfn f() -> i64 {\n    0\n}\n");
    assert!(
        out.contains("#[target(cpu)]"),
        "bare [target(cpu)] should normalize to #[target(cpu)]; got:\n{out}"
    );
    assert!(
        !out.contains("\n[target") && !out.starts_with("[target"),
        "no bare [target should remain; got:\n{out}"
    );
}

#[test]
fn formatter_does_not_erase_function_attributes() {
    // Regression guard: emit_fn_def previously dropped attributes entirely,
    // so `mindc fmt` silently erased `[deterministic]` on functions.
    let out = fmt("[deterministic]\nfn f() -> i64 {\n    0\n}\n");
    assert!(
        out.contains("#[deterministic]"),
        "function attribute must survive formatting; got:\n{out}"
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
fn formatter_normalizes_struct_attribute() {
    // The shared emit_attrs helper must canonicalize struct/enum/const/alias
    // attributes too, not just function ones.
    let out = fmt("[repr(C)]\nstruct S {\n    x: i64,\n}\n");
    assert!(out.contains("#[repr(C)]"), "struct attr should normalize; got:\n{out}");
}
