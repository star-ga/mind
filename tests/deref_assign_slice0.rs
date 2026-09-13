// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Slice 0 (deref-assign track): `*p` and `*p = v` PARSE, FORMAT (round-trip),
//! and are carried in the AST — but are REFUSED with the EXACT diagnostics
//! `E2028` (dereference) / `E2029` (deref-assign), with NO executable support:
//! the reference/place ABI is unimplemented and under architecture review.
//! Record-identity place replacement (mind-spec v1.0/types.md:65-99) is NOT a
//! field copy; this slice defines no executable meaning.
//!
//! Generic types only — no private consumer (MindLLM/PageTable) source here.

mod common;
use common::mindc_bin;

use libmind::fmt::format_source;
use libmind::parser;
use libmind::project::MindcraftFormatConfig;
use libmind::type_checker::{self, TypeEnv};

use std::process::Command;

/// The exact diagnostic codes emitted for a checked module (empty when clean).
fn codes(src: &str) -> Vec<String> {
    let module = parser::parse(src).expect("Slice 0 source must PARSE");
    let env = TypeEnv::default();
    type_checker::check_module_types(&module, src, &env)
        .iter()
        .map(|d| d.code.to_string())
        .collect()
}

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default()).expect("Slice 0 source must FORMAT")
}

#[test]
fn deref_read_round_trips_and_is_refused_with_E2028() {
    let src = "fn f(p: &mut i64) -> i64 {\n    return *p\n}\n";
    let once = fmt(src);
    assert!(
        once.contains("*p"),
        "deref must round-trip through fmt: {once}"
    );
    assert_eq!(once, fmt(&once), "fmt must be idempotent on deref");
    let cs = codes(src);
    assert!(
        cs.contains(&"E2028".to_string()),
        "expected E2028; got {cs:?}"
    );
}

#[test]
fn deref_assign_round_trips_and_is_refused_with_E2029() {
    let src = "fn f(p: &mut i64, v: i64) {\n    *p = v\n}\n";
    let once = fmt(src);
    assert!(
        once.contains("*p = v"),
        "deref-assign must round-trip: {once}"
    );
    assert_eq!(once, fmt(&once), "fmt must be idempotent on deref-assign");
    let cs = codes(src);
    assert!(
        cs.contains(&"E2029".to_string()),
        "expected E2029; got {cs:?}"
    );
}

#[test]
fn record_deref_assign_is_refused_not_cloned_E2029() {
    // Real motivation is records (identity-preserving place replacement), NOT a
    // field copy. Slice 0 refuses with E2029 rather than defining a clone.
    let src = "struct Cell {\n    a: i64,\n    b: i64\n}\nfn r(p: &mut Cell, next: Cell) {\n    *p = next\n}\n";
    assert!(
        fmt(src).contains("*p = next"),
        "record deref-assign must round-trip"
    );
    assert!(
        codes(src).contains(&"E2029".to_string()),
        "record `*p = next` must be E2029"
    );
}

#[test]
fn valid_multiply_produces_no_diagnostics() {
    // Prefix `*` (deref) must not disturb the infix `*` (multiply): a plain
    // multiply type-checks with ZERO diagnostics — no E2028/E2029, no others.
    let src = "fn f(a: i64, b: i64) -> i64 {\n    return a * b\n}\n";
    assert!(fmt(src).contains("a * b"), "infix multiply must round-trip");
    let cs = codes(src);
    assert!(
        cs.is_empty(),
        "a valid multiply must produce NO diagnostics; got {cs:?}"
    );
}

#[test]
fn deref_nested_in_expressions_is_refused() {
    // `*p` buried inside a call argument and an arithmetic expression is still
    // reached by the whole-module refusal scan.
    let src = "fn g(x: i64) -> i64 {\n    return x\n}\nfn f(p: &mut i64) -> i64 {\n    return g(*p) + 1\n}\n";
    assert!(
        codes(src).contains(&"E2028".to_string()),
        "nested `*p` must be E2028"
    );
}

#[test]
fn deref_assign_inside_control_flow_is_refused() {
    // `*p = v` inside an `if` / loop body is reached by the scan.
    let src = "fn f(p: &mut i64, v: i64, c: i64) {\n    if c == 1 {\n        *p = v\n    }\n}\n";
    assert!(
        codes(src).contains(&"E2029".to_string()),
        "`*p = v` in control flow must be E2029"
    );
}

/// No apparent executable support: through the SAME `--emit-shared` entrypoint,
/// a valid control MUST emit an artifact while a deref-assign program MUST fail
/// with the intended refusal (E2029) and write NO artifact. The valid control
/// is what stops a broken backend/formatter/linker from making the negative
/// case look green. Gated on `mlir-build` (native emission); without it this
/// test is NOT EXECUTED — never counted as proof. Unique per-run temp dir so
/// concurrent runs cannot delete each other's artifacts.
#[cfg(feature = "mlir-build")]
#[test]
fn deref_assign_emits_no_artifact_while_valid_control_does() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped("deref_assign_slice0", "mindc not found; not executed");
        return;
    }
    let dir = tempfile::TempDir::new().expect("unique temp dir");
    let emit = |name: &str, body: &str| -> (bool, bool, String) {
        let src = dir.path().join(format!("{name}.mind"));
        let so = dir.path().join(format!("{name}.so"));
        std::fs::write(&src, body).expect("write src");
        let out = Command::new(&mindc)
            .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
            .output()
            .expect("run mindc");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.success(), so.exists(), text)
    };

    // (a) Valid control through the SAME entrypoint. If native emission is
    //     unavailable here, treat as NOT EXECUTED rather than proof.
    let (ok, so, ctrl_text) = emit(
        "valid_control",
        "fn f(a: i64, v: i64) -> i64 {\n    return a + v;\n}\n",
    );
    if !ok {
        crate::common::gate::skipped(
            "deref_assign_slice0",
            "native --emit-shared backend unavailable for the valid control; not executed",
        );
        return;
    }
    assert!(
        so,
        "valid control must emit a `.so` through --emit-shared:\n{ctrl_text}"
    );

    // (b) Deref-assign through the SAME entrypoint: must FAIL with the intended
    //     E2029 refusal and write NO artifact.
    let (d_ok, d_so, d_text) = emit(
        "deref_assign",
        "fn f(p: &mut i64, v: i64) {\n    *p = v;\n}\n",
    );
    assert!(
        !d_ok,
        "deref-assign must NOT emit an artifact; got success:\n{d_text}"
    );
    assert!(
        d_text.contains("E2029"),
        "the failure must be the deref-assign refusal E2029, not an unrelated CLI/formatter/linker error:\n{d_text}"
    );
    assert!(
        !d_so,
        "no `.so` may be written for a refused deref-assign:\n{d_text}"
    );
}
