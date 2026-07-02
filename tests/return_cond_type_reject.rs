// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Type-checker fail-closed-early regression gate for E2010 / E2011.
//!
//! Two constructs that used to pass `mindc check` and only fail LATE at
//! `mlir-opt` with an opaque error now reject EARLY at the check phase:
//!   E2010 — a `return <float>` from an integer-declared function
//!           (`fn f() -> i64 { return 1.5 }`) previously passed check, then
//!           `mlir-opt` reported "type of return operand 0 ('f64') doesn't
//!           match function result type ('i64')".
//!   E2011 — a FLOAT-class `if`/`while` condition (`if 1.5 { .. }`) previously
//!           passed check, then `mlir-opt` emitted `arith.cmpi` on an `f64`.
//! Both now fail `mindc check`. This gate asserts the bad shapes are REJECTED
//! and that the sound sibling programs still pass with NO false positives:
//!   - an `i32` return from an `-> i64` fn (same scalar class, different width),
//!   - a correct float return from an `-> f64` fn,
//!   - normal integer `if x > 0` / `while i < n`,
//!   - a FLOAT comparison `if a > b` (bool-intent, must NOT be rejected),
//!   - an untyped same-file call in an `-> f64` body (loose i64 ABI default —
//!     must NOT false-positive E2010).
//!
//! We assert on the presence/absence of the diagnostic CODE in stdout+stderr
//! rather than the process exit status: `mindc check` also emits a `fmt::drift`
//! diagnostic for an unformatted file, so the exit code muddies a pure
//! pass/fail read. Whether E2010/E2011 fires is the behavior under test.

use std::io::Write;
use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn write_tmp(name: &str, src: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&p).expect("create tmp");
    f.write_all(src.as_bytes()).expect("write tmp");
    p
}

/// Combined stdout+stderr of `mindc check <path>` (the default `human` reporter
/// prints diagnostics to stdout).
fn check_out(path: &std::path::Path) -> String {
    let out = mindc()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("spawn mindc");
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn float_return_from_int_fn_rejected() {
    let bad = write_tmp(
        "mind_ret_type_bad.mind",
        "pub fn bad() -> i64 {\n\
         \x20   return 1.5\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "float return from int fn not rejected (late mlir-opt failure); out: {out}"
    );
}

#[test]
fn width_sibling_return_accepted() {
    // i32 into an -> i64 fn: same scalar CLASS, different width — must pass.
    let good = write_tmp(
        "mind_ret_type_width_ok.mind",
        "pub fn ok(x: i32) -> i64 {\n\
         \x20   return x\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "i32-into-i64 sibling return falsely rejected: {out}"
    );
}

#[test]
fn correct_float_return_accepted() {
    let good = write_tmp(
        "mind_ret_type_float_ok.mind",
        "pub fn ok() -> f64 {\n\
         \x20   return 1.5\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "valid float return falsely rejected: {out}"
    );
}

#[test]
fn untyped_call_into_float_fn_accepted() {
    // The loose i64 ABI defaults an untyped same-file call to ScalarI64; the
    // E2010 rule must be asymmetric (only a confidently-FLOAT value fires) so
    // this valid `-> f64` body is NOT rejected.
    let good = write_tmp(
        "mind_ret_type_call_ok.mind",
        "pub fn ok() -> f64 {\n\
         \x20   return helper()\n\
         }\n\
         pub fn helper() -> i64 {\n\
         \x20   return 3\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "untyped call into f64 fn falsely rejected E2010: {out}"
    );
}

#[test]
fn float_if_condition_rejected() {
    let bad = write_tmp(
        "mind_cond_if_bad.mind",
        "pub fn bad() -> i64 {\n\
         \x20   if 1.5 { return 1 } else { return 0 }\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2011"),
        "float `if` condition not rejected (late mlir-opt failure); out: {out}"
    );
}

#[test]
fn float_while_condition_rejected() {
    let bad = write_tmp(
        "mind_cond_while_bad.mind",
        "pub fn bad() -> i64 {\n\
         \x20   let mut i = 0\n\
         \x20   while 1.5 { i = i + 1 }\n\
         \x20   return i\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2011"),
        "float `while` condition not rejected (late mlir-opt failure); out: {out}"
    );
}

#[test]
fn int_conditions_accepted() {
    let good = write_tmp(
        "mind_cond_int_ok.mind",
        "pub fn ok(n: i64) -> i64 {\n\
         \x20   let mut i = 0\n\
         \x20   while i < n { i = i + 1 }\n\
         \x20   if i > 0 { return 1 } else { return 0 }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2011"),
        "valid integer conditions falsely rejected: {out}"
    );
}

#[test]
fn float_comparison_condition_accepted() {
    // `a > b` over floats is a boolean-intent comparison, not a raw float
    // condition — `infer_expr` mistypes it as ScalarF64, so it MUST be excluded.
    let good = write_tmp(
        "mind_cond_fcmp_ok.mind",
        "pub fn ok(a: f64, b: f64) -> i64 {\n\
         \x20   if a > b { return 1 } else { return 0 }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2011"),
        "float comparison condition falsely rejected E2011: {out}"
    );
}
