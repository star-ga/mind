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
//!   - a call to an UNANNOTATED-return callee in an `-> f64` body (class is not
//!     confidently known — must NOT false-positive E2010).
//!
//! The confidence-gated scalar-class checks (RFC 0011 — no implicit int↔float
//! coercion) extend this gate with the previously-missing directions:
//!   E2010 (new direction) — an integer value returned from a float-declared fn,
//!   E2013 — a `Node::Binary` mixing an int and a float operand (incl. `i64<f64`),
//!   E2014 — a declared-`u64` value in a sign-sensitive context,
//!   E2015 — a `let`/assign whose annotation class ≠ its value class,
//!   E2016 — a numeric `as bool` cast.
//! Each fires ONLY on annotation/literal-derived classes; the GREEN cases pin
//! the loose-typed constructs (fields, unannotated calls, loop vars) that MUST
//! stay invisible.
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
fn typed_int_call_into_float_fn_rejected() {
    // The confidence-gated class checker resolves a call to its callee's
    // DECLARED return annotation. `helper()` is declared `-> i64`, so returning
    // it from an `-> f64` fn is a genuine implicit int→float coercion (RFC 0011)
    // that fails LATE at `mlir-opt` ("return operand ('i64') doesn't match
    // result type ('f64')"). This is exactly the new E2010 direction — the
    // checker catches it EARLY. (Previously this program slipped through the
    // check phase because the E2010 rule only fired in the float-value
    // direction; the remediation closes the return-direction asymmetry.)
    let bad = write_tmp(
        "mind_ret_type_call_typed.mind",
        "pub fn ok() -> f64 {\n\
         \x20   return helper()\n\
         }\n\
         pub fn helper() -> i64 {\n\
         \x20   return 3\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "typed i64 call into f64 fn not rejected (late mlir-opt failure); out: {out}"
    );
}

#[test]
fn unannotated_call_into_float_fn_no_false_positive() {
    // Confidence gate: a call whose callee carries NO declared return annotation
    // resolves to class `None` — the E2010 new direction must NOT fire on it
    // (the checker only rejects on a *confident* class mismatch, never a guess).
    // This preserves the loose-ABI non-false-positive discipline the original
    // pin was written for.
    let good = write_tmp(
        "mind_ret_type_call_unann.mind",
        "pub fn ok() -> f64 {\n\
         \x20   return helper()\n\
         }\n\
         pub fn helper() {\n\
         \x20   return 3\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "unannotated-callee return falsely rejected E2010: {out}"
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

// `while` loops are only parsed by the formatter/front-end under `std-surface`;
// without it this source fails at parse before the type-checker can raise the
// float-condition E2011, so the test is gated to the feature that makes the
// `while` form reachable.
#[test]
#[cfg(feature = "std-surface")]
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

// ── Confidence-gated scalar-class checks (RFC 0011) ──────────────────────────
//   E2010 (new direction) — an integer value returned from a float-declared fn.
//   E2013 — a `Node::Binary` mixing a confident-Int and a confident-Float
//           operand (arithmetic AND comparison, incl. `i64 < f64`).
//   E2014 — a declared-`u64` value in a sign-sensitive context (`as f32/f64`,
//           `< <= > >= / % >>`) whose current lowering is (wrongly) signed.
//   E2015 — a `let`/assign whose scalar-annotation class ≠ the value class.
//   E2016 — a numeric `as bool` cast.
// Every check fires ONLY on annotation/literal-derived classes; the GREEN block
// pins the loose-typed constructs (fields, untyped calls, loop vars) that MUST
// stay invisible so the checker never over-rejects valid code.

#[test]
fn int_return_from_float_fn_rejected() {
    // E2010, the previously-missing direction: `fn g() -> f64 { return 5 }`.
    let bad = write_tmp(
        "mind_ret_int_into_float_bad.mind",
        "pub fn g() -> f64 {\n\
         \x20   return 5\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "int return from float fn not rejected (late mlir-opt failure); out: {out}"
    );
}

#[test]
fn let_float_ann_int_value_rejected() {
    let bad = write_tmp(
        "mind_let_float_int_bad.mind",
        "pub fn bad() -> i64 {\n\
         \x20   let x: f64 = 5\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2015"),
        "`let x: f64 = 5` not rejected; out: {out}"
    );
}

#[test]
fn let_int_ann_float_value_rejected() {
    let bad = write_tmp(
        "mind_let_int_float_bad.mind",
        "pub fn bad() -> i64 {\n\
         \x20   let y: i64 = 1.5\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2015"),
        "`let y: i64 = 1.5` not rejected; out: {out}"
    );
}

#[test]
fn mixed_int_float_arithmetic_rejected() {
    let bad = write_tmp(
        "mind_mixed_arith_bad.mind",
        "pub fn bad(a: i64, b: f64) -> i64 {\n\
         \x20   let z = a + b\n\
         \x20   return z\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2013"),
        "mixed `i64 + f64` not rejected; out: {out}"
    );
}

#[test]
fn mixed_int_float_comparison_rejected() {
    // `i64 < f64` — comparison direction of E2013 (closes the condition hole
    // without touching `cond_is_boolean_intent`).
    let bad = write_tmp(
        "mind_mixed_cmp_bad.mind",
        "pub fn bad(a: i64, b: f64) -> i64 {\n\
         \x20   if a < b { return 1 } else { return 0 }\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2013"),
        "mixed `i64 < f64` comparison not rejected; out: {out}"
    );
}

#[test]
fn cast_to_bool_rejected() {
    let bad = write_tmp(
        "mind_as_bool_bad.mind",
        "pub fn bad() -> i64 {\n\
         \x20   let x = 3 as bool\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2016"),
        "`3 as bool` not rejected; out: {out}"
    );
}

#[test]
fn u64_to_float_cast_rejected() {
    // `<u64> as f64` lowers to signed `sitofp` today — wrong for values ≥ 2^63.
    let bad = write_tmp(
        "mind_u64_as_float_bad.mind",
        "pub fn bad(x: u64) -> i64 {\n\
         \x20   let y = x as f64\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2014"),
        "`(u64) as f64` not rejected; out: {out}"
    );
}

#[test]
fn u64_comparison_now_unsigned() {
    // `u64 < u64` now lowers to UNSIGNED `cmpi ult` (issue #99 Stage 2: first-class
    // `ScalarU64` unsigned lowering), so it is deterministic and no longer rejected.
    // It must type-check clean — E2014 must NOT fire for the integer sign-sensitive
    // ops (compare/div/rem/shr) now that they have a deterministic unsigned lowering.
    let good = write_tmp(
        "mind_u64_cmp_ok.mind",
        "pub fn ok(a: u64, b: u64) -> i64 {\n\
         \x20   if a < b { return 1 } else { return 0 }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2014"),
        "`u64 < u64` now has an unsigned lowering and must NOT be rejected; out: {out}"
    );
}

// ── GREEN: loose-typed constructs the checker MUST leave invisible ───────────

#[test]
fn float_field_return_accepted() {
    // A struct-field read `p.x` (float field) returned from an `-> f64` fn:
    // field access resolves to class `None`, so E2010 must NOT fire.
    let good = write_tmp(
        "mind_field_return_ok.mind",
        "struct P { x: f64 }\n\
         pub fn ok(p: P) -> f64 {\n\
         \x20   return p.x\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "float field return falsely rejected E2010: {out}"
    );
}

#[test]
fn float_comparison_not_mixed_binop() {
    // `a > b` over two floats is same-class — must NOT trip E2013.
    let good = write_tmp(
        "mind_fcmp_not_mixed_ok.mind",
        "pub fn ok(a: f64, b: f64) -> i64 {\n\
         \x20   if a > b { return 1 } else { return 0 }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2013"),
        "float-vs-float comparison falsely rejected E2013: {out}"
    );
}

#[test]
#[cfg(feature = "std-surface")]
fn loop_counter_arithmetic_accepted() {
    // Unannotated loop counters/accumulators resolve to class `None`; neither
    // E2013 nor E2014 may fire on `s = s + i` / `i < n`.
    let good = write_tmp(
        "mind_loop_counter_ok.mind",
        "pub fn ok(n: i64) -> i64 {\n\
         \x20   let mut s = 0\n\
         \x20   let mut i = 0\n\
         \x20   while i < n {\n\
         \x20       s = s + i\n\
         \x20       i = i + 1\n\
         \x20   }\n\
         \x20   return s\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2013") && !out.contains("E2014"),
        "loop-counter arithmetic falsely rejected: {out}"
    );
}

#[test]
fn let_float_ann_undeclared_call_accepted() {
    // `let x: f64 = f()` where `f` is undeclared: the call resolves to class
    // `None` (no intra-module signature), so E2015 must NOT fire.
    let good = write_tmp(
        "mind_let_undeclared_call_ok.mind",
        "pub fn ok() -> f64 {\n\
         \x20   let x: f64 = f()\n\
         \x20   return x\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2015"),
        "`let x: f64 = f()` (f undeclared) falsely rejected E2015: {out}"
    );
}
