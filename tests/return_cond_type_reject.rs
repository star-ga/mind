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

/// Combined stdout+stderr of a full `mindc <path> --emit-shared <tmp.so>` COMPILE.
/// Unlike `mindc check`, the compile pipeline runs `desugar_traits` (impls lifted
/// to `{type}_{method}` free fns) BEFORE type-checking — the only path on which
/// the RFC 0011 method-call class check fires (`mindc check` rejects any raw
/// trait/impl at E2001 before type inference). A scalar-class error (E2027) is
/// raised during the type-check phase and returned BEFORE lowering/`mlir-opt`, so
/// the reject shows up even in a build without a working MLIR toolchain; a clean
/// program proceeds to emit. Requires `std-surface` (the receiver-type map /
/// E2027 check) and `mlir-build` (the `--emit-shared` entry point), so the four
/// method-call tests below are gated on both — a `cargo test --no-default-features`
/// run (no struct-resolver, no `--emit-shared`) compiles them out entirely.
#[cfg(all(feature = "std-surface", feature = "mlir-build"))]
fn build_out(path: &std::path::Path) -> String {
    let so = std::env::temp_dir().join(format!(
        "{}.so",
        path.file_stem().unwrap().to_str().unwrap()
    ));
    let out = mindc()
        .args([
            path.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("spawn mindc build");
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
fn u64_to_float_cast_now_unsigned() {
    // `<u64> as f32|f64` now lowers via UNSIGNED `uitofp` (issue #99: first-class
    // `ScalarU64`), so a `u64` value ≥ 2^63 converts correctly instead of via a
    // wrong signed `sitofp`. It must type-check clean — no E2014 (that diagnostic
    // is fully retired now that u64 has a deterministic lowering everywhere).
    let good = write_tmp(
        "mind_u64_as_float_ok.mind",
        "pub fn ok(x: u64) -> i64 {\n\
         \x20   let y = x as f64\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2014"),
        "`(u64) as f64` now has an unsigned lowering and must NOT be rejected; out: {out}"
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

// ── #230: RFC 0011 extended to the call-argument (E2027) and implicit
//    trailing-expression return (E2010) positions ─────────────────────────────
// Both used to pass `mindc check` (rc=0, fail-open) while `mlir-opt` rejected
// the program with a `'f64' vs 'i64'` scalar-ABI conflict. The BAD cases pin
// that the code now fires EARLY at check; the GREEN cases pin the loose-typed /
// enum-ctor / valid / width-sibling forms that must stay invisible (zero
// over-coverage). NON-VACUOUS: every BAD case returns rc=0 with NO E2027/E2010
// on the pre-#230 compiler and is rejected only after the fix.

#[test]
fn typed_int_arg_into_float_param_rejected() {
    // The #230 arg-position repro: `n: i64` flows into `scale`'s `x: f64`.
    let bad = write_tmp(
        "mind_arg_int_into_float_bad.mind",
        "pub fn scale(x: f64) -> f64 {\n\
         \x20   x + 1.0\n\
         }\n\
         pub fn driver(n: i64) -> f64 {\n\
         \x20   scale(n)\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2027"),
        "i64 argument into f64 parameter not rejected at check (fail-open); out: {out}"
    );
}

#[test]
fn typed_float_arg_into_int_param_rejected() {
    // Mirror direction: a confident-float arg into an i64 parameter.
    let bad = write_tmp(
        "mind_arg_float_into_int_bad.mind",
        "pub fn add1(x: i64) -> i64 {\n\
         \x20   x + 1\n\
         }\n\
         pub fn driver(y: f64) -> i64 {\n\
         \x20   add1(y)\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2027"),
        "f64 argument into i64 parameter not rejected at check (fail-open); out: {out}"
    );
}

#[test]
fn implicit_int_tail_return_from_float_fn_rejected() {
    // The #230 return-position repro: the implicit trailing expression `n`
    // (i64) is the return value of an `-> f64` fn. Not a `Node::Return`, so the
    // pre-#230 E2010 checks (which only match `Node::Return`) missed it.
    let bad = write_tmp(
        "mind_tail_int_into_float_bad.mind",
        "pub fn f(n: i64) -> f64 {\n\
         \x20   n\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "implicit i64 tail return from f64 fn not rejected at check (fail-open); out: {out}"
    );
}

#[test]
fn implicit_float_tail_return_from_int_fn_rejected() {
    // Mirror: implicit trailing float value returned from an `-> i64` fn.
    let bad = write_tmp(
        "mind_tail_float_into_int_bad.mind",
        "pub fn f(n: f64) -> i64 {\n\
         \x20   n\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "implicit f64 tail return from i64 fn not rejected at check (fail-open); out: {out}"
    );
}

// GREEN — must stay invisible (zero over-coverage)

#[test]
fn enum_ctor_arg_no_false_positive() {
    // An enum-variant constructor in argument position resolves to class `None`
    // (`confident_scalar_class` returns `None`), so E2027 must NOT fire.
    let good = write_tmp(
        "mind_arg_enum_ctor_ok.mind",
        "enum Mode {\n\
         \x20   On,\n\
         \x20   Off,\n\
         }\n\
         pub fn g(m: Mode) -> i64 {\n\
         \x20   0\n\
         }\n\
         pub fn h() -> i64 {\n\
         \x20   g(Mode::On)\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2027"),
        "enum-ctor argument falsely rejected E2027: {out}"
    );
}

#[test]
fn valid_float_arg_into_float_param_accepted() {
    let good = write_tmp(
        "mind_arg_float_ok.mind",
        "pub fn scale(x: f64) -> f64 {\n\
         \x20   x + 1.0\n\
         }\n\
         pub fn driver(n: f64) -> f64 {\n\
         \x20   scale(n)\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2027") && !out.contains("E2010"),
        "valid float→float call falsely rejected: {out}"
    );
}

#[test]
fn valid_int_arg_into_int_param_accepted() {
    let good = write_tmp(
        "mind_arg_int_ok.mind",
        "pub fn add1(x: i64) -> i64 {\n\
         \x20   x + 1\n\
         }\n\
         pub fn use_it(n: i64) -> i64 {\n\
         \x20   add1(n)\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2027") && !out.contains("E2010"),
        "valid int→int call falsely rejected: {out}"
    );
}

#[test]
fn width_sibling_arg_accepted() {
    // i32 argument into an i64 parameter: SAME scalar class, different width —
    // must NOT trip E2027 (the class check is int-vs-float, never width).
    let good = write_tmp(
        "mind_arg_width_ok.mind",
        "pub fn add1(x: i64) -> i64 {\n\
         \x20   x + 1\n\
         }\n\
         pub fn use_it(n: i32) -> i64 {\n\
         \x20   add1(n)\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2027"),
        "i32-into-i64 width-sibling argument falsely rejected E2027: {out}"
    );
}

// ── #233: RFC 0011 scalar-class checks extended into match arms + the implicit
//    trailing-return that precedes a binding ──────────────────────────────────
// The class checks previously never entered a `match` arm (a `Match` fell to the
// terminal `_ => {}`), and the tail-return check used `body.last()` so a return
// value preceding a trailing `let` was unchecked. Both are the SAME class of
// early-diagnostic gap (build already fail-closes). BAD cases are non-vacuous:
// each returns NO E20xx on the pre-#233 compiler. GREEN cases pin the
// pattern-bound-shadow / valid forms that must stay invisible.

#[test]
fn match_arm_arg_class_mismatch_rejected() {
    // Float literal into an i64 param, inside a match arm body — previously the
    // arm was never walked so E2027 was silently missed.
    let bad = write_tmp(
        "mind_match_arm_arg_bad.mind",
        "pub fn takes_i64(x: i64) -> i64 {\n\
         \x20   x\n\
         }\n\
         pub fn f(m: i64) -> i64 {\n\
         \x20   match m {\n\
         \x20       _ => takes_i64(1.0),\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2027"),
        "f64 arg into i64 param inside a match arm not rejected; out: {out}"
    );
}

#[test]
fn match_arm_mixed_binop_rejected() {
    // Mixed int/float binop inside a match arm body — E2013 now reaches it.
    let bad = write_tmp(
        "mind_match_arm_binop_bad.mind",
        "pub fn f(a: i64, b: f64, m: i64) -> i64 {\n\
         \x20   match m {\n\
         \x20       _ => a + b,\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2013"),
        "mixed `i64 + f64` inside a match arm not rejected; out: {out}"
    );
}

#[test]
fn match_arm_pattern_bound_shadow_no_false_positive() {
    // The pattern binds `n`, shadowing the outer `n: f64` param. The per-arm
    // ctx must DROP the pattern-bound `n` so `takes_i64(n)` sees class `None`
    // (never the stale outer f64) — otherwise E2027 falsely fires.
    let good = write_tmp(
        "mind_match_arm_shadow_ok.mind",
        "enum Opt {\n\
         \x20   Some(i64),\n\
         \x20   None,\n\
         }\n\
         pub fn takes_i64(x: i64) -> i64 {\n\
         \x20   x\n\
         }\n\
         pub fn h(o: Opt, n: f64) -> i64 {\n\
         \x20   match o {\n\
         \x20       Opt::Some(n) => takes_i64(n),\n\
         \x20       Opt::None => 0,\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2027"),
        "pattern-bound `n` shadowing an outer f64 falsely rejected E2027: {out}"
    );
}

#[test]
fn tail_let_terminated_return_rejected() {
    // The implicit return is `n` (i64), which PRECEDES a trailing `let` — the
    // lowerer returns the last non-binding statement. `body.last()` (the `let`)
    // missed it; the backward scan now checks `n` against the f64 return.
    let bad = write_tmp(
        "mind_tail_let_bad.mind",
        "pub fn f(n: i64) -> f64 {\n\
         \x20   n\n\
         \x20   let _u: i64 = 5\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "i64 return value preceding a trailing `let` in an f64 fn not rejected; out: {out}"
    );
}

#[test]
fn tail_let_terminated_valid_accepted() {
    // Same shape but the tail value is f64 into an f64 fn — must stay clean.
    let good = write_tmp(
        "mind_tail_let_ok.mind",
        "pub fn f(x: f64) -> f64 {\n\
         \x20   x\n\
         \x20   let _u: i64 = 5\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "valid f64 tail value preceding a trailing `let` falsely rejected E2010: {out}"
    );
}

// ── #233 follow-up: inner-scope annotated `let` as a branch tail ─────────────
// A branch-local annotated `let` shadowing an outer binding of the same name
// must resolve its OWN class in the tail check (seeded per-branch ctx), not the
// stale outer class; and a `LetTuple` rebind must DROP the name (never keep a
// stale scalar class). GREEN cases were FALSELY rejected before this fix.

#[test]
fn inner_scope_annotated_tail_no_false_positive() {
    // then-branch's inner `n: f64` shadows the outer `n: i64`; the tail `n` is
    // f64 into an f64 fn — valid. Before the per-branch seed it was mis-classed
    // as the outer i64 → false E2010.
    let good = write_tmp(
        "mind_inner_scope_ok.mind",
        "pub fn f(n: i64, c: i64) -> f64 {\n\
         \x20   if c == 0 {\n\
         \x20       let n: f64 = 1.0\n\
         \x20       n\n\
         \x20   } else {\n\
         \x20       0.0\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "inner-scope `let n: f64` branch tail falsely rejected E2010: {out}"
    );
}

#[test]
fn inner_scope_still_catches_real_mismatch() {
    // then-branch's inner `m: i64` IS returned from an f64 fn — a genuine
    // mismatch that build rejects; E2010 must still fire (seed resolves m=Int).
    let bad = write_tmp(
        "mind_inner_scope_bad.mind",
        "pub fn f(c: i64) -> f64 {\n\
         \x20   if c == 0 {\n\
         \x20       let m: i64 = 1\n\
         \x20       m\n\
         \x20   } else {\n\
         \x20       0.0\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2010"),
        "inner-scope i64 branch tail into f64 fn not rejected; out: {out}"
    );
}

#[test]
fn inner_scope_lettuple_rebind_no_false_positive() {
    // `let n: f64` then `let (n, w) = mkpair()` rebinds `n` to i64 (tuple ABI).
    // The LetTuple binder must DROP `n` (→ class None), else the stale f64 from
    // the first let would false-fire E2010 on the i64 tail `n`.
    let good = write_tmp(
        "mind_inner_lettuple_ok.mind",
        "pub fn mkpair() -> (i64, i64) {\n\
         \x20   (10, 20)\n\
         }\n\
         pub fn f(c: i64) -> i64 {\n\
         \x20   if c == 0 {\n\
         \x20       let n: f64 = 1.0\n\
         \x20       let (n, w) = mkpair()\n\
         \x20       n\n\
         \x20   } else {\n\
         \x20       5\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "LetTuple-rebound `n` falsely rejected E2010 (stale class not dropped): {out}"
    );
}

#[test]
fn inner_scope_shadow_after_statement_no_false_positive() {
    // audit concern: a NON-BINDER statement (`g(x)`) precedes the shadowing
    // `let x: f64`. The per-branch seed must be a FULL sequential scan (not just
    // the leading-let prefix) so the inner `x: f64` is seeded and the tail `x`
    // resolves Float — otherwise the stale outer `x: i64` survives and E2010
    // false-fires on this build-accepted program.
    let good = write_tmp(
        "mind_inner_shadow_after_stmt_ok.mind",
        "pub fn g(x: i64) -> i64 {\n\
         \x20   x\n\
         }\n\
         pub fn f(x: i64, c: i64) -> f64 {\n\
         \x20   if c == 0 {\n\
         \x20       g(x)\n\
         \x20       let x: f64 = 1.5\n\
         \x20       x\n\
         \x20   } else {\n\
         \x20       2.0\n\
         \x20   }\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2010"),
        "shadow `let x: f64` after a non-binder statement falsely rejected E2010 \
         (seed is prefix-only, not full scan): {out}"
    );
}

// ---------------------------------------------------------------------------
// #233 follow-up — E2027 in the METHOD-CALL argument position (RFC 0011).
//
// A `receiver.method(arg)` on a struct-typed receiver lowers (UFCS static
// dispatch) to the free fn `{type}_{method}(self, arg...)` that `desugar_traits`
// lifts *before* type inference — but ONLY on the compile (`--emit-shared`)
// pipeline. `mindc check` does not run `desugar_traits`, so a raw trait/impl is
// rejected there at E2001 before type inference; the method-call class check
// therefore fires on the BUILD path, where it turns a confident Int/Float
// argument into an oppositely-classed declared method parameter — the exact
// scalar-ABI mismatch `mlir-opt` rejects late (`'f64' vs 'i64'`) — into an EARLY
// type-check-phase E2027, returned before lowering. NON-VACUOUS: on the pre-fix
// compiler the method-call position had NO class check (the arm fell to the
// terminal `_ => {}`), so `s.scale(2)` compiled its type-check phase clean and
// only failed at `mlir-opt`. Uses `build_out` (a real compile), not `check_out`.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "std-surface", feature = "mlir-build"))]
#[test]
fn method_arg_int_into_float_param_rejected() {
    // `s.scale(2)` — int literal into an `f64`-declared method param. The lifted
    // free fn is `sc_scale(self: Sc, k: f64)`; arg 0 (`2` → Int) vs param 1
    // (`f64` → Float) mismatch → E2027 at the type-check phase of the build,
    // returned before `mlir-opt` ever runs.
    let bad = write_tmp(
        "mind_method_arg_int_into_f64.mind",
        "struct Sc { tag: i64 }\n\
         trait Scale {\n\
         \x20   fn scale(self, k: f64) -> f64\n\
         }\n\
         impl Scale for Sc {\n\
         \x20   fn scale(self, k: f64) -> f64 {\n\
         \x20       return k\n\
         \x20   }\n\
         }\n\
         pub fn bad() -> f64 {\n\
         \x20   let s = Sc { tag: 0 }\n\
         \x20   return s.scale(2)\n\
         }\n",
    );
    let out = build_out(&bad);
    assert!(
        out.contains("E2027"),
        "int literal `2` into `f64` method param NOT rejected E2027 (method-call \
         arg class check missing on the build path): {out}"
    );
}

#[cfg(all(feature = "std-surface", feature = "mlir-build"))]
#[test]
fn method_arg_float_into_int_param_rejected() {
    // Mirror direction: `c.bump(1.5)` — float literal into an `i64`-declared
    // method param. Lifted `ct_bump(self: Ct, n: i64)`; arg 0 (`1.5` → Float) vs
    // param 1 (`i64` → Int) → E2027.
    let bad = write_tmp(
        "mind_method_arg_float_into_i64.mind",
        "struct Ct { c: i64 }\n\
         trait Bump {\n\
         \x20   fn bump(self, n: i64) -> i64\n\
         }\n\
         impl Bump for Ct {\n\
         \x20   fn bump(self, n: i64) -> i64 {\n\
         \x20       return n\n\
         \x20   }\n\
         }\n\
         pub fn bad() -> i64 {\n\
         \x20   let c = Ct { c: 3 }\n\
         \x20   return c.bump(1.5)\n\
         }\n",
    );
    let out = build_out(&bad);
    assert!(
        out.contains("E2027"),
        "float literal `1.5` into `i64` method param NOT rejected E2027: {out}"
    );
}

#[cfg(all(feature = "std-surface", feature = "mlir-build"))]
#[test]
fn method_arg_correct_class_no_false_positive() {
    // The GREEN control: `s.scale(2.0)` passes the SAME arg into the SAME `f64`
    // param in the correct class. Must NOT false-fire E2027 — a valid program
    // that compiles to a `.so` end-to-end (zero over-coverage is load-bearing).
    let good = write_tmp(
        "mind_method_arg_correct_class_ok.mind",
        "struct Sc { tag: i64 }\n\
         trait Scale {\n\
         \x20   fn scale(self, k: f64) -> f64\n\
         }\n\
         impl Scale for Sc {\n\
         \x20   fn scale(self, k: f64) -> f64 {\n\
         \x20       return k\n\
         \x20   }\n\
         }\n\
         pub fn okc() -> f64 {\n\
         \x20   let s = Sc { tag: 0 }\n\
         \x20   return s.scale(2.0)\n\
         }\n",
    );
    let out = build_out(&good);
    assert!(
        !out.contains("E2027"),
        "correctly-classed `s.scale(2.0)` falsely rejected E2027: {out}"
    );
}

#[cfg(all(feature = "std-surface", feature = "mlir-build"))]
#[test]
fn method_zero_arg_call_no_false_positive() {
    // The zero-arg method-as-field case audit flagged: `foo.val()` takes no
    // args, so the self-inclusive arity gate holds (`param_types.len() == 1 ==
    // 0 + 1`) and the arg loop is empty — no E2027, no panic (the `skip(1)`
    // self-drop stays in bounds). This is the proven `trait_static_dispatch_run`
    // shape (compiles + returns 42).
    let good = write_tmp(
        "mind_method_zero_arg_ok.mind",
        "struct Foo { x: i64, y: i64 }\n\
         trait Speak {\n\
         \x20   fn val(self) -> i64\n\
         }\n\
         impl Speak for Foo {\n\
         \x20   fn val(self) -> i64 {\n\
         \x20       return self.x + self.y\n\
         \x20   }\n\
         }\n\
         pub fn okz() -> i64 {\n\
         \x20   let foo = Foo { x: 40, y: 2 }\n\
         \x20   return foo.val()\n\
         }\n",
    );
    let out = build_out(&good);
    assert!(
        !out.contains("E2027"),
        "zero-arg `foo.val()` falsely rejected E2027 (arity/self-skip wrong): {out}"
    );
}
