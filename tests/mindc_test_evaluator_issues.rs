// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc test` executes a test body ONCE, in source order, through the
//! interpreter — and a body that errors cannot pass (#240, #241, #243).
//!
//! The runner used to evaluate the body, discard the result, then walk the AST
//! a second time evaluating only `assert` nodes. Three reported false verdicts
//! followed from that one design, and each shape below was measured against the
//! pre-fix binary before it was pinned here:
//!
//! * **#243** — `read_oob()` errored in the first pass; the second pass
//!   re-evaluated the `let`, dropped the error, and a later `assert n == 1`
//!   graded the test `ok`. A bare out-of-bounds load statement was skipped
//!   outright. An evaluation error now fails the test unconditionally, with
//!   the interpreter's own diagnostic (address AND requested allocation extent).
//! * **#241** — the second pass could not replay memory effects: it
//!   re-allocated a fresh zeroed block for `let x = __mind_alloc(8)`, skipped
//!   the store, and read `0`, so BOTH issue fixtures failed. Asserts are now
//!   evaluated at their own point with every preceding effect applied.
//! * **#240** — the tuple form `assert(cond, "msg")` parsed as a 2-tuple
//!   condition and read as "truthy" whatever `cond` was. It is refused with a
//!   message naming the form; the grammar itself is a separate report.
//!
//! Every negative shape has a positive control in the SAME file so a pass here
//! is evidence the runner ran, not evidence that it refused everything.
//!
//! Gate: `cargo test --test mindc_test_evaluator_issues`

mod common;
use common::{require_mindc, scratch_dir};

use std::process::Command;

/// Write `src` as `<stem>.mind` in this target's private scratch dir, run
/// `mindc test` on it, and return `(exit code, stdout+stderr)`.
fn run_mindc_test(stem: &str, src: &str) -> (i32, String) {
    let path = scratch_dir("mindc_test_evaluator_issues").join(format!("{stem}.mind"));
    std::fs::write(&path, src).expect("write fixture");
    let out = Command::new(require_mindc())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), combined)
}

fn assert_line(out: &str, needle: &str) {
    assert!(
        out.contains(needle),
        "expected `{needle}` in mindc test output:\n{out}"
    );
}

fn assert_no_line(out: &str, needle: &str) {
    assert!(
        !out.contains(needle),
        "did NOT expect `{needle}` in mindc test output:\n{out}"
    );
}

// ---------------------------------------------------------------------------
// #243 — an evaluation error fails the test, whatever a later assert says
// ---------------------------------------------------------------------------

/// The issue's own repro, verbatim: the OOB read is CONSUMED by a `let`, then an
/// unrelated true assertion follows. `test_no_oob` is the positive control.
#[cfg(feature = "std-surface")] // the repro's `while` loop is a std-surface construct
const ISSUE_243_EXACT: &str = r#"
fn read_oob() -> i64 {
    let buf: i64 = __mind_alloc(8);
    __mind_store_i64(buf, 1);
    let mut i: i64 = 0;
    let mut acc: i64 = 0;
    while i < 65536 {
        acc = acc + __mind_load_i64(buf + i * 8);
        i = i + 1;
    }
    acc
}

#[test]
fn test_after_oob_read() {
    let r: i64 = read_oob();
    let n: i64 = 1;
    assert n == 1, "binding after OOB-reading call";
}

#[test]
fn test_no_oob() {
    let buf: i64 = __mind_alloc(8);
    __mind_store_i64(buf, 1);
    let n: i64 = 1;
    assert n == 1, "control";
}
"#;

#[test]
#[cfg(feature = "std-surface")] // the repro's `while` loop is a std-surface construct
fn issue243_exact_repro_fails_the_oob_test_and_passes_the_control() {
    let (code, out) = run_mindc_test("issue243_exact", ISSUE_243_EXACT);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "test issue243_exact::test_after_oob_read ... FAILED");
    assert_line(&out, "memory access out of bounds");
    assert_line(&out, "requested allocation extent");
    // Positive control: the runner still grades a sound memory-using body.
    assert_line(&out, "test issue243_exact::test_no_oob ... ok");
    assert_line(&out, "1 passed; 1 failed");
}

/// The OOB value is consumed by a `let`, then an UNRELATED true assertion
/// follows. This graded `ok` before the fix (root receipt, fixture
/// `issue243_oob_then_unrelated_pass`).
#[test]
fn issue243_consumed_oob_then_unrelated_true_assert_fails() {
    let src = r#"
fn read_oob() -> i64 {
    let buf: i64 = __mind_alloc(8);
    __mind_store_i64(buf, 1);
    return __mind_load_i64(buf + 8);
}

#[test]
fn oob_then_unrelated_pass_must_still_fail() {
    let r: i64 = read_oob();
    assert 1 == 1, "unrelated true assertion";
}
"#;
    let (code, out) = run_mindc_test("issue243_consumed", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "oob_then_unrelated_pass_must_still_fail ... FAILED");
    assert_line(&out, "memory access out of bounds: [16, 16+8)");
    assert_line(&out, "requested allocation extent [8, 16)");
    assert_no_line(&out, "unrelated true assertion");
}

/// The OOB load is a bare expression statement — its value is never bound or
/// used — followed by an unrelated true assertion. The old second pass skipped
/// expression statements entirely, so this graded `ok`.
#[test]
fn issue243_unused_oob_load_then_unrelated_true_assert_fails() {
    let src = r#"
#[test]
fn unused_oob_load_must_fail() {
    let buf: i64 = __mind_alloc(8);
    __mind_store_i64(buf, 1);
    __mind_load_i64(buf + 8);
    assert 1 == 1, "unrelated true assertion";
}
"#;
    let (code, out) = run_mindc_test("issue243_unused", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "unused_oob_load_must_fail ... FAILED");
    assert_line(&out, "memory access out of bounds: [16, 16+8)");
    assert_line(&out, "requested allocation extent [8, 16)");
}

/// The OOB value is USED by the assert. The extent diagnostic must be the
/// failure text, with no downstream `unknown variable` symptom attached — the
/// body aborted at the load and nothing after it was evaluated.
#[test]
fn issue243_oob_value_used_by_assert_reports_the_extent_only() {
    let src = r#"
#[test]
fn oob_value_used_by_assert_must_fail() {
    let buf: i64 = __mind_alloc(1);
    let v: i64 = __mind_load_i8(buf + 1);
    assert v == 0, "read past a one-byte allocation";
}
"#;
    let (code, out) = run_mindc_test("issue243_used", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "memory access out of bounds: [9, 9+1)");
    assert_line(&out, "requested allocation extent [8, 9)");
    assert_no_line(&out, "unknown variable");
    assert_no_line(&out, "read past a one-byte allocation");
}

// ---------------------------------------------------------------------------
// #241 — asserts see memory as it is AT THEIR POINT
// ---------------------------------------------------------------------------

/// The issue's own repro, verbatim. Both tests must pass: the condition was
/// true when the assert ran, and the later store must not change that.
const ISSUE_241_EXACT: &str = r#"
#[test]
fn test_assert_then_mutate() {
    let x: i64 = __mind_alloc(8);
    __mind_store_i8(x, 1);
    assert __mind_load_i8(x) == 1, "reads before mutation";
    __mind_store_i8(x, 2);
}

#[test]
fn test_assert_then_mutate_letbound() {
    let x: i64 = __mind_alloc(8);
    __mind_store_i8(x, 1);
    let v: i64 = __mind_load_i8(x);
    assert v == 1, "let-bound reads before mutation";
    __mind_store_i8(x, 2);
}
"#;

#[test]
fn issue241_exact_repro_both_tests_pass() {
    let (code, out) = run_mindc_test("issue241_exact", ISSUE_241_EXACT);
    assert_eq!(code, 0, "both tests must pass:\n{out}");
    assert_line(&out, "test issue241_exact::test_assert_then_mutate ... ok");
    assert_line(
        &out,
        "test issue241_exact::test_assert_then_mutate_letbound ... ok",
    );
    assert_line(&out, "2 passed; 0 failed");
}

/// Mirror image: the condition is FALSE at the assert's point and only the
/// LATER store would make it true. A runner reading final state passes this.
#[test]
fn issue241_false_at_its_point_fails_even_though_a_later_store_makes_it_true() {
    let src = r#"
#[test]
fn false_at_its_point_must_fail() {
    let x: i64 = __mind_alloc(8);
    __mind_store_i8(x, 1);
    assert __mind_load_i8(x) == 2, "negative control: false before the later store";
    __mind_store_i8(x, 2);
}
"#;
    let (code, out) = run_mindc_test("issue241_negative", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "false_at_its_point_must_fail ... FAILED");
    assert_line(&out, "negative control: false before the later store");
}

/// A scalar assign after the assert, both directions, in one file: the true-
/// at-its-point test passes and the false-at-its-point test fails.
#[test]
fn issue241_scalar_assign_after_assert_both_directions() {
    let src = r#"
#[test]
fn true_then_reassigned() {
    let mut x: i64 = 1;
    assert x == 1, "x is 1 here";
    x = 2;
}

#[test]
fn false_then_reassigned() {
    let mut y: i64 = 1;
    assert y == 2, "y is not 2 here";
    y = 2;
}
"#;
    let (code, out) = run_mindc_test("issue241_scalar", src);
    assert_ne!(code, 0, "one of the two must fail:\n{out}");
    assert_line(&out, "issue241_scalar::true_then_reassigned ... ok");
    assert_line(&out, "issue241_scalar::false_then_reassigned ... FAILED");
    assert_line(&out, "y is not 2 here");
    assert_line(&out, "1 passed; 1 failed");
}

/// Indexed setup assignments in a `#[test]` body must use the same statement
/// executor as ordinary function bodies. The stale-value test is the mutation
/// control: it must fail after the write instead of passing on the initializer.
#[test]
fn indexed_and_nested_assignments_execute_in_test_bodies() {
    let src = r#"
#[test]
fn indexed_write_persists() {
    let mut xs: [i64; 2] = [0, 0];
    let i: i64 = 1;
    xs[i] = 7;
    assert xs[1] == 7, "indexed write persists";
}

#[test]
fn if_body_write_persists() {
    let mut n: i64 = 0;
    if 1 == 1 {
        n = 5;
    }
    assert n == 5, "if-body write persists";
}

#[test]
fn stale_array_value_must_fail() {
    let mut xs: [i64; 1] = [0];
    xs[0] = 7;
    assert xs[0] == 0, "indexed write must invalidate the initializer";
}
"#;

    let (code, out) = run_mindc_test("test_body_assignments", src);
    assert_ne!(
        code, 0,
        "mutation control must make the command fail:\n{out}"
    );
    assert_line(&out, "indexed_write_persists ... ok");
    assert_line(&out, "if_body_write_persists ... ok");
    assert_line(&out, "stale_array_value_must_fail ... FAILED");
    assert_line(&out, "indexed write must invalidate the initializer");
    assert_line(&out, "2 passed; 1 failed");
}

/// A scalar receiver used to evaluate only the RHS and pass on the old value.
/// Keep an ordinary array write beside it so refusal cannot make every write red.
#[test]
fn unsupported_indexed_assignment_receiver_fails_closed() {
    let src = r#"
#[test]
fn ordinary_array_control() {
    let mut xs: [i64; 1] = [0];
    xs[0] = 7;
    assert xs[0] == 7, "ordinary indexed write";
}

#[test]
fn scalar_receiver_is_an_error() {
    let mut x: i64 = 0;
    x[0] = 7;
    assert x == 0, "unchanged scalar cannot attest to setup";
}
"#;

    let (code, out) = run_mindc_test("indexed_assignment_fail_closed", src);
    assert_ne!(
        code, 0,
        "unsupported write must make the command fail:\n{out}"
    );
    assert_line(&out, "ordinary_array_control ... ok");
    assert_line(&out, "scalar_receiver_is_an_error ... FAILED");
    assert_line(&out, "indexed assignment receiver `x` must be an array");
    assert_line(&out, "1 passed; 1 failed");
}

#[test]
fn nested_indexed_assignment_receiver_fails_closed() {
    let src = r#"
#[test]
fn nested_receiver_is_an_error() {
    let mut rows: [[i64; 1]; 1] = [[0]];
    rows[0][0] = 7;
    assert rows[0][0] == 0, "nested write was discarded";
}
"#;
    let (code, out) = run_mindc_test("nested_indexed_assignment", src);
    assert_ne!(code, 0, "unsupported nested write must fail:\n{out}");
    assert_line(&out, "nested_receiver_is_an_error ... FAILED");
    assert_line(
        &out,
        "indexed assignment receiver must be a simple array variable",
    );
    assert_line(&out, "0 passed; 1 failed");
}

#[test]
fn missing_indexed_assignment_receiver_fails_closed() {
    let src = r#"
#[test]
fn missing_receiver_is_an_error() {
    missing[0] = 7;
    assert 1 == 1, "later assertion cannot hide the missing receiver";
}
"#;
    let (code, out) = run_mindc_test("missing_indexed_assignment", src);
    assert_ne!(code, 0, "missing receiver must fail:\n{out}");
    assert_line(&out, "missing_receiver_is_an_error ... FAILED");
    assert_line(&out, "unknown variable: missing");
    assert_line(&out, "0 passed; 1 failed");
}

// ---------------------------------------------------------------------------
// #240 — call-result and struct bindings are visible; precise failure causes
// ---------------------------------------------------------------------------

const ISSUE_240_EXACT: &str = r#"
struct Pair {
    a: i64,
    b: i64,
}

fn mk() -> Pair {
    return Pair { a: 1, b: 2 };
}

fn forty_two() -> i64 {
    return 42;
}

#[test]
fn test_plain_assert() {
    assert(42 == 42);
}

#[test]
fn test_literal_let_assert() {
    let x = 42;
    assert(x == 42);
}

#[test]
fn test_call_let_assert() {
    let x = forty_two();
    assert(x == 42);
}

#[test]
fn test_struct_let_assert() {
    let p = mk();
    assert(p.a == 1);
}
"#;

#[test]
fn issue240_exact_repro_all_four_pass() {
    let (code, out) = run_mindc_test("issue240_exact", ISSUE_240_EXACT);
    assert_eq!(code, 0, "all four must pass:\n{out}");
    assert_line(&out, "issue240_exact::test_plain_assert ... ok");
    assert_line(&out, "issue240_exact::test_literal_let_assert ... ok");
    assert_line(&out, "issue240_exact::test_call_let_assert ... ok");
    assert_line(&out, "issue240_exact::test_struct_let_assert ... ok");
    assert_line(&out, "4 passed; 0 failed");
}

/// Negative control for the struct binding: a WRONG field value must fail, with
/// the plain-assert default message — never `unknown variable: p`.
#[test]
fn issue240_wrong_struct_field_fails_with_the_assertion_not_an_unknown_variable() {
    let src = r#"
struct Pair {
    a: i64,
    b: i64,
}

fn mk() -> Pair {
    return Pair { a: 1, b: 2 };
}

#[test]
fn wrong_struct_field_must_fail() {
    let p = mk();
    assert(p.a == 9);
}
"#;
    let (code, out) = run_mindc_test("issue240_negative", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "wrong_struct_field_must_fail ... FAILED");
    assert_line(&out, "assertion failed");
    assert_no_line(&out, "unknown variable");
}

/// The message form: `assert cond, "msg"` — true passes, false fails with the
/// author's message verbatim.
#[test]
fn issue240_message_form_true_passes_false_fails_with_its_message() {
    let src = r#"
struct Pair {
    a: i64,
    b: i64,
}

fn mk() -> Pair {
    return Pair { a: 1, b: 2 };
}

#[test]
fn message_form_true() {
    let p = mk();
    assert p.a == 1, "a is one";
}

#[test]
fn message_form_false() {
    let p = mk();
    assert p.a == 9, "a is not nine";
}
"#;
    let (code, out) = run_mindc_test("issue240_message", src);
    assert_ne!(code, 0, "the false one must fail:\n{out}");
    assert_line(&out, "issue240_message::message_form_true ... ok");
    assert_line(&out, "issue240_message::message_form_false ... FAILED");
    assert_line(&out, "a is not nine");
    assert_line(&out, "1 passed; 1 failed");
}

// ---------------------------------------------------------------------------
// Struct field assignment — fail closed until interpreter mutation semantics
// are explicitly defined (the native lowering path is a separate contract).
// ---------------------------------------------------------------------------

const STRUCT_FIELD_ASSIGNMENT_REFUSAL: &str = r#"
struct Point {
    x: i64,
}

fn read_point() -> i64 {
    let p: Point = Point { x: 1 };
    return p.x;
}

fn store_direct() -> i64 {
    let mut p: Point = Point { x: 1 };
    p.x = 9;
    return p.x;
}

fn store_helper() -> i64 {
    return store_direct();
}

#[test]
fn struct_read_only_helper_still_passes() {
    assert read_point() == 1, "read-only struct helper"
}

#[test]
fn struct_store_direct_refuses() {
    let got = store_direct();
    assert got == 9, "store must not silently retain the old value"
}

#[test]
fn struct_store_called_helper_refuses() {
    let got = store_helper();
    assert got == 9, "store in a called helper must not silently disappear"
}

#[test]
fn struct_store_old_value_control_refuses() {
    let got = store_direct();
    assert got == 1, "old-value assertion must not pass when the store is unsupported"
}
"#;

#[test]
fn struct_field_assignment_refuses_in_direct_and_called_helpers() {
    let (code, out) = run_mindc_test(
        "struct_field_assignment_refusal",
        STRUCT_FIELD_ASSIGNMENT_REFUSAL,
    );
    assert_ne!(code, 0, "field mutation must not report green:\n{out}");
    assert_line(&out, "struct_read_only_helper_still_passes ... ok");
    assert_line(&out, "struct_store_direct_refuses ... FAILED");
    assert_line(&out, "struct_store_called_helper_refuses ... FAILED");
    assert_line(&out, "struct_store_old_value_control_refuses ... FAILED");
    assert_line(
        &out,
        "struct field assignment `.x` is unsupported by the interpreter; mutation semantics are not defined",
    );
    assert_line(&out, "1 passed; 3 failed");
}

#[cfg(feature = "std-surface")]
#[test]
fn struct_field_assignment_refuses_inside_loop() {
    let src = r#"
struct Point {
    x: i64,
}

fn store_loop() -> i64 {
    let mut p: Point = Point { x: 1 };
    let mut i: i64 = 0;
    while i < 1 {
        p.x = 9;
        i = i + 1;
    }
    return p.x;
}

#[test]
fn struct_store_loop_refuses() {
    let got = store_loop();
    assert got == 9, "store in a loop must not silently disappear"
}
"#;
    let (code, out) = run_mindc_test("struct_field_assignment_loop_refusal", src);
    assert_ne!(code, 0, "loop field mutation must not report green:\n{out}");
    assert_line(&out, "struct_store_loop_refuses ... FAILED");
    assert_line(
        &out,
        "struct field assignment `.x` is unsupported by the interpreter; mutation semantics are not defined",
    );
    assert_line(&out, "0 passed; 1 failed");
}

/// `assert(cond, "msg")` — the parenthesised comma form — parsed as a 2-tuple
/// CONDITION and graded `ok` regardless of `cond`; the evaluator then refused it
/// for a TRUE and a FALSE `cond` alike. The parser now splits it into the
/// condition and the message, so it asserts `cond` like `assert cond, "msg"`.
#[test]
fn issue240_parenthesised_comma_form_asserts_its_condition() {
    let src = r#"
struct Pair {
    a: i64,
    b: i64,
}

fn mk() -> Pair {
    return Pair { a: 1, b: 2 };
}

#[test]
fn tuple_form_true_cond() {
    let p = mk();
    assert(p.a == 1, "message");
}

#[test]
fn tuple_form_false_cond() {
    let p = mk();
    assert(p.a == 9, "message");
}
"#;
    let (code, out) = run_mindc_test("issue240_tuple_form", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "issue240_tuple_form::tuple_form_true_cond ... ok");
    assert_line(
        &out,
        "issue240_tuple_form::tuple_form_false_cond ... FAILED",
    );
    assert_line(&out, "message");
    assert_line(&out, "1 passed; 1 failed");
}

// ---------------------------------------------------------------------------
// Asserts are EXECUTED where the old walker refused or could not see them
// ---------------------------------------------------------------------------

/// An assert inside a CALLED helper fn is checked like any other statement.
#[test]
fn assert_inside_a_called_helper_is_checked() {
    let src = r#"
fn check(v: i64) {
    assert v == 1, "helper saw the wrong value";
}

#[test]
fn helper_assert_holds() {
    check(1);
}

#[test]
fn helper_assert_fires() {
    check(2);
}
"#;
    let (code, out) = run_mindc_test("helper_assert", src);
    assert_ne!(code, 0, "the firing one must fail:\n{out}");
    assert_line(&out, "helper_assert::helper_assert_holds ... ok");
    assert_line(&out, "helper_assert::helper_assert_fires ... FAILED");
    assert_line(&out, "helper saw the wrong value");
    assert_line(&out, "1 passed; 1 failed");
}

/// An assert inside a `for` body runs per iteration: a bound that holds for
/// every `i` passes, one that fails at `i == 2` fails — the walker used to
/// refuse both.
#[test]
fn assert_inside_a_for_loop_is_executed_per_iteration() {
    let src = r#"
#[test]
fn loop_assert_holds() {
    for i in 0..3 {
        assert i < 3, "in range";
    }
}

#[test]
fn loop_assert_fires() {
    for i in 0..3 {
        assert i < 2, "loop assert fires at i == 2";
    }
}
"#;
    let (code, out) = run_mindc_test("loop_assert", src);
    assert_ne!(code, 0, "the firing one must fail:\n{out}");
    assert_line(&out, "loop_assert::loop_assert_holds ... ok");
    assert_line(&out, "loop_assert::loop_assert_fires ... FAILED");
    assert_line(&out, "loop assert fires at i == 2");
    assert_no_line(&out, "assertion inside a loop body");
    assert_line(&out, "1 passed; 1 failed");
}

/// Same for a `while` loop whose counter is mutated in the body.
#[test]
#[cfg(feature = "std-surface")]
fn assert_inside_a_while_loop_is_executed_per_iteration() {
    let src = r#"
#[test]
fn while_assert_holds() {
    let mut i: i64 = 0;
    while i < 3 {
        assert i < 3, "in range";
        i = i + 1;
    }
    assert i == 3, "loop ran to completion";
}

#[test]
fn while_assert_fires() {
    let mut i: i64 = 0;
    while i < 3 {
        assert i < 2, "while assert fires at i == 2";
        i = i + 1;
    }
}
"#;
    let (code, out) = run_mindc_test("while_assert", src);
    assert_ne!(code, 0, "the firing one must fail:\n{out}");
    assert_line(&out, "while_assert::while_assert_holds ... ok");
    assert_line(&out, "while_assert::while_assert_fires ... FAILED");
    assert_line(&out, "while assert fires at i == 2");
    assert_line(&out, "1 passed; 1 failed");
}

/// A `-> bool` test is graded on its value: `true` passes, `false` fails.
#[test]
fn bool_returning_test_is_graded_on_its_value() {
    let src = r#"
#[test]
fn returns_true() -> bool {
    true
}

#[test]
fn returns_false() -> bool {
    false
}
"#;
    let (code, out) = run_mindc_test("bool_return", src);
    assert_ne!(code, 0, "the false one must fail:\n{out}");
    assert_line(&out, "bool_return::returns_true ... ok");
    assert_line(&out, "bool_return::returns_false ... FAILED");
    assert_line(&out, "test returned false (0)");
    assert_line(&out, "1 passed; 1 failed");
}

/// An unresolved name inside a CALLEE is the root cause, reported by name; the
/// correctly-bound caller variable is never blamed (#242, the sibling symptom).
#[test]
fn unresolved_name_in_callee_is_the_reported_cause() {
    let src = r#"
fn poisoned(out: i64) -> i64 {
    __mind_store_i8(out, 7);
    return UNDEFINED_CONST_XYZ;
}

#[test]
fn caller_is_not_blamed() {
    let out: i64 = __mind_alloc(8);
    let rc: i64 = poisoned(out);
    assert rc == 7, "rc is bound; the real error is the undefined const";
}
"#;
    let (code, out) = run_mindc_test("root_cause", src);
    assert_ne!(code, 0, "exit code must be non-zero:\n{out}");
    assert_line(&out, "unknown variable: UNDEFINED_CONST_XYZ");
    assert_no_line(&out, "unknown variable: rc");
}
