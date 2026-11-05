// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc test` runs five constructs the way the compiled build runs them:
//! a float comparison combined by `&&` / `||` / `!` or used as a condition,
//! a struct field store, `break` / `continue`, a for-each loop, and the
//! std string builders a program calls without an import.
//!
//! Each case is ONE program with a `fn score() -> i64`. The interpreter leg
//! runs it under `mindc test` with `assert score() == EXPECTED`; the native
//! leg builds the same source with `fn main() -> i64 { return score(); }`
//! (`mindc build`), runs the artifact it reports, and reads the score from the
//! exit status. Both legs are held to the same EXPECTED, so they agree with
//! each other. Every interpreter leg also runs with EXPECTED + 1 and must
//! FAIL: a runner that passes everything cannot pass this file.
//!
//! The native leg needs the MLIR backend; without it the step is a coded
//! capability skip ([`common::gate`]), which `MIND_BENCH_REQUIRE=1` turns
//! into a failure.
//!
//! Gate: `cargo test --release --no-default-features
//!        --features mlir-build,std-surface,cross-module-imports
//!        --test mindc_test_eval_parity`

#![cfg(feature = "std-surface")]

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// `&&` / `||` / `!` over float comparisons, a false float comparison as an
/// `if` condition, and a float `while` condition that turns false.
const FLOAT_CONDITIONS: &str = r#"
fn in_range(s: f64) -> bool {
    return s >= 1.0 && s <= 5.0;
}

fn outside(s: f64) -> bool {
    return s < 1.0 || s > 5.0;
}

fn score() -> i64 {
    let mut code: i64 = 0;
    if in_range(3.0) {
        code = code + 1;
    }
    if !in_range(7.0) {
        code = code + 2;
    }
    if outside(0.5) {
        code = code + 4;
    }
    if !outside(2.0) {
        code = code + 8;
    }
    if 2.0 < 1.0 {
        code = code + 100;
    } else {
        code = code + 16;
    }
    let mut x: f64 = 0.0;
    let mut steps: i64 = 0;
    while x < 1.0 {
        x = x + 0.25;
        steps = steps + 1;
    }
    return code + steps * 32;
}
"#;

/// A field store through the binding, through a second binding of the same
/// record, and through a `&mut` parameter: all three reach one record.
const FIELD_STORE_IDENTITY: &str = r#"
struct P {
    x: i64,
    y: i64,
}

fn bump(p: &mut P) {
    p.x = p.x + 10;
}

fn score() -> i64 {
    let mut p: P = P { x: 1, y: 2 };
    p.x = 7;
    let mut q: P = p;
    q.x = q.x + 1;
    bump(&mut p);
    return p.x * 10 + p.y;
}
"#;

/// Field stores inside `while` / `if` / `else` / range-`for` bodies survive
/// the block they were made in.
const FIELD_STORE_IN_LOOPS: &str = r#"
struct P {
    x: i64,
    y: i64,
}

fn score() -> i64 {
    let mut p: P = P { x: 0, y: 0 };
    let mut i: i64 = 0;
    while i < 5 {
        if i % 2 == 0 {
            p.x = p.x + i;
        } else {
            p.y = p.y + 1;
        }
        i = i + 1;
    }
    for k in 0..3 {
        p.y = p.y + k;
    }
    return p.x * 10 + p.y;
}
"#;

/// An array element is a record reference: a store through a binding read out
/// of the array is seen through the array.
const FIELD_STORE_ARRAY_ELEMENT: &str = r#"
struct P {
    x: i64,
}

fn score() -> i64 {
    let ps: array<P> = [P { x: 1 }, P { x: 2 }];
    let first: P = ps[0];
    first.x = 40;
    return ps[0].x + ps[1].x;
}
"#;

/// A fixed-array field is a value container: an element store changes the
/// field, not an array copied out of it before the store.
const FIXED_ARRAY_FIELD_STORE: &str = r#"
struct Bank {
    xs: [i64; 3],
}

fn score() -> i64 {
    let mut b: Bank = Bank { xs: [1, 2, 3] };
    let before: [i64; 3] = b.xs;
    b.xs[1] = 20;
    return b.xs[1] * 10 + before[1];
}
"#;

/// `break` inside `if` inside `while 1`, `continue` in a `while`, both in a
/// range `for`, and a `break` that leaves only the inner of two loops.
const BREAK_CONTINUE: &str = r#"
fn first_ge(n: i32) -> i32 {
    let mut i: i32 = 0;
    while 1 {
        if i >= n {
            break;
        }
        i = i + 1;
    }
    return i;
}

fn odd_sum(n: i64) -> i64 {
    let mut i: i64 = 0;
    let mut s: i64 = 0;
    while i < n {
        i = i + 1;
        if i % 2 == 0 {
            continue;
        }
        s = s + i;
    }
    return s;
}

fn range_break() -> i64 {
    let mut s: i64 = 0;
    for i in 0..100 {
        if i == 5 {
            break;
        }
        s = s + i;
    }
    return s;
}

fn range_continue() -> i64 {
    let mut s: i64 = 0;
    for i in 0..6 {
        if i % 2 == 1 {
            continue;
        }
        s = s + i;
    }
    return s;
}

fn inner_only() -> i64 {
    let mut c: i64 = 0;
    for i in 0..3 {
        let mut j: i64 = 0;
        while 1 {
            if j == 2 {
                break;
            }
            j = j + 1;
            c = c + 1;
        }
    }
    return c;
}

fn score() -> i64 {
    return (first_ge(3) as i64) + odd_sum(5) + range_break() + range_continue() + inner_only();
}
"#;

/// For-each over an array variable and an enum literal, with `break` and
/// `continue`, and an element store ahead of the loop position, which the
/// compiled loop reads back on a later iteration.
const FOR_EACH: &str = r#"
enum Boundary {
    Reflective,
    Periodic,
    Zero,
}

fn sum_var() -> i64 {
    let xs: array<i64> = [4, 5, 6];
    let mut s: i64 = 0;
    for v in xs {
        s = s + v;
    }
    return s;
}

fn count_enums() -> i64 {
    let mut n: i64 = 0;
    for bc in [Boundary.Reflective, Boundary.Periodic, Boundary.Zero] {
        if bc == Boundary.Periodic {
            n = n + 10;
        }
        n = n + 1;
    }
    return n;
}

fn with_break() -> i64 {
    let xs: array<i64> = [1, 2, 3, 4];
    let mut s: i64 = 0;
    for v in xs {
        if v == 3 {
            break;
        }
        s = s + v;
    }
    return s;
}

fn with_continue() -> i64 {
    let xs: array<i64> = [1, 2, 3, 4];
    let mut s: i64 = 0;
    for v in xs {
        if v == 2 {
            continue;
        }
        s = s + v;
    }
    return s;
}

fn store_ahead() -> i64 {
    let mut xs: array<i64> = [1, 1, 1];
    let mut s: i64 = 0;
    for v in xs {
        xs[2] = 10;
        s = s + v;
    }
    return s;
}

fn score() -> i64 {
    return sum_var() + count_enums() + with_break() + with_continue() + store_ahead();
}
"#;

/// The std.string builders, called with no import (the compiled build links
/// them from its runtime support), including growth past the first capacity.
const STRING_BUILDERS: &str = r#"
fn score() -> i64 {
    let mut s: String = string_new();
    s = string_push_byte(s, 65);
    s = string_push_byte(s, 66);
    let mut t: String = string_new();
    t = string_push_str(t, s);
    t = string_push_i64(t, 42);
    let mut code: i64 = string_len(t) * 10;
    code = code + string_get_byte(t, 3) - 48;
    code = code + string_eq(s, s) + string_starts_with(t, s) * 100;
    let mut long: String = string_new();
    let mut i: i64 = 0;
    while i < 20 {
        long = string_push_byte(long, 97 + i);
        i = i + 1;
    }
    code = code + string_len(long) + string_get_byte(long, 19) - 116;
    return code;
}
"#;

fn scratch(label: &str) -> PathBuf {
    let dir = common::scratch_dir("mindc_test_eval_parity").join(label);
    fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Run `mindc test` on `program` plus a `parity` test asserting
/// `score() == expected`; returns `(exit code, output)`.
fn mindc_test(stem: &str, program: &str, expected: i64) -> (i32, String) {
    let path = scratch(&format!("{stem}-expect{expected}")).join(format!("{stem}.mind"));
    let src = format!(
        "{program}\n#[test]\nfn parity() {{\n    let got: i64 = score();\n    \
         assert got == {expected}, \"score\";\n}}\n"
    );
    fs::write(&path, src).expect("write test source");
    let out = Command::new(common::require_mindc())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    (out.status.code().unwrap_or(-1), text(&out))
}

/// The interpreter leg: the right score passes, a wrong one fails.
fn assert_interpreter(stem: &str, program: &str, expected: i64) {
    let (code, out) = mindc_test(stem, program, expected);
    assert_eq!(
        code, 0,
        "mindc test must pass with score {expected}:\n{out}"
    );
    assert!(
        out.contains(&format!("test {stem}::parity ... ok")) && out.contains("1 passed; 0 failed"),
        "expected the parity test to run and pass:\n{out}"
    );
    let (code, out) = mindc_test(stem, program, expected + 1);
    assert_ne!(code, 0, "a wrong expected score must fail:\n{out}");
    assert!(
        out.contains(&format!("test {stem}::parity ... FAILED")),
        "expected the wrong-score control to fail its assert:\n{out}"
    );
}

/// The native leg: build `program` with a `main` returning `score()`, run the
/// artifact, and compare its exit status with `expected`.
fn assert_native(stem: &str, program: &str, expected: i64) {
    assert!((0..256).contains(&expected), "an exit status holds 0..=255");
    let dir = scratch(&format!("{stem}-native"));
    fs::create_dir_all(dir.join(".git")).expect("project boundary");
    fs::write(
        dir.join("Mind.toml"),
        format!(
            "[package]\nname = \"{stem}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("write manifest");
    fs::write(
        dir.join("main.mind"),
        format!("{program}\nfn main() -> i64 {{\n    return score();\n}}\n"),
    )
    .expect("write entry");
    let built = common::run_build_captured(&common::require_mindc(), &dir, &[]);
    if !common::gate::compiled(&format!("mindc_test_eval_parity::{stem}"), &built) {
        return;
    }
    let artifact = common::reported_artifact(&built);
    let ran = Command::new(&artifact)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e}", artifact.display()));
    assert_eq!(
        ran.status.code(),
        Some(expected as i32),
        "the compiled {stem} program must exit with score {expected}:\n{}",
        text(&ran)
    );
}

fn assert_parity(stem: &str, program: &str, expected: i64) {
    assert_interpreter(stem, program, expected);
    assert_native(stem, program, expected);
}

#[test]
fn float_conditions_agree_with_the_compiled_build() {
    assert_parity("float_conditions", FLOAT_CONDITIONS, 159);
}

#[test]
fn field_store_reaches_the_one_record_like_the_compiled_build() {
    assert_parity("field_store_identity", FIELD_STORE_IDENTITY, 182);
}

#[test]
fn field_store_in_loop_bodies_agrees_with_the_compiled_build() {
    assert_parity("field_store_in_loops", FIELD_STORE_IN_LOOPS, 65);
}

#[test]
fn field_store_through_an_array_element_agrees_with_the_compiled_build() {
    assert_parity("field_store_array_element", FIELD_STORE_ARRAY_ELEMENT, 42);
}

#[test]
fn fixed_array_field_store_agrees_with_the_compiled_build() {
    assert_parity("fixed_array_field_store", FIXED_ARRAY_FIELD_STORE, 202);
}

#[test]
fn break_and_continue_agree_with_the_compiled_build() {
    assert_parity("break_continue", BREAK_CONTINUE, 34);
}

#[test]
fn for_each_agrees_with_the_compiled_build() {
    assert_parity("for_each", FOR_EACH, 51);
}

#[test]
fn ambient_string_builders_agree_with_the_compiled_build() {
    assert_parity("string_builders", STRING_BUILDERS, 163);
}

/// Interpreter only: the compiled build refuses an all-constant literal as a
/// for-each collection at MLIR lowering (`non-i64 argument to call vec_len`),
/// so there is no native result to compare with.
#[test]
fn for_each_over_a_constant_literal_runs() {
    let program = "fn score() -> i64 {\n    let mut s: i64 = 0;\n    for v in [1, 2, 3] {\n        \
                   s = s + v;\n    }\n    return s;\n}\n";
    assert_interpreter("for_each_const_literal", program, 6);
}

/// The compiled loop keeps reading the handle it started with when the body
/// rebinds the collection variable; the interpreter cannot model that, so it
/// refuses. Iterating a copy is the positive control.
#[test]
fn for_each_body_rebinding_its_collection_is_refused() {
    let refused = "fn score() -> i64 {\n    let mut xs: array<i64> = [1, 2];\n    let mut s: i64 = 0;\n    \
                   for v in xs {\n        xs = [7, 7];\n        s = s + v;\n    }\n    return s;\n}\n";
    let (code, out) = mindc_test("for_each_rebind", refused, 3);
    assert_ne!(
        code, 0,
        "a for-each that rebinds its collection must not pass:\n{out}"
    );
    assert!(
        out.contains("for-each body reassigns its collection `xs`"),
        "expected the rebinding refusal:\n{out}"
    );
    let copy = "fn score() -> i64 {\n    let mut xs: array<i64> = [1, 2];\n    let ys: array<i64> = xs;\n    \
                let mut s: i64 = 0;\n    for v in ys {\n        xs = [7, 7];\n        s = s + v;\n    }\n    \
                return s + xs[0];\n}\n";
    assert_interpreter("for_each_copy", copy, 10);
}

/// A `break` with no loop in its own function is an error there; it must not
/// stop the loop of the caller.
#[test]
fn break_does_not_cross_a_function_boundary() {
    let program = "fn stop() -> i64 {\n    break;\n    return 0;\n}\n\nfn score() -> i64 {\n    \
                   let mut n: i64 = 0;\n    while n < 3 {\n        n = n + 1;\n        stop();\n    }\n    \
                   return n;\n}\n";
    let (code, out) = mindc_test("break_in_callee", program, 3);
    assert_ne!(
        code, 0,
        "a `break` outside any loop of its fn must fail:\n{out}"
    );
    assert!(
        out.contains("`break` outside of a loop"),
        "expected the fenced-break error:\n{out}"
    );
}
