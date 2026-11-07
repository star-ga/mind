// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc test` runs two more constructs the way the compiled build runs them:
//! an `array<T>` slot that holds a `vec_zeroed` buffer, and `Ok(..)` / `Err(..)`
//! values.
//!
//! **Zeroed arrays.** The slot's declared `array<T>` type makes it a std.vec
//! handle, so an element store (`a[i] = v`, `b.xs[i] = v`), `a.len()` and a
//! `&mut [T]` argument all reach ONE shared backing: the callee's stores are
//! the caller's, and an alias copied before the store sees it. Under
//! `mindc test` the buffer used to be an opaque integer (`indexed assignment
//! receiver `a` must be an array`, `no method .len() for Int(8)`).
//!
//! The compiled build cannot link `vec_zeroed` (the runtime support exports
//! `__mind_vec_zeroed` only; `undefined reference to `vec_zeroed``), so a
//! program written with `ZEROED(n)` runs under `mindc test` as `vec_zeroed(n)`
//! and is BUILT with `n` zero literals, the std.vec handle the build passes for
//! an `array<T>` slot. Both legs are held to one expected score.
//!
//! **Results.** `Ok(..)` / `Err(..)` resolve to `Result::Ok` / `Result::Err`
//! in the build, whichever spelling constructs or matches them. Constructing
//! one under `mindc test` used to stop at `unsupported operation`.
//!
//! Each case is ONE program with a `fn score() -> i64`. The interpreter leg runs
//! it under `mindc test` with the expected score; the native leg builds it with
//! `fn main() -> i64 { return score(); }`, runs the artifact and reads the exit
//! status. Every interpreter leg also runs with a WRONG expected value and must
//! FAIL: a runner that passes everything cannot pass this file.
//!
//! The native leg needs the MLIR backend; without it the step is a coded
//! capability skip ([`common::gate`]), which `MIND_BENCH_REQUIRE=1` turns into a
//! failure.
//!
//! Gate: `cargo test --release --no-default-features
//!        --features mlir-build,std-surface,cross-module-imports
//!        --test mindc_test_eval_arrays_results_parity`

#![cfg(feature = "std-surface")]

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// The issue's reproduction: an element store, a read-back and `.len()`.
const INDEXED_STORE_AND_LEN: &str = r#"
fn score() -> i64 {
    let mut a: array<i32> = ZEROED(2);
    a[0] = 7;
    return (a[0] as i64) * 10 + (a.len() as i64);
}
"#;

/// A store in a loop body, and two buffers that stay independent.
const LOOP_STORES_AND_TWO_BUFFERS: &str = r#"
fn score() -> i64 {
    let mut a: array<i64> = ZEROED(4);
    let mut b: array<i64> = ZEROED(2);
    let mut i: i64 = 0;
    while i < 4 {
        a[i] = i * 3;
        i = i + 1;
    }
    b[0] = 5;
    return a[0] + a[1] + a[2] + a[3] + b[0] * 10 + b[1];
}
"#;

/// A callee's stores through a `&mut [T]` parameter are the caller's, one hop
/// and two hops down.
const MUT_SLICE_WRITE_BACK: &str = r#"
fn fill(xs: &mut [i32]) -> i64 {
    xs[0] = 5;
    xs[1] = 6;
    return xs.len() as i64;
}

fn bump(xs: &mut [i32]) -> i64 {
    xs[1] = xs[1] + 4;
    return 0;
}

fn outer(xs: &mut [i32]) -> i64 {
    xs[0] = 3;
    return bump(xs);
}

fn score() -> i64 {
    let mut a: array<i32> = ZEROED(2);
    let n: i64 = fill(a);
    let mut c: array<i32> = ZEROED(2);
    let z: i64 = outer(c);
    return (a[0] as i64) * 20 + (a[1] as i64) * 3 + n + (c[0] as i64) * 10 + (c[1] as i64) + z;
}
"#;

/// A handle copy shares the backing: an alias made before the store sees it.
const ALIAS_SHARES_THE_BACKING: &str = r#"
fn score() -> i64 {
    let mut a: array<i32> = ZEROED(2);
    let c: array<i32> = a;
    a[0] = 9;
    return (c[0] as i64) * 10 + (a.len() as i64);
}
"#;

/// A buffer returned from a fn keeps its stores and its length.
const RETURN_FROM_FN: &str = r#"
fn make() -> array<i32> {
    let mut a: array<i32> = ZEROED(2);
    a[1] = 6;
    return a;
}

fn score() -> i64 {
    let a: array<i32> = make();
    return (a[1] as i64) * 10 + (a.len() as i64);
}
"#;

/// A struct field declared `array<T>` holds the buffer: an element store
/// through the field, `.len()`, and the field passed on as a `&mut [T]`.
const STRUCT_FIELD_ELEMENT_STORE: &str = r#"
struct B {
    xs: array<i32>,
}

fn fill(xs: &mut [i32]) -> i64 {
    xs[2] = 8;
    return 0;
}

fn score() -> i64 {
    let mut b: B = B { xs: ZEROED(3) };
    b.xs[1] = 4;
    let z: i64 = fill(b.xs);
    return (b.xs[1] as i64) * 10 + (b.xs.len() as i64) + (b.xs[2] as i64) * 20 + z;
}
"#;

/// `Ok(..)` / `Err(..)` constructed and matched in the bare spelling.
const RESULT_BARE: &str = r#"
fn v(x: i64) -> Result<i64, i64> {
    if x > 0 {
        return Ok(x);
    }
    return Err(9);
}

fn unwrap_or(r: Result<i64, i64>) -> i64 {
    return match r {
        Ok(x) => x,
        Err(e) => e + 100,
    };
}

fn score() -> i64 {
    return unwrap_or(v(3)) + unwrap_or(v(0));
}
"#;

/// The three ways to cross the bare and qualified spellings of one variant.
const RESULT_SPELLINGS: &str = r#"
fn bare(x: i64) -> Result<i64, i64> {
    if x > 0 {
        return Ok(x);
    }
    return Err(9);
}

fn qualified(x: i64) -> Result<i64, i64> {
    if x > 0 {
        return Result::Ok(x);
    }
    return Result::Err(9);
}

fn read_bare(r: Result<i64, i64>) -> i64 {
    return match r {
        Ok(x) => x,
        Err(e) => e + 100,
    };
}

fn read_qualified(r: Result<i64, i64>) -> i64 {
    return match r {
        Result::Ok(x) => x,
        Result::Err(e) => e + 100,
    };
}

fn score() -> i64 {
    let a: i64 = read_qualified(bare(3)) + read_qualified(bare(0));
    let b: i64 = read_bare(qualified(3)) + read_bare(qualified(0));
    return a - b + 112;
}
"#;

/// A `match` over a `Result` in statement position: the taken arm's assignment
/// to an outer name survives the `match`.
const RESULT_MATCH_ASSIGNS: &str = r#"
fn v(ok: i64) -> Result<i64, i64> {
    if ok > 0 {
        return Ok(3);
    }
    return Err(7);
}

fn score() -> i64 {
    let mut s: i64 = 1;
    match v(1) {
        Ok(x) => {
            s = s + x;
        },
        Err(e) => {
            s = s + 50;
        },
    }
    match v(0) {
        Ok(x) => {
            s = s + x * 100;
        },
        Err(e) => {
            s = s + e * 10;
        },
    }
    return s;
}
"#;

/// A payload computed from the call, a unit payload, and a `Result` passed on
/// through a second fn.
const RESULT_PAYLOADS: &str = r#"
fn pair(x: i64) -> Result<i64, i64> {
    if x > 1 {
        return Ok(x * 2);
    }
    return Err(x);
}

fn chain(x: i64) -> i64 {
    let r: Result<i64, i64> = pair(x);
    return match r {
        Ok(v) => v + 1,
        Err(e) => e + 40,
    };
}

fn check(x: i64) -> Result<(), i64> {
    if x > 0 {
        return Ok(());
    }
    return Err(x + 20);
}

fn unit_code(x: i64) -> i64 {
    return match check(x) {
        Ok(u) => 1,
        Err(e) => e,
    };
}

fn score() -> i64 {
    return chain(5) + chain(1) + unit_code(1) + unit_code(0);
}
"#;

/// Replace each `ZEROED(n)` with `with(n)`.
fn expand_zeroed(program: &str, with: impl Fn(usize) -> String) -> String {
    let mut out = String::new();
    let mut rest = program;
    while let Some(at) = rest.find("ZEROED(") {
        out.push_str(&rest[..at]);
        let after = &rest[at + "ZEROED(".len()..];
        let close = after.find(')').expect("ZEROED( closes");
        let n: usize = after[..close].parse().expect("ZEROED(n) takes a count");
        out.push_str(&with(n));
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

/// What `mindc test` runs: the buffer is `vec_zeroed(n)`.
fn eval_form(program: &str) -> String {
    expand_zeroed(program, |n| format!("vec_zeroed({n})"))
}

/// What the build compiles: the same buffer as `n` zero literals.
fn build_form(program: &str) -> String {
    expand_zeroed(program, |n| format!("[{}]", vec!["0"; n].join(", ")))
}

fn scratch(label: &str) -> PathBuf {
    let dir = common::scratch_dir("mindc_test_eval_arrays_results_parity").join(label);
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
        "{}\n#[test]\nfn parity() {{\n    let got: i64 = score();\n    \
         assert got == {expected}, \"score\";\n}}\n",
        eval_form(program)
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
        format!(
            "{}\nfn main() -> i64 {{\n    return score();\n}}\n",
            build_form(program)
        ),
    )
    .expect("write entry");
    let built = common::run_build_captured(&common::require_mindc(), &dir, &[]);
    if !common::gate::compiled(
        &format!("mindc_test_eval_arrays_results_parity::{stem}"),
        &built,
    ) {
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
fn zeroed_array_store_and_len_agree_with_the_compiled_build() {
    assert_parity("zeroed_store_len", INDEXED_STORE_AND_LEN, 72);
}

#[test]
fn zeroed_array_loop_stores_and_independent_buffers_agree() {
    assert_parity("zeroed_loop_stores", LOOP_STORES_AND_TWO_BUFFERS, 68);
}

#[test]
fn zeroed_array_write_back_through_mut_slice_agrees() {
    assert_parity("zeroed_write_back", MUT_SLICE_WRITE_BACK, 154);
}

#[test]
fn zeroed_array_alias_shares_the_backing() {
    assert_parity("zeroed_alias", ALIAS_SHARES_THE_BACKING, 92);
}

#[test]
fn zeroed_array_returned_from_a_fn_agrees() {
    assert_parity("zeroed_return", RETURN_FROM_FN, 62);
}

#[test]
fn zeroed_array_struct_field_element_store_agrees() {
    assert_parity("zeroed_struct_field", STRUCT_FIELD_ELEMENT_STORE, 203);
}

#[test]
fn result_bare_construct_and_match_agree() {
    assert_parity("result_bare", RESULT_BARE, 112);
}

#[test]
fn result_bare_and_qualified_spellings_are_one_variant() {
    assert_parity("result_spellings", RESULT_SPELLINGS, 112);
}

#[test]
fn result_match_arm_assignments_survive_the_match() {
    assert_parity("result_match_assigns", RESULT_MATCH_ASSIGNS, 74);
}

#[test]
fn result_payloads_agree() {
    assert_parity("result_payloads", RESULT_PAYLOADS, 73);
}

/// Interpreter only: the build cannot link `vec_zeroed`, and it refuses to
/// rebind an `array<T>` variable to an array literal (`non-i64 argument to call
/// `vec_set``), so a buffer returned or passed straight on, or taken by a
/// reassignment, has no native twin. The slot rule is the one the parity cases
/// above hold to the build. `make().len()` reads the returned value with no
/// slot of its own, and `size(vec_zeroed(3))` is taken by the parameter alone.
#[test]
fn zeroed_buffer_taken_directly_by_a_return_or_an_argument() {
    let program = "fn make() -> array<i32> {\n    return vec_zeroed(2);\n}\n\n\
                   fn size(xs: &[i32]) -> i64 {\n    return xs.len() as i64;\n}\n\n\
                   fn score() -> i64 {\n    let a: array<i32> = make();\n    \
                   return size(a) * 10 + size(vec_zeroed(3)) + (make().len() as i64);\n}\n";
    assert_interpreter("zeroed_direct", program, 25);
}

#[test]
fn zeroed_buffer_taken_by_a_reassignment() {
    let program = "fn score() -> i64 {\n    let mut a: array<i32> = vec_zeroed(2);\n    \
                   a[0] = 4;\n    a = vec_zeroed(3);\n    a[2] = 5;\n    \
                   return (a[0] as i64) * 100 + (a[2] as i64) * 10 + (a.len() as i64);\n}\n";
    assert_interpreter("zeroed_reassign", program, 53);
}

/// A zeroed array is bounds-checked: an index past its length fails the test
/// instead of reading a neighbouring cell, and a slot declared `i64` keeps the
/// plain integer the arena routines address.
#[test]
fn zeroed_buffer_bounds_and_integer_slot() {
    let past = "fn score() -> i64 {\n    let a: array<i32> = vec_zeroed(2);\n    \
                return a[2] as i64;\n}\n";
    let (code, out) = mindc_test("zeroed_past_end", past, 0);
    assert_ne!(code, 0, "an index past the length must fail:\n{out}");
    assert!(
        out.contains("out of bounds"),
        "expected the bounds error:\n{out}"
    );
    let raw = "fn score() -> i64 {\n    let z: i64 = vec_zeroed(2);\n    \
               __mind_store_i64(z + 8, 5);\n    return __mind_load_i64(z + 8) + __mind_load_i64(z);\n}\n";
    let (code, out) = mindc_test("zeroed_integer_slot", raw, 5);
    assert_eq!(code, 0, "an i64 slot keeps the arena address:\n{out}");
}

/// A test fn that returns `Err(..)` fails with its payload; one that returns
/// `Ok(())` passes (RFC 0008 section 5.1).
#[test]
fn a_test_returning_a_result_is_graded_by_its_variant() {
    let dir = scratch("test_returns_result");
    let path = dir.join("graded.mind");
    fs::write(
        &path,
        "#[test]\nfn passes() -> Result<(), i64> {\n    return Ok(());\n}\n\n\
         #[test]\nfn fails() -> Result<(), i64> {\n    return Err(41);\n}\n",
    )
    .expect("write test source");
    let out = Command::new(common::require_mindc())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    let log = text(&out);
    assert_ne!(
        out.status.code(),
        Some(0),
        "an Err result must fail:\n{log}"
    );
    assert!(
        log.contains("test graded::passes ... ok") && log.contains("test graded::fails ... FAILED"),
        "expected one pass and one fail:\n{log}"
    );
    assert!(
        log.contains("test returned Err(41)"),
        "expected the Err payload in the failure:\n{log}"
    );
}
