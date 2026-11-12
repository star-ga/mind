// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc test` runs five more constructs the way the compiled build runs
//! them — the silent-wrong-result gaps this PR closes. Each program's
//! expected value is the NATIVE build's measured result (the reference;
//! never re-derived from the interpreter):
//!
//! * a module-level `for v in xs` threads its loop-carried stores (the loop
//!   used to run over a clone of the env and drop them, so an
//!   `assert total == 0` passed against a loop that had summed 6);
//! * `b.xs[i] = v` on a field declared `array<T>` writes the SHARED element
//!   run (the compiled build stores a std.vec handle there, and an alias
//!   copied out before the store sees the write: 20*10+20, not 20*10+2);
//! * `for v in b.xs` reads its elements through that same shared run each
//!   iteration, so a store ahead of the loop position is read back (12);
//! * a struct `const` is re-materialised FRESH at every use (the compiled
//!   build inlines the literal, so two uses never share one record: 121,
//!   not a leaked 131);
//! * an ambient std fn that the runtime support does NOT link (`vec_free`)
//!   is refused, the way the compiled build fails to link it — ambient
//!   admission is exactly the linkable surface.
//!
//! Each positive case is ONE program with a `fn score() -> i64`. The
//! interpreter leg runs it under `mindc test` with the expected score; the
//! native leg builds the same source with `fn main() -> i64 { return
//! score(); }`, runs the artifact, and reads the exit status. Every
//! interpreter leg also runs with a WRONG expected value and must FAIL: a
//! runner that passes everything cannot pass this file.
//!
//! The native leg needs the MLIR backend; without it the step is a coded
//! capability skip ([`common::gate`]), which `MIND_BENCH_REQUIRE=1` turns
//! into a failure.
//!
//! Gate: `cargo test --release --test mindc_test_eval_blockers_parity`

#![cfg(feature = "std-surface")]

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// A module-level `for v in xs` must thread its loop-carried stores: after
/// the loop `total` is the SUM (6), so a test asserting the stale pre-loop
/// value (0) fails. The loop used to run over a clone of the env, which made
/// the stale assert pass — the silent wrong result this pins shut.
const MODULE_LEVEL_FOR_EACH: &str = r#"
let xs: array<i64> = [1, 2, 3];
let mut total: i64 = 0;
for v in xs {
    total = total + v;
}
"#;

/// `b.xs[i] = v` on a field declared `array<i64>` goes through the shared
/// element run, so `before` — an alias copied out BEFORE the store — reads
/// the stored element too: 20*10 + 20.
const FIELD_ELEMENT_STORE_SHARES: &str = r#"
struct B {
    xs: array<i64>,
}

fn score() -> i64 {
    let mut b: B = B { xs: [1, 2, 3] };
    let before: array<i64> = b.xs;
    b.xs[1] = 20;
    return b.xs[1] * 10 + before[1];
}
"#;

/// `for v in b.xs` reads each element through the field's shared run at the
/// top of the iteration, so `b.xs[2] = 10` (ahead of the loop position) is
/// read back: 1 + 1 + 10.
const FOR_EACH_FIELD_STORE_AHEAD: &str = r#"
struct B {
    xs: array<i64>,
}

fn score() -> i64 {
    let mut b: B = B { xs: [1, 1, 1] };
    let mut s: i64 = 0;
    for v in b.xs {
        b.xs[2] = 10;
        s = s + v;
    }
    return s;
}
"#;

/// A struct `const` re-materialises at every use: each `bump()` starts from
/// a fresh record, so both calls answer 11 (11*10 + 11). One shared record
/// leaked the first bump's store into the second call (11*10 + 21).
const STRUCT_CONST_FRESH_PER_USE: &str = r#"
struct P {
    x: i64,
}

const ORIGIN: P = P { x: 1 };

fn bump() -> i64 {
    let p: P = ORIGIN;
    p.x = p.x + 10;
    return p.x;
}

fn score() -> i64 {
    let a: i64 = bump();
    let b: i64 = bump();
    return a * 10 + b;
}
"#;

/// `vec_free` is a std.vec fn the runtime support does NOT link: calling it
/// with no import must be refused, exactly as the compiled build fails to
/// link it. The `vec_len` control is a linkable ambient name and passes.
const AMBIENT_VEC_FREE: &str = r#"
fn score() -> i64 {
    let v: Vec = vec_new();
    return vec_free(v) + 5;
}
"#;

const AMBIENT_VEC_LEN_CONTROL: &str = r#"
fn score() -> i64 {
    let v: Vec = vec_new();
    return vec_len(v) + 5;
}
"#;

fn scratch(label: &str) -> PathBuf {
    let dir = common::scratch_dir("mindc_test_eval_blockers_parity").join(label);
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

/// Run `mindc test` on a MODULE-LEVEL program plus a `parity` test asserting
/// `total == expected`.
fn mindc_test_module_total(stem: &str, program: &str, expected: i64) -> (i32, String) {
    let path = scratch(&format!("{stem}-expect{expected}")).join(format!("{stem}.mind"));
    let src = format!(
        "{program}\n#[test]\nfn parity() {{\n    assert total == {expected}, \
         \"total\";\n}}\n"
    );
    fs::write(&path, src).expect("write test source");
    let out = Command::new(common::require_mindc())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    (out.status.code().unwrap_or(-1), text(&out))
}

/// The interpreter leg for a score-based case: the right score passes, a
/// wrong one fails.
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
    if !common::gate::compiled(&format!("mindc_test_eval_blockers_parity::{stem}"), &built) {
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
fn module_level_for_each_threads_its_loop_stores() {
    // The module-level program's native reference: the same loop inside a
    // `main` exits with the summed total.
    let native_program = "fn score() -> i64 {\n    let xs: array<i64> = [1, 2, 3];\n    \
                           let mut total: i64 = 0;\n    for v in xs {\n        total = total + \
                           v;\n    }\n    return total;\n}\n";
    let (code, out) = mindc_test_module_total("module_foreach", MODULE_LEVEL_FOR_EACH, 6);
    assert_eq!(
        code, 0,
        "a module-level for-each must thread its stores (total == 6):\n{out}"
    );
    assert!(
        out.contains("test module_foreach::parity ... ok"),
        "expected the parity test to run and pass:\n{out}"
    );
    // The stale pre-loop value must FAIL — on the old env-clone path this
    // control passed, which is exactly the silent wrong result.
    let (code, out) = mindc_test_module_total("module_foreach", MODULE_LEVEL_FOR_EACH, 0);
    assert_ne!(
        code, 0,
        "asserting the stale pre-loop total (0) must fail:\n{out}"
    );
    assert!(
        out.contains("test module_foreach::parity ... FAILED"),
        "expected the stale-total control to fail its assert:\n{out}"
    );
    assert_native("module_foreach", native_program, 6);
}

#[test]
fn field_element_store_shares_the_dynamic_array_field() {
    assert_parity("field_element_store", FIELD_ELEMENT_STORE_SHARES, 220);
}

#[test]
fn for_each_over_a_field_collection_reads_body_stores() {
    assert_parity("for_each_field_store", FOR_EACH_FIELD_STORE_AHEAD, 12);
}

#[test]
fn struct_const_materialises_fresh_at_every_use() {
    assert_parity("struct_const_fresh", STRUCT_CONST_FRESH_PER_USE, 121);
}

#[test]
fn ambient_std_admits_only_what_runtime_support_links() {
    // The unlinkable name is refused under `mindc test` — the compiled build
    // cannot link it, and the interpreter must not pass what it cannot.
    let (code, out) = mindc_test("ambient_vec_free", AMBIENT_VEC_FREE, 5);
    assert_ne!(
        code, 0,
        "vec_free is not in the runtime support the compiled build links; \
         mindc test must refuse it:\n{out}"
    );
    assert!(
        out.contains("unsupported operation"),
        "expected the same refusal the unlinked call gives:\n{out}"
    );
    // The linkable control keeps passing: ambient admission is the linkable
    // surface, not nothing.
    assert_interpreter("ambient_vec_len", AMBIENT_VEC_LEN_CONTROL, 5);
    // Native leg: the artifact must FAIL TO LINK with the missing symbol. An
    // executable names a MIND function `mind.<name>` at the object level.
    let dir = scratch("ambient-vec-free-native");
    fs::create_dir_all(dir.join(".git")).expect("project boundary");
    fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"ambient_vec_free\"\nversion = \"0.1.0\"\n\n[build]\nentry = \
         \"main.mind\"\n",
    )
    .expect("write manifest");
    fs::write(
        dir.join("main.mind"),
        format!("{AMBIENT_VEC_FREE}\nfn main() -> i64 {{\n    return score();\n}}\n"),
    )
    .expect("write entry");
    let built = common::run_build_captured(&common::require_mindc(), &dir, &[]);
    let log = text(&built);
    if built.status.success() {
        panic!("the compiled build must not link an ambient-only vec_free call:\n{log}");
    }
    if common::gate::is_capability_gap(&String::from_utf8_lossy(&built.stderr)) {
        return;
    }
    assert!(
        log.contains("undefined reference to `mind.vec_free'"),
        "expected the linker's missing-symbol refusal:\n{log}"
    );
}
