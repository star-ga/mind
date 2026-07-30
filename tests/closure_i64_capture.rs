// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Closures #267 Phase 1 — i64-capture, direct-apply RUNTIME differential gate.
//!
//! A single-param closure that captures one `i64` by value, bound via
//! `let f = |k; x: i64| -> i64 { x + k }` and applied by a DIRECT call `f(5)` in
//! the same fn scope, must desugar (BEFORE IR emit) into an env struct + a
//! top-level fn + a struct literal, monomorphized to a direct call
//! `__cl_{span}(env, 5)` — and RUN correct: `f(5)` with `k = 10` is `15`.
//!
//! The capture-rewrite and closure-call rewriters walk an EXHAUSTIVE child
//! visitor, so a captured ident inside a `match` body (`run_match` /
//! `run_shadow_match`) and a direct call `f(i)` inside a `while` body
//! (`run_while`) are rewritten too — the shapes a single-`Binary`-body test
//! misses. `run_shadow_match` is the reviewer's silent-miscompile reproducer: a
//! module `const k = 999` colliding with the capture `k`, which a non-exhaustive
//! walker left resolving to `999` instead of the captured `10`.
//!
//! Also witnesses that ordinary (non-closure) code in the same module is
//! unaffected — `plain(3, 4) == 7`.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test closure_i64_capture`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// #267 Phase 1 examples (narrowed to DIRECT apply — `apply(f, x)` indirection is
// deferred to Phase 2). A module-level `const k = 999` deliberately collides with
// the capture name `k` to prove the capture (10) wins over the global (999).
const SRC: &str = r#"
const k: i64 = 999

// Canonical: `k = 10`, `f(5) = 5 + 10 = 15`.
pub fn run() -> i64 {
    let k: i64 = 10
    let f = |k; x: i64| -> i64 { x + k }
    return f(5)
}

// Capture used inside a `match` body — the exhaustive walker must rewrite `k`
// to `env.k` here too.
pub fn run_match() -> i64 {
    let k: i64 = 10
    let f = |k; x: i64| -> i64 { match x { _ => x + k } }
    return f(5)
}

// Reviewer reproducer: `match x { _ => k }` with a colliding global `const k`.
// A non-exhaustive walker left `k` as the global 999; the fix yields 10.
pub fn run_shadow_match() -> i64 {
    let k: i64 = 10
    let f = |k; x: i64| -> i64 { match x { _ => k } }
    return f(5)
}

// Direct call `f(i)` inside a `while` body — the exhaustive walker must rewrite
// it to `__cl_{span}(f, i)`. f(0)+f(1)+f(2) = 10 + 11 + 12 = 33.
pub fn run_while() -> i64 {
    let k: i64 = 10
    let f = |k; x: i64| -> i64 { x + k }
    let mut acc: i64 = 0
    let mut i: i64 = 0
    while i < 3 {
        acc = acc + f(i)
        i = i + 1
    }
    return acc
}

// Witness: a plain fn with no closure lowers exactly as before.
pub fn plain(a: i64, b: i64) -> i64 {
    return a + b
}
"#;

// Multi-capture is a Phase-2 shape; Phase 1 must REJECT it fail-closed (E2600),
// never silently compile an untested arity.
const MULTI_CAPTURE_SRC: &str = r#"
pub fn run() -> i64 {
    let a: i64 = 1
    let b: i64 = 2
    let f = |a, b; x: i64| -> i64 { x + a + b }
    return f(5)
}
"#;

#[test]
fn closure_i64_capture_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("closure-i64-capture: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_closure_i64_capture.mind");
    let so = dir.join("mind_closure_i64_capture.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("closure-i64-capture: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("closure-i64-capture: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for name in ('run','run_match','run_shadow_match','run_while'):\n\
         \x20   fn = getattr(lib, name); fn.restype = ctypes.c_int64; fn.argtypes = []\n\
         lib.plain.restype = ctypes.c_int64; lib.plain.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         r = lib.run();              assert r == 15, 'run()=' + str(r) + ' (want 15)'\n\
         r = lib.run_match();        assert r == 15, 'run_match()=' + str(r) + ' (want 15)'\n\
         r = lib.run_shadow_match(); assert r == 10, 'run_shadow_match()=' + str(r) + ' (want 10; 999 = capture NOT rewritten)'\n\
         r = lib.run_while();        assert r == 33, 'run_while()=' + str(r) + ' (want 33)'\n\
         r = lib.plain(3, 4);        assert r == 7,  'plain(3,4)=' + str(r) + ' (want 7)'\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "closure-i64-capture value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn closure_multi_capture_rejected() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("closure-multi-capture-reject: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_closure_multi_capture.mind");
    let so = dir.join("mind_closure_multi_capture.so");
    std::fs::write(&src, MULTI_CAPTURE_SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("mlir-build") && stderr.contains("requires") {
        println!("closure-multi-capture-reject: mindc --emit-shared needs mlir-build; skipping");
        return;
    }
    assert!(
        !out.status.success(),
        "multi-capture closure must be rejected fail-closed, but mindc succeeded"
    );
    assert!(
        stderr.contains("E2600") || stderr.contains("closure"),
        "expected an E2600 closure diagnostic, got:\n{stderr}"
    );
}
