// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! Range-`for` hygiene (audit #4 / #5i) — RUNTIME gate for the Rust-compiled
//! substrate.
//!
//! `for VAR in START..END { BODY }` used to desugar unconditionally to
//! `let VAR = START; while VAR < END { BODY; VAR = VAR + 1 }`. That form has
//! two observable divergences from the interpreter oracle
//! (`src/eval/mod.rs` For arm, which evaluates the range bound EXACTLY ONCE and
//! binds `VAR` FRESH each iteration):
//!
//!  1. `END` was re-lowered into the `while` condition and re-evaluated EVERY
//!     iteration — a call or a body-mutated bound changed the trip count.
//!  2. `VAR` was bound under its own name and escaped the loop, clobbering a
//!     shadowed outer binding; a body write `VAR = …` corrupted the counter.
//!
//! The `For` arm in `src/eval/lower.rs` now branches on a hygiene gate:
//! `env.contains_key(var) || body_assigns(var) || end_has_call || end_reads_body_assigned`.
//! Gate OFF keeps the old desugar BYTE-FOR-BYTE (keystone + cross-substrate
//! canaries prove zero drift); gate ON emits a hygienic form (span-unique
//! counter, `END` pre-lowered ONCE, per-iteration `let mut VAR` copy).
//!
//! This test compiles the four gated shapes through the Rust `--emit-shared`
//! backend, RUNS each exported function, and asserts the result equals an
//! independent hand-computed reference (the third, self-host native-ELF
//! substrate leg + its Step-2 xfail bookkeeping lives in
//! `examples/mindc_mind/self_host_for_smoke.py`).
//!
//! NOTE on the "interpreter" leg: `mindc` exposes no interpreter subcommand —
//! both `mindc run` and `mindc build` compile through this same Rust backend
//! (`run_project`). The independent oracle here is therefore the hand-computed
//! reference; the Rust-compiled ELF is the fix-under-test. `end_reads_body`
//! would HANG on the pre-fix compiler (END `h` re-evaluated as it grows), so
//! the whole run is under `timeout` — a regression fails LOUD (rc=124) instead
//! of stalling CI.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test for_hygiene_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// Each function is a gated shape. Hand-computed references:
//  * shadow_outer   — `let i = 100; for i in 0..3 {} return i` — VAR must not
//                     escape; outer `i` stays 100.
//  * body_assign    — `for i in 0..3 { i = i + 5; c = c + 1 }` — the body write
//                     hits a per-iteration copy, the counter is untouched, so
//                     the loop runs its full 3 iterations -> c == 3.
//  * end_reads_body — `let mut h = 3; for i in 0..h { h = h + 1; c = c + 1 }` —
//                     the bound is frozen at its once-evaluated 3 -> c == 3
//                     (this HANGS on the pre-fix compiler).
//  * end_call       — `for i in 0..hi()` — the call bound is pre-lowered once
//                     -> c == 3.
const SRC: &str = r#"
fn shadow_outer() -> i64 {
    let i = 100;
    for i in 0..3 { }
    return i;
}
fn body_assign() -> i64 {
    let mut c = 0;
    for i in 0..3 {
        i = i + 5;
        c = c + 1;
    }
    return c;
}
fn end_reads_body() -> i64 {
    let mut h = 3;
    let mut c = 0;
    for i in 0..h {
        h = h + 1;
        c = c + 1;
    }
    return c;
}
fn hi() -> i64 { return 3; }
fn end_call() -> i64 {
    let mut c = 0;
    for i in 0..hi() {
        c = c + 1;
    }
    return c;
}
fn main() -> i64 { return 0; }
"#;

#[test]
fn range_for_hygiene_runs_correctly() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("for-hygiene-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_for_hygiene_run.mind");
    let so = dir.join("mind_for_hygiene_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("for-hygiene-run: needs mlir-build; skipping");
            return;
        }
        panic!("for-hygiene-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         def f(name):\n\
         \x20   fn = getattr(lib, name); fn.restype = ctypes.c_int64\n\
         \x20   fn.argtypes = []; return fn\n\
         assert f('shadow_outer')()   == 100, 'shadow_outer='  +str(f('shadow_outer')())\n\
         assert f('body_assign')()    == 3,   'body_assign='   +str(f('body_assign')())\n\
         assert f('end_reads_body')() == 3,   'end_reads_body='+str(f('end_reads_body')())\n\
         assert f('end_call')()       == 3,   'end_call='      +str(f('end_call')())\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    // `timeout` kills the end_reads_body hang a pre-fix compiler would produce
    // (rc=124) instead of stalling CI.
    let out = Command::new("timeout")
        .args(["60", "python3", "-c", &py])
        .output()
        .expect("timeout python3");
    assert!(
        out.status.success(),
        "for-hygiene-run check failed (rc={:?}; 124=HANG regression):\n\
         stdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
