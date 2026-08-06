// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Codex PR #216 correctness regression gate — two confirmed silent-miscompile /
//! fail-to-compile bugs in the narrow-int + tuple-index lowering, plus the
//! broader-sweep coverage.
//!
//! * Finding 1 (silent miscompile) — the `while` body did NOT snapshot/restore
//!   `NARROW_LOCALS`, unlike the `Block`/`If` arms. A wide re-let inside the loop
//!   (`let c: i64 = …`) that SHADOWS an outer `u8` local permanently deleted the
//!   outer's narrow metadata (even on zero iterations), so a later assignment to
//!   the outer var after the loop emitted UNMASKED — an out-of-range value.
//!   `while_shadow` reproduces it (buggy = 300, correct = 300 & 255 == 44), and
//!   `for_shadow` proves the `For`→`While` desugar routes through the same fix.
//!
//! * Finding 2 (fail-to-compile) — a chained numeric tuple index `t.0.1` panicked
//!   because the resolver only accepted an `Ident` receiver; the outer `.1`'s
//!   receiver is the `FieldAccess` `t.0`. The resolver now recovers the receiver's
//!   tuple element TYPE recursively. `nested_tuple_*` compile + run to the correct
//!   nested element.
//!
//! Each function reproduces a bug and asserts the correct value via
//! `--emit-shared` + ctypes. RED on base (Finding 1 returns 300 not 44; Finding 2
//! fails to compile), GREEN after the fix.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test narrow_tuple_pr216_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// ── Finding 1: a wide re-let inside a WHILE body shadowing an outer u8 must NOT
//    delete the outer's narrow metadata; the post-loop `c = 300` masks to 44.
pub fn while_shadow(x0: i64) -> i64 {
    let mut c: u8 = 5;
    let mut i: i64 = 0;
    while i < 1 {
        let c: i64 = x0;
        let d: i64 = c + 1;
        i = i + 1;
    }
    c = 300;
    return c as i64;
}

// ── Finding 1 (desugar coverage): the same via a FOR loop, which desugars to the
//    same While arm — so the one guard covers every loop body.
pub fn for_shadow(x0: i64) -> i64 {
    let mut c: u8 = 5;
    for i in 0..1 {
        let c: i64 = x0 + i;
        let d: i64 = c + 1;
    }
    c = 300;
    return c as i64;
}

// ── Finding 1 control: a genuine loop-carried narrow reassignment (NOT a re-let)
//    inside the body must KEEP its mask: c starts 5, `c = c + 250` each of 2 iters
//    → (5+250)&255 = 255, then (255+250)&255 = 249.
pub fn while_carried_mask() -> i64 {
    let mut c: u8 = 5;
    let mut i: i64 = 0;
    while i < 2 {
        c = c + 250;
        i = i + 1;
    }
    return c as i64;
}

// ── Finding 2: chained numeric tuple index `t.0.1` on a nested-tuple-typed t.
pub fn nested_tuple_01() -> i64 {
    let t: ((i64, i64), i64) = ((10, 20), 30);
    return t.0.1;
}
pub fn nested_tuple_00() -> i64 {
    let t: ((i64, i64), i64) = ((10, 20), 30);
    return t.0.0;
}
pub fn nested_tuple_outer() -> i64 {
    let t: ((i64, i64), i64) = ((10, 20), 30);
    return t.1;
}
// ── Finding 2: three levels deep — t.0.1.0.
pub fn nested_tuple_deep() -> i64 {
    let t: (((i64, i64), i64), i64) = (((10, 20), 30), 40);
    return t.0.0.1;
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn narrow_tuple_pr216_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-tuple-pr216-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_narrow_tuple_pr216_run.mind");
    let so = dir.join("mind_narrow_tuple_pr216_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("narrow-tuple-pr216-run: needs mlir-build; skipping");
            return;
        }
        panic!("narrow-tuple-pr216-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         def sig(n):\n\
         \x20   f = getattr(lib, n); f.restype = ctypes.c_int64; return f\n\
         cases = [\n\
         \x20 ('while_shadow',       lambda f: f(ctypes.c_int64(1000)),  44),\n\
         \x20 ('for_shadow',         lambda f: f(ctypes.c_int64(1000)),  44),\n\
         \x20 ('while_carried_mask', lambda f: f(),                     249),\n\
         \x20 ('nested_tuple_01',    lambda f: f(),                      20),\n\
         \x20 ('nested_tuple_00',    lambda f: f(),                      10),\n\
         \x20 ('nested_tuple_outer', lambda f: f(),                      30),\n\
         \x20 ('nested_tuple_deep',  lambda f: f(),                      20),\n\
         ]\n\
         for name, call, exp in cases:\n\
         \x20   r = call(sig(name))\n\
         \x20   assert r == exp, name + '=' + str(r) + ' expected ' + str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "narrow-tuple-pr216-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
