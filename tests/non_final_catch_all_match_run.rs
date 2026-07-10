// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! Non-final catch-all match arm — RUNTIME correctness gate (audit rank 6).
//!
//! `match x { 0 => 100, _ => 200, 1 => 300 }` used to make `desugar_match_to_if`
//! bail (the non-final `_` hit the comparison-RHS `return None`), dropping the
//! whole match into the scrutinee-IGNORING sequential fallback that returns the
//! LAST arm's value for EVERY input — so `classify(0) == classify(1) ==
//! classify(7) == 300`, a silent wrong-result rc=0 `.so`. The fix truncates the
//! arm list at the first catch-all (every later arm is unreachable), lowering the
//! well-formed `0 => 100, _ => 200` and yielding the correct first-match result.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test non_final_catch_all_match_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// Non-final wildcard catch-all followed by an (unreachable) `1 => 300` arm, plus
// a normal catch-all-last function to confirm the common case is unchanged.
const SRC: &str = r#"
fn classify(x: i64) -> i64 {
    return match x { 0 => 100, _ => 200, 1 => 300 };
}
fn normal(x: i64) -> i64 {
    return match x { 0 => 100, _ => 200 };
}
fn main() -> i64 { return 0; }
"#;

#[test]
fn non_final_catch_all_match_runs_correct() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("non-final-catch-all-match-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_non_final_catch_all_match_run.mind");
    let so = dir.join("mind_non_final_catch_all_match_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("non-final-catch-all-match-run: needs mlir-build; skipping");
            return;
        }
        panic!("non-final-catch-all-match-run: mindc --emit-shared failed:\n{stderr}");
    }

    // classify: first-match wins — 0 => 100, everything else via the catch-all
    // => 200 (the unreachable `1 => 300` must NEVER be observed). normal: the
    // common catch-all-last form, unchanged.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         c = lib.classify; c.restype = ctypes.c_int64; c.argtypes = [ctypes.c_int64]\n\
         n = lib.normal;   n.restype = ctypes.c_int64; n.argtypes = [ctypes.c_int64]\n\
         for arg, exp in ((0,100),(1,200),(7,200)):\n\
         \x20   got = c(arg)\n\
         \x20   assert got == exp, 'classify('+str(arg)+'): got='+str(got)+' expected='+str(exp)\n\
         for arg, exp in ((0,100),(1,200),(7,200)):\n\
         \x20   got = n(arg)\n\
         \x20   assert got == exp, 'normal('+str(arg)+'): got='+str(got)+' expected='+str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "non-final-catch-all-match-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
