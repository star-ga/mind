// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Phase 17.4 — `const` array EXECUTABLE lowering RUNTIME gate.
//!
//! A module that declares `const NAME: [i64; N] = [...]` and indexes it
//! (`NAME[i]`) inside a loop used to fail executable lowering with
//! `missing type information for value ValueId(N) while lowering binop`: the
//! `ArrayLoad` result carried no registered type, so the enclosing `s + PSI[i]`
//! binop could not resolve its operand kind. The IR paths (`--emit-mic3` /
//! `--emit-evidence` / `mindc test`) always accepted it because they never query
//! the lowering type map — which is exactly why the gap survived under the
//! IR-shape-only `tests/std_surface_array_literals.rs`. This test compiles the
//! construct to a real `.so`, dlopen-calls it, and asserts the summed value.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test const_array_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// PSI sums to 1 - 3 + 2 - 6 + 4 + 18 = 16. A second table (negatives + a
// duplicate index pattern) guards the index_cast + tensor.extract element read.
const SRC: &str = r#"
const PSI: [i64; 6] = [1, -3, 2, -6, 4, 18];
const W: [i64; 4] = [10, 20, 30, 40];

pub fn psi_sum() -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < 6 { s = s + PSI[i]; i = i + 1; }
    s
}

// Dot-product-ish read: sum W[i] * (i+1) = 10*1 + 20*2 + 30*3 + 40*4 = 300.
pub fn w_weighted() -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < 4 { s = s + W[i] * (i + 1); i = i + 1; }
    s
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn const_array_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("const-array-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_const_array_run.mind");
    let so = dir.join("mind_const_array_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("const-array-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("const-array-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.psi_sum.restype = ctypes.c_int64; lib.psi_sum.argtypes = []\n\
         lib.w_weighted.restype = ctypes.c_int64; lib.w_weighted.argtypes = []\n\
         r = lib.psi_sum(); assert r == 16, 'psi_sum()=' + repr(r)\n\
         r = lib.w_weighted(); assert r == 300, 'w_weighted()=' + repr(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "const-array-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
