// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Postfix `?` operator (W1.5f) RUNTIME gate.
//!
//! `expr?` desugars to the `match` error-propagation machinery: for a
//! `Result`-returning fn, `x?` is `match x { Ok(v) => v, Err(e) => return Err(e) }`
//! (and the `Some`/`None` twin for an `Option`-returning fn). The parser fixes the
//! family from the enclosing fn's return type; lowering emits the SAME match/return
//! IR the hand-written form does. Parsing + type-rejection are pinned by
//! `parse_try_operator.rs`; THIS gate proves the desugar RUNS correctly on real
//! hardware for BOTH the Ok/Some UNWRAP path and the Err/None EARLY-RETURN path.
//!
//! It compiles to a `.so` and dlopen-calls EXPORTED functions with ctypes, so each
//! `?`-using fn is exercised via a real call (not a const-foldable `main`) — the
//! honest, cache-contamination-proof shape.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test try_operator_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
fn qm_div(a: i64, b: i64) -> Result<i64, i64> {
    if b == 0 {
        return Err(7)
    }
    Ok(a / b)
}

// Two `?` in one fn — the Ok path unwraps twice; a failing first `?` early-returns
// its Err and the second `?` never runs.
fn qm_chain(a: i64, b: i64, c: i64) -> Result<i64, i64> {
    let x = qm_div(a, b)?
    let y = qm_div(x, c)?
    Ok(y)
}

// Ok path: div(120,3)=40, div(40,2)=20 -> Ok(20). `?` unwrapped both.
pub fn run_ok() -> i64 {
    match qm_chain(120, 3, 2) {
        Ok(v) => v,
        Err(e) => 0 - 1,
    }
}

// Err path: div(120,0) -> Err(7); the first `?` early-returns Err(7) out of
// qm_chain (the second div never runs).
pub fn run_err() -> i64 {
    match qm_chain(120, 0, 2) {
        Ok(v) => 0 - 1,
        Err(e) => e,
    }
}

fn qo_pos(x: i64) -> Option<i64> {
    if x == 0 {
        return None
    }
    Some(x)
}

fn qo_chain(x: i64) -> Option<i64> {
    let a = qo_pos(x)?
    Some(a + 1)
}

// Option Some path: Some(41) -> `?` unwraps 41 -> Some(42).
pub fn run_opt_some() -> i64 {
    match qo_chain(41) {
        Some(v) => v,
        Nothing => 0 - 1,
    }
}

// Option None path: qo_pos(0) -> None; the `?` early-returns None out of qo_chain.
pub fn run_opt_none() -> i64 {
    match qo_chain(0) {
        Some(v) => 0 - 1,
        Nothing => 99,
    }
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn try_operator_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("try-operator-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_try_operator_run.mind");
    let so = dir.join("mind_try_operator_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("try-operator-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("try-operator-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('run_ok','run_err','run_opt_some','run_opt_none'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.run_ok(); assert r == 20, 'run_ok=' + str(r)\n\
         r = lib.run_err(); assert r == 7, 'run_err=' + str(r)\n\
         r = lib.run_opt_some(); assert r == 42, 'run_opt_some=' + str(r)\n\
         r = lib.run_opt_none(); assert r == 99, 'run_opt_none=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "try-operator-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
