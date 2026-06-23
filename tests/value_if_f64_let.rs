// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A value-`if` whose branch declares a `let` of a non-i64 (f64) value used to
//! MISCOMPILE.
//!
//! A `let` inside a value-`if` branch is recorded as a branch "write", so it
//! becomes a one-sided merge phi (defined in one branch, absent in the other).
//! The absent-branch placeholder was hardcoded `ConstI64(0)` — i64 — so for an
//! `f64` `let` the phi typed `i64` while the then-edge supplied the `f64` value:
//! `cf.br ^merge(%v : i64)` over an f64 `%v` → `mlir-opt: 'i64' vs 'f64'`.
//! `if c { 1.5 } else { 2.5 }` (no branch `let`) lowered fine; only a branch
//! `let` triggered it. The placeholder is now typed by the side that DEFINES the
//! binding (an f64 `let` → `ConstF64(0.0)`), so the merge phi types `f64`. An
//! i64 `let` keeps `ConstI64(0)` → every all-i64 program is byte-identical
//! (keystone 7/7 + cross-substrate 8/8).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test value_if_f64_let`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// f64 value-`if`, `let` in the THEN branch only (the original miscompile).
fn one_sided(c: i64) -> f64 {
    if c == 0 { let v = 1.5
 v } else { 2.5 }
}

// f64 value-`if`, a `let` in EACH branch (two-sided).
fn two_sided(c: i64) -> f64 {
    if c == 0 { let v = 1.5
 v } else { let w = 2.5
 w }
}

// The branch `let` feeds f64 arithmetic in the arm.
fn arith(c: i64) -> f64 {
    if c == 0 { let v = 1.5
 v + v } else { 2.5 }
}

// i64 value-`if` with a branch `let` — the byte-identical path; must still work.
fn i64_let(c: i64) -> i64 {
    if c == 0 { let v = 5
 v } else { 9 }
}

pub fn t_one_then() -> f64 { one_sided(0) }
pub fn t_one_else() -> f64 { one_sided(1) }
pub fn t_two_then() -> f64 { two_sided(0) }
pub fn t_two_else() -> f64 { two_sided(1) }
pub fn t_arith() -> f64 { arith(0) }
pub fn t_arith_else() -> f64 { arith(1) }
pub fn t_i64_then() -> i64 { i64_let(0) }
pub fn t_i64_else() -> i64 { i64_let(1) }
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn value_if_with_f64_branch_let_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("value-if-f64-let: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_value_if_f64_let.mind");
    let so = dir.join("mind_value_if_f64_let.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("value-if-f64-let: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("value-if-f64-let: mindc --emit-shared failed:\n{stderr}");
    }

    // Flat statements (the `\`-continuation strips leading whitespace).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('t_one_then','t_one_else','t_two_then','t_two_else','t_arith','t_arith_else'): getattr(lib,_n).restype = ctypes.c_double\n\
         for _n in ('t_i64_then','t_i64_else'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.t_one_then(); assert r == 1.5, 't_one_then=' + str(r)\n\
         r = lib.t_one_else(); assert r == 2.5, 't_one_else=' + str(r)\n\
         r = lib.t_two_then(); assert r == 1.5, 't_two_then=' + str(r)\n\
         r = lib.t_two_else(); assert r == 2.5, 't_two_else=' + str(r)\n\
         r = lib.t_arith(); assert r == 3.0, 't_arith=' + str(r)\n\
         r = lib.t_arith_else(); assert r == 2.5, 't_arith_else=' + str(r)\n\
         r = lib.t_i64_then(); assert r == 5, 't_i64_then=' + str(r)\n\
         r = lib.t_i64_else(); assert r == 9, 't_i64_else=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "value-if-f64-let value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
