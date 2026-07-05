// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Scalar `f64` cross-function CALL-ARGUMENT runtime gate.
//!
//! Passing an `f64` SCALAR value across a function-call boundary must lower to a
//! `func.call @callee(%x) : (f64, ...) -> f64` — a native MLIR scalar slot, NOT
//! the aggregate call ABI (RFC 0005 phase 2+). The MLIR-build call lowering used
//! to reject every non-i64 call argument with "non-i64 argument to call"; the
//! scalar-f64 case is a bounded, self-contained fix distinct from the aggregate
//! machinery (the callee param slot is typed from `fn_signatures`, the f64 value
//! passes untouched since `phys == target`, and the result is tracked at the
//! callee's `ScalarF64` return kind).
//!
//! Determinism: the f64 arithmetic that produces the argument, and the callee
//! body, stay on the STRICT float path — `arith.mulf` + `arith.addf` as SEPARATE
//! ops (no FMA-contraction) and no reassociation. A call only forwards the value,
//! so it introduces no reordering. This compiles a two-function program that
//! passes an f64 across a call to a `.so`, dlopen-calls it, and asserts the
//! results are an EXACT f64 bit-match.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test f64_call_arg_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// Callee: takes two f64 SCALAR args, returns f64. `x * k + 1.5` lowers to
// arith.mulf + arith.addf (separate ops — strict, no FMA contraction).
pub fn scale(x: f64, k: f64) -> f64 {
    return x * k + 1.5
}

// Caller: passes f64 SCALARS across the call boundary (both a variable and a
// literal argument), then combines two call results with strict f64 add.
pub fn driver() -> f64 {
    let a: f64 = 2.5
    let b: f64 = 4.0
    let r: f64 = scale(a, b)
    return r + scale(r, 2.0)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn f64_call_arg_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("f64-call-arg-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_f64_call_arg_run.mind");
    let so = dir.join("mind_f64_call_arg_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("f64-call-arg-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("f64-call-arg-run: mindc --emit-shared failed:\n{stderr}");
    }

    // Reference computed with the same IEEE-754 f64 ops the .mind performs, so
    // the assertion is an EXACT bit-match (single-substrate).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.scale.restype = ctypes.c_double\n\
         lib.scale.argtypes = [ctypes.c_double, ctypes.c_double]\n\
         lib.driver.restype = ctypes.c_double; lib.driver.argtypes = []\n\
         s = lib.scale(2.5, 4.0)\n\
         assert s == 11.5, 'scale(2.5,4.0)=' + repr(s)\n\
         r = 2.5 * 4.0 + 1.5\n\
         expect = r + (r * 2.0 + 1.5)\n\
         d = lib.driver()\n\
         assert d == expect, 'driver()=' + repr(d) + ' expected ' + repr(expect)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "f64-call-arg-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
