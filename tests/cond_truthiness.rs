// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `if`/`while` condition truthiness gate.
//!
//! A non-`i1` (integer) condition is TRUE iff it is non-zero. The lowering used
//! to `arith.trunci` an i64 condition to i1, which tests only the LOW BIT — so an
//! even non-zero value (`2`, `4`, …) wrongly branched FALSE (`pick(2,10,20)`
//! returned `20` instead of `10`, and a `while c` countdown skipped even values).
//! The fix compares the condition against `0` (`cmpi "ne"`). An already-`i1`
//! comparison result is unaffected (it never took the trunci path), so the
//! keystone + cross-substrate byte-identity gates are untouched.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test cond_truthiness`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
pub fn pick(c: i64, a: i64, b: i64) -> i64 {
    if c {
        return a
    }
    return b
}
pub fn count(n: i64) -> i64 {
    let mut c: i64 = n
    let mut acc: i64 = 0
    while c {
        acc = acc + c
        c = c - 1
    }
    return acc
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn integer_condition_is_true_iff_nonzero() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("cond-truthiness: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_cond_truthiness.mind");
    let so = dir.join("mind_cond_truthiness.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("cond-truthiness: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("cond-truthiness: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.pick.restype = ctypes.c_int64\n\
         lib.pick.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]\n\
         lib.count.restype = ctypes.c_int64\n\
         lib.count.argtypes = [ctypes.c_int64]\n\
         # an `if <int>` branches on non-zero, NOT on the low bit\n\
         r = lib.pick(0, 10, 20); assert r == 20, 'pick(0)=' + str(r)\n\
         r = lib.pick(1, 10, 20); assert r == 10, 'pick(1)=' + str(r)\n\
         r = lib.pick(2, 10, 20); assert r == 10, 'pick(2)=' + str(r)  # the bug case\n\
         r = lib.pick(4, 10, 20); assert r == 10, 'pick(4)=' + str(r)  # the bug case\n\
         # a `while <int>` countdown visits every value down to 0\n\
         r = lib.count(4); assert r == 10, 'count(4)=' + str(r)\n\
         r = lib.count(5); assert r == 15, 'count(5)=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "cond-truthiness check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
