// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `u64 as i64` signed-compare RUNTIME gate (issue #210).
//!
//! Regression for a verified silent miscompile: casting a `u64` value to `i64`
//! carried the source's `ScalarU64` kind forward, so a downstream
//! sign-sensitive op picked the UNSIGNED variant — `if (x as i64) < 0` emitted
//! `arith.cmpi "ult"` (always false for a `< 0` test). The fix re-tags the
//! `__mind_conv_i64` result `ScalarI64` for an integer `u64` source, so the
//! compare emits `slt`. This test compiles `2^64-1` through the cast and
//! asserts the signed compare actually sees `-1 < 0`, plus guards that plain
//! u64 compares stay unsigned and plain i64 compares stay signed.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test u64_cast_signed_compare_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
fn g(x: u64) -> i64 {
    if (x as i64) < 0 {
        return 1
    }
    return 0
}

// 2^64-1 as i64 is -1: the compare after the cast must be SIGNED -> 1.
pub fn kat_scmp() -> i64 {
    let big: u64 = (0 - 1) as u64
    return g(big)
}

// u64 compares stay UNSIGNED: 2^64-1 < 1 is false -> 0.
pub fn kat_u64_ult() -> i64 {
    let big: u64 = (0 - 1) as u64
    let one: u64 = 1 as u64
    if big < one {
        return 1
    }
    return 0
}

// Plain i64 compares stay SIGNED: -1 < 0 is true -> 1.
pub fn kat_i64_slt() -> i64 {
    let neg: i64 = 0 - 1
    if neg < 0 {
        return 1
    }
    return 0
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn u64_cast_signed_compare_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("u64-cast-signed-compare-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_u64_cast_signed_compare_run.mind");
    let so = dir.join("mind_u64_cast_signed_compare_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("u64-cast-signed-compare-run: needs mlir-build; skipping");
            return;
        }
        panic!("u64-cast-signed-compare-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('kat_scmp','kat_u64_ult','kat_i64_slt'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.kat_scmp();    assert r == 1, 'kat_scmp=' + str(r)\n\
         r = lib.kat_u64_ult(); assert r == 0, 'kat_u64_ult=' + str(r)\n\
         r = lib.kat_i64_slt(); assert r == 1, 'kat_i64_slt=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "u64-cast-signed-compare-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
