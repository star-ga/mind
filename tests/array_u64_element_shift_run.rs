// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `array<u64>` element read keeps its UNSIGNEDNESS through a `vec_get`.
//!
//! Regression gate for a HIGH correctness bug: an in-bounds `array<u64>`
//! element read (`a[i]`) lowered to a bare `vec_get` whose result was an
//! UNTYPED i64 SSA value, so a downstream sign-sensitive op saw a SIGNED i64.
//! For `u64::MAX` (all-ones i64 = -1) `a[0] >> 63` selected the ARITHMETIC
//! (signed) shift and returned -1 instead of the logical-shift result 1.
//!
//! The `vec_get` result is now re-materialised at the array's declared element
//! type (`index_element_narrow_ty` + `mask_narrow_let`), so a `u64` element
//! carries the `__mind_conv_u64` marker and MLIR selects the UNSIGNED variant
//! (`arith.shrui`, `cmpi ult`). Controls prove the fix is narrow: an
//! `array<i64>` element `>>` stays ARITHMETIC (-1), and an in-range `u64`
//! element still shifts/compares correctly.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test array_u64_element_shift_run`

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
// The bug: u64::MAX element logical-shifted right by 63 == 1 (was -1).
pub fn f() -> i64 {
    let x: u64 = (0 - 1) as u64
    let a: array<u64> = [x]
    let v = a[0]
    return v >> 63
}

// Control: an array<i64> element keeps the ARITHMETIC (signed) shift (-1).
pub fn ctrl_i64_arith() -> i64 {
    let x: i64 = 0 - 1
    let a: array<i64> = [x]
    let v = a[0]
    return v >> 63
}

// Control: a u64 element in an unsigned compare — u64::MAX > 5 is true (1).
pub fn ctrl_u64_cmp() -> i64 {
    let x: u64 = (0 - 1) as u64
    let a: array<u64> = [x]
    let v = a[0]
    if v > 5 {
        return 1
    }
    return 0
}

// Control: an in-range u64 element still shifts correctly (1024 >> 2 == 256).
pub fn ctrl_u64_inrange() -> i64 {
    let x: u64 = 1024 as u64
    let a: array<u64> = [x]
    let v = a[0]
    return v >> 2
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn array_u64_element_shift_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("array-u64-element-shift-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_array_u64_element_shift_run.mind");
    let so = dir.join("mind_array_u64_element_shift_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("array-u64-element-shift-run: needs mlir-build; skipping");
            return;
        }
        panic!("array-u64-element-shift-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('f','ctrl_i64_arith','ctrl_u64_cmp','ctrl_u64_inrange'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.f(); assert r == 1, 'f=' + str(r)\n\
         r = lib.ctrl_i64_arith(); assert r == -1, 'ctrl_i64_arith=' + str(r)\n\
         r = lib.ctrl_u64_cmp(); assert r == 1, 'ctrl_u64_cmp=' + str(r)\n\
         r = lib.ctrl_u64_inrange(); assert r == 256, 'ctrl_u64_inrange=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "array-u64-element-shift-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
