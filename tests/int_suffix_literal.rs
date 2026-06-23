// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Issue #205: integer-type suffixes (`2u32`, `-1i32`) in expression position.
//!
//! Before the fix the parser had no concept of a numeric-literal type suffix, so
//! `2u32 * c` died with `expected ')', found Some('i')` and `-1i32` / `d == 2u32 * c`
//! likewise. The suffix is now consumed at a word boundary on any integer literal
//! and desugared into the existing `expr as type` cast — so `2u32` is exactly
//! `(2 as u32)`. This test compiles a program that uses suffixes in multiply,
//! unary-negate and equality position to a `.so`, dlopen-calls it, and asserts the
//! runtime values to prove the symptom is gone (parse + typecheck + codegen + run),
//! not merely that it parses.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test int_suffix_literal`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// `2u32 * c` — suffix on a literal used in a multiply
pub fn mul_suffix(c: i64) -> i64 {
    let d: i64 = 2u32 * c
    return d
}

// `-1i32` — suffix under unary negation
pub fn neg_suffix() -> i64 {
    let e: i64 = -1i32
    return e
}

// `d == 2u32 * c` — suffix in an equality comparison
pub fn eq_suffix(c: i64) -> i64 {
    let d: i64 = 2u32 * c
    if d == 2u32 * c {
        return 1
    }
    return 0
}

// i64 suffix on the runnable-scalar path
pub fn i64_suffix() -> i64 {
    return 7i64 + 3i64
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn int_suffix_literals_parse_typecheck_and_run() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("int-suffix: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_int_suffix_literal.mind");
    let so = dir.join("mind_int_suffix_literal.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("int-suffix: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("int-suffix: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.mul_suffix.restype = ctypes.c_int64\n\
         lib.mul_suffix.argtypes = [ctypes.c_int64]\n\
         r = lib.mul_suffix(5); assert r == 10, 'mul_suffix=' + str(r)\n\
         lib.neg_suffix.restype = ctypes.c_int64\n\
         r = lib.neg_suffix(); assert r == -1, 'neg_suffix=' + str(r)\n\
         lib.eq_suffix.restype = ctypes.c_int64\n\
         lib.eq_suffix.argtypes = [ctypes.c_int64]\n\
         r = lib.eq_suffix(3); assert r == 1, 'eq_suffix=' + str(r)\n\
         lib.i64_suffix.restype = ctypes.c_int64\n\
         r = lib.i64_suffix(); assert r == 10, 'i64_suffix=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "int-suffix value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
