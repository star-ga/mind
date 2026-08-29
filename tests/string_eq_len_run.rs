// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native string `==` / `!=` / `.len()` RUNTIME gate (issue #245).
//!
//! Before this gate, `==` between two strings compared the two `__mind_alloc`
//! heap-record POINTERS — never equal for two distinct literals — so
//! `"abc" == "abc"` was FALSE, and an annotated `let s: string` was tracked by
//! nothing, so `s.len()` missed the `string_<method>` dispatch and lowered to
//! `const.i64 0`. Both built clean to a real ELF, exited 0, and silently
//! computed the wrong answer with no diagnostic and no JIT-fallback banner.
//!
//! This asserts VALUES through the real runtime, not that the source parses.
//! The integer arm is the control: `==` on i64 must be untouched.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_eq_len_run`

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
// Two EQUAL string literals must compare equal (was 0 — pointer compare).
pub fn lit_eq() -> i64 {
    if "abc" == "abc" { return 1 }
    return 0
}

// Two DIFFERENT literals must still compare unequal.
pub fn lit_neq() -> i64 {
    if "abc" == "xyz" { return 1 }
    return 0
}

// `.len()` on an annotated string local (was 0 — unresolved receiver).
pub fn ann_len() -> i64 {
    let s: string = "abcd"
    return s.len()
}

// Equality through VARIABLES, not just literals.
pub fn var_eq() -> i64 {
    let a: string = "hello"
    let b: string = "hello"
    if a == b { return 1 }
    return 0
}

// `!=` is the negation of string_eq, not a pointer compare.
pub fn ne_diff() -> i64 {
    let a: string = "hello"
    let b: string = "world"
    if a != b { return 1 }
    return 0
}

// CONTROL: i64 `==` must be byte-for-byte unaffected by the string routing.
pub fn int_eq_unchanged() -> i64 {
    let a: i64 = 3
    if a == 3 { return 7 }
    return 0
}
"#;

#[test]
fn string_eq_and_len_run() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("string-eq-len-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_string_eq_len_run.mind");
    let so = dir.join("mind_string_eq_len_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("string-eq-len-run: needs mlir-build; skipping");
            return;
        }
        panic!("string-eq-len-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('lit_eq','lit_neq','ann_len','var_eq','ne_diff','int_eq_unchanged'):\n\
         \x20   getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.lit_eq(); assert r == 1, 'lit_eq=' + str(r)\n\
         r = lib.lit_neq(); assert r == 0, 'lit_neq=' + str(r)\n\
         r = lib.ann_len(); assert r == 4, 'ann_len=' + str(r)\n\
         r = lib.var_eq(); assert r == 1, 'var_eq=' + str(r)\n\
         r = lib.ne_diff(); assert r == 1, 'ne_diff=' + str(r)\n\
         r = lib.int_eq_unchanged(); assert r == 7, 'int_eq_unchanged=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-eq-len-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
