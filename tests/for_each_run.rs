// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! For-each `for x in coll { … }` RUNTIME gate.
//!
//! For-each over an `array<T>` (std.vec handle) flat-desugars to an indexed
//! `while` over `vec_len` / `vec_get`, reusing the loop-carried-SSA machinery of
//! the `For`/`While` arms (no nested Block). Hidden index/length/collection
//! bindings are span-unique so a NESTED for-each does not collide. Compiles a
//! program with a simple and a nested for-each to a `.so`, dlopen-calls it, and
//! asserts the accumulated values.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test for_each_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// Sum a literal array via for-each: 5+10+15+20 = 50.
pub fn sum_each() -> i64 {
    let a: array<i64> = [5, 10, 15, 20]
    let mut s = 0
    for x in a {
        s = s + x
    }
    return s
}

// Nested for-each (exercises span-unique hidden vars): sum of x*y over the
// cross product = (1+2+3) * (10+20) = 180.
pub fn nested() -> i64 {
    let a: array<i64> = [1, 2, 3]
    let b: array<i64> = [10, 20]
    let mut s = 0
    for x in a {
        for y in b {
            s = s + x * y
        }
    }
    return s
}

// For-each over an `array<T>` PARAMETER (collection threaded across a call).
fn count_positive(xs: array<i64>) -> i64 {
    let mut n = 0
    for v in xs {
        if v > 0 {
            n = n + 1
        }
    }
    return n
}

pub fn over_param() -> i64 {
    let xs: array<i64> = [3, 0, 7, 0, 9]
    return count_positive(xs)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn for_each_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("for-each-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_for_each_run.mind");
    let so = dir.join("mind_for_each_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("for-each-run: needs mlir-build; skipping");
            return;
        }
        panic!("for-each-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('sum_each','nested','over_param'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.sum_each(); assert r == 50, 'sum_each=' + str(r)\n\
         r = lib.nested(); assert r == 180, 'nested=' + str(r)\n\
         r = lib.over_param(); assert r == 3, 'over_param=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "for-each-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
