// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `array<T>` dynamic-array surface RUNTIME gate.
//!
//! The `array<T>` surface lowers onto the std.vec heap runtime (i64 handles):
//! a literal `[a, b, c]` → `vec_new` + `vec_push` chain; `arr.push(x)` →
//! `vec_push`; `arr.get(i)` → `vec_get`; `arr[i]` → `vec_get`; `arr.len` /
//! `arr.length` → `vec_len`; and an `array<T>` PARAMETER carries the same vec
//! sentinel so methods/index/length resolve in the callee. Compiles a program
//! exercising each form to a `.so`, dlopen-calls it, and asserts the values —
//! proving the runtime works, not merely that it compiles.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test array_surface_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// Literal + index + .length + .get + .push (push appends in place when the
// capacity fits, so after push the length is 4): 20 + 4 + 10 = 34.
pub fn mixed() -> i64 {
    let a: array<i64> = [10, 20, 30]
    let _ = a.push(40)
    return a[1] + a.length + a.get(0)
}

// `array<T>` parameter: iterate by index, summing. 1+2+3+4 = 10.
fn sum(xs: array<i64>) -> i64 {
    let mut s = 0
    let mut i = 0
    while i < xs.length {
        s = s + xs[i]
        i = i + 1
    }
    return s
}

pub fn sum_param() -> i64 {
    let xs: array<i64> = [1, 2, 3, 4]
    return sum(xs)
}

// Empty literal + push growth past the initial capacity (4 → 8): length 5.
pub fn grow() -> i64 {
    let v: array<i64> = []
    let _ = v.push(1)
    let _ = v.push(2)
    let _ = v.push(3)
    let _ = v.push(4)
    let _ = v.push(5)
    return v.length
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn array_surface_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("array-surface-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_array_surface_run.mind");
    let so = dir.join("mind_array_surface_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("array-surface-run: needs mlir-build; skipping");
            return;
        }
        panic!("array-surface-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('mixed','sum_param','grow'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.mixed(); assert r == 34, 'mixed=' + str(r)\n\
         r = lib.sum_param(); assert r == 10, 'sum_param=' + str(r)\n\
         r = lib.grow(); assert r == 5, 'grow=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "array-surface-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
