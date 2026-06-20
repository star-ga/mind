// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Tuple construction + destructuring `let` RUNTIME gate.
//!
//! A tuple literal `(a, b, ...)` lowers to an anonymous all-i64 product type —
//! the same `__mind_alloc(8*n)` + per-slot `__mind_store_i64` machinery as an
//! all-i64 struct / multi-payload enum variant — and a destructuring
//! `let (x, y) = t` reads each slot back with `__mind_load_i64(t + 8*i)`. The old
//! `Node::Tuple` lowering was a STUB that returned only the LAST element, so a
//! tuple silently collapsed and any destructure read garbage; it shipped because
//! no test ever RAN tuple code. This compiles a program to a `.so`, dlopen-calls
//! it, and asserts the values — it would have caught the collapse.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test tuple_destructure_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// Direct destructure of a 3-tuple literal.
pub fn direct() -> i64 {
    let (a, b, c) = (1, 20, 300)
    return a + b + c
}

// Destructure through a `let`-bound tuple value (the tuple flows through a
// binding before being taken apart).
pub fn via_binding() -> i64 {
    let p = (7, 9)
    let (x, y) = p
    return x * 10 + y
}

// Element order: first slot vs second slot must not be swapped.
pub fn order() -> i64 {
    let (first, second) = (3, 5)
    return first - second
}

// Sequential destructures that reference earlier bindings — `q` uses `x`/`y`.
pub fn chained() -> i64 {
    let (x, y) = (100, 5)
    let (p, q) = (x + y, x - y)
    return p * q
}

// A 2-tuple whose elements are themselves expressions with calls.
pub fn helper(n: i64) -> i64 { return n + n }
pub fn exprs() -> i64 {
    let (lo, hi) = (helper(4), helper(10))
    return hi - lo
}

// An all-i64 tuple RETURNED across a fn boundary: the callee returns the heap
// aggregate's base pointer (i64), and the caller destructures it. This is the
// pattern that the `safety::tuple_return_unsupported` guard now allows for
// all-i64 tuples (only float-bearing tuples stay rejected).
fn make_pair(a: i64, b: i64) -> (i64, i64) {
    return (a, b)
}
pub fn returned() -> i64 {
    let (x, y) = make_pair(10, 32)
    return x + y
}
pub fn returned_order() -> i64 {
    let (x, y) = make_pair(3, 70)
    return y - x
}
"#;

fn mindc_bin() -> PathBuf {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d = m.join("target").join("debug").join("mindc");
    if d.exists() {
        d
    } else {
        m.join("target").join("release").join("mindc")
    }
}

#[test]
fn tuple_destructure_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("tuple-destructure-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_tuple_destructure_run.mind");
    let so = dir.join("mind_tuple_destructure_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("tuple-destructure-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("tuple-destructure-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('direct','via_binding','order','chained','exprs','returned','returned_order'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.direct(); assert r == 321, 'direct=' + str(r)\n\
         r = lib.via_binding(); assert r == 79, 'via_binding=' + str(r)\n\
         r = lib.order(); assert r == -2, 'order=' + str(r)\n\
         r = lib.chained(); assert r == 9975, 'chained=' + str(r)\n\
         r = lib.exprs(); assert r == 12, 'exprs=' + str(r)\n\
         r = lib.returned(); assert r == 42, 'returned=' + str(r)\n\
         r = lib.returned_order(); assert r == 67, 'returned_order=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "tuple-destructure-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
