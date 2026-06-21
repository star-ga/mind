// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Statement-position collection mutation RUNTIME gate.
//!
//! std.map's `map_insert` and (on realloc) std.vec's `vec_push` return a FRESH
//! handle — so a bare `m.insert(k,v)` / `v.push(x)` STATEMENT (result discarded)
//! would silently lose the change. A lowering pre-pass rewrites such statements
//! into `m = m.insert(k,v)` / `v = v.push(x)` so the handle rebinds. Emitting a
//! real assignment (not an SSA-env rebind) keeps loop-carried-SSA detection
//! intact, so the mutation persists across LOOP iterations — the critical case.
//! Compiles a program that mutates a map and a vec inside `while` loops, then
//! reads them back, and asserts the accumulated state.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test statement_mutation_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// Statement `id_of.insert(...)` inside a loop — the inserts must persist
// (loop-carried). get(3)=300 + len=4 = 304.
pub fn build_map() -> i64 {
    let mut id_of: map<i64, i64> = {}
    let mut i = 0
    while i < 4 {
        id_of.insert(i, i * 100)
        i = i + 1
    }
    return id_of.get(3) + id_of.len
}

// Statement `v.push(...)` inside a loop, past the initial capacity (realloc) —
// the pushes must persist. len=6 + v[5]=5 = 11.
pub fn build_vec() -> i64 {
    let mut v: array<i64> = []
    let mut i = 0
    while i < 6 {
        v.push(i)
        i = i + 1
    }
    return v.length + v[5]
}

// Statement mutation inside a for-each over a collection.
pub fn from_each() -> i64 {
    let src: array<i64> = [10, 20, 30]
    let mut acc: array<i64> = []
    for x in src {
        acc.push(x + 1)
    }
    return acc.length * 100 + acc[2]
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
fn statement_mutation_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("statement-mutation-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_statement_mutation_run.mind");
    let so = dir.join("mind_statement_mutation_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("statement-mutation-run: needs mlir-build; skipping");
            return;
        }
        panic!("statement-mutation-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('build_map','build_vec','from_each'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.build_map(); assert r == 304, 'build_map=' + str(r)\n\
         r = lib.build_vec(); assert r == 11, 'build_vec=' + str(r)\n\
         r = lib.from_each(); assert r == 331, 'from_each=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "statement-mutation-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
