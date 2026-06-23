// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A collection mutating-method (`push`/`insert`/`set`/`add`) used in
//! EXPRESSION/ARGUMENT position — e.g. `w.push(v.push(5))` — must be refused
//! fail-loud (#306), because the std collection returns a fresh handle on
//! realloc that can only be rebound at statement position; in expression
//! position the mutation would be silently lost (a silent miscompile).
//! Statement-position mutation must keep working unchanged.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test collection_mutation_expr_position_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

fn mindc_bin() -> PathBuf {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d = m.join("target").join("debug").join("mindc");
    if d.exists() {
        d
    } else {
        m.join("target").join("release").join("mindc")
    }
}

/// `w.push(v.push(5))` — mutation in expression position — must FAIL to compile.
#[test]
fn collection_mutation_in_expr_position_is_rejected() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("collmut-expr: mindc not found; skipping");
        return;
    }
    let src = "pub fn run() -> i64 {\n\
               \x20   let mut w: array<i64> = array<i64>.new()\n\
               \x20   let mut v: array<i64> = array<i64>.new()\n\
               \x20   w.push(v.push(5))\n\
               \x20   return 1\n\
               }\n";
    let dir = std::env::temp_dir();
    let s = dir.join("mind_collmut_expr.mind");
    let so = dir.join("mind_collmut_expr.so");
    std::fs::write(&s, src).expect("write");
    let out = Command::new(&mindc)
        .args([s.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("mlir-build") && err.contains("requires") {
        println!("collmut-expr: needs mlir-build; skipping");
        return;
    }
    assert!(
        !out.status.success(),
        "collmut: expr-position mutation must FAIL to compile, but it succeeded"
    );
    assert!(
        err.contains("expression position") && err.contains("silently lost"),
        "collmut: expected the expr-position fail-loud diagnostic, got:\n{err}"
    );
}

/// `w.push(5)` as its own statement must still compile + run (count == 2).
#[test]
fn statement_position_mutation_still_works() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("collmut-stmt: mindc not found; skipping");
        return;
    }
    let src = "pub fn run() -> i64 {\n\
               \x20   let mut w: array<i64> = array<i64>.new()\n\
               \x20   w.push(5)\n\
               \x20   w.push(7)\n\
               \x20   return w.length\n\
               }\n";
    let dir = std::env::temp_dir();
    let s = dir.join("mind_collmut_stmt.mind");
    let so = dir.join("mind_collmut_stmt.so");
    std::fs::write(&s, src).expect("write");
    let out = Command::new(&mindc)
        .args([s.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("mlir-build") && err.contains("requires") {
            println!("collmut-stmt: needs mlir-build; skipping");
            return;
        }
        panic!("collmut-stmt: statement-position push must compile:\n{err}");
    }
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\nlib.run.restype=ctypes.c_int64\nassert lib.run()==2, lib.run()\nprint('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3").args(["-c", &py]).output().expect("py");
    assert!(
        out.status.success(),
        "collmut-stmt: run() != 2\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
