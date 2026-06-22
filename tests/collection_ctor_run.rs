// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Generic-type static constructor RUNTIME gate — `map<K,V>.new()` /
//! `set<T>.new()` / `array<T>.new()` in expression position. The parser parses
//! the balanced `<…>` type args + `.new()` and emits the runtime constructor
//! (`map_new`/`vec_new`); a real `map < x` comparison is untouched (position is
//! restored if it isn't a `<…>.method(…)` form).

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface", feature = "cross-module-imports"))]
use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
struct Bag { h: map<string, string>, v: array<i64> }
pub fn run() -> i64 {
    let b = Bag { h: map<string, string>.new(), v: array<i64>.new() }
    return map_len(b.h) + vec_len(b.v) + 7
}
"#;

fn mindc_bin() -> PathBuf {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d = m.join("target").join("debug").join("mindc");
    if d.exists() { d } else { m.join("target").join("release").join("mindc") }
}

#[test]
fn collection_ctor_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() { println!("skip: no mindc"); return; }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_collection_ctor_run.mind");
    let so = dir.join("mind_collection_ctor_run.so");
    std::fs::write(&src, SRC).expect("write");
    let out = Command::new(&mindc).args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()]).output().expect("run");
    if !out.status.success() {
        let e = String::from_utf8_lossy(&out.stderr);
        if e.contains("mlir-build") && e.contains("requires") { println!("skip: needs mlir-build"); return; }
        panic!("compile failed:\n{e}");
    }
    let py = format!("import ctypes\nl=ctypes.CDLL(r'{}')\nl.run.restype=ctypes.c_int64\nr=l.run();assert r==7,'run='+str(r)\nprint('ok')\n", so.to_string_lossy());
    let out = Command::new("python3").args(["-c", &py]).output().expect("py");
    assert!(out.status.success(), "check failed:\n{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}
