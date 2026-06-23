// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `set<T>` surface RUNTIME gate.
//!
//! A set literal `{ a, b, c }` (comma-separated, no colons — disambiguated from
//! the map literal `{ k: v }`) is backed by the std.map runtime (a set is a map
//! keyed by its elements, value 1). A `set<T>`-annotated binding carries a set
//! sentinel so `.contains`/`.has` → map_contains_key (_str for string elements),
//! `.add`/`.insert` → map_insert(recv, x, 1), and `.len` → map_len. Statement
//! `s.add(x)` inside a loop rebinds (non-mutating map_insert). Compiles a program
//! exercising the literal + membership + loop-built sets to a `.so`, dlopen-calls
//! it, and asserts the values.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test set_surface_run`

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
// Set literal + membership + .len.
// contains(7)=1*100 + contains(5)=0 + len=3 = 103.
pub fn literal() -> i64 {
    let s: set<i64> = { 3, 7, 9 }
    let a = s.contains(7)
    let b = s.contains(5)
    return a * 100 + b * 10 + s.len
}

// Empty set built with `.add` STATEMENTS inside a loop (the inserts must
// persist — loop-carried via the statement-mutation rebind). set{0,2,4,6}:
// contains(4)=1*10 + len=4 = 14.
pub fn build() -> i64 {
    let mut seen: set<i64> = {}
    let mut i = 0
    while i < 4 {
        seen.add(i * 2)
        i = i + 1
    }
    return seen.contains(4) * 10 + seen.len
}

// String-element set: membership by CONTENT (distinct allocations match).
// contains("bar")=1*10 + contains("zzz")=0 + len=2 = 12.
pub fn strings() -> i64 {
    let s: set<string> = { "foo", "bar" }
    return s.contains("bar") * 10 + s.contains("zzz") + s.len
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn set_surface_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("set-surface-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_set_surface_run.mind");
    let so = dir.join("mind_set_surface_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("set-surface-run: needs mlir-build; skipping");
            return;
        }
        panic!("set-surface-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('literal','build','strings'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.literal(); assert r == 103, 'literal=' + str(r)\n\
         r = lib.build(); assert r == 14, 'build=' + str(r)\n\
         r = lib.strings(); assert r == 12, 'strings=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "set-surface-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
