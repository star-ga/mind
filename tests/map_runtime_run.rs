// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! std.map lookup runtime RUNTIME gate (`map_get` / `map_contains_key`).
//!
//! The insertion-ordered std.map gained linear-scan lookups: `map_get` (value
//! for a key, 0 if absent) and `map_contains_key` (1/0), plus string-key
//! content-equality variants `map_get_str` / `map_contains_key_str` for
//! `map<string, _>` (a handle `==` would compare pointers, not bytes). This
//! exercises the i64-key path end-to-end (insert several entries, look up
//! present + absent keys) via a dlopen-called `.so`. The string-key variants
//! are correct by construction (byte compare of the String record) and are
//! exercised through the `map<string, _>` surface in the compiler test suite.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test map_runtime_run`

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
import std.map

// Insert (7→700, 9→900, 3→300); look up present + absent keys.
// get(9)=900 + contains(7)=1 + contains(5)=0 + get(3)=300 = 1201.
pub fn lookups() -> i64 {
    let m = map_new()
    let m = map_insert(m, 7, 700)
    let m = map_insert(m, 9, 900)
    let m = map_insert(m, 3, 300)
    return map_get(m, 9) + map_contains_key(m, 7) + map_contains_key(m, 5) + map_get(m, 3)
}

// Absent key returns 0 (not a garbage tail read).
pub fn absent() -> i64 {
    let m = map_new()
    let m = map_insert(m, 1, 111)
    return map_get(m, 42)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn map_runtime_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("map-runtime-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_map_runtime_run.mind");
    let so = dir.join("mind_map_runtime_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("map-runtime-run: needs mlir-build; skipping");
            return;
        }
        panic!("map-runtime-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('lookups','absent'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.lookups(); assert r == 1201, 'lookups=' + str(r)\n\
         r = lib.absent(); assert r == 0, 'absent=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "map-runtime-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
