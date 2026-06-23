// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `map<K, V>` surface RUNTIME gate.
//!
//! The `map<K, V>` surface lowers onto the std.map heap runtime (i64 handles):
//! a literal `{}` / `{ k: v }` → `map_new` + `map_insert` chain; `m.insert(k,v)`
//! → `map_insert`; `m.get(k)` → `map_get`; `m.contains_key(k)` →
//! `map_contains_key`; `m.len` → `map_len`. A `map<string, V>` binding routes
//! `.get` / `.contains_key` to the CONTENT-equality variants
//! (`map_get_str` / `map_contains_key_str`) — a handle `==` would compare String
//! pointers, not bytes. Compiles a program exercising both key tiers to a `.so`,
//! dlopen-calls it, and asserts the values (proving the runtime, incl. the
//! string-key content-equality path, not merely that it compiles).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test map_surface_run`

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
// i64-key map: insert (value-position rebind), get present/absent, contains.
// get(9)=900 + contains(7)=1 + contains(5)=0 = 901.
pub fn imap() -> i64 {
    let m: map<i64, i64> = {}
    let m = m.insert(7, 700)
    let m = m.insert(9, 900)
    return m.get(9) + m.contains_key(7) + m.contains_key(5)
}

// string-key map: keys are distinct String allocations with the SAME content as
// the lookup keys → must match by CONTENT (map_get_str / map_contains_key_str).
// get("bar")=22 + contains("foo")=1 + contains("zzz")=0 = 23.
pub fn smap() -> i64 {
    let m: map<string, i64> = {}
    let m = m.insert("foo", 11)
    let m = m.insert("bar", 22)
    return m.get("bar") + m.contains_key("foo") + m.contains_key("zzz")
}

// Non-empty map literal `{ k: v, ... }`.
pub fn literal() -> i64 {
    let m: map<i64, i64> = { 1: 10, 2: 20, 3: 30 }
    return m.get(2) + m.len
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn map_surface_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("map-surface-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_map_surface_run.mind");
    let so = dir.join("mind_map_surface_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("map-surface-run: needs mlir-build; skipping");
            return;
        }
        panic!("map-surface-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('imap','smap','literal'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.imap(); assert r == 901, 'imap=' + str(r)\n\
         r = lib.smap(); assert r == 23, 'smap=' + str(r)\n\
         r = lib.literal(); assert r == 23, 'literal=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "map-surface-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
