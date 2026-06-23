// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct-FIELD collection access RUNTIME gate.
//!
//! A collection method/index on a struct FIELD — `s.field.contains_key(k)`,
//! `s.field.get(k)`, `s.field.contains(x)`, `s.field.len` — resolves the field's
//! declared `map<K,V>` / `set<T>` / `array<T>` type (via `struct_field_types`,
//! cross-module via the global struct registry) and routes to the std.map /
//! std.vec runtime — not only an Ident-bound collection. Compiles a program that
//! builds a struct holding collection fields, passes it to functions that query
//! those fields, dlopen-calls it, and asserts the values.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test struct_field_collection_run`

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
struct Analyzed {
    ids:    map<string, i64>,
    flags:  set<string>,
}

// Collection methods on a struct-FIELD via a struct PARAMETER.
fn lookup(a: Analyzed) -> i64 {
    let has = a.ids.contains_key("x")
    let val = a.ids.get("x")
    let flagged = a.flags.contains("on")
    let missing = a.flags.contains("off")
    return val * 1000 + has * 100 + flagged * 10 + missing + a.ids.len
}

pub fn run() -> i64 {
    let mut ids: map<string, i64> = {}
    let ids = ids.insert("x", 7)
    let mut flags: set<string> = {}
    let flags = flags.insert("on")
    let a = Analyzed { ids: ids, flags: flags }
    return lookup(a)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn struct_field_collection_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("struct-field-collection-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_struct_field_collection_run.mind");
    let so = dir.join("mind_struct_field_collection_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("struct-field-collection-run: needs mlir-build; skipping");
            return;
        }
        panic!("struct-field-collection-run: mindc --emit-shared failed:\n{stderr}");
    }

    // val=7*1000 + has=1*100 + flagged=1*10 + missing=0 + ids.len=1 = 7111.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 7111, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "struct-field-collection-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
