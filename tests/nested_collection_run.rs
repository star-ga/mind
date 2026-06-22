// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Nested-collection RUNTIME gate — a map held as an ELEMENT of a struct's
//! `array<map<K,V>>` field, mutated through an index and persisted.
//!
//! Three things compose here, each previously unresolved:
//!   * `let next = r` aliases the struct-typed binding `r` so `next.tables…`
//!     resolves (`let x = y` struct/collection type propagation).
//!   * `next.tables[0]` — indexing a struct `array<map<…>>` field resolves the
//!     ELEMENT's collection sentinel (map), so `.insert` desugars to `map_insert`.
//!   * `next.tables[0].insert(k, v)` — the fresh-on-realloc map handle is written
//!     back via an `IndexAssign` rebind (`next.tables[0] = next.tables[0].insert`)
//!     so the growth persists; a bare statement would silently drop it.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test nested_collection_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
struct Reg { tables: array<map<i64, i64>> }

fn add_entries(r: Reg) -> Reg {
    let next = r
    next.tables[0].insert(5, 100)
    next.tables[0].insert(6, 200)
    return next
}

pub fn run() -> i64 {
    let m: map<i64, i64> = {}
    let r = Reg { tables: [m] }
    let r = add_entries(r)
    return map_get(r.tables[0], 5) + map_get(r.tables[0], 6)
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
fn nested_collection_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("nested-collection-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_nested_collection_run.mind");
    let so = dir.join("mind_nested_collection_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("nested-collection-run: needs mlir-build; skipping");
            return;
        }
        panic!("nested-collection-run: mindc --emit-shared failed:\n{stderr}");
    }

    // insert (5->100) and (6->200) into the element map, persisted, read back → 300.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 300, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "nested-collection-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
