// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `map.get` value-type inference + const/field map receivers RUNTIME gate.
//!
//!   * `TABLE.get(k)` where `const TABLE: map<K,V>` — a module-const map used as
//!     a method receiver resolves its collection sentinel (so `.get` desugars to
//!     `map_get`) via the const's declared type.
//!   * `g.vals.get(k)` — `.get` on a struct map FIELD likewise resolves.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test map_get_inference_run`

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
const TABLE: map<i64, i64> = {7: 42, 8: 99}

struct G { vals: map<i64, i64> }

fn from_const(k: i64) -> i64 {
    return TABLE.get(k)
}

fn from_field(g: G) -> i64 {
    return g.vals.get(3)
}

pub fn run() -> i64 {
    let g = G { vals: TABLE }
    return from_const(7) + from_const(8) + from_field(g)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn map_get_inference_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("map-get-inference-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_map_get_inference_run.mind");
    let so = dir.join("mind_map_get_inference_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("map-get-inference-run: needs mlir-build; skipping");
            return;
        }
        panic!("map-get-inference-run: mindc --emit-shared failed:\n{stderr}");
    }

    // from_const(7)=42, from_const(8)=99, from_field reads TABLE[3] (absent → 0)
    // → 42 + 99 + 0 = 141.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 141, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "map-get-inference-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
