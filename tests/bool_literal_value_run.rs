// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Bare boolean literal in VALUE position RUNTIME gate.
//!
//! `true` / `false` used as a value — a `let` initialiser, a condition, a call
//! argument (`Err(false)`), and a struct field (`T { flag: true }`) — must lower
//! to the i64 truthiness encoding (1 / 0), MIND's bool ABI, rather than being
//! parsed as an unknown identifier. (The pattern side already maps `true`/`false`
//! to 1/0; this covers the value side.) Compiles a program exercising each
//! position to a `.so`, dlopen-calls it, and asserts the encoded values.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test bool_literal_value_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
struct Flag { on: i64 }

// `let` initialiser + condition.
pub fn from_let() -> i64 {
    let b = false
    if b { return 100 }
    let c = true
    if c { return 7 }
    return 0
}

// Struct field initialised from a bare bool literal (false → 0, true → 1).
pub fn from_field() -> i64 {
    let f = Flag { on: true }
    f.on
}

// Bare bool literal returned directly (true → 1).
pub fn direct() -> i64 {
    return true
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn bool_literal_value_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bool-literal-value-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_bool_literal_value_run.mind");
    let so = dir.join("mind_bool_literal_value_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("bool-literal-value-run: needs mlir-build; skipping");
            return;
        }
        panic!("bool-literal-value-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('from_let','from_field','direct'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.from_let(); assert r == 7, 'from_let=' + str(r)\n\
         r = lib.from_field(); assert r == 1, 'from_field=' + str(r)\n\
         r = lib.direct(); assert r == 1, 'direct=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "bool-literal-value-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
