// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Built-in Result/Option PRELUDE RUNTIME gate.
//!
//! MIND has no source-level prelude, but Rust-style code expects `Result<T,E>` /
//! `Option<T>` with bare `Ok`/`Err`/`Some`/`None` WITHOUT defining the enums.
//! The compiler registers them in the boxed-enum side-tables (no emitted
//! instructions, so the keystone — which uses neither — stays byte-identical),
//! so construction and `match` resolve via the bare-constructor path. This
//! compiles a program that uses Result/Option WITHOUT declaring either, builds a
//! `.so`, dlopen-calls it, and asserts the values — including the fieldless
//! `None` (which must lower to a real `[tag,0]` record, not a bare ordinal).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test result_option_prelude_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// No `enum Result` / `enum Option` declared anywhere — the prelude provides them.
fn dbl(r: Result<i64, i64>) -> Result<i64, i64> {
    match r {
        Ok(v) => Ok(v + v),
        Err(e) => Err(e),
    }
}
pub fn r_ok() -> i64 {
    let r = dbl(Ok(21))
    match r { Ok(v) => v, Err(e) => 0 - 1 }
}
pub fn r_err() -> i64 {
    let r = dbl(Err(7))
    match r { Ok(v) => v, Err(e) => e }
}

fn pick(flag: i64) -> Option<i64> {
    if flag > 0 { return Some(99) }
    return None
}
pub fn o_some() -> i64 {
    match pick(1) { Some(v) => v, None => 0 - 1 }
}
pub fn o_none() -> i64 {
    match pick(0) { Some(v) => v, None => 0 - 1 }
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn result_option_prelude_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("result-option-prelude-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_result_option_prelude_run.mind");
    let so = dir.join("mind_result_option_prelude_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("result-option-prelude-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("result-option-prelude-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('r_ok','r_err','o_some','o_none'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.r_ok(); assert r == 42, 'r_ok=' + str(r)\n\
         r = lib.r_err(); assert r == 7, 'r_err=' + str(r)\n\
         r = lib.o_some(); assert r == 99, 'o_some=' + str(r)\n\
         r = lib.o_none(); assert r == -1, 'o_none=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "result-option-prelude-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
