// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Decimal digit-separator literal RUNTIME gate.
//!
//! A decimal integer literal may carry `_` digit separators between digits
//! (`120_000`, `1_000_000`), exactly like the radix-prefixed path already
//! accepts. The separators are dropped — the value of `120_000` is `120000`.
//! This shape is pervasive in real source (`const DEFAULT_TIMEOUT_MS: u64 =
//! 120_000`). The keystone source uses no separators, so its emit stays
//! byte-identical.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test digit_separator_run`

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
const DEFAULT_TIMEOUT_MS: u64 = 120_000

pub fn run() -> i64 {
    return DEFAULT_TIMEOUT_MS as i64
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn digit_separator_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("digit-separator-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_digit_separator_run.mind");
    let so = dir.join("mind_digit_separator_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("digit-separator-run: needs mlir-build; skipping");
            return;
        }
        panic!("digit-separator-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 120000, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "digit-separator-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
