// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! Typed / radix integer literals in MATCH PATTERN position — RUNTIME gate.
//!
//! The pattern parser accepted only bare decimal int literals: a radix-prefixed
//! literal (`0x00`, `0o7`, `0b10`) or a type-suffixed one (`0u8`, `1i8`) in an
//! arm pattern hit `error[parse][E1001]: expected `=>` after match pattern`
//! (the suffix / hex tail was left unconsumed). The fix mirrors the
//! expression-position numeric parser: read the radix digits and consume the
//! optional `u8`/`i8`/… width suffix (a pattern matches on the VALUE, so the
//! width is annotation only). This is the exact form `mind-flow`'s
//! `bitnet.mind` uses (`match code { 0x00u8 => 0i8, … }`).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test typed_literal_match_pattern_run`

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
fn classify(c: i64) -> i64 {
    match c {
        0x00   => 100,
        0xFF   => 200,
        0o17   => 700,
        0b101  => 500,
        42u8   => 142,
        _      => 999,
    }
}
"#;

#[test]
fn typed_literal_match_patterns_run() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("typed-literal-match-pattern-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_typed_literal_match_pattern_run.mind");
    let so = dir.join("mind_typed_literal_match_pattern_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("typed-literal-match-pattern-run: needs mlir-build; skipping");
            return;
        }
        panic!("typed-literal-match-pattern-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.classify; f.restype = ctypes.c_int64; f.argtypes = [ctypes.c_int64]\n\
         for arg, exp in ((0,100),(255,200),(15,700),(5,500),(42,142),(7,999)):\n\
         \x20   got = f(arg)\n\
         \x20   assert got == exp, 'classify('+str(arg)+'): got='+str(got)+' expected='+str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "typed-literal-match-pattern-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
