// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Bare (unqualified) enum-variant constructor + pattern RUNTIME gate.
//!
//! MIND links every module into one global unit, so a variant with a SOLE
//! owning enum can be referenced UNQUALIFIED (`Good(v)` / `Just(x)`) in BOTH
//! construction position and `match` patterns. Previously only the qualified
//! `Opt::Just(42)` form worked; a bare `Just(42)` failed `E2003 unsupported
//! call`, and a bare `Just(v)` pattern fell through to the catch-all (wrong
//! arm). This compiles a program to a `.so`, dlopen-calls it, and asserts the
//! rewrap values.
//!
//! The USER enums here deliberately use names with NO prelude collision
//! (`Good`/`Bad`/`Just`, not `Ok`/`Err`/`Some`): after audit finding #8, a bare
//! name owned by BOTH a user enum and the `Option`/`Result` prelude is
//! fail-closed ambiguous (qualify it), so a fixture testing SOLE-owner bare
//! resolution must not shadow prelude names. Bare resolution of the prelude
//! names themselves (no shadowing enum → sole owner) is covered by
//! `result_option_prelude_run`; the ambiguity fail-closed by
//! `bare_variant_ambiguity_run`.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test bare_variant_ctor_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
enum Res { Good(i64), Bad(i64) }

// Bare Good/Bad in BOTH construction and pattern position, with a re-wrap.
fn dbl(r: Res) -> Res {
    match r {
        Good(v) => Good(v + v),
        Bad(e) => Bad(e),
    }
}

pub fn run_ok() -> i64 {
    let r = dbl(Good(21))
    match r {
        Good(v) => v,
        Bad(e) => 0 - 1,
    }
}

pub fn run_err() -> i64 {
    let r = dbl(Bad(7))
    match r {
        Good(v) => v,
        Bad(e) => e,
    }
}

enum Opt { Just(i64), Nothing }

// Bare Just construction + bare Just(v) payload-binding pattern.
pub fn some_val() -> i64 {
    let o = Just(42)
    match o {
        Just(v) => v,
        Nothing => 0,
    }
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn bare_variant_ctor_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bare-variant-ctor-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_bare_variant_ctor_run.mind");
    let so = dir.join("mind_bare_variant_ctor_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("bare-variant-ctor-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("bare-variant-ctor-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('run_ok','run_err','some_val'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.run_ok(); assert r == 42, 'run_ok=' + str(r)\n\
         r = lib.run_err(); assert r == 7, 'run_err=' + str(r)\n\
         r = lib.some_val(); assert r == 42, 'some_val=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "bare-variant-ctor-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
