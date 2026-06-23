// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Nested-block lowering RUNTIME gate — tuple-destructuring lets, dotted-ident
//! enum patterns, and tuple-payload match patterns inside `if`/`while` bodies.
//!
//! These constructs reach the value-position lowering path through the
//! `if`-then / `if`-else / `while`-body block loops (not the fn-body statement
//! sequence), so each needs an explicit statement arm there:
//!   * `let (a, b) = pair()` — `LetTuple` in an `if`/`while` body.
//!   * `match k { Kind.Scalar => … }` — a fieldless enum variant written with
//!     DOT notation parses as a bare `Ident` pattern; it must resolve to its
//!     discriminant tag rather than a catch-all binding.
//!   * `match parse(x) { Ok((a, b)) => acc = a + b, … }` — a tuple sub-pattern
//!     inside an enum payload, with an `Assign` arm body.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test nested_block_surface_run`

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
enum Kind { Scalar, Tensor, Struct }

fn classify(k: Kind) -> i64 {
    match k {
        Kind.Scalar => return 1,
        Kind.Tensor => return 2,
        Kind.Struct => return 3,
    }
}

fn pair() -> (i64, i64) {
    return (10, 20)
}

fn use_lettuple_in_if(c: i64) -> i64 {
    if c > 0 {
        let (a, b) = pair()
        return a + b
    }
    return 0
}

fn use_lettuple_in_while(n: i64) -> i64 {
    let mut i = 0
    let mut acc = 0
    while i < n {
        let (a, b) = pair()
        acc = acc + a + b
        i = i + 1
    }
    return acc
}

fn parse(x: i64) -> Result<(i64, i64), i64> {
    if x > 0 {
        return Ok((x, x + 1))
    }
    return Err(7)
}

fn use_tuple_payload(x: i64) -> i64 {
    let mut acc = 0
    match parse(x) {
        Ok((a, b)) => acc = a + b,
        Err(e) => acc = e,
    }
    return acc
}

pub fn run() -> i64 {
    let a = classify(Kind.Tensor)
    let b = use_lettuple_in_if(1)
    let c = use_tuple_payload(5)
    let d = use_tuple_payload(-1)
    let e = use_lettuple_in_while(3)
    return a * 100000 + b * 1000 + c * 100 + d * 10 + e
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn nested_block_surface_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("nested-block-surface-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_nested_block_surface_run.mind");
    let so = dir.join("mind_nested_block_surface_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("nested-block-surface-run: needs mlir-build; skipping");
            return;
        }
        panic!("nested-block-surface-run: mindc --emit-shared failed:\n{stderr}");
    }

    // a=2 (Kind.Tensor), b=30 (10+20), c=11 (Ok(5,6)), d=7 (Err(7)),
    // e=90 (3 * (10+20)) → 2*100000 + 30*1000 + 11*100 + 7*10 + 90 = 231260.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 231260, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "nested-block-surface-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
