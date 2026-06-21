// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Match-arm STATEMENT bodies RUNTIME gate.
//!
//! A match arm body may be a `{ … }` block, an expression, OR a statement:
//! `return …`, `continue` / `break`, or a bare assignment `lhs = rhs`
//! (Rust-ish mutating arms — `Ok(p) => set = f(set, p)`, `Err(_) => continue`).
//! `parse_expr` handles none of the statement forms, so without explicit
//! handling the match loop mis-reads the remainder as a new arm pattern
//! ("expected pattern"). Compiles a program exercising the assignment + continue
//! arm bodies to a `.so`, dlopen-calls it, and asserts the accumulated value.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test match_arm_stmt_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
enum R { Keep(i64), Skip }

// Assignment arm body (`acc = acc + p`) + `continue` arm body. Sums 0..5 = 10.
pub fn accumulate() -> i64 {
    let mut acc = 0
    let mut i = 0
    while i < 5 {
        let r = R.Keep(i)
        match r {
            R.Keep(p) => acc = acc + p,
            R.Skip => continue,
        }
        i = i + 1
    }
    return acc
}

// `break` arm body: stop at the first Skip. Sums 0,1,2 then breaks at 3 = 3.
pub fn stop_early() -> i64 {
    let mut acc = 0
    let mut i = 0
    while i < 10 {
        let r = if i < 3 { R.Keep(i) } else { R.Skip }
        match r {
            R.Keep(p) => acc = acc + p,
            R.Skip => break,
        }
        i = i + 1
    }
    return acc
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
fn match_arm_stmt_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("match-arm-stmt-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_match_arm_stmt_run.mind");
    let so = dir.join("mind_match_arm_stmt_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("match-arm-stmt-run: needs mlir-build; skipping");
            return;
        }
        panic!("match-arm-stmt-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('accumulate','stop_early'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.accumulate(); assert r == 10, 'accumulate=' + str(r)\n\
         r = lib.stop_early(); assert r == 3, 'stop_early=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "match-arm-stmt-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
