// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `loop { … }` unconditional-loop RUNTIME gate.
//!
//! `loop { body }` desugars to `while 1 { body }`, reusing the while machinery
//! (break/continue, region-scoped exit SSA). The body must `break` or `return` to
//! terminate. This compiles a program to a `.so`, dlopen-calls it, and asserts
//! both exit paths produce the right value.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test loop_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// Terminate via `break`, accumulating across iterations.
pub fn sum_to(n: i64) -> i64 {
    let mut i = 0
    let mut sum = 0
    loop {
        if i >= n {
            break
        }
        sum = sum + i
        i = i + 1
    }
    return sum
}

// Terminate via `return` directly out of the loop.
pub fn first_divisor(n: i64) -> i64 {
    let mut i = 2
    loop {
        if n - (n / i) * i == 0 {
            return i
        }
        i = i + 1
    }
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
fn loop_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("loop-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_loop_run.mind");
    let so = dir.join("mind_loop_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("loop-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("loop-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.sum_to.restype = ctypes.c_int64; lib.sum_to.argtypes = [ctypes.c_int64]\n\
         lib.first_divisor.restype = ctypes.c_int64; lib.first_divisor.argtypes = [ctypes.c_int64]\n\
         r = lib.sum_to(5); assert r == 10, 'sum_to(5)=' + str(r)\n\
         r = lib.first_divisor(15); assert r == 3, 'first_divisor(15)=' + str(r)\n\
         r = lib.first_divisor(49); assert r == 7, 'first_divisor(49)=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "loop-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
