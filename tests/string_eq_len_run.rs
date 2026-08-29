// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native string `.len()` RUNTIME gate (issue #245, `.len()` half).
//!
//! An annotated `let s: string` was tracked by nothing, so `s.len()` missed the
//! `string_<method>` dispatch, fell through to the struct/field path and lowered
//! to `const.i64 0`. It built clean to a real ELF, exited 0, and silently
//! computed 0 with no diagnostic and no JIT-fallback banner.
//!
//! This asserts VALUES through the real runtime, not that the source parses.
//! The integer arm is the control: `==` on i64 must be untouched.
//!
//! The `==`-on-strings half of #245 is NOT covered here — it is still open; see
//! that issue for why the lowering-side interception was withdrawn.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_eq_len_run`

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

// `.len()` on an annotated string local (was 0 — unresolved receiver).
pub fn ann_len() -> i64 {
    let s: string = "abcd"
    return s.len()
}


// CONTROL: i64 `==` must be byte-for-byte unaffected by the string routing.
pub fn int_eq_unchanged() -> i64 {
    let a: i64 = 3
    if a == 3 { return 7 }
    return 0
}
"#;

#[test]
fn string_len_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("string-eq-len-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_string_eq_len_run.mind");
    let so = dir.join("mind_string_eq_len_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("string-eq-len-run: needs mlir-build; skipping");
            return;
        }
        panic!("string-eq-len-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('ann_len','int_eq_unchanged'):\n\
         \x20   getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.ann_len(); assert r == 4, 'ann_len=' + str(r)\n\
         r = lib.int_eq_unchanged(); assert r == 7, 'int_eq_unchanged=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-eq-len-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
