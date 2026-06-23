// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Value-`if` whose branches yield a comparison (i1) into an i64 merge.
//!
//! `let b: bool = if c { x > 10 } else { y > 100 }` produces an `i1` (the
//! `cmpi` result) in each branch, but the merge block argument is typed i64, so
//! the branch `cf.br ^merge(%cmp : i64)` mismatched (`'i64' vs 'i1'`) and
//! `mlir-opt` failed. The branch sub-contexts now bubble their `i1_values` up so
//! the merge recognises the value is physically i1 and widens it (`extui`) to
//! i64 before the block argument. A value-`if` over ordinary i64 values never
//! takes this path, so the keystone + cross-substrate byte-identity gates are
//! untouched.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test value_if_comparison`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
pub fn f(c: i64, x: i64, y: i64) -> i64 {
    let b: bool = if c > 0 {
        x > 10
    } else {
        y > 100
    }
    if b {
        return 1
    }
    return 0
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn value_if_yielding_a_comparison_lowers() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("value-if-comparison: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_value_if_comparison.mind");
    let so = dir.join("mind_value_if_comparison.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("value-if-comparison: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("value-if-comparison: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.f.restype = ctypes.c_int64\n\
         lib.f.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]\n\
         r = lib.f(1, 20, 0); assert r == 1, 'f(1,20,0)=' + str(r)   # c>0 -> x>10 -> 20>10 true\n\
         r = lib.f(1, 5, 0);  assert r == 0, 'f(1,5,0)=' + str(r)    # x>10 -> 5>10 false\n\
         r = lib.f(0, 0, 200); assert r == 1, 'f(0,0,200)=' + str(r) # else y>100 -> 200>100 true\n\
         r = lib.f(0, 0, 50);  assert r == 0, 'f(0,0,50)=' + str(r)  # else y>100 -> 50>100 false\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "value-if-comparison check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
