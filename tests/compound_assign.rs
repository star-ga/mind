// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Compound-assignment operator gate (`+= -= *= /= %= &= |= ^= <<= >>=`).
//!
//! These desugar at PARSE time to `lhs = lhs OP rhs` (zero new IR), so the gate
//! both proves the parse succeeds (the operators were previously a hard
//! `expected expression` error — e.g. the `examples/policy.mind` showcase used
//! `+=` and did not build) and that each desugars to the correct arithmetic.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test compound_assign`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
pub fn t_add() -> i64 {
    let mut i: i64 = 0
    i += 5
    i += 3
    return i
}
pub fn t_sub() -> i64 {
    let mut i: i64 = 10
    i -= 4
    return i
}
pub fn t_mul() -> i64 {
    let mut i: i64 = 3
    i *= 4
    return i
}
pub fn t_div() -> i64 {
    let mut i: i64 = 20
    i /= 5
    return i
}
pub fn t_mod() -> i64 {
    let mut i: i64 = 17
    i %= 5
    return i
}
pub fn t_shl() -> i64 {
    let mut i: i64 = 1
    i <<= 4
    return i
}
pub fn t_shr() -> i64 {
    let mut i: i64 = 256
    i >>= 2
    return i
}
pub fn t_or() -> i64 {
    let mut i: i64 = 1
    i |= 2
    return i
}
pub fn t_and() -> i64 {
    let mut i: i64 = 6
    i &= 3
    return i
}
pub fn t_xor() -> i64 {
    let mut i: i64 = 5
    i ^= 1
    return i
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
fn compound_assignment_desugars_and_computes() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("compound-assign: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_compound_assign.mind");
    let so = dir.join("mind_compound_assign.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("compound-assign: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("compound-assign: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.t_add.restype = ctypes.c_int64\n\
         r = lib.t_add(); assert r == 8, 't_add=' + str(r)\n\
         lib.t_sub.restype = ctypes.c_int64\n\
         r = lib.t_sub(); assert r == 6, 't_sub=' + str(r)\n\
         lib.t_mul.restype = ctypes.c_int64\n\
         r = lib.t_mul(); assert r == 12, 't_mul=' + str(r)\n\
         lib.t_div.restype = ctypes.c_int64\n\
         r = lib.t_div(); assert r == 4, 't_div=' + str(r)\n\
         lib.t_mod.restype = ctypes.c_int64\n\
         r = lib.t_mod(); assert r == 2, 't_mod=' + str(r)\n\
         lib.t_shl.restype = ctypes.c_int64\n\
         r = lib.t_shl(); assert r == 16, 't_shl=' + str(r)\n\
         lib.t_shr.restype = ctypes.c_int64\n\
         r = lib.t_shr(); assert r == 64, 't_shr=' + str(r)\n\
         lib.t_or.restype = ctypes.c_int64\n\
         r = lib.t_or(); assert r == 3, 't_or=' + str(r)\n\
         lib.t_and.restype = ctypes.c_int64\n\
         r = lib.t_and(); assert r == 2, 't_and=' + str(r)\n\
         lib.t_xor.restype = ctypes.c_int64\n\
         r = lib.t_xor(); assert r == 4, 't_xor=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "compound-assign value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
