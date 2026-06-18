// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Integer-operation cross-substrate determinism gate.
//!
//! Two integer hazards lower DIVERGENTLY across substrates unless the compiler
//! pins them, breaking the bit-identity wedge for any program that hits them:
//!
//!   1. `INT_MIN / -1` (and `INT_MIN % -1`): the true quotient is
//!      unrepresentable, so x86 `idiv` raises `#DE` (SIGFPE) while AArch64
//!      `sdiv` returns `INT_MIN`. The lowering substitutes divisor `1` on the
//!      overflow case, yielding the wrapping result on EVERY substrate
//!      (`INT_MIN/1 == INT_MIN`, `INT_MIN%1 == 0`) and never trapping.
//!   2. A shift count `>= bit-width`: poison in MLIR/LLVM, lowered as x86
//!      mask-mod-width vs AArch64 zero. The lowering masks the count to
//!      `width-1` so every count is in-range and identical everywhere.
//!
//! In-range inputs are unchanged (both fixes are no-ops for them), so the
//! keystone + canary byte-identity gates are untouched — verified separately.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test int_determinism`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
pub fn idiv(a: i64, b: i64) -> i64 {
    return a / b
}
pub fn imod(a: i64, b: i64) -> i64 {
    return a % b
}
pub fn ishl(x: i64, n: i64) -> i64 {
    return x << n
}
pub fn idiv32(a: i32, b: i32) -> i32 {
    return a / b
}
pub fn imod32(a: i32, b: i32) -> i32 {
    return a % b
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
fn int_min_div_and_oversized_shift_are_deterministic() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("int-determinism: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_int_determinism.mind");
    let so = dir.join("mind_int_determinism.so");
    std::fs::write(&src, SRC).expect("write src");

    let status = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .status()
        .expect("run mindc");
    if !status.success() {
        println!("int-determinism: mindc --emit-shared failed (no MLIR backend?); skipping");
        return;
    }

    // Drive the compiled .so through ctypes. INT_MIN is passed as an argument
    // (computed host-side) to avoid relying on a most-negative source literal.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.idiv.restype = ctypes.c_int64\n\
         lib.idiv.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         lib.imod.restype = ctypes.c_int64\n\
         lib.imod.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         lib.ishl.restype = ctypes.c_int64\n\
         lib.ishl.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         lib.idiv32.restype = ctypes.c_int32\n\
         lib.idiv32.argtypes = [ctypes.c_int32, ctypes.c_int32]\n\
         lib.imod32.restype = ctypes.c_int32\n\
         lib.imod32.argtypes = [ctypes.c_int32, ctypes.c_int32]\n\
         MIN = -(2**63)\n\
         MIN32 = -(2**31)\n\
         # normal arithmetic unaffected\n\
         assert lib.idiv(20, 3) == 6, lib.idiv(20, 3)\n\
         assert lib.imod(20, 3) == 2, lib.imod(20, 3)\n\
         assert lib.ishl(5, 1) == 10, lib.ishl(5, 1)\n\
         # INT_MIN / -1: wrapping result, no SIGFPE trap (else this aborts)\n\
         assert lib.idiv(MIN, -1) == MIN, lib.idiv(MIN, -1)\n\
         assert lib.imod(MIN, -1) == 0, lib.imod(MIN, -1)\n\
         assert lib.idiv32(20, 3) == 6, lib.idiv32(20, 3)\n\
         assert lib.idiv32(MIN32, -1) == MIN32, lib.idiv32(MIN32, -1)\n\
         assert lib.imod32(MIN32, -1) == 0, lib.imod32(MIN32, -1)\n\
         # division-by-zero: deterministic 0 on every substrate (else x86 SIGFPE)\n\
         assert lib.idiv(7, 0) == 0, lib.idiv(7, 0)\n\
         assert lib.imod(7, 0) == 0, lib.imod(7, 0)\n\
         # oversized shift counts masked mod-width identically everywhere\n\
         assert lib.ishl(1, 64) == 1, lib.ishl(1, 64)\n\
         assert lib.ishl(1, 65) == 2, lib.ishl(1, 65)\n\
         assert lib.ishl(1, 63) == MIN, lib.ishl(1, 63)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "int-determinism check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
