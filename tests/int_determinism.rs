// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Integer-operation cross-substrate determinism gate.
//!
//! Two integer hazards lower DIVERGENTLY across substrates unless the compiler
//! pins them, breaking the bit-identity wedge for any program that hits them:
//!
//!   1. `INT_MIN / -1` (and `INT_MIN % -1`) and `x / 0` (and `x % 0`): x86
//!      `idiv` raises `#DE` (SIGFPE) while AArch64 `sdiv` returns a value.
//!      MIND pins both (W2-trap, docs/errors.md "Integer division trap"): for
//!      i64 every one of the four cases terminates the process with exit
//!      status 79 on EVERY substrate, never a signal and never a made-up value.
//!      For the narrow signed widths (i32 here) a zero divisor traps the same
//!      way, while `MIN / -1` wraps to `MIN` (remainder 0), which is what the
//!      native backend computes too. A trapping call ends the process, so each
//!      trap case runs in its own child Python process and the test asserts
//!      that child's exit status (79) and that nothing after the call ran.
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

mod common;
use common::mindc_bin;

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

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn int_min_div_and_oversized_shift_are_deterministic() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "int_determinism",
            "int-determinism: mindc not found; skipping",
        );
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_int_determinism.mind");
    let so = dir.join("mind_int_determinism.so");
    std::fs::write(&src, SRC).expect("write src");

    let compile_out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !crate::common::gate::compiled("int_determinism", &compile_out) {
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
         # neighbours of the trap edge are ordinary divisions\n\
         assert lib.idiv(MIN, -2) == 2**62, lib.idiv(MIN, -2)\n\
         assert lib.imod(MIN, 1) == 0, lib.imod(MIN, 1)\n\
         assert lib.idiv(-7, 2) == -3, lib.idiv(-7, 2)\n\
         assert lib.imod(-7, 2) == -1, lib.imod(-7, 2)\n\
         # narrow i32: MIN / -1 wraps to MIN (remainder 0), no trap\n\
         assert lib.idiv32(20, 3) == 6, lib.idiv32(20, 3)\n\
         assert lib.idiv32(MIN32, -1) == MIN32, lib.idiv32(MIN32, -1)\n\
         assert lib.imod32(MIN32, -1) == 0, lib.imod32(MIN32, -1)\n\
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

    // W2-trap: each trapping call runs in its own child process, which must
    // exit with status 79 right at the call (the line after it never prints).
    let traps = [
        "lib.idiv(MIN, -1)",
        "lib.imod(MIN, -1)",
        "lib.idiv(7, 0)",
        "lib.imod(7, 0)",
        "lib.idiv(MIN, 0)",
        "lib.idiv32(7, 0)",
        "lib.imod32(7, 0)",
    ];
    for call in traps {
        let py = format!(
            "import ctypes\n\
             lib = ctypes.CDLL(r'{}')\n\
             for n in ('idiv', 'imod'):\n    \
             getattr(lib, n).restype = ctypes.c_int64\n    \
             getattr(lib, n).argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
             for n in ('idiv32', 'imod32'):\n    \
             getattr(lib, n).restype = ctypes.c_int32\n    \
             getattr(lib, n).argtypes = [ctypes.c_int32, ctypes.c_int32]\n\
             MIN = -(2**63)\n\
             print('before', flush=True)\n\
             {call}\n\
             print('after', flush=True)\n",
            so.to_string_lossy()
        );
        let out = Command::new("python3")
            .args(["-c", &py])
            .output()
            .expect("python3");
        assert_eq!(
            out.status.code(),
            Some(79),
            "`{call}` must exit 79 (division trap); status {:?}\nstdout: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "before\n",
            "`{call}`: the trap must end the process at the call"
        );
    }
}
