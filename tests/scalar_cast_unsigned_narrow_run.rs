// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unsigned-narrow cast `u8`/`u16`/`u32` truncation RUNTIME gate.
//!
//! Regression for a verified silent miscompile: unsigned narrowing casts
//! (`x as u8/u16/u32` and the call-form `u8(x)`/`u16(x)`/`u32(x)`) used to lower
//! to a NO-OP — the full i64 passed through with the high bits intact. The fix
//! emits a zero-extend `val & mask` (BitAnd against an i64 const). This test
//! compiles out-of-range inputs and asserts the result is actually TRUNCATED.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test scalar_cast_unsigned_narrow_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// 70000 & 0xFFFF == 4464 — call-form u16 must drop the high bits.
pub fn cast_u16_call() -> i64 {
    return i64(u16(70000))
}

// 70000 as u16 == 4464 — postfix `as` form.
pub fn cast_u16_as() -> i64 {
    let x = 70000
    return i64(x as u16)
}

// 257 & 0xFF == 1 — u8 narrowing.
pub fn cast_u8_call() -> i64 {
    return i64(u8(257))
}

pub fn cast_u8_as() -> i64 {
    let x = 257
    return i64(x as u8)
}

// 0x1_0000_0001 & 0xFFFF_FFFF == 1 — u32 narrowing (call form -> ScalarU32).
pub fn cast_u32_call() -> i64 {
    return i64(u32(4294967297))
}

// postfix `as u32` -> ScalarU32 path.
pub fn cast_u32_as() -> i64 {
    let x = 4294967297
    return i64(x as u32)
}

// In-range values must be preserved (no spurious truncation).
pub fn cast_inrange() -> i64 {
    return i64(u16(4464)) + i64(u8(1)) + i64(u32(1))
}

// Signed casts must STILL sign-extend (regression guard for the signed path).
// 257 as i8 == 1 ; -1 as i8 stays -1 ; 0x1FF as i16 -> 511.
pub fn cast_i8_signext() -> i64 {
    let neg = 0 - 1
    return (257 as i8) + (neg as i8) + (511 as i16)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn scalar_cast_unsigned_narrow_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("scalar-cast-unsigned-narrow-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_scalar_cast_unsigned_narrow_run.mind");
    let so = dir.join("mind_scalar_cast_unsigned_narrow_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("scalar-cast-unsigned-narrow-run: needs mlir-build; skipping");
            return;
        }
        panic!("scalar-cast-unsigned-narrow-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         names = ('cast_u16_call','cast_u16_as','cast_u8_call','cast_u8_as',\n\
                  'cast_u32_call','cast_u32_as','cast_inrange','cast_i8_signext')\n\
         for _n in names: getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.cast_u16_call(); assert r == 4464, 'cast_u16_call=' + str(r)\n\
         r = lib.cast_u16_as();   assert r == 4464, 'cast_u16_as=' + str(r)\n\
         r = lib.cast_u8_call();  assert r == 1,    'cast_u8_call=' + str(r)\n\
         r = lib.cast_u8_as();    assert r == 1,    'cast_u8_as=' + str(r)\n\
         r = lib.cast_u32_call(); assert r == 1,    'cast_u32_call=' + str(r)\n\
         r = lib.cast_u32_as();   assert r == 1,    'cast_u32_as=' + str(r)\n\
         r = lib.cast_inrange();  assert r == 4466, 'cast_inrange=' + str(r)\n\
         r = lib.cast_i8_signext(); assert r == 511, 'cast_i8_signext=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "scalar-cast-unsigned-narrow-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
