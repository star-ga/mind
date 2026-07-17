// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Narrow-int (`i8`/`u8`/`i16`/`u16`) FUNCTION-SIGNATURE ABI gate.
//!
//! The i64-SLOT narrow-signature ABI (`src/eval/lower.rs`): an `i8/u8/i16/u16`
//! param arrives in an i64 slot and is materialised at its declared width on
//! fn ENTRY (zext `BitAnd` mask for unsigned, sext shift-pair for signed), and
//! a narrow RETURN is masked to its declared width at every return site — so
//! the value handed back in the i64 slot is the canonical zero/sign-extended
//! representation. The MLIR `func.func` signature stays i64-typed throughout
//! (unlike `i32`/`u32`, which lower via the physical-i32 ABI covered by
//! `narrow_call_abi.rs`).
//!
//! Adversarial cases asserted (not just the happy path):
//!   * u8 sum that FITS (120+80 = 200) and one that WRAPS (200+100 = 300 → 44);
//!   * u16 fit (60000) and wrap (70000 → 4464);
//!   * i8 ENTRY sign-extension: a raw 200 in the slot must be seen as -56
//!     (`sign_i8(200) == -1`), proving the callee masks on entry rather than
//!     trusting the caller's slot bits;
//!   * i8 RETURN sign-extension: `neg_i8(56) == -56` as a full i64 (the slot
//!     carries the sign-extended value, not a zero-padded 200);
//!   * early narrow return inside an if-branch, and a narrow IMPLICIT tail
//!     return (both masked at their own return sites).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test narrow_sig_abi_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
pub fn add_u8(a: u8, b: u8) -> u8 {
    return a + b
}
pub fn add_u16(a: u16, b: u16) -> u16 {
    return a + b
}
pub fn neg_i8(x: i8) -> i8 {
    return 0 - x
}
pub fn sign_i8(x: i8) -> i64 {
    if x < 0 {
        return 0 - 1
    }
    return 1
}
// early narrow return inside an if-branch (masked at the branch return site)
pub fn clamp_u8(x: u8) -> u8 {
    if x > 100 {
        return x + 200
    }
    return x
}
// narrow IMPLICIT tail return (masked at the FnDef tail site)
pub fn tail_u8(a: u8) -> u8 {
    a + 250
}

pub fn t_u8_fits() -> i64 {
    return add_u8(120, 80)
}
pub fn t_u8_wraps() -> i64 {
    return add_u8(200, 100)
}
pub fn t_u16_fits() -> i64 {
    return add_u16(30000, 30000)
}
pub fn t_u16_wraps() -> i64 {
    return add_u16(60000, 10000)
}
pub fn t_i8_entry_sext() -> i64 {
    return sign_i8(200)
}
pub fn t_i8_ret_sext() -> i64 {
    return neg_i8(56)
}
pub fn t_i8_ret_wrap() -> i64 {
    return neg_i8(200)
}
pub fn t_u8_branch_wrap() -> i64 {
    return clamp_u8(120)
}
pub fn t_u8_tail_wrap() -> i64 {
    return tail_u8(10)
}
"#;

#[test]
fn narrow_signature_i64_slot_abi() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-sig-abi: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_narrow_sig_abi.mind");
    let so = dir.join("mind_narrow_sig_abi.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("narrow-sig-abi: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("narrow-sig-abi: mindc --emit-shared failed:\n{stderr}");
    }

    // Flat top-level python statements only — see narrow_call_abi.rs for why
    // (no indented blocks survive the `\`-continuation).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         checks = [('t_u8_fits', 200), ('t_u8_wraps', 44), ('t_u16_fits', 60000), \
         ('t_u16_wraps', 4464), ('t_i8_entry_sext', -1), ('t_i8_ret_sext', -56), \
         ('t_i8_ret_wrap', 56), ('t_u8_branch_wrap', 64), ('t_u8_tail_wrap', 4)]\n\
         call = lambda f: (setattr(f, 'restype', ctypes.c_int64), f())[1]\n\
         results = [(name, want, call(getattr(lib, name))) for name, want in checks]\n\
         bad = [r for r in results if r[1] != r[2]]\n\
         assert not bad, 'FAIL: ' + repr(bad)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "narrow-sig-abi value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
