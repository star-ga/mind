// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Fable IR-audit #1 — narrow-scalar `extern "C"` RETURN ABI, end-to-end.
//!
//! Regression gate for a HIGH silent-miscompile: an `extern "C" { fn f(...) ->
//! i32 }` used to be declared `llvm.func @f(...) -> i64`, so the call read the
//! FULL `rax`. A C callee returning `int` writes only `eax`, so a `-1` result
//! (`eax = 0xFFFF_FFFF`, `rax` high bits ABI-unspecified) was mis-recovered as
//! `4294967295` instead of `-1`. Downstream `if fd < 0` error checks (std/fs
//! `open`/`mkdir`, std/process `spawn`) then NEVER fired — a silent error swallow
//! — and sub-32-bit widths read undefined high bits (a cross-substrate break).
//!
//! The fix declares the REAL narrow width and sign/zero-extends the result to
//! the i64 MIND value at the call site. This test links a C stub returning
//! known narrow values and asserts MIND reads them correctly:
//!   * `i8`/`i16`/`i32` = -1  →  MIND reads -1 (sign-extended, NOT 255/65535/2^32-1);
//!   * `u8`  = 200, `u16` = 60000, `u32` = 4294967295  →  read verbatim (zero-extended);
//!   * `_Bool` = 1  →  read as 1 (zero-extended to {0,1}).
//!
//! The poisoned-high-bits stub (`ret_i32_neg1_dirty`) deliberately dirties the
//! high 32 bits of `rax` before returning `int -1`, proving the fix reads only
//! `eax` (the old i64 declaration would have surfaced the poison).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test extern_narrow_ret_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const C_STUB: &str = r#"
#include <stdint.h>
int8_t   ret_i8_neg1(void)   { return -1; }
int16_t  ret_i16_neg1(void)  { return -1; }
int32_t  ret_i32_neg1(void)  { return -1; }
uint8_t  ret_u8_200(void)    { return 200; }
uint16_t ret_u16_60000(void) { return 60000; }
uint32_t ret_u32_max(void)   { return 4294967295u; }
_Bool    ret_bool_true(void) { return 1; }

/* Dirty the high 32 bits of rax, then return int -1 (writes only eax). A
 * correct narrow-return ABI reads only eax and recovers -1; the old i64
 * declaration would surface the poisoned high word. */
int32_t ret_i32_neg1_dirty(void) {
    register uint64_t poison asm("rax") = 0xDEADBEEF00000000ULL;
    asm volatile("" : "+r"(poison));
    return -1;
}
"#;

const MIND_SRC: &str = r#"
extern "C" {
    fn ret_i8_neg1() -> i8
    fn ret_i16_neg1() -> i16
    fn ret_i32_neg1() -> i32
    fn ret_u8_200() -> u8
    fn ret_u16_60000() -> u16
    fn ret_u32_max() -> u32
    fn ret_bool_true() -> bool
    fn ret_i32_neg1_dirty() -> i32
}

pub fn t_i8_neg1() -> i64 {
    return ret_i8_neg1()
}
pub fn t_i16_neg1() -> i64 {
    return ret_i16_neg1()
}
pub fn t_i32_neg1() -> i64 {
    return ret_i32_neg1()
}
pub fn t_u8_200() -> i64 {
    return ret_u8_200()
}
pub fn t_u16_60000() -> i64 {
    return ret_u16_60000()
}
pub fn t_u32_max() -> i64 {
    return ret_u32_max()
}
pub fn t_bool_true() -> i64 {
    return ret_bool_true()
}
pub fn t_i32_dirty() -> i64 {
    return ret_i32_neg1_dirty()
}
// The live-impact pattern: an i32 error return of -1 must make `< 0` fire.
pub fn t_error_check_fires() -> i64 {
    let fd = ret_i32_neg1()
    if fd < 0 {
        return 1
    }
    return 0
}
"#;

#[test]
fn extern_narrow_return_sign_zero_extension() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("extern-narrow-ret: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let cstub = dir.join("mind_extern_narrow_ret_stub.c");
    let cso = dir.join("mind_extern_narrow_ret_stub.so");
    let src = dir.join("mind_extern_narrow_ret.mind");
    let so = dir.join("mind_extern_narrow_ret.so");
    std::fs::write(&cstub, C_STUB).expect("write c stub");
    std::fs::write(&src, MIND_SRC).expect("write mind src");

    // Compile the C stub into its own shared object (RTLD_GLOBAL-loaded first so
    // the MIND .so resolves the extern symbols against it).
    let cc = Command::new("cc")
        .args([
            "-shared",
            "-fPIC",
            "-O0",
            cstub.to_str().unwrap(),
            "-o",
            cso.to_str().unwrap(),
        ])
        .output()
        .expect("run cc");
    assert!(
        cc.status.success(),
        "cc failed to build stub:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("extern-narrow-ret: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("extern-narrow-ret: mindc --emit-shared failed:\n{stderr}");
    }

    // Flat top-level python statements only (see narrow_call_abi.rs for why).
    let py = format!(
        "import ctypes\n\
         stub = ctypes.CDLL(r'{cso}', mode=ctypes.RTLD_GLOBAL)\n\
         lib = ctypes.CDLL(r'{so}')\n\
         checks = [('t_i8_neg1', -1), ('t_i16_neg1', -1), ('t_i32_neg1', -1), \
         ('t_u8_200', 200), ('t_u16_60000', 60000), ('t_u32_max', 4294967295), \
         ('t_bool_true', 1), ('t_i32_dirty', -1), ('t_error_check_fires', 1)]\n\
         call = lambda f: (setattr(f, 'restype', ctypes.c_int64), f())[1]\n\
         results = [(name, want, call(getattr(lib, name))) for name, want in checks]\n\
         bad = [r for r in results if r[1] != r[2]]\n\
         assert not bad, 'FAIL: ' + repr(bad)\n\
         print('ok:', results)\n",
        cso = cso.to_string_lossy(),
        so = so.to_string_lossy(),
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "extern-narrow-ret value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    println!("{}", String::from_utf8_lossy(&out.stdout));
}
