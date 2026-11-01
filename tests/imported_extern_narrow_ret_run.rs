// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! IR-audit #1 — narrow-scalar `extern "C"` RETURN ABI, end-to-end.
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
//!                   --test imported_extern_narrow_ret_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const C_STUB: &str = r#"
#include <stdint.h>
int8_t echo_i8(int8_t x) { return x; }
int16_t echo_i16(int16_t x) { return x; }
int32_t echo_i32(int32_t x) { return x; }
uint32_t echo_u32(uint32_t x) { return x; }
int8_t   ret_i8_neg1(void)   { return -1; }
int16_t  ret_i16_neg1(void)  { return -1; }
int32_t  ret_i32_neg1(void)  { return -1; }
uint8_t  ret_u8_200(void)    { return 200; }
uint16_t ret_u16_60000(void) { return 60000; }
uint32_t ret_u32_max(void)   { return 4294967295u; }
_Bool    ret_bool_true(void) { return 1; }

/* Dirty the high 32 bits of rax, then return int -1 (writes only eax). A
 * correct narrow-return ABI reads only eax and recovers -1; the old i64
 * declaration would surface the poisoned high word. The register name is
 * x86-64 only; other targets return -1 without that poison. */
#if defined(__x86_64__)
int32_t ret_i32_neg1_dirty(void) {
    register uint64_t poison asm("rax") = 0xDEADBEEF00000000ULL;
    asm volatile("" : "+r"(poison));
    return -1;
}
#else
int32_t ret_i32_neg1_dirty(void) { return -1; }
#endif
"#;

const MIND_SRC: &str = r#"
extern "C" {
    fn echo_i8(x: i8) -> i8
    fn echo_i16(x: i16) -> i16
    fn echo_i32(x: i32) -> i32
    fn echo_u32(x: u32) -> u32
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
pub fn t_arg_i8() -> i64 { return echo_i8(0 - 7); }
pub fn t_arg_i16() -> i64 { return echo_i16(0 - 300); }
pub fn t_arg_i32() -> i64 { return echo_i32(0 - 70000); }
pub fn t_arg_u32() -> i64 { return echo_u32(4294967295); }
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
fn imported_extern_narrow_return_sign_zero_extension() {
    imported_abi(false, false);
}

#[test]
fn explicitly_exported_extern_narrow_arguments_and_returns() {
    imported_abi(true, false);
}

#[test]
fn unimported_extern_does_not_change_selected_abi() {
    imported_abi(true, true);
}

fn imported_abi(explicit: bool, conflicting_sibling: bool) {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "imported_extern_narrow_ret_run",
            "extern-narrow-ret: mindc not found; skipping",
        );
        return;
    }
    let scratch = tempfile::tempdir().expect("scratch");
    let dir = scratch.path();
    std::fs::write(
        dir.join("Mind.toml"),
        r#"[package]
name = "imported_abi"
version = "0.1.0"
[build]
entry = "main.mind"
"#,
    )
    .unwrap();
    std::fs::create_dir(dir.join("src")).unwrap();
    let cstub = dir.join("mind_extern_narrow_ret_stub.c");
    let cso = dir.join("mind_extern_narrow_ret_stub.so");
    let src = dir.join("mind_extern_narrow_ret.mind");
    let so = dir.join("mind_extern_narrow_ret.so");
    std::fs::write(&cstub, C_STUB).expect("write c stub");
    let (declarations, body) = MIND_SRC.split_once("pub fn t_i8_neg1").unwrap();
    if conflicting_sibling {
        std::fs::write(
            dir.join("src/unrelated.mind"),
            "export { ret_i32_neg1 }\nextern \"C\" { fn ret_i32_neg1() -> i64 }\n",
        )
        .unwrap();
    }
    let exports = if explicit {
        "export { ret_i8_neg1, ret_i16_neg1, ret_i32_neg1, ret_u8_200, ret_u16_60000, ret_u32_max, ret_bool_true, ret_i32_neg1_dirty, echo_i8, echo_i16, echo_i32, echo_u32 }\n"
    } else {
        ""
    };
    std::fs::write(dir.join("src/ffi.mind"), format!("{exports}{declarations}")).unwrap();
    std::fs::write(&src, format!("import ffi;\npub fn t_i8_neg1{body}")).expect("write mind src");
    std::fs::write(dir.join("main.mind"), "fn main() -> i64 { return 0; }").unwrap();

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
        .current_dir(dir)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Flat top-level python statements only (see narrow_call_abi.rs for why).
    let py = format!(
        "import ctypes\n\
         stub = ctypes.CDLL(r'{cso}', mode=ctypes.RTLD_GLOBAL)\n\
         lib = ctypes.CDLL(r'{so}')\n\
         checks = [('t_i8_neg1', -1), ('t_i16_neg1', -1), ('t_i32_neg1', -1), \
         ('t_u8_200', 200), ('t_u16_60000', 60000), ('t_u32_max', 4294967295), \
         ('t_bool_true', 1), ('t_i32_dirty', -1), ('t_error_check_fires', 1), ('t_arg_i8', -7), ('t_arg_i16', -300), ('t_arg_i32', -70000), ('t_arg_u32', 4294967295)]\n\
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

/// Narrow arguments passed as narrow MIND values, linked to a `clang -O2`
/// stub. `clang -O2` trusts `signext` / `zeroext`, so a swapped attribute
/// changes i8/i16/u8/u16/bool. i32 and u32 already occupy a 32-bit register
/// on this ABI, so those two are checked on the `llvm.func` declaration.
const WIDEN_C: &str = r#"
#include <stdint.h>
int32_t widen_i8(int8_t x) { return x; }
int32_t widen_i16(int16_t x) { return x; }
int32_t widen_i32(int32_t x) { return x; }
int32_t widen_u8(uint8_t x) { return x; }
int32_t widen_u16(uint16_t x) { return x; }
int32_t widen_u32(uint32_t x) { return x; }
int32_t widen_bool(_Bool x) { return x; }
"#;

const WIDEN_MIND: &str = r#"
extern "C" {
    fn widen_i8(x: i8) -> i32
    fn widen_i16(x: i16) -> i32
    fn widen_i32(x: i32) -> i32
    fn widen_u8(x: u8) -> i32
    fn widen_u16(x: u16) -> i32
    fn widen_u32(x: u32) -> i32
    fn widen_bool(x: bool) -> i32
}
"#;

const WIDEN_BODY: &str = r#"
import ffi;
pub fn t_i8() -> i64 { let x: i8 = (0 - 7) as i8; return widen_i8(x); }
pub fn t_i16() -> i64 { let x: i16 = (0 - 300) as i16; return widen_i16(x); }
pub fn t_i32() -> i64 { let x: i32 = (0 - 7) as i32; return widen_i32(x); }
pub fn t_u8() -> i64 { let x: u8 = 200 as u8; return widen_u8(x); }
pub fn t_u16() -> i64 { let x: u16 = 60000 as u16; return widen_u16(x); }
pub fn t_u32() -> i64 { let x: u32 = 3000000000 as u32; return widen_u32(x); }
pub fn t_bool() -> i64 { let b: bool = 0 < 1; return widen_bool(b); }
"#;

struct WidenRun {
    values: std::collections::BTreeMap<String, i64>,
    mlir: String,
}

fn widen_run() -> &'static WidenRun {
    use std::sync::OnceLock;
    static CELL: OnceLock<WidenRun> = OnceLock::new();
    CELL.get_or_init(build_widen_run)
}

fn clang_o2(cstub: &std::path::Path, cso: &std::path::Path) {
    let mut detail = String::new();
    for cc in ["clang", "clang-20", "clang-23"] {
        match Command::new(cc)
            .args([
                "-shared",
                "-fPIC",
                "-O2",
                cstub.to_str().unwrap(),
                "-o",
                cso.to_str().unwrap(),
            ])
            .output()
        {
            Ok(out) if out.status.success() => return,
            Ok(out) => {
                detail = format!("{cc}: {}", String::from_utf8_lossy(&out.stderr));
            }
            Err(err) => detail = format!("{cc}: {err}"),
        }
    }
    panic!("clang -O2 did not build the narrow-argument stub: {detail}");
}

fn build_widen_run() -> WidenRun {
    let mindc = mindc_bin();
    assert!(
        mindc.exists(),
        "mindc is required for the narrow-argument gate"
    );
    let scratch = tempfile::tempdir().expect("scratch");
    let dir = scratch.path();
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"widen_abi\"\nversion = \"0.1.0\"\n[build]\nentry = \"main.mind\"\n",
    )
    .unwrap();
    std::fs::create_dir(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/ffi.mind"), WIDEN_MIND).unwrap();
    std::fs::write(dir.join("main.mind"), "fn main() -> i64 { return 0; }\n").unwrap();
    let src = dir.join("widen.mind");
    std::fs::write(&src, WIDEN_BODY).unwrap();
    let cstub = dir.join("widen.c");
    let cso = dir.join("widen.so");
    let so = dir.join("widen_mind.so");
    std::fs::write(&cstub, WIDEN_C).unwrap();
    clang_o2(&cstub, &cso);
    let out = Command::new(&mindc)
        .current_dir(dir)
        .args([
            src.to_str().unwrap(),
            "--emit-mlir",
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    assert!(
        out.status.success(),
        "mindc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mlir = String::from_utf8_lossy(&out.stdout).to_string();
    let py = format!(
        "import ctypes\n\
         ctypes.CDLL(r'{cso}', mode=ctypes.RTLD_GLOBAL)\n\
         lib = ctypes.CDLL(r'{so}')\n\
         names = ['t_i8','t_i16','t_i32','t_u8','t_u16','t_u32','t_bool']\n\
         call = lambda f: (setattr(f, 'restype', ctypes.c_int64), f())[1]\n\
         print('\\n'.join(n + ' ' + str(call(getattr(lib, n))) for n in names))\n",
        cso = cso.to_string_lossy(),
        so = so.to_string_lossy(),
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        ran.status.success(),
        "narrow-argument run failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );
    let mut values = std::collections::BTreeMap::new();
    for line in String::from_utf8_lossy(&ran.stdout).lines() {
        let mut parts = line.split_whitespace();
        let (Some(name), Some(raw)) = (parts.next(), parts.next()) else {
            continue;
        };
        values.insert(name.to_string(), raw.parse::<i64>().expect("value"));
    }
    assert_eq!(values.len(), 7, "expected seven narrow-argument results");
    WidenRun { values, mlir }
}

fn expect_widen(name: &str, value: i64, decl: &str) {
    let run = widen_run();
    assert_eq!(run.values[name], value, "{name} value");
    assert!(
        run.mlir.contains(decl),
        "{name} declaration missing `{decl}`\n{}",
        run.mlir
            .lines()
            .filter(|line| line.contains("llvm.func @widen_"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn narrow_i8_argument_is_sign_extended() {
    expect_widen("t_i8", -7, "llvm.func @widen_i8(i8 {llvm.signext})");
}

#[test]
fn narrow_i16_argument_is_sign_extended() {
    expect_widen("t_i16", -300, "llvm.func @widen_i16(i16 {llvm.signext})");
}

#[test]
fn narrow_u8_argument_is_zero_extended() {
    expect_widen("t_u8", 200, "llvm.func @widen_u8(i8 {llvm.zeroext})");
}

#[test]
fn narrow_u16_argument_is_zero_extended() {
    expect_widen("t_u16", 60000, "llvm.func @widen_u16(i16 {llvm.zeroext})");
}

#[test]
fn narrow_bool_argument_is_zero_extended() {
    expect_widen("t_bool", 1, "llvm.func @widen_bool(i1 {llvm.zeroext})");
}

#[test]
fn narrow_i32_argument_declares_signext() {
    expect_widen("t_i32", -7, "llvm.func @widen_i32(i32 {llvm.signext})");
}

#[test]
fn narrow_u32_argument_declares_zeroext() {
    // 3000000000 does not fit in signed i32. The C return is that bit pattern.
    expect_widen(
        "t_u32",
        -1_294_967_296,
        "llvm.func @widen_u32(i32 {llvm.zeroext})",
    );
}
