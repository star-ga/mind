// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! UNSIGNED narrow div/rem by zero must yield the deterministic contract value
//! `0` (matching the signed-i32 and i64 div guards) — NOT trap. The narrow
//! BinOp arm guarded the SIGNED `divsi`/`remsi` paths (INT_MIN/-1 + divisor-0)
//! but routed `u32` `divui`/`remui` straight to a bare `arith.divui : i32`,
//! which lowers to x86 `divl` and raises `#DE` (SIGFPE / process crash) on a
//! zero divisor, while AArch64 `udiv` returns 0 with no trap — a cross-substrate
//! divergence AND a crash where every other integer width returns 0. The fix
//! adds the unsigned divisor-0 guard (substitute 1, force the result to 0).
//!
//! Repro before the fix: `f(100, 0)` for `a / b : u32` crashed with SIGFPE.
//! After the fix it returns 0, and every non-zero divisor is unchanged.
//!
//! Gate: `cargo test --release --features "std-surface mlir-build
//!                   cross-module-imports" --test narrow_unsigned_div_zero_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Compile `fn f(a: <ty>, b: <ty>) -> <ty> { <body> }`, then call `f(a, b)`
/// through ctypes (reading the result as an UNSIGNED 32-bit value). Returns
/// the printed integer, or `i64::MIN` if the call crashed / produced no output
/// (a SIGFPE leaves stdout empty — that is exactly the pre-fix failure).
fn call_f(ty: &str, body: &str, a: i64, b: i64, tag: &str) -> i64 {
    let mindc = mindc_bin();
    let dir = std::env::temp_dir();
    let s = dir.join(format!("mind_udivz_{tag}.mind"));
    let so = dir.join(format!("mind_udivz_{tag}.so"));
    let src = format!("pub fn f(a: {ty}, b: {ty}) -> {ty} {{\n{body}\n}}\n");
    std::fs::write(&s, src).expect("write");
    let out = Command::new(&mindc)
        .args([s.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    assert!(
        out.status.success(),
        "unsigned-div compile failed ({tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // c_uint32 result + c_int64 args (the i64 ABI slot carries narrow args).
    let py = format!(
        "import ctypes\n\
         lib=ctypes.CDLL(r'{}')\n\
         lib.f.argtypes=[ctypes.c_int64, ctypes.c_int64]\n\
         lib.f.restype=ctypes.c_uint32\n\
         print(lib.f({a}, {b}))\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("py");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(i64::MIN)
}

fn mlir_build_available() -> bool {
    let mindc = mindc_bin();
    if !mindc.exists() {
        return false;
    }
    let dir = std::env::temp_dir();
    let s = dir.join("mind_udivz_probe.mind");
    std::fs::write(&s, "pub fn run() -> i64 { return 0 }\n").unwrap();
    let o = Command::new(&mindc)
        .args([
            s.to_str().unwrap(),
            "--emit-shared",
            dir.join("udivz_probe.so").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let e = String::from_utf8_lossy(&o.stderr);
    !(e.contains("mlir-build") && e.contains("requires"))
}

#[test]
fn unsigned_narrow_div_by_zero_is_deterministic_zero() {
    if !mlir_build_available() {
        println!("unsigned-div-zero: mindc/mlir-build unavailable; skipping");
        return;
    }

    // THE BUG: `u32 / 0` and `u32 % 0` used to SIGFPE-crash. Now they must
    // return the deterministic contract value 0 (same as signed/i64). A crash
    // would leave stdout empty -> i64::MIN, so this assert is the regression.
    assert_eq!(call_f("u32", "    return a / b", 100, 0, "div0"), 0);
    assert_eq!(call_f("u32", "    return a % b", 100, 0, "mod0"), 0);

    // Non-zero divisors are UNCHANGED and still use UNSIGNED semantics:
    // 0xFFFFFFFF / 2 == 2147483647 (unsigned), not 0 (which `-1 / 2` signed gives).
    assert_eq!(call_f("u32", "    return a / b", 4_294_967_295, 2, "udiv"), 2_147_483_647);
    // 0xFFFFFFFF % 3 == 0 unsigned (4294967295 = 3 * 1431655765).
    assert_eq!(call_f("u32", "    return a % b", 4_294_967_295, 3, "umod"), 0);
    // Plain in-range arithmetic is untouched.
    assert_eq!(call_f("u32", "    return a / b", 100, 7, "div7"), 14);
    assert_eq!(call_f("u32", "    return a % b", 100, 7, "mod7"), 2);

    // A constant non-zero divisor ELIDES the guard but must still compute the
    // same unsigned value (the elision is purely a size optimisation).
    assert_eq!(call_f("u32", "    return a / 7", 100, 0, "divc"), 14);
}
