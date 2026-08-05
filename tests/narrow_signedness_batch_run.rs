// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Narrow-integer signedness/width silent-miscompile batch regression gate.
//!
//! Each function reproduces a NET-VERIFIED wrong-value bug and asserts the
//! correct value via `--emit-shared` + ctypes. Bundled findings:
//!
//! * Finding 1 (regression from the narrow-locals registry) — a narrow `let`
//!   shadow-widened to i64 (`shadow_wide`), or a narrow `let` inside a branch
//!   whose entry LEAKED past the `if` (`scope_leak`), was still masked to 8
//!   bits: `1010/2` gave `505>>` masked `121`. `record_narrow_let` now clears
//!   the entry on a wide re-let, and each branch body block-scopes its
//!   narrow-local additions.
//! * Finding 2 — a narrow value produced by a CALL (`-> u8`), a struct FIELD
//!   (`u8`), or a CAST (`x as u8`) escaped the intermediate-wrap mask.
//!   `infer_narrow_arith_ty` now resolves those operand kinds.
//! * Finding 3 — a narrow (8/16-bit) shift result was not re-masked and its
//!   COUNT was masked mod 64 not mod width: `1u8 << 8` gave 0 not 1.
//! * Finding 4 — an i64-LITERAL left operand of a shift computed at the u32
//!   count's width: `(1 << 33) as i64` gave 2 not 8589934592.
//!
//! Controls prove each fix is NARROW: the already-correct `u8`/`u32`/`i32`
//! cases stay correct.
//!
//! Findings 5 (Option/Result<u64> payload signedness) and 6 (tuple u64
//! element) are DEFERRED — see the `// deferred: Finding 5` marker in
//! `src/eval/lower.rs` and the batch report; Finding 6's `t.0` numeric tuple
//! index does not parse under public `mindc`.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test narrow_signedness_batch_run`

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
// ── Finding 1: shadow-widen — a u8 let re-let at i64 must NOT re-mask.
pub fn shadow_wide(x0: i64) -> i64 { let x: u8 = 5; let x: i64 = x0; return (x + 10) / 2 }
// ── Finding 1: branch-scope leak — a u8 let inside a branch must not leak.
pub fn scope_leak(x0: i64) -> i64 { if x0 > 0 { let t: u8 = 1 }; let t: i64 = x0; return (t + 10) / 2 }
// ── Finding 1 control: a genuine u8 let still wraps: (250+10)&255 / 2 == 2.
pub fn keep_u8() -> i64 { let x: u8 = 250; return (x + 10) / 2 }

// ── Finding 2: narrow value from a CALL escapes the intermediate-wrap mask.
fn gu8() -> u8 { return 250 }
pub fn call_arith_gap() -> i64 { return (gu8() + 10) / 2 }
// ── Finding 2: narrow value from a struct FIELD.
struct S8 { a: u8 }
pub fn field_arith_gap() -> i64 { let s: S8 = S8 { a: 250 }; return (s.a + 10) / 2 }
// ── Finding 2: narrow value from a CAST source.
pub fn cast_arith_gap(x: i64) -> i64 { return ((x as u8) + 10) / 2 }

// ── Finding 3: narrow (u8) shift result masked to width; (400&255)/2 == 72.
pub fn shl_arith_gap(x: u8) -> i64 { return (x << 1) / 2 }
// ── Finding 3: u8/u16 variable shift COUNT masked mod width: 1<<(8 mod 8)==1.
pub fn shl_u8_u8(a: u8, b: u8) -> i64 { return (a << b) as i64 }
pub fn shl_u16_u16(a: u16, b: u16) -> i64 { return (a << b) as i64 }

// ── Finding 3/4 controls: the already-correct u32/i32 shift cases.
pub fn shl_u32_u8(a: u32, b: u8) -> i64 { return (a << b) as i64 }
pub fn shr_u32_sign(a: u32, b: u32) -> i64 { return (a >> b) as i64 }

// ── Finding 4: i64-literal left operand of a shift used as i64 == 1<<33.
pub fn shl_lit_u32(n: u32) -> i64 { return (1 << n) as i64 }
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn narrow_signedness_batch_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-signedness-batch-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_narrow_signedness_batch_run.mind");
    let so = dir.join("mind_narrow_signedness_batch_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("narrow-signedness-batch-run: needs mlir-build; skipping");
            return;
        }
        panic!("narrow-signedness-batch-run: mindc --emit-shared failed:\n{stderr}");
    }

    // (fn, ctypes-arg exprs, expected) — one case per finding + controls.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         def sig(n):\n\
         \x20   f = getattr(lib, n); f.restype = ctypes.c_int64; return f\n\
         cases = [\n\
         \x20 ('shadow_wide',    lambda f: f(ctypes.c_int64(1000)),                 505),\n\
         \x20 ('scope_leak',     lambda f: f(ctypes.c_int64(1000)),                 505),\n\
         \x20 ('keep_u8',        lambda f: f(),                                       2),\n\
         \x20 ('call_arith_gap', lambda f: f(),                                       2),\n\
         \x20 ('field_arith_gap',lambda f: f(),                                       2),\n\
         \x20 ('cast_arith_gap', lambda f: f(ctypes.c_int64(250)),                    2),\n\
         \x20 ('shl_arith_gap',  lambda f: f(ctypes.c_uint8(200)),                   72),\n\
         \x20 ('shl_u8_u8',      lambda f: f(ctypes.c_uint8(1), ctypes.c_uint8(8)),   1),\n\
         \x20 ('shl_u16_u16',    lambda f: f(ctypes.c_uint16(1), ctypes.c_uint16(16)),1),\n\
         \x20 ('shl_u32_u8',     lambda f: f(ctypes.c_uint32(1), ctypes.c_uint8(32)), 1),\n\
         \x20 ('shr_u32_sign',   lambda f: f(ctypes.c_uint32(0x80000000), ctypes.c_uint32(31)), 1),\n\
         \x20 ('shl_lit_u32',    lambda f: f(ctypes.c_uint32(33)),           8589934592),\n\
         ]\n\
         for name, call, exp in cases:\n\
         \x20   r = call(sig(name))\n\
         \x20   assert r == exp, name + '=' + str(r) + ' expected ' + str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "narrow-signedness-batch-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
