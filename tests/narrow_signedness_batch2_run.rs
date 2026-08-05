// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Narrow-integer signedness silent-miscompile batch 2 regression gate.
//!
//! Each function reproduces a NET-VERIFIED wrong-value bug and asserts the
//! correct value via `--emit-shared` + ctypes. Bundled findings:
//!
//! * Finding 5 — a builtin `Option`/`Result` match payload bound at the
//!   prelude's generic `[ScalarI64]`, so `o: Option<u64>` holding `u64::MAX`
//!   read back as `-1` and `v < 1` was wrongly true. `desugar_match_to_if` now
//!   synthesizes the payload type from the scrutinee's DECLARED generic args
//!   (tracked in `NARROW_LOCALS`), restricted to INTEGER args.
//! * Finding 6 — the numeric tuple index `t.0` did not parse, and a tuple's
//!   `u64` element had no typed read path. The parser now emits a digit-field
//!   `FieldAccess`; lowering resolves it against the receiver's declared tuple
//!   annotation, bounds-checks, loads `base + 8*N`, and re-materialises the
//!   element at its declared type. Any unresolved/out-of-range use fails LOUD
//!   at compile time (asserted below) — never a struct-resolution fallthrough.
//! * Finding 7 — a mixed-width narrow binop (`a: u8 + b: u16`) wrapped at the
//!   LEFT operand's width, so `(a + b) / 2` and `(b + a) / 2` disagreed.
//!   `join_narrow_tys` now picks the WIDEST operand; equal widths keep the
//!   LEFT operand (byte-preserving for `u8+u8` and `i8+u8`).
//!
//! Controls prove each fix is NARROW: `Option<i64>`/unannotated matches, the
//! `i64` tuple element, same-width and one-sided narrow arithmetic all keep
//! their prior values.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test narrow_signedness_batch2_run`

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
// ── Finding 5: Option<u64> payload must bind UNSIGNED (u64::MAX < 1 is false).
pub fn opt_u64_lt() -> i64 {
    let x: u64 = (0 - 1) as u64
    let o: Option<u64> = Some(x)
    match o {
        Some(v) => if v < 1 { 1 } else { 0 },
        None => 2
    }
}
// ── Finding 5: Result<u64, _> Ok payload.
pub fn res_u64_ok() -> i64 {
    let x: u64 = (0 - 1) as u64
    let r: Result<u64, i64> = Ok(x)
    match r {
        Ok(v) => if v < 1 { 1 } else { 0 },
        Err(e) => 2
    }
}
// ── Finding 5: Result<_, u64> Err payload uses args[1].
pub fn res_u64_err() -> i64 {
    let x: u64 = (0 - 1) as u64
    let r: Result<i64, u64> = Err(x)
    match r {
        Ok(v) => 2,
        Err(e) => if e < 1 { 1 } else { 0 }
    }
}
// ── Finding 5 control: Option<i64> stays SIGNED (-1 < 1 is true).
pub fn opt_i64_ctl() -> i64 {
    let x: i64 = 0 - 1
    let o: Option<i64> = Some(x)
    match o {
        Some(v) => if v < 1 { 1 } else { 0 },
        None => 2
    }
}
// ── Finding 5 control: an UNANNOTATED scrutinee keeps today's path.
pub fn opt_unann_ctl() -> i64 {
    let o = Some(7)
    match o {
        Some(v) => v,
        None => 0
    }
}

// ── Finding 6: tuple u64 element compares UNSIGNED via `t.0`.
pub fn tup_u64_elem() -> i64 {
    let t: (u64, i64) = ((0 - 1) as u64, 5)
    if t.0 < 1 { 1 } else { 0 }
}
// ── Finding 6 control: the i64 element reads its plain value.
pub fn tup_i64_elem() -> i64 {
    let t: (u64, i64) = ((0 - 1) as u64, 5)
    return t.1
}

// ── Finding 7: mixed-width narrow binop joins to the WIDEST operand, so both
//    operand orders agree: (250 + 1000) fits u16 → 1250/2 == 625.
pub fn mixed_lr(a: u8, b: u16) -> i64 { return (a + b) / 2 }
pub fn mixed_rl(a: u8, b: u16) -> i64 { return (b + a) / 2 }
// ── Finding 7 control: same-width u8+u8 unchanged ((250+4)&255)/2 == 127.
pub fn same_u8(a: u8, b: u8) -> i64 { return (a + b) / 2 }
// ── Finding 7 control: right operand non-narrow → left width, unchanged.
pub fn right_none(x: u8) -> i64 { return (x + 10) / 2 }
// ── Finding 7 control: EQUAL width keeps the LEFT operand (i8 vs u8):
//    (-6 + 4) at i8 → -2/2 == -1; at u8 → 254/2 == 127 (pre-change values).
pub fn i8_u8(a: i8, b: u8) -> i64 { return (a + b) / 2 }
pub fn u8_i8(a: i8, b: u8) -> i64 { return (b + a) / 2 }
"#;

// Finding 6 negative control: an OUT-OF-BOUNDS tuple index must be a loud
// compile-time error, never a silent load past the aggregate.
const SRC_OOB: &str = r#"
pub fn oob() -> i64 {
    let t: (u64, i64) = (1, 5)
    return t.2
}
"#;

// Finding 6 negative control: a receiver WITHOUT a declared tuple annotation
// must fail loud, never fall through to struct-field/const-0 resolution.
const SRC_UNANN: &str = r#"
pub fn unann(x: i64) -> i64 {
    let t = x
    return t.0
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn narrow_signedness_batch2_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-signedness-batch2-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_narrow_signedness_batch2_run.mind");
    let so = dir.join("mind_narrow_signedness_batch2_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("narrow-signedness-batch2-run: needs mlir-build; skipping");
            return;
        }
        panic!("narrow-signedness-batch2-run: mindc --emit-shared failed:\n{stderr}");
    }

    // (fn, ctypes-arg exprs, expected) — one case per finding + controls.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         def sig(n):\n\
         \x20   f = getattr(lib, n); f.restype = ctypes.c_int64; return f\n\
         cases = [\n\
         \x20 ('opt_u64_lt',   lambda f: f(),                                        0),\n\
         \x20 ('res_u64_ok',   lambda f: f(),                                        0),\n\
         \x20 ('res_u64_err',  lambda f: f(),                                        0),\n\
         \x20 ('opt_i64_ctl',  lambda f: f(),                                        1),\n\
         \x20 ('opt_unann_ctl',lambda f: f(),                                        7),\n\
         \x20 ('tup_u64_elem', lambda f: f(),                                        0),\n\
         \x20 ('tup_i64_elem', lambda f: f(),                                        5),\n\
         \x20 ('mixed_lr',     lambda f: f(ctypes.c_uint8(250), ctypes.c_uint16(1000)), 625),\n\
         \x20 ('mixed_rl',     lambda f: f(ctypes.c_uint8(250), ctypes.c_uint16(1000)), 625),\n\
         \x20 ('same_u8',      lambda f: f(ctypes.c_uint8(250), ctypes.c_uint8(4)),  127),\n\
         \x20 ('right_none',   lambda f: f(ctypes.c_uint8(250)),                       2),\n\
         \x20 ('i8_u8',        lambda f: f(ctypes.c_int8(-6), ctypes.c_uint8(4)),     -1),\n\
         \x20 ('u8_i8',        lambda f: f(ctypes.c_int8(-6), ctypes.c_uint8(4)),    127),\n\
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
        "narrow-signedness-batch2-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // Finding 6 negative controls: both must FAIL to compile (loud reject).
    for (label, bad_src, needle) in [
        ("oob", SRC_OOB, "out of bounds"),
        ("unann", SRC_UNANN, "declared tuple annotation"),
    ] {
        let bad = dir.join(format!("mind_narrow_signedness_batch2_{label}.mind"));
        let bad_so = dir.join(format!("mind_narrow_signedness_batch2_{label}.so"));
        std::fs::write(&bad, bad_src).expect("write bad src");
        let out = Command::new(&mindc)
            .args([
                bad.to_str().unwrap(),
                "--emit-shared",
                bad_so.to_str().unwrap(),
            ])
            .output()
            .expect("run mindc");
        assert!(
            !out.status.success(),
            "narrow-signedness-batch2-run: `{label}` tuple-index misuse must be \
             a loud compile error, but mindc succeeded"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(needle),
            "narrow-signedness-batch2-run: `{label}` failed for the wrong \
             reason (expected `{needle}`):\n{stderr}"
        );
    }
}
