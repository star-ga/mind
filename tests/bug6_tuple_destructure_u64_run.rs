// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! #290 corr2 BUG6 — tuple-DESTRUCTURE loses a `u64` element's unsigned tag.
//!
//! A tuple destructure `let (v, ignored) = t` reads each slot with a bare
//! `__mind_load_i64` and bound the name with NO width/signedness. A full-width
//! `u64` element therefore lost the `__mind_conv_u64`/`ScalarU64` tag it carried
//! before being stored into the tuple heap slot, so a later `v >> 63` lowered to
//! `arith.shrsi` (ARITHMETIC shift) instead of `arith.shrui` (LOGICAL):
//!
//!   let x: u64 = (0 - 1) as u64   // 2^64-1
//!   let t = (x, 0)
//!   let (v, ignored) = t
//!   return v >> 63                // returned -1 (signed), MUST be 1
//!
//! (net-verified via ctypes.) The fix mirrors the tuple-INDEX element read
//! (`t.N`): the destructure re-materialises each element at its declared or
//! SYNTHESISED tuple type (`mask_narrow_let`), and an unannotated `let t = (x, 0)`
//! synthesises the same `TypeAnn::Tuple` an explicit `let t: (u64, i64) = …`
//! records. All three destructure element-load sites (fn-body, block-local, and
//! the shared `lower_lettuple_stmt` used by if/while blocks) are fixed.
//!
//! Adversarial controls prove the fix is not over-applied: an i64 element still
//! shifts SIGNED (`kat_i64_signed_shift` -> -1), and the u64 tuple-INDEX path is
//! byte-untouched (`kat_u64_index` -> 1).
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test bug6_tuple_destructure_u64_run`

#![cfg(all(unix, feature = "std-surface", feature = "mlir-build"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// RED repro: unannotated let-bound tuple, destructured through an Ident RHS.
// 2^64-1 >> 63 unsigned is 1 (was -1, arithmetic shift of the signed repr).
pub fn kat_u64_destr() -> i64 {
    let x: u64 = (0 - 1) as u64
    let t = (x, 0)
    let (v, ignored) = t
    return (v >> 63) + ignored
}

// Direct-literal destructure (RHS is the tuple literal itself, not an Ident) —
// exercises the per-element inference branch of the resolver.
pub fn kat_u64_destr_lit() -> i64 {
    let x: u64 = (0 - 1) as u64
    let (v, ignored) = (x, 0)
    return (v >> 63) + ignored
}

// Explicitly ANNOTATED tuple destructure — resolves the u64 element from the
// declared `TypeAnn::Tuple` (no synthesis), so the destructure re-tags unsigned.
pub fn kat_u64_destr_annotated() -> i64 {
    let t: (u64, i64) = ((0 - 1) as u64, 0)
    let (v, ignored) = t
    return (v >> 63) + ignored
}

// GREEN control: an i64-element destructure returns its correct value.
pub fn kat_i64_destr_value() -> i64 {
    let a: i64 = 42
    let t = (a, 7)
    let (v, w) = t
    return v + w
}

// ADVERSARIAL control: an i64 element must still shift SIGNED (arith.shrsi).
// -1 >> 63 signed is -1; proves the fix does NOT tag i64 elements unsigned.
pub fn kat_i64_signed_shift() -> i64 {
    let a: i64 = 0 - 1
    let t = (a, 0)
    let (v, w) = t
    return (v >> 63) + w
}

// CONTROL: the u64 tuple-INDEX path (`t.0`) is untouched and still returns 1.
pub fn kat_u64_index() -> i64 {
    let x: u64 = (0 - 1) as u64
    let t: (u64, i64) = (x, 0)
    return (t.0 >> 63) + t.1
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn bug6_tuple_destructure_u64_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug6-tuple-destructure-u64-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_bug6_tuple_destructure_u64_run.mind");
    let so = dir.join("mind_bug6_tuple_destructure_u64_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("bug6-tuple-destructure-u64-run: needs mlir-build; skipping");
            return;
        }
        panic!("bug6-tuple-destructure-u64-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         names = ('kat_u64_destr','kat_u64_destr_lit','kat_u64_destr_annotated',\n\
         \x20        'kat_i64_destr_value','kat_i64_signed_shift','kat_u64_index')\n\
         for _n in names: getattr(lib, _n).restype = ctypes.c_int64\n\
         r = lib.kat_u64_destr();           assert r == 1,  'kat_u64_destr=' + str(r)\n\
         r = lib.kat_u64_destr_lit();       assert r == 1,  'kat_u64_destr_lit=' + str(r)\n\
         r = lib.kat_u64_destr_annotated(); assert r == 1,  'kat_u64_destr_annotated=' + str(r)\n\
         r = lib.kat_i64_destr_value();     assert r == 49, 'kat_i64_destr_value=' + str(r)\n\
         r = lib.kat_i64_signed_shift();    assert r == -1, 'kat_i64_signed_shift=' + str(r)\n\
         r = lib.kat_u64_index();           assert r == 1,  'kat_u64_index=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "bug6-tuple-destructure-u64-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
