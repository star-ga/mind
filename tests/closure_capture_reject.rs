// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! #290 corr2 BUG7 regression — non-i64 closure capture is rejected fail-closed.
//!
//! Phase-1 closures store every capture in a fixed `i64` env field. A captured
//! value whose declared type occupies that slot DIFFERENTLY than `i64` —
//! `u64`/`usize` (bit 63 makes shift arithmetic-vs-logical and compare/div/rem
//! signed-vs-unsigned diverge) or `f32`/`f64` (bit reinterpretation) — was
//! SILENTLY MISCOMPUTED: `let k: u64 = u64::MAX; |k; x| { k >> 63 }` returned
//! `-1` (arithmetic shift of the signed reinterpretation) instead of `1`
//! (net-verified via ctypes). The desugarer now threads a name->declared-type
//! scope (fn params + preceding annotated `let`s) into `build_closure_items` and
//! REJECTS an i64-slot-divergent capture with E2600 rather than fake-support it.
//!
//! Load-bearing non-over-rejection: `bool` and `i8..i64` and `u8..u32` sext/zext
//! into the i64 slot EXACTLY (a <=32-bit unsigned value never sets bit 63), so
//! those captures MUST still compile — asserted by the two GREEN controls.
//!
//! The reject fires during `desugar_closures` (compile path only; `mindc check`
//! does not desugar), and the repros build via `--emit-shared`, so this gate
//! needs `mlir-build` + `std-surface` — a `cargo test --no-default-features` run
//! compiles it out entirely (the CI Test step; the mlir feature set runs it).
#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::io::Write;
use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// Combined stdout+stderr of a full `mindc <path> --emit-shared <tmp.so>` compile.
fn build_out(name: &str, src: &str) -> String {
    let dir = std::env::temp_dir();
    let mp = dir.join(format!("{name}.mind"));
    let sp = dir.join(format!("{name}.so"));
    let mut f = std::fs::File::create(&mp).expect("write src");
    f.write_all(src.as_bytes()).expect("write src bytes");
    let out = mindc()
        .args([mp.to_str().unwrap(), "--emit-shared", sp.to_str().unwrap()])
        .output()
        .expect("spawn mindc");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn closure_u64_capture_rejected() {
    // A `u64` capture would be shifted arithmetically in the i64 env slot
    // (`u64::MAX >> 63` == -1 not 1). Must reject fail-closed (E2600), not emit
    // a silently-wrong `.so`.
    let out = build_out(
        "mind_bug7_u64_capture",
        "pub fn f() -> i64 {\n\
         \x20   let k: u64 = (0 - 1) as u64\n\
         \x20   let g = |k; x: i64| -> i64 { k >> 63 }\n\
         \x20   return g(0)\n\
         }\n",
    );
    assert!(
        out.contains("E2600"),
        "u64 closure capture NOT rejected (silent >>63 miscompile): {out}"
    );
}

#[test]
fn closure_bool_capture_accepted() {
    // `bool` zero-extends into the i64 slot exactly — must still compile.
    let out = build_out(
        "mind_bug7_bool_capture",
        "pub fn f() -> i64 {\n\
         \x20   let b: bool = true\n\
         \x20   let g = |b; x: i64| -> i64 { if b { 1 } else { 0 } }\n\
         \x20   return g(0)\n\
         }\n",
    );
    assert!(
        !out.contains("E2600"),
        "bool closure capture falsely rejected (over-match): {out}"
    );
}

#[test]
fn closure_i64_capture_accepted() {
    // The ordinary i64 capture is exact and must be untouched.
    let out = build_out(
        "mind_bug7_i64_capture",
        "pub fn f() -> i64 {\n\
         \x20   let k: i64 = 10\n\
         \x20   let g = |k; x: i64| -> i64 { k + x }\n\
         \x20   return g(1)\n\
         }\n",
    );
    assert!(
        !out.contains("E2600"),
        "i64 closure capture falsely rejected: {out}"
    );
}
