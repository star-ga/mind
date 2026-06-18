// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Boxed-enum `match` RUNTIME gate — payload + fieldless variants together.
//!
//! A payload-carrying enum (`Opt::Some(i64)`) lowered to a 2-field heap record
//! `[tag @ +0, payload @ +8]`, but a FIELDLESS sibling (`Opt::None`) lowered to a
//! BARE ordinal. The match reads the tag with `__mind_load_i64(scrutinee + 0)`,
//! so a `None` scrutinee dereferenced the ordinal `1` AS AN ADDRESS → SEGFAULT.
//! `match o { Opt::Some(v) => v, Opt::None => d }` therefore crashed for every
//! Option/Result-shaped enum — the most common case. It shipped because the only
//! enum test (`enum_soundness`) checks the program COMPILES, never RUNS it.
//!
//! The fix: an enum with ≥1 payload variant is "boxed", and EVERY constructor
//! (including fieldless) lowers to the uniform `[tag, payload]` record (fieldless
//! → payload 0); a `match` on a boxed enum always reads the tag from the record.
//! This test compiles a multi-enum program to a `.so`, dlopen-calls it, and
//! asserts the values — it would have caught the crash.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test enum_match_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
enum Opt {
    Some(i64),
    None,
}

enum Res {
    Ok(i64),
    Err(i64),
}

// Fieldless variant (Zero) BETWEEN two payload variants — exercises tag 1 as a
// boxed fieldless record and a non-zero payload tag (Pos = 2).
enum Sign {
    Neg(i64),
    Zero,
    Pos(i64),
}

// Boxed enum matched on ONLY fieldless variants (no payload binding anywhere) —
// the match must load the tag from the record, not compare the record POINTER
// against an ordinal (which never matches).
enum E {
    A(i64),
    B,
    C,
}

fn unwrap_or(o: Opt, d: i64) -> i64 {
    match o {
        Opt::Some(v) => v,
        Opt::None => d,
    }
}

fn fold(r: Res) -> i64 {
    match r {
        Res::Ok(v) => v,
        Res::Err(e) => 0 - e,
    }
}

fn classify(s: Sign) -> i64 {
    match s {
        Sign::Neg(n) => 0 - n,
        Sign::Zero => 100,
        Sign::Pos(p) => p,
    }
}

fn name(e: E) -> i64 {
    match e {
        E::B => 22,
        E::C => 33,
        _ => 0,
    }
}

// Option: payload extraction + fieldless default (the original segfault).
pub fn t_some() -> i64 { unwrap_or(Opt::Some(42), 7) }
pub fn t_none() -> i64 { unwrap_or(Opt::None, 7) }

// Result: two payload variants.
pub fn t_ok() -> i64 { fold(Res::Ok(55)) }
pub fn t_err() -> i64 { fold(Res::Err(9)) }

// Mixed 3-variant with a fieldless middle variant.
pub fn t_neg() -> i64 { classify(Sign::Neg(4)) }
pub fn t_zero() -> i64 { classify(Sign::Zero) }
pub fn t_pos() -> i64 { classify(Sign::Pos(7)) }

// Boxed enum, fieldless-only match arms.
pub fn t_e_a() -> i64 { name(E::A(5)) }
pub fn t_e_b() -> i64 { name(E::B) }
pub fn t_e_c() -> i64 { name(E::C) }
"#;

fn mindc_bin() -> PathBuf {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d = m.join("target").join("debug").join("mindc");
    if d.exists() {
        d
    } else {
        m.join("target").join("release").join("mindc")
    }
}

#[test]
fn boxed_enum_match_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("enum-match-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_enum_match_run.mind");
    let so = dir.join("mind_enum_match_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("enum-match-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("enum-match-run: mindc --emit-shared failed:\n{stderr}");
    }

    // Flat top-level statements only (Rust's `\`-line-continuation strips each
    // continued line's leading whitespace, so no indented Python blocks). Each
    // function defaults to a 0-arg i64 return; one assert per case.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('t_some','t_none','t_ok','t_err','t_neg','t_zero','t_pos','t_e_a','t_e_b','t_e_c'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.t_some(); assert r == 42, 't_some=' + str(r)\n\
         r = lib.t_none(); assert r == 7, 't_none=' + str(r)\n\
         r = lib.t_ok(); assert r == 55, 't_ok=' + str(r)\n\
         r = lib.t_err(); assert r == -9, 't_err=' + str(r)\n\
         r = lib.t_neg(); assert r == -4, 't_neg=' + str(r)\n\
         r = lib.t_zero(); assert r == 100, 't_zero=' + str(r)\n\
         r = lib.t_pos(); assert r == 7, 't_pos=' + str(r)\n\
         r = lib.t_e_a(); assert r == 0, 't_e_a=' + str(r)\n\
         r = lib.t_e_b(); assert r == 22, 't_e_b=' + str(r)\n\
         r = lib.t_e_c(); assert r == 33, 't_e_c=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "enum-match-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
