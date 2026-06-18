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

// Multi-field payload variants — every field is stored into its own record slot
// and bound positionally; a `_` skips a field without shifting later offsets.
enum Tri {
    T(i64, i64, i64),
    N,
}

enum Pick {
    Two(i64, i64),
    Zero,
}

// f64 payloads: the field is stored as raw bits in the i64 record slot
// (__mind_f64_to_bits) and loaded back (__mind_bits_to_f64), bit-exact.
enum FOpt {
    Some(f64),
    None,
}

enum FRes {
    Ok(f64),
    Err(f64),
}

// A variant mixing an i64 and an f64 field — each coerced independently.
enum Mix {
    Pair(i64, f64),
    Empty,
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

// Wildcard payload `Some(_)` — discriminate by tag, bind nothing. Previously
// bailed the desugar and SILENTLY returned the wrong arm (0 for both).
fn is_some(o: Opt) -> i64 {
    match o {
        Opt::Some(_) => 1,
        Opt::None => 0,
    }
}

// Three-field payload bound positionally.
fn combine(t: Tri) -> i64 {
    match t {
        Tri::T(a, b, c) => a + b * 10 + c * 100,
        Tri::N => 0 - 1,
    }
}

// Mixed binding/wildcard — `_` must not shift the other field's offset.
fn first_only(p: Pick) -> i64 {
    match p {
        Pick::Two(a, _) => a,
        Pick::Zero => 0,
    }
}

fn second_only(p: Pick) -> i64 {
    match p {
        Pick::Two(_, b) => b,
        Pick::Zero => 0,
    }
}

// f64 payload extraction + f64 arithmetic in the arm.
fn f_unwrap(o: FOpt) -> f64 {
    match o {
        FOpt::Some(v) => v,
        FOpt::None => 0.0,
    }
}

fn f_double(o: FOpt) -> f64 {
    match o {
        FOpt::Some(v) => v + v,
        FOpt::None => 0.0,
    }
}

fn f_fold(r: FRes) -> f64 {
    match r {
        FRes::Ok(v) => v,
        FRes::Err(e) => 0.0 - e,
    }
}

fn mix_f(m: Mix) -> f64 {
    match m {
        Mix::Pair(a, b) => b,
        Mix::Empty => 0.0,
    }
}

fn mix_i(m: Mix) -> i64 {
    match m {
        Mix::Pair(a, b) => a,
        Mix::Empty => 0,
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

// Wildcard payload match.
pub fn t_is_some() -> i64 { is_some(Opt::Some(99)) }
pub fn t_is_none() -> i64 { is_some(Opt::None) }

// Multi-field payloads.
pub fn t_tri() -> i64 { combine(Tri::T(7, 8, 9)) }
pub fn t_tri_n() -> i64 { combine(Tri::N) }
pub fn t_first() -> i64 { first_only(Pick::Two(11, 22)) }
pub fn t_second() -> i64 { second_only(Pick::Two(11, 22)) }
pub fn t_pick_zero() -> i64 { first_only(Pick::Zero) }

// f64 payloads.
pub fn t_fsome() -> f64 { f_unwrap(FOpt::Some(3.5)) }
pub fn t_fnone() -> f64 { f_unwrap(FOpt::None) }
pub fn t_fdouble() -> f64 { f_double(FOpt::Some(2.25)) }
pub fn t_fok() -> f64 { f_fold(FRes::Ok(7.5)) }
pub fn t_ferr() -> f64 { f_fold(FRes::Err(2.5)) }
pub fn t_mixf() -> f64 { mix_f(Mix::Pair(9, 3.25)) }
pub fn t_mixi() -> i64 { mix_i(Mix::Pair(9, 3.25)) }
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
         for _n in ('t_some','t_none','t_ok','t_err','t_neg','t_zero','t_pos','t_e_a','t_e_b','t_e_c','t_is_some','t_is_none','t_tri','t_tri_n','t_first','t_second','t_pick_zero','t_mixi'): getattr(lib,_n).restype = ctypes.c_int64\n\
         for _n in ('t_fsome','t_fnone','t_fdouble','t_fok','t_ferr','t_mixf'): getattr(lib,_n).restype = ctypes.c_double\n\
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
         r = lib.t_is_some(); assert r == 1, 't_is_some=' + str(r)\n\
         r = lib.t_is_none(); assert r == 0, 't_is_none=' + str(r)\n\
         r = lib.t_tri(); assert r == 987, 't_tri=' + str(r)\n\
         r = lib.t_tri_n(); assert r == -1, 't_tri_n=' + str(r)\n\
         r = lib.t_first(); assert r == 11, 't_first=' + str(r)\n\
         r = lib.t_second(); assert r == 22, 't_second=' + str(r)\n\
         r = lib.t_pick_zero(); assert r == 0, 't_pick_zero=' + str(r)\n\
         r = lib.t_fsome(); assert r == 3.5, 't_fsome=' + str(r)\n\
         r = lib.t_fnone(); assert r == 0.0, 't_fnone=' + str(r)\n\
         r = lib.t_fdouble(); assert r == 4.5, 't_fdouble=' + str(r)\n\
         r = lib.t_fok(); assert r == 7.5, 't_fok=' + str(r)\n\
         r = lib.t_ferr(); assert r == -2.5, 't_ferr=' + str(r)\n\
         r = lib.t_mixf(); assert r == 3.25, 't_mixf=' + str(r)\n\
         r = lib.t_mixi(); assert r == 9, 't_mixi=' + str(r)\n\
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
