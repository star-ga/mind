// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Match scrutinee EVALUATED-ONCE runtime gate (Fable audit #7).
//!
//! The `match`-desugar embeds the scrutinee node into EVERY arm's discriminant
//! test and each payload bind (~6 clone sites in `desugar_match_to_if`), so
//! before the fix a `match f() { 0 => …, 1 => …, _ => … }` re-lowered and
//! re-evaluated `f()` up to once PER ARM TESTED at runtime — duplicating any
//! side effects — whereas the interpreter oracle evaluates the scrutinee
//! EXACTLY once. The fix gates on `expr_contains_call(scrutinee)`: an effectful
//! scrutinee is pre-bound ONCE into a span-unique hidden `let __match_scrut_*`
//! and that ident is embedded everywhere; a pure scrutinee keeps the
//! byte-identical embed-verbatim path.
//!
//! Call-count is made OBSERVABLE deterministically via a shared-mutable heap
//! counter (`bytes[8].zero()` + `__mind_{load,store}_i64`): `tick` bumps the
//! counter and returns the tag, so the counter advances once per REAL
//! evaluation. Each function is compiled to a `.so`, dlopen-called, and its
//! return (result * scale + observed call-count) is asserted against the hand
//! oracle. Covers: a plain call scrutinee (gate ON), a payload-carrying enum
//! scrutinee (gate ON, more embed sites), a NESTED gated match in an arm body
//! (Fable caveat F — distinct span-unique temps must not collide), and a pure
//! ident scrutinee (gate OFF — the byte-neutral common path).
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test match_scrutinee_once`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
enum E { A(i64), B(i64) }

// Effectful scrutinee helper: bump the shared counter at `ctr`, return `tag`.
fn tick(ctr: i64, tag: i64) -> i64 {
    let cur = __mind_load_i64(ctr)
    let _ = __mind_store_i64(ctr, cur + 1)
    return tag
}

// Effectful enum-producing scrutinee: bump the counter, return E.A/E.B(payload).
fn tick_e(ctr: i64, tag: i64, payload: i64) -> E {
    let cur = __mind_load_i64(ctr)
    let _ = __mind_store_i64(ctr, cur + 1)
    if tag == 0 { return E.A(payload) }
    return E.B(payload)
}

// GATE ON: int-literal arms over a call scrutinee. tag 1 => 200, called ONCE.
// fix => 200*10 + 1 = 2001 ; per-arm bug (tests arm0 then arm1) => 2002.
pub fn scrut_once() -> i64 {
    let ctr = bytes[8].zero()
    let r = match tick(ctr, 1) {
        0 => 100,
        1 => 200,
        _ => 300,
    }
    return r * 10 + __mind_load_i64(ctr)
}

// GATE ON: payload-carrying enum scrutinee — the desugar embeds the scrutinee
// at MORE sites (tag load + payload field bind), so per-arm re-eval would
// advance the counter further. E.B(42) => 42+2 = 44, called ONCE.
// fix => 44*10 + 1 = 441.
pub fn scrut_once_payload() -> i64 {
    let ctr = bytes[8].zero()
    let r = match tick_e(ctr, 1, 42) {
        E.A(v) => v + 1,
        E.B(v) => v + 2,
        _ => 0,
    }
    return r * 10 + __mind_load_i64(ctr)
}

// NESTED gated match inside an arm body (Fable caveat F): each match binds its
// OWN span-unique __match_scrut_*, so the inner pre-bind cannot collide with the
// outer one. Both scrutinees must evaluate exactly once.
// outer tag 1 -> inner tag 2 -> 77 ; outer called ONCE, inner called ONCE.
// fix => 77*10000 + 1*100 + 1 = 770101.
pub fn nested_scrut() -> i64 {
    let outer = bytes[8].zero()
    let inner = bytes[8].zero()
    let r = match tick(outer, 1) {
        0 => 10,
        1 => {
            let s = match tick(inner, 2) {
                2 => 77,
                _ => 88,
            }
            s
        }
        _ => 99,
    }
    return r * 10000 + __mind_load_i64(outer) * 100 + __mind_load_i64(inner)
}

// GATE OFF: pure (ident) scrutinee — the byte-neutral common path. Must still
// select the correct arm. tag 1 => 200.
pub fn scrut_pure() -> i64 {
    let tag = 1
    let r = match tag {
        0 => 100,
        1 => 200,
        _ => 300,
    }
    return r
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn match_scrutinee_evaluated_once() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("match-scrutinee-once: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_match_scrutinee_once.mind");
    let so = dir.join("mind_match_scrutinee_once.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("match-scrutinee-once: needs mlir-build; skipping");
            return;
        }
        panic!("match-scrutinee-once: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('scrut_once','scrut_once_payload','nested_scrut','scrut_pure'):\n\
         \x20   getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.scrut_once(); assert r == 2001, 'scrut_once=' + str(r)\n\
         r = lib.scrut_once_payload(); assert r == 441, 'scrut_once_payload=' + str(r)\n\
         r = lib.nested_scrut(); assert r == 770101, 'nested_scrut=' + str(r)\n\
         r = lib.scrut_pure(); assert r == 200, 'scrut_pure=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "match-scrutinee-once check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
