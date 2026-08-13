// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! corr-audit BUG (F4) — closure capture rewrite ignored inner shadowing.
//!
//! `rewrite_captures` in `src/eval/closures.rs` rewrote EVERY identifier whose
//! spelling was in the closure's capture set to `env.<capture>`, with no
//! lexical-scope stack. A `let`/`let (…)`/`for`/`match`-pattern binding INSIDE
//! the closure body that shadowed a capture name was therefore wrongly rewritten
//! to the captured outer value — a silent miscompile:
//!
//!   let k: i64 = 10
//!   let g = |k; x: i64| -> i64 { match x { k => k } }
//!   g(7)                       // returned 10 (captured k), MUST be 7 (bound k)
//!
//!   let g = |k; x: i64| -> i64 { let k: i64 = 100; x + k }
//!   g(1)                       // returned 11 (10+1), MUST be 101 (100+1)
//!
//! Fix: thread a lexical shadow set — once an inner binder (re)binds a capture
//! name, references in that binding's scope are the local, not the capture.
//!
//! Adversarial controls prove the fix is NOT over-applied: an ordinary
//! capture-by-value (`k + x`), a capture read in a WILDCARD match arm (binds
//! nothing), and a FREE capture deep inside nested if/match (no shadowing
//! binder) must all STILL rewrite to `env.<capture>`.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test bug_f4_closure_shadow_run`
//!
//! Needs `tests/common` (`mindc_bin` — CARGO_BIN_EXE_mindc, staleness-free).

#![cfg(all(unix, feature = "std-surface", feature = "mlir-build"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// RED repro #1: a match pattern `k` binds the scrutinee (7); body `k` is the
// bound value, NOT the captured outer k (10). Returned 10 before the fix.
pub fn kat_match_shadow() -> i64 {
    let k: i64 = 10;
    let g = |k; x: i64| -> i64 { match x { k => k } };
    g(7)
}

// RED repro #2: an inner `let k` shadows the capture for the REST of the body.
// x + k = 1 + 100 = 101. Returned 11 before the fix.
pub fn kat_let_shadow() -> i64 {
    let k: i64 = 10;
    let g = |k; x: i64| -> i64 { let k: i64 = 100; x + k };
    g(1)
}

// CONTROL: ordinary i64 capture-by-value — the free capture reference k must
// still rewrite to env.k (=10). 10 + 1 = 11.
pub fn kat_plain_capture() -> i64 {
    let k: i64 = 10;
    let g = |k; x: i64| -> i64 { k + x };
    g(1)
}

// CONTROL: a WILDCARD match arm binds nothing, so `k` in the arm body is a free
// capture and MUST rewrite to env.k (=10).
pub fn kat_wildcard_capture() -> i64 {
    let k: i64 = 10;
    let g = |k; x: i64| -> i64 { match x { _ => k } };
    g(7)
}

// CONTROL: a genuine free capture deep inside nested if/match still rewrites.
// x=8 > 0 -> match arm `_ => k + x` = 42 + 8 = 50.
pub fn kat_deep_free_capture() -> i64 {
    let k: i64 = 42;
    let g = |k; x: i64| -> i64 { if x > 0 { match x { _ => k + x } } else { k } };
    g(8)
}

// ADVERSARIAL: match guard AND body see the pattern binding k (=scrutinee 7);
// the `_` arm would see the free capture (100). guard 7 > 5 true -> returns 7.
pub fn kat_guard_shadow() -> i64 {
    let k: i64 = 100;
    let g = |k; x: i64| -> i64 { match x { k if k > 5 => k, _ => k } };
    g(7)
}

// ADVERSARIAL: a for-loop var shadows the capture k in the body; the loop END
// bound `..k` uses the FREE capture (3). sum(0..3)=0+1+2=3, + x(100) = 103.
pub fn kat_for_var_shadow() -> i64 {
    let k: i64 = 3;
    let g = |k; x: i64| -> i64 { let mut s: i64 = 0; for k in 0..k { s = s + k }; s + x };
    g(100)
}

// ADVERSARIAL: an inner `let k` shadows the capture; a LATER sibling stmt must
// see the local (1), not the capture (10). let y = k + x = 1 + 5 = 6.
pub fn kat_relet_shadow() -> i64 {
    let k: i64 = 10;
    let g = |k; x: i64| -> i64 { let k: i64 = 1; let y: i64 = k + x; y };
    g(5)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn bug_f4_closure_shadow_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug-f4-closure-shadow-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_bug_f4_closure_shadow_run.mind");
    let so = dir.join("mind_bug_f4_closure_shadow_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("bug-f4-closure-shadow-run: needs mlir-build; skipping");
            return;
        }
        panic!("bug-f4-closure-shadow-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         names = ('kat_match_shadow','kat_let_shadow','kat_plain_capture',\n\
         \x20        'kat_wildcard_capture','kat_deep_free_capture','kat_guard_shadow',\n\
         \x20        'kat_for_var_shadow','kat_relet_shadow')\n\
         for _n in names: getattr(lib, _n).restype = ctypes.c_int64\n\
         r = lib.kat_match_shadow();      assert r == 7,   'kat_match_shadow=' + str(r)\n\
         r = lib.kat_let_shadow();        assert r == 101, 'kat_let_shadow=' + str(r)\n\
         r = lib.kat_plain_capture();     assert r == 11,  'kat_plain_capture=' + str(r)\n\
         r = lib.kat_wildcard_capture();  assert r == 10,  'kat_wildcard_capture=' + str(r)\n\
         r = lib.kat_deep_free_capture(); assert r == 50,  'kat_deep_free_capture=' + str(r)\n\
         r = lib.kat_guard_shadow();      assert r == 7,   'kat_guard_shadow=' + str(r)\n\
         r = lib.kat_for_var_shadow();    assert r == 103, 'kat_for_var_shadow=' + str(r)\n\
         r = lib.kat_relet_shadow();      assert r == 6,   'kat_relet_shadow=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "bug-f4-closure-shadow-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
