// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! Pattern guards W1.5a (`<pattern> if <bool-expr> => <body>`) — RUNTIME
//! correctness gate, plus the drift #131 co-fix (a guarded arm is NOT a
//! catch-all).
//!
//! A guarded arm matches only when the pattern matches AND the guard is truthy;
//! a false guard falls through to the next arm. The desugar is a pure front-end
//! rewrite into the existing if-else chain — no new IR opcode. This gate proves
//! the emitted `.so` returns the right value for both guard outcomes, and that a
//! guarded catch-all binding does NOT swallow inputs its guard rejects (#131).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test pattern_guard_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// `classify` — guarded bare-ident catch (`n if n > 0`) + a real wildcard: the
//   guard decides between 1 and 0; the guarded arm must NOT act as a catch-all.
// `select` — guarded int-literal arm (`5 if x > 100`, guard always false for
//   x==5) followed by an unguarded `5` arm: proves a failed guard falls through
//   to the next arm with the SAME pattern rather than short-circuiting.
const SRC: &str = r#"
fn classify(x: i64) -> i64 {
    return match x { n if n > 0 => 1, _ => 0 };
}
fn select(x: i64) -> i64 {
    return match x { 5 if x > 100 => 111, 5 => 222, _ => 999 };
}
fn main() -> i64 { return 0; }
"#;

#[test]
fn pattern_guard_runs_correct() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("pattern-guard-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_pattern_guard_run.mind");
    let so = dir.join("mind_pattern_guard_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("pattern-guard-run: needs mlir-build; skipping");
            return;
        }
        panic!("pattern-guard-run: mindc --emit-shared failed:\n{stderr}");
    }

    // classify: guard `n > 0` picks 1 for positives, else the wildcard's 0 (the
    //   guarded ident arm is refutable, NOT a catch-all — #131).
    // select: x==5 fails the `5 if x > 100` guard, so the NEXT `5` arm wins
    //   (222); a non-5 value hits the wildcard (999).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         c = lib.classify; c.restype = ctypes.c_int64; c.argtypes = [ctypes.c_int64]\n\
         s = lib.select;   s.restype = ctypes.c_int64; s.argtypes = [ctypes.c_int64]\n\
         for arg, exp in ((7,1),(1,1),(0,0),(-3,0)):\n\
         \x20   got = c(arg)\n\
         \x20   assert got == exp, 'classify('+str(arg)+'): got='+str(got)+' expected='+str(exp)\n\
         for arg, exp in ((5,222),(9,999)):\n\
         \x20   got = s(arg)\n\
         \x20   assert got == exp, 'select('+str(arg)+'): got='+str(got)+' expected='+str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "pattern-guard-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
