// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Loop-lowering alias-clobber RUNTIME regression gate (fix: c0fd22d).
//!
//! `let mut j = start` lowered as pure env aliasing — `j` and `start` shared one
//! ValueId — and the MLIR While emitter's textual `substitute_ids` rewrote every
//! occurrence of that id to the loop-carried block-arg, silently clobbering any
//! read of the alias SOURCE into the loop counter. Three native-verified shapes:
//!   [A] alias source read in the COND  → always-true condition (hang / OOB)
//!   [B] alias source read in the BODY  → wrong value accumulated
//!   [C] two carried vars share one init → the wrong carried slot is written
//! The fix mints a fresh copy id in the dominating parent block when a shared-id
//! source is actually read. This compiles the three shapes to a `.so`, runs them,
//! and asserts the correct results — a regression re-introduces a silent
//! miscompile these assertions catch (shape [A] is capped so a regression FAILS
//! on the cap instead of hanging CI).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test alias_miscompile_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// [A] alias source read in the loop CONDITION: `j` aliases `start`, so
// `start + len` must read the real `start`, not the loop counter `j`.
// A clobber makes the condition `j < j + len` (always true); the `n >= 100`
// cap turns that regression into a wrong RESULT rather than an infinite loop.
pub fn count_a(start: i64, len: i64) -> i64 {
    let mut j: i64 = start
    let mut n: i64 = 0
    while j < start + len {
        if n >= 100 {
            break
        }
        n = n + 1
        j = j + 1
    }
    return n
}

// [B] alias source read in the loop BODY: `j` aliases `n`, so `acc + n` must
// add the real `n`, not the loop counter `j`.
pub fn g(n: i64) -> i64 {
    let mut j: i64 = n
    let mut acc: i64 = 0
    while j < 10 {
        acc = acc + n
        j = j + 1
    }
    return acc
}

// [C] two carried vars share one init id (both start as `n`): `y * 2` must
// write y's slot, not x's.
pub fn f(n: i64) -> i64 {
    let mut x: i64 = n
    let mut y: i64 = n
    let mut i: i64 = 0
    while i < 3 {
        x = x + 10
        y = y * 2
        i = i + 1
    }
    return x * 1000 + y
}
"#;

#[test]
fn alias_clobber_shapes_run_correctly() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("alias-miscompile-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_alias_miscompile_run.mind");
    let so = dir.join("mind_alias_miscompile_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("alias-miscompile-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("alias-miscompile-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for name in ('count_a','g','f'):\n\
         \x20   fn = getattr(lib, name); fn.restype = ctypes.c_int64\n\
         \x20   fn.argtypes = [ctypes.c_int64] * (2 if name=='count_a' else 1)\n\
         r = lib.count_a(0, 4); assert r == 4, '[A] count_a(0,4)=' + str(r) + ' (clobber -> 100)'\n\
         r = lib.g(3);          assert r == 21, '[B] g(3)=' + str(r) + ' (clobber -> 42)'\n\
         r = lib.f(1);          assert r == 31008, '[C] f(1)=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "alias-miscompile-run value check failed (silent miscompile regression?):\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
