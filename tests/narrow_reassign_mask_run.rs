// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A REASSIGNMENT of a NARROW-typed local (`u8`/`u16`/`u32`/`i8`/`i16`) must
//! re-mask / sign-extend to the declared width — the `let` initializer was
//! already masked (see `narrow_local_mask_run`), but `c = c + 100` (and the
//! `c += 100` that desugars to it) carried no annotation and silently kept the
//! full-width i64 value (`200 + 100 == 300` instead of the wrapped `u8` `44`).
//! The drop spanned top-level, then/else-branch and loop bodies, so each is
//! exercised here. The non-narrow `i64` path is asserted unchanged.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test narrow_reassign_mask_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Compile `pub fn run(flag: i64) -> i64 { <body> }` to a `.so` and call it with
/// `arg`, returning the i64 result. `flag` is always declared (ignored by bodies
/// that don't branch on it) so the one signature serves every case.
fn run_with(body: &str, tag: &str, arg: i64) -> i64 {
    let mindc = mindc_bin();
    let dir = std::env::temp_dir();
    let s = dir.join(format!("mind_nreassign_{tag}.mind"));
    let so = dir.join(format!("mind_nreassign_{tag}.so"));
    let src = format!("pub fn run(flag: i64) -> i64 {{\n{body}\n}}\n");
    std::fs::write(&s, src).expect("write");
    let out = Command::new(&mindc)
        .args([s.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    assert!(
        out.status.success(),
        "narrow-reassign compile failed ({tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\nlib.run.restype=ctypes.c_int64\n\
         lib.run.argtypes=[ctypes.c_int64]\nprint(lib.run({arg}))\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("py");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(i64::MIN)
}

#[test]
fn narrow_reassign_re_masks_to_declared_width() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-reassign: mindc not found; skipping");
        return;
    }
    // probe for mlir-build
    {
        let dir = std::env::temp_dir();
        let s = dir.join("mind_nreassign_probe.mind");
        std::fs::write(&s, "pub fn run() -> i64 { return 0 }\n").unwrap();
        let o = Command::new(&mindc)
            .args([
                s.to_str().unwrap(),
                "--emit-shared",
                dir.join("p_nr.so").to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let e = String::from_utf8_lossy(&o.stderr);
        if e.contains("mlir-build") && e.contains("requires") {
            println!("narrow-reassign: needs mlir-build; skipping");
            return;
        }
    }

    // ── top-level plain reassignment: 200 + 100 == 300, wraps to u8 == 44.
    assert_eq!(
        run_with("    let c: u8 = 200\n    c = c + 100\n    return c", "top", 0),
        44
    );
    // ── compound-assign desugars to the same Assign; same wrap.
    assert_eq!(
        run_with("    let c: u8 = 200\n    c += 100\n    return c", "cmpd", 0),
        44
    );
    // ── u16 reassignment masks to 16 bits.
    assert_eq!(
        run_with("    let c: u16 = 65000\n    c = c + 1000\n    return c", "u16", 0),
        464 // 66000 & 0xFFFF == 66000 - 65536
    );
    // ── signed i8 reassignment sign-extends: 100 + 100 == 200 -> -56.
    assert_eq!(
        run_with("    let d: i8 = 100\n    d = d + 100\n    return d", "i8", 0),
        -56
    );
    // ── then-branch reassignment re-masks.
    assert_eq!(
        run_with(
            "    let c: u8 = 200\n    if flag > 0 {\n        c = c + 100\n    }\n    return c",
            "then",
            1
        ),
        44
    );
    // ── else-branch reassignment re-masks.
    assert_eq!(
        run_with(
            "    let c: u8 = 200\n    if flag > 0 {\n        c = c + 1\n    } else {\n        c = c + 100\n    }\n    return c",
            "else",
            0
        ),
        44
    );
    // ── loop-carried narrow reassignment re-masks each iteration; the final
    //    accumulated value is masked to the declared width: 5*100 == 500 -> 244.
    assert_eq!(
        run_with(
            "    let c: u8 = 0\n    let i: i64 = 0\n    while i < 5 {\n        c = c + 100\n        i = i + 1\n    }\n    return c",
            "loop",
            0
        ),
        244 // 500 & 0xFF
    );

    // ── the i64 (non-narrow) path is UNCHANGED: no mask, full value kept.
    assert_eq!(
        run_with("    let c: i64 = 200\n    c = c + 100\n    return c", "i64", 0),
        300
    );
}
