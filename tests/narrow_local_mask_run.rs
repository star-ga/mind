// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A `let` with an explicit NARROW integer type (`u8`/`u16`/`u32`/`i8`/`i16`)
//! must mask / sign-extend its initializer to the declared width — it used to
//! be silently widened to i64 with no masking, so `let c: u8 = 200 * 2` kept
//! the full value 400 instead of 144. Unsigned narrows AND-mask; signed
//! narrows sign-extend; i64/u64 are unchanged.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test narrow_local_mask_run`

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

fn run_returning(body: &str, tag: &str) -> i64 {
    let mindc = mindc_bin();
    let dir = std::env::temp_dir();
    let s = dir.join(format!("mind_nlocal_{tag}.mind"));
    let so = dir.join(format!("mind_nlocal_{tag}.so"));
    let src = format!("pub fn run() -> i64 {{\n{body}\n}}\n");
    std::fs::write(&s, src).expect("write");
    let out = Command::new(&mindc)
        .args([s.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    assert!(
        out.status.success(),
        "narrow-local compile failed ({tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\nlib.run.restype=ctypes.c_int64\nprint(lib.run())\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3").args(["-c", &py]).output().expect("py");
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(i64::MIN)
}

#[test]
fn narrow_local_masks_and_sign_extends() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-local: mindc not found; skipping");
        return;
    }
    // probe for mlir-build
    {
        let dir = std::env::temp_dir();
        let s = dir.join("mind_nlocal_probe.mind");
        std::fs::write(&s, "pub fn run() -> i64 { return 0 }\n").unwrap();
        let o = Command::new(&mindc)
            .args([s.to_str().unwrap(), "--emit-shared", dir.join("p.so").to_str().unwrap()])
            .output()
            .unwrap();
        let e = String::from_utf8_lossy(&o.stderr);
        if e.contains("mlir-build") && e.contains("requires") {
            println!("narrow-local: needs mlir-build; skipping");
            return;
        }
    }
    // unsigned narrows AND-mask
    assert_eq!(run_returning("    let c: u8 = 200 * 2\n    return c", "u8"), 144); // 400 & 0xFF
    assert_eq!(run_returning("    let c: u16 = 70000\n    return c", "u16"), 4464); // & 0xFFFF
    assert_eq!(run_returning("    let c: u8 = 5\n    return c", "u8ok"), 5); // in-range preserved
    // signed narrows sign-extend
    assert_eq!(run_returning("    let d: i8 = 200\n    return d", "i8"), -56);
    assert_eq!(run_returning("    let d: i8 = 5\n    return d", "i8ok"), 5);
}
