// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct field populated from a narrow (i32/u32) SSA value.
//!
//! A struct with a sub-i64 field whose value is a narrow SSA value (the natural
//! `P { x: a }` for `a: i32`) failed to compile in v0.9.0: the generic
//! `Instr::Call` arm rejected the non-`ScalarI64` argument to `__mind_store_i32`
//! BEFORE the narrow-store handler ran, and the handler itself blind-`trunc`'d
//! the value assuming i64. The fix exempts the narrow mem-intrinsics from the
//! blanket rejection and coerces the value from its real physical width. This
//! also closes the ABI-gate inconsistency the audit flagged (gate said "clean"
//! then lowering hard-errored): a gate-clean struct-lit now genuinely lowers.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test struct_narrow_field`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
struct P {
    x: i32,
    y: i32,
}
pub fn mk(a: i32, b: i32) -> i64 {
    let p = P { x: a, y: b }
    return p.x + p.y
}
pub fn t_struct_narrow() -> i64 {
    return mk(100, 23)
}

struct U {
    a: u32,
    b: u32,
}
pub fn umk(a: u32, b: u32) -> i64 {
    let u = U { a: a, b: b }
    return u.a + u.b
}
pub fn t_struct_u32() -> i64 {
    return umk(4000000000, 1)
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn struct_field_from_narrow_ssa_value_lowers() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("struct-narrow-field: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_struct_narrow_field.mind");
    let so = dir.join("mind_struct_narrow_field.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("struct-narrow-field: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("struct-narrow-field: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.t_struct_narrow.restype = ctypes.c_int64\n\
         r = lib.t_struct_narrow(); assert r == 123, 't_struct_narrow=' + str(r)\n\
         lib.t_struct_u32.restype = ctypes.c_int64\n\
         r = lib.t_struct_u32(); assert r == 4000000001, 't_struct_u32=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "struct-narrow-field value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
