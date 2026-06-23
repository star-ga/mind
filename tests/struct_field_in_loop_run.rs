// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct-FIELD read inside a LOOP BODY — RUNTIME gate (regression).
//!
//! Reading a struct field on a receiver whose struct type is only
//! resolvable through the per-fn `struct_env` (Step 1) or the
//! `receiver_types` side-table (Step 2) must produce the STORED value,
//! not the `ConstI64(0)` placeholder, when that read sits inside a
//! `while` (or desugared `for`) loop body.
//!
//! Two paths regressed identically and both produced a silent `0`:
//!   * Step 1 — a `let p = T { .. }` declared INSIDE the loop body: the
//!     loop-body `Let` handler tracked only collection sentinels, not
//!     StructLit-bound struct types, so a later `p.field` fell through to
//!     the placeholder.
//!   * Step 2 — `make().field` inside the loop body: `struct_resolver`'s
//!     `walk_expr` never descended into `While`/`ForEach` bodies, so the
//!     `receiver_types` side-table had no entry for the in-loop access and
//!     it too fell through to the placeholder.
//!
//! Both are SILENT miscompiles (runnable .so, EXIT=0, wrong value), the
//! exact `FieldAccess None -> ConstI64(0)` hazard class. The identical
//! reads OUTSIDE a loop already compile to the correct load.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test struct_field_in_loop_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
struct P { a: i64, b: i64 }

fn make() -> P {
    return P { a: 111, b: 222 }
}

// Step-1 path: struct var declared INSIDE the loop body, field read inside.
// Expect 77, not 0.
pub fn inner_var_in_loop() -> i64 {
    let mut acc: i64 = 0
    let mut i: i64 = 0
    while i < 1 {
        let p: P = P { a: 5, b: 77 }
        acc = acc + p.b
        i = i + 1
    }
    return acc
}

// Step-2 path: fn-returned struct field read INSIDE the loop body.
// Expect 222, not 0.
pub fn fn_ret_in_loop() -> i64 {
    let mut acc: i64 = 0
    let mut i: i64 = 0
    while i < 1 {
        let r: i64 = make().b
        acc = acc + r
        i = i + 1
    }
    return acc
}

// Control: the SAME fn-returned field read OUTSIDE any loop already worked.
// Expect 222 — guards against a fix that breaks the non-loop path.
pub fn outside() -> i64 {
    let r: i64 = make().b
    return r
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn struct_field_in_loop_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("struct-field-in-loop-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_struct_field_in_loop_run.mind");
    let so = dir.join("mind_struct_field_in_loop_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("struct-field-in-loop-run: needs mlir-build; skipping");
            return;
        }
        panic!("struct-field-in-loop-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for fn, exp in (('inner_var_in_loop', 77), ('fn_ret_in_loop', 222), ('outside', 222)):\n\
         \x20   f = getattr(lib, fn); f.restype = ctypes.c_int64; f.argtypes = []\n\
         \x20   got = f()\n\
         \x20   assert got == exp, fn + ': got=' + str(got) + ' expected=' + str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "struct-field-in-loop-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
