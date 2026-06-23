// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `array<T>` static-constructor + push/get/length RUNTIME gate (task #24).
//!
//! Complements `array_surface_run.rs` (which builds from a literal `[..]`) by
//! exercising the canonical *empty-constructed* pattern: `array<i64>.new()` →
//! three `.push(x)` → assert `.length == 3` and `.get(1)` == the SECOND pushed
//! value (222, not a stale literal element). Every form lowers onto the existing
//! `std.vec` heap runtime — `vec_new` / `vec_push` / `vec_get` / `vec_len` — so
//! this proves the constructor + grow + index-read chain is sound, not merely
//! that it parses. Gated on `std-surface` so the keystone stays byte-identical.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test array_ctor_push_get_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// Empty constructor + 3 pushes; .length is the logical count (3).
pub fn build_len() -> i64 {
    let mut a: array<i64> = array<i64>.new()
    a.push(111)
    a.push(222)
    a.push(333)
    return a.length
}

// .get(1) is the SECOND pushed value (222), proving the heap store/load
// roundtrips through vec_push/vec_get rather than returning a const-0.
pub fn build_get1() -> i64 {
    let mut a: array<i64> = array<i64>.new()
    a.push(111)
    a.push(222)
    a.push(333)
    return a.get(1)
}

// Empty literal `[]` constructor + push + index read: 7 + 8 = 15.
pub fn empty_lit_idx() -> i64 {
    let mut a: array<i64> = []
    a.push(7)
    a.push(8)
    return a[0] + a[1]
}
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
fn array_ctor_push_get_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("array-ctor-push-get-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_array_ctor_push_get_run.mind");
    let so = dir.join("mind_array_ctor_push_get_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("array-ctor-push-get-run: needs mlir-build; skipping");
            return;
        }
        panic!("array-ctor-push-get-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('build_len','build_get1','empty_lit_idx'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.build_len(); assert r == 3, 'build_len=' + str(r)\n\
         r = lib.build_get1(); assert r == 222, 'build_get1=' + str(r)\n\
         r = lib.empty_lit_idx(); assert r == 15, 'empty_lit_idx=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "array-ctor-push-get-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
