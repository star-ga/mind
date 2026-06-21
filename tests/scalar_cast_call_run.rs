// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Scalar cast-call `u32(x)` / `i64(x)` RUNTIME gate.
//!
//! A call whose callee is a reserved scalar type name with one argument is a
//! CAST written in call form (`u32(i + 1)`, mind-flow idiom) — desugared to
//! `x as <type>`, reusing the existing cast typecheck/codegen. Compiles a
//! program exercising the narrowing + widening casts, dlopen-calls it, and
//! asserts the values.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test scalar_cast_call_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// u32(301) then i64(...) widen back — value preserved (301 fits u32).
pub fn roundtrip() -> i64 {
    let a = u32(300 + 1)
    let b = i64(a)
    return b
}

// Cast-call in argument position + chained casts on in-range values, the
// idiom mind-flow uses (`u32(i + 1)` for 1-based ids). 1-based id of index 6.
pub fn chained() -> i64 {
    let i = 6
    let id = u32(i + 1)
    return i64(id) * 10
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
fn scalar_cast_call_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("scalar-cast-call-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_scalar_cast_call_run.mind");
    let so = dir.join("mind_scalar_cast_call_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("scalar-cast-call-run: needs mlir-build; skipping");
            return;
        }
        panic!("scalar-cast-call-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('roundtrip','chained'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.roundtrip(); assert r == 301, 'roundtrip=' + str(r)\n\
         r = lib.chained(); assert r == 70, 'chained=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "scalar-cast-call-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
