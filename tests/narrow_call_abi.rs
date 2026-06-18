// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Narrow-int (i32/u32/bool) INTER-FUNCTION ABI gate.
//!
//! v0.9.0 lowered narrow leaf functions correctly but the generic `func.call`
//! arm hardcoded `(i64..) -> i64`, so ANY narrow param/return on a function that
//! is CALLED by another made the whole module fail `mlir-opt` (type mismatch) —
//! and a narrow early `return` inside an if/else/while body emitted
//! `return %x : i64` against an `-> i32` result, because `fn_ret_abi` was never
//! propagated into the branch sub-context. Both shipped CI-green: nothing ran
//! `mlir-opt` over a composed narrow program. This test compiles a multi-function
//! narrow program to a `.so`, dlopen-calls it, and asserts the values — covering
//! narrow returns/params across a call boundary, early narrow returns inside
//! if-then / if-else / while bodies, and bool + u32 (zero-extended) returns.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test narrow_call_abi`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// narrow RETURN consumed across a call boundary
pub fn produce() -> i32 {
    return 1000000
}
pub fn t_narrow_return() -> i64 {
    return produce()
}

// narrow PARAM passed across a call boundary (i64 literal -> i32 param)
pub fn dbl(x: i32) -> i32 {
    return x + x
}
pub fn t_narrow_param() -> i64 {
    return dbl(21)
}

// early narrow return inside if-then / if-else bodies
pub fn guard(a: i32) -> i32 {
    if a > 0 {
        return 5
    }
    return 9
}
pub fn t_early_then() -> i64 {
    return guard(7)
}
pub fn t_early_else() -> i64 {
    return guard(0)
}

// early narrow return inside a while body
pub fn wguard(n: i32) -> i32 {
    let mut i: i32 = 0
    while i < n {
        if i == 2 {
            return 42
        }
        i = i + 1
    }
    return 99
}
pub fn t_while_early() -> i64 {
    return wguard(5)
}
pub fn t_while_noearly() -> i64 {
    return wguard(1)
}

// bool return consumed across a call boundary
pub fn ispos(x: i32) -> bool {
    return x > 0
}
pub fn t_bool_call() -> i64 {
    if ispos(5) {
        return 100
    }
    return 200
}

// u32 return must ZERO-extend at the use site (not sign-extend)
pub fn uprod() -> u32 {
    return 4000000000
}
pub fn t_u32_zero_extend() -> i64 {
    return uprod()
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
fn narrow_int_inter_function_abi() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("narrow-call-abi: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_narrow_call_abi.mind");
    let so = dir.join("mind_narrow_call_abi.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        // A genuinely-missing MLIR backend is a skip; a type-mismatch / lowering
        // error is the regression this gate exists to catch — fail loud.
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("narrow-call-abi: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("narrow-call-abi: mindc --emit-shared failed:\n{stderr}");
    }

    // NOTE: flat top-level statements only — Rust's `\`-line-continuation strips
    // each continued line's leading whitespace, so a Python block needing
    // indentation (for/def) would lose it. `;` keeps each check on one line.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.t_narrow_return.restype = ctypes.c_int64\n\
         r = lib.t_narrow_return(); assert r == 1000000, 't_narrow_return=' + str(r)\n\
         lib.t_narrow_param.restype = ctypes.c_int64\n\
         r = lib.t_narrow_param(); assert r == 42, 't_narrow_param=' + str(r)\n\
         lib.t_early_then.restype = ctypes.c_int64\n\
         r = lib.t_early_then(); assert r == 5, 't_early_then=' + str(r)\n\
         lib.t_early_else.restype = ctypes.c_int64\n\
         r = lib.t_early_else(); assert r == 9, 't_early_else=' + str(r)\n\
         lib.t_while_early.restype = ctypes.c_int64\n\
         r = lib.t_while_early(); assert r == 42, 't_while_early=' + str(r)\n\
         lib.t_while_noearly.restype = ctypes.c_int64\n\
         r = lib.t_while_noearly(); assert r == 99, 't_while_noearly=' + str(r)\n\
         lib.t_bool_call.restype = ctypes.c_int64\n\
         r = lib.t_bool_call(); assert r == 100, 't_bool_call=' + str(r)\n\
         lib.t_u32_zero_extend.restype = ctypes.c_int64\n\
         r = lib.t_u32_zero_extend(); assert r == 4000000000, 't_u32_zero_extend=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "narrow-call-abi value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
