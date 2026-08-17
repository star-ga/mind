// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Phase 17.4 hardening — `ArrayLoad` bounds + element-type RUNTIME gate.
//!
//! Regression coverage for the audit HIGH findings against the const-array
//! element-load lowering (`src/mlir/lowering.rs`, `Instr::ArrayLoad`):
//!
//! * **A1 (bounds, wedge):** an out-of-bounds runtime index used to lower to an
//!   out-of-bounds `tensor.extract` — UB after bufferization whose result differs
//!   by substrate (a byte-identity break on the executable path). The load now
//!   clamps the index to `[0, len-1]`, giving OOB a PINNED, substrate-independent
//!   result (element 0 for a negative index, element len-1 for `>= len`). This
//!   test dlopens a real `.so` and asserts that determinism.
//! * **A2/A3 (element type):** the recovery used `elem_dtype.as_str()` (yielding
//!   the invalid MLIR `tensor<Nxq16>`) and fell every non-f32/f64 element through
//!   to `ScalarI64` (an i64-arith width miscompile on an i32/f16 value). An
//!   unsupported element type (q16/f16/bf16) is now a LOUD compile error instead
//!   of a silent miscompile / unparseable IR. This test asserts the compile FAILS
//!   with that message rather than emitting a wrong artifact.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test array_load_bounds_and_dtype`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// A1: a runtime index into a fixed const array. `at(k) = PSI[k]`.
const OOB_SRC: &str = r#"
const PSI: [i64; 6] = [1, -3, 2, -6, 4, 18];
pub fn at(k: i64) -> i64 { PSI[k] }
"#;

// A2/A3: an f16 dense-tensor element load — an element type OUTSIDE the
// proven-correct executable set {i64,i32,f32,f64}. Must fail loud, never emit.
const UNSUPPORTED_SRC: &str = r#"
pub fn pick(i: i64) -> f16 {
    let t: tensor<f16[3]> = [1.0, 2.0, 3.0];
    t[i]
}
"#;

/// A1 — an out-of-bounds index is deterministic (clamp to the array edge), not
/// substrate-dependent UB.
#[test]
fn oob_index_is_deterministic_clamp() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("array-load-bounds: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_array_load_oob.mind");
    let so = dir.join("mind_array_load_oob.so");
    std::fs::write(&src, OOB_SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("array-load-bounds: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("array-load-bounds: mindc --emit-shared failed:\n{stderr}");
    }

    // Negative → element 0 (=1); >= len → element len-1 (=18); in-bounds pass through.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.at.restype = ctypes.c_int64; lib.at.argtypes = [ctypes.c_int64]\n\
         assert lib.at(-1) == 1, 'at(-1)=' + repr(lib.at(-1))\n\
         assert lib.at(-100) == 1, 'at(-100)=' + repr(lib.at(-100))\n\
         assert lib.at(100) == 18, 'at(100)=' + repr(lib.at(100))\n\
         assert lib.at(6) == 18, 'at(6)=' + repr(lib.at(6))\n\
         assert lib.at(0) == 1, 'at(0)=' + repr(lib.at(0))\n\
         assert lib.at(2) == 2, 'at(2)=' + repr(lib.at(2))\n\
         assert lib.at(5) == 18, 'at(5)=' + repr(lib.at(5))\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "array-load-bounds OOB determinism check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// A2/A3 — an element type outside {i64,i32,f32,f64} fails loud at compile time
/// rather than emitting invalid IR or a width-miscompiled artifact.
#[test]
fn unsupported_element_type_fails_loud() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("array-load-dtype: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_array_load_f16.mind");
    let so = dir.join("mind_array_load_f16.so");
    std::fs::write(&src, UNSUPPORTED_SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("mlir-build") && stderr.contains("requires") {
        println!("array-load-dtype: mindc --emit-shared needs mlir-build; skipping");
        return;
    }
    assert!(
        !out.status.success(),
        "array-load-dtype: f16 element load unexpectedly SUCCEEDED (silent \
         miscompile risk):\nstdout: {}\nstderr: {stderr}",
        String::from_utf8_lossy(&out.stdout),
    );
    assert!(
        stderr.contains("not supported on the executable path"),
        "array-load-dtype: failed for the wrong reason (expected the \
         unsupported-element-type message):\n{stderr}",
    );
}
