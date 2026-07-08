// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Track #15 — 2-D STATIC-SHAPE tensor PARAMETER ABI run smoke.
//!
//! The static-shape tensor-parameter ABI (`param_non_i64` in
//! `src/eval/abi_gate.rs`) is rank-generic: a `tensor<f64[R,C]>` parameter
//! lowers through the same `one-shot-bufferize{bufferize-function-boundaries=
//! true}` boundary conversion and the same rank-generic pinned canonical fold
//! (`MlirEmitter::emit_tensor_reduce_pinned`, an odometer over all dims) that
//! the landed 1-D case reuses. This PINS that a 2-D memref parameter
//! (unpacked C descriptor `{alloc_ptr, aligned_ptr, offset, size0, size1,
//! stride0, stride1}`) actually reduces at RUNTIME — the runtime-fed param
//! defeats constant-folding, so the artifact must contain a real unfused
//! left-to-right scalar `arith.addf` chain, not a baked constant.
//!
//! `msum2([[1,2],[3,4]]) == 10.0` (exact, no float error — a fixed-order chain
//! of IEEE-754 round-to-nearest adds over exactly-representable integers).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test tensor_param_2d_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
// A 2-D static-shape tensor parameter, reduced to a scalar. The runtime memref
// argument defeats constant-folding, so the body is a real add-chain.
pub fn msum2(t: tensor<f64[2,2]>) -> f64 {
    return t.sum()
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn tensor_param_2d_sum_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("tensor-param-2d-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_tensor_param_2d_run.mind");
    let so = dir.join("mind_tensor_param_2d_run.so");
    // Clean slate so a stale `.so` can't mask a missing write.
    let _ = std::fs::remove_file(&so);
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("tensor-param-2d-run: needs mlir-build; skipping");
            return;
        }
        panic!("tensor-param-2d-run: mindc --emit-shared failed:\n{stderr}");
    }
    assert!(
        so.exists(),
        "tensor-param-2d-run: no `.so` written despite success"
    );

    // Call msum2 through the unpacked 2-D memref C ABI descriptor:
    //   (alloc_ptr, aligned_ptr, offset, size0, size1, stride0, stride1) -> f64
    // Data is row-major [[1,2],[3,4]] => contiguous [1,2,3,4], strides (2,1).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         f = lib.msum2\n\
         f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong,\n\
         \x20             ctypes.c_longlong, ctypes.c_longlong]\n\
         f.restype = ctypes.c_double\n\
         buf = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)\n\
         p = ctypes.cast(buf, ctypes.c_void_p)\n\
         r = f(p, p, 0, 2, 2, 2, 1)\n\
         assert r == 10.0, 'msum2=' + repr(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "tensor-param-2d-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
