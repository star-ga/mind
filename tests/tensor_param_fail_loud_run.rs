// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Track #15 — a `tensor`-typed function parameter/return is NOT in the runnable
//! i64-scalar ABI subset. A fleet audit feared `mindc x.mind --emit-ir` emitting
//! `const.i64 0` for `pub fn r(x: tensor<f32[2]>) -> tensor<f32[2]> { tensor.relu(x) }`
//! was a SILENT MISCOMPILE. Investigation showed `--emit-ir` prints only the
//! module-TOP IR — a function-only module emits the FnDef declaration's unit
//! placeholder (`const.i64 0`), IDENTICAL for a scalar `fn r(x: i64) -> i64`, so
//! that output is benign inspection, not a tensor-specific miscompile.
//!
//! The RUNNABLE paths already fail LOUD (the #306 fail-closed philosophy).
//! A tensor param/return is refused by the ABI gate (`lower::non_i64_param` /
//! `lower::non_i64_return`, file:line span); a tensor used INTERNALLY in an
//! i64-signature fn is refused at MLIR lowering (`error[mlir]: missing type
//! information ... while lowering relu`).
//! In both cases `--emit-shared` exits non-zero and writes NO `.so` — never a
//! wrong artifact. This test PINS that contract so a future ABI change cannot
//! silently regress it into a const-0 (or any other) miscompiled artifact.
//!
//! deferred: real tensor-param ABI (carry `TypeAnn::Tensor{dtype,dims}` through
//! FnDef params + `func.call` results into `tensor<...>` / memref descriptors,
//! lowered via `one-shot-bufferize{bufferize-function-boundaries=true}` so the
//! body emits a real relu and the artifact is ctypes-callable). Until then the
//! fail-loud refusal IS the correct contract — upgrade path: implement the
//! tensor function-boundary ABI, then flip these asserts to a compiled+run smoke.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test tensor_param_fail_loud_run`

#![cfg(all(unix, feature = "std-surface", feature = "cross-module-imports"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Compile `src` single-file via `--emit-shared` and return
/// (compile_succeeded, so_was_written, combined stdout+stderr).
fn emit_shared(src: &str, tag: &str) -> (bool, bool, String) {
    let mindc = mindc_bin();
    let dir = std::env::temp_dir();
    let path = dir.join(format!("mind_track15_{tag}.mind"));
    let so = dir.join(format!("mind_track15_{tag}.so"));
    // Start from a clean slate so a stale `.so` can't mask a missing-write.
    let _ = std::fs::remove_file(&so);
    std::fs::write(&path, src).expect("write src");
    let out = Command::new(&mindc)
        .args([
            path.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), so.exists(), combined)
}

/// A tensor-typed PARAMETER (and return) must FAIL LOUD on the runnable path:
/// non-zero exit, a diagnostic naming the tensor construct, and NO `.so` written.
#[test]
fn tensor_param_emit_shared_fails_loud() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("track15: mindc not found; skipping");
        return;
    }
    let src = "pub fn r(x: tensor<f32[2]>) -> tensor<f32[2]> {\n\
               \x20   return tensor.relu(x)\n\
               }\n";
    let (ok, so_written, output) = emit_shared(src, "param");
    assert!(
        !ok,
        "track15: a tensor-typed parameter must FAIL to lower to a runnable artifact, but it succeeded\n{output}"
    );
    assert!(
        !so_written,
        "track15: no `.so` may be written when lowering is refused (would be a silent miscompile)\n{output}"
    );
    assert!(
        output.contains("lower::non_i64_param") && output.contains("tensor-typed parameter/return"),
        "track15: expected the non_i64_param tensor diagnostic, got:\n{output}"
    );
    // And the return type is flagged too.
    assert!(
        output.contains("lower::non_i64_return"),
        "track15: expected the non_i64_return tensor diagnostic, got:\n{output}"
    );
}

/// A tensor used INTERNALLY inside an i64-signature fn (so the signature ABI gate
/// does not fire) must STILL fail loud — at MLIR lowering — and write no `.so`.
/// This closes the "slip past the signature gate" escape path.
#[test]
fn internal_tensor_in_i64_fn_fails_loud() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("track15: mindc not found; skipping");
        return;
    }
    let src = "pub fn r(n: i64) -> i64 {\n\
               \x20   let y: tensor<f32[2]> = tensor.zeros()\n\
               \x20   let z: tensor<f32[2]> = tensor.relu(y)\n\
               \x20   return n\n\
               }\n";
    let (ok, so_written, output) = emit_shared(src, "internal");
    assert!(
        !ok,
        "track15: an internal tensor in an i64-signature fn must FAIL to lower, but it succeeded\n{output}"
    );
    assert!(
        !so_written,
        "track15: no `.so` may be written when the tensor body cannot lower\n{output}"
    );
    assert!(
        output.contains("missing type information") && output.contains("relu"),
        "track15: expected the MLIR missing-type-info relu diagnostic, got:\n{output}"
    );
}

/// No false positive: a plain i64-scalar function still lowers to a runnable
/// artifact (the gate is narrow — it must not refuse the i64 subset).
#[test]
fn scalar_i64_fn_still_compiles() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("track15: mindc not found; skipping");
        return;
    }
    let src = "pub fn r(x: i64) -> i64 { return x + 1 }\n";
    let (ok, so_written, output) = emit_shared(src, "scalar_ok");
    assert!(
        ok && so_written,
        "track15: a plain i64 function must still lower to a runnable `.so`:\n{output}"
    );
    assert!(
        !output.contains("lower::non_i64_param") && !output.contains("lower::non_i64_return"),
        "track15: an i64 signature must NOT trigger the tensor ABI gate:\n{output}"
    );
}
