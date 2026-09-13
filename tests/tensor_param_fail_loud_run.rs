// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Tensor-boundary refusal controls for executable shared artifacts.
//!
//! Static-shape tensor parameters have an admitted memref descriptor path.
//! These tests cover that positive path and the tensor returns, nested tensor
//! composites, and unsupported internal operation that must still be refused.
//! An unsupported boundary must exit non-zero and write no shared artifact.
//! Module-top `--emit-ir` unit placeholders are not executable tensor evidence.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test tensor_param_fail_loud_run`

// `--emit-shared` shells out to the MLIR/LLVM toolchain and is only available
// under `mlir-build` (without it mindc exits "--emit-shared requires building
// with the 'mlir-build' feature"), so — as the module gate above documents —
// these runnable-artifact assertions require it. A non-`mlir-build` test run
// therefore compiles this file to an empty set instead of failing on a build
// that structurally cannot produce a `.so`.
#![cfg(all(
    unix,
    feature = "std-surface",
    feature = "cross-module-imports",
    feature = "mlir-build"
))]

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
        crate::common::gate::skipped(
            "tensor_param_fail_loud_run",
            "track15: mindc not found; skipping",
        );
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
    // The tensor RETURN is what is refused now. The two assertions above — the
    // ones that make this a fail-loud control (non-zero exit, no `.so` written)
    // — are unchanged and still pass.
    //
    // This previously required `lower::non_i64_param`. Commit f9f68e5b ("allow
    // static-shape tensor param") deliberately split `param_non_i64` out of
    // `sig_non_i64` (src/eval/abi_gate.rs:107) so a STATIC-SHAPE tensor PARAMETER
    // lowers through a real memref C ABI. For this fixture the parameter
    // therefore no longer gates and the RETURN does, so the old assertion pinned
    // a rejection the compiler had intentionally stopped making.
    //
    // Both spellings are accepted rather than only the new one, so a future
    // change that re-gates the parameter cannot red this control. The shared
    // prose is asserted SEPARATELY below, so this cannot be satisfied by an
    // unrelated error that merely happens to be a lowering refusal.
    assert!(
        output.contains("lower::non_i64_return") || output.contains("lower::non_i64_param"),
        "track15: expected a tensor ABI-boundary refusal, got:\n{output}"
    );
    assert!(
        output.contains("tensor-typed parameter/return"),
        "track15: the refusal must name the tensor construct, got:\n{output}"
    );
    // And the return type is flagged too.
    assert!(
        output.contains("lower::non_i64_return"),
        "track15: expected the non_i64_return tensor diagnostic, got:\n{output}"
    );
}

/// A static-shape tensor PARAMETER must lower even when the body never TOUCHES it.
///
/// `param_non_i64` (src/eval/abi_gate.rs:107) deliberately admits a static-shape
/// tensor parameter, on the stated grounds that bufferization converts the
/// boundary to a memref. That conversion only happens under the `arith-linalg`
/// preset, and `preset_for_mlir` (src/eval/mlir_build.rs) chose it by scanning the
/// emitted MLIR for tensor OPERATIONS — `linalg.*`, `tensor.empty`,
/// `tensor.extract`, a dense tensor constant. A signature carrying `tensor<..>`
/// with no tensor op in the body matched none of them, fell to the scalar `core`
/// pipeline, and the tensor argument survived to translation:
///
///   error[build]: subprocess mlir-translate failed: cannot be converted to LLVM
///   IR: missing `LLVMTranslationDialectInterface` registration
///     func.func @s(%arg0: tensor<2xf32>) -> i64
///
/// A raw subprocess error for a construct the MIND-level ABI gate had just
/// admitted. The gap was SHAPE-DEPENDENT, which is why it survived: the same
/// function WITH a use emits `tensor.extract`, matches the old predicate, and
/// builds fine. Both shapes are asserted here so a future predicate that only
/// notices operations reds immediately.
#[test]
fn tensor_param_lowers_whether_or_not_the_body_uses_it() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "tensor_param_fail_loud_run",
            "track15: mindc not found; skipping",
        );
        return;
    }

    // UNUSED — the shape that regressed.
    let (ok, so_written, output) = emit_shared(
        "pub fn s(x: tensor<f32[2]>) -> i64 {\n    return 7\n}\n",
        "param_unused",
    );
    assert!(
        ok && so_written,
        "track15: a static-shape tensor param must lower even with no tensor op in \
         the body — the ABI gate admits it, so the pipeline must handle it:\n{output}"
    );

    // USED — must not regress while fixing the unused case.
    let (ok2, so2, out2) = emit_shared(
        "pub fn u(x: tensor<f32[2]>) -> f32 {\n    return x[0]\n}\n",
        "param_used",
    );
    assert!(
        ok2 && so2,
        "track15: a USED static-shape tensor param must still lower:\n{out2}"
    );
}

/// A tensor used INTERNALLY inside an i64-signature fn (so the signature ABI gate
/// does not fire) must STILL fail loud — at MLIR lowering — and write no `.so`.
/// This closes the "slip past the signature gate" escape path.
#[test]
fn internal_tensor_in_i64_fn_fails_loud() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "tensor_param_fail_loud_run",
            "track15: mindc not found; skipping",
        );
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
        crate::common::gate::skipped(
            "tensor_param_fail_loud_run",
            "track15: mindc not found; skipping",
        );
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

/// A tensor NESTED inside a composite type (`&tensor`, `(tensor, i64)`,
/// `Option<tensor>`, `[tensor; N]`, `&[tensor]`) must ALSO fail loud. Before the
/// recursive `sig_non_i64` fix (src/eval/abi_gate.rs) the ABI gate matched a
/// tensor ONLY as the outermost type node, so every one of these fell to
/// `_ => None`, `type_ann_to_abi_mlir` lowered the composite to `i64`, and
/// `--emit-shared` wrote an rc=0 `.so` whose C signature did not match the
/// declared type — a silent miscompile of an evidence-signable artifact
/// (MEASURED rc=0 + `.so` on the pre-fix binary for all five). The
/// `-> i64 { return t }` case additionally leaked the raw i64 slot as the
/// result (`r3(0x1234) == 0x1234`). Each must now be a loud refusal, no `.so`.
#[test]
fn nested_tensor_in_composite_fails_loud() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "tensor_param_fail_loud_run",
            "track15: mindc not found; skipping",
        );
        return;
    }
    // (tag, source) — every one currently mis-lowered to a raw i64 slot.
    let cases = [
        (
            "ref_param",
            "pub fn r1(t: &tensor<f32[4]>) -> i64 { return 0 }\n",
        ),
        (
            "tuple_param",
            "pub fn tp(t: (tensor<f32[4]>, i64)) -> i64 { return 0 }\n",
        ),
        (
            "option_param",
            "pub fn og(t: Option<tensor<f32[4]>>) -> i64 { return 0 }\n",
        ),
        (
            "slice_param",
            "pub fn sp(t: &[tensor<f32[4]>]) -> i64 { return 0 }\n",
        ),
        // The raw-slot RETURN leak: the param gate fires first, so this is
        // refused before the erased return can ship a wrong value.
        (
            "ret_leak",
            "pub fn r3(t: &tensor<f32[4]>) -> i64 {\n    return t\n}\n",
        ),
    ];
    for (tag, src) in cases {
        let (ok, so_written, output) = emit_shared(src, tag);
        assert!(
            !ok,
            "track15/{tag}: a tensor nested in a composite must FAIL to lower to a \
             runnable artifact, but it succeeded (silent miscompile)\n{output}"
        );
        assert!(
            !so_written,
            "track15/{tag}: no `.so` may be written when a nested-tensor boundary is \
             refused (would be a silent miscompile)\n{output}"
        );
        assert!(
            output.contains("tensor-typed parameter/return"),
            "track15/{tag}: the refusal must name the tensor construct, got:\n{output}"
        );
    }
}

/// No false positive from the recursive gate: a composite that carries NO tensor
/// must not trigger the tensor ABI gate. The recursion added to `sig_non_i64`
/// descends `Ref` / `Slice` / `Array` / `Generic` / `Tuple`, so this pins that
/// it fires ONLY on a tensor, never on an all-scalar/struct composite.
#[test]
fn non_tensor_composite_does_not_trigger_tensor_gate() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "tensor_param_fail_loud_run",
            "track15: mindc not found; skipping",
        );
        return;
    }
    let cases = [
        ("slice_i64", "pub fn a(x: &[i64]) -> i64 { return 0 }\n"),
        ("tuple_i64", "pub fn b(x: (i64, i64)) -> i64 { return 0 }\n"),
        (
            "option_i64",
            "pub fn c(x: Option<i64>) -> i64 { return 0 }\n",
        ),
    ];
    for (tag, src) in cases {
        let (_ok, _so, output) = emit_shared(src, tag);
        // May or may not lower for unrelated reasons; the ONLY claim here is
        // that the recursive tensor gate does not misfire on a tensor-free
        // composite.
        assert!(
            !output.contains("tensor-typed parameter/return"),
            "track15/{tag}: a tensor-free composite must NOT trigger the tensor ABI \
             gate, but it did:\n{output}"
        );
    }
}
