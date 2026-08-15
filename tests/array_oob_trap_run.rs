// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP` executable regression gate.
//!
//! A runtime out-of-bounds array index is a DETERMINISTIC BOUNDS TRAP — never a
//! clamp to `[0, len-1]`, never a silent wrong value. This gate compiles small
//! programs whose index is a genuine RUNTIME value (a function parameter, so the
//! compiler cannot fold it), runs the real artifact, and asserts:
//!   * OOB (negative, `== len`, and `> len`) exits with the pinned OOB code 77
//!     on BOTH the MLIR/clang backend AND the pure-MIND native-ELF backend
//!     (OOB_RUST_MLIR_NATIVE_PARITY);
//!   * an in-bounds index returns the correct element (the trap adds no
//!     behavioural change on the happy path);
//!   * OOB is NOT CLAMPED — a negative index does not return element 0 and a
//!     `>= len` index does not return element `len-1` (the exact regression a
//!     re-introduced `arith.maxsi/minsi` clamp would cause);
//!   * a COMPILE-TIME-PROVABLE OOB (a literal constant index) is REJECTED by
//!     `mindc check` (deterministic compile-time rejection, E2621), never lowered.
//!
//! The clamp lived in `src/mlir/lowering.rs` (`ArrayLoad`); it was replaced by a
//! call to the single trap authority `__mind_oob_check` (runtime-support), which
//! is emitted BEFORE the `tensor.extract` and whose result the extract consumes,
//! so the trap strictly dominates the memory access.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test array_oob_trap_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

const OOB_CODE: i32 = 77;

/// `at(k)` indexes a LOCAL `[i64;3]` by the runtime parameter `k`, so the index
/// is never a compile-time constant (never const-folded / never compile-rejected)
/// and always exercises the runtime bounds-trap path. `main` calls `at(<k>)`, so
/// the process exit is the in-range element for a valid `k`, or the trap for OOB.
fn runtime_src(k: i64) -> String {
    format!(
        "fn at(k: i64) -> i64 {{\n    let a: [i64; 3] = [10, 20, 30];\n    return a[k];\n}}\n\
         fn main() -> i64 {{ return at({k}); }}\n"
    )
}

/// Compile a cleanly-compiling program and run it; returns its real exit code.
/// `native` selects the pure-MIND native-ELF backend, else the MLIR/clang path.
/// Panics with the compiler diagnostics if the artifact is not produced (a
/// masked compile failure — build emits a JIT launcher stub — is caught here so
/// it can never be mistaken for a clean run).
fn build_and_run(mindc: &Path, src: &str, native: bool) -> i32 {
    let dir = std::env::temp_dir().join(format!(
        "mind_oob_{}_{}",
        if native { "nat" } else { "mlir" },
        // vary by content so parallel cases never collide on one path
        src.len()
    ));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    let out = dir.join("p.bin");
    let _ = std::fs::remove_file(&out);
    std::fs::write(&srcf, src).expect("write src");

    let mut cmd = Command::new(mindc);
    cmd.arg("build");
    if native {
        cmd.arg("--backend=native");
    } else {
        cmd.arg("--emit=binary");
    }
    // Run from the repo root so the native backend resolves `std/` modules.
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(format!("--out={}", out.display()))
        .arg(&srcf);
    let c = cmd.output().expect("run mindc build");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    // A build that could not natively compile emits a launcher stub, not a real
    // binary — treat that as a hard failure, never a clean run.
    assert!(
        !blob.contains("not natively compiled")
            && out.exists()
            && out.metadata().unwrap().len() > 0,
        "compile did not produce a real binary (native={native}):\n{blob}"
    );
    let mut perm = std::fs::metadata(&out).unwrap().permissions();
    perm.set_mode(0o755);
    std::fs::set_permissions(&out, perm).unwrap();

    let r = Command::new(&out).output().expect("run artifact");
    r.status.code().unwrap_or(-1)
}

/// True iff `mindc check` rejects the program with an out-of-bounds diagnostic.
fn check_rejects(mindc: &Path, src: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("mind_oob_chk_{}", src.len()));
    let _ = std::fs::create_dir_all(&dir);
    let srcf = dir.join("main.mind");
    std::fs::write(&srcf, src).expect("write src");
    let c = Command::new(mindc)
        .arg("check")
        .arg(&srcf)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run mindc check");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&c.stderr),
        String::from_utf8_lossy(&c.stdout)
    );
    (
        !c.status.success() && blob.to_lowercase().contains("out of bounds"),
        blob,
    )
}

#[test]
fn array_oob_is_a_deterministic_trap_not_a_clamp() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("array-oob-trap: mindc not found; skipping");
        return;
    }

    // Runtime OOB on BOTH backends -> pinned trap code, identical (parity).
    for k in [-1i64, 3, 5] {
        let src = runtime_src(k);
        let mlir = build_and_run(&mindc, &src, false);
        let native = build_and_run(&mindc, &src, true);
        assert_eq!(
            mlir, OOB_CODE,
            "MLIR OOB k={k} should trap {OOB_CODE}, got {mlir}"
        );
        assert_eq!(
            native, OOB_CODE,
            "native OOB k={k} should trap {OOB_CODE}, got {native}"
        );
        assert_eq!(mlir, native, "OOB_RUST_MLIR_NATIVE_PARITY broken at k={k}");
    }

    // In-bounds -> correct element, both backends (happy path unchanged).
    for (k, want) in [(0i64, 10i32), (1, 20), (2, 30)] {
        let src = runtime_src(k);
        assert_eq!(
            build_and_run(&mindc, &src, false),
            want,
            "MLIR in-bounds k={k}"
        );
        assert_eq!(
            build_and_run(&mindc, &src, true),
            want,
            "native in-bounds k={k}"
        );
    }

    // NOT-CLAMPED: a clamp would make a[-1] read a[0]==10 and a[5] read
    // a[len-1]==30. The trap makes both 77 instead — assert they are NOT the
    // clamped values on either backend.
    for native in [false, true] {
        let neg = build_and_run(&mindc, &runtime_src(-1), native);
        let high = build_and_run(&mindc, &runtime_src(5), native);
        assert_ne!(
            neg, 10,
            "negative index was CLAMPED to a[0] (native={native})"
        );
        assert_ne!(
            high, 30,
            "high index was CLAMPED to a[len-1] (native={native})"
        );
    }

    // Compile-time-provable OOB (literal index) -> deterministic compile-time
    // rejection, never a runnable artifact.
    for k in [3i64, 5, -1] {
        let src = format!(
            "fn main() -> i64 {{\n    let a: [i64; 3] = [10, 20, 30];\n    return a[{k}];\n}}\n"
        );
        let (rejected, blob) = check_rejects(&mindc, &src);
        assert!(
            rejected,
            "constant OOB a[{k}] must be rejected by `mindc check`:\n{blob}"
        );
    }
}

// The tree-evaluator's substrate representation of the SAME bounds-trap event —
// `EvalError::BoundsTrap` — is proven directly in a `src/eval/mod.rs` unit test
// (`eval_array_index_oob_is_bounds_trap`), which evaluates an out-of-bounds index
// and asserts the typed error. It is NOT exercised through `mindc test` here: the
// `mindc test` assert-condition evaluator has a pre-existing scope limitation
// (function-body locals are "unknown variable" inside an `assert` condition),
// orthogonal to the bounds-trap contract, that prevents reaching the index sites.
