// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! `mindc build` / `mindc run` must consult the runnable-artifact ABI gate (#54).
//!
//! `products.runnable_blockers` — the constructs the shipped i64-scalar backend
//! would SILENTLY MISCOMPILE — was consulted ONLY on the single-file
//! `--emit-obj` / `--emit-shared` path (`src/bin/mindc.rs`). The `mindc build` /
//! `mindc run` compile path (`build::run_build` -> `project::compile_sources` ->
//! `compile_single_source`, and the cdylib `build_cdylib_from_entry`) NEVER
//! checked it — so a program that `--emit-shared` fail-loud REJECTS built GREEN
//! (rc=0) and ran WRONG via the primary commands. The fix consults
//! `runnable_blockers` in the build/run compile path and fails non-zero.
//!
//! This test uses the enum-handle-in-scalar-return blocker (`divide(_,0)` returns
//! `Res::Err(0)` where `-> i64` is declared — a leaked heap-record pointer),
//! confirms `--emit-shared` rejects it, then asserts `mindc build` AND `mindc run`
//! now exit NON-ZERO (they returned rc=0 before).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test build_run_runnable_blocker_gate`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::fs;
use std::process::Command;

// Bare-scalar `-> i64` return that yields an enum constructor handle on one path
// — a `runnable_blocker` (`lower::enum_handle_in_scalar_return`).
const BLOCKER_MIND: &str = r#"enum Res { Ok(i64), Err(i64) }
fn divide(a: i64, b: i64) -> i64 {
    if b == 0 { return Res::Err(0); }
    return a / b;
}
fn main() -> i64 { return divide(4, 2); }
"#;

fn manifest(entry: &str) -> String {
    format!(
        "[package]\nname = \"blockerproj\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"{entry}\"\noutput = \"blockerproj\"\n"
    )
}

#[test]
fn build_and_run_reject_runnable_blocker() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("build-run-runnable-blocker-gate: mindc not found; skipping");
        return;
    }
    let td = tempfile::tempdir().expect("tempdir");
    let src_dir = td.path().join("src");
    fs::create_dir_all(&src_dir).expect("mkdir src");
    fs::write(src_dir.join("main.mind"), BLOCKER_MIND).expect("write src");
    fs::write(td.path().join("Mind.toml"), manifest("src/main.mind")).expect("write manifest");

    // (0) Baseline: the single-file `--emit-shared` path already rejects it.
    let so = td.path().join("blocker.so");
    let es = Command::new(&mindc)
        .args([
            src_dir.join("main.mind").to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc --emit-shared");
    let es_err = String::from_utf8_lossy(&es.stderr);
    if es_err.contains("mlir-build") && es_err.contains("requires") {
        println!("build-run-runnable-blocker-gate: needs mlir-build; skipping");
        return;
    }
    assert!(
        !es.status.success(),
        "--emit-shared must reject the runnable_blocker but exited 0"
    );
    assert!(
        es_err.contains("enum_handle_in_scalar_return"),
        "expected the enum-handle-scalar-return diagnostic; got:\n{es_err}"
    );

    // (1) `mindc build` must now ALSO reject it (was rc=0 before the fix).
    let build = Command::new(&mindc)
        .arg("build")
        .current_dir(td.path())
        .output()
        .expect("run mindc build");
    assert!(
        !build.status.success(),
        "mindc build must fail non-zero on a runnable_blocker; stdout={} stderr={}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr),
    );
    assert!(
        String::from_utf8_lossy(&build.stderr).contains("runnable artifact"),
        "mindc build error should name the runnable-artifact refusal; got:\n{}",
        String::from_utf8_lossy(&build.stderr),
    );

    // (2) `mindc run` must now ALSO reject it (was rc=0 before the fix).
    let run = Command::new(&mindc)
        .arg("run")
        .current_dir(td.path())
        .output()
        .expect("run mindc run");
    assert!(
        !run.status.success(),
        "mindc run must fail non-zero on a runnable_blocker; stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
}
