#![cfg(feature = "mlir-build")]

use std::fs;
use std::process::Command;

use mind::eval;
use tempfile::tempdir;

fn ensure_toolchain() -> bool {
    eval::resolve_mlir_build_tools().map(|_| true).unwrap_or_else(|err| {
        eprintln!("Skipping CLI build test: {err}");
        false
    })
}

#[test]
fn cli_emits_artifacts() {
    if !ensure_toolchain() {
        return;
    }
    let dir = tempdir().expect("tempdir");
    let mlir_path = dir.path().join("cli.mlir");
    let llvm_path = dir.path().join("cli.ll");

    let status = Command::new(env!("CARGO_BIN_EXE_mind"))
        .arg("eval")
        .arg("--emit-mlir-file")
        .arg(&mlir_path)
        .arg("--emit-llvm-file")
        .arg(&llvm_path)
        .arg("--mlir-lower")
        .arg("core")
        .arg("1+2")
        .status()
        .expect("spawn mind cli");

    if !status.success() {
        panic!("CLI build failed: {status}");
    }

    let mlir_contents = fs::read_to_string(&mlir_path).expect("read cli mlir");
    assert!(mlir_contents.contains("module"));
    let llvm_contents = fs::read_to_string(&llvm_path).expect("read cli llvm");
    assert!(llvm_contents.contains("define"));
}

#[test]
fn cli_reports_missing_tool() {
    let dir = tempdir().expect("tempdir");
    let llvm_path = dir.path().join("cli.ll");

    let output = Command::new(env!("CARGO_BIN_EXE_mind"))
        .arg("eval")
        .arg("--emit-llvm-file")
        .arg(&llvm_path)
        .arg("1+1")
        .env("MLIR_TRANSLATE", "definitely_missing_mlir_translate")
        .output()
        .expect("spawn mind cli");

    assert!(!output.status.success(), "command unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("tool not found") || stderr.contains("mlir-build"),
        "missing tool diagnostic should mention missing tool; stderr: {stderr}"
    );
}
