use std::fs;
use std::process::Command;

use tempfile::tempdir;

#[test]
fn stdout_emit_default_lowering() {
    let out = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--no-default-features",
            "--",
            "eval",
            "1+2",
            "--emit-mlir",
            "--mlir-lower",
            "none",
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "stdout run failed: {:?}", out);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("module"));
    assert!(s.contains("arith.constant"));
}

#[test]
fn file_emit_with_preset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.mlir");

    let status = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--no-default-features",
            "--",
            "eval",
            "let x: Tensor[f32,(2,3)] = 0; x+1",
            "--emit-mlir-file",
            path.to_str().unwrap(),
            "--mlir-lower",
            "arith-linalg",
        ])
        .status()
        .expect("run");
    assert!(status.success());

    let txt = fs::read_to_string(&path).expect("read");
    assert!(txt.contains("tensor.empty") || txt.contains("linalg.fill"));
    assert!(txt.contains("arith.constant"));
}
