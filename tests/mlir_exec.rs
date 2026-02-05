// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::path::PathBuf;

/// Get the path to the mind binary from the cargo target directory
fn mind_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");

    #[cfg(debug_assertions)]
    path.push("debug");
    #[cfg(not(debug_assertions))]
    path.push("release");

    #[cfg(target_os = "windows")]
    path.push("mind.exe");
    #[cfg(not(target_os = "windows"))]
    path.push("mind");

    path
}

#[cfg(feature = "mlir-exec")]
#[test]
fn mlir_exec_scalar_add() {
    let binary = mind_binary();
    if !binary.exists() {
        eprintln!("Skipping: mind binary not found at {:?}", binary);
        return;
    }

    if which::which("mlir-opt").is_err() || which::which("mlir-cpu-runner").is_err() {
        eprintln!("skipping: mlir tools not found");
        return;
    }

    let output = std::process::Command::new(&binary)
        .args(["eval", "--mlir-exec", "1", "+", "2"])
        .output()
        .expect("run mlir exec");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("3"), "stdout: {}", stdout);
}

#[cfg(all(feature = "mlir-exec", feature = "cpu-exec"))]
#[test]
fn parity_cpu_vs_mlir_exec_simple() {
    let binary = mind_binary();
    if !binary.exists() {
        eprintln!("Skipping: mind binary not found at {:?}", binary);
        return;
    }

    if which::which("mlir-opt").is_err() || which::which("mlir-cpu-runner").is_err() {
        eprintln!("skipping: mlir tools not found");
        return;
    }

    let src = "let x: Tensor[f32,(2,2)] = 1; tensor.sum(x + 2)";

    let cpu = std::process::Command::new(&binary)
        .args(["eval", "--exec", src])
        .output()
        .expect("run cpu exec");
    assert!(
        cpu.status.success(),
        "cpu stderr: {}",
        String::from_utf8_lossy(&cpu.stderr)
    );
    let cpu_stdout = String::from_utf8_lossy(&cpu.stdout).trim().to_string();

    let mlir = std::process::Command::new(&binary)
        .args(["eval", "--mlir-exec", src])
        .output()
        .expect("run mlir exec");
    assert!(
        mlir.status.success(),
        "mlir stderr: {}",
        String::from_utf8_lossy(&mlir.stderr)
    );
    let mlir_stdout = String::from_utf8_lossy(&mlir.stdout).trim().to_string();

    assert!(
        !cpu_stdout.is_empty(),
        "cpu stdout empty: stderr={}",
        String::from_utf8_lossy(&cpu.stderr)
    );
    assert_eq!(cpu_stdout, mlir_stdout, "cpu vs mlir mismatch");
}
