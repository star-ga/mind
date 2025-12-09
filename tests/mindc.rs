// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::process::Command;

#[cfg(not(debug_assertions))]
#[ignore]
#[test]
fn _ignore_in_release_mode() {}

#[test]
fn mindc_emits_ir() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/simple.mind",
            "--emit-ir",
        ])
        .output()
        .expect("run mindc");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.to_lowercase().contains("output"), "{stdout}");
}

#[cfg(feature = "autodiff")]
#[test]
fn mindc_emits_grad_ir() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--features",
            "autodiff",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/autodiff.mind",
            "--func",
            "main",
            "--autodiff",
            "--emit-grad-ir",
        ])
        .output()
        .expect("run mindc autodiff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.to_lowercase().contains("output"), "{stdout}");
}

#[cfg(all(feature = "mlir-lowering", feature = "autodiff"))]
#[test]
fn mindc_emits_mlir() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--features",
            "mlir-lowering autodiff",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/autodiff.mind",
            "--func",
            "main",
            "--autodiff",
            "--emit-mlir",
        ])
        .output()
        .expect("run mindc mlir");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("func.func @main"), "{stdout}");
}

#[test]
fn mindc_verify_only_mode() {
    let status = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/simple.mind",
            "--verify-only",
        ])
        .status()
        .expect("run mindc verify");

    assert!(status.success());
}

#[test]
fn mindc_reports_prefixed_errors() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/invalid.mind",
        ])
        .output()
        .expect("run mindc error path");

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error[parse]")
            || stderr.contains("error[type-check]")
            || stderr.contains("error[ir-verify]"),
        "stderr should include standardized prefix: {stderr}"
    );
}
