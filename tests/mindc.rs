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

#[test]
fn mindc_accepts_cpu_target_flag() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/simple.mind",
            "--emit-ir",
            "--target",
            "cpu",
        ])
        .output()
        .expect("run mindc with cpu target");

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

#[test]
fn mindc_reports_unavailable_gpu_backend() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/simple.mind",
            "--target",
            "gpu",
        ])
        .output()
        .expect("run mindc gpu target");

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(stderr.contains("error[backend]"));
    assert!(stderr.contains("backend not available"));
}

#[test]
fn mindc_prints_json_diagnostics_with_flag() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/invalid.mind",
            "--diagnostic-format",
            "json",
        ])
        .output()
        .expect("run mindc json diagnostics");

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    let first_line = stderr.lines().next().unwrap_or("");
    let value: serde_json::Value = serde_json::from_str(first_line).expect("json diagnostic");
    assert!(value["phase"].is_string());
    assert_eq!(value["severity"], "error");
}

#[test]
fn mindc_color_env_overridden_by_flag() {
    let output = Command::new("cargo")
        .env("MINDC_COLOR", "always")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "tests/fixtures/invalid.mind",
            "--color",
            "never",
        ])
        .output()
        .expect("run mindc color flag");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("\u{1b}["),
        "stderr should be uncolored when flag forces never"
    );
}

#[test]
fn mindc_runs_conformance_suite() {
    let status = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "mindc",
            "--",
            "conformance",
            "--profile",
            "cpu",
        ])
        .status()
        .expect("run mindc conformance");

    assert!(status.success());
}
