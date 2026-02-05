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
use std::process::Command;

/// Get the path to the mindc binary from the cargo target directory
fn mindc_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");

    // Use release or debug based on build profile
    #[cfg(debug_assertions)]
    path.push("debug");
    #[cfg(not(debug_assertions))]
    path.push("release");

    #[cfg(target_os = "windows")]
    path.push("mindc.exe");
    #[cfg(not(target_os = "windows"))]
    path.push("mindc");

    path
}

/// Check if the mindc binary exists, skip test if not
fn require_mindc() -> PathBuf {
    let binary = mindc_binary();
    if !binary.exists() {
        eprintln!("Skipping: mindc binary not found at {:?}", binary);
    }
    binary
}

#[cfg(not(debug_assertions))]
#[ignore]
#[test]
fn _ignore_in_release_mode() {}

#[test]
fn mindc_emits_ir() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args(["tests/fixtures/simple.mind", "--emit-ir"])
        .output()
        .expect("run mindc");

    assert!(
        output.status.success(),
        "mindc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.to_lowercase().contains("output"), "{stdout}");
}

#[test]
fn mindc_accepts_cpu_target_flag() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args([
            "tests/fixtures/simple.mind",
            "--emit-ir",
            "--target",
            "cpu",
        ])
        .output()
        .expect("run mindc with cpu target");

    assert!(
        output.status.success(),
        "mindc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.to_lowercase().contains("output"), "{stdout}");
}

#[cfg(feature = "autodiff")]
#[test]
fn mindc_emits_grad_ir() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args([
            "tests/fixtures/autodiff.mind",
            "--func",
            "main",
            "--autodiff",
            "--emit-grad-ir",
        ])
        .output()
        .expect("run mindc autodiff");

    assert!(
        output.status.success(),
        "mindc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.to_lowercase().contains("output"), "{stdout}");
}

#[test]
fn mindc_verify_only_mode() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let status = Command::new(&binary)
        .args(["tests/fixtures/simple.mind", "--verify-only"])
        .status()
        .expect("run mindc verify");

    assert!(status.success());
}

#[test]
fn mindc_reports_prefixed_errors() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args(["tests/fixtures/invalid.mind"])
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
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args(["tests/fixtures/simple.mind", "--target", "gpu"])
        .output()
        .expect("run mindc gpu target");

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(stderr.contains("error[backend]"));
    assert!(stderr.contains("no backend available"));
}

#[test]
fn mindc_prints_json_diagnostics_with_flag() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args([
            "tests/fixtures/invalid.mind",
            "--diagnostic-format",
            "json",
        ])
        .output()
        .expect("run mindc json diagnostics");

    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Find the first line that looks like JSON (starts with '{')
    let json_line = stderr
        .lines()
        .find(|line| line.trim().starts_with('{'))
        .expect("should have json diagnostic line");
    let value: serde_json::Value = serde_json::from_str(json_line).expect("json diagnostic");
    assert!(value["phase"].is_string());
    assert_eq!(value["severity"], "error");
}

#[test]
fn mindc_reports_shape_errors_with_codes() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .args([
            "tests/fixtures/invalid_broadcast.mind",
            "--diagnostic-format",
            "json",
        ])
        .output()
        .expect("run mindc shape error");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Find the first line that looks like JSON (starts with '{')
    let json_line = stderr
        .lines()
        .find(|line| line.trim().starts_with('{'))
        .expect("should have json diagnostic line");
    let value: serde_json::Value = serde_json::from_str(json_line).expect("json diagnostic");
    assert_eq!(value["code"], "E2101");
    assert_eq!(value["phase"], "type-check");
}

#[test]
fn mindc_color_env_overridden_by_flag() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let output = Command::new(&binary)
        .env("MINDC_COLOR", "always")
        // Ensure no terminal is detected
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .args([
            "tests/fixtures/invalid.mind",
            "--color",
            "never",
        ])
        .output()
        .expect("run mindc color flag");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Find lines that look like mindc output (not cargo warnings)
    let mindc_output: String = stderr
        .lines()
        .filter(|line| line.contains("error[") || line.contains("warning["))
        .collect::<Vec<_>>()
        .join("\n");
    // If no mindc-specific output found, check the full stderr
    let to_check = if mindc_output.is_empty() {
        stderr.to_string()
    } else {
        mindc_output
    };
    assert!(
        !to_check.contains("\u{1b}["),
        "mindc output should be uncolored when flag forces never: {to_check}"
    );
}

#[test]
fn mindc_runs_conformance_suite() {
    let binary = require_mindc();
    if !binary.exists() {
        return;
    }

    let status = Command::new(&binary)
        .args(["conformance", "--profile", "cpu"])
        .status()
        .expect("run mindc conformance");

    assert!(status.success());
}
