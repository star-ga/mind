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

#[cfg(feature = "cpu-exec")]
#[test]
fn cli_runs_exec() {
    let binary = mind_binary();
    if !binary.exists() {
        eprintln!("Skipping: mind binary not found at {:?}", binary);
        return;
    }

    let output = std::process::Command::new(&binary)
        .args([
            "eval",
            "--exec",
            "let x: Tensor[f32,(2,2)] = 1; tensor.matmul(x,x)",
        ])
        .output()
        .expect("run");
    assert!(
        output.status.success(),
        "process failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Tensor["));
    assert!(stdout.contains("(2,2)"));
    // In open-core build, runtime stubs return Unsupported so exec falls back
    // to preview mode with "fill=" output. Materialized output requires the
    // proprietary mind-runtime backend.
    assert!(
        stdout.contains("materialized") || stdout.contains("fill="),
        "expected materialized or preview output, got: {stdout}"
    );
}
