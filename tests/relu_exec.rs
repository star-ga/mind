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

#[cfg(feature = "cpu-exec")]
use std::path::PathBuf;
#[cfg(feature = "cpu-exec")]
use std::process::Command;

/// Get the path to the mind binary from the cargo target directory
#[cfg(feature = "cpu-exec")]
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
fn relu_exec_non_negative() {
    let binary = mind_binary();
    if !binary.exists() {
        eprintln!("Skipping: mind binary not found at {:?}", binary);
        return;
    }

    let program = "let x: Tensor[f32,(1,4)] = 0; x = x - 3; tensor.relu(x + 1)";
    let output = Command::new(&binary)
        .args(["eval", "--exec", program])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "process failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Tensor[F32,(1,4)]"));
}
