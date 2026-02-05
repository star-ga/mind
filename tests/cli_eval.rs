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

use std::path::PathBuf;
use std::process::Command;

/// Get the path to the mind binary from the cargo target directory
fn mind_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");

    // Use release or debug based on build profile
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

#[cfg(not(debug_assertions))]
#[ignore]
#[test]
fn _ignore_in_release_mode() {}

#[test]
fn mind_eval_basic_expr() {
    let binary = mind_binary();
    if !binary.exists() {
        // Skip if binary not built yet (cargo test without prior build)
        eprintln!("Skipping: mind binary not found at {:?}", binary);
        return;
    }

    let output = Command::new(&binary)
        .args(["eval", "2 + 3 * 4"])
        .output()
        .expect("failed to execute mind binary");

    assert!(
        output.status.success(),
        "mind eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    assert!(trimmed.contains("--- Lowered IR ---"), "{trimmed}");
    assert!(trimmed.contains("--- Result ---"), "{trimmed}");
    assert!(trimmed.ends_with("14"), "{trimmed}");
}
