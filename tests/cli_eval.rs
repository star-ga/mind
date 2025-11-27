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
fn mind_eval_basic_expr() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--no-default-features",
            "--",
            "eval",
            "2 + 3 * 4",
        ])
        .output()
        .expect("run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    assert!(trimmed.contains("--- Lowered IR ---"), "{trimmed}");
    assert!(trimmed.contains("--- Result ---"), "{trimmed}");
    assert!(trimmed.ends_with("14"), "{trimmed}");
}
