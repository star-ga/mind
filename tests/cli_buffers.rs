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

#[cfg(feature = "cpu-buffers")]
#[test]
fn cli_shows_materialized_note() {
    let out = std::process::Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--features",
            "cpu-buffers",
            "--",
            "eval",
            "let x: Tensor[f32,(2,3)] = 1; tensor.materialize(x)",
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("materialized"));
}
