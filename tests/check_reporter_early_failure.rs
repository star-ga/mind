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

//! `mindc check --reporter json|lsp` must report failures that stop the check
//! before any file is checked.
//!
//! A path that does not resolve, or (under `cross-module-imports`) a project
//! scope that cannot be installed, used to print only to stderr and exit 1 with
//! NOTHING on stdout. A CI step that parses the JSON then sees an empty result
//! next to a failing status, and a pipeline that reads only stdout reports the
//! file as clean. These drive the binary and parse its stdout.

mod common;

use std::process::{Command, Output};

fn check_json(args: &[&str]) -> Output {
    Command::new(common::require_mindc())
        .arg("check")
        .arg("--no-fmt")
        .args(args)
        .output()
        .expect("spawn mindc check")
}

/// Parse stdout as a JSON array and return its items.
fn items(out: &Output) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "stdout must be a JSON array, got {e}: stdout={text:?} stderr={:?}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    value
        .as_array()
        .unwrap_or_else(|| panic!("stdout must be a JSON array, got {value}"))
        .clone()
}

#[test]
fn json_reporter_reports_an_unresolvable_path() {
    let dir = common::scratch_dir("check_reporter_missing_path");
    let missing = dir.join("does_not_exist.mind");
    let path = missing.to_string_lossy().into_owned();
    let out = check_json(&["--reporter", "json", &path]);
    assert_eq!(out.status.code(), Some(1), "a missing path must fail check");
    let diags = items(&out);
    assert_eq!(
        diags.len(),
        1,
        "exactly one diagnostic for the failure: {diags:?}"
    );
    assert_eq!(diags[0]["severity"], "error", "{diags:?}");
    assert!(
        !diags[0]["message"].as_str().unwrap_or("").is_empty(),
        "the diagnostic must carry the reason: {diags:?}"
    );
}

#[test]
fn lsp_reporter_reports_an_unresolvable_path() {
    let dir = common::scratch_dir("check_reporter_missing_path_lsp");
    let missing = dir.join("does_not_exist.mind");
    let path = missing.to_string_lossy().into_owned();
    let out = check_json(&["--reporter", "lsp", &path]);
    assert_eq!(out.status.code(), Some(1), "a missing path must fail check");
    let diags = items(&out);
    assert_eq!(
        diags.len(),
        1,
        "exactly one diagnostic for the failure: {diags:?}"
    );
    assert_eq!(diags[0]["source"], "mindc", "{diags:?}");
}

#[test]
fn json_reporter_on_a_clean_file_stays_an_empty_array() {
    // Negative control: a clean file still prints `[]` and exits 0, so the
    // tests above fail on the early failure and on nothing else.
    let dir = common::scratch_dir("check_reporter_clean");
    let src = dir.join("clean.mind");
    std::fs::write(&src, "fn main() -> i64 {\n    return 0;\n}\n").expect("write fixture");
    let path = src.to_string_lossy().into_owned();
    let out = check_json(&["--reporter", "json", &path]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(items(&out).is_empty());
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn json_reporter_reports_a_missing_project_for_local_imports() {
    // A single file with a local import and no enclosing Mind.toml fails while
    // the project scope is installed, before any file is checked.
    let dir = common::scratch_dir("check_reporter_no_project");
    let src = dir.join("main.mind");
    std::fs::write(
        &src,
        "import sibling;\n\nfn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("write fixture");
    let path = src.to_string_lossy().into_owned();
    let out = check_json(&["--reporter", "json", &path]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a missing project must fail check"
    );
    let diags = items(&out);
    assert_eq!(diags.len(), 1, "exactly one diagnostic: {diags:?}");
    assert_eq!(diags[0]["severity"], "error", "{diags:?}");
    assert_eq!(diags[0]["rule_id"], "type_check::E2003", "{diags:?}");
    assert!(
        diags[0]["message"]
            .as_str()
            .unwrap_or("")
            .contains("sibling"),
        "the message must name the import: {diags:?}"
    );
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn json_reporter_reports_a_sibling_that_does_not_parse() {
    let dir = common::scratch_dir("check_reporter_bad_sibling");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"bad_sibling\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\nemit = \"cdylib\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"util.mind\"]\n",
    )
    .expect("write manifest");
    std::fs::write(dir.join("util.mind"), "pub fn inc(a: i64 -> i64 {\n")
        .expect("write broken sibling");
    std::fs::write(
        dir.join("main.mind"),
        "import util;\n\nfn main() -> i64 {\n    return util.inc(1);\n}\n",
    )
    .expect("write main");
    let path = dir.join("main.mind").to_string_lossy().into_owned();
    let out = check_json(&["--reporter", "json", &path]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a broken sibling must fail check"
    );
    let diags = items(&out);
    assert!(
        !diags.is_empty(),
        "the failure must reach stdout as a diagnostic, not only stderr; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(diags.iter().any(|d| d["severity"] == "error"), "{diags:?}");
}
