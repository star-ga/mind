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

//! RFC 0007 Phase 6 — `mindc check --reporter lsp` integration tests.
//!
//! Assertions:
//! - stdout is valid JSON (a JSON array).
//! - Each element matches the LSP Diagnostic shape:
//!   { uri, range: { start: { line, character }, end: { line, character } },
//!     severity, message, source, code }
//! - `uri` begins with "file://".
//! - `range.start.line` and `range.start.character` are non-negative integers.
//! - `severity` is 1 (Error), 2 (Warning), or 3 (Information).
//! - `source` is "mindc".
//! - `code` matches the rule_id (e.g. "fmt::drift").

use std::fs;
use std::process::Command;

use tempfile::tempdir;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// A MIND source with fmt drift (extra spaces).
const DRIFTED: &str = "fn add(a: i64, b: i64) -> i64 {\n    a  +  b\n}\n";

/// A canonically formatted, lint-clean source string.
const CLEAN: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

// ---------------------------------------------------------------------------
// Test 1: stdout is valid JSON array
// ---------------------------------------------------------------------------

#[test]
fn lsp_reporter_produces_valid_json() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("drifted.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = mindc()
        .args(["check", "--reporter", "lsp"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout is not valid JSON: {e}\nstdout: {stdout}"));

    assert!(
        parsed.is_array(),
        "--reporter lsp must produce a JSON array; got: {parsed}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: each diagnostic object has the required LSP fields
// ---------------------------------------------------------------------------

#[test]
fn lsp_reporter_diagnostic_shape() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("drift.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = mindc()
        .args(["check", "--reporter", "lsp"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let array: serde_json::Value = serde_json::from_str(&stdout)
        .expect("valid JSON");
    let items = array.as_array().expect("JSON array");

    assert!(!items.is_empty(), "expected at least one diagnostic for a drifted file");

    for item in items {
        // uri
        let uri = item["uri"].as_str().expect("uri must be a string");
        assert!(
            uri.starts_with("file://"),
            "uri must start with 'file://'; got: {uri}"
        );

        // range.start.line (0-based integer)
        let line = item["range"]["start"]["line"]
            .as_u64()
            .expect("range.start.line must be a non-negative integer");
        let _char = item["range"]["start"]["character"]
            .as_u64()
            .expect("range.start.character must be a non-negative integer");
        let _ = line;

        // range.end mirrors start (single-position diagnostics)
        item["range"]["end"]["line"]
            .as_u64()
            .expect("range.end.line must be present");
        item["range"]["end"]["character"]
            .as_u64()
            .expect("range.end.character must be present");

        // severity: 1=Error, 2=Warning, 3=Information
        let sev = item["severity"].as_u64().expect("severity must be an integer");
        assert!(
            (1..=3).contains(&sev),
            "severity must be 1, 2, or 3; got: {sev}"
        );

        // message
        assert!(
            item["message"].is_string(),
            "message must be a string"
        );

        // source = "mindc"
        assert_eq!(
            item["source"].as_str(),
            Some("mindc"),
            "source must be 'mindc'"
        );

        // code = rule_id
        assert!(
            item["code"].is_string(),
            "code must be a string (rule_id)"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: fmt::drift diagnostic has severity 1 (Error)
// ---------------------------------------------------------------------------

#[test]
fn lsp_fmt_drift_is_error_severity() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("drift2.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = mindc()
        .args(["check", "--reporter", "lsp"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let array: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let items = array.as_array().expect("array");

    let drift = items
        .iter()
        .find(|d| d["code"].as_str() == Some("fmt::drift"))
        .expect("expected a fmt::drift diagnostic");

    assert_eq!(
        drift["severity"].as_u64(),
        Some(1),
        "fmt::drift must have LSP severity 1 (Error)"
    );
}

// ---------------------------------------------------------------------------
// Test 4: clean file produces an empty JSON array
// ---------------------------------------------------------------------------

#[test]
fn lsp_reporter_empty_array_for_clean_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("clean.mind");
    fs::write(&file, CLEAN).expect("write");

    let out = mindc()
        .args(["check", "--reporter", "lsp"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    assert_eq!(
        out.status.code(),
        Some(0),
        "clean file should exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let array: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let items = array.as_array().expect("JSON array");

    assert!(
        items.is_empty(),
        "clean file should produce an empty LSP array; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: uri field encodes the absolute path to the file
// ---------------------------------------------------------------------------

#[test]
fn lsp_reporter_uri_encodes_file_path() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("uri_check.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = mindc()
        .args(["check", "--reporter", "lsp"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let array: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let items = array.as_array().expect("array");

    assert!(!items.is_empty(), "expected diagnostics");

    let uri = items[0]["uri"].as_str().expect("uri string");
    let file_str = file.to_string_lossy();
    assert!(
        uri.contains(file_str.as_ref()),
        "uri should contain the file path; uri={uri}, path={file_str}"
    );
}
