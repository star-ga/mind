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

//! RFC 0007 Phase 6 — `mindc check --fix` integration tests.
//!
//! Assertions:
//! (a) File is rewritten (fmt drift + trailing whitespace both fixed).
//! (b) After fix, `mindc check` exits 0 on the same file.
//! (c) Summary line is printed to stdout.

use std::fs;
use std::process::Command;

use tempfile::tempdir;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// Drifted source with extra spaces AND trailing whitespace on a line.
/// The formatter will normalise `a  +  b` to `a + b` (fmt drift).
/// The trailing spaces on line 2 will trigger lint::trailing_whitespace.
fn drifted_with_trailing_ws() -> String {
    // Line 1: fn decl (clean)
    // Line 2: body with extra spaces + trailing whitespace
    // Line 3: closing brace (clean)
    "fn add(a: i64, b: i64) -> i64 {\n    a  +  b   \n}\n".to_string()
}

/// Canonical form expected after all fixes applied.
const EXPECTED_FIXED: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

/// A canonically formatted, lint-clean source string.
const CLEAN: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

// ---------------------------------------------------------------------------
// Test A: file is rewritten
// ---------------------------------------------------------------------------

#[test]
fn check_fix_rewrites_drifted_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("fix_me.mind");
    fs::write(&file, drifted_with_trailing_ws()).expect("write");

    let out = mindc()
        .args(["check", "--fix"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    // check --fix exits 0 when all diagnostics are fixable.
    assert_eq!(
        out.status.code(),
        Some(0),
        "check --fix should exit 0 when all diagnostics are fixable; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let after = fs::read_to_string(&file).expect("read back");
    assert_eq!(
        after, EXPECTED_FIXED,
        "check --fix should rewrite the file to canonical form"
    );
}

// ---------------------------------------------------------------------------
// Test B: post-fix `mindc check` exits 0
// ---------------------------------------------------------------------------

#[test]
fn check_fix_then_check_passes() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("fix_then_check.mind");
    fs::write(&file, drifted_with_trailing_ws()).expect("write");

    // Apply fixes.
    let fix_out = mindc()
        .args(["check", "--fix"])
        .arg(&file)
        .output()
        .expect("spawn check --fix");

    assert_eq!(
        fix_out.status.code(),
        Some(0),
        "--fix should exit 0; stderr: {}",
        String::from_utf8_lossy(&fix_out.stderr)
    );

    // Re-check the fixed file — should be clean.
    let check_out = mindc()
        .args(["check"])
        .arg(&file)
        .output()
        .expect("spawn check after fix");

    assert_eq!(
        check_out.status.code(),
        Some(0),
        "post-fix check should exit 0; stdout: {} stderr: {}",
        String::from_utf8_lossy(&check_out.stdout),
        String::from_utf8_lossy(&check_out.stderr)
    );

    let stdout = String::from_utf8_lossy(&check_out.stdout);
    assert!(
        stdout.trim().is_empty(),
        "post-fix check should produce no diagnostics; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test C: summary line is printed
// ---------------------------------------------------------------------------

#[test]
fn check_fix_prints_summary() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("summary.mind");
    fs::write(&file, drifted_with_trailing_ws()).expect("write");

    let out = mindc()
        .args(["check", "--fix"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Fixed") && stdout.contains("unfixable"),
        "expected summary line containing 'Fixed ... unfixable ...' in stdout; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test D: clean file — exits 0, reports 0 fixed
// ---------------------------------------------------------------------------

#[test]
fn check_fix_clean_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("clean.mind");
    fs::write(&file, CLEAN).expect("write");

    let out = mindc()
        .args(["check", "--fix"])
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
    assert!(
        stdout.contains("Fixed 0 files"),
        "clean file should report 0 fixed; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test E: only trailing whitespace (no fmt drift) — auto-fixed
// ---------------------------------------------------------------------------

#[test]
fn check_fix_only_trailing_whitespace() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("ws_only.mind");
    // This is already fmt-canonical but has trailing whitespace.
    let source = "fn add(a: i64, b: i64) -> i64 {\n    a + b   \n}\n";
    let expected = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";
    fs::write(&file, source).expect("write");

    let out = mindc()
        .args(["check", "--fix"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    assert_eq!(
        out.status.code(),
        Some(0),
        "trailing-ws-only file should be fixable; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let after = fs::read_to_string(&file).expect("read back");
    assert_eq!(after, expected, "trailing whitespace should be removed");
}
