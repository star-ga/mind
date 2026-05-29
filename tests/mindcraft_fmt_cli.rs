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

//! CLI integration tests for `mindc fmt`.
//!
//! Tests shell out to the compiled `mindc` binary via `CARGO_BIN_EXE_mindc`.
//! The test suite covers:
//!
//! 1. `--check` exits 0 on already-formatted input.
//! 2. `--check` exits 1 on drifted input.
//! 3. `--diff` prints a diff to stdout.
//! 4. `--stdin` with valid input writes formatted source to stdout.
//! 5. Default mode writes the file back atomically (tempdir).
//! 6. Directory walk picks up nested `*.mind` files.
//! 7. Exit 2 on invalid CLI usage (`--stdin` + positional paths).

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// The first already-formatted fixture — guaranteed to exit 0 under --check.
fn formatted_fixture_path() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/mindcraft/fmt/01_indent_if_else.out.mind"
    )
}

/// A trivially formatted MIND source string.
const ALREADY_FORMATTED: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

/// A MIND source string with extra internal whitespace that the formatter
/// will normalise (collapse `1  +  2` → `1 + 2`).
const NEEDS_FORMAT: &str = "fn add(a: i64, b: i64) -> i64 {\n    a  +  b\n}\n";

// ---------------------------------------------------------------------------
// Test 1: --check exits 0 on already-formatted file
// ---------------------------------------------------------------------------

#[test]
fn check_exits_0_on_already_formatted() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("clean.mind");
    fs::write(&file, ALREADY_FORMATTED).expect("write");

    let status = mindc()
        .args(["fmt", "--check"])
        .arg(&file)
        .status()
        .expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(0),
        "--check should exit 0 on already-formatted file"
    );
}

// ---------------------------------------------------------------------------
// Test 2: --check exits 1 on drifted file
// ---------------------------------------------------------------------------

#[test]
fn check_exits_1_on_drifted_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("dirty.mind");
    fs::write(&file, NEEDS_FORMAT).expect("write");

    let status = mindc()
        .args(["fmt", "--check"])
        .arg(&file)
        .status()
        .expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(1),
        "--check should exit 1 on a file that needs formatting"
    );
}

// ---------------------------------------------------------------------------
// Test 3: --diff prints a diff to stdout
// ---------------------------------------------------------------------------

#[test]
fn diff_prints_diff_to_stdout() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("dirty.mind");
    fs::write(&file, NEEDS_FORMAT).expect("write");

    let output = mindc()
        .args(["fmt", "--diff"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(1),
        "--diff should exit 1 when file would change"
    );
    // The diff must contain a `@@` hunk header.
    assert!(
        stdout.contains("@@"),
        "--diff output should contain a hunk header:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 3b: --diff exits 0 and emits nothing when file is already formatted
// ---------------------------------------------------------------------------

#[test]
fn diff_exits_0_on_already_formatted() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("clean.mind");
    fs::write(&file, ALREADY_FORMATTED).expect("write");

    let output = mindc()
        .args(["fmt", "--diff"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    assert_eq!(
        output.status.code(),
        Some(0),
        "--diff should exit 0 on an already-formatted file"
    );
    assert!(
        output.stdout.is_empty(),
        "--diff should emit nothing when file is clean"
    );
}

// ---------------------------------------------------------------------------
// Test 4: --stdin with valid input writes formatted source to stdout
// ---------------------------------------------------------------------------

#[test]
fn stdin_formats_and_writes_to_stdout() {
    let mut child = mindc()
        .args(["fmt", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mindc");

    {
        let stdin = child.stdin.as_mut().expect("stdin handle");
        stdin
            .write_all(NEEDS_FORMAT.as_bytes())
            .expect("write stdin");
    }

    let output = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(
        output.status.code(),
        Some(0),
        "--stdin should exit 0 on valid input; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The formatted output should not contain multiple consecutive spaces
    // between `a` and `+` or `+` and `b`.
    assert!(
        !stdout.contains("a  +"),
        "formatted output should not contain 'a  +': {stdout:?}"
    );
    assert!(
        !stdout.is_empty(),
        "--stdin should produce non-empty output"
    );
}

// ---------------------------------------------------------------------------
// Test 4b: --stdin with already-formatted source echoes it unchanged
// ---------------------------------------------------------------------------

#[test]
fn stdin_already_formatted_is_idempotent() {
    let mut child = mindc()
        .args(["fmt", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mindc");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin
            .write_all(ALREADY_FORMATTED.as_bytes())
            .expect("write");
    }

    let output = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        stdout.as_ref(),
        ALREADY_FORMATTED,
        "--stdin should preserve already-formatted source"
    );
}

// ---------------------------------------------------------------------------
// Test 5: default mode writes file back atomically
// ---------------------------------------------------------------------------

#[test]
fn default_mode_writes_file_in_place() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("dirty.mind");
    fs::write(&file, NEEDS_FORMAT).expect("write");

    let status = mindc().arg("fmt").arg(&file).status().expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(0),
        "default mode should exit 0 after formatting"
    );

    let after = fs::read_to_string(&file).expect("re-read");
    assert_ne!(
        after, NEEDS_FORMAT,
        "file should have been rewritten by the formatter"
    );
    // A second format pass should be idempotent.
    let status2 = mindc()
        .args(["fmt", "--check"])
        .arg(&file)
        .status()
        .expect("spawn mindc");
    assert_eq!(
        status2.code(),
        Some(0),
        "--check after in-place format should exit 0 (idempotence)"
    );
}

// ---------------------------------------------------------------------------
// Test 5b: default mode does not leave a .mind.tmp artefact behind
// ---------------------------------------------------------------------------

#[test]
fn default_mode_leaves_no_tmp_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("dirty.mind");
    let tmp = dir.path().join("dirty.mind.tmp");
    fs::write(&file, NEEDS_FORMAT).expect("write");

    mindc().arg("fmt").arg(&file).status().expect("spawn mindc");

    assert!(
        !tmp.exists(),
        ".mind.tmp artefact should be cleaned up after atomic rename"
    );
}

// ---------------------------------------------------------------------------
// Test 6: directory walk picks up nested *.mind files
// ---------------------------------------------------------------------------

#[test]
fn directory_walk_finds_nested_mind_files() {
    let dir = tempdir().expect("tempdir");
    // Create a nested structure.
    let sub = dir.path().join("sub");
    fs::create_dir_all(&sub).expect("mkdir sub");

    let file_top = dir.path().join("top.mind");
    let file_nested = sub.join("nested.mind");
    fs::write(&file_top, NEEDS_FORMAT).expect("write top");
    fs::write(&file_nested, NEEDS_FORMAT).expect("write nested");

    let status = mindc()
        .arg("fmt")
        .arg(dir.path())
        .status()
        .expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(0),
        "directory walk should exit 0 after formatting all files"
    );

    // Both files should now be formatted (--check exits 0).
    let c1 = mindc()
        .args(["fmt", "--check"])
        .arg(&file_top)
        .status()
        .expect("spawn");
    let c2 = mindc()
        .args(["fmt", "--check"])
        .arg(&file_nested)
        .status()
        .expect("spawn");

    assert_eq!(c1.code(), Some(0), "top-level file should be formatted");
    assert_eq!(c2.code(), Some(0), "nested file should be formatted");
}

// ---------------------------------------------------------------------------
// Test 7: exit 2 on --stdin combined with positional paths
// ---------------------------------------------------------------------------

#[test]
fn stdin_with_positional_paths_exits_2() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("x.mind");
    fs::write(&file, ALREADY_FORMATTED).expect("write");

    let status = mindc()
        .args(["fmt", "--stdin"])
        .arg(&file)
        .status()
        .expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(2),
        "--stdin with positional paths should exit 2 (usage error)"
    );
}

// ---------------------------------------------------------------------------
// Hard gate: --check on a .out.mind fixture exits 0
// ---------------------------------------------------------------------------

#[test]
fn hard_gate_check_on_formatted_fixture_exits_0() {
    let path = formatted_fixture_path();
    assert!(Path::new(path).exists(), "fixture not found: {path}");

    let status = mindc()
        .args(["fmt", "--check", path])
        .status()
        .expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(0),
        "--check on a .out.mind fixture should exit 0"
    );
}
