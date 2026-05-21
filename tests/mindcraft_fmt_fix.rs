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

//! RFC 0007 Phase 6 — `mindc fmt --fix` integration tests.
//!
//! Assertions:
//! (a) File is rewritten to canonical form.
//! (b) Summary line "Formatted N files, M unchanged." is printed to stdout.
//! (c) Exit code is 0.

use std::fs;
use std::process::Command;

use tempfile::tempdir;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// A MIND source string with extra internal whitespace — formatter normalises.
const DRIFTED: &str = "fn add(a: i64, b: i64) -> i64 {\n    a  +  b\n}\n";

/// The expected formatted form of `DRIFTED`.
const FORMATTED: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

/// An already-formatted MIND source string.
const CLEAN: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

// ---------------------------------------------------------------------------
// Test A: file is rewritten
// ---------------------------------------------------------------------------

#[test]
fn fmt_fix_rewrites_drifted_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("drifted.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = mindc()
        .args(["fmt", "--fix"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    assert_eq!(
        out.status.code(),
        Some(0),
        "exit code should be 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let after = fs::read_to_string(&file).expect("read back");
    assert_eq!(
        after, FORMATTED,
        "--fix should rewrite the file to canonical form"
    );
}

// ---------------------------------------------------------------------------
// Test B: summary line is printed to stdout
// ---------------------------------------------------------------------------

#[test]
fn fmt_fix_prints_summary_line() {
    let dir = tempdir().expect("tempdir");
    let drifted = dir.path().join("drifted.mind");
    let clean = dir.path().join("clean.mind");
    fs::write(&drifted, DRIFTED).expect("write drifted");
    fs::write(&clean, CLEAN).expect("write clean");

    let out = mindc()
        .args(["fmt", "--fix"])
        .arg(&drifted)
        .arg(&clean)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Formatted 1 file") && stdout.contains("1 unchanged"),
        "expected summary 'Formatted 1 file, 1 unchanged.' in stdout; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test C: exit code 0
// ---------------------------------------------------------------------------

#[test]
fn fmt_fix_exits_0() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("any.mind");
    fs::write(&file, DRIFTED).expect("write");

    let status = mindc()
        .args(["fmt", "--fix"])
        .arg(&file)
        .status()
        .expect("spawn mindc");

    assert_eq!(
        status.code(),
        Some(0),
        "mindc fmt --fix should always exit 0"
    );
}

// ---------------------------------------------------------------------------
// Test D: clean file — summary shows 0 formatted, 1 unchanged
// ---------------------------------------------------------------------------

#[test]
fn fmt_fix_clean_file_summary() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("clean.mind");
    fs::write(&file, CLEAN).expect("write");

    let out = mindc()
        .args(["fmt", "--fix"])
        .arg(&file)
        .output()
        .expect("spawn mindc");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Formatted 0 files") && stdout.contains("1 unchanged"),
        "expected 'Formatted 0 files, 1 unchanged.' for a clean file; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test E: idempotence — running --fix twice on a drifted file is a no-op
//          on the second invocation (file already clean after first).
// ---------------------------------------------------------------------------

#[test]
fn fmt_fix_is_idempotent() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("idem.mind");
    fs::write(&file, DRIFTED).expect("write");

    // First pass: rewrites.
    mindc()
        .args(["fmt", "--fix"])
        .arg(&file)
        .status()
        .expect("first pass");

    let after_first = fs::read_to_string(&file).expect("read after first");
    assert_eq!(after_first, FORMATTED, "first pass should format the file");

    // Second pass: file is already canonical — summary shows 0 formatted.
    let out = mindc()
        .args(["fmt", "--fix"])
        .arg(&file)
        .output()
        .expect("second pass");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Formatted 0 files"),
        "second pass should report no files formatted; got: {stdout}"
    );

    let after_second = fs::read_to_string(&file).expect("read after second");
    assert_eq!(after_second, FORMATTED, "second pass should not change the file");
}
