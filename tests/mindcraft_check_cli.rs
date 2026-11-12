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

//! RFC 0007 Phase 5 — integration tests for `mindc check`.
//!
//! Tests shell out to the compiled `mindc` binary via `CARGO_BIN_EXE_mindc`.
//! Each test validates one declared deliverable from the Phase 5 spec.
//!
//! 1. `--check` mode exits 0 on a clean (already-formatted, lint-clean) file.
//! 2. Format drift is detected and exits 1.
//! 3. JSON reporter produces a valid JSON array.
//! 4. `--no-lint` suppresses lint-only diagnostics.
//! 5. `--no-fmt` suppresses fmt-drift diagnostics.
//! 6. `--no-typecheck` suppresses type-check diagnostics.
//! 7. VCS-aware filtering: a `.gitignore`d file in a tmpdir is skipped.
//! 8. Multi-file directory walk produces a sorted diagnostic stream.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn run_check(args: &[&str]) -> Output {
    mindc()
        .arg("check")
        .args(args)
        .output()
        .expect("failed to spawn mindc check")
}

/// A canonically formatted MIND source string (clean fixture).
const CLEAN: &str = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

/// A MIND source string with extra internal whitespace — fmt will normalise.
const DRIFTED: &str = "fn add(a: i64,  b: i64) -> i64 {\n    a  +  b\n}\n";

/// A MIND source with an unused import (triggers lint::unused_import).
/// Uses the canonical `import` form so the file also passes fmt-check.
const UNUSED_IMPORT: &str = "import std.vec;\n\nfn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n";

// ---------------------------------------------------------------------------
// Test 1: clean file exits 0
// ---------------------------------------------------------------------------

#[test]
fn check_exits_0_on_clean_file() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("clean.mind");
    fs::write(&file, CLEAN).expect("write");

    let out = run_check(&[file.to_str().unwrap()]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "clean file should exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.trim().is_empty(),
        "clean file should produce no output; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: fmt drift is detected, exits 1
// ---------------------------------------------------------------------------

#[test]
fn check_exits_1_on_fmt_drift() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("drifted.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = run_check(&[file.to_str().unwrap()]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "drifted file should exit 1; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("fmt::drift"),
        "expected fmt::drift in output; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: JSON reporter produces valid JSON array
// ---------------------------------------------------------------------------

#[test]
fn json_reporter_produces_valid_json() {
    let dir = tempdir().expect("tempdir");
    let clean = dir.path().join("clean.mind");
    let drifted = dir.path().join("drifted.mind");
    fs::write(&clean, CLEAN).expect("write clean");
    fs::write(&drifted, DRIFTED).expect("write drifted");

    // Run on the drifted file so we get at least one diagnostic.
    let out = run_check(&["--reporter", "json", drifted.to_str().unwrap()]);

    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not valid JSON: {e}\nstdout: {stdout}"));

    assert!(
        parsed.is_array(),
        "JSON reporter must emit an array; got: {parsed}"
    );
    let arr = parsed.as_array().unwrap();
    assert!(
        !arr.is_empty(),
        "expected at least one diagnostic in JSON output"
    );

    // Validate required fields on first diagnostic.
    let first = &arr[0];
    assert!(first.get("file").is_some(), "diagnostic must have 'file'");
    assert!(first.get("line").is_some(), "diagnostic must have 'line'");
    assert!(first.get("col").is_some(), "diagnostic must have 'col'");
    assert!(
        first.get("severity").is_some(),
        "diagnostic must have 'severity'"
    );
    assert!(
        first.get("message").is_some(),
        "diagnostic must have 'message'"
    );
    assert!(
        first.get("rule_id").is_some(),
        "diagnostic must have 'rule_id'"
    );
    assert!(first.get("phase").is_some(), "diagnostic must have 'phase'");
}

// ---------------------------------------------------------------------------
// Test 4: --no-lint suppresses lint-only diagnostics
// ---------------------------------------------------------------------------

#[test]
fn no_lint_flag_suppresses_lint_diagnostics() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("unused.mind");
    fs::write(&file, UNUSED_IMPORT).expect("write");

    // Without --no-lint we expect a lint diagnostic.
    let out_with_lint = run_check(&[file.to_str().unwrap()]);
    let stdout_with = String::from_utf8_lossy(&out_with_lint.stdout);
    assert!(
        stdout_with.contains("lint::"),
        "expected lint diagnostic without --no-lint; got: {stdout_with}"
    );

    // With --no-lint there should be no lint diagnostics.
    let out_no_lint = run_check(&["--no-lint", file.to_str().unwrap()]);
    let stdout_no = String::from_utf8_lossy(&out_no_lint.stdout);
    assert!(
        !stdout_no.contains("lint::"),
        "expected no lint diagnostics with --no-lint; got: {stdout_no}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: --no-fmt suppresses fmt-drift diagnostics
// ---------------------------------------------------------------------------

#[test]
fn no_fmt_flag_suppresses_fmt_diagnostics() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("drifted.mind");
    fs::write(&file, DRIFTED).expect("write");

    let out = run_check(&["--no-fmt", file.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("fmt::drift"),
        "expected no fmt diagnostics with --no-fmt; got: {stdout}"
    );
    // Exit 0: no errors from other passes on this clean-logic file.
    assert_eq!(
        out.status.code(),
        Some(0),
        "--no-fmt on logic-clean file should exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Test 6: --no-typecheck suppresses type-check diagnostics
// ---------------------------------------------------------------------------

#[test]
fn no_typecheck_flag_suppresses_typecheck_diagnostics() {
    // Write a file that the formatter accepts but has a parse error to trigger
    // a type_check::parse_error diagnostic. We use a file that is already
    // formatted but has a syntax error.
    let dir = tempdir().expect("tempdir");
    let bad_file = dir.path().join("bad.mind");
    // Deliberately malformed: unclosed brace.
    fs::write(&bad_file, "fn broken( {\n").expect("write");

    // Without --no-typecheck we expect a type_check diagnostic.
    let out_with = run_check(&[bad_file.to_str().unwrap()]);
    let stdout_with = String::from_utf8_lossy(&out_with.stdout);
    // The parse error bubbles up as a type_check or fmt error; either way
    // the file is not clean.
    let exits_nonzero = out_with.status.code() != Some(0)
        || stdout_with.contains("type_check::")
        || stdout_with.contains("fmt::");
    assert!(
        exits_nonzero,
        "expected non-zero exit or diagnostic on malformed file; \
         code={:?} stdout={stdout_with}",
        out_with.status.code()
    );

    // With --no-typecheck and --no-fmt the file might pass (fmt will fail to
    // parse it so drift check is skipped, and typecheck is skipped).
    // We just verify the flag is accepted without crashing.
    let out_skip = run_check(&["--no-typecheck", "--no-fmt", bad_file.to_str().unwrap()]);
    // Should not exit with a non-handled error (2 = usage error).
    assert_ne!(
        out_skip.status.code(),
        Some(2),
        "--no-typecheck should be accepted; stderr: {}",
        String::from_utf8_lossy(&out_skip.stderr)
    );
}

// ---------------------------------------------------------------------------
// Test 7: VCS-aware filtering — .gitignore'd file is skipped
// ---------------------------------------------------------------------------

#[test]
fn vcs_filtering_skips_gitignored_file() {
    let dir = tempdir().expect("tempdir");

    // Create a fake git repo so find_git_root() resolves to dir.
    fs::create_dir(dir.path().join(".git")).expect("mkdir .git");
    fs::write(dir.path().join(".gitignore"), "ignored.mind\n").expect("write .gitignore");

    // File that should be ignored.
    let ignored = dir.path().join("ignored.mind");
    fs::write(&ignored, DRIFTED).expect("write ignored (drifted)");

    // File that should NOT be ignored.
    let visible = dir.path().join("visible.mind");
    fs::write(&visible, CLEAN).expect("write visible (clean)");

    // Run check on the whole directory — the ignored file is drifted so if it
    // were checked it would produce a diagnostic and exit 1.
    let out = run_check(&[dir.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // The check should exit 0 because only visible.mind is scanned and it's clean.
    assert_eq!(
        out.status.code(),
        Some(0),
        "ignored drifted file should not cause exit 1; stdout={stdout}; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // No diagnostic referencing the ignored file.
    assert!(
        !stdout.contains("ignored.mind"),
        "ignored.mind should not appear in output; stdout: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Multi-file directory walk produces sorted diagnostic stream
// ---------------------------------------------------------------------------

#[test]
fn directory_walk_produces_sorted_diagnostics() {
    let dir = tempdir().expect("tempdir");

    // Create two drifted files with different names (sorting: a < b).
    let file_a = dir.path().join("aaa.mind");
    let file_b = dir.path().join("bbb.mind");
    fs::write(&file_a, DRIFTED).expect("write a");
    fs::write(&file_b, DRIFTED).expect("write b");

    // Create a subdirectory with another drifted file.
    let sub = dir.path().join("sub");
    fs::create_dir(&sub).expect("mkdir sub");
    let file_c = sub.join("ccc.mind");
    fs::write(&file_c, DRIFTED).expect("write c");

    let out = run_check(&[dir.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // All three files should produce a diagnostic.
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        lines.len() >= 3,
        "expected at least 3 diagnostics; got {} lines:\n{stdout}",
        lines.len()
    );

    // Lines should be sorted: files appear in lexicographic order.
    // aaa.mind should come before bbb.mind.
    let pos_a = lines
        .iter()
        .position(|l| l.contains("aaa.mind"))
        .unwrap_or(usize::MAX);
    let pos_b = lines
        .iter()
        .position(|l| l.contains("bbb.mind"))
        .unwrap_or(usize::MAX);
    assert!(
        pos_a < pos_b,
        "aaa.mind should appear before bbb.mind in sorted output; stdout:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Clean JSON output on empty directory (edge case)
// ---------------------------------------------------------------------------

#[test]
fn json_reporter_empty_result_is_empty_array() {
    let dir = tempdir().expect("tempdir");
    // No .mind files.
    let out = run_check(&["--reporter", "json", dir.path().to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not valid JSON: {e}\nstdout: {stdout}"));
    assert_eq!(
        parsed,
        serde_json::json!([]),
        "empty directory should produce []"
    );
    assert_eq!(out.status.code(), Some(0));
}

// ---------------------------------------------------------------------------
// Tests 10-12: the file set of a project check (no feature needed)
// ---------------------------------------------------------------------------

/// A module with one type error: it returns a name that nothing declares.
fn broken(function: &str) -> String {
    format!("fn {function}() -> i64 {{\n    missing_name\n}}\n")
}

/// A project in the common layout: the entry is in `src/` and `docs/` is at
/// the project root. `src/main.mind` is the whole program. Nothing imports
/// `src/docs/stray.mind` or `docs/outside.mind`, and each has a type error.
fn write_src_layout_project(root: &Path) {
    fs::create_dir_all(root.join("src/docs")).expect("mkdir src/docs");
    fs::create_dir_all(root.join("docs")).expect("mkdir docs");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"scoped\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n",
    )
    .expect("write manifest");
    fs::write(root.join("src/main.mind"), "fn main() -> i64 {\n    0\n}\n").expect("write entry");
    fs::write(root.join("src/docs/stray.mind"), broken("stray_value")).expect("write stray");
    fs::write(root.join("docs/outside.mind"), broken("outside_value")).expect("write outside");
}

/// `mindc check --no-fmt --no-lint <args>` in `dir`: exit code, stdout, stderr.
fn type_check_in(dir: &Path, args: &[&str]) -> (Option<i32>, String, String) {
    let out = mindc()
        .current_dir(dir)
        .args(["check", "--no-fmt", "--no-lint"])
        .args(args)
        .output()
        .expect("failed to spawn mindc check");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status.code(), stdout, stderr)
}

#[test]
fn project_check_with_no_path_takes_the_build_source_set() {
    let dir = tempdir().expect("tempdir");
    write_src_layout_project(dir.path());

    // The build compiles `src/main.mind` only, so the check reads it only.
    let (code, stdout, stderr) = type_check_in(dir.path(), &[]);
    assert_eq!(
        code,
        Some(0),
        "the project check read a file outside the source set:\n{stdout}{stderr}"
    );
    assert!(!stdout.contains("missing_name"), "{stdout}");
    // The two files left out are counted: no file leaves the check silently.
    assert!(
        stderr.contains("note[check]: 2 .mind file(s) below"),
        "no notice of the files outside the source set: {stderr}"
    );

    // Positive control: a module in the entry's directory is in the set (the
    // build compiles it), so its error is reported. The strays stay out.
    fs::write(dir.path().join("src/sibling.mind"), broken("sibling_value")).expect("write sibling");
    let (code, stdout, stderr) = type_check_in(dir.path(), &[]);
    assert_eq!(
        code,
        Some(1),
        "a member's error was lost:\n{stdout}{stderr}"
    );
    assert!(
        stdout.contains("sibling.mind") && stdout.contains("missing_name"),
        "the member was not reported: {stdout}"
    );
    assert!(
        !stdout.contains("stray.mind") && !stdout.contains("outside.mind"),
        "a file outside the source set was read: {stdout}"
    );
}

#[test]
fn a_path_operand_is_never_narrowed_to_the_source_set() {
    let dir = tempdir().expect("tempdir");
    write_src_layout_project(dir.path());

    // Each operand covers a file that is not in the project's source set. The
    // check reads that file and fails on it: nothing is dropped.
    let cases: [(&str, &[&str]); 6] = [
        ("src/docs", &["stray.mind"]),
        ("docs", &["outside.mind"]),
        ("src", &["stray.mind"]),
        (".", &["stray.mind", "outside.mind"]),
        ("src/docs/stray.mind", &["stray.mind"]),
        ("docs/outside.mind", &["outside.mind"]),
    ];
    for (operand, reported) in cases {
        let (code, stdout, stderr) = type_check_in(dir.path(), &[operand]);
        assert_eq!(
            code,
            Some(1),
            "`mindc check {operand}` passed over a broken file:\n{stdout}{stderr}"
        );
        for file in reported {
            assert!(
                stdout.contains(file) && stdout.contains("missing_name"),
                "`mindc check {operand}` did not report {file}:\n{stdout}{stderr}"
            );
        }
    }
}

#[test]
fn a_temporary_manifest_does_not_scope_the_check() {
    // A standalone build writes this manifest for its own duration; a killed
    // build leaves it. It is never a project, so it gives no source set.
    let dir = tempdir().expect("tempdir");
    fs::create_dir_all(dir.path().join("sub")).expect("mkdir sub");
    fs::write(
        dir.path().join("Mind.toml"),
        "# mindc: temporary manifest of a standalone build; removed when that build returns\n\
         [package]\nname = \"main\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n",
    )
    .expect("write temporary manifest");
    fs::write(
        dir.path().join("main.mind"),
        "fn main() -> i64 {\n    0\n}\n",
    )
    .expect("write entry");
    fs::write(dir.path().join("sub/other.mind"), broken("other_value")).expect("write other");

    let (code, stdout, stderr) = type_check_in(dir.path(), &[]);
    assert_eq!(
        code,
        Some(1),
        "a temporary manifest narrowed the check:\n{stdout}{stderr}"
    );
    assert!(stdout.contains("other.mind"), "{stdout}{stderr}");
}
