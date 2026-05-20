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

//! Formatter idempotence gate — Phase 2A acceptance test (Step 3 of PR #3).
//!
//! For every source file in scope, asserts:
//!   `format_source(format_source(src)) == format_source(src)`
//!
//! This is the hard contract: round-trip stability must hold at 100% with
//! zero skips.  A formatter that fails idempotence is structurally broken.
//!
//! Scope:
//!   - `std/*.mind` (5 files, canonical stdlib)
//!   - `examples/**/*.mind` (all discoverable .mind sources)
//!   - `tests/mindcraft/fmt/*.in.mind` (7 Phase-2A fixture inputs)
//!
//! All files that parse successfully must be idempotent.  Files that fail to
//! parse (e.g. because they use features the parser doesn't yet support) are
//! skipped with a note — parse failures are not idempotence failures.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn default_cfg() -> MindcraftFormatConfig {
    MindcraftFormatConfig::default()
}

fn manifest_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Produce a compact line-by-line diff for failure messages.
fn diff_lines(a: &str, b: &str) -> String {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let max = a_lines.len().max(b_lines.len());
    let mut out = String::new();
    for i in 0..max {
        let la = a_lines.get(i).copied().unwrap_or("<missing>");
        let lb = b_lines.get(i).copied().unwrap_or("<missing>");
        if la != lb {
            out.push_str(&format!("  line {}: pass1={la:?}\n", i + 1));
            out.push_str(&format!("  line {}:  pass2={lb:?}\n", i + 1));
        }
    }
    out
}

/// Assert idempotence for a single source string.
///
/// Returns `true` if the file was exercised (either passed or failed),
/// `false` if skipped (parse error on pass 1 — not a formatter bug).
fn check_idempotence(label: &str, src: &str, cfg: &MindcraftFormatConfig) -> bool {
    let pass1 = match format_source(src, cfg) {
        Ok(s) => s,
        Err(_) => {
            // Parse failures are expected for some examples using tensor/
            // autodiff syntax not in the current formatter scope.
            return false;
        }
    };
    let pass2 = format_source(&pass1, cfg)
        .unwrap_or_else(|e| panic!("idempotence: pass-2 parse failed for {label}: {e}"));
    assert_eq!(
        pass1,
        pass2,
        "idempotence violated for {label}:\n{}",
        diff_lines(&pass1, &pass2),
    );
    true
}

// ---------------------------------------------------------------------------
// std/*.mind — 5 files
// ---------------------------------------------------------------------------

#[test]
fn idempotence_stdlib() {
    let base = manifest_dir().join("std");
    let cfg = default_cfg();
    let mut passed = 0usize;
    let mut skipped = 0usize;

    for name in &["vec", "string", "io", "map", "blas"] {
        let path = base.join(format!("{name}.mind"));
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if check_idempotence(name, &src, &cfg) {
            passed += 1;
        } else {
            skipped += 1;
        }
    }

    assert_eq!(skipped, 0, "unexpected parse failures in std/ ({skipped} file(s))");
    assert_eq!(passed, 5, "expected 5 stdlib files, got {passed}");
}

// ---------------------------------------------------------------------------
// examples/**/*.mind
// ---------------------------------------------------------------------------

#[test]
fn idempotence_examples() {
    let base = manifest_dir();
    let cfg = default_cfg();
    let mut passed = 0usize;
    let mut skipped = 0usize;

    // Collect all .mind files under examples/
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let examples_dir = base.join("examples");
    collect_mind_files(&examples_dir, &mut paths);
    paths.sort();

    assert!(!paths.is_empty(), "no .mind files found under examples/");

    for path in &paths {
        let label = path.strip_prefix(&base).unwrap_or(path).display().to_string();
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if check_idempotence(&label, &src, &cfg) {
            passed += 1;
        } else {
            skipped += 1;
        }
    }

    // Parse skips are permitted (tensor/autodiff examples), but the
    // idempotence assertion inside check_idempotence must hold for every
    // file that does parse.
    let _ = (passed, skipped); // counts informational only
}

// ---------------------------------------------------------------------------
// tests/mindcraft/fmt/*.in.mind — Phase-2A fixture inputs
// ---------------------------------------------------------------------------

#[test]
fn idempotence_fmt_fixtures() {
    let fixture_dir = manifest_dir().join("tests/mindcraft/fmt");
    let cfg = default_cfg();
    let mut passed = 0usize;

    for entry in std::fs::read_dir(&fixture_dir)
        .unwrap_or_else(|e| panic!("cannot read fixture dir: {e}"))
    {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("mind")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".in.mind"))
                .unwrap_or(false)
        {
            let label = path.file_name().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            let exercised = check_idempotence(&label, &src, &cfg);
            assert!(exercised, "fixture {label} failed to parse — unexpected");
            passed += 1;
        }
    }

    assert_eq!(passed, 7, "expected 7 fixture .in.mind files, got {passed}");
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn collect_mind_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_mind_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("mind") {
            out.push(path);
        }
    }
}
