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

//! `mindc fmt` subcommand dispatcher.
//!
//! Handles path resolution, config loading, and the three output modes:
//! - Default: format files in-place via atomic rename.
//! - `--check`: exit 1 if any file would change; no write.
//! - `--diff`: print unified diff to stdout; no write.
//! - `--stdin`: read from stdin, write to stdout.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::fmt::format_source;
use crate::project::{MindcraftFormatConfig, governing_project_root, load_manifest};

/// Entry point called from `mindc.rs`.
///
/// Returns an exit code: 0 = clean, 1 = drift/error, 2 = usage error.
///
/// The `fix` flag is an explicit alias for the default write mode: files are
/// formatted in-place and a summary line `Formatted N files, M unchanged` is
/// printed to stdout.
pub fn run_fmt(paths: &[String], check: bool, diff: bool, stdin: bool, fix: bool) -> i32 {
    // --stdin is mutually exclusive with positional paths.
    if stdin && !paths.is_empty() {
        eprintln!("error[fmt]: --stdin cannot be combined with positional paths");
        return 2;
    }

    let cfg = load_fmt_config();

    if stdin {
        return run_stdin_mode(&cfg);
    }

    let files = match resolve_paths(paths) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error[fmt]: {e}");
            return 1;
        }
    };

    if files.is_empty() {
        if fix {
            println!("Formatted 0 files, 0 unchanged.");
        }
        return 0;
    }

    let mut exit_code = 0_i32;
    let mut formatted_count = 0usize;
    let mut unchanged_count = 0usize;

    for path in &files {
        match process_file(path, check, diff, &cfg) {
            Ok(FileResult::Clean) => {
                unchanged_count += 1;
            }
            Ok(FileResult::Written) => {
                formatted_count += 1;
            }
            Ok(FileResult::Drifted) => {
                exit_code = 1;
            }
            Err(e) => {
                eprintln!("error[fmt]: {}: {e}", path.display());
                exit_code = 1;
            }
        }
    }

    if fix {
        println!(
            "Formatted {formatted_count} file{}, {unchanged_count} unchanged.",
            if formatted_count == 1 { "" } else { "s" },
        );
    }

    exit_code
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

fn load_fmt_config() -> MindcraftFormatConfig {
    // Walk upward from CWD for Mind.toml; fall back to defaults silently.
    if let Ok(root) = governing_project_root() {
        if let Ok(manifest) = load_manifest(&root) {
            return manifest.mindcraft.format;
        }
    }
    MindcraftFormatConfig::default()
}

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/// Resolve positional path arguments to a list of `.mind` files.
///
/// - Empty `paths` defaults to the current directory.
/// - Files are taken as-is.
/// - Directories are walked recursively; only `*.mind` entries are kept.
fn resolve_paths(paths: &[String]) -> Result<Vec<PathBuf>, String> {
    let roots: Vec<PathBuf> = if paths.is_empty() {
        vec![std::env::current_dir().map_err(|e| format!("cannot read current directory: {e}"))?]
    } else {
        paths.iter().map(PathBuf::from).collect()
    };

    let mut out = Vec::new();
    for root in roots {
        if root.is_file() {
            out.push(root);
        } else if root.is_dir() {
            collect_mind_files(&root, &mut out).map_err(|e| format!("{}: {e}", root.display()))?;
        } else {
            return Err(format!("'{}' does not exist", root.display()));
        }
    }
    Ok(out)
}

/// Recursively collect `*.mind` files under `dir`.
fn collect_mind_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let entries = fs::read_dir(dir)?;
    // Collect and sort for deterministic ordering.
    let mut children: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    children.sort();

    for child in children {
        if child.is_dir() {
            collect_mind_files(&child, out)?;
        } else if child.extension().is_some_and(|ext| ext == "mind") {
            out.push(child);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-file processing
// ---------------------------------------------------------------------------

enum FileResult {
    /// Source was already canonical — no write needed.
    Clean,
    /// Source differed from canonical — file was written successfully (default mode).
    Written,
    /// Source differed from canonical — `--check`/`--diff` mode, caller must exit 1.
    Drifted,
}

fn process_file(
    path: &Path,
    check: bool,
    diff: bool,
    cfg: &MindcraftFormatConfig,
) -> Result<FileResult, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("cannot read: {e}"))?;

    let formatted = format_source(&src, cfg).map_err(|e| format!("format error: {e}"))?;

    if formatted == src {
        return Ok(FileResult::Clean);
    }

    if check {
        eprintln!("would reformat: {}", path.display());
        return Ok(FileResult::Drifted);
    }

    if diff {
        let header = format!("--- {0}\n+++ {0}\n", path.display());
        print!("{}", header);
        print_unified_diff(&src, &formatted);
        return Ok(FileResult::Drifted);
    }

    // Default: write in-place via atomic rename.
    write_atomic(path, &formatted).map_err(|e| format!("cannot write: {e}"))?;

    Ok(FileResult::Written)
}

// ---------------------------------------------------------------------------
// --stdin mode
// ---------------------------------------------------------------------------

fn run_stdin_mode(cfg: &MindcraftFormatConfig) -> i32 {
    let mut src = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut src) {
        eprintln!("error[fmt]: failed to read stdin: {e}");
        return 1;
    }

    match format_source(&src, cfg) {
        Ok(formatted) => {
            if let Err(e) = io::stdout().write_all(formatted.as_bytes()) {
                eprintln!("error[fmt]: failed to write stdout: {e}");
                return 1;
            }
            0
        }
        Err(e) => {
            eprintln!("error[fmt]: {e}");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic write
// ---------------------------------------------------------------------------

fn write_atomic(path: &Path, content: &str) -> io::Result<()> {
    let tmp_path = path.with_extension("mind.tmp");
    fs::write(&tmp_path, content)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unified diff (hand-rolled, no external deps)
// ---------------------------------------------------------------------------

/// Print a unified diff between `old` and `new` to stdout.
///
/// Uses Myers LCS to compute the edit script. Context lines: 3.
pub fn print_unified_diff(old: &str, new: &str) {
    let diff = unified_diff_str(old, new, 3);
    print!("{diff}");
}

/// Produce a unified diff string between `old` and `new`.
///
/// `context` is the number of surrounding unchanged lines to include in each hunk.
pub fn unified_diff_str(old: &str, new: &str, context: usize) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let edit_script = compute_lcs_diff(&old_lines, &new_lines);
    build_unified_hunks(&old_lines, &new_lines, &edit_script, context)
}

/// Edit operations relative to the old sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edit {
    /// Keep: appears in both sequences.
    Keep,
    /// Delete: only in old.
    Delete,
    /// Insert: only in new.
    Insert,
}

/// Compute the diff between `old` and `new` via LCS (O(n*m) DP).
///
/// Builds the longest-common-subsequence table, then backtracks to produce
/// a flat `Edit` sequence aligned with `old` (deletes and keeps) and `new`
/// (inserts and keeps).
fn compute_lcs_diff(old: &[&str], new: &[&str]) -> Vec<Edit> {
    let n = old.len();
    let m = new.len();

    // dp[i][j] = length of LCS of old[..i] and new[..j].
    // Flat row-major layout: index as dp[i * (m+1) + j].
    let mut dp = vec![0u32; (n + 1) * (m + 1)];
    for i in 1..=n {
        for j in 1..=m {
            dp[i * (m + 1) + j] = if old[i - 1] == new[j - 1] {
                dp[(i - 1) * (m + 1) + (j - 1)] + 1
            } else {
                dp[(i - 1) * (m + 1) + j].max(dp[i * (m + 1) + (j - 1)])
            };
        }
    }

    // Backtrack to build the edit sequence (reversed, then flipped).
    let mut edits: Vec<Edit> = Vec::with_capacity(n + m);
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
            edits.push(Edit::Keep);
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i * (m + 1) + (j - 1)] >= dp[(i - 1) * (m + 1) + j]) {
            edits.push(Edit::Insert);
            j -= 1;
        } else {
            edits.push(Edit::Delete);
            i -= 1;
        }
    }
    edits.reverse();
    edits
}

/// Build the unified-diff string from an edit script.
fn build_unified_hunks(old: &[&str], new: &[&str], edits: &[Edit], context: usize) -> String {
    // Build a flat list of per-line items: (old_line, new_line, edit).
    // old_line / new_line are the 1-based line numbers (0 = not present).
    struct Line {
        old_no: usize,
        new_no: usize,
        edit: Edit,
    }

    let mut lines: Vec<Line> = Vec::with_capacity(old.len() + new.len());
    let mut oi = 0usize;
    let mut ni = 0usize;

    for &edit in edits {
        match edit {
            Edit::Keep => {
                lines.push(Line {
                    old_no: oi + 1,
                    new_no: ni + 1,
                    edit,
                });
                oi += 1;
                ni += 1;
            }
            Edit::Delete => {
                lines.push(Line {
                    old_no: oi + 1,
                    new_no: 0,
                    edit,
                });
                oi += 1;
            }
            Edit::Insert => {
                lines.push(Line {
                    old_no: 0,
                    new_no: ni + 1,
                    edit,
                });
                ni += 1;
            }
        }
    }

    // Group into hunks: runs of changed lines padded with `context` lines.
    let mut out = String::new();
    let n = lines.len();
    let changed: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.edit != Edit::Keep)
        .map(|(k, _)| k)
        .collect();

    for (hunk_start, hunk_end) in hunk_ranges(&changed, n, context) {
        // Compute hunk header @@ -old_start,old_count +new_start,new_count @@
        let old_start = lines[hunk_start].old_no.max(1);
        let new_start = lines[hunk_start].new_no.max(1);
        let old_count = lines[hunk_start..hunk_end]
            .iter()
            .filter(|l| l.edit != Edit::Insert)
            .count();
        let new_count = lines[hunk_start..hunk_end]
            .iter()
            .filter(|l| l.edit != Edit::Delete)
            .count();

        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start, old_count, new_start, new_count
        ));

        for line in &lines[hunk_start..hunk_end] {
            let text = match line.edit {
                Edit::Keep => {
                    let idx = line.old_no.saturating_sub(1);
                    old.get(idx).copied().unwrap_or("")
                }
                Edit::Delete => {
                    let idx = line.old_no.saturating_sub(1);
                    old.get(idx).copied().unwrap_or("")
                }
                Edit::Insert => {
                    let idx = line.new_no.saturating_sub(1);
                    new.get(idx).copied().unwrap_or("")
                }
            };
            let prefix = match line.edit {
                Edit::Keep => ' ',
                Edit::Delete => '-',
                Edit::Insert => '+',
            };
            out.push(prefix);
            out.push_str(text);
            out.push('\n');
        }
    }

    out
}

/// Split the changed-line indices into hunk ranges `[start, end)` over `n` lines.
///
/// `changed` is strictly increasing with every index below `n`.  Changes
/// separated by at most `2 * context` unchanged lines share one hunk, because
/// their context windows touch or overlap.  Each hunk is padded with `context`
/// lines on both sides and clamped to `[0, n)`, so `start < end <= n` always
/// holds and consecutive hunks never overlap.
fn hunk_ranges(changed: &[usize], n: usize, context: usize) -> Vec<(usize, usize)> {
    let max_gap = context.saturating_mul(2);
    let mut ranges = Vec::new();
    let mut k = 0;
    while k < changed.len() {
        let first = changed[k];
        let mut last = first;
        k += 1;
        while k < changed.len() && changed[k].saturating_sub(last + 1) <= max_gap {
            last = changed[k];
            k += 1;
        }
        ranges.push((
            first.saturating_sub(context),
            last.saturating_add(1).saturating_add(context).min(n),
        ));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_source_is_not_written() {
        let dir = tempfile::tempdir().expect("temporary fixture directory");
        let path = dir.path().join("broken.mind");
        let source = b"fn broken() { let content = format!(\"{}\", 1); }\n";
        fs::write(&path, source).expect("write fixture");

        let result = process_file(&path, false, false, &MindcraftFormatConfig::default());

        assert!(result.is_err(), "unsupported syntax must fail closed");
        assert_eq!(fs::read(&path).expect("read fixture"), source);
    }

    #[test]
    fn diff_identical_is_empty() {
        assert_eq!(unified_diff_str("a\nb\nc\n", "a\nb\nc\n", 3), "");
    }

    #[test]
    fn diff_single_insert() {
        let result = unified_diff_str("a\nc\n", "a\nb\nc\n", 3);
        assert!(result.contains("+b"), "expected +b in diff:\n{result}");
    }

    #[test]
    fn diff_single_delete() {
        let result = unified_diff_str("a\nb\nc\n", "a\nc\n", 3);
        assert!(result.contains("-b"), "expected -b in diff:\n{result}");
    }

    #[test]
    fn diff_replace() {
        let result = unified_diff_str("hello\n", "world\n", 3);
        assert!(result.contains("-hello"), "expected -hello:\n{result}");
        assert!(result.contains("+world"), "expected +world:\n{result}");
    }

    // -- hunk grouping -----------------------------------------------------
    //
    // Expected outputs below are what `diff -U3` prints for the same two files.

    /// `n` lines `l1..ln`, with each 1-based line in `replaced` spelled `L<k>`.
    fn numbered(n: usize, replaced: &[usize]) -> String {
        (1..=n)
            .map(|k| {
                if replaced.contains(&k) {
                    format!("L{k}\n")
                } else {
                    format!("l{k}\n")
                }
            })
            .collect()
    }

    fn diff_of(n: usize, replaced: &[usize]) -> String {
        unified_diff_str(&numbered(n, &[]), &numbered(n, replaced), 3)
    }

    #[test]
    fn diff_changes_seven_lines_apart_make_two_hunks() {
        // Seven unchanged lines between the changes: the panic input.
        let want = "\
@@ -1,5 +1,5 @@
 l1
-l2
+L2
 l3
 l4
 l5
@@ -7,7 +7,7 @@
 l7
 l8
 l9
-l10
+L10
 l11
 l12
 l13
";
        assert_eq!(diff_of(20, &[2, 10]), want);
    }

    #[test]
    fn diff_changes_eight_lines_apart_make_two_hunks() {
        let want = "\
@@ -1,5 +1,5 @@
 l1
-l2
+L2
 l3
 l4
 l5
@@ -8,7 +8,7 @@
 l8
 l9
 l10
-l11
+L11
 l12
 l13
 l14
";
        assert_eq!(diff_of(20, &[2, 11]), want);
    }

    #[test]
    fn diff_changes_six_lines_apart_share_one_hunk() {
        // Six unchanged lines is exactly 2 * context: the windows touch, so the
        // changes merge; one more line (seven, above) splits them.
        let want = "\
@@ -1,12 +1,12 @@
 l1
-l2
+L2
 l3
 l4
 l5
 l6
 l7
 l8
-l9
+L9
 l10
 l11
 l12
";
        assert_eq!(diff_of(20, &[2, 9]), want);
    }

    #[test]
    fn diff_overlapping_changes_chain_into_one_hunk_then_split() {
        let chained = "\
@@ -1,17 +1,17 @@
 l1
-l2
+L2
 l3
 l4
 l5
 l6
 l7
-l8
+L8
 l9
 l10
 l11
 l12
 l13
-l14
+L14
 l15
 l16
 l17
";
        assert_eq!(diff_of(20, &[2, 8, 14]), chained);

        let tail = "\
@@ -19,6 +19,6 @@
 l19
 l20
 l21
-l22
+L22
 l23
 l24
";
        assert_eq!(diff_of(24, &[2, 8, 14, 22]), format!("{chained}{tail}"));
    }

    #[test]
    fn diff_first_and_last_line_changed() {
        let want = "\
@@ -1,4 +1,4 @@
-l1
+L1
 l2
 l3
 l4
@@ -9,4 +9,4 @@
 l9
 l10
 l11
-l12
+L12
";
        assert_eq!(diff_of(12, &[1, 12]), want);
    }

    #[test]
    fn diff_change_near_end_clamps_hunk_end() {
        // Only one line of trailing context exists; the hunk must stop at the end.
        let want = "\
@@ -6,5 +6,5 @@
 l6
 l7
 l8
-l9
+L9
 l10
";
        assert_eq!(diff_of(10, &[9]), want);
    }

    #[test]
    fn diff_clustered_changes_keep_full_trailing_context() {
        let want = "\
@@ -1,6 +1,6 @@
-l1
+L1
 l2
-l3
+L3
 l4
 l5
 l6
";
        assert_eq!(diff_of(12, &[1, 3]), want);
    }

    #[test]
    fn diff_unchanged_inputs_emit_nothing() {
        assert_eq!(diff_of(25, &[]), "");
        assert_eq!(unified_diff_str("", "", 3), "");
        assert_eq!(unified_diff_str("a\nb", "a\nb", 3), "");
    }

    #[test]
    fn hunk_ranges_stay_in_bounds_for_every_change_pattern() {
        for context in 0..=4usize {
            for n in 0..=12usize {
                for mask in 0u32..(1 << n) {
                    let changed: Vec<usize> = (0..n).filter(|k| (mask >> *k) & 1 == 1).collect();
                    let ranges = hunk_ranges(&changed, n, context);
                    let tag = format!("n={n} context={context} changed={changed:?} -> {ranges:?}");

                    assert_eq!(ranges.is_empty(), changed.is_empty(), "{tag}");
                    let mut covered = Vec::new();
                    for &(start, end) in &ranges {
                        assert!(start < end && end <= n, "out of bounds: {tag}");
                        let inside: Vec<usize> = changed
                            .iter()
                            .copied()
                            .filter(|k| (start..end).contains(k))
                            .collect();
                        let (first, last) = (inside[0], inside[inside.len() - 1]);
                        assert_eq!(start, first.saturating_sub(context), "start: {tag}");
                        assert_eq!(end, (last + 1 + context).min(n), "end: {tag}");
                        covered.extend(inside);
                    }
                    assert_eq!(covered, changed, "every change in exactly one hunk: {tag}");
                    for pair in ranges.windows(2) {
                        assert!(pair[0].1 < pair[1].0, "hunks touch or overlap: {tag}");
                    }
                }
            }
        }
    }

    /// Apply a unified diff (no file header) to `old`, checking every context
    /// and deleted line against it and that hunks are ordered and disjoint.
    fn apply_unified(old: &str, diff: &str) -> String {
        let old_lines: Vec<&str> = old.lines().collect();
        let mut out: Vec<&str> = Vec::new();
        let mut pos = 0usize;
        for line in diff.lines() {
            if let Some(header) = line.strip_prefix("@@ -") {
                let start: usize = header
                    .split([',', ' '])
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("hunk header start");
                assert!(start > pos, "hunks overlap or are out of order:\n{diff}");
                out.extend(&old_lines[pos..start - 1]);
                pos = start - 1;
            } else if let Some(text) = line.strip_prefix('+') {
                out.push(text);
            } else {
                let text = &line[1..];
                assert_eq!(old_lines[pos], text, "context/delete mismatch:\n{diff}");
                if line.starts_with(' ') {
                    out.push(text);
                }
                pos += 1;
            }
        }
        out.extend(&old_lines[pos..]);
        out.iter().map(|l| format!("{l}\n")).collect()
    }

    #[test]
    fn diff_applies_back_to_the_new_text_for_scattered_edits() {
        // Deterministic pseudo-random edits at several densities, every context.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % 100
        };
        for density in [3u64, 8, 20, 45, 80] {
            for round in 0..40 {
                let old: Vec<String> = (0..40).map(|k| format!("o{k}")).collect();
                let mut new: Vec<String> = Vec::new();
                for (k, line) in old.iter().enumerate() {
                    let r = next();
                    if r >= density {
                        new.push(line.clone());
                    } else if r % 3 == 0 {
                        new.push(format!("n{round}_{k}"));
                    } else if r % 3 == 1 {
                        new.push(line.clone());
                        new.push(format!("i{round}_{k}"));
                    } // else: line deleted
                }
                let (old, new) = (old.join("\n") + "\n", new.join("\n") + "\n");
                for context in 1..=3 {
                    let diff = unified_diff_str(&old, &new, context);
                    assert_eq!(apply_unified(&old, &diff), new, "context={context}\n{diff}");
                }
            }
        }
    }
}
