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

//! RFC 0007 (Mindcraft) Phase 5 — `mindc check` project-walker driver.
//!
//! Combines format-check + lint + type-check into a single fast pass over a
//! set of `.mind` files. Emits aggregated diagnostics with `file:line:col`
//! positions.
//!
//! Public entry point: [`run_check`].

pub mod gitignore;
pub mod reporter;

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::fmt::format_source;
use crate::lint::rules::register_defaults;
use crate::lint::{check_source as lint_source, rule::RuleRegistry};
use crate::parser::parse_with_trivia;
use crate::project::{MindcraftConfig, RuleSeverity, find_project_root, load_manifest};
#[cfg(not(feature = "cross-module-imports"))]
use crate::type_checker::check_module_types_in_file;

use gitignore::GitignoreFilter;

// ---------------------------------------------------------------------------
// Public diagnostic schema
// ---------------------------------------------------------------------------

/// The source pass that produced a [`CheckDiagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckPhase {
    Fmt,
    Lint,
    TypeCheck,
}

/// Severity of a [`CheckDiagnostic`].
///
/// Mirrors [`RuleSeverity`] but without the `Off` variant (off rules produce
/// no diagnostics) and adds parity with the compiler's `Severity::Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckSeverity {
    Info,
    Warn,
    Error,
}

impl CheckSeverity {
    fn as_str(self) -> &'static str {
        match self {
            CheckSeverity::Info => "info",
            CheckSeverity::Warn => "warn",
            CheckSeverity::Error => "error",
        }
    }

    fn is_error(self) -> bool {
        self == CheckSeverity::Error
    }
}

impl From<RuleSeverity> for CheckSeverity {
    fn from(s: RuleSeverity) -> Self {
        match s {
            RuleSeverity::Off | RuleSeverity::Info => CheckSeverity::Info,
            RuleSeverity::Warn => CheckSeverity::Warn,
            RuleSeverity::Error => CheckSeverity::Error,
        }
    }
}

/// A single diagnostic produced by the `mindc check` pass.
///
/// The schema is the stable contract for `--reporter json` (RFC 0007 §6).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CheckDiagnostic {
    /// Source file.
    pub file: PathBuf,
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number.
    pub col: usize,
    /// Severity.
    pub severity: CheckSeverity,
    /// Human-readable message.
    pub message: String,
    /// Stable rule / phase identifier, e.g. `"lint::trailing_whitespace"` or
    /// `"fmt::drift"` or `"type_check::E001"`.
    pub rule_id: String,
    /// Which toolchain pass produced this diagnostic.
    pub phase: CheckPhase,
    /// Optional human-readable fix hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
    /// Byte-range replacement sourced from lint auto-fix (not serialised).
    ///
    /// Present only on diagnostics that have a machine-applicable fix.
    /// Used internally by `--fix`; not emitted in JSON/LSP reporters.
    #[serde(skip)]
    pub auto_fix: Option<crate::lint::rule::Fix>,
}

impl CheckDiagnostic {
    /// Return the human-reporter line:
    /// `<path>:<line>:<col>: <severity>: <message> [<rule_id>]`
    pub fn human_line(&self) -> String {
        format!(
            "{}:{}:{}: {}: {} [{}]",
            self.file.display(),
            self.line,
            self.col,
            self.severity.as_str(),
            self.message,
            self.rule_id,
        )
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Which passes to run. All default to `true`.
#[derive(Debug, Clone)]
pub struct CheckOptions {
    pub run_fmt: bool,
    pub run_lint: bool,
    pub run_typecheck: bool,
    pub reporter: ReporterKind,
    pub paths: Vec<String>,
    /// When `true`, apply machine-applicable fixes and rewrite files.
    pub fix: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            run_fmt: true,
            run_lint: true,
            run_typecheck: true,
            reporter: ReporterKind::Human,
            paths: Vec::new(),
            fix: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReporterKind {
    Human,
    Json,
    /// Emit LSP-compatible Diagnostic JSON (RFC 0007 Phase 6).
    ///
    /// Each diagnostic is serialised as an LSP `Diagnostic` object:
    /// `{ uri, range: { start: { line, character }, end: { line, character } },
    ///   severity, message, source, code }`.
    Lsp,
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------

/// Run `mindc check` and return the exit code (0 = clean, 1 = errors).
///
/// When `opts.fix` is `true`, machine-applicable fixes are applied iteratively
/// (up to 5 rounds) and the summary is printed to stdout. The function itself
/// does not call `process::exit`.
pub fn run_check(opts: &CheckOptions) -> i32 {
    if opts.fix {
        return run_check_fix(opts);
    }

    let config = load_check_config();
    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    let files = match resolve_paths(&opts.paths, &config) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error[check]: {e}");
            return 1;
        }
    };

    if files.is_empty() {
        match opts.reporter {
            ReporterKind::Json | ReporterKind::Lsp => println!("[]"),
            ReporterKind::Human => {}
        }
        return 0;
    }

    let mut all_diags: Vec<CheckDiagnostic> = Vec::new();

    for path in &files {
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error[check]: {}: {e}", path.display());
                continue;
            }
        };

        let mut file_diags = check_file(path, &source, &config, &registry, opts);
        file_diags.sort_by_key(|d| (d.line, d.col));
        all_diags.extend(file_diags);
    }

    // Sort globally by (file, line, col) for deterministic multi-file output.
    all_diags.sort_by(|a, b| {
        a.file
            .cmp(&b.file)
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.col.cmp(&b.col))
    });

    let has_errors = all_diags.iter().any(|d| d.severity.is_error());

    match opts.reporter {
        ReporterKind::Human => {
            for d in &all_diags {
                println!("{}", d.human_line());
            }
        }
        ReporterKind::Json => match serde_json::to_string_pretty(&all_diags) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("error[check]: JSON serialisation failed: {e}");
                return 1;
            }
        },
        ReporterKind::Lsp => match emit_lsp_diagnostics(&all_diags) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("error[check]: LSP serialisation failed: {e}");
                return 1;
            }
        },
    }

    if has_errors { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// --fix driver
// ---------------------------------------------------------------------------

/// Maximum number of fix-recheck iterations before giving up.
const MAX_FIX_ITERATIONS: usize = 5;

/// Run `mindc check --fix`: apply machine-applicable fixes iteratively.
///
/// For every `fmt::drift` diagnostic, write the formatted version.
/// For every lint diagnostic with an `auto_fix`, apply the byte-range edit.
/// Re-check until no fixable diagnostics remain or `MAX_FIX_ITERATIONS` is
/// reached.  Prints a summary line and returns the exit code.
fn run_check_fix(opts: &CheckOptions) -> i32 {
    let config = load_check_config();
    let mut registry = RuleRegistry::new();
    register_defaults(&mut registry);

    let files = match resolve_paths(&opts.paths, &config) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error[check]: {e}");
            return 1;
        }
    };

    if files.is_empty() {
        println!("Fixed 0 files, 0 unfixable diagnostics remaining.");
        return 0;
    }

    let mut total_fixed_files = 0usize;
    let mut converged = true;

    for path in &files {
        let fixed = fix_file_iteratively(path, &config, &registry, opts);
        match fixed {
            Ok(FixResult::Fixed) => total_fixed_files += 1,
            Ok(FixResult::Clean) => {}
            Ok(FixResult::DidNotConverge) => {
                converged = false;
                eprintln!(
                    "warning[check]: --fix did not converge after {MAX_FIX_ITERATIONS} \
                     iterations for {}",
                    path.display()
                );
            }
            Err(e) => {
                eprintln!("error[check]: {}: {e}", path.display());
            }
        }
    }

    // Final pass: collect remaining unfixable diagnostics.
    let mut remaining = 0usize;
    for path in &files {
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let diags = check_file(path, &source, &config, &registry, opts);
        remaining += diags
            .iter()
            .filter(|d| d.auto_fix.is_none() && d.phase != CheckPhase::Fmt)
            .count();
    }

    println!(
        "Fixed {total_fixed_files} file{}, {remaining} unfixable diagnostic{} remaining.",
        if total_fixed_files == 1 { "" } else { "s" },
        if remaining == 1 { "" } else { "s" },
    );

    if !converged {
        eprintln!("warning[check]: some files did not converge in {MAX_FIX_ITERATIONS} iterations");
    }

    if remaining > 0 { 1 } else { 0 }
}

#[derive(Debug)]
enum FixResult {
    Clean,
    Fixed,
    DidNotConverge,
}

/// Apply fixes to a single file, re-checking up to `MAX_FIX_ITERATIONS` times.
fn fix_file_iteratively(
    path: &Path,
    config: &crate::project::MindcraftConfig,
    registry: &RuleRegistry,
    opts: &CheckOptions,
) -> Result<FixResult, String> {
    let mut any_written = false;

    for _iter in 0..MAX_FIX_ITERATIONS {
        let source = fs::read_to_string(path).map_err(|e| format!("cannot read: {e}"))?;

        let diags = check_file(path, &source, config, registry, opts);

        // Separate fixable from unfixable.
        let fixable: Vec<&CheckDiagnostic> = diags
            .iter()
            .filter(|d| d.auto_fix.is_some() || d.phase == CheckPhase::Fmt)
            .collect();

        if fixable.is_empty() {
            return Ok(if any_written {
                FixResult::Fixed
            } else {
                FixResult::Clean
            });
        }

        // Prioritise fmt::drift: if the file is not formatted, write the
        // formatted version and re-check in the next iteration.  Do NOT also
        // apply lint byte-range edits in the same pass — those offsets are for
        // the original source and would be invalid after reformatting.
        let has_fmt_drift = fixable.iter().any(|d| d.phase == CheckPhase::Fmt);
        if has_fmt_drift {
            let formatted = crate::fmt::format_source(&source, &config.format)
                .map_err(|e| format!("format error: {e}"))?;
            if formatted != source {
                write_atomic(path, &formatted).map_err(|e| format!("cannot write: {e}"))?;
                any_written = true;
                continue; // re-check from scratch with the formatted source
            }
            // If fmt produced the same source (shouldn't happen if fmt::drift
            // was reported), fall through to lint fixes below.
        }

        // No fmt drift (or fmt was already applied above).  Collect and apply
        // lint auto-fixes (byte-range edits) against the current source.
        let mut edits: Vec<crate::lint::rule::Fix> =
            fixable.iter().filter_map(|d| d.auto_fix.clone()).collect();

        if edits.is_empty() {
            return Ok(if any_written {
                FixResult::Fixed
            } else {
                FixResult::Clean
            });
        }

        // Apply edits in ascending byte-offset order (apply_edits handles this).
        edits.sort_by_key(|e| e.range.start);
        // Remove overlapping edits: keep the first (lowest offset).
        edits.dedup_by(|a, b| {
            // `a` comes after `b` in sorted order.
            // Drop `a` if it overlaps with `b`.
            a.range.start < b.range.end
        });

        let fixed_source = apply_edits(&source, &edits)?;

        if fixed_source == source {
            return Ok(if any_written {
                FixResult::Fixed
            } else {
                FixResult::Clean
            });
        }

        write_atomic(path, &fixed_source).map_err(|e| format!("cannot write: {e}"))?;
        any_written = true;
    }

    Ok(FixResult::DidNotConverge)
}

/// Apply a sorted (reverse-offset) list of non-overlapping byte-range edits.
fn apply_edits(source: &str, edits: &[crate::lint::rule::Fix]) -> Result<String, String> {
    let src_bytes = source.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(src_bytes.len());
    let mut cursor = 0usize;

    // Edits must be sorted in reverse order (largest start first) so we
    // process them back-to-front and rebuild the string front-to-back by
    // reversing the list here.
    let mut ordered: Vec<_> = edits.iter().collect();
    ordered.sort_by_key(|e| e.range.start);

    for edit in &ordered {
        if edit.range.start > src_bytes.len() || edit.range.end > src_bytes.len() {
            return Err(format!(
                "fix range {}..{} out of bounds (source len {})",
                edit.range.start,
                edit.range.end,
                src_bytes.len()
            ));
        }
        if edit.range.start < cursor {
            // Overlapping edit — skip.
            continue;
        }
        out.extend_from_slice(&src_bytes[cursor..edit.range.start]);
        out.extend_from_slice(edit.replacement.as_bytes());
        cursor = edit.range.end;
    }
    out.extend_from_slice(&src_bytes[cursor..]);

    String::from_utf8(out).map_err(|e| format!("fix produced invalid UTF-8: {e}"))
}

/// Atomic file write via temp-file + rename.
fn write_atomic(path: &Path, content: &str) -> io::Result<()> {
    let tmp = path.with_extension("mind.fix.tmp");
    fs::write(&tmp, content)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// LSP reporter
// ---------------------------------------------------------------------------

/// LSP severity codes (as per the LSP specification).
///
/// 1=Error, 2=Warning, 3=Information, 4=Hint.
fn lsp_severity(s: CheckSeverity) -> u8 {
    match s {
        CheckSeverity::Error => 1,
        CheckSeverity::Warn => 2,
        CheckSeverity::Info => 3,
    }
}

/// Emit diagnostics as a JSON array of LSP `Diagnostic` objects.
///
/// Schema (per LSP specification §3.17):
/// ```json
/// [
///   {
///     "uri": "file:///absolute/path/to/file.mind",
///     "range": {
///       "start": { "line": 0, "character": 0 },
///       "end":   { "line": 0, "character": 5 }
///     },
///     "severity": 1,
///     "message": "...",
///     "source": "mindc",
///     "code": "lint::trailing_whitespace"
///   }
/// ]
/// ```
///
/// LSP line/character values are 0-based; `CheckDiagnostic` uses 1-based.
fn emit_lsp_diagnostics(diags: &[CheckDiagnostic]) -> Result<String, serde_json::Error> {
    use serde_json::{Value, json};

    let items: Vec<Value> = diags
        .iter()
        .map(|d| {
            let line0 = d.line.saturating_sub(1);
            let char0 = d.col.saturating_sub(1);
            // For LSP `end`, we use the same line/col as start since we only
            // have one position. Downstream tooling can use the `code` to
            // look up the full span if needed.
            let uri = format!("file://{}", d.file.display());
            json!({
                "uri": uri,
                "range": {
                    "start": { "line": line0, "character": char0 },
                    "end":   { "line": line0, "character": char0 }
                },
                "severity": lsp_severity(d.severity),
                "message": d.message,
                "source": "mindc",
                "code": d.rule_id,
            })
        })
        .collect();

    serde_json::to_string_pretty(&items)
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

fn load_check_config() -> MindcraftConfig {
    if let Ok(root) = find_project_root() {
        if let Ok(manifest) = load_manifest(&root) {
            return manifest.mindcraft;
        }
    }
    MindcraftConfig::default()
}

// ---------------------------------------------------------------------------
// Path resolution with VCS filtering
// ---------------------------------------------------------------------------

fn resolve_paths(paths: &[String], config: &MindcraftConfig) -> Result<Vec<PathBuf>, String> {
    let roots: Vec<PathBuf> = if paths.is_empty() {
        vec![std::env::current_dir().map_err(|e| format!("cannot read current directory: {e}"))?]
    } else {
        paths.iter().map(PathBuf::from).collect()
    };

    // Determine repo root: prefer a git root found relative to the first
    // input path/directory, falling back to CWD-relative discovery.
    // This allows tests that create synthetic git repos in tmpdirs to work
    // correctly even when the test binary's CWD is a different repo.
    let repo_root = {
        let candidate_start = roots.first().map(|p| {
            if p.is_file() {
                p.parent()
                    .map(|d| d.to_path_buf())
                    .unwrap_or_else(|| p.clone())
            } else {
                p.clone()
            }
        });
        candidate_start
            .and_then(|start| find_git_root_from(&start))
            .or_else(find_git_root)
    };

    // Build gitignore filter if VCS integration is enabled.
    let filter = if config.vcs.use_ignore_file {
        Some(build_gitignore_filter_from(repo_root.as_deref()))
    } else {
        None
    };

    let mut out: Vec<PathBuf> = Vec::new();
    for root in roots {
        if root.is_file() {
            if should_include(&root, filter.as_ref(), repo_root.as_deref()) {
                out.push(root);
            }
        } else if root.is_dir() {
            collect_mind_files(&root, &mut out, filter.as_ref(), repo_root.as_deref())
                .map_err(|e| format!("{}: {e}", root.display()))?;
        } else {
            return Err(format!("'{}' does not exist", root.display()));
        }
    }

    out.sort(); // deterministic ordering
    Ok(out)
}

/// Walk `dir` recursively, collecting `*.mind` files that pass the filter.
fn collect_mind_files(
    dir: &Path,
    out: &mut Vec<PathBuf>,
    filter: Option<&GitignoreFilter>,
    repo_root: Option<&Path>,
) -> io::Result<()> {
    let mut children: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    children.sort();

    for child in children {
        if child.is_dir() {
            collect_mind_files(&child, out, filter, repo_root)?;
        } else if child.extension().is_some_and(|ext| ext == "mind")
            && should_include(&child, filter, repo_root)
        {
            out.push(child);
        }
    }
    Ok(())
}

/// Return `true` if `path` should be included in the check run.
fn should_include(path: &Path, filter: Option<&GitignoreFilter>, repo_root: Option<&Path>) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    let rel = match repo_root {
        Some(root) => path.strip_prefix(root).unwrap_or(path),
        None => path,
    };
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    !filter.is_ignored(&rel_str)
}

/// Build a [`GitignoreFilter`] from the repo's `.gitignore` and
/// `.git/info/exclude` (if present), rooted at `root`.
fn build_gitignore_filter_from(root: Option<&Path>) -> GitignoreFilter {
    let mut f = GitignoreFilter::default();
    if let Some(r) = root {
        f.load_file(&r.join(".gitignore"));
        f.load_file(&r.join(".git/info/exclude"));
    } else {
        // Fallback: look in current directory.
        f.load_file(Path::new(".gitignore"));
    }
    f
}

/// Walk upward from `start` to find the git repository root (dir that contains
/// `.git`). Returns `None` if not inside a git repo.
fn find_git_root_from(start: &Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    // Canonicalize so relative paths work.
    if let Ok(abs) = cur.canonicalize() {
        cur = abs;
    }
    loop {
        if cur.join(".git").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

/// Walk upward from CWD to find the git repository root (dir that contains
/// `.git`). Returns `None` if not inside a git repo.
fn find_git_root() -> Option<PathBuf> {
    let cur = std::env::current_dir().ok()?;
    find_git_root_from(&cur)
}

// ---------------------------------------------------------------------------
// Per-file check
// ---------------------------------------------------------------------------

fn check_file(
    path: &Path,
    source: &str,
    config: &MindcraftConfig,
    registry: &RuleRegistry,
    opts: &CheckOptions,
) -> Vec<CheckDiagnostic> {
    let mut out: Vec<CheckDiagnostic> = Vec::new();

    // ── Format-check pass ───────────────────────────────────────────────────
    if opts.run_fmt {
        check_fmt(path, source, config, &mut out);
    }

    // ── Lint pass ───────────────────────────────────────────────────────────
    if opts.run_lint {
        check_lint(path, source, config, registry, &mut out);
    }

    // ── Type-check pass ─────────────────────────────────────────────────────
    if opts.run_typecheck {
        check_types(path, source, &mut out);
    }

    out
}

/// Format-check: parse + format; if output differs from source, emit a
/// `fmt::drift` diagnostic pointing at the first differing line.
fn check_fmt(path: &Path, source: &str, config: &MindcraftConfig, out: &mut Vec<CheckDiagnostic>) {
    let formatted = match format_source(source, &config.format) {
        Ok(f) => f,
        Err(_) => return, // parse error — let type-check surface it
    };

    if formatted == source {
        return;
    }

    // Find the first differing line for a precise location.
    let (line, col) = first_diff_position(source, &formatted);

    out.push(CheckDiagnostic {
        file: path.to_path_buf(),
        line,
        col,
        severity: CheckSeverity::Error,
        message: "file is not formatted; run `mindc fmt` to fix".to_string(),
        rule_id: "fmt::drift".to_string(),
        phase: CheckPhase::Fmt,
        help: Some("run `mindc fmt <file>` to auto-format".to_string()),
        // fmt::drift is fixed by rewriting the whole file; no byte-range fix.
        auto_fix: None,
    });
}

/// Lint pass: run all registered rules via `check_source`.
fn check_lint(
    path: &Path,
    source: &str,
    config: &MindcraftConfig,
    registry: &RuleRegistry,
    out: &mut Vec<CheckDiagnostic>,
) {
    let lint_diags = lint_source(source, path, config, registry);
    for d in lint_diags {
        let (line, col) = offset_to_line_col(source, d.span.start);
        out.push(CheckDiagnostic {
            file: path.to_path_buf(),
            line,
            col,
            severity: CheckSeverity::from(d.severity),
            message: d.message,
            rule_id: d.rule_id,
            phase: CheckPhase::Lint,
            help: d.help,
            auto_fix: d.auto_fix,
        });
    }
}

/// Type-check pass: parse + type-check; emit any type errors.
fn check_types(path: &Path, source: &str, out: &mut Vec<CheckDiagnostic>) {
    let parse_result = parse_with_trivia(source);
    let (module, _trivia) = match parse_result {
        Ok(pair) => pair,
        Err(errors) => {
            // Surface parse errors as type_check phase diagnostics.
            for e in errors {
                let (line, col) = offset_to_line_col(source, e.offset);
                out.push(CheckDiagnostic {
                    file: path.to_path_buf(),
                    line,
                    col,
                    severity: CheckSeverity::Error,
                    message: e.message.clone(),
                    rule_id: "type_check::parse_error".to_string(),
                    phase: CheckPhase::TypeCheck,
                    help: None,
                    auto_fix: None,
                });
            }
            return;
        }
    };

    let file_name = path.to_str();
    // RFC 0005 Phase C — seed the cross-module resolver with the bundled std
    // surface so single-file `mindc check` resolves `use std.vec` /
    // `use std.string` / … imports the same way the project build path does
    // (task #23). Without this seeding the resolver sees an empty table and a
    // call to an imported `vec_new` would mis-fire as an undefined call (E2003).
    // Default / no-cross-module build keeps the byte-identical empty-table path.
    #[cfg(feature = "cross-module-imports")]
    let type_diags = {
        use crate::project::module_table::build_module_table;
        use crate::project::stdlib::parsed_stdlib_modules;
        use crate::type_checker::check_module_types_with_modules;
        let std_mods = parsed_stdlib_modules();
        let refs: Vec<(String, &crate::ast::Module)> =
            std_mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);
        check_module_types_with_modules(&module, source, file_name, &HashMap::new(), &table)
    };
    #[cfg(not(feature = "cross-module-imports"))]
    let type_diags = check_module_types_in_file(&module, source, file_name, &HashMap::new());

    for d in type_diags {
        let (line, col) = match &d.span {
            Some(span) => (span.line, span.column),
            None => (1, 1),
        };
        let severity = match d.severity {
            crate::diagnostics::Severity::Error => CheckSeverity::Error,
            crate::diagnostics::Severity::Warning => CheckSeverity::Warn,
        };
        out.push(CheckDiagnostic {
            file: path.to_path_buf(),
            line,
            col,
            severity,
            message: d.message.clone(),
            rule_id: format!("type_check::{}", d.code),
            phase: CheckPhase::TypeCheck,
            help: d.help,
            auto_fix: None,
        });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a byte offset into 1-based (line, col).
pub fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    let mut count = 0usize;
    for ch in source.chars() {
        if count >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
        count += ch.len_utf8();
    }
    (line, col)
}

/// Find the (1-based line, 1-based col) of the first byte where `a` and `b`
/// differ. Falls back to (1, 1) if identical (should not occur when called
/// after a drift check).
fn first_diff_position(a: &str, b: &str) -> (usize, usize) {
    let diff_offset = a
        .bytes()
        .zip(b.bytes())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()));
    offset_to_line_col(a, diff_offset)
}
