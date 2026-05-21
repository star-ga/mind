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
use crate::project::{find_project_root, load_manifest, MindcraftConfig, RuleSeverity};
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
    /// Optional machine-readable fix hint (not applied in Phase 5).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
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
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            run_fmt: true,
            run_lint: true,
            run_typecheck: true,
            reporter: ReporterKind::Human,
            paths: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReporterKind {
    Human,
    Json,
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------

/// Run `mindc check` and return the exit code (0 = clean, 1 = errors).
///
/// Diagnostics are emitted to stdout (human) or stdout (json array) as they
/// accumulate. The function itself does not call `process::exit`.
pub fn run_check(opts: &CheckOptions) -> i32 {
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
        if opts.reporter == ReporterKind::Json {
            println!("[]");
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
        // Sort by (line, col) for deterministic output within a file.
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
        ReporterKind::Json => {
            match serde_json::to_string_pretty(&all_diags) {
                Ok(json) => println!("{json}"),
                Err(e) => {
                    eprintln!("error[check]: JSON serialisation failed: {e}");
                    return 1;
                }
            }
        }
    }

    if has_errors { 1 } else { 0 }
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
        vec![std::env::current_dir()
            .map_err(|e| format!("cannot read current directory: {e}"))?]
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
                p.parent().map(|d| d.to_path_buf()).unwrap_or_else(|| p.clone())
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
        } else if child.extension().map_or(false, |ext| ext == "mind") {
            if should_include(&child, filter, repo_root) {
                out.push(child);
            }
        }
    }
    Ok(())
}

/// Return `true` if `path` should be included in the check run.
fn should_include(
    path: &Path,
    filter: Option<&GitignoreFilter>,
    repo_root: Option<&Path>,
) -> bool {
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
                });
            }
            return;
        }
    };

    let file_name = path.to_str();
    let type_diags =
        check_module_types_in_file(&module, source, file_name, &HashMap::new());

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

