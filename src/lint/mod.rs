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

//! RFC 0007 (Mindcraft) Phase 3 — lint rule infrastructure.
//!
//! This module provides the core surfaces that Phase 4's named rules plug into:
//!
//! - [`Diagnostic`] + [`SourceSpan`] — the shared diagnostic schema.
//! - [`LintRule`] — the trait every rule implements.
//! - [`LintCtx`] — immutable context threaded into every rule's `check` call.
//! - [`RuleRegistry`] — registration, severity resolution, unknown-id detection.
//! - [`run_lint`] — top-level driver that runs all enabled rules and collects
//!   diagnostics sorted by `(file, span.start)`.
//!
//! Phase 3 also ships one proof-of-life rule
//! ([`rules::trailing_whitespace`]) to validate that the pipeline plumbs
//! through end-to-end before Phase 4 adds the five named production rules.

pub mod rule;
pub mod rules;

pub use rule::{LintCtx, LintRule};

use std::path::Path;

use crate::project::{MindcraftConfig, RuleSeverity};

use rule::RuleRegistry;

/// A byte-range in the original source string.
///
/// Offsets are into the `source` field of [`LintCtx`] — the raw `.mind` file
/// content before any comment stripping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

/// A single lint finding produced by a [`LintRule`].
///
/// The schema is shared verbatim with `mindc check` and is the stable contract
/// for the `--reporter json` output (RFC 0007 §6).
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Stable rule identifier, e.g. `"lint::trailing_whitespace"`.
    pub rule_id: String,
    /// Resolved severity for this file (after glob-scoped override merging).
    pub severity: RuleSeverity,
    /// Human-readable description of the finding.
    pub message: String,
    /// Source file that triggered the diagnostic.
    pub file: std::path::PathBuf,
    /// Byte span within `file` where the finding was detected.
    pub span: SourceSpan,
    /// Optional suggested fix description (not machine-applicable in Phase 3).
    pub help: Option<String>,
}

/// Run all enabled rules against `ctx`, returning diagnostics sorted by
/// `(span.start)`.
///
/// Rules whose effective severity is [`RuleSeverity::Off`] for the given
/// file are skipped entirely — their `check` method is not called.
pub fn run_lint(ctx: &LintCtx<'_>, registry: &RuleRegistry) -> Vec<Diagnostic> {
    let mut diagnostics: Vec<Diagnostic> = registry
        .rules()
        .iter()
        .filter_map(|rule| {
            let sev = registry.effective_severity(rule.id(), ctx.config, ctx.file);
            if sev == RuleSeverity::Off {
                return None;
            }
            let raw = rule.check(ctx);
            // Override the severity on every emitted diagnostic with the
            // resolved value so callers always see the file-local effective
            // severity, not the rule's compile-time default.
            Some(raw.into_iter().map(move |mut d| {
                d.severity = sev;
                d
            }))
        })
        .flatten()
        .collect();

    diagnostics.sort_by_key(|d| d.span.start);
    diagnostics
}

/// Convenience: build a [`LintCtx`], run lint, return diagnostics.
///
/// This is the integration point used by the `mindc check` / `mindc lint`
/// CLI drivers once they are wired up in Phase 5.
pub fn check_source(
    source: &str,
    file: &Path,
    config: &MindcraftConfig,
    registry: &RuleRegistry,
) -> Vec<Diagnostic> {
    use crate::parser::parse_with_trivia;
    use crate::ast::Module;
    use crate::parser::TriviaStream;

    // Best-effort: if the source doesn't parse, return no diagnostics.
    // A parse-error is surfaced by the compile pipeline, not the lint pipeline.
    let (module, trivia): (Module, TriviaStream) = match parse_with_trivia(source) {
        Ok(pair) => pair,
        Err(_) => return vec![],
    };

    let ctx = LintCtx {
        module: &module,
        trivia: &trivia,
        source,
        file,
        config,
    };
    run_lint(&ctx, registry)
}
