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

//! [`LintRule`] trait, [`LintCtx`] context, and [`RuleRegistry`].

use std::path::Path;

use crate::ast::Module;
use crate::parser::TriviaStream;
use crate::project::{MindcraftConfig, MindcraftOverride, RuleSeverity};

use super::Diagnostic;

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Immutable context passed to every [`LintRule::check`] invocation.
pub struct LintCtx<'a> {
    /// Typed AST of the file under analysis.
    pub module: &'a Module,
    /// Trivia stream (comments, blank lines) from the same parse.
    pub trivia: &'a TriviaStream,
    /// Raw source text of the file — byte offsets in [`Diagnostic::span`]
    /// refer into this string.
    pub source: &'a str,
    /// Canonical path of the file, used for glob-scoped override matching.
    pub file: &'a Path,
    /// Active Mindcraft configuration for the project.
    pub config: &'a MindcraftConfig,
}

// ---------------------------------------------------------------------------
// Rule trait
// ---------------------------------------------------------------------------

/// A single lint rule.
///
/// Each rule is registered with the [`RuleRegistry`] once at startup.  The
/// registry calls `check` only when the resolved severity is not
/// [`RuleSeverity::Off`].
pub trait LintRule: Send + Sync {
    /// Stable rule identifier, e.g. `"lint::trailing_whitespace"`.
    ///
    /// Must be unique within a registry.  By convention MIND rules use the
    /// `lint::` namespace; project-local rules may use any non-`lint::` prefix.
    fn id(&self) -> &'static str;

    /// Severity when no project configuration overrides it.
    fn default_severity(&self) -> RuleSeverity;

    /// Short human-readable description shown in `mindc lint --list`.
    fn description(&self) -> &'static str;

    /// Analyse `ctx` and return all findings.
    ///
    /// The registry sets `Diagnostic::severity` to the resolved value after
    /// `check` returns; rules should emit their `default_severity()` or any
    /// placeholder — it will be overwritten.
    fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic>;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Holds all registered lint rules and provides severity resolution.
#[derive(Default)]
pub struct RuleRegistry {
    /// Rules in registration order.  Iteration order is deterministic.
    pub(crate) rules: Vec<Box<dyn LintRule>>,
}

impl RuleRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rule.  Rules are stored in registration order.
    pub fn register(&mut self, rule: impl LintRule + 'static) {
        self.rules.push(Box::new(rule));
    }

    /// Iterate over registered rules (in registration order).
    pub fn rules(&self) -> &[Box<dyn LintRule>] {
        &self.rules
    }

    /// Return rule ids present in `config.rules` that are not registered.
    ///
    /// These are reported by `mindc check` as config-level diagnostics so
    /// typos don't silently become no-ops (RFC 0007 §5).
    pub fn unknown_rules(&self, config: &MindcraftConfig) -> Vec<String> {
        let registered: std::collections::HashSet<&str> =
            self.rules.iter().map(|r| r.id()).collect();
        let mut unknown: Vec<String> = config
            .rules
            .keys()
            .filter(|id| !registered.contains(id.as_str()))
            .cloned()
            .collect();
        unknown.sort(); // deterministic output
        unknown
    }

    /// Resolve the effective severity for `rule_id` when linting `file_path`.
    ///
    /// Merge order (later wins):
    /// 1. Rule's `default_severity()` (or [`RuleSeverity::Warn`] when the
    ///    rule is not registered — callers may query hypothetical ids).
    /// 2. `config.rules` base map.
    /// 3. Each `[[mindcraft.overrides]]` entry whose glob matches `file_path`,
    ///    in declaration order (later entries take precedence).
    pub fn effective_severity(
        &self,
        rule_id: &str,
        config: &MindcraftConfig,
        file_path: &Path,
    ) -> RuleSeverity {
        // Start from the rule's built-in default.
        let mut sev = self
            .rules
            .iter()
            .find(|r| r.id() == rule_id)
            .map(|r| r.default_severity())
            .unwrap_or(RuleSeverity::Warn);

        // Apply the base config map.
        if let Some(&base_sev) = config.rules.get(rule_id) {
            sev = base_sev;
        }

        // Apply each matching override layer in declaration order.
        for ov in &config.overrides {
            if override_matches(ov, file_path) {
                if let Some(&ov_sev) = ov.rules.get(rule_id) {
                    sev = ov_sev;
                }
            }
        }

        sev
    }
}

// ---------------------------------------------------------------------------
// Glob matching (no external crate)
// ---------------------------------------------------------------------------

/// Return true if `file_path` is matched by the override entry.
///
/// A file matches when it satisfies at least one `includes` glob AND none of
/// the `excludes` globs.  An empty `includes` list matches nothing (the
/// override is effectively disabled); an empty `excludes` list never excludes.
fn override_matches(ov: &MindcraftOverride, file_path: &Path) -> bool {
    if ov.includes.is_empty() {
        return false;
    }
    let path_str = file_path.to_string_lossy();
    let included = ov.includes.iter().any(|pat| glob_match(pat, &path_str));
    if !included {
        return false;
    }
    let excluded = ov.excludes.iter().any(|pat| glob_match(pat, &path_str));
    !excluded
}

/// Minimal glob pattern matching: `*` matches any sequence of non-separator
/// characters; `**` matches any sequence including path separators; `?`
/// matches exactly one character.
///
/// Both `/` and `\` are treated as path separators so patterns are
/// portable across platforms.
fn glob_match(pattern: &str, text: &str) -> bool {
    glob_match_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_inner(pat: &[u8], txt: &[u8]) -> bool {
    // `**` — must be detected by inspecting pat alone, not txt.
    if pat.first() == Some(&b'*') && pat.get(1) == Some(&b'*') {
        let rest_pat = &pat[2..];
        // Skip any trailing separator after `**`.
        let rest_pat = if rest_pat.first() == Some(&b'/') || rest_pat.first() == Some(&b'\\') {
            &rest_pat[1..]
        } else {
            rest_pat
        };
        // Try matching rest_pat against every suffix of txt (including empty).
        for i in 0..=txt.len() {
            if glob_match_inner(rest_pat, &txt[i..]) {
                return true;
            }
        }
        return false;
    }

    match (pat.first(), txt.first()) {
        // Both exhausted — match.
        (None, None) => true,
        // Pattern exhausted but text remains — no match.
        (None, Some(_)) => false,
        // `*` — matches zero or more non-separator characters.
        (Some(&b'*'), _) => {
            let rest_pat = &pat[1..];
            for i in 0..=txt.len() {
                if i > 0 && is_sep(txt[i - 1]) {
                    break; // `*` never crosses a separator
                }
                if glob_match_inner(rest_pat, &txt[i..]) {
                    return true;
                }
            }
            false
        }
        // `?` — matches exactly one non-separator character.
        (Some(&b'?'), Some(&c)) if !is_sep(c) => glob_match_inner(&pat[1..], &txt[1..]),
        (Some(&b'?'), _) => false,
        // Literal — must match exactly (case-sensitive).
        (Some(&p), Some(&t)) if p == t => glob_match_inner(&pat[1..], &txt[1..]),
        // Mismatch.
        _ => false,
    }
}

#[inline]
fn is_sep(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

// ---------------------------------------------------------------------------
// Unit tests for glob_match
// ---------------------------------------------------------------------------

#[cfg(test)]
mod glob_tests {
    use super::glob_match;

    #[test]
    fn literal_match() {
        assert!(glob_match("foo.mind", "foo.mind"));
        assert!(!glob_match("foo.mind", "bar.mind"));
    }

    #[test]
    fn star_matches_within_segment() {
        assert!(glob_match("tests/*.mind", "tests/foo.mind"));
        assert!(!glob_match("tests/*.mind", "tests/sub/foo.mind"));
    }

    #[test]
    fn double_star_crosses_separators() {
        assert!(glob_match("tests/**", "tests/sub/foo.mind"));
        assert!(glob_match("tests/**/*.mind", "tests/sub/foo.mind"));
    }

    #[test]
    fn question_mark() {
        assert!(glob_match("fo?.mind", "foo.mind"));
        assert!(!glob_match("fo?.mind", "fo.mind"));
    }
}
