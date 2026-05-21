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

//! Minimal in-process `.gitignore` parser for VCS-aware file filtering.
//!
//! Implements enough of the gitignore spec to be useful for project walks:
//! - Blank lines and `#`-comments are skipped.
//! - Patterns beginning with `!` negate (un-ignore) a previously ignored path.
//! - A leading `/` anchors to the repo root; without it the pattern may match
//!   at any depth (gitignore § "A pattern without a slash …").
//! - A trailing `/` matches directories only (we skip directory detection
//!   here — we test path strings, so trailing-slash patterns match any path
//!   that starts with the prefix).
//! - `*` matches any sequence of non-`/` characters; `**` matches across `/`.
//! - `?` matches one character except `/`.
//!
//! This is intentionally not a full git-core reimplementation. It covers the
//! patterns encountered in typical `.gitignore` files: `target/`, `*.o`,
//! `build/**`, `!src/generated/keep.mind`, etc.

use std::path::Path;

/// A compiled set of gitignore rules loaded from one or more ignore files.
#[derive(Default)]
pub struct GitignoreFilter {
    /// Rules in file order; later rules override earlier ones when they
    /// produce conflicting decisions for the same path.
    rules: Vec<Rule>,
}

struct Rule {
    /// The normalised glob pattern (anchoring prefix stripped).
    pattern: String,
    /// If true the rule starts with `!` — it un-ignores matched paths.
    negated: bool,
    /// If true the original pattern started with `/` — match only from root.
    anchored: bool,
    /// If true the original pattern ended with `/` — directory hint (stored
    /// for future use; path-string testing always treats it as a prefix match).
    #[allow(dead_code)]
    dir_only: bool,
}

impl GitignoreFilter {
    /// Load rules from a file; silently skips unreadable files.
    pub fn load_file(&mut self, path: &Path) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        for raw in text.lines() {
            self.add_line(raw);
        }
    }

    /// Parse and append one line from an ignore file.
    fn add_line(&mut self, raw: &str) {
        // Trailing inline `\` escapes are rare; skip full unescape for now.
        let line = raw.trim_end();
        // Blank and comment lines.
        if line.is_empty() || line.starts_with('#') {
            return;
        }

        let negated = line.starts_with('!');
        let line = if negated { &line[1..] } else { line };

        // A leading `/` anchors to the root.
        let anchored = line.starts_with('/');
        let line = if anchored { &line[1..] } else { line };

        // A trailing `/` is a directory hint.
        let dir_only = line.ends_with('/');
        let pattern = if dir_only {
            line[..line.len() - 1].to_string()
        } else {
            line.to_string()
        };

        if pattern.is_empty() {
            return;
        }

        self.rules.push(Rule {
            pattern,
            negated,
            anchored,
            dir_only,
        });
    }

    /// Return `true` if `path` (relative to the repo root) should be ignored.
    ///
    /// Rules are evaluated in order; the last matching rule wins.
    pub fn is_ignored(&self, rel_path: &str) -> bool {
        let mut ignored = false;
        let path_str = rel_path.replace('\\', "/");

        for rule in &self.rules {
            if rule_matches(rule, &path_str) {
                ignored = !rule.negated;
            }
        }
        ignored
    }
}

fn rule_matches(rule: &Rule, path: &str) -> bool {
    if rule.anchored {
        // Anchored: pattern must match from the start of the path.
        //
        // A pattern like `/build` should match `build/foo.mind` (the entry and
        // everything under it) but NOT `sub/build/foo.mind`.
        // Strategy: try an exact glob match first; if the pattern contains no
        // wildcard and no `/`, also accept any path that starts with
        // `<pattern>/` (i.e. files inside the named root entry).
        if glob_match_path(&rule.pattern, path) {
            return true;
        }
        // If pattern has no wildcards, treat it as a directory prefix.
        if !rule.pattern.contains('*') && !rule.pattern.contains('?') {
            let prefix = format!("{}/", rule.pattern);
            if path.starts_with(&*prefix) {
                return true;
            }
        }
        // For dir-only: the path just needs to be inside that directory.
        if rule.dir_only {
            let prefix = format!("{}/", rule.pattern);
            return path.starts_with(&*prefix) || path == rule.pattern.as_str();
        }
        return false;
    }

    // For directory-only patterns without anchoring (e.g. `target/`):
    // match any path that is inside the named directory at any depth.
    if rule.dir_only {
        // Check if any path component or prefix equals the pattern.
        let parts: Vec<&str> = path.split('/').collect();
        for i in 0..parts.len() {
            if glob_match_path(&rule.pattern, parts[i]) {
                // This component matches — the path is under this directory.
                return true;
            }
        }
        return false;
    }

    // Without anchoring, check the full path first.
    if glob_match_path(&rule.pattern, path) {
        return true;
    }
    // Also check each path component suffix (e.g. `*.o` should match `src/foo.o`).
    if !rule.pattern.contains('/') {
        // No slash in pattern — can match any segment.
        let parts: Vec<&str> = path.split('/').collect();
        // Try matching the basename.
        if let Some(&basename) = parts.last() {
            if glob_match_path(&rule.pattern, basename) {
                return true;
            }
        }
    }
    false
}

/// Match `pattern` against the full `text` using gitignore glob semantics.
fn glob_match_path(pattern: &str, text: &str) -> bool {
    glob_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_inner(pat: &[u8], txt: &[u8]) -> bool {
    // `**` — matches any sequence including `/`.
    if pat.starts_with(b"**") {
        let rest_pat = &pat[2..];
        let rest_pat = rest_pat.strip_prefix(b"/").unwrap_or(rest_pat);
        if rest_pat.is_empty() {
            return true; // `**` with nothing after matches everything.
        }
        for i in 0..=txt.len() {
            if glob_inner(rest_pat, &txt[i..]) {
                return true;
            }
        }
        return false;
    }

    match (pat.first(), txt.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(b'*'), _) => {
            let rest_pat = &pat[1..];
            for i in 0..=txt.len() {
                if i > 0 && txt[i - 1] == b'/' {
                    break; // `*` never crosses `/`
                }
                if glob_inner(rest_pat, &txt[i..]) {
                    return true;
                }
            }
            false
        }
        (Some(b'?'), Some(&c)) if c != b'/' => glob_inner(&pat[1..], &txt[1..]),
        (Some(b'?'), _) => false,
        (Some(&p), Some(&t)) if p == t => glob_inner(&pat[1..], &txt[1..]),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(lines: &[&str]) -> GitignoreFilter {
        let mut f = GitignoreFilter::default();
        for l in lines {
            f.add_line(l);
        }
        f
    }

    #[test]
    fn ignores_target_dir() {
        let f = filter(&["target/"]);
        assert!(f.is_ignored("target/debug/foo.mind"));
        assert!(!f.is_ignored("src/target.mind"));
    }

    #[test]
    fn wildcard_extension() {
        let f = filter(&["*.tmp"]);
        assert!(f.is_ignored("foo.tmp"));
        assert!(f.is_ignored("src/foo.tmp"));
        assert!(!f.is_ignored("src/foo.mind"));
    }

    #[test]
    fn negation_un_ignores() {
        let f = filter(&["*.mind", "!keep.mind"]);
        assert!(f.is_ignored("other.mind"));
        assert!(!f.is_ignored("keep.mind"));
    }

    #[test]
    fn anchored_only_root() {
        let f = filter(&["/build"]);
        assert!(f.is_ignored("build/foo.mind"));
        assert!(!f.is_ignored("sub/build/foo.mind"));
    }

    #[test]
    fn double_star() {
        let f = filter(&["build/**"]);
        assert!(f.is_ignored("build/sub/foo.mind"));
        assert!(f.is_ignored("build/foo.mind"));
    }

    #[test]
    fn comment_and_blank_skipped() {
        let f = filter(&["# this is a comment", "", "*.tmp"]);
        assert!(f.is_ignored("foo.tmp"));
        assert_eq!(f.rules.len(), 1);
    }
}
