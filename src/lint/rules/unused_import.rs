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

//! Rule `lint::unused_import` — detect `use`/`import` declarations where no
//! symbol from the imported module is referenced in the module body.
//!
//! ## Detection strategy (conservative)
//!
//! The MIND AST does not carry resolved import bindings at the lint layer.
//! This rule uses a text-scan heuristic:
//!
//! 1. Collect all `Node::Import` items (which represent both `import` and
//!    `use` statements, since `parse_use` also produces `Node::Import`).
//! 2. For each import, derive the "local name" from the **last segment** of
//!    the dotted path (e.g. `use std.vec` → `"vec"`).
//! 3. Scan the raw `source` text for any occurrence of the local name as a
//!    word boundary match (i.e. surrounded by non-alphanumeric / non-`_`
//!    characters, or at start/end of file).
//! 4. The import span itself contains the local name, so subtract one
//!    occurrence to avoid counting the declaration itself.
//! 5. If the count after subtracting the declaration's occurrence is zero,
//!    the import is unused.
//!
//! This is intentionally conservative: any ambiguous or partial match is
//! treated as "used" to avoid false positives.

use crate::ast::Node;
use crate::lint::rule::{LintCtx, LintRule};
use crate::lint::{Diagnostic, SourceSpan};
use crate::project::RuleSeverity;

/// Lint rule: unused `use`/`import` declaration.
pub struct UnusedImport;

impl LintRule for UnusedImport {
    fn id(&self) -> &'static str {
        "lint::unused_import"
    }

    fn default_severity(&self) -> RuleSeverity {
        RuleSeverity::Warn
    }

    fn description(&self) -> &'static str {
        "unused import — the imported module is never referenced in this file"
    }

    fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let source = ctx.source;

        for item in &ctx.module.items {
            let Node::Import { path, span } = item else {
                continue;
            };
            let local_name = match path.last() {
                Some(n) => n.as_str(),
                None => continue,
            };

            // Count word-boundary occurrences of `local_name` in the source,
            // excluding comment text (// ... to end of line).
            // The import declaration itself contributes exactly 1 occurrence
            // (the last path segment in `use std.vec`). If total non-comment
            // count is only 1, nothing outside the declaration references it.
            let non_comment_count = count_word_occurrences_non_comment(source, local_name);

            if non_comment_count <= 1 {
                diagnostics.push(Diagnostic {
                    rule_id: self.id().to_string(),
                    severity: self.default_severity(),
                    message: format!("unused import `{}`", path.join(".")),
                    file: ctx.file.to_path_buf(),
                    span: SourceSpan {
                        start: span.start(),
                        end: span.end(),
                    },
                    help: Some(
                        "remove the `use` declaration or use the imported symbol".to_string(),
                    ),
                    auto_fix: None,
                });
            }
        }

        diagnostics
    }
}

/// Count word-boundary occurrences of `needle` in `haystack`, skipping bytes
/// that are inside a `//` line comment (i.e. after `//` up to the next `\n`).
///
/// The import declaration itself contributes one occurrence (the last path
/// segment in `use std.vec`), so callers compare the result against `<= 1`
/// to detect truly unused imports.
fn count_word_occurrences_non_comment(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let needle_len = needle_bytes.len();
    let len = bytes.len();

    // First pass: mark comment bytes.
    let mut comment = vec![false; len];
    {
        let mut j = 0usize;
        let mut in_str = false;
        let mut in_comment = false;
        while j < len {
            let b = bytes[j];
            if in_str {
                comment[j] = false;
                if b == b'\\' && j + 1 < len {
                    comment[j + 1] = false;
                    j += 2;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
            } else if in_comment {
                comment[j] = true;
                if b == b'\n' {
                    in_comment = false;
                }
            } else if b == b'"' {
                in_str = true;
                comment[j] = false;
            } else if b == b'/' && j + 1 < len && bytes[j + 1] == b'/' {
                comment[j] = true;
                in_comment = true;
            } else {
                comment[j] = false;
            }
            j += 1;
        }
    }

    // Second pass: count "prefix" matches outside comments.
    //
    // A match is counted when:
    //  - The byte before the match is not an alphanumeric/underscore character
    //    (i.e. the match starts at a word boundary), AND
    //  - The match is not followed by a non-underscore alphanumeric character
    //    that would make this a substring of a longer, unrelated identifier.
    //    (Allow `vec_new` as a reference to `vec` — the `_` suffix is okay.)
    //
    // In practice: left-boundary required (don't match `xvec`), right-boundary
    // flexible (match both `vec` alone and `vec_new`, `vec::new`, etc.).
    let mut count = 0usize;
    let mut i = 0usize;
    while i + needle_len <= len {
        if comment[i] {
            i += 1;
            continue;
        }
        if bytes[i..i + needle_len] == *needle_bytes {
            // Left boundary: must not be preceded by a word char.
            let left_ok = i == 0 || !is_word_char(bytes[i - 1]);
            if left_ok {
                count += 1;
            }
        }
        i += 1;
    }
    count
}

#[inline]
fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
