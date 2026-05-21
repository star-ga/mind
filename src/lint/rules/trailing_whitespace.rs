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

//! Rule `lint::trailing_whitespace`.
//!
//! Emits a [`Diagnostic`] for every line that ends with one or more space or
//! tab characters (before the newline, or at end-of-file without a newline).
//!
//! This rule supports auto-fix (RFC 0007 Phase 6): the trailing whitespace
//! bytes are deleted by replacing the span with an empty string.

use crate::lint::rule::{Fix, LintCtx, LintRule};
use crate::lint::{Diagnostic, SourceSpan};
use crate::project::RuleSeverity;

/// Lint rule: trailing whitespace on any line.
pub struct TrailingWhitespace;

impl LintRule for TrailingWhitespace {
    fn id(&self) -> &'static str {
        "lint::trailing_whitespace"
    }

    fn default_severity(&self) -> RuleSeverity {
        RuleSeverity::Warn
    }

    fn description(&self) -> &'static str {
        "trailing whitespace (space or tab) at end of line"
    }

    fn check(&self, ctx: &LintCtx<'_>) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let src = ctx.source.as_bytes();
        let len = src.len();
        let mut line_start = 0usize;

        while line_start <= len {
            // Find the end of the current line (position of '\n' or EOF).
            let line_end = src[line_start..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|rel| line_start + rel)
                .unwrap_or(len);

            // Scan backwards from the newline (or EOF) for trailing whitespace.
            let mut trailing_end = line_end;
            while trailing_end > line_start
                && (src[trailing_end - 1] == b' ' || src[trailing_end - 1] == b'\t')
            {
                trailing_end -= 1;
            }

            if trailing_end < line_end {
                // There is at least one trailing space/tab on this line.
                diagnostics.push(Diagnostic {
                    rule_id: self.id().to_string(),
                    severity: self.default_severity(),
                    message: "trailing whitespace".to_string(),
                    file: ctx.file.to_path_buf(),
                    span: SourceSpan {
                        start: trailing_end,
                        end: line_end,
                    },
                    help: Some("remove trailing spaces/tabs".to_string()),
                    auto_fix: None, // populated by auto_fix() below
                });
            }

            // Advance past the newline.
            if line_end >= len {
                break;
            }
            line_start = line_end + 1;
        }

        diagnostics
    }

    /// Delete the trailing whitespace bytes (replace span with empty string).
    fn auto_fix(&self, _ctx: &LintCtx<'_>, diagnostic: &Diagnostic) -> Option<Fix> {
        Some(Fix {
            range: diagnostic.span.start..diagnostic.span.end,
            replacement: String::new(),
        })
    }
}
