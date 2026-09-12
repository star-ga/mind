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

//! `mindc fmt` — Phase 2A canonical formatter.
//!
//! Entry point: [`format_source`].  Parses the source through the trivia-aware
//! front-end, then walks the AST with `printer::print_module` to emit a
//! canonicalised string.
//!
//! Phase 2A rules (no soft line-wrap):
//! - Indent `cfg.indent_width` spaces per nesting; never tabs.
//! - Single blank line between top-level items; no leading/trailing blanks.
//! - Whitespace normalisation inside expressions.
//! - Comment re-attachment from the trivia stream.
//! - String-literal contents passed through bytewise.

pub mod cli;
mod printer;

use crate::parser::{ParseError, parse_with_trivia};
use crate::project::MindcraftFormatConfig;

/// Error type for formatting operations.
#[derive(Debug, Clone)]
pub enum FmtError {
    /// The source could not be parsed.
    ParseError(Vec<ParseError>),
    /// Formatting would DESTROY source: identifiers present in the input are
    /// absent from the output.  The formatter refuses to write rather than
    /// emit a mangled file.
    ///
    /// MEASURED 2026-09-11 on `MindLLM/mind/governance_bridge.mind`: `mindc fmt`
    /// reported success and cut the file from 122 lines to 80, deleting the
    /// `module governance_bridge` declaration, a `struct`, a `fn`, and four
    /// `inference_auth_hash` call sites.  A layout formatter must never change
    /// which identifiers a file mentions, so any loss is a bug in the
    /// parser/printer pair and writing the result would lose the user's code.
    LossyFormat(Vec<(String, usize)>),
}

impl std::fmt::Display for FmtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FmtError::LossyFormat(lost) => {
                write!(
                    f,
                    "refusing to write: formatting would DELETE source. \
                     Identifiers present in the input are missing from the output"
                )?;
                for (name, n) in lost {
                    write!(f, "; `{name}` x{n}")?;
                }
                write!(
                    f,
                    ". This is a formatter bug, not a problem with your file — the \
                     file is unchanged. Please report it with the source attached."
                )
            }
            FmtError::ParseError(errs) => {
                write!(f, "parse error(s):")?;
                for e in errs {
                    write!(f, " {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for FmtError {}

/// Format a MIND source string according to `cfg`, returning the canonical
/// representation.
///
/// The output is idempotent: `format_source(format_source(s, c), c) ==
/// format_source(s, c)`.
///
/// # Errors
///
/// Returns [`FmtError::ParseError`] when `src` cannot be parsed.
pub fn format_source(src: &str, cfg: &MindcraftFormatConfig) -> Result<String, FmtError> {
    let (module, trivia) = parse_with_trivia(src).map_err(FmtError::ParseError)?;
    let out = printer::print_module(&module, &trivia, cfg, src);
    // FAIL CLOSED ON DATA LOSS. See `FmtError::LossyFormat`.
    if let Some(lost) = lost_identifiers(src, &out) {
        return Err(FmtError::LossyFormat(lost));
    }
    Ok(out)
}

/// Canonicalisations the printer performs DELIBERATELY, as `(input, output)`
/// keyword pairs.  Listed explicitly so an intentional rewrite is never
/// mistaken for data loss — and so adding one is a decision someone records
/// here rather than a silent widening of what counts as "not lost".
const CANONICAL_KEYWORD_REWRITES: &[(&str, &str)] = &[("use", "import")];

/// Identifiers and keywords the output drops relative to the input, or `None`
/// when nothing is lost.
///
/// A layout formatter must not change WHICH identifiers a file mentions, so
/// this is a whole-file invariant rather than a heuristic.  Comments and string
/// literals are excluded: their contents are passed through bytewise and are
/// not code, so a word appearing only inside them says nothing about the AST.
///
/// Deliberately one-directional — it reports LOSSES, not gains.  A printer that
/// inserts an identifier is suspect too, but this guard exists to stop the
/// user's code disappearing, and a narrow check that always means what it says
/// is worth more than a broad one that needs exceptions.
fn lost_identifiers(src: &str, out: &str) -> Option<Vec<(String, usize)>> {
    use std::collections::BTreeMap;

    let before = identifier_counts(src);
    let mut after = identifier_counts(out);

    // Credit each deliberate rewrite: every `import` in the output may account
    // for one `use` in the input.
    for (from, to) in CANONICAL_KEYWORD_REWRITES {
        let produced = after.get(*to).copied().unwrap_or(0);
        if produced > 0 {
            *after.entry((*from).to_string()).or_insert(0) += produced;
        }
    }

    let mut lost: BTreeMap<String, usize> = BTreeMap::new();
    for (name, n) in &before {
        let kept = after.get(name).copied().unwrap_or(0);
        if *n > kept {
            lost.insert(name.clone(), n - kept);
        }
    }
    if lost.is_empty() {
        None
    } else {
        Some(lost.into_iter().collect())
    }
}

/// Multiset of identifier-shaped words, with comments and string literals
/// removed.  Hand-scanned rather than lexed because the formatter must be able
/// to run this check even on input whose finer structure the parser mishandled
/// — which is exactly the situation the guard is for.
fn identifier_counts(src: &str) -> std::collections::BTreeMap<String, usize> {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut word = String::new();

    let flush = |word: &mut String, counts: &mut std::collections::BTreeMap<String, usize>| {
        if !word.is_empty() {
            *counts.entry(std::mem::take(word)).or_insert(0) += 1;
        }
    };

    while i < bytes.len() {
        let b = bytes[i];
        // line comment
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            flush(&mut word, &mut counts);
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // block comment
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            flush(&mut word, &mut counts);
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        // string literal (double-quoted, backslash escapes)
        if b == b'"' {
            flush(&mut word, &mut counts);
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            i = (i + 1).min(bytes.len());
            continue;
        }
        if b.is_ascii_alphanumeric() || b == b'_' {
            if word.is_empty() && b.is_ascii_digit() {
                // a numeric literal is not an identifier; skip the run
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
                {
                    i += 1;
                }
                continue;
            }
            word.push(b as char);
            i += 1;
            continue;
        }
        flush(&mut word, &mut counts);
        i += 1;
    }
    flush(&mut word, &mut counts);
    counts
}
