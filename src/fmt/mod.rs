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
}

impl std::fmt::Display for FmtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
/// Line endings are normalised to LF before any processing.  This ensures
/// consistent byte-offset arithmetic in the trivia and span indices on every
/// platform, including Windows where `std::fs::read_to_string` may return
/// CRLF content when `core.autocrlf` is active.
///
/// # Errors
///
/// Returns [`FmtError::ParseError`] when `src` cannot be parsed.
pub fn format_source(src: &str, cfg: &MindcraftFormatConfig) -> Result<String, FmtError> {
    // Normalise CR+LF and bare CR to LF so the parser, the trivia collector,
    // and the printer all see a consistent LF-only byte stream.  Without this,
    // on Windows (where git may check out files with CRLF endings), the trivia
    // byte-offsets (recorded in the CRLF source) diverge from the
    // `stripped_idx` used by the printer (built from an LF-only stripped copy),
    // causing comments to be placed at the wrong indent level on the first
    // formatting pass and producing a non-idempotent result.  The second pass
    // always receives LF-only output from the printer, so without normalisation
    // pass 1 and pass 2 differ — violating the idempotence contract.
    let normalized;
    let src = if src.contains('\r') {
        normalized = src.replace("\r\n", "\n").replace('\r', "\n");
        normalized.as_str()
    } else {
        src
    };
    let (module, trivia) = parse_with_trivia(src).map_err(FmtError::ParseError)?;
    Ok(printer::print_module(&module, &trivia, cfg, src))
}
