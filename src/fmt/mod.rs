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

mod printer;
pub mod cli;

use crate::parser::{parse_with_trivia, ParseError};
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
/// # Errors
///
/// Returns [`FmtError::ParseError`] when `src` cannot be parsed.
pub fn format_source(src: &str, cfg: &MindcraftFormatConfig) -> Result<String, FmtError> {
    let (module, trivia) = parse_with_trivia(src).map_err(FmtError::ParseError)?;
    Ok(printer::print_module(&module, &trivia, cfg, src))
}
