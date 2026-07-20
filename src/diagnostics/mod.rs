// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Structured diagnostics for the Core v1 compiler pipeline.

use std::fmt::Write as _;
use std::io::{IsTerminal, Write};

use colored::Colorize;
use serde::Serialize;

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// Normalized source location for a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Span {
    pub file: Option<String>,
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

impl Span {
    /// Construct a span from byte offsets within a source string.
    pub fn from_offsets(source: &str, start: usize, end: usize, file: Option<&str>) -> Self {
        let (line, column) = offset_to_line_col(source, start);
        let length = end.saturating_sub(start).max(1);
        Span {
            file: file.map(str::to_string),
            line,
            column,
            length,
        }
    }
}

thread_local! {
    // (src.as_ptr() as usize, src.len(), line_start_byte_offsets)
    static LINE_INDEX: std::cell::RefCell<Option<(usize, usize, Vec<usize>)>> =
        const { std::cell::RefCell::new(None) };
}

/// Force the next `offset_to_line_col` call to rebuild its line-start index.
///
/// The cache below is validated by `(src.as_ptr(), src.len())`, a fast but
/// address-based check: it is sound ONLY as long as no two DIFFERENT source
/// strings ever share both an address and a length while both are live
/// candidates for lookup. Within the recursive type-check of ONE source that
/// invariant holds trivially (the same `&str` is threaded through unchanged).
/// Across DIFFERENT top-level sources — `mindc check std/ examples/` type-checks
/// many files in one process; test binaries type-check many small sources on
/// one thread — the previous source's `String` is dropped and a new one can be
/// allocated at the SAME address with the SAME length (especially likely with
/// `mindc`'s pooled `SmallHeapAlloc`), which would silently serve the WRONG
/// line:col for the new source. Callers MUST call this once at every genuine
/// "new top-level source" boundary — see `type_checker::check_module_types_in_file`,
/// the sole public entry point all such boundaries route through.
pub(crate) fn reset_line_index_cache() {
    LINE_INDEX.with(|cell| *cell.borrow_mut() = None);
}

/// Byte offset → 1-based (line, char-column).
///
/// A naive left-to-right `src.chars()` scan is O(offset) per call, and the type
/// checker calls this once per diagnostic span. On a large source (the ~25k-line
/// self-host `main.mind`) that made `mindc check` quadratic — ~83% of a 6.5s
/// check was spent here. Instead cache a line-start byte-offset index (one entry
/// per `\n`) in a thread-local, keyed by the source's (ptr, len) so it is built
/// once per file and reused for every span, then binary-search the line and count
/// only the chars within that one line. O(log lines + line_len) per call.
///
/// Diagnostics-only: this feeds error/warning `line:col`, never emitted bytes, so
/// it is outside the byte-identity wedge. The result is bit-for-bit the same
/// (line, col) the old scan produced (char-based columns), which the type-error
/// span tests pin. See `reset_line_index_cache` for the address-reuse caveat this
/// cache's callers must respect.
fn offset_to_line_col(src: &str, offset: usize) -> (usize, usize) {
    LINE_INDEX.with(|cell| {
        let mut cell = cell.borrow_mut();
        let key = (src.as_ptr() as usize, src.len());
        if cell.as_ref().map(|(p, l, _)| (*p, *l)) != Some(key) {
            // starts[0] = 0; starts[k] = byte offset just past the k-th '\n'.
            let mut starts = Vec::with_capacity(src.len() / 24 + 1);
            starts.push(0usize);
            for (i, b) in src.bytes().enumerate() {
                if b == b'\n' {
                    starts.push(i + 1);
                }
            }
            *cell = Some((key.0, key.1, starts));
        }
        let starts = &cell.as_ref().unwrap().2;
        // Line containing `offset` (1-based). Mirrors the old scan: an offset that
        // lands exactly on a line start belongs to that new line.
        let line = match starts.binary_search(&offset) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        let line_start = starts[line - 1];
        // Column = 1 + chars from the line start up to `offset` (char-based, as
        // before). Clamp to a valid char boundary to stay panic-free on a span
        // that (defensively) lands mid-char.
        let end = offset.min(src.len());
        let col = if end <= line_start {
            1
        } else {
            // The old left-to-right scan counted a char as soon as its START byte
            // was < offset, so a mid-char offset includes the char containing it.
            // Round `end` UP to the next char boundary to reproduce that exactly
            // (a no-op for the real, always-char-aligned span offsets).
            let mut e = end;
            while e < src.len() && !src.is_char_boundary(e) {
                e += 1;
            }
            src[line_start..e].chars().count() + 1
        };
        (line, col)
    })
}

/// Machine-readable diagnostic emitted by the compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    pub phase: &'static str,
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    #[serde(default)]
    pub notes: Vec<String>,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn error(phase: &'static str, code: &'static str, message: impl Into<String>) -> Self {
        Diagnostic {
            phase,
            code,
            severity: Severity::Error,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Attach a file name if it was missing.
    pub fn fill_file(mut self, file: Option<&str>) -> Self {
        if let (Some(name), Some(span)) = (file, self.span.as_mut()) {
            if span.file.is_none() {
                span.file = Some(name.to_string());
            }
        }
        self
    }
}

/// CLI-facing diagnostic output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticFormat {
    Human,
    Short,
    Json,
}

impl DiagnosticFormat {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "human" => Some(DiagnosticFormat::Human),
            "short" => Some(DiagnosticFormat::Short),
            "json" => Some(DiagnosticFormat::Json),
            _ => None,
        }
    }
}

/// Color handling for CLI diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

impl ColorChoice {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "auto" => Some(ColorChoice::Auto),
            "always" => Some(ColorChoice::Always),
            "never" => Some(ColorChoice::Never),
            _ => None,
        }
    }

    fn should_color(self, is_terminal: bool) -> bool {
        match self {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => is_terminal,
        }
    }
}

/// Emitter that renders diagnostics in human, short, or JSON formats.
pub struct DiagnosticEmitter {
    format: DiagnosticFormat,
    color: ColorChoice,
    is_tty: bool,
}

impl DiagnosticEmitter {
    pub fn new(format: DiagnosticFormat, color: ColorChoice) -> Self {
        let is_tty = std::io::stderr().is_terminal();
        DiagnosticEmitter {
            format,
            color,
            is_tty,
        }
    }

    pub fn emit(&self, diag: &Diagnostic, source: Option<&str>, mut writer: impl Write) {
        match self.format {
            DiagnosticFormat::Json => {
                let _ = serde_json::to_writer(&mut writer, diag);
                let _ = writeln!(writer);
            }
            DiagnosticFormat::Short => {
                let _ = writeln!(writer, "{}", self.render_short(diag));
            }
            DiagnosticFormat::Human => {
                let rendered = self.render_human(diag, source);
                let _ = writeln!(writer, "{rendered}");
            }
        }
    }

    pub fn emit_all(&self, diags: &[Diagnostic], source: Option<&str>) {
        for diag in diags {
            self.emit(diag, source, std::io::stderr());
        }
    }

    fn render_prefix(&self, diag: &Diagnostic) -> String {
        let mut prefix = format!("{}[{}]", diag.severity.as_str(), diag.phase);
        if !diag.code.is_empty() {
            let _ = write!(prefix, "[{}]", diag.code);
        }
        if self.color.should_color(self.is_tty) {
            match diag.severity {
                Severity::Error => prefix = prefix.red().bold().to_string(),
                Severity::Warning => prefix = prefix.yellow().bold().to_string(),
            }
        }
        prefix
    }

    fn render_location(&self, span: &Span) -> String {
        match &span.file {
            Some(file) => format!("{file}:{}:{}", span.line, span.column),
            None => format!("line {} column {}", span.line, span.column),
        }
    }

    fn render_short(&self, diag: &Diagnostic) -> String {
        let prefix = self.render_prefix(diag);
        let mut out = format!("{prefix}: {}", diag.message);
        if let Some(span) = &diag.span {
            let loc = self.render_location(span);
            let _ = write!(out, " ({loc})");
        }
        out
    }

    pub fn render_human(&self, diag: &Diagnostic, source: Option<&str>) -> String {
        let mut out = String::new();
        let prefix = self.render_prefix(diag);
        let _ = write!(out, "{prefix}: {}", diag.message);

        if let Some(span) = &diag.span {
            let loc = self.render_location(span);
            let _ = write!(out, "\n  --> {loc}");
            if let Some(src) = source {
                if let Some(line_str) = source_line(src, span.line) {
                    let indicator = caret_line(span.column, span.length);
                    let _ = write!(out, "\n   | {line_str}\n   | {indicator}");
                }
            }
        }

        for note in &diag.notes {
            let label = if self.color.should_color(self.is_tty) {
                "note".cyan().bold().to_string()
            } else {
                "note".to_string()
            };
            let _ = write!(out, "\n{label}: {note}");
        }

        if let Some(help) = &diag.help {
            let label = if self.color.should_color(self.is_tty) {
                "help".green().bold().to_string()
            } else {
                "help".to_string()
            };
            let _ = write!(out, "\n{label}: {help}");
        }

        out
    }
}

fn source_line(src: &str, line: usize) -> Option<String> {
    src.lines()
        .nth(line.saturating_sub(1))
        .map(|l| l.to_string())
}

fn caret_line(col: usize, len: usize) -> String {
    let mut out = String::new();
    for _ in 1..col {
        out.push(' ');
    }
    for _ in 0..len.max(1) {
        out.push('^');
    }
    out
}

/// Convenience wrapper for human diagnostics without ANSI colors.
pub fn render(source: &str, diag: &Diagnostic) -> String {
    DiagnosticEmitter::new(DiagnosticFormat::Human, ColorChoice::Never)
        .render_human(diag, Some(source))
}
