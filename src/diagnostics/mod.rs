//! Pretty diagnostics: spans, line/col, caret-highlights.

use std::ops::Range;

/// Byte-span in the original source (inclusive start, exclusive end).
pub type Span = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub line: usize, // 1-based
    pub col: usize,  // 1-based (Unicode-agnostic; counts bytes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
    pub start: Location,
    pub end: Location,
}

/// Compute (line, col) from byte offset.
fn offset_to_loc(src: &str, offset: usize) -> Location {
    let mut line = 1usize;
    let mut col = 1usize;
    let mut count = 0usize;
    for ch in src.chars() {
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
    Location { line, col }
}

/// Extract the single source line containing `span.start`.
fn line_at(src: &str, span: Span) -> (String, usize /* line_start_offset */) {
    let bytes = src.as_bytes();
    let mut b = span.start;
    while b > 0 && bytes[b - 1] != b'\n' {
        b -= 1;
    }
    let mut e = span.start;
    while e < bytes.len() && bytes[e] != b'\n' {
        e += 1;
    }
    (src[b..e].to_string(), b)
}

/// Render caret-highlight under the selected span (single-line best effort).
pub fn render(src: &str, diag: &Diagnostic) -> String {
    let span = diag.span.clone();
    let (line_str, line_off) = line_at(src, span.clone());
    let caret_start = span.start.saturating_sub(line_off);
    let caret_len = span.end.saturating_sub(span.start).max(1);

    let mut carets = String::new();
    for _ in 0..caret_start {
        carets.push(' ');
    }
    for _ in 0..caret_len {
        carets.push('^');
    }

    format!(
        "error: {}\n--> line {}, col {}\n{}\n{}",
        diag.message, diag.start.line, diag.start.col, line_str, carets
    )
}

impl Diagnostic {
    /// Construct from a chumsky `Simple` error.
    pub fn from_chumsky(src: &str, e: chumsky::error::Simple<char>) -> Self {
        let span = e.span();
        let start = offset_to_loc(src, span.start);
        let end = offset_to_loc(src, span.end);
        let message = e.to_string();
        Diagnostic {
            message,
            span,
            start,
            end,
        }
    }
}
