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

//! Trivia layer for the MIND parser.
//!
//! Trivia records capture source elements (comments, blank lines) that the
//! main AST discards.  The trivia layer is a parallel, optional output of
//! [`parse_with_trivia`] — zero overhead when not requested.
//!
//! [`parse_with_trivia`]: super::parse_with_trivia

/// Classification of a trivia record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriviaKind {
    /// A `//`-prefixed comment that is NOT a doc-comment.
    ///
    /// Includes the `//` prefix up to (but not including) the trailing newline.
    LineComment,

    /// A `///`-prefixed doc-comment (`///` followed by a non-`/` byte, or
    /// by end-of-line).  `////` and beyond are [`LineComment`][TriviaKind::LineComment].
    ///
    /// Includes the `///` prefix up to (but not including) the trailing newline.
    DocComment,

    /// A line that contains only whitespace (spaces, tabs, `\r`) between two
    /// newline characters (or between start-of-file and a newline, or between a
    /// newline and end-of-file).
    ///
    /// `text` is always the empty string for blank lines.
    BlankLine,
}

/// A single trivia record captured during parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trivia {
    /// Classification of this trivia item.
    pub kind: TriviaKind,

    /// Byte offset of this record in the **original** (pre-comment-strip) source.
    ///
    /// For comments this is the offset of the first `/`.
    /// For blank lines this is the offset of the `\n` that opens the blank line
    /// (i.e. the newline immediately preceding the whitespace-only content), or
    /// `0` when the blank line occurs at the very start of the file.
    pub byte_offset: usize,

    /// Raw text of the comment, including the `//` or `///` prefix.
    ///
    /// For [`BlankLine`][TriviaKind::BlankLine] this is always `""`.
    pub text: String,
}

/// An ordered list of trivia records collected during a single parse.
///
/// Records are emitted in source order (ascending `byte_offset`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TriviaStream(pub Vec<Trivia>);

/// Internal accumulator threaded through `strip_comments_with_trivia` and
/// `P::skip_ws_and_newlines_trivia`.
///
/// Not part of the public API; callers use [`TriviaStream`] via
/// [`parse_with_trivia`][super::parse_with_trivia].
pub(super) struct TriviaCollector {
    records: Vec<Trivia>,
}

impl TriviaCollector {
    pub(super) fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, trivia: Trivia) {
        self.records.push(trivia);
    }

    pub(super) fn into_stream(self) -> TriviaStream {
        TriviaStream(self.records)
    }
}

/// Strip single-line comments (`// ...`) from `input`, optionally collecting
/// trivia into `collector`.
///
/// Returns `(stripped, offset_map)` where:
/// - `stripped` is the comment-stripped source (same line count as `input`).
/// - `offset_map[i]` is the byte offset in `input` corresponding to byte `i`
///   in `stripped`.  Only built when `collector.is_some()`; otherwise returns
///   an empty `Vec`.
///
/// Correctly handles `//` inside string literals (does not strip them).
///
/// `////...` is treated as a regular line comment (not a doc-comment).  Only
/// `///` followed by a non-`/` byte (or end-of-line) is a doc-comment.
pub(super) fn strip_comments_with_trivia(
    input: &str,
    collector: &mut Option<TriviaCollector>,
) -> (String, Vec<usize>) {
    // When no collector requested, take the fast path (no offset tracking).
    if collector.is_none() {
        let stripped = strip_comments_fast(input);
        return (stripped, Vec::new());
    }

    let col = collector.as_mut().unwrap();
    let mut stripped_bytes: Vec<u8> = Vec::with_capacity(input.len());
    let mut offset_map: Vec<usize> = Vec::with_capacity(input.len());

    // Walk through the input line by line.  We track both the input byte
    // cursor (for original offsets) and build stripped output with the map.
    let input_bytes = input.as_bytes();
    let mut orig_pos: usize = 0; // cursor in `input`

    // `.lines()` skips the trailing newline which makes rejoining tricky; we
    // instead iterate manually so we control newline handling.
    // Stop when all input bytes have been consumed (do not process the
    // phantom empty "line" after a trailing newline).
    while orig_pos < input_bytes.len() {
        // Find end of this line (position of '\n', or end of input).
        let line_start = orig_pos;
        let newline_pos = input_bytes[orig_pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|rel| orig_pos + rel)
            .unwrap_or(input_bytes.len());

        let line_bytes = &input_bytes[line_start..newline_pos];

        // Scan the line for the first `//` that is not inside a string.
        let comment_col = find_comment_start(line_bytes);

        // The "content" portion of the line (before any comment).
        let content_end = comment_col.unwrap_or(line_bytes.len());
        let content_bytes = &line_bytes[..content_end];

        // Blank line detection: a line whose ENTIRE original content (the full
        // `line_bytes`, including any comment text) is entirely whitespace.
        // A comment-only line like `// foo` has `/` in it and is NOT blank.
        // Only `""` or `"   "` (whitespace-only) lines qualify.
        //
        // The byte_offset for a blank line is `line_start` — the original-source
        // position of the first byte of the blank line's content (the `\n` that
        // terminates the blank line when the content is empty, which is also the
        // start offset of the blank line in the original source).
        //
        // Note: the very first line in a file with blank content gets
        // byte_offset = 0 (line_start = 0).
        if is_blank_line(line_bytes) {
            col.push(Trivia {
                kind: TriviaKind::BlankLine,
                byte_offset: line_start,
                text: String::new(),
            });
        }

        // Append content bytes and build offset map entries.
        for rel in 0..content_end {
            stripped_bytes.push(content_bytes[rel]);
            offset_map.push(line_start + rel);
        }

        // If there is a comment, record it.
        if let Some(col_idx) = comment_col {
            let comment_start_orig = line_start + col_idx;
            let comment_bytes = &line_bytes[col_idx..];
            let comment_text =
                std::str::from_utf8(comment_bytes).unwrap_or("").to_string();

            let kind = classify_comment(comment_bytes);
            col.push(Trivia {
                kind,
                byte_offset: comment_start_orig,
                text: comment_text,
            });
        }

        // Emit the newline (if present) into stripped and map it.
        if newline_pos < input_bytes.len() {
            stripped_bytes.push(b'\n');
            offset_map.push(newline_pos);
            orig_pos = newline_pos + 1;
        } else {
            // End of input reached; stop.
            break;
        }
    }

    let stripped = String::from_utf8(stripped_bytes)
        .expect("stripped source must be valid UTF-8");
    (stripped, offset_map)
}

/// Fast path for `strip_comments` with no trivia collection.
/// Matches the behaviour of the original `strip_comments` exactly.
fn strip_comments_fast(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let bytes = line.as_bytes();
            match find_comment_start(bytes) {
                Some(i) => &line[..i],
                None => line,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Find the byte index within `line` of the first `//` that is not inside a
/// double-quoted string.  Returns `None` if the line contains no such comment.
pub(super) fn find_comment_start(line: &[u8]) -> Option<usize> {
    let mut in_string = false;
    let mut i = 0;
    while i < line.len() {
        if in_string {
            if line[i] == b'\\' && i + 1 < line.len() {
                i += 2;
                continue;
            }
            if line[i] == b'"' {
                in_string = false;
            }
        } else if line[i] == b'"' {
            in_string = true;
        } else if line[i] == b'/' && i + 1 < line.len() && line[i + 1] == b'/' {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Classify comment bytes (starting with `//`) as `DocComment` or
/// `LineComment`.
///
/// Rule: `///` where the fourth byte (index 3) is NOT `/` (or the comment is
/// exactly 3 bytes long) → `DocComment`.  All other `//`-prefixed text →
/// `LineComment`.
fn classify_comment(bytes: &[u8]) -> TriviaKind {
    // Must start with `//`; check for third `/`.
    if bytes.len() >= 3 && bytes[2] == b'/' {
        // Potential doc-comment.  The fourth byte must be non-`/` (or absent).
        let fourth_is_slash = bytes.get(3) == Some(&b'/');
        if !fourth_is_slash {
            return TriviaKind::DocComment;
        }
    }
    TriviaKind::LineComment
}

/// Detect whether a sequence of bytes (a single line's content after
/// comment-stripping) is entirely whitespace.
///
/// Returns `true` for empty slices, and for slices containing only
/// `' '`, `'\t'`, or `'\r'`.
pub(super) fn is_blank_line(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\r')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_line_comment() {
        assert_eq!(classify_comment(b"// hello"), TriviaKind::LineComment);
        assert_eq!(classify_comment(b"//no space"), TriviaKind::LineComment);
        assert_eq!(classify_comment(b"////"), TriviaKind::LineComment);
        assert_eq!(classify_comment(b"////doc"), TriviaKind::LineComment);
    }

    #[test]
    fn classify_doc_comment() {
        assert_eq!(classify_comment(b"/// doc"), TriviaKind::DocComment);
        assert_eq!(classify_comment(b"///"), TriviaKind::DocComment);
        assert_eq!(classify_comment(b"/// "), TriviaKind::DocComment);
    }

    #[test]
    fn find_comment_skips_string() {
        assert_eq!(find_comment_start(b"let s = \"// not a comment\";"), None);
        assert_eq!(find_comment_start(b"let s = \"x\"; // real"), Some(13));
    }

    #[test]
    fn blank_line_detection() {
        assert!(is_blank_line(b""));
        assert!(is_blank_line(b"   "));
        assert!(is_blank_line(b"\t\r"));
        assert!(!is_blank_line(b"  x  "));
    }

    #[test]
    fn strip_preserves_line_count() {
        let src = "fn foo() {\n    // comment\n    let x = 1;\n}\n";
        let (stripped, _) = strip_comments_with_trivia(src, &mut None);
        assert_eq!(
            src.lines().count(),
            stripped.lines().count(),
            "line count must be preserved"
        );
    }

    #[test]
    fn strip_with_trivia_collects_comments() {
        let src = "// line comment\n/// doc\nlet x = 1; // trailing\n";
        let mut col = Some(TriviaCollector::new());
        let (stripped, map) = strip_comments_with_trivia(src, &mut col);
        let stream = col.unwrap().into_stream();

        assert_eq!(stream.0.len(), 3);
        assert_eq!(stream.0[0].kind, TriviaKind::LineComment);
        assert_eq!(stream.0[0].byte_offset, 0);
        assert_eq!(stream.0[0].text, "// line comment");

        assert_eq!(stream.0[1].kind, TriviaKind::DocComment);
        assert_eq!(stream.0[1].byte_offset, 16);
        assert_eq!(stream.0[1].text, "/// doc");

        assert_eq!(stream.0[2].kind, TriviaKind::LineComment);
        // "let x = 1; " is 11 chars; comment starts at offset 16+8+11 = 35
        // line 3 starts at offset 16 + 8 = 24; "let x = 1; " = 11 bytes → 35
        assert_eq!(stream.0[2].byte_offset, 35);
        assert_eq!(stream.0[2].text, "// trailing");

        // Offset map must cover all stripped bytes
        assert_eq!(map.len(), stripped.len());
        // Every mapped offset must be within the original source
        for &orig in &map {
            assert!(orig < src.len(), "offset {orig} out of bounds");
        }
    }

    #[test]
    fn offset_map_none_is_empty() {
        let src = "let x = 1; // comment\n";
        let (_, map) = strip_comments_with_trivia(src, &mut None);
        assert!(map.is_empty());
    }
}
