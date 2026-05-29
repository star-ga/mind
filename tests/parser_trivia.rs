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

//! Round-trip trivia recovery tests for the parallel CST trivia layer.
//!
//! Hard gate: `parse_with_trivia(src).0` must equal `parse(src)` for every
//! input — the AST shape is unchanged by trivia collection.

use libmind::parser::{Trivia, TriviaKind, parse, parse_with_trivia};

/// Convenience: parse a MIND source snippet that is expected to succeed.
fn must_parse(src: &str) -> libmind::ast::Module {
    parse(src).unwrap_or_else(|errs| {
        panic!(
            "parse failed: {:?}\nsrc: {src:?}",
            errs.iter().map(|e| e.message.clone()).collect::<Vec<_>>()
        )
    })
}

// ---------------------------------------------------------------------------
// AST parity: parse_with_trivia.0 == parse
// ---------------------------------------------------------------------------

#[test]
fn ast_parity_simple_expr() {
    let src = "1 + 2 * 3\n";
    let (module_trivia, _stream) = parse_with_trivia(src).unwrap();
    let module_plain = must_parse(src);
    assert_eq!(
        module_trivia, module_plain,
        "AST must be identical whether or not trivia is collected"
    );
}

#[test]
fn ast_parity_with_comments() {
    let src = "// header\nlet x = 1; // trailing\n/// doc\nlet y = 2;\n";
    let (module_trivia, _) = parse_with_trivia(src).unwrap();
    let module_plain = must_parse(src);
    assert_eq!(module_trivia, module_plain);
}

#[test]
fn ast_parity_with_blank_lines() {
    let src = "let x = 1;\n\nlet y = 2;\n\n\nlet z = 3;\n";
    let (module_trivia, _) = parse_with_trivia(src).unwrap();
    let module_plain = must_parse(src);
    assert_eq!(module_trivia, module_plain);
}

#[test]
fn ast_parity_empty_source() {
    let src = "";
    let (module_trivia, stream) = parse_with_trivia(src).unwrap();
    let module_plain = must_parse(src);
    assert_eq!(module_trivia, module_plain);
    assert!(stream.0.is_empty());
}

// ---------------------------------------------------------------------------
// Comment capture: byte offsets and text
// ---------------------------------------------------------------------------

#[test]
fn line_comment_at_start_of_file() {
    let src = "// hello world\nlet x = 1;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let comments: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::LineComment)
        .collect();

    assert_eq!(comments.len(), 1, "expected exactly one line comment");
    let c = &comments[0];
    assert_eq!(c.byte_offset, 0, "// at start of file → offset 0");
    assert_eq!(c.text, "// hello world");
}

#[test]
fn trailing_line_comment_offset() {
    // "let x = 1; // trailing\n"
    //  0123456789012345678
    //  "let x = 1; " = 11 bytes → comment starts at offset 11
    let src = "let x = 1; // trailing\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let comments: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::LineComment)
        .collect();

    assert_eq!(comments.len(), 1);
    assert_eq!(comments[0].byte_offset, 11);
    assert_eq!(comments[0].text, "// trailing");
}

#[test]
fn doc_comment_recognised() {
    let src = "/// A doc comment\nlet x = 1;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let docs: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::DocComment)
        .collect();

    assert_eq!(docs.len(), 1, "expected exactly one doc comment");
    assert_eq!(docs[0].byte_offset, 0);
    assert_eq!(docs[0].text, "/// A doc comment");
}

#[test]
fn quadruple_slash_is_line_comment_not_doc() {
    // `////` must be classified as LineComment, not DocComment.
    let src = "//// not a doc comment\nlet x = 1;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let kinds: Vec<&TriviaKind> = stream.0.iter().map(|t| &t.kind).collect();
    assert!(
        !kinds.contains(&&TriviaKind::DocComment),
        "`////` must not be classified as DocComment; got: {kinds:?}"
    );
    assert!(
        kinds.contains(&&TriviaKind::LineComment),
        "`////` must be classified as LineComment; got: {kinds:?}"
    );
}

#[test]
fn triple_slash_no_space_is_doc() {
    // `///x` — no space after `///` — still a doc comment.
    let src = "///doc\nlet x = 1;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let docs: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::DocComment)
        .collect();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].text, "///doc");
}

#[test]
fn multiple_comments_in_order() {
    let src = "// first\n/// second\nlet x = 1; // third\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let comments: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind != TriviaKind::BlankLine)
        .collect();

    assert_eq!(comments.len(), 3, "expected three comments");

    // Records must be in source order (ascending byte_offset).
    let offsets: Vec<usize> = comments.iter().map(|t| t.byte_offset).collect();
    let mut sorted = offsets.clone();
    sorted.sort_unstable();
    assert_eq!(offsets, sorted, "trivia records must be in source order");

    assert_eq!(comments[0].kind, TriviaKind::LineComment);
    assert_eq!(comments[0].text, "// first");

    assert_eq!(comments[1].kind, TriviaKind::DocComment);
    assert_eq!(comments[1].text, "/// second");

    assert_eq!(comments[2].kind, TriviaKind::LineComment);
    assert_eq!(comments[2].text, "// third");
}

// ---------------------------------------------------------------------------
// Comment inside string literal: must NOT be captured as trivia
// ---------------------------------------------------------------------------

#[test]
fn comment_lookalike_inside_string_not_captured() {
    let src = "let s = \"// not a comment\";\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let comments: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind != TriviaKind::BlankLine)
        .collect();
    assert!(
        comments.is_empty(),
        "`//` inside a string must not be captured as trivia; got: {comments:?}"
    );
}

// ---------------------------------------------------------------------------
// Blank line capture: byte offsets
// ---------------------------------------------------------------------------

#[test]
fn blank_line_between_items() {
    // The blank line between `let x` and `let y` must appear in the stream.
    let src = "let x = 1;\n\nlet y = 2;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let blanks: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::BlankLine)
        .collect();

    assert_eq!(blanks.len(), 1, "expected exactly one blank line");
    assert_eq!(blanks[0].text, "", "blank line text must be empty");
    // "let x = 1;\n" = 11 bytes (0-10 content + newline at 10).
    // The blank line starts at offset 11 (the second \n in "\n\n").
    assert_eq!(blanks[0].byte_offset, 11);
}

#[test]
fn whitespace_only_line_is_blank() {
    // A line with only spaces is a blank line.
    let src = "let x = 1;\n   \nlet y = 2;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let blanks: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::BlankLine)
        .collect();

    assert_eq!(
        blanks.len(),
        1,
        "space-only line must be recorded as BlankLine"
    );
}

#[test]
fn multiple_consecutive_blank_lines() {
    let src = "let x = 1;\n\n\nlet y = 2;\n";
    let (_, stream) = parse_with_trivia(src).unwrap();

    let blanks: Vec<&Trivia> = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::BlankLine)
        .collect();

    assert_eq!(blanks.len(), 2, "two consecutive blank lines = two records");
}

// ---------------------------------------------------------------------------
// Comment-heavy source: comprehensive round-trip
// ---------------------------------------------------------------------------

#[test]
fn comment_heavy_source_round_trip() {
    // A realistic snippet with all three trivia kinds.
    let src = concat!(
        "/// Module doc comment\n",
        "// Copyright notice\n",
        "\n",
        "let foo = 1;\n",
        "\n",
        "// Single line\n",
        "let bar = foo + 1; // inline\n",
        "\n",
        "/// Another doc\n",
        "let baz = bar * 2;\n",
    );

    let (module_trivia, stream) = parse_with_trivia(src).unwrap();
    let module_plain = must_parse(src);

    // Gate 1: AST parity.
    assert_eq!(module_trivia, module_plain, "AST must be identical");

    // Gate 2: expected trivia counts.
    let line_comments = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::LineComment)
        .count();
    let doc_comments = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::DocComment)
        .count();
    let blank_lines = stream
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::BlankLine)
        .count();

    assert_eq!(line_comments, 3, "expected 3 line comments");
    assert_eq!(doc_comments, 2, "expected 2 doc comments");
    assert_eq!(blank_lines, 3, "expected 3 blank lines");

    // Gate 3: all byte_offsets are within the original source.
    for t in &stream.0 {
        assert!(
            t.byte_offset < src.len(),
            "byte_offset {} out of range for source of len {}",
            t.byte_offset,
            src.len()
        );
    }

    // Gate 4: byte_offsets are in ascending order (source order).
    let offsets: Vec<usize> = stream.0.iter().map(|t| t.byte_offset).collect();
    let mut sorted = offsets.clone();
    sorted.sort_unstable();
    assert_eq!(offsets, sorted, "trivia must be emitted in source order");

    // Gate 5: for each comment, the source slice at byte_offset matches text.
    for t in stream.0.iter().filter(|t| t.kind != TriviaKind::BlankLine) {
        let slice = &src[t.byte_offset..t.byte_offset + t.text.len()];
        assert_eq!(
            slice, t.text,
            "source slice at byte_offset must match recorded text"
        );
    }
}
