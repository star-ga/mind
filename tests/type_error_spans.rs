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

use libmind::diagnostics;

use libmind::parser;

use libmind::type_checker;

#[test]
fn unknown_ident_points_to_name() {
    let src = "let n: i32 = x + 1";
    let module = parser::parse_with_diagnostics(src).expect("parse failed");
    let diags = type_checker::check_module_types(&module, src, &Default::default());
    assert!(!diags.is_empty(), "expected type error diagnostic");
    let rendered = diagnostics::render(src, &diags[0]);
    assert!(
        rendered.contains("x + 1"),
        "diagnostic missing offending line: {rendered}"
    );
    let line = "let n: i32 = x + 1";
    let x_idx = line.find('x').unwrap();
    let caret_line = rendered.lines().last().unwrap_or("");
    let caret_pos = caret_line.find('^').unwrap_or(usize::MAX);
    assert!(
        caret_pos >= x_idx.saturating_sub(2),
        "caret not near identifier: {rendered}"
    );
}

/// 1-based `(line, column)` of a byte offset — mirrors the diagnostic renderer
/// for pure-ASCII source (which these fixtures are).
fn byte_to_line_col(src: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in src.char_indices() {
        if i == offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Assert the E2004 implicit-narrowing diagnostic points at the value `big` in
/// `let small: i32 = big` — the real offending token — not a comment or the
/// `let` keyword.
fn assert_narrowing_points_at_value(src: &str) {
    let module = parser::parse_with_diagnostics(src).expect("parse failed");
    let diags = type_checker::check_module_types(&module, src, &Default::default());
    let narrowing = diags
        .iter()
        .find(|d| d.code == "E2004")
        .expect("expected E2004 narrowing diagnostic");
    let span = narrowing
        .span
        .as_ref()
        .expect("diagnostic must carry a span");

    // The offending value is the `big` immediately after `= ` in the assignment.
    let assign_idx = src.find("= big").expect("source must contain `= big`");
    let big_idx = assign_idx + "= ".len();
    let (exp_line, exp_col) = byte_to_line_col(src, big_idx);
    assert_eq!(
        (span.line, span.column),
        (exp_line, exp_col),
        "narrowing diagnostic must point at `big` (the value); got line {}, col {}",
        span.line,
        span.column
    );
}

/// Regression (#23 prerequisite): a comment appearing before the offending
/// statement must NOT shift the diagnostic column. Comment stripping blanks
/// comments with spaces (preserving byte offsets) rather than deleting them, so
/// token spans still index the original source. Before the fix, each preceding
/// comment shifted the reported column earlier by its byte length — eventually
/// landing the caret on a comment line far from the real token.
#[test]
fn comment_before_token_preserves_span_module_level() {
    // Without the comment the diagnostic correctly points at `big`; the comment
    // on the line above must not move it.
    let src = "let big: i64 = 4294967297;\n// a comment\nlet small: i32 = big;\nsmall\n";
    assert_narrowing_points_at_value(src);
}

#[test]
fn comment_before_token_preserves_span_fn_body() {
    // Same property inside a function body (the path the #23 audit flagged).
    let src = "fn f() -> i32 {\n    // comment line 2\n    let big: i64 = 4294967297;\n    let small: i32 = big;\n    small\n}\n";
    assert_narrowing_points_at_value(src);
}
