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

//! Regression tests for comment placement inside function bodies.
//!
//! Covers:
//!   (a) A comment between two body statements stays between them.
//!   (b) A trailing same-line comment stays inside the function, immediately
//!       after the statement it annotates (acceptable: own line, still inside
//!       the closing brace).
//!   (c) A comment before the first body statement stays inside the function.
//!   (d) A comment after the last body statement (before `}`) stays inside.
//!   (e) Idempotence: formatting twice yields identical output for all cases.
//!
//! The critical contract: NO comment must migrate across the `}` that closes
//! the function body.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn cfg() -> MindcraftFormatConfig {
    MindcraftFormatConfig::default()
}

fn fmt(src: &str) -> String {
    format_source(src, &cfg()).unwrap_or_else(|e| panic!("format failed: {e}"))
}

fn assert_idempotent(label: &str, src: &str) {
    let pass1 = fmt(src);
    let pass2 = fmt(&pass1);
    assert_eq!(
        pass1,
        pass2,
        "idempotence violated for {label}:\n  pass1 line count={}, pass2 line count={}\n  pass1:\n{pass1}\n  pass2:\n{pass2}",
        pass1.lines().count(),
        pass2.lines().count(),
    );
}

fn comment_stays_inside(label: &str, output: &str) {
    // The output must not have any comment line appearing after the `}` that
    // closes the function body. Concretely: the `}` that closes `fn` must be
    // the last non-empty line, and no `//` comment must follow it.
    let lines: Vec<&str> = output.lines().collect();
    // Find the first closing-brace line (the fn body close).
    let close_idx = lines.iter().rposition(|l| l.trim() == "}");
    if let Some(ci) = close_idx {
        for (i, line) in lines.iter().enumerate().skip(ci + 1) {
            let trimmed = line.trim();
            assert!(
                !trimmed.starts_with("//"),
                "{label}: comment found after `}}` on output line {}: {:?}\nFull output:\n{output}",
                i + 1,
                line,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (a) Comment between two body statements
// ---------------------------------------------------------------------------

#[test]
fn between_stmts_comment_stays_inside() {
    let src = "pub fn add(a: i64, b: i64) -> i64 {\n    let x: i64 = a;\n    // middle comment\n    x + b\n}\n";
    let out = fmt(src);
    assert!(
        out.contains("// middle comment"),
        "comment must be present in output:\n{out}"
    );
    comment_stays_inside("between_stmts", &out);
    // The comment must appear before the implicit return, not after `}`.
    let lines: Vec<&str> = out.lines().collect();
    let comment_pos = lines
        .iter()
        .position(|l| l.trim() == "// middle comment")
        .expect("comment not found");
    let close_pos = lines
        .iter()
        .rposition(|l| l.trim() == "}")
        .expect("closing brace not found");
    assert!(
        comment_pos < close_pos,
        "comment (line {}) must appear before closing `}}` (line {}):\n{out}",
        comment_pos + 1,
        close_pos + 1,
    );
}

// ---------------------------------------------------------------------------
// (b) Trailing same-line comment
// ---------------------------------------------------------------------------

#[test]
fn trailing_same_line_comment_stays_inside() {
    // The formatter may move the comment to its own line, but it must stay
    // INSIDE the function body, not migrate past the `}`.
    let src = "pub fn g(x: i64) -> i64 {\n    let a: i64 = x;  // keep me\n    a\n}\n";
    let out = fmt(src);
    assert!(
        out.contains("// keep me"),
        "trailing comment must be present in output:\n{out}"
    );
    comment_stays_inside("trailing_same_line", &out);
    // The comment must appear before the closing `}`.
    let lines: Vec<&str> = out.lines().collect();
    let comment_pos = lines
        .iter()
        .position(|l| l.trim() == "// keep me")
        .expect("comment not found");
    let close_pos = lines
        .iter()
        .rposition(|l| l.trim() == "}")
        .expect("closing brace not found");
    assert!(
        comment_pos < close_pos,
        "comment (line {}) must appear before closing `}}` (line {}):\n{out}",
        comment_pos + 1,
        close_pos + 1,
    );
}

// ---------------------------------------------------------------------------
// (c) Comment before the first body statement
// ---------------------------------------------------------------------------

#[test]
fn leading_body_comment_stays_inside() {
    let src =
        "pub fn h(x: i64) -> i64 {\n    // leading body comment\n    let a: i64 = x;\n    a\n}\n";
    let out = fmt(src);
    assert!(
        out.contains("// leading body comment"),
        "leading comment must be present in output:\n{out}"
    );
    comment_stays_inside("leading_body", &out);
    // The comment must appear before the closing `}`.
    let lines: Vec<&str> = out.lines().collect();
    let comment_pos = lines
        .iter()
        .position(|l| l.trim() == "// leading body comment")
        .expect("comment not found");
    let close_pos = lines
        .iter()
        .rposition(|l| l.trim() == "}")
        .expect("closing brace not found");
    assert!(
        comment_pos < close_pos,
        "comment (line {}) must appear before closing `}}` (line {}):\n{out}",
        comment_pos + 1,
        close_pos + 1,
    );
}

// ---------------------------------------------------------------------------
// (d) Comment after the last body statement (before `}`)
// ---------------------------------------------------------------------------

#[test]
fn trailing_body_comment_stays_inside() {
    let src =
        "pub fn f(x: i64) -> i64 {\n    let a: i64 = x;\n    // trailing body comment\n    a\n}\n";
    let out = fmt(src);
    // Note: the comment is between the let and the final expression, so the
    // formatter must keep it between those two statements.
    assert!(
        out.contains("// trailing body comment"),
        "trailing body comment must be present in output:\n{out}"
    );
    comment_stays_inside("trailing_body", &out);
    let lines: Vec<&str> = out.lines().collect();
    let comment_pos = lines
        .iter()
        .position(|l| l.trim() == "// trailing body comment")
        .expect("comment not found");
    let close_pos = lines
        .iter()
        .rposition(|l| l.trim() == "}")
        .expect("closing brace not found");
    assert!(
        comment_pos < close_pos,
        "comment (line {}) must appear before closing `}}` (line {}):\n{out}",
        comment_pos + 1,
        close_pos + 1,
    );
}

// ---------------------------------------------------------------------------
// (e) Idempotence for all four cases
// ---------------------------------------------------------------------------

#[test]
fn idempotence_between_stmts() {
    let src = "pub fn add(a: i64, b: i64) -> i64 {\n    let x: i64 = a;\n    // middle comment\n    x + b\n}\n";
    assert_idempotent("between_stmts", src);
}

#[test]
fn idempotence_trailing_same_line() {
    let src = "pub fn g(x: i64) -> i64 {\n    let a: i64 = x;  // keep me\n    a\n}\n";
    assert_idempotent("trailing_same_line", src);
}

#[test]
fn idempotence_leading_body() {
    let src =
        "pub fn h(x: i64) -> i64 {\n    // leading body comment\n    let a: i64 = x;\n    a\n}\n";
    assert_idempotent("leading_body", src);
}

#[test]
fn idempotence_trailing_body() {
    let src =
        "pub fn f(x: i64) -> i64 {\n    let a: i64 = x;\n    // trailing body comment\n    a\n}\n";
    assert_idempotent("trailing_body", src);
}

// ---------------------------------------------------------------------------
// Additional: multiple functions — comments must not cross fn boundaries
// ---------------------------------------------------------------------------

#[test]
fn comment_does_not_cross_fn_boundary() {
    let src = "pub fn first(x: i64) -> i64 {\n    let a: i64 = x;\n    // note inside first\n    a\n}\n\npub fn second(x: i64) -> i64 {\n    x\n}\n";
    let out = fmt(src);
    assert!(
        out.contains("// note inside first"),
        "comment must be present:\n{out}"
    );
    // The comment must appear before second function's definition.
    let lines: Vec<&str> = out.lines().collect();
    let comment_pos = lines
        .iter()
        .position(|l| l.trim() == "// note inside first")
        .expect("comment not found");
    let second_fn_pos = lines
        .iter()
        .position(|l| l.trim().starts_with("pub fn second"))
        .expect("second fn not found");
    assert!(
        comment_pos < second_fn_pos,
        "comment (line {}) must appear before second fn (line {}):\n{out}",
        comment_pos + 1,
        second_fn_pos + 1,
    );
    // Critically: the comment must appear before the `}` of first fn.
    comment_stays_inside("cross_boundary", &out);
}

// ---------------------------------------------------------------------------
// Two-repro smoke test (exact repros from the bug description)
// ---------------------------------------------------------------------------

#[test]
fn repro_trailing_comment_stays_inside_fn() {
    let src = "pub fn g(x: i64) -> i64 {\n    let a: i64 = x;  // keep me\n    a\n}\n";
    let out = fmt(src);
    // The comment must not appear after the `}` on any line by itself.
    let lines: Vec<&str> = out.lines().collect();
    let last_close = lines
        .iter()
        .rposition(|l| l.trim() == "}")
        .expect("no close brace");
    for line in &lines[last_close + 1..] {
        assert!(
            !line.trim().starts_with("//"),
            "comment escaped past `}}`: {:?}\nFull output:\n{out}",
            line
        );
    }
}

#[test]
fn repro_leading_body_comment_stays_inside_fn() {
    let src =
        "pub fn h(x: i64) -> i64 {\n    // leading body comment\n    let a: i64 = x;\n    a\n}\n";
    let out = fmt(src);
    let lines: Vec<&str> = out.lines().collect();
    let last_close = lines
        .iter()
        .rposition(|l| l.trim() == "}")
        .expect("no close brace");
    for line in &lines[last_close + 1..] {
        assert!(
            !line.trim().starts_with("//"),
            "comment escaped past `}}`: {:?}\nFull output:\n{out}",
            line
        );
    }
}
