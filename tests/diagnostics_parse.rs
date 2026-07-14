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

use libmind::parser;

#[test]
fn shows_pretty_error_for_unexpected_paren() {
    let src = ")";
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected error");
    };
    let joined = diags
        .iter()
        .map(|d| libmind::diagnostics::render(src, d))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("error"));
    assert!(joined.contains("line 1"));
    assert!(joined.contains("^")); // caret present
}

#[test]
fn shows_error_for_unclosed_paren() {
    let src = "(";
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected error");
    };
    let s = libmind::diagnostics::render(src, &diags[0]);
    assert!(s.contains("line 1"));
    assert!(s.contains("^"));
}

// Issue #200: an unsupported prefix token must be reported AT that token's own
// span, naming the token — never unwound to an earlier expression checkpoint in
// a different, well-formed function. The `@` below sits on line 6, column 5.
#[test]
fn unsupported_prefix_token_reports_at_token_span() {
    let src = "fn aaa() -> i64 {\n    let a = 1\n    a\n}\nfn bbb() -> i64 {\n    @foo\n}\n";
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected a parse error");
    };
    let span = diags[0]
        .span
        .as_ref()
        .expect("diagnostic must carry a span");
    assert_eq!(span.line, 6, "error must point at the `@` line, not unwind");
    assert_eq!(span.column, 5, "error must point at the `@` column");
    assert!(
        diags[0].message.contains("unexpected `@`"),
        "message should name the offending token, got: {}",
        diags[0].message
    );
}

// Issue #200: same guarantee inside a deeper `else if` block-expression — the
// historical failure mode re-reported at an earlier `else if` checkpoint. The
// `@` below is on line 7, column 9.
#[test]
fn unsupported_prefix_in_else_if_branch_reports_at_token() {
    let src = concat!(
        "fn f(n: i64) -> i64 {\n",
        "    if n == 0 {\n",
        "        10\n",
        "    } else if n == 1 {\n",
        "        20\n",
        "    } else if n == 2 {\n",
        "        @bad\n",
        "    } else {\n",
        "        40\n",
        "    }\n",
        "}\n",
    );
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected a parse error");
    };
    let span = diags[0].span.as_ref().expect("span");
    assert_eq!(span.line, 7);
    assert_eq!(span.column, 9);
    assert!(diags[0].message.contains("unexpected `@`"));
}

// A duplicate enum variant name must be rejected loud at parse. Without this the
// resolver collects variants into a `BTreeSet`, which silently absorbs the dup —
// leaving the enum with fewer distinct tags than written (a fail-open the
// downstream discriminant / niceness machinery can only catch at runtime).
#[test]
fn rejects_duplicate_enum_variant() {
    let src = "enum Currency {\n    AUD,\n    JPY,\n    AUD,\n    USD,\n}\n";
    let Err(diags) = parser::parse_with_diagnostics(src) else {
        panic!("expected a parse error for the duplicate variant");
    };
    let joined = diags
        .iter()
        .map(|d| d.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("duplicate enum variant") && joined.contains("AUD"),
        "expected a duplicate-variant diagnostic naming `AUD`, got: {joined}"
    );
}

// A well-formed enum with distinct variant names must still parse cleanly.
#[test]
fn accepts_distinct_enum_variants() {
    let src = "enum Currency {\n    AUD,\n    JPY,\n    USD,\n}\n";
    assert!(
        parser::parse_with_diagnostics(src).is_ok(),
        "a distinct-variant enum must parse without error"
    );
}
