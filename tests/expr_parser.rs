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

use libmind::eval;
use libmind::parser;

#[test]
fn precedence_and_parens() {
    let module = parser::parse("1 + 2 * 3").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);

    let module = parser::parse("(1 + 2) * 3").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 9);
}

#[test]
fn division_and_zero_guard() {
    let module = parser::parse("8 / 2").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 4);

    // Integer division by zero is a VALUE, not an error: `x / 0 == 0`. That is the
    // language contract both compiled backends implement — the native emitter's
    // `nb_div_guarded` zero-guard (pinned by examples/mindc_mind/div_shift_cmp_edge_smoke.py)
    // and the MLIR `div_zero_guard` (tests/narrow_unsigned_div_zero_run.rs) — and the
    // interpreter now honours it too. This assertion used to require an error, which
    // pinned the interpreter DISAGREEING with the artifact it models.
    let module = parser::parse("1 / 0").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 0);
}

/// A `!` welded to an identifier is a macro invocation, and MIND has no macros.
///
/// Before this was rejected the front-end RECOVERED the shape — `format` became
/// a bare expression statement and `!("{}", 1)` a second, unary-not statement,
/// because MIND statements need no separator. Every consumer then saw a program
/// nobody wrote; `mindc fmt` rewrote such a file in place and exited 0.
#[test]
fn macro_invocation_is_rejected_not_recovered() {
    for src in [
        "fn f() { let c = format!(\"{}\", 1); }",
        "fn f() { let c = vec![1, 2]; }",
        "fn f() { let c = m!{1}; }",
        "fn f() { let c = std.fmt.format!(\"{}\", 1); }",
    ] {
        let errs = parser::parse(src).expect_err(&format!("must not recover: {src:?}"));
        assert!(
            errs.iter().any(|e| e.message.contains("macro invocation")),
            "diagnostic must name the construct, got {:?} for {src:?}",
            errs.iter().map(|e| e.message.clone()).collect::<Vec<_>>()
        );
    }
}

/// The rule is adjacency and nothing else. `!` is a prefix operator binding at
/// atom precedence, so `!( … )` is the ONLY spelling for negating a compound
/// expression — rejecting it would break ordinary source. Keywords never reach
/// the identifier arm, and `!=` is the two-byte comparison operator.
#[test]
fn unary_not_and_not_equal_still_parse() {
    for src in [
        "fn f(a: bool, b: bool) -> bool { return !(a && b); }",
        "fn f(a: bool, b: bool) -> i64 { if !(a && b) { return 1; } return 0; }",
        "fn f(a: bool, b: bool) -> bool { let c = !(a || b); return c; }",
        "fn f(a: i64, b: i64) -> bool { return a != b; }",
        "fn f(a: i64, b: i64) -> bool { return a!=b; }",
    ] {
        parser::parse(src).unwrap_or_else(|e| {
            panic!(
                "valid source must parse: {src:?}: {:?}",
                e.iter().map(|x| x.message.clone()).collect::<Vec<_>>()
            )
        });
    }
}

#[test]
fn unary_not_in_while_respects_surface_profile() {
    let src = "fn f(a: bool, b: bool) -> i64 { while !(a && b) { return 1; } return 0; }";
    let parsed = parser::parse(src);
    #[cfg(feature = "std-surface")]
    assert!(
        parsed.is_ok(),
        "while must parse with std-surface: {parsed:?}"
    );
    #[cfg(not(feature = "std-surface"))]
    assert!(
        parsed.is_err(),
        "bare grammar must refuse the while fixture"
    );
}
