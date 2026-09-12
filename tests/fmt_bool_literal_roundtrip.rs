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

//! `true` / `false` must survive the formatter as `true` / `false`.
//!
//! WHAT THIS CAUGHT. The parser desugared a bool literal to `Literal::Int(1)` / `Int(0)`,
//! and the formatter round-trips through that same AST — so `mindc fmt` rewrote
//! `return false;` as `return 0;`. The identifier-loss guard refused to write the file
//! (correctly), with the consequence that **any** source containing a bool literal could
//! never clear `fmt::drift`, and so could never pass `mindc check` at all. Measured
//! 2026-09-11: 15 of 26 MindLLM modules and 8 of 42 stdlib files mention `true`/`false`.
//! The stdlib ones only mention them in COMMENTS, which is why the whole gap went unnoticed
//! — no shipped source exercised the value path.
//!
//! WHY THE ASSERTIONS ARE SHAPED THIS WAY. Each case asserts on the FORMATTED TEXT, not on
//! idempotence. `format(format(src)) == format(src)` held throughout the bug: `0` formats
//! to `0` forever. Idempotence is the wrong instrument for a lossy desugar, and this file
//! exists because the existing idempotence gate could not see the defect.

use libmind::fmt::format_source;
use libmind::project::MindcraftFormatConfig;

fn fmt(src: &str) -> String {
    format_source(src, &MindcraftFormatConfig::default())
        .unwrap_or_else(|e| panic!("format_source failed on:\n{src}\nerror: {e:?}"))
}

/// Every position a bool literal can appear in, each spelled back as written.
///
/// `0`/`1` are asserted ABSENT as well as `true`/`false` present: the failure mode was a
/// substitution, so checking only for the presence of `false` would still pass on output
/// that contained both.
#[test]
fn bool_literals_survive_formatting() {
    let cases: &[(&str, &str)] = &[
        ("trailing expression", "pub fn f() -> bool {\n    false\n}\n"),
        ("explicit return", "pub fn f(x: i64) -> bool {\n    if x < 0 {\n        return true;\n    }\n    false\n}\n"),
        ("let initializer", "pub fn f() -> bool {\n    let b: bool = true;\n    b\n}\n"),
        ("both spellings in one fn", "pub fn f(x: i64) -> bool {\n    if x > 0 {\n        return true;\n    }\n    return false;\n}\n"),
    ];

    for (what, src) in cases {
        let out = fmt(src);
        let want_true = src.contains("true");
        let want_false = src.contains("false");

        assert_eq!(
            out.contains("true"),
            want_true,
            "{what}: `true` presence changed.\n--- in ---\n{src}\n--- out ---\n{out}"
        );
        assert_eq!(
            out.contains("false"),
            want_false,
            "{what}: `false` presence changed.\n--- in ---\n{src}\n--- out ---\n{out}"
        );

        // The specific regression: the desugared integer must not appear in its place.
        for digit in ["return 0;", "return 1;", "= 0;", "= 1;", "\n    0\n", "\n    1\n"] {
            assert!(
                !out.contains(digit),
                "{what}: bool literal was rewritten as an integer ({digit:?} in output).\n\
                 --- in ---\n{src}\n--- out ---\n{out}"
            );
        }
    }
}

/// POSITIVE CONTROL for the assertions above.
///
/// Without this, `bool_literals_survive_formatting` would still pass if `format_source`
/// started returning its input verbatim, or if the digit patterns were simply unmatchable.
/// Here an integer literal IS expected to print as a digit — so the same patterns that must
/// stay absent above must be present here, which proves they can match at all.
#[test]
fn integer_literals_still_print_as_digits() {
    let out = fmt("pub fn f(x: i64) -> i64 {\n    if x < 0 {\n        return 0;\n    }\n    1\n}\n");
    assert!(
        out.contains("return 0;"),
        "positive control failed: an integer literal did not print as a digit.\n{out}"
    );
    assert!(
        !out.contains("false"),
        "an integer literal printed as a bool spelling.\n{out}"
    );
}

/// A bool literal must reach the AST as the INTEGER it always was, and must LOWER.
///
/// THIS IS THE GATE WHOSE ABSENCE SHIPPED A COMPILER PANIC. The first version of this fix
/// introduced a `Literal::Bool` variant. `mindc check` returned 0 and all 1647 tests passed —
/// and `mindc build` PANICKED on any program containing `true`, because `lower_expr` had arms
/// for `Literal::Int` and none for the new variant, so a bool literal hit the fail-closed
/// catch-all. Five further consumers (`opt/fold.rs`, `opt/comptime.rs`, `opt/scev.rs`,
/// `eval/lower.rs`'s narrow-width inference) silently stopped treating a bool as a constant,
/// which changes the optimisation path and therefore the mic@3 digest of such a program.
/// Keystone byte-identity could not see it: the parser records that this path never fires for
/// the keystone self-compile.
///
/// The fix was to stop putting spelling in the shared enum at all — the value stays
/// `Literal::Int`, the spelling lives in `ast::SourceSpelling::bool_literals`. These
/// assertions pin both halves of that, so the variant cannot come back by accident.
#[test]
fn bool_literals_are_integers_in_the_ast_and_lower() {
    use libmind::ast::{Literal, Node};

    let m = libmind::parser::parse("fn main() -> i64 {\n    let b: i64 = true;\n    return 6;\n}\n")
        .expect("parse");

    // The spelling is recorded for the formatter…
    assert_eq!(
        m.spelling.bool_literals.len(),
        1,
        "the bool spelling must be recorded in the side table, or fmt deletes it"
    );
    assert!(m.spelling.bool_literals[0].1, "`true` must be recorded as true");

    // …and the VALUE in the tree is an ordinary integer, which is what every consumer matches.
    let dump = format!("{m:?}");
    assert!(
        dump.contains("Int(1)"),
        "a `true` literal must reach the AST as Int(1); anything else silently changes which \
         consumers treat it as a constant.\n{dump}"
    );

    // AND it must lower. `lower_to_ir` fail-closes by panicking on a node it has no arm for,
    // so reaching this line at all is the assertion.
    let ir = libmind::eval::lower::lower_to_ir(&m);
    assert!(
        !ir.instrs.is_empty(),
        "lowering a program with a bool literal produced no instructions"
    );

    // Positive control: the same program with `1` must lower to the same instruction count —
    // if it does not, the bool path is taking a different route through lowering.
    let m2 = libmind::parser::parse("fn main() -> i64 {\n    let b: i64 = 1;\n    return 6;\n}\n")
        .expect("parse");
    let ir2 = libmind::eval::lower::lower_to_ir(&m2);
    assert_eq!(
        ir.instrs.len(),
        ir2.instrs.len(),
        "`true` and `1` must lower identically — a divergence here is a different optimisation \
         path, and therefore a different mic@3 digest, for a spelling change"
    );
    let _ = Literal::Int(0);
    let _: Option<&Node> = m.items.first();
}
