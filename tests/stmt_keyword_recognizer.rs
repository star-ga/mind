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

// Part of the MIND project (Machine Intelligence Native Design).

//! Equivalence + determinism gate for the `parse_stmt` statement-keyword
//! recogniser (the compile-time perfect hash that replaced the sequential
//! `at_keyword(b"…")` ladder).
//!
//! The recogniser itself is private to `src/parser`, so this suite gates it
//! through the public `parser::parse` surface — the only thing that actually
//! matters: the recogniser must dispatch EXACTLY the statements the ladder did,
//! and must reject every near-miss identifier the ladder rejected.
//!
//! DETERMINISM RED LINE. The discriminator is `(word length, word[0], word[2])`
//! — a fixed structural function of the key bytes, textually pinned in the
//! source. There is no seed, no search, no RNG, no clock, no hostname, no
//! `HashMap` iteration, no pointer value and no `-march` dependence anywhere in
//! it, so the same source selects the same statement parser on x86 and on ARM.
//! `recognizer_selection_is_a_pure_function_of_the_key_set` asserts the
//! observable consequence: repeated parses of the same source, and parses of the
//! same statements in permuted textual order, produce the identical AST debug
//! rendering — i.e. the decision depends on the key and nothing ambient.

use libmind::parser;

/// Every statement-leading keyword in the closed set, with a minimal statement
/// that must still dispatch to its own parser.
const KEYWORD_STMTS: &[(&str, &str)] = &[
    ("import", "import foo\n"),
    ("use", "use foo\n"),
    ("const", "const K: i64 = 1;\n"),
    ("type", "type Alias = i64;\n"),
    ("struct", "struct S { a: i64 }\n"),
    ("enum", "enum E { A, B }\n"),
    ("fn", "fn f() -> i64 { 1 }\n"),
    ("pub fn", "pub fn g() -> i64 { 1 }\n"),
    ("pub struct", "pub struct T { a: i64 }\n"),
    ("pub const", "pub const P: i64 = 2;\n"),
    ("assert", "fn f() -> i64 { assert 1 == 1; 1 }\n"),
    ("return", "fn f() -> i64 { return 3; }\n"),
    ("let", "fn f() -> i64 { let x: i64 = 1; x }\n"),
    ("if", "fn f() -> i64 { if 1 > 0 { 1 } else { 2 } }\n"),
    (
        "while",
        "fn f() -> i64 { let mut i: i64 = 0; while i < 3 { i = i + 1; } i }\n",
    ),
    (
        "loop",
        "fn f() -> i64 { let mut i: i64 = 0; loop { i = i + 1; break; } i }\n",
    ),
    (
        "break/continue",
        "fn f() -> i64 { let mut i: i64 = 0; while i < 9 { i = i + 1; if i > 3 { break; } continue; } i }\n",
    ),
    (
        "for",
        "fn f() -> i64 { let mut s: i64 = 0; for i in 0..6 { s = s + i; } s }\n",
    ),
    ("print", "fn f() -> i64 { print(\"x\"); 1 }\n"),
    ("module", "module m { fn f() -> i64 { 1 } }\n"),
    ("export", "export { checked }\n"),
    ("extern", "extern \"C\" { }\n"),
    ("region", "fn f() -> i64 { region { } 1 }\n"),
    ("invariant", "invariant my_inv { }\n"),
];

#[test]
fn every_statement_keyword_still_dispatches() {
    for (name, src) in KEYWORD_STMTS {
        let parsed = parser::parse(src);
        assert!(
            parsed.is_ok(),
            "statement keyword `{name}` no longer parses: {:?}",
            parsed.err()
        );
        let m = parsed.unwrap();
        assert!(
            !m.items.is_empty(),
            "statement keyword `{name}` produced an empty module"
        );
    }
}

/// The discriminator `(len, w[0], w[2])` is NOT injective over arbitrary
/// identifiers — `expand` lands in the same `(6, b'e')` cell as `export`, `retry`
/// shares `(5, b'r')`-shaped neighbourhoods, and so on. The trailing full-slice
/// compare is what makes the recogniser exact. If it were ever dropped, these
/// identifiers would be mis-dispatched into a keyword parser and the source below
/// would stop compiling — which is precisely what this test detects.
#[test]
fn near_miss_identifiers_are_not_keywords() {
    // Each of these is a plain identifier that COLLIDES with a keyword under the
    // (len, byte0[, byte2]) discriminator alone, plus the prefix-extension cases
    // (`letx`, `iffy`) the word-boundary rule must reject.
    let idents = [
        "expand",     // (6,'e','p') == export's cell
        "extend",     // (6,'e','t') == extern's cell
        "regain",     // (6,'r','g') == region's cell
        "reduce",     // (6,'r','d') -> return's else-branch
        "assign",     // (6,'a')     == assert's cell
        "impose",     // (6,'i')     == import's cell
        "moduli",     // (6,'m')     == module's cell
        "stride",     // (6,'s')     == struct's cell
        "brake",      // (5,'b')     == break's cell
        "count",      // (5,'c')     == const's cell
        "primo",      // (5,'p')     == print's cell
        "whole",      // (5,'w')     == while's cell
        "each",       // (4,'e')     == enum's cell
        "line",       // (4,'l')     == loop's cell
        "tint",       // (4,'t')     == type's cell
        "far",        // (3,'f')     == for's cell
        "lit",        // (3,'l')     == let's cell
        "pun",        // (3,'p')     == pub's cell
        "urn",        // (3,'u')     == use's cell
        "fx",         // (2,'f')     == fn's cell
        "ix",         // (2,'i')     == if's cell
        "letx",       // prefix extension of `let`
        "iffy",       // prefix extension of `if`
        "forge",      // prefix extension of `for`
        "constant_x", // prefix extension of `const`
        "continued",  // prefix extension of `continue`
        "invariants", // prefix extension of `invariant`
    ];
    for id in idents {
        // Used as a plain local variable: if the recogniser mis-fired, the word
        // would be eaten by a keyword parser and this would fail to parse.
        let src = format!("fn f() -> i64 {{ let {id}: i64 = 7; {id} }}\n");
        let parsed = parser::parse(&src);
        assert!(
            parsed.is_ok(),
            "identifier `{id}` was mis-recognised as a statement keyword: {:?}",
            parsed.err()
        );
    }
}

/// DETERMINISM. Selection must be a pure function of the key set (the source
/// bytes) and nothing ambient. Two observable consequences are asserted:
///
///  * re-parsing the SAME source many times yields a bit-identical AST rendering
///    (no clock / RNG / address / hasher-seed leakage into the decision), and
///  * the SAME statement text placed in a different textual ORDER dispatches the
///    same way statement-for-statement (no iteration-order dependence — the
///    ladder's order was never load-bearing and the recogniser's is not either).
///
/// A `HashMap`-seeded or `-march`-conditioned recogniser would be free to differ
/// across runs/hosts and would fail the first assertion in a `RandomState`-seeded
/// process (the seed differs per process, so the 32-run loop below is run under
/// one process AND the whole test binary is re-run by CI on both x86 and ARM —
/// the cross-substrate keystone/canary gates are the arch half of this proof).
#[test]
fn recognizer_selection_is_a_pure_function_of_the_key_set() {
    let src = "\
fn body() -> i64 {
    let mut acc: i64 = 0;
    let x: i64 = 3;
    if x > 0 { acc = acc + x; }
    while acc < 100 { acc = acc + 1; if acc > 50 { break; } }
    loop { acc = acc + 1; break; }
    return acc;
}
";
    let first = format!("{:?}", parser::parse(src).expect("must parse"));
    for i in 0..32 {
        let again = format!("{:?}", parser::parse(src).expect("must parse"));
        assert_eq!(first, again, "parse {i} diverged — selection is not pure");
    }

    // Order permutation: the same three item kinds, declared in two different
    // orders, must each dispatch to their own parser (item count and kinds are
    // preserved). The ladder was order-insensitive because the keywords are
    // pairwise distinct under the word-boundary rule; the recogniser is too.
    let order_a = "const A: i64 = 1;\nstruct S { a: i64 }\nfn f() -> i64 { 1 }\n";
    let order_b = "fn f() -> i64 { 1 }\nconst A: i64 = 1;\nstruct S { a: i64 }\n";
    let ma = parser::parse(order_a).expect("order A must parse");
    let mb = parser::parse(order_b).expect("order B must parse");
    assert_eq!(ma.items.len(), 3);
    assert_eq!(mb.items.len(), 3);
    let kind = |n: &libmind::ast::Node| {
        format!("{n:?}")
            .split_whitespace()
            .next()
            .unwrap()
            .to_string()
    };
    let mut ka: Vec<String> = ma.items.iter().map(kind).collect();
    let mut kb: Vec<String> = mb.items.iter().map(kind).collect();
    ka.sort();
    kb.sort();
    assert_eq!(ka, kb, "textual order changed which parser was selected");
}

/// The self-host compiler source is the real workload the recogniser runs on
/// (~29.7k lines). It must still parse, and parse identically across repeats.
#[test]
fn selfhost_source_parses_identically_across_repeats() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/mindc_mind/main.mind");
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return, // source not present in this checkout — nothing to gate
    };
    let a = parser::parse(&src).expect("self-host source must parse");
    let b = parser::parse(&src).expect("self-host source must parse");
    assert_eq!(a.items.len(), b.items.len());
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}
