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

//! RFC 0005 Phase B — intra-module call signature soundness.
//!
//! Before this slice the module-level FnDef pre-registration pass put every
//! function name into the type env as a bare `ScalarI64` placeholder with NO
//! signature, so an intra-module call like `g(1, 2, 3)` to a 1-arg `g`
//! type-checked silently — a soundness hole. A side-table now captures every
//! fn's arity (built for ALL fns before ANY call is checked, so forward
//! references and recursion resolve), and each intra-module call site validates
//! arity, emitting `E2005` on a mismatch.
//!
//! Per-arg *type* checking is intentionally NOT enforced: under the loose i64
//! ABI it is not soundly determinable (intra-module calls return `ScalarI64`,
//! literals widen freely, struct/aggregate values are i64 heap addresses, and
//! tensor-shape agreement is RFC 0012's symbolic pass). Arity is the property
//! that is always knowable from the AST, so it is the false-positive-free check.
//!
//! This file also covers match exhaustiveness on sum/enum types
//! (`match::non_exhaustive`): a match on a known enum that omits a variant with
//! no wildcard/binding catch-all is an error; an exhaustive match, a match with
//! a wildcard, and an integer/literal match (which legitimately relies on a `_`
//! arm) are all accepted.

use libmind::parser;
use libmind::type_checker::check_module_types;

/// Parse `src` and run the module type-checker with an empty env, returning the
/// diagnostic codes produced (in order).
fn check_codes(src: &str) -> Vec<&'static str> {
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let env = std::collections::HashMap::new();
    check_module_types(&module, src, &env)
        .iter()
        .map(|d| d.code)
        .collect()
}

/// True when at least one diagnostic carries `code`.
fn has_code(src: &str, code: &'static str) -> bool {
    check_codes(src).contains(&code)
}

// ── HOLE 1: intra-module call arity ──────────────────────────────────────────

#[test]
fn wrong_arity_intra_module_call_errors_e2005() {
    // `g` takes 2 args; the call passes 3 — previously accepted silently.
    let src = r#"
fn g(a: i64, b: i64) -> i64 {
    a + b
}

fn caller() -> i64 {
    g(1, 2, 3)
}
"#;
    assert!(
        has_code(src, "E2005"),
        "a wrong-arity intra-module call must emit E2005; got {:?}",
        check_codes(src)
    );
}

#[test]
fn too_few_arguments_also_errors_e2005() {
    let src = r#"
fn g(a: i64, b: i64) -> i64 {
    a + b
}

fn caller() -> i64 {
    g(1)
}
"#;
    assert!(
        has_code(src, "E2005"),
        "too-few-arguments must also emit E2005; got {:?}",
        check_codes(src)
    );
}

#[test]
fn correct_arity_intra_module_call_passes() {
    let src = r#"
fn g(a: i64, b: i64) -> i64 {
    a + b
}

fn caller() -> i64 {
    g(1, 2)
}
"#;
    assert!(
        !has_code(src, "E2005"),
        "a correct-arity call must not emit E2005; got {:?}",
        check_codes(src)
    );
}

#[test]
fn forward_reference_call_is_checked_not_skipped() {
    // `helper` is called BEFORE it is defined in source order. The side-table
    // is built for all fns first, so the forward call is still arity-checked.
    let bad = r#"
fn caller() -> i64 {
    helper(1, 2)
}

fn helper(x: i64) -> i64 {
    x
}
"#;
    assert!(
        has_code(bad, "E2005"),
        "a wrong-arity forward-reference call must emit E2005; got {:?}",
        check_codes(bad)
    );

    let ok = r#"
fn caller() -> i64 {
    helper(1)
}

fn helper(x: i64) -> i64 {
    x
}
"#;
    assert!(
        !has_code(ok, "E2005"),
        "a correct-arity forward-reference call must pass; got {:?}",
        check_codes(ok)
    );
}

#[test]
fn recursion_does_not_false_positive() {
    let src = r#"
fn countdown(x: i64) -> i64 {
    if x > 0 {
        countdown(x - 1)
    } else {
        0
    }
}
"#;
    assert!(
        !has_code(src, "E2005"),
        "a correct self-recursive call must not emit E2005; got {:?}",
        check_codes(src)
    );
}

#[test]
fn generic_fn_arity_is_checked_types_are_not() {
    // A generic fn's arg can be anything, so only arity is enforced.
    let ok = r#"
fn id<T>(x: T) -> T {
    x
}

fn caller() -> i64 {
    id(42)
}
"#;
    assert!(
        !has_code(ok, "E2005"),
        "a correct-arity call to a generic fn must pass; got {:?}",
        check_codes(ok)
    );

    let bad = r#"
fn id<T>(x: T) -> T {
    x
}

fn caller() -> i64 {
    id(1, 2)
}
"#;
    assert!(
        has_code(bad, "E2005"),
        "a wrong-arity call to a generic fn must still emit E2005; got {:?}",
        check_codes(bad)
    );
}

#[test]
fn varied_scalar_arg_types_do_not_false_positive() {
    // Integer literals (ScalarI32) into i64 params, the loose i64 ABI — must
    // not trigger any call-signature error.
    let src = r#"
fn add3(a: i64, b: i64, c: i64) -> i64 {
    a + b + c
}

fn caller() -> i64 {
    add3(1, 2, 3)
}
"#;
    let codes = check_codes(src);
    assert!(
        !codes.iter().any(|c| *c == "E2005" || *c == "E2006"),
        "loose i64-ABI scalar args must not produce a call-signature error; got {codes:?}"
    );
}

// ── HOLE 2: match exhaustiveness on sum/enum types ───────────────────────────

#[test]
fn non_exhaustive_enum_match_errors() {
    let src = r#"
enum Color {
    Red,
    Green,
    Blue,
}

fn name_of(c: i64) -> i64 {
    match c {
        Color::Red => 1,
        Color::Green => 2,
    }
}
"#;
    assert!(
        has_code(src, "match::non_exhaustive"),
        "a non-exhaustive enum match must emit match::non_exhaustive; got {:?}",
        check_codes(src)
    );
}

#[test]
fn exhaustive_enum_match_passes() {
    let src = r#"
enum Color {
    Red,
    Green,
    Blue,
}

fn name_of(c: i64) -> i64 {
    match c {
        Color::Red => 1,
        Color::Green => 2,
        Color::Blue => 3,
    }
}
"#;
    assert!(
        !has_code(src, "match::non_exhaustive"),
        "an exhaustive enum match must not be flagged; got {:?}",
        check_codes(src)
    );
}

#[test]
fn enum_match_with_wildcard_passes() {
    let src = r#"
enum Color {
    Red,
    Green,
    Blue,
}

fn name_of(c: i64) -> i64 {
    match c {
        Color::Red => 1,
        _ => 0,
    }
}
"#;
    assert!(
        !has_code(src, "match::non_exhaustive"),
        "an enum match with a wildcard arm must not be flagged; got {:?}",
        check_codes(src)
    );
}

#[test]
fn integer_match_without_wildcard_is_not_flagged() {
    // Integer matches cannot be "exhaustive" without a wildcard and must NOT be
    // flagged — this is exactly the shape main.mind's 51 tag-matches take.
    let src = r#"
fn classify(x: i64) -> i64 {
    match x {
        0 => 10,
        1 => 20,
    }
}
"#;
    assert!(
        !has_code(src, "match::non_exhaustive"),
        "an integer/literal match must never be flagged non-exhaustive; got {:?}",
        check_codes(src)
    );
}
