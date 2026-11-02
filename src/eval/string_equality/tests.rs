// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit controls for typed string equality normalization.

use super::*;
use crate::ast::Span;

fn count_calls(node: &Node, name: &str) -> usize {
    let here = matches!(node, Node::Call { callee, .. } if callee == name) as usize;
    here + match node {
        Node::FnDef(fd, _) => fd.body.iter().map(|n| count_calls(n, name)).sum::<usize>(),
        _ => super::super::closures::node_children_ref(node)
            .into_iter()
            .map(|n| count_calls(n, name))
            .sum::<usize>(),
    }
}

fn set_comparison_span(node: &mut Node, replacement: Span) {
    if let Node::Binary {
        op: BinOp::Eq | BinOp::Ne,
        span,
        ..
    } = node
    {
        *span = replacement;
    }
    for child in super::super::closures::node_children_mut(node) {
        set_comparison_span(child, replacement);
    }
}

#[test]
fn rewrites_only_locally_proven_strings() {
    let module = crate::parser::parse(
        r#"fn same() -> i64 {
            let a: string = "a"
            let b: string = "b"
            let local: string = "x"
            if a == b { return 1 }
            if local != "y" { return 2 }
            return 0
        }"#,
    )
    .unwrap();
    let rewritten = normalize(&module).expect("two proven comparisons");
    assert_eq!(
        rewritten
            .items
            .iter()
            .map(|n| count_calls(n, "__mind_string_eq"))
            .sum::<usize>(),
        2
    );
}

#[test]
fn numeric_and_invalidated_facts_are_no_op() {
    let module = crate::parser::parse(
        r#"fn control() -> i64 {
            let value: string = "x"
            value = 7
            if value == "x" { return 1 }
            if 2 != 3 { return 2 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    assert!(analysis.rewritten.is_none());
    // The reassigned name has no fact, so its comparison with text is refused.
    assert_eq!(analysis.refused.len(), 1);
}

#[test]
fn identical_cross_source_spans_do_not_rewrite_numeric_comparisons() {
    let mut numeric = crate::parser::parse(
        "fn numeric(a: i64, b: i64) -> i64 { if a == b { return 1 }; return 0 }",
    )
    .unwrap();
    let mut textual = crate::parser::parse(
        "fn textual() -> i64 { let a = \"a\"; let b = \"b\"; if a == b { return 1 }; return 0 }",
    )
    .unwrap();
    let shared = Span::new(40, 46);
    set_comparison_span(&mut numeric.items[0], shared);
    set_comparison_span(&mut textual.items[0], shared);
    numeric.items.extend(textual.items);

    let rewritten = normalize(&numeric).expect("the string comparison is proven");
    assert_eq!(count_calls(&rewritten.items[0], "__mind_string_eq"), 0);
    assert_eq!(count_calls(&rewritten.items[1], "__mind_string_eq"), 1);
}

#[test]
fn an_assigned_name_is_never_treated_as_immutable_string_proof() {
    let module = crate::parser::parse(
        r#"fn control(flag: bool) -> i64 {
            let value: string = "x"
            if flag { value = 7; let value: string = "x" }
            if value == "x" { return 1 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    assert!(analysis.rewritten.is_none());
    assert_eq!(analysis.refused.len(), 1);
}

/// A `string` parameter compared with text was a handle compare that returned
/// false for equal bytes; it is now byte-wise. A numeric value under a string
/// annotation proves nothing, and an integer operand stays a handle test.
#[test]
fn string_parameters_compare_bytes_numeric_annotations_do_not() {
    let module = crate::parser::parse(
        r#"fn control(real: string) -> i64 {
            let fake: string = 7
            if fake == "x" { return 1 }
            if real == 7 { return 2 }
            if real == "x" { return 3 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    let rewritten = analysis
        .rewritten
        .expect("the parameter comparison is typed");
    assert_eq!(count_calls(&rewritten.items[0], "__mind_string_eq"), 1);
    assert_eq!(analysis.refused.len(), 1, "`fake == \"x\"` is refused");
}

#[test]
fn call_results_fields_and_constants_are_typed_strings() {
    let module = crate::parser::parse(
        r#"struct Dec { name: string, n: i64 }
        const MODE: string = "cache"
        fn label() -> string { return "x" }
        fn control(d: Dec, s: string) -> i64 {
            if d.name == "cache" { return 1 }
            if label() != "x" { return 2 }
            if s == MODE { return 3 }
            if d.n == 3 { return 4 }
            if s == 0 { return 5 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    assert!(analysis.refused.is_empty());
    let rewritten = analysis.rewritten.expect("three typed string comparisons");
    let calls: usize = rewritten
        .items
        .iter()
        .map(|n| count_calls(n, "__mind_string_eq"))
        .sum();
    assert_eq!(calls, 3);
}

#[test]
fn text_against_an_untyped_operand_is_refused() {
    let module = crate::parser::parse(
        r#"const MODE: string = "cache"
        fn control(h: i64, s: string) -> i64 {
            let MODE = 5
            if h == "x" { return 1 }
            if s != MODE { return 2 }
            if h == 0 { return 3 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    assert!(analysis.rewritten.is_none());
    let ops: Vec<BinOp> = analysis.refused.iter().map(|r| r.op).collect();
    assert_eq!(ops, vec![BinOp::Eq, BinOp::Ne]);
}

#[test]
fn lexical_shadowing_does_not_change_outer_proof() {
    let module = crate::parser::parse(
        r#"fn control() -> i64 {
            let value: string = "outer"
            if true { let value = 7 }
            if value == "outer" { return 1 }
            return 0
        }"#,
    )
    .unwrap();
    let rewritten = normalize(&module).expect("outer immutable string remains proven");
    assert_eq!(count_calls(&rewritten.items[0], "__mind_string_eq"), 1);
}

#[test]
fn string_array_elements_are_typed_strings() {
    let module = crate::parser::parse(
        r#"fn control(parts: array<string>, ns: array<i64>) -> i64 {
            if parts[0] == "a" { return 1 }
            for p in parts {
                if p != "b" { return 2 }
            }
            if ns[0] == 3 { return 3 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    assert!(analysis.refused.is_empty());
    let rewritten = analysis.rewritten.expect("two element comparisons");
    assert_eq!(count_calls(&rewritten.items[0], "__mind_string_eq"), 2);
}

#[test]
fn a_string_binding_stays_a_string_after_string_assignments() {
    let module = crate::parser::parse(
        r#"fn control(flag: i64, p: string) -> i64 {
            let mut s: string = "abc"
            s = "abd"
            if s != "abc" { return 1 }
            p = "x"
            if p == "x" { return 2 }
            return 0
        }"#,
    )
    .unwrap();
    let analysis = analyze(&module);
    assert!(analysis.refused.is_empty());
    let rewritten = analysis.rewritten.expect("both comparisons stay typed");
    assert_eq!(count_calls(&rewritten.items[0], "__mind_string_eq"), 2);
}
