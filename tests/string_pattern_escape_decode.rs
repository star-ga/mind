// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! String-literal MATCH PATTERNS must decode escapes identically to string
//! LITERALS (audit finding — parser raw-vs-decoded mismatch).
//!
//! `parse_string_lit` decodes `"\n"` to the single byte `0x0A`, but the pattern
//! parser used to copy the RAW slice, storing the two bytes `\` `n`. A
//! `match s { "\n" => … }` pattern therefore never matched a scrutinee built
//! from the `"\n"` literal (their byte strings differed: 1 vs 2). The runnable
//! lowering rejects string patterns fail-loud today, so this was a LATENT
//! silent-miscompile; the fix routes both through the shared `decode_string_body`
//! so a string pattern and a string literal always carry the same bytes.
//!
//! This is a PARSE-level gate (no codegen), so it needs only the parser.

use libmind::ast::{Literal, Node, Pattern};
use libmind::parser::parse;

/// Depth-first: return the first `Match` node's arms.
fn first_match_arms(node: &Node) -> Option<&Vec<libmind::ast::MatchArm>> {
    match node {
        Node::Match { arms, .. } => Some(arms),
        Node::Return { value: Some(v), .. } => first_match_arms(v),
        Node::FnDef { body, .. } | Node::Block { stmts: body, .. } => {
            body.iter().find_map(first_match_arms)
        }
        _ => None,
    }
}

fn first_match_arms_in_module(src: &str) -> Vec<libmind::ast::MatchArm> {
    let m = parse(src).expect("parse");
    m.items
        .iter()
        .find_map(first_match_arms)
        .expect("a match node")
        .clone()
}

#[test]
fn string_pattern_decodes_newline_escape() {
    // First arm pattern `"\n"` must decode to the single byte 0x0A — NOT the raw
    // 2-byte `\` `n`.
    let arms = first_match_arms_in_module(
        r#"fn f(s: String) -> i64 { return match s { "\n" => 1, _ => 0 }; }"#,
    );
    match &arms[0].pattern {
        Pattern::Literal(Literal::Str(s)) => {
            assert_eq!(
                s.as_bytes(),
                &[0x0A],
                "string pattern `\"\\n\"` must decode to one byte 0x0A, got {:?}",
                s.as_bytes()
            );
        }
        other => panic!("expected a string-literal pattern, got {other:?}"),
    }
}

#[test]
fn string_pattern_matches_string_literal_bytes() {
    // The pattern's decoded bytes must equal the SAME literal's decoded bytes,
    // so a `match "\t" { "\t" => … }` can actually match.
    let arms = first_match_arms_in_module(
        r#"fn f(s: String) -> i64 { return match s { "\t" => 1, _ => 0 }; }"#,
    );
    let pat_bytes = match &arms[0].pattern {
        Pattern::Literal(Literal::Str(s)) => s.clone().into_bytes(),
        other => panic!("expected a string-literal pattern, got {other:?}"),
    };

    // Decode the same escape as an expression literal via a tiny program.
    let m = parse(r#"fn g() -> i64 { let x = "\t"; return 0; }"#).expect("parse");
    let lit_bytes = {
        fn find_str_lit(node: &Node) -> Option<Vec<u8>> {
            match node {
                Node::Lit(Literal::Str(s), _) => Some(s.clone().into_bytes()),
                Node::Let { value, .. }
                | Node::Return {
                    value: Some(value), ..
                } => find_str_lit(value),
                Node::FnDef { body, .. } | Node::Block { stmts: body, .. } => {
                    body.iter().find_map(find_str_lit)
                }
                _ => None,
            }
        }
        m.items
            .iter()
            .find_map(find_str_lit)
            .expect("a string literal")
    };

    assert_eq!(
        pat_bytes, lit_bytes,
        "string pattern bytes must equal string literal bytes for the same escape"
    );
    assert_eq!(pat_bytes, vec![0x09], "`\\t` decodes to one byte 0x09");
}

#[test]
fn string_pattern_plain_unchanged() {
    // A non-escaped pattern is byte-identical to before (regression guard).
    let arms = first_match_arms_in_module(
        r#"fn f(s: String) -> i64 { return match s { "abc" => 1, _ => 0 }; }"#,
    );
    match &arms[0].pattern {
        Pattern::Literal(Literal::Str(s)) => assert_eq!(s, "abc"),
        other => panic!("expected a string-literal pattern, got {other:?}"),
    }
}
