// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 6.2b — Gap 3: unsigned-i64 literal reinterpret-cast.
//!
//! Verifies that integer literals in the range `(i64::MAX, u64::MAX]` are
//! accepted by the parser and stored as signed i64 bit-patterns (two's
//! complement, identical to Rust's `u64 as i64`).
//!
//! Run with:
//!   cargo test --test parser_unsigned_i64_literals

use libmind::ast::{Literal, Node};
use libmind::parser;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Parse a source string that is a single integer expression and return the
/// `i64` value stored in the resulting `Node::Lit(Literal::Int(_), _)`.
fn parse_int(src: &str) -> i64 {
    let module = parser::parse(src).unwrap_or_else(|e| {
        panic!("parse failed for {:?}: {e:#?}", src);
    });
    let first = module.items.first().expect("module must have one item");
    match first {
        Node::Lit(Literal::Int(v), _) => *v,
        other => panic!("expected Lit(Int(_)), got {other:#?}"),
    }
}

/// Parse a source string expected to fail and assert the error message
/// contains the given substring.
fn parse_expect_err(src: &str, expected_msg: &str) {
    let err = parser::parse(src).expect_err("expected parse error");
    let msg = err
        .iter()
        .map(|e| e.message.as_str())
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        msg.contains(expected_msg),
        "expected error containing {:?}, got: {msg:?}",
        expected_msg
    );
}

// ── Gap 3 test cases ─────────────────────────────────────────────────────────

/// i64::MAX stays positive — taken by the normal i64 parse path.
#[test]
fn i64_max_stays_positive() {
    let expected: i64 = i64::MAX;
    assert_eq!(parse_int("9223372036854775807"), expected);
}

/// i64::MAX + 1 = first value that needs u64 fallback.
/// Rust: 9223372036854775808u64 as i64 == i64::MIN == -9223372036854775808.
#[test]
fn i64_max_plus_one_reinterprets_to_i64_min() {
    let expected: i64 = 9223372036854775808_u64 as i64; // == i64::MIN
    assert_eq!(expected, i64::MIN);
    assert_eq!(parse_int("9223372036854775808"), expected);
}

/// FNV-1a 64-bit offset basis: 0xCBF29CE484222325 = 14695981039346656037.
/// Rust: 14695981039346656037u64 as i64 == -3750763034362895579.
#[test]
fn fnv1a_offset_basis_reinterprets_correctly() {
    let u: u64 = 14695981039346656037;
    let expected: i64 = u as i64; // -3750763034362895579
    assert_eq!(expected, -3750763034362895579_i64);
    assert_eq!(parse_int("14695981039346656037"), expected);
}

/// u64::MAX = 18446744073709551615.
/// Rust: u64::MAX as i64 == -1.
#[test]
fn u64_max_reinterprets_to_minus_one() {
    let expected: i64 = u64::MAX as i64; // -1
    assert_eq!(expected, -1_i64);
    assert_eq!(parse_int("18446744073709551615"), expected);
}

/// u64::MAX + 1 = 18446744073709551616 — must still be rejected.
#[test]
fn u64_max_plus_one_is_overflow() {
    parse_expect_err("18446744073709551616", "integer overflow");
}

// ── Byte-level round-trip sanity ─────────────────────────────────────────────

/// The stored i64 bit-pattern must be byte-identical to what Rust produces
/// from the same `u64 as i64` cast.  This is the contract the design doc
/// mandates ("byte-identical to a Rust `as i64` cast").
#[test]
fn byte_patterns_match_rust_cast() {
    let cases: &[(&str, u64)] = &[
        ("14695981039346656037", 14695981039346656037_u64),
        ("18446744073709551615", 18446744073709551615_u64),
        ("9223372036854775808", 9223372036854775808_u64),
        ("9223372036854775807", 9223372036854775807_u64), // i64::MAX — i64 path
    ];
    for (src, u) in cases {
        let got = parse_int(src);
        let want = *u as i64;
        assert_eq!(
            got.to_le_bytes(),
            want.to_le_bytes(),
            "byte mismatch for {src}: got {got}, want {want}"
        );
    }
}

// ── fn-body context (full function parse) ────────────────────────────────────

/// Verify the literal is accepted inside a fn body, matching the original
/// failing reproducer from the Phase 6.3 type-checker work.
#[test]
fn fnv_offset_inside_fn_body_parses() {
    let src = "pub fn fnv_offset() -> i64 { 14695981039346656037 }";
    let module = parser::parse(src).unwrap_or_else(|e| {
        panic!("parse failed: {e:#?}");
    });
    // We just need it to parse; IR round-trip is a separate smoke test.
    assert_eq!(module.items.len(), 1, "expected one top-level fn");
}

/// u64::MAX inside a fn body also accepted.
#[test]
fn u64_max_inside_fn_body_parses() {
    let src = "pub fn sentinel() -> i64 { 18446744073709551615 }";
    let module = parser::parse(src).unwrap_or_else(|e| {
        panic!("parse failed: {e:#?}");
    });
    assert_eq!(module.items.len(), 1);
}

// ── Range-syntax path (for N..M) ─────────────────────────────────────────────

/// The range-syntax disambiguation branch inside `parse_number` (the `N..M`
/// path that avoids treating `..` as a decimal point) uses the same
/// `parse_i64_literal` helper.  Confirm a large literal used as a for-loop
/// bound is accepted.
#[test]
fn large_literal_in_for_range_bound_accepted() {
    // The lower bound 9223372036854775808 exceeds i64::MAX; the upper bound
    // 9223372036854775810 also does.  Both should parse via u64 fallback.
    let src = "fn f() -> i64 { for i in 9223372036854775808..9223372036854775810 { } 0 }";
    parser::parse(src).unwrap_or_else(|e| {
        panic!("for-range with large bounds failed: {e:#?}");
    });
}
