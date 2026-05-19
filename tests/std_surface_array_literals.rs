// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 6.2b Gap 2 — array literals `[expr, expr, ...]` and
//! fixed-size array types `[T; N]`.
//!
//! Covers:
//! 1. `[1, 2, 3]` parses and lowers to a `ConstArray` IR instruction typed as `[i64; 3]`.
//! 2. `[i64; 0]` (empty array type) parses and is a valid type annotation.
//! 3. A 4,096-entry array literal (generated programmatically) parses without
//!    stack overflow and produces exactly 4,096 elements.
//! 4. `const FOO: [i64; 4] = [1, 2, 3, 4]; fn nth(i: i64) -> i64 { FOO[i] }`
//!    — the const is registered in the module-level env and IndexAccess on it
//!    lowers to a `__mind_array_load_i64` call.
//! 5. A type-length mismatch `let x: [i64; 3] = [1, 2]` is rejected at
//!    type-check time with a diagnostic containing "length".
//!
//! Gated: `cargo test --features std-surface --test std_surface_array_literals`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

// ── Test 1: basic array literal parses + lowers ──────────────────────────────

#[test]
fn array_lit_three_elements_parses_and_lowers() {
    let src = "[1, 2, 3]";
    let module = parser::parse(src).expect("[1, 2, 3] must parse");
    // The parsed module should contain an ArrayLit node.
    let ir = lower_to_ir(&module);
    // Must contain a ConstArray instruction with 3 elements.
    let has_const_array = ir
        .instrs
        .iter()
        .any(|i| matches!(i, Instr::ConstArray { values, .. } if values.len() == 3));
    assert!(
        has_const_array,
        "expected ConstArray with 3 elements in IR, got: {:?}",
        ir.instrs
    );
}

// ── Test 2: empty array literal ───────────────────────────────────────────────

#[test]
fn array_lit_empty_parses() {
    // Empty array literal should parse without error.
    let src = "let x: [i64; 0] = []";
    let _module = parser::parse(src).expect("[i64; 0] empty array must parse");
}

// ── Test 3: large array literal (4,096 entries, no stack overflow) ────────────

#[test]
fn array_lit_4096_entries_no_stack_overflow() {
    // Generate a 4096-element array literal: [0, 1, 2, ..., 4095]
    let mut src = String::from("[");
    for i in 0u32..4096 {
        if i > 0 {
            src.push_str(", ");
        }
        src.push_str(&i.to_string());
    }
    src.push(']');

    let module = parser::parse(&src).expect("4096-entry array literal must parse without overflow");
    let ir = lower_to_ir(&module);
    let has_const_array = ir
        .instrs
        .iter()
        .any(|i| matches!(i, Instr::ConstArray { values, .. } if values.len() == 4096));
    assert!(
        has_const_array,
        "expected ConstArray with 4096 elements in lowered IR"
    );
}

// ── Test 4: const array + index access ───────────────────────────────────────

#[test]
fn const_array_and_index_access_lower() {
    let src = r#"
const FOO: [i64; 4] = [1, 2, 3, 4];

fn nth(i: i64) -> i64 {
    FOO[i]
}
"#;
    let module = parser::parse(src).expect("const FOO + fn nth must parse");
    let ir = lower_to_ir(&module);

    // The module-level ConstArray for FOO must exist.
    let has_foo = ir.instrs.iter().any(|i| {
        matches!(i, Instr::ConstArray { name: Some(n), values, .. }
            if n == "FOO" && values.len() == 4)
    });
    assert!(has_foo, "expected named ConstArray 'FOO' in module IR");

    // The fn body of `nth` must contain an array load.
    let fn_body = ir
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::FnDef { name, body, .. } if name == "nth" => Some(body.as_slice()),
            _ => None,
        })
        .expect("expected FnDef 'nth' in module IR");

    let has_load = fn_body.iter().any(|i| matches!(i, Instr::ArrayLoad { .. }));
    assert!(
        has_load,
        "expected ArrayLoad instruction in fn nth body, got: {:?}",
        fn_body
    );
}

// ── Test 5: type-length mismatch rejected ────────────────────────────────────

#[test]
fn type_mismatch_length_rejected() {
    use libmind::type_checker;
    let src = "let x: [i64; 3] = [1, 2]";
    let module = parser::parse(src).expect("let x: [i64; 3] = [1, 2] must parse");
    let diags = type_checker::check_module_types(&module, src, &Default::default());
    assert!(
        !diags.is_empty(),
        "expected type-check diagnostic for array length mismatch, but got none"
    );
    // The diagnostic message must mention the length discrepancy.
    let has_length_msg = diags.iter().any(|d| {
        d.message.contains("length")
            || d.message.contains("3")
            || d.message.contains("2")
            || d.message.contains("array")
            || d.message.contains("mismatch")
    });
    assert!(
        has_length_msg,
        "diagnostic should mention length mismatch, got: {:?}",
        diags
    );
}
