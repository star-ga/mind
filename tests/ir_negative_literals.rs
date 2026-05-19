// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Regression: bare negative integer literals must lower to the correct
//! signed constant — identical to the binary-subtraction form `(0 - N)`.
//!
//! Before the fix, `lower_expr` had no `ast::Node::Neg` arm, so every unary
//! minus fell through to the catch-all and was silently lowered to
//! `const.i64 0`. The defect was surfaced by the pure-MIND lookup-table
//! work: generated tables containing entries like `-524288` / `-65536`
//! were silently zeroed. The `(0 - N)` source form was never affected.
//!
//! Run with:
//!   cargo test --test ir_negative_literals --features std-surface

use libmind::eval;
use libmind::ir::Instr;
use libmind::parser;

/// Lower `src` and collect every `ConstI64` immediate that appears inside a
/// function body (the FnDef body is not printed by `--emit-ir`, so we read
/// the IR structurally).
fn fn_body_const_i64s(src: &str) -> Vec<i64> {
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:#?}"));
    let ir = eval::lower_to_ir(&module);
    let mut out = Vec::new();
    for instr in &ir.instrs {
        if let Instr::FnDef { body, .. } = instr {
            for b in body {
                if let Instr::ConstI64(_, v) = b {
                    out.push(*v);
                }
            }
        }
    }
    out
}

fn fn_body_const_f64s(src: &str) -> Vec<f64> {
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:#?}"));
    let ir = eval::lower_to_ir(&module);
    let mut out = Vec::new();
    for instr in &ir.instrs {
        if let Instr::FnDef { body, .. } = instr {
            for b in body {
                if let Instr::ConstF64(_, v) = b {
                    out.push(*v);
                }
            }
        }
    }
    out
}

/// Core reproducer from the bug report: `let a: i64 = -65536; return a;`
/// must emit the constant -65536, not 0.
#[test]
fn neg_65536_lowers_to_minus_65536() {
    let consts = fn_body_const_i64s("pub fn f() -> i64 { let a: i64 = -65536; return a; }");
    assert!(
        consts.contains(&-65536),
        "expected ConstI64(-65536) in fn body, got {consts:?}"
    );
    assert!(
        !consts.contains(&0),
        "buggy `const.i64 0` placeholder must not appear, got {consts:?}"
    );
}

#[test]
fn neg_one() {
    assert_eq!(
        fn_body_const_i64s("pub fn f() -> i64 { let a: i64 = -1; return a; }"),
        vec![-1]
    );
}

#[test]
fn neg_i32_min() {
    assert_eq!(
        fn_body_const_i64s("pub fn f() -> i64 { let a: i64 = -2147483648; return a; }"),
        vec![-2147483648]
    );
}

#[test]
fn neg_i64_max_magnitude() {
    assert_eq!(
        fn_body_const_i64s("pub fn f() -> i64 { let a: i64 = -9223372036854775807; return a; }"),
        vec![-9223372036854775807]
    );
}

/// INT64_MIN edge: `-9223372036854775808` must behave identically to
/// `(0 - 9223372036854775808)`. In two's complement both equal i64::MIN
/// (`0 - INT64_MIN` wraps back to INT64_MIN, and so does `wrapping_neg`).
#[test]
fn neg_i64_min_matches_subtraction_form() {
    let neg_form =
        fn_body_const_i64s("pub fn f() -> i64 { let a: i64 = -9223372036854775808; return a; }");
    assert_eq!(neg_form, vec![i64::MIN]);

    let sub_form = fn_body_const_i64s(
        "pub fn f() -> i64 { let a: i64 = (0 - 9223372036854775808); return a; }",
    );
    // sub form: ConstI64(0), ConstI64(i64::MIN), then BinOp Sub.
    assert!(
        sub_form.contains(&i64::MIN),
        "subtraction form must also reach i64::MIN, got {sub_form:?}"
    );
}

/// `-N` in arithmetic position: `x + -5`.
#[test]
fn negative_literal_in_arithmetic() {
    let consts = fn_body_const_i64s("pub fn f(x: i64) -> i64 { return x + -5; }");
    assert!(
        consts.contains(&-5),
        "expected ConstI64(-5) for `x + -5`, got {consts:?}"
    );
}

/// `-N` as a call argument: `g(-3)`.
#[test]
fn negative_literal_as_argument() {
    let src = "pub fn g(y: i64) -> i64 { return y; } pub fn f() -> i64 { return g(-3); }";
    let consts = fn_body_const_i64s(src);
    assert!(
        consts.contains(&-3),
        "expected ConstI64(-3) call arg, got {consts:?}"
    );
}

/// `-N` inside an array literal: `[-7, 9]`.
#[test]
fn negative_literal_in_array() {
    let module =
        parser::parse("pub fn f() -> [i64; 2] { let a: [i64; 2] = [-7, 9]; return a; }").unwrap();
    let ir = eval::lower_to_ir(&module);
    let mut found = false;
    for instr in &ir.instrs {
        if let Instr::FnDef { body, .. } = instr {
            for b in body {
                if let Instr::ConstArray { values, .. } = b {
                    assert_eq!(values, &vec![-7, 9], "array literal must keep signs");
                    found = true;
                }
            }
        }
    }
    assert!(found, "expected a ConstArray in the fn body");
}

/// Negative float literal: `-1.5` must fold to ConstF64(-1.5), not 0.
#[test]
fn negative_float_literal() {
    let consts = fn_body_const_f64s("pub fn f() -> f64 { let a: f64 = -1.5; return a; }");
    assert!(
        consts.contains(&-1.5),
        "expected ConstF64(-1.5), got {consts:?}"
    );
}

/// Double negation `-(-8)`: the inner `-8` folds to ConstI64(-8); the outer
/// negate (operand is itself a `Neg`, not a bare literal) lowers as
/// `0 - (-8)` via `BinOp::Sub`, which equals 8 — identical to the
/// hand-written subtraction source form.
#[test]
fn double_negation_lowers_as_subtraction() {
    let module = parser::parse("pub fn f() -> i64 { let a: i64 = -(-8); return a; }").unwrap();
    let ir = eval::lower_to_ir(&module);
    for instr in &ir.instrs {
        if let Instr::FnDef { body, .. } = instr {
            let consts: Vec<i64> = body
                .iter()
                .filter_map(|b| match b {
                    Instr::ConstI64(_, v) => Some(*v),
                    _ => None,
                })
                .collect();
            assert!(
                consts.contains(&0) && consts.contains(&-8),
                "expected ConstI64(0) and ConstI64(-8) for `0 - (-8)`, got {consts:?}"
            );
            let has_sub = body.iter().any(|b| {
                matches!(
                    b,
                    Instr::BinOp {
                        op: libmind::ir::BinOp::Sub,
                        ..
                    }
                )
            });
            assert!(has_sub, "outer negate must lower to a Sub BinOp");
            return;
        }
    }
    panic!("no FnDef in lowered IR");
}
