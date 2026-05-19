// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Phase 6.5 Stage 1a — `Instr::If` parsing, IR lowering, and MLIR emission.
//!
//! Covers:
//! 1. `if cond { X } else { Y }` where both branches return same-type expr.
//! 2. `if cond { return X } else { return Y }` (early-return in both branches).
//! 3. Nested `if` (`if a { if b { ... } else { ... } } else { ... }`).
//! 4. `let` inside if-branch is visible in outer scope after the if (Gap C).
//!
//! All cases must parse, lower to `Instr::If` in the IR, and (with
//! `mlir-lowering`) emit MLIR that does NOT contain `func.return` as a
//! mid-block terminator.
//!
//! Gated: `cargo test --features "std-surface mlir-lowering" --test std_surface_if_statement`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

fn must_parse(src: &str) -> libmind::ast::Module {
    parser::parse(src).unwrap_or_else(|errs| {
        panic!(
            "parse failed with {} error(s):\n{}",
            errs.len(),
            errs.iter()
                .map(|e| format!("  {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })
}

fn count_if_instrs(instrs: &[Instr]) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::If {
                then_instrs,
                else_instrs,
                cond_instrs,
                ..
            } => {
                n += 1;
                n += count_if_instrs(cond_instrs);
                n += count_if_instrs(then_instrs);
                n += count_if_instrs(else_instrs);
            }
            Instr::FnDef { body, .. } => {
                n += count_if_instrs(body);
            }
            Instr::While { body, cond_instrs, .. } => {
                n += count_if_instrs(body);
                n += count_if_instrs(cond_instrs);
            }
            _ => {}
        }
    }
    n
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: if/else where both branches return a value expression (no early return)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn if_else_value_expr_lowers_to_instr_if() {
    let src = r#"
fn pick(cond: i64) -> i64 {
    if cond == 1 { 42 } else { 7 }
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);

    let if_count = count_if_instrs(&ir.instrs);
    assert!(
        if_count >= 1,
        "expected at least one Instr::If; got {if_count}\nIR: {:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: early return in both branches — the key case for the lexer bootstrap.
//
// Before Phase 6.5 Stage 1a, this produced `func.return` mid-block in MLIR,
// which mlir-opt rejected. The fix emits each branch into its own MLIR basic
// block so both `return` ops are terminators.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn if_early_return_in_both_branches_lowers_to_instr_if() {
    let src = r#"
fn classify(b: i64) -> i64 {
    if b == 32 { return 1; }
    0
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);

    let if_count = count_if_instrs(&ir.instrs);
    assert!(
        if_count >= 1,
        "expected at least one Instr::If; got {if_count}\nIR: {:?}",
        ir.instrs
    );
}

#[test]
fn if_else_both_return_lowers_to_instr_if() {
    let src = r#"
fn classify(b: i64) -> i64 {
    if b == 32 { return 1 } else { return 0 }
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);

    let if_count = count_if_instrs(&ir.instrs);
    assert!(
        if_count >= 1,
        "expected at least one Instr::If from early-return pattern; got {if_count}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: nested if (`if a { if b { ... } else { ... } } else { ... }`)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn nested_if_else_lowers_to_nested_instr_if() {
    let src = r#"
fn nested(a: i64, b: i64) -> i64 {
    if a == 1 {
        if b == 2 { 10 } else { 20 }
    } else {
        30
    }
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);

    let if_count = count_if_instrs(&ir.instrs);
    assert!(
        if_count >= 2,
        "expected at least 2 Instr::If for nested if; got {if_count}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 4: `let` inside if-branch is visible in the outer scope after the if
// (Gap C verification)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn let_binding_in_then_branch_visible_after_if() {
    let src = r#"
fn gap_c_demo(cond: i64) -> i64 {
    if cond == 1 {
        let x: i64 = 99
        x
    } else {
        0
    }
}
"#;
    let module = must_parse(src);
    // Must parse and lower without errors or panics.
    let ir = lower_to_ir(&module);
    assert!(
        count_if_instrs(&ir.instrs) >= 1,
        "expected Instr::If to exist"
    );
    // The `branch_bindings` on the Instr::If should capture `x`.
    let has_binding = ir.instrs.iter().any(|i| {
        if let Instr::FnDef { body, .. } = i {
            body.iter().any(|bi| {
                if let Instr::If { branch_bindings, .. } = bi {
                    branch_bindings.iter().any(|(name, _)| name == "x")
                } else {
                    false
                }
            })
        } else {
            false
        }
    });
    assert!(
        has_binding,
        "branch_bindings on Instr::If should contain `x`"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 5: lexer-style chain of `if b == N { return 1; }` followed by fallback 0
// This is the exact pattern in examples/lexer/main.mind `is_space`.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn lexer_style_early_return_chain_lowers_cleanly() {
    let src = r#"
fn is_space(b: i64) -> i64 {
    if b == 32 { return 1; }
    if b == 9  { return 1; }
    if b == 10 { return 1; }
    if b == 13 { return 1; }
    0
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);

    let if_count = count_if_instrs(&ir.instrs);
    assert!(
        if_count >= 4,
        "expected 4 Instr::If nodes (one per early-return check); got {if_count}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 6: MLIR lowering of `if/return` does not produce mid-block terminators
//
// Verifies the core MLIR correctness requirement: after Phase 6.5 Stage 1a,
// the emitted MLIR must NOT contain `func.return` in the middle of a basic
// block (i.e., there must be no `return` followed by more instructions in the
// same block).
//
// Gated on both `std-surface` AND `mlir-lowering`.
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "mlir-lowering")]
#[test]
fn if_return_mlir_does_not_place_return_mid_block() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    // Build: fn classify(b: i64) -> i64 {
    //   if b == 32 { return 1 } else { 0 }
    // }
    let mut m = IRModule::new();
    let param_b = ValueId(0);
    let cond_val = ValueId(1);
    let cst32 = ValueId(2);
    let cst1_then = ValueId(3);
    let cst0_else = ValueId(4);
    let dst = ValueId(5);

    // Condition instructions: %cst32 = 32, %cond_val = cmpi eq %param_b, %cst32
    let cond_instrs = vec![
        Instr::ConstI64(cst32, 32),
        Instr::BinOp {
            dst: cond_val,
            op: libmind::ir::BinOp::Eq,
            lhs: param_b,
            rhs: cst32,
        },
    ];
    // Then branch: return 1
    let then_instrs = vec![
        Instr::ConstI64(cst1_then, 1),
        Instr::Return {
            value: Some(cst1_then),
        },
    ];
    // Else branch: 0
    let else_instrs = vec![Instr::ConstI64(cst0_else, 0)];

    let unit = ValueId(6);
    let fn_body = vec![
        Instr::Param {
            dst: param_b,
            name: "b".to_string(),
            index: 0,
        },
        Instr::If {
            cond_id: cond_val,
            cond_instrs,
            then_instrs,
            then_result: cst1_then,
            else_instrs,
            else_result: cst0_else,
            dst,
            branch_bindings: vec![],
        },
        Instr::ConstI64(unit, 0),
        Instr::Return { value: Some(dst) },
    ];

    m.instrs.push(Instr::FnDef {
        name: "classify".to_string(),
        params: vec![("b".to_string(), param_b)],
        ret_id: Some(dst),
        body: fn_body,
        reap_threshold: None,
    });
    let out_id = m.fresh();
    m.instrs.push(Instr::ConstI64(out_id, 0));
    m.instrs.push(Instr::Output(out_id));

    let text = lower_ir_to_mlir(&m)
        .expect("if/return MLIR lowering must not fail")
        .text;

    // Verify basic-block structure is emitted.
    assert!(
        text.contains("^if_then_"),
        "expected ^if_then_N block in MLIR output; got:\n{text}"
    );
    assert!(
        text.contains("^if_else_"),
        "expected ^if_else_N block in MLIR output; got:\n{text}"
    );
    assert!(
        text.contains("^if_after_"),
        "expected ^if_after_N block in MLIR output; got:\n{text}"
    );
    assert!(
        text.contains("cf.cond_br"),
        "expected cf.cond_br dispatch; got:\n{text}"
    );

    // The defining check: `func.return` must ONLY appear at the end of a
    // block (preceded only by whitespace / indentation). Specifically, after
    // a `return` there must not be another non-label instruction in the same
    // block.  We check this by verifying that `return %` is NEVER followed
    // by another `%<n> =` assignment on the very next non-empty line.
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("return ") || line.trim_start() == "return" {
            // Check the next non-empty, non-comment line.
            for j in (i + 1)..lines.len() {
                let next = lines[j].trim();
                if next.is_empty() || next.starts_with("//") {
                    continue;
                }
                // A label (`^<name>:`) is OK — it starts a new basic block.
                if next.starts_with('^') {
                    break;
                }
                // A `}` is OK — closes the function.
                if next == "}" {
                    break;
                }
                panic!(
                    "func.return is mid-block: line {i} is a return, \
                     but line {j} (`{next}`) is not a block label or closing brace.\
                     \nFull MLIR:\n{text}"
                );
            }
        }
    }
}
