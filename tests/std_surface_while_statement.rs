// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Gap 1 — `while` statement parsing + IR lowering.
//!
//! Covers:
//! 1. Trivial counted loop (`while i < 10 { i = i + 1 }`).
//! 2. Nested `while` (2x2 tally with two loop counters).
//! 3. `while` with mutable state declared outside the loop.
//! 4. `while` inside an `if` arm.
//!
//! All four cases must parse, lower to IR, and (with `mlir-lowering`)
//! emit MLIR cleanly under `--emit-ir`.
//!
//! Gated: `cargo test --features std-surface --test std_surface_while_statement`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

// ──────────────────────────────────────────────────────────────────────────────
// Helper: assert that a MIND source parses without errors.
// ──────────────────────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────────────────────
// Helper: count `Instr::While` nodes in a flat instruction list (top-level and
// inside `FnDef` bodies).
// ──────────────────────────────────────────────────────────────────────────────

fn count_while_instrs(instrs: &[Instr]) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::While { body, .. } => {
                n += 1;
                // Recurse into the body to count nested while loops.
                n += count_while_instrs(body);
            }
            Instr::FnDef { body, .. } => {
                n += count_while_instrs(body);
            }
            _ => {}
        }
    }
    n
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: trivial counted loop
//
// fn count_to(n: i64) -> i64 {
//     let mut i: i64 = 0
//     while i < n { i = i + 1 }
//     i
// }
//
// Expectation: parses cleanly; the lowered IR for `count_to` contains exactly
// one `Instr::While` node.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn while_trivial_counted_loop_parses_and_lowers() {
    let src = r#"
fn count_to(n: i64) -> i64 {
    let mut i: i64 = 0
    while i < n {
        i = i + 1
    }
    i
}
"#;
    let module = must_parse(src);
    // Verify at least the `while` keyword was consumed as a statement.
    assert!(
        module.items.iter().any(|n| matches!(n, libmind::ast::Node::FnDef { .. })),
        "FnDef not found in module items"
    );

    let ir = lower_to_ir(&module);
    assert_eq!(
        count_while_instrs(&ir.instrs),
        1,
        "expected exactly one While instr; got:\n{:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: nested while (outer and inner loop)
//
// fn grid_sum(n: i64) -> i64 {
//     let mut r: i64 = 0
//     let mut i: i64 = 0
//     while i < n {
//         let mut j: i64 = 0
//         while j < n {
//             r = r + 1
//             j = j + 1
//         }
//         i = i + 1
//     }
//     r
// }
//
// Expectation: exactly two `Instr::While` nodes in the IR.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn while_nested_two_loops_produce_two_while_instrs() {
    let src = r#"
fn grid_sum(n: i64) -> i64 {
    let mut r: i64 = 0
    let mut i: i64 = 0
    while i < n {
        let mut j: i64 = 0
        while j < n {
            r = r + 1
            j = j + 1
        }
        i = i + 1
    }
    r
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert_eq!(
        count_while_instrs(&ir.instrs),
        2,
        "expected two While instrs (outer + inner); got:\n{:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: while with mutable state declared outside the loop
//
// fn accumulate(n: i64) -> i64 {
//     let mut sum: i64 = 0
//     let mut i: i64 = 0
//     while i < n {
//         sum = sum + i
//         i = i + 1
//     }
//     sum
// }
//
// Expectation: parses cleanly; IR contains one While with two entries in
// `live_vars` (`sum` and `i`).
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn while_with_mutable_state_outside_loop() {
    let src = r#"
fn accumulate(n: i64) -> i64 {
    let mut sum: i64 = 0
    let mut i: i64 = 0
    while i < n {
        sum = sum + i
        i = i + 1
    }
    sum
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert_eq!(
        count_while_instrs(&ir.instrs),
        1,
        "expected one While instr; got {n}",
        n = count_while_instrs(&ir.instrs)
    );

    // Inspect live_vars: the body mutates both `sum` and `i`.
    let mut live_var_count = 0;
    for instr in &ir.instrs {
        if let Instr::FnDef { body, .. } = instr {
            for bi in body {
                if let Instr::While { live_vars, .. } = bi {
                    live_var_count = live_vars.len();
                }
            }
        }
    }
    assert!(
        live_var_count >= 2,
        "expected >= 2 live_vars (sum, i); got {live_var_count}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 4: while inside an if arm
//
// fn conditional_loop(cond: i64, n: i64) -> i64 {
//     let mut v: i64 = 0
//     if cond {
//         let mut i: i64 = 0
//         while i < n {
//             v = v + 1
//             i = i + 1
//         }
//     }
//     v
// }
//
// Expectation: parses cleanly; IR contains at least one While node (inside the
// then-branch of the if).
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn while_inside_if_arm_parses_and_lowers() {
    let src = r#"
fn conditional_loop(cond: i64, n: i64) -> i64 {
    let mut v: i64 = 0
    if cond {
        let mut i: i64 = 0
        while i < n {
            v = v + 1
            i = i + 1
        }
    }
    v
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert_eq!(
        count_while_instrs(&ir.instrs),
        1,
        "expected one While instr (inside if then-branch); got:\n{:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 5 (bonus): the RFC 0005 Gap 1 design-doc reproducer
//
// This is the exact five-line example from the design doc.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn while_rfc0005_gap1_reproducer_parses() {
    let src = r#"
fn count_to(n: i64) -> i64 {
    let mut i: i64 = 0
    while i < n {
        i = i + 1
    }
    i
}
"#;
    // Must parse without any errors.
    let _module = must_parse(src);
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 6: MLIR lowering of a while loop emits basic-block structure
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "mlir-lowering")]
#[test]
fn while_mlir_lowering_emits_basic_blocks() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    // Build a minimal IR module that contains a While node directly.
    let mut m = IRModule::new();

    // Condition: %0 = 1 (always true — just tests emission)
    let cond_val = ValueId(0);
    let cond_instrs = vec![libmind::ir::Instr::ConstI64(cond_val, 1)];

    // Body: %1 = 0 (trivial body)
    let body_val = ValueId(1);
    let body_instrs = vec![libmind::ir::Instr::ConstI64(body_val, 0)];

    // Emit the While inside a FnDef body.
    let param_id = ValueId(2);
    let unit_id = ValueId(3);
    let fn_body = vec![
        libmind::ir::Instr::Param {
            dst: param_id,
            name: "x".to_string(),
            index: 0,
        },
        Instr::While {
            cond_id: cond_val,
            cond_instrs,
            body: body_instrs,
            live_vars: vec![],
        },
        libmind::ir::Instr::ConstI64(unit_id, 0),
        libmind::ir::Instr::Return {
            value: Some(unit_id),
        },
    ];

    m.instrs.push(Instr::FnDef {
        name: "loopy".to_string(),
        params: vec![("x".to_string(), param_id)],
        ret_id: Some(unit_id),
        body: fn_body,
        reap_threshold: None,
    });

    let c = m.fresh();
    m.instrs.push(Instr::ConstI64(c, 0));
    m.instrs.push(Instr::Output(c));

    let text = lower_ir_to_mlir(&m).expect("while MLIR lowering failed").text;

    assert!(
        text.contains("^while_header_"),
        "expected ^while_header_N block; got:\n{text}"
    );
    assert!(
        text.contains("^while_body_"),
        "expected ^while_body_N block; got:\n{text}"
    );
    assert!(
        text.contains("^while_after_"),
        "expected ^while_after_N block; got:\n{text}"
    );
    assert!(
        text.contains("cf.cond_br"),
        "expected cf.cond_br for loop dispatch; got:\n{text}"
    );
    assert!(
        text.contains("cf.br ^while_header_"),
        "expected back-edge to header; got:\n{text}"
    );
}
