// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! IR-level proof that `for` / `for-each` inject the loop STEP before every
//! in-scope `continue` (audit rank 5) — the desugar fix in
//! `inject_step_before_continue`.
//!
//! `for`/`for-each` desugar to a `while` whose counter increment sits at the
//! body TAIL. A `continue` jumps to the header and would SKIP that increment →
//! infinite loop. The fix splices the step (`VAR = VAR + 1` / `idx = idx + 1`,
//! which lowers to a `BinOp::Add`) directly before each in-scope `continue`.
//!
//! Range-`for` is also proven end-to-end (compile + run) by
//! `for_continue_advances_run.rs`; this IR gate additionally covers `for-each`,
//! which cannot yet be lowered to a runnable `.so` (the array-literal `vec_len`
//! aggregate-call ABI is the separate RFC 0005 phase-2+ gap), so an IR-shape
//! assertion is the strongest available proof for the for-each arm.
//!
//! Gate: `cargo test --features "std-surface"
//!                   --test for_continue_step_injection`

#![cfg(feature = "std-surface")]

use libmind::eval::lower;
use libmind::ir::{BinOp, Instr};
use libmind::parser;

/// The immediate child instruction lists of an instruction (the nested blocks a
/// `continue` can live in). Mirrors the containers `inject_step_before_continue`
/// walks: fn bodies, loop bodies + conditions, both if branches, regions.
fn child_blocks(instr: &Instr) -> Vec<&[Instr]> {
    match instr {
        Instr::FnDef { body, .. } => vec![body.as_slice()],
        Instr::While {
            cond_instrs, body, ..
        } => vec![cond_instrs.as_slice(), body.as_slice()],
        Instr::If {
            cond_instrs,
            then_instrs,
            else_instrs,
            ..
        } => vec![
            cond_instrs.as_slice(),
            then_instrs.as_slice(),
            else_instrs.as_slice(),
        ],
        Instr::Region { body, .. } => vec![body.as_slice()],
        // A bare braced block in a loop body lowers to a Region; Region already
        // covers it. No standalone block Instr to descend here.
        _ => vec![],
    }
}

/// Walk every instruction block. Returns `(saw_continue, every_continue_stepped)`
/// where `every_continue_stepped` is false if any block holds a `Continue` with
/// no `BinOp::Add` before it in the SAME block (the injected step).
fn audit(instrs: &[Instr], saw: &mut bool, ok: &mut bool) {
    // Does this block have an Add before its first Continue?
    if let Some(cont_at) = instrs
        .iter()
        .position(|i| matches!(i, Instr::Continue { .. }))
    {
        *saw = true;
        let stepped = instrs[..cont_at]
            .iter()
            .any(|i| matches!(i, Instr::BinOp { op: BinOp::Add, .. }));
        if !stepped {
            *ok = false;
        }
    }
    for instr in instrs {
        for block in child_blocks(instr) {
            audit(block, saw, ok);
        }
    }
}

fn assert_stepped(src: &str, label: &str) {
    let module = parser::parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e:?}"));
    let ir = lower::lower_to_ir(&module);
    let mut saw = false;
    let mut ok = true;
    audit(&ir.instrs, &mut saw, &mut ok);
    assert!(
        saw,
        "{label}: expected a `continue` marker in the lowered IR"
    );
    assert!(
        ok,
        "{label}: a `continue` block has no injected step (BinOp::Add) before it \
         — the loop counter would never advance (infinite loop)"
    );
}

#[test]
fn range_for_continue_injects_step() {
    // `if i == 2 { continue; }` — the continue sits in the if's then-block; the
    // fix must place `i = i + 1` (a BinOp::Add) before it.
    assert_stepped(
        "fn f(n: i64) -> i64 { let mut s = 0; for i in 0..n { if i == 2 { continue; } s = s + i; } return s; }",
        "range-for",
    );
}

#[test]
fn for_each_continue_injects_step() {
    // for-each over an array literal — the index step `idx = idx + 1` must be
    // spliced before the continue (can't be run-tested: RFC 0005 phase-2+ ABI).
    assert_stepped(
        "fn f() -> i64 { let xs = [0, 1, 2, 3]; let mut s = 0; for x in xs { if x == 2 { continue; } s = s + x; } return s; }",
        "for-each",
    );
}

#[test]
fn bare_block_continue_injects_step() {
    // `continue` nested inside a bare braced block `{ … }` in the loop body — the
    // walker must descend into the `Node::Block`, else the step is missed and the
    // loop hangs (NET-verified before the Block arm was added).
    assert_stepped(
        "fn f(n: i64) -> i64 { let mut s = 0; for i in 0..n { { if i == 2 { continue; } } s = s + i; } return s; }",
        "bare-block",
    );
}

#[test]
fn nested_loop_continue_targets_own_loop() {
    // Inner + outer continue: both must be stepped, and the outer step must NOT
    // be grafted onto the inner continue (the boundary). Structural check only —
    // every continue block has its Add.
    assert_stepped(
        "fn f(n: i64, m: i64) -> i64 { let mut s = 0; \
         for i in 0..n { for j in 0..m { if j == 1 { continue; } s = s + 1; } if i == 0 { continue; } s = s + 100; } \
         return s; }",
        "nested",
    );
}
