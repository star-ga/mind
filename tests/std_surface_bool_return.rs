// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `-> bool` return-ABI lowering — a comparison result is an MLIR `i1`, but
//! every user fn lowers under the uniform `-> i64` std-surface ABI. When the
//! `i1` flows straight into the return slot, the emitter must zero-extend it
//! (`arith.extui %v : i1 to i64`) or mlir-opt rejects `return %v : i64`
//! ("'i64' vs 'i1'") and the whole `-> bool` function fails to lower.
//!
//! This regression covers BOTH return sites:
//!   * the implicit fall-off-the-end return (synthesised from `ret_id`), and
//!   * the explicit `Instr::Return { value }`.
//!
//! The fix is additive: only a return-of-comparison (which previously did not
//! compile at all) changes emitted text, so existing artifacts — and the
//! keystone byte-identity — are unaffected.
//!
//! Gated: `cargo test --features std-surface,mlir-lowering
//! --test std_surface_bool_return`.

#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]

use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use libmind::mlir::lower_ir_to_mlir;

/// fn body: `x <cmp> 0` producing an i1 comparison result.
/// Returns (body-without-terminator, comparison ValueId).
fn cmp_zero_body(op: BinOp) -> (Vec<Instr>, ValueId) {
    let x = ValueId(0);
    let zero = ValueId(1);
    let cmp = ValueId(2);
    let body = vec![
        Instr::Param {
            dst: x,
            name: "x".to_string(),
            index: 0,
        },
        Instr::ConstI64(zero, 0),
        Instr::BinOp {
            dst: cmp,
            op,
            lhs: x,
            rhs: zero,
        },
    ];
    (body, cmp)
}

fn module_with_fn(name: &str, body: Vec<Instr>, ret: ValueId) -> IRModule {
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: name.to_string(),
        params: vec![("x".to_string(), ValueId(0))],
        ret_id: Some(ret),
        body,
        reap_threshold: None,
    });
    // At least one main-level op so the assembler emits @main.
    let c = m.fresh();
    m.instrs.push(Instr::ConstI64(c, 0));
    m.instrs.push(Instr::Output(c));
    m
}

#[test]
fn implicit_bool_return_zero_extends_i1_to_i64() {
    // `fn is_pos(x) -> bool { x > 0 }` — comparison is the trailing expression,
    // so it lowers through the synthesised (implicit) return path.
    let (body, cmp) = cmp_zero_body(BinOp::Gt);
    let m = module_with_fn("is_pos", body, cmp);

    let text = lower_ir_to_mlir(&m).expect("lower").text;

    assert!(
        text.contains("arith.cmpi \"sgt\""),
        "expected a signed-greater-than comparison; got:\n{text}"
    );
    assert!(
        text.contains(&format!("arith.extui %{} : i1 to i64", cmp.0)),
        "the i1 comparison result must be zero-extended before the i64 return; got:\n{text}"
    );
    assert!(
        text.contains(&format!("return %bext{} : i64", cmp.0)),
        "the widened value (not the raw i1) must be returned; got:\n{text}"
    );
    // The raw i1 must NOT be returned directly (that is the bug).
    assert!(
        !text.contains(&format!("return %{} : i64", cmp.0)),
        "must not return the raw i1 as i64; got:\n{text}"
    );
}

#[test]
fn explicit_bool_return_zero_extends_i1_to_i64() {
    // `fn is_eq0(x) -> bool { return x == 0; }` — explicit Return of the
    // comparison, exercising the `Instr::Return` arm.
    let (mut body, cmp) = cmp_zero_body(BinOp::Eq);
    body.push(Instr::Return { value: Some(cmp) });
    let m = module_with_fn("is_eq0", body, cmp);

    let text = lower_ir_to_mlir(&m).expect("lower").text;

    assert!(
        text.contains(&format!("arith.extui %{} : i1 to i64", cmp.0)),
        "explicit return of an i1 must be zero-extended; got:\n{text}"
    );
    assert!(
        text.contains(&format!("return %bext{} : i64", cmp.0)),
        "explicit return must return the widened i64 value; got:\n{text}"
    );
}

#[test]
fn non_bool_return_is_unchanged_no_extui() {
    // A plain arithmetic return must NOT gain an extui — guards against the
    // fix accidentally widening genuine i64 values (which would change
    // existing artifacts / the keystone).
    let x = ValueId(0);
    let one = ValueId(1);
    let sum = ValueId(2);
    let body = vec![
        Instr::Param {
            dst: x,
            name: "x".to_string(),
            index: 0,
        },
        Instr::ConstI64(one, 1),
        Instr::BinOp {
            dst: sum,
            op: BinOp::Add,
            lhs: x,
            rhs: one,
        },
    ];
    let m = module_with_fn("add_one", body, sum);

    let text = lower_ir_to_mlir(&m).expect("lower").text;

    assert!(
        !text.contains("arith.extui"),
        "a non-bool (i64) return must not emit an extui; got:\n{text}"
    );
    assert!(
        text.contains(&format!("return %{} : i64", sum.0)),
        "the i64 sum must be returned directly; got:\n{text}"
    );
}
