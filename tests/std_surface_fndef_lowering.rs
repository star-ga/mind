// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 P0d — `Instr::FnDef` -> MLIR `func.func` lowering.
//!
//! Phase 0 (committed `8b2199f`) made `Instr::Call` reach MLIR for
//! external symbols. P0d (this test) proves *user-defined* functions
//! also reach MLIR as `func.func @name(%pN: i64...) -> i64 { ... }`
//! emitted as a sibling top-level symbol before `@main`. This is the
//! foundation for Phase 2 — pure-MIND `std.vec` defines `vec_new`,
//! `vec_push`, `vec_get` etc. as ordinary fns; without P0d each of
//! those would silently fall through the MLIR catch-all.
//!
//! The struct-codegen blocker (P0e) is *not* in scope here — these
//! tests use only `i64` parameters and return.
//!
//! Gated: `cargo test --features std-surface,mlir-lowering
//! --test std_surface_fndef_lowering`.

#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]

use libmind::ir::{IRModule, Instr, ValueId};
use libmind::mlir::lower_ir_to_mlir;

/// Convenience: build a fn body that takes one i64, adds 1, returns it.
fn add_one_body(start_id: usize) -> (Vec<Instr>, ValueId) {
    let param = ValueId(start_id);
    let one = ValueId(start_id + 1);
    let sum = ValueId(start_id + 2);
    let body = vec![
        Instr::Param {
            dst: param,
            name: "x".to_string(),
            index: 0,
        },
        Instr::ConstI64(one, 1),
        Instr::BinOp {
            dst: sum,
            op: libmind::ir::BinOp::Add,
            lhs: param,
            rhs: one,
        },
        Instr::Return { value: Some(sum) },
    ];
    (body, sum)
}

#[test]
fn fndef_lowers_to_func_func_with_i64_signature() {
    let mut m = IRModule::new();
    let (body, ret) = add_one_body(0);
    m.instrs.push(Instr::FnDef {
        name: "add_one".to_string(),
        params: vec![("x".to_string(), ValueId(0))],
        ret_id: Some(ret),
        body,
        reap_threshold: None,
    });
    // Need at least one main-level op so the assembler emits @main.
    let c = m.fresh();
    m.instrs.push(Instr::ConstI64(c, 0));
    m.instrs.push(Instr::Output(c));

    let text = lower_ir_to_mlir(&m).expect("lower").text;
    assert!(
        text.contains("func.func @add_one(%0: i64) -> i64 {"),
        "expected `func.func @add_one(%0: i64) -> i64`; got:\n{text}"
    );
    assert!(
        text.contains("return %"),
        "fn body must end with a return; got:\n{text}"
    );
}

#[test]
fn fndef_appears_before_main_in_emitted_text() {
    let mut m = IRModule::new();
    let (body, ret) = add_one_body(0);
    m.instrs.push(Instr::FnDef {
        name: "helper".to_string(),
        params: vec![("x".to_string(), ValueId(0))],
        ret_id: Some(ret),
        body,
        reap_threshold: None,
    });
    let c = m.fresh();
    m.instrs.push(Instr::ConstI64(c, 0));
    m.instrs.push(Instr::Output(c));

    let text = lower_ir_to_mlir(&m).expect("lower").text;
    let helper_pos = text
        .find("func.func @helper")
        .expect("helper definition missing");
    let main_pos = text.find("func.func @main").expect("@main missing");
    assert!(
        helper_pos < main_pos,
        "user fn must be emitted BEFORE @main so calls in @main resolve;\nhelper@{helper_pos} main@{main_pos}\nfull:\n{text}"
    );
}

#[test]
fn fndef_and_call_to_it_compose_cleanly() {
    // FnDef `double(x) = x + x`, then @main calls it.
    let mut m = IRModule::new();
    let p = ValueId(0);
    let s = ValueId(1);
    let body = vec![
        Instr::Param {
            dst: p,
            name: "x".to_string(),
            index: 0,
        },
        Instr::BinOp {
            dst: s,
            op: libmind::ir::BinOp::Add,
            lhs: p,
            rhs: p,
        },
        Instr::Return { value: Some(s) },
    ];
    m.instrs.push(Instr::FnDef {
        name: "double".to_string(),
        params: vec![("x".to_string(), p)],
        ret_id: Some(s),
        body,
        reap_threshold: None,
    });

    let c = m.fresh();
    m.instrs.push(Instr::ConstI64(c, 21));
    let r = m.fresh();
    m.instrs.push(Instr::Call {
        dst: r,
        name: "double".to_string(),
        args: vec![c],
    });
    m.instrs.push(Instr::Output(r));

    let text = lower_ir_to_mlir(&m).expect("lower").text;
    assert!(
        text.contains("func.func @double("),
        "definition missing; got:\n{text}"
    );
    assert!(
        text.contains("func.call @double(%"),
        "call site missing; got:\n{text}"
    );
    // P0d: a private forward decl would clash with the definition —
    // the assembler must NOT emit one when the symbol is defined.
    assert!(
        !text.contains("func.func private @double"),
        "must NOT emit a private fwd-decl for a locally defined fn; got:\n{text}"
    );
}

#[test]
fn external_intrinsic_still_gets_private_fwd_decl() {
    // A call to `__mind_alloc` with no FnDef must still produce its
    // `func.func private @__mind_alloc(i64) -> i64` declaration — P0d
    // only suppresses the decl for *locally defined* symbols.
    let mut m = IRModule::new();
    let n = m.fresh();
    m.instrs.push(Instr::ConstI64(n, 64));
    let r = m.fresh();
    m.instrs.push(Instr::Call {
        dst: r,
        name: "__mind_alloc".to_string(),
        args: vec![n],
    });
    m.instrs.push(Instr::Output(r));

    let text = lower_ir_to_mlir(&m).expect("lower").text;
    assert!(
        text.contains("func.func private @__mind_alloc(i64) -> i64"),
        "intrinsic forward decl must still be emitted; got:\n{text}"
    );
}
