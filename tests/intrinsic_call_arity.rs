// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A call to a registered `__mind_*` intrinsic carries exactly the declared
//! arity. `__mind_write(fd, buf, count, offset)` with three arguments used to
//! pass verification and build, so the C runtime read an undefined offset.

#![cfg(feature = "std-surface")]

use libmind::ir::{IRModule, Instr, ValueId, verify_module};

fn module_calling_write(arg_count: usize) -> IRModule {
    let mut body: Vec<Instr> = (0..arg_count)
        .map(|i| Instr::ConstI64(ValueId(i), 1))
        .collect();
    body.push(Instr::Call {
        dst: ValueId(arg_count),
        name: "__mind_write".to_string(),
        args: (0..arg_count).map(ValueId).collect(),
        resolved_callee: None,
    });
    body.push(Instr::Return {
        value: Some(ValueId(arg_count)),
    });
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: "main".to_string(),
        params: vec![],
        ret_id: Some(ValueId(arg_count)),
        body,
        reap_threshold: None,
        semantic_types: None,
        value_types: std::collections::BTreeMap::new(),
    });
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::Output(ValueId(0)));
    m.next_id = arg_count + 1;
    m
}

#[test]
fn a_short_intrinsic_call_fails_verification() {
    let err = verify_module(&module_calling_write(3))
        .expect_err("a 3-argument __mind_write must not verify");
    assert!(
        err.to_string()
            .contains("intrinsic `__mind_write` expects 4 i64 argument(s)"),
        "the error names the intrinsic and its arity: {err}"
    );
}

#[test]
fn an_extra_argument_fails_verification() {
    assert!(verify_module(&module_calling_write(5)).is_err());
}

#[test]
fn the_declared_arity_verifies() {
    verify_module(&module_calling_write(4)).expect("a 4-argument __mind_write verifies");
}
