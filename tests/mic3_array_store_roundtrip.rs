// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Regression: the mic@3 wire format must round-trip `Instr::ArrayStore`
//! (aggregate element write `arr[idx] = value`, opcode 0x2B — #320 Step D
//! aggregate mutation) EXACTLY — all four operands (`dst`, `base`, `index`,
//! `value`) preserved across emit -> parse and a fixed point under
//! emit -> parse -> emit (the property every committed `trace_hash` relies on).
//!
//! `ArrayStore` is a pure additive opcode appended in previously-unused op-space
//! (0x2A = `OP_CONST_DENSE_TENSOR` was the prior max), so no `MIC3_VERSION` byte
//! bump is required; the keystone + cross_substrate byte-identity gates PROVE the
//! append perturbs no existing stream.

#![cfg(feature = "std-surface")]

use libmind::ir::compact::{emit_mic3, parse_mic3};
use libmind::ir::{IRModule, Instr, ValueId};

/// A module whose exported fn builds a 3-element array, then stores `9` into
/// slot `0` and returns the fresh post-store aggregate id — exercising the full
/// `ArrayStore { dst, base, index, value }` operand set on the wire.
fn module_with_array_store() -> IRModule {
    let mut m = IRModule::new();
    let base = m.fresh();
    let idx = m.fresh();
    let val = m.fresh();
    let stored = m.fresh();
    let r = m.fresh();
    m.exports.insert("store".to_string());
    m.instrs.push(Instr::FnDef {
        name: "store".to_string(),
        params: vec![],
        ret_id: Some(r),
        body: vec![
            Instr::ConstArray {
                dst: base,
                name: None,
                values: vec![1, 2, 3],
            },
            Instr::ConstI64(idx, 0),
            Instr::ConstI64(val, 9),
            Instr::ArrayStore {
                dst: stored,
                base,
                index: idx,
                value: val,
            },
            // Read back through the fresh incarnation so the store is observed.
            Instr::ArrayLoad {
                dst: r,
                base: stored,
                index: idx,
            },
            Instr::Return { value: Some(r) },
        ],
        reap_threshold: None,
        #[cfg(feature = "std-surface")]
        value_types: std::collections::BTreeMap::new(),
    });
    m
}

/// Extract the first `ArrayStore`'s (dst, base, index, value) operand ids.
fn first_store(m: &IRModule) -> Option<(ValueId, ValueId, ValueId, ValueId)> {
    fn walk(instrs: &[Instr]) -> Option<(ValueId, ValueId, ValueId, ValueId)> {
        for i in instrs {
            match i {
                Instr::ArrayStore {
                    dst,
                    base,
                    index,
                    value,
                } => return Some((*dst, *base, *index, *value)),
                Instr::FnDef { body, .. } => {
                    if let Some(x) = walk(body) {
                        return Some(x);
                    }
                }
                _ => {}
            }
        }
        None
    }
    walk(&m.instrs)
}

#[test]
fn array_store_survives_mic3_round_trip() {
    let m = module_with_array_store();
    let src = first_store(&m).expect("fixture has an array store");

    let bytes = emit_mic3(&m);
    let parsed = parse_mic3(&bytes).expect("mic@3 must re-parse an ArrayStore module");
    let got = first_store(&parsed).expect("parsed module retains the array store");

    assert_eq!(
        got, src,
        "ArrayStore operands (dst,base,index,value) must survive emit->parse exactly"
    );

    // Fixed point: emit(parse(emit(m))) == emit(m). This is the invariant every
    // committed trace_hash depends on — a re-emitted parsed module is byte-stable.
    let bytes2 = emit_mic3(&parsed);
    assert_eq!(
        bytes, bytes2,
        "mic@3 emit->parse->emit must be a byte-identical fixed point for ArrayStore"
    );

    // The 0x2B opcode byte must actually appear (guards against a silent
    // encode-nothing regression that would still round-trip an empty body).
    assert!(
        bytes.contains(&0x2Bu8),
        "the ArrayStore opcode 0x2B must be present in the emitted mic@3 bytes"
    );
}
