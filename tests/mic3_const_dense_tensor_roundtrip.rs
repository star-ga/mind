// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Regression: the mic@3 wire format must round-trip `Instr::ConstDenseTensor`
//! (dense tensor literal `[1.0, 2.0, 3.0]`, opcode 0x2A) EXACTLY — dtype, shape,
//! and the per-element bit patterns. A dense literal is the in-function tensor
//! constructor introduced for the tensor-RUNS track; the old i64 `ConstArray`
//! path silently coerced float elements to `[0, 0, 0]`, so the new opcode must
//! preserve the exact IEEE bits across emit -> parse and be a fixed point under
//! emit -> parse -> emit (the property every committed `trace_hash` relies on).

#![cfg(feature = "std-surface")]

use libmind::ir::compact::{emit_mic3, parse_mic3};
use libmind::ir::{IRModule, Instr};
use libmind::types::{DType, ShapeDim};

/// A module whose exported fn materialises an f32 dense tensor with
/// deliberately non-trivial bit patterns (2.5, -3.75) so a float→0 coercion or
/// a width/endianness bug is observable, not masked by all-zero data.
fn module_with_dense_tensor() -> IRModule {
    let mut m = IRModule::new();
    let t = m.fresh();
    let r = m.fresh();
    m.exports.insert("dense".to_string());
    m.instrs.push(Instr::FnDef {
        name: "dense".to_string(),
        params: vec![],
        ret_id: Some(r),
        body: vec![
            Instr::ConstDenseTensor {
                dst: t,
                dtype: DType::F32,
                shape: vec![ShapeDim::Known(3)],
                data: vec![
                    (1.0f32).to_bits() as u64,
                    (2.5f32).to_bits() as u64,
                    (-3.75f32).to_bits() as u64,
                ],
            },
            Instr::ConstI64(r, 0),
            Instr::Return { value: Some(r) },
        ],
        reap_threshold: None,
    });
    m
}

fn first_dense(m: &IRModule) -> Option<(DType, Vec<ShapeDim>, Vec<u64>)> {
    fn walk(instrs: &[Instr]) -> Option<(DType, Vec<ShapeDim>, Vec<u64>)> {
        for i in instrs {
            match i {
                Instr::ConstDenseTensor {
                    dtype, shape, data, ..
                } => return Some((dtype.clone(), shape.clone(), data.clone())),
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
fn const_dense_tensor_survives_mic3_round_trip() {
    let m = module_with_dense_tensor();
    let (d0, s0, data0) = first_dense(&m).expect("fixture has a dense tensor");

    let bytes = emit_mic3(&m);
    let parsed = parse_mic3(&bytes).expect("mic@3 must re-parse");
    let (d1, s1, data1) = first_dense(&parsed).expect("parsed module retains the dense tensor");

    assert_eq!(d0, d1, "dtype must survive emit->parse");
    assert_eq!(s0, s1, "shape must survive emit->parse");
    assert_eq!(
        data0, data1,
        "exact element bits must survive (a float-coercion or width bug shows here)"
    );

    // Full fixed point: emit(parse(emit(m))) == emit(m).
    let twice = emit_mic3(&parsed);
    assert_eq!(
        bytes, twice,
        "mic@3 must be a fixed point under emit->parse->emit for ConstDenseTensor"
    );
}
