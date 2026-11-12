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
        semantic_types: None,
        #[cfg(feature = "std-surface")]
        value_types: std::collections::BTreeMap::new(),
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

/// Spec 3.3 table (ruling section 5): each row is a Q16.16 dense element. The
/// expected i32 values come from exact rational arithmetic (the ruling's
/// reference script), never from the compiler. `(row, literal, Some(q))` must
/// lower to `q`; `None` must be refused by name (`Q16ConstantOutOfRange`).
const Q16_ROWS: &[(u8, &str, Option<i32>)] = &[
    (1, "7.62939453125e-06", Some(0)),
    (1, "-7.62939453125e-06", Some(0)),
    (2, "2.288818359375e-05", Some(2)),
    (2, "-2.288818359375e-05", Some(-2)),
    (3, "3.814697265625e-05", Some(2)),
    (3, "-3.814697265625e-05", Some(-2)),
    (4, "7.629394531250002e-06", Some(1)),
    (4, "7.629394531249999e-06", Some(0)),
    (5, "32767.99998474121", Some(i32::MAX)),
    (5, "32767.999992370602", Some(i32::MAX)),
    (6, "32767.999992370605", None),
    (6, "32768.0", None),
    (7, "-32768.0", Some(i32::MIN)),
    (7, "-32768.000007629395", Some(i32::MIN)),
    (8, "-32768.0000076294", None),
    (8, "-32768.00001525879", None),
    (9, "1e999", None),
    (9, "-1e999", None),
    (10, "5e-324", Some(0)),
    (10, "-5e-324", Some(0)),
    (10, "2.2250738585072014e-308", Some(0)),
    (10, "-0.0", Some(0)),
    (11, "0.1", Some(6554)),
    (11, "1e+30", None),
    (11, "-1e+30", None),
    (11, "1.7976931348623157e+308", None),
];

fn q16_dense(literal: &str) -> Result<Vec<u64>, String> {
    let src = format!("fn main() -> i64 {{ let t: Tensor<q16, [2]> = [{literal}, 1]; return 0; }}");
    let ast = libmind::parser::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let ir = libmind::eval::lower_to_ir(&ast).map_err(|e| e.to_string())?;
    let (dtype, _, data) = first_dense(&ir).ok_or("no dense constant")?;
    assert_eq!(dtype, DType::Q16, "{literal}");
    Ok(data)
}

#[test]
fn q16_dense_elements_follow_spec_3_3() {
    let mut red = Vec::new();
    for &(row, literal, want) in Q16_ROWS {
        let got = q16_dense(literal);
        let ok = match (want, &got) {
            // The integer literal 1 is Q(1.0) = 65536 (spec 3.3).
            (Some(q), Ok(data)) => *data == [q as u32 as u64, 65536],
            (None, Err(e)) => e.starts_with("Q16ConstantOutOfRange"),
            _ => false,
        };
        if !ok {
            red.push(format!("row {row} `{literal}`: want {want:?}, got {got:?}"));
        }
    }
    assert!(red.is_empty(), "spec 3.3 rows red:\n{}", red.join("\n"));
    // The emitted constant survives mic@3 exactly.
    let src = "fn main() -> i64 { let t: Tensor<q16, [1]> = [0.1]; return 0; }";
    let ir = libmind::eval::lower_to_ir(&libmind::parser::parse(src).unwrap()).unwrap();
    let parsed = parse_mic3(&emit_mic3(&ir)).expect("mic@3 re-parses");
    assert_eq!(first_dense(&parsed).map(|d| d.2), Some(vec![6554]));
}

/// Lower `let t: <ty> = <lit>;` and return the first dense constant's shape and
/// element bits, or the refusal text.
fn lowered_dense(ty: &str, lit: &str) -> Result<(DType, Vec<ShapeDim>, Vec<u64>), String> {
    let src = format!("fn main() -> i64 {{ let t: {ty} = {lit}; return 0; }}");
    let ast = libmind::parser::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let ir = libmind::eval::lower_to_ir(&ast).map_err(|e| e.to_string())?;
    first_dense(&ir).ok_or_else(|| "no dense constant".to_string())
}

fn f32_bits(values: &[f32]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits() as u64).collect()
}

/// A nested constant literal flattens row-major into one dense constant of the
/// declared shape. The expected bits come from the flat element list the test
/// writes out, never from the compiler. Ranks 2 and 3, f32 and i64, and the
/// `linear_regression` weights `[4, 1]`, `[3, 4]` and `[3, 1]`.
#[test]
fn nested_dense_literals_flatten_row_major() {
    let w = [0.1f32, 0.2, -0.1, 0.3];
    let x = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let y = [2.0f32, 6.0, 10.0];
    let ints = |v: &[i64]| v.iter().map(|n| *n as u64).collect::<Vec<u64>>();
    type Row = (&'static str, &'static str, DType, Vec<usize>, Vec<u64>);
    let rows: Vec<Row> = vec![
        (
            "diff tensor<f32[4, 1]>",
            "[[0.1], [0.2], [-0.1], [0.3]]",
            DType::F32,
            vec![4, 1],
            f32_bits(&w),
        ),
        (
            "diff tensor<f32[3, 4]>",
            "[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]",
            DType::F32,
            vec![3, 4],
            f32_bits(&x),
        ),
        (
            "diff tensor<f32[3, 1]>",
            "[[2.0], [6.0], [10.0]]",
            DType::F32,
            vec![3, 1],
            f32_bits(&y),
        ),
        (
            "tensor<i64[2, 2]>",
            "[[1, 2], [3, 0 - 4]]",
            DType::I64,
            vec![2, 2],
            ints(&[1, 2, 3, -4]),
        ),
        (
            "tensor<i64[2, 2, 2]>",
            "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            DType::I64,
            vec![2, 2, 2],
            ints(&[1, 2, 3, 4, 5, 6, 7, 8]),
        ),
    ];
    for (ty, lit, dtype, dims, bits) in rows {
        let (d, s, data) = lowered_dense(ty, lit).unwrap_or_else(|e| panic!("{ty} {lit}: {e}"));
        let want: Vec<ShapeDim> = dims.iter().map(|n| ShapeDim::Known(*n)).collect();
        assert_eq!((d, s, &data), (dtype, want, &bits), "{ty} {lit}");
    }
    // The constant survives mic@3 exactly.
    let ast = libmind::parser::parse(
        "fn main() -> i64 { let t: tensor<i64[2, 2]> = [[1, 2], [3, 4]]; return 0; }",
    )
    .unwrap();
    let ir = libmind::eval::lower_to_ir(&ast).unwrap();
    let parsed = parse_mic3(&emit_mic3(&ir)).expect("mic@3 re-parses");
    assert_eq!(first_dense(&parsed).map(|d| d.2), Some(vec![1, 2, 3, 4]));
}

/// A nested literal that is not a regular nest of constants of the declared
/// shape is refused by name: never padded, never truncated, never stored as 0.
#[test]
fn nested_dense_literals_refuse_by_name() {
    let rows: [(&str, &str, &str); 6] = [
        ("tensor<i64[2, 1]>", "[[x], [2]]", "NonConstantElement"),
        ("tensor<i64[2, 3]>", "[[1, 2], [3, 4]]", "DenseLeafCount"),
        ("tensor<i64[2, 2]>", "[[1, 2], [3]]", "RaggedNest"),
        ("tensor<i64[2, 2]>", "[[1, 2, 3, 4]]", "RaggedNest"),
        ("tensor<i64[2]>", "[[1], [2]]", "RaggedNest"),
        ("tensor<i64[2, 2]>", "[[1, 2], 3]", "RaggedNest"),
    ];
    for (ty, lit, name) in rows {
        // `x` is a runtime value in scope for the non-constant row.
        let src = format!("fn main() -> i64 {{ let x: i64 = 3; let t: {ty} = {lit}; return 0; }}");
        let ast = libmind::parser::parse(&src).unwrap();
        let got = libmind::eval::lower_to_ir(&ast).map_err(|e| e.to_string());
        assert!(
            matches!(&got, Err(e) if e.starts_with(name)),
            "{ty} {lit}: want {name}, got {:?}",
            got.map(|_| "built")
        );
    }
}
