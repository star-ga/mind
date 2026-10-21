// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Tests for the frozen native-profile admission predicate.

use super::*;
use crate::ir::{Instr, ValueId};

/// Test-side mirror of [`profile_frozen_admits`] for a bare instruction slice.
/// It builds the defined-function set with the SAME `collect_fn_names` pass the
/// public entry uses, so a test can never accidentally prove a property under a
/// different admission context than production.
fn admits(instrs: &[Instr]) -> Result<(), FrozenProfileRejection> {
    let mut defined = std::collections::BTreeSet::new();
    collect_fn_names(instrs, &mut defined);
    let module = IRModule::new();
    let mut aggs = std::collections::BTreeSet::new();
    admit_instrs_in(instrs, &defined, &module, &mut aggs)
}

fn fndef(body: Vec<Instr>) -> Instr {
    Instr::FnDef {
        name: "f".into(),
        params: vec![],
        ret_id: None,
        body,
        reap_threshold: None,
        semantic_types: None,
        #[cfg(feature = "std-surface")]
        value_types: Default::default(),
    }
}

fn matmul() -> Instr {
    Instr::MatMul {
        dst: ValueId(2),
        a: ValueId(0),
        b: ValueId(1),
    }
}

#[test]
fn scalar_body_is_admitted() {
    // fn f() -> i64 { return 42 } — pure scalar, in profile.
    let instrs = vec![fndef(vec![
        Instr::ConstI64(ValueId(0), 42),
        Instr::Return {
            value: Some(ValueId(0)),
        },
    ])];
    assert_eq!(admits(&instrs), Ok(()));
}

#[test]
fn tensor_matmul_is_rejected_by_name() {
    let err = admits(&[matmul()]).unwrap_err();
    assert_eq!(err.construct, "tensor.matmul");
}

#[test]
fn tensor_hidden_in_fn_body_is_still_rejected() {
    // The offender must not hide inside a nested function body.
    let err = admits(&[fndef(vec![matmul()])]).unwrap_err();
    assert_eq!(err.construct, "tensor.matmul");
}

#[test]
fn public_wrapper_delegates() {
    // The IRModule wrapper walks module.instrs — smoke it via the helper it
    // delegates to, so this file needs no full IRModule literal.
    assert!(admits(&[Instr::Output(ValueId(0))]).is_ok());
}

fn binop(op: BinOp) -> Instr {
    Instr::BinOp {
        dst: ValueId(2),
        op,
        lhs: ValueId(0),
        rhs: ValueId(1),
    }
}

#[test]
fn corpus_proven_binops_are_admitted() {
    // The 5-program readiness corpus exercises `+` (Add) and `<` (Lt, the for-loop
    // desugar); Sub/Mul/other-compares are substrate-invariant on the same axis.
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Lt, BinOp::Eq] {
        assert_eq!(admits(&[binop(op)]), Ok(()), "{op:?} should be admitted");
    }
}

#[test]
fn divergent_binops_are_rejected_by_name() {
    // The H5 exploit ops: Div/Mod (i64::MIN/-1, sign-of-remainder) are not in the
    // byte-identity corpus and must be named, not silently admitted.
    assert_eq!(
        admits(&[binop(BinOp::Div)]).unwrap_err().construct,
        "binop.div"
    );
    assert_eq!(
        admits(&[binop(BinOp::Mod)]).unwrap_err().construct,
        "binop.mod"
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn shifts_are_rejected_by_name() {
    // Shl (count >= width UB) / Shr (arith-vs-logical signedness split) — the ops
    // that keep the Q16.16 `(a*b)>>16` ecosystem tier out of the frozen profile.
    assert_eq!(
        admits(&[binop(BinOp::Shl)]).unwrap_err().construct,
        "binop.shl"
    );
    assert_eq!(
        admits(&[binop(BinOp::Shr)]).unwrap_err().construct,
        "binop.shr"
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn array_read_is_admitted_but_array_store_is_rejected() {
    // Row 12: reading a fixed array is corpus-proven (readiness-gate program
    // `array_idx_loop`); WRITING one is not ported to the frozen stage1.elf.
    // They share a constructor family, so admitting them together was the
    // constructor-vs-denotation defect — this pins the split.
    assert_eq!(
        admits(&[Instr::ArrayLoad {
            dst: ValueId(2),
            base: ValueId(0),
            index: ValueId(1),
        }]),
        Ok(())
    );
    assert_eq!(
        admits(&[Instr::ArrayStore {
            dst: ValueId(3),
            base: ValueId(0),
            index: ValueId(1),
            value: ValueId(2),
        }])
        .unwrap_err()
        .construct,
        "aggregate.array_store"
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn array_store_hidden_in_fn_body_is_rejected() {
    // Must not slip through the FnDef recursion, same as the Div case below.
    let err = admits(&[fndef(vec![Instr::ArrayStore {
        dst: ValueId(3),
        base: ValueId(0),
        index: ValueId(1),
        value: ValueId(2),
    }])])
    .unwrap_err();
    assert_eq!(err.construct, "aggregate.array_store");
}

#[test]
fn intrinsic_callee_is_rejected_but_user_callee_is_admitted() {
    // The aggregate ABI reaches the IR as intrinsic calls; pin BOTH directions
    // so a future intrinsic allowlist has to update this test deliberately.
    let call = |name: &str| Instr::legacy_call(ValueId(1), name, vec![]);
    assert_eq!(
        admits(&[call("__mind_alloc")]).unwrap_err().construct,
        "call.undefined_or_builtin"
    );
    // `f` IS defined by `fndef`, so a call to it is in profile.
    assert_eq!(admits(&[fndef(vec![]), call("f")]), Ok(()));
}

#[test]
fn divergent_binop_hidden_in_fn_body_is_rejected() {
    // The op-blind admission bug: a Div buried in a function body must not slip
    // through the FnDef recursion.
    let err = admits(&[fndef(vec![binop(BinOp::Div)])]).unwrap_err();
    assert_eq!(err.construct, "binop.div");
}

// ---- row 10 FLOAT_LANGUAGE_COVERAGE: the NaN-compare divergence ----

/// A float literal alone stays in profile: `float_as_i64` (`(2.5+4.0) as i64`) is a
/// readiness-corpus IN-profile program and carries no comparison.
#[test]
fn float_arithmetic_without_comparison_is_admitted() {
    let instrs = vec![fndef(vec![
        Instr::ConstF64(ValueId(0), 2.5),
        Instr::ConstF64(ValueId(1), 4.0),
        binop(BinOp::Add),
        Instr::Return {
            value: Some(ValueId(2)),
        },
    ])];
    assert_eq!(admits(&instrs), Ok(()));
}

/// A float comparison is now IN profile: the native lowering is IEEE-ordered
/// (swapped-operand `seta`/`setae` for `<`/`<=`, `sete AND setnp` for `==`,
/// `setne OR setp` for `!=`) and `ri_d1_frozen_profile_gate.py` proves native == MLIR
/// == an IEEE oracle over NaN / +-0.0 / +-inf. Before that proof this exact shape was
/// refused as `binop.compare_in_float_module` (old native exit 3 vs MLIR exit 0).
#[test]
fn comparison_in_a_float_module_is_admitted() {
    for op in [
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::Eq,
        BinOp::Ne,
    ] {
        let instrs = vec![fndef(vec![
            Instr::ConstF64(ValueId(0), 0.0),
            Instr::ConstF64(ValueId(1), f64::NAN),
            binop(op),
        ])];
        assert_eq!(
            admits(&instrs),
            Ok(()),
            "{op:?} over f64 operands is native == MLIR proven and must be admitted"
        );
    }
}

/// Relaxing the float-COMPARE arm must not relax the operators it never proved:
/// `Div` (and `Mod`) in a float module stay refused by name, exactly as in an
/// integer module.
#[test]
fn div_in_a_float_module_is_still_rejected() {
    for (op, label) in [(BinOp::Div, "binop.div"), (BinOp::Mod, "binop.mod")] {
        let instrs = vec![fndef(vec![
            Instr::ConstF64(ValueId(0), 1.0),
            Instr::ConstF64(ValueId(1), 3.0),
            binop(BinOp::Lt),
            binop(op),
        ])];
        assert_eq!(admits(&instrs).unwrap_err().construct, label);
    }
}

/// Integer comparisons keep the profile's proven for-loop (`array_idx_loop`).
#[test]
fn integer_comparison_without_float_is_still_admitted() {
    assert_eq!(admits(&[binop(BinOp::Lt)]), Ok(()));
}

/// A comparison over float params whose literal sits in ANOTHER fn (the shape the old
/// module-scoped taint existed for) is admitted too — the admission no longer depends
/// on where a float literal appears.
#[test]
fn float_in_caller_with_comparison_in_another_fn_is_admitted() {
    let caller = Instr::FnDef {
        name: "main".into(),
        params: vec![],
        ret_id: None,
        body: vec![Instr::ConstF64(ValueId(0), 1.5)],
        reap_threshold: None,
        semantic_types: None,
        #[cfg(feature = "std-surface")]
        value_types: Default::default(),
    };
    let callee = fndef(vec![binop(BinOp::Lt)]);
    assert_eq!(admits(&[caller, callee]), Ok(()));
}

fn source_admits(src: &str) -> Result<(), FrozenProfileRejection> {
    let module = crate::parser::parse(src).expect("test source parses");
    profile_frozen_admits_source(&module)
}

/// The IR half cannot see a declared f32: both literals lower to `ConstF64` and the
/// compare to a bare `Lt`, so the IR predicate admits it. The source half refuses it.
#[test]
fn declared_f32_comparison_is_rejected_on_the_source() {
    let src = "fn main() -> i64 { let x: f32 = 1.0; let y: f32 = 2.0; \
                   if x < y { return 1; } return 0; }";
    let module = crate::parser::parse(src).expect("parses");
    let ir = crate::eval::lower_to_ir(&module).expect("lowers");
    assert_eq!(profile_frozen_admits(&ir), Ok(()), "IR half is type-blind");
    assert_eq!(
        source_admits(src).unwrap_err().construct,
        "type.non_f64_float"
    );
}

/// Every spelling that puts a non-f64 float into the program is refused: params,
/// returns, casts, the call-form cast, struct fields and nested element types.
#[test]
fn every_non_f64_float_spelling_is_rejected() {
    for src in [
        "fn f(x: f32) -> i64 { return 0; }",
        "fn f() -> f32 { return 1.0; }",
        "fn main() -> i64 { let a: f64 = 1.0; let b = a as f32; return 0; }",
        "fn main() -> i64 { let a: f64 = 1.0; let b = f32(a); return 0; }",
        "struct P { x: f32 }",
        "fn f(xs: [f32; 4]) -> i64 { return 0; }",
        "fn f(x: f16) -> i64 { return 0; }",
        "fn f(x: bf16) -> i64 { return 0; }",
    ] {
        assert_eq!(
            source_admits(src).map_err(|r| r.construct),
            Err("type.non_f64_float"),
            "{src}"
        );
    }
}

/// Positive control: f64 and integer programs, including f64 comparisons, pass.
#[test]
fn f64_and_integer_sources_are_admitted() {
    for src in [
        "fn main() -> i64 { let x: f64 = 1.0; let y: f64 = 2.0; \
             if x < y { return 1; } return 0; }",
        "fn d(x: f64, y: f64) -> f64 { return x * y; }",
        "fn main() -> i64 { let n: i64 = 3; return n; }",
        "fn main() -> i64 { let a: f64 = 1.5; return a as i64; }",
        // A tensor element dtype is not a scalar width; the tensor itself is refused
        // elsewhere, under its own name.
        "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; }",
    ] {
        assert_eq!(source_admits(src), Ok(()), "{src}");
    }
}

#[cfg(feature = "std-surface")]
fn array_binop(op: BinOp) -> Vec<Instr> {
    vec![fndef(vec![
        Instr::ConstArray {
            dst: ValueId(0),
            name: None,
            values: vec![7],
        },
        Instr::ConstArray {
            dst: ValueId(1),
            name: None,
            values: vec![8],
        },
        Instr::BinOp {
            dst: ValueId(2),
            op,
            lhs: ValueId(0),
            rhs: ValueId(1),
        },
    ])]
}

/// Every binary operator on two array values is the same hole. The name is
/// the operand class, including operators that a scalar use already refuses.
#[cfg(feature = "std-surface")]
#[test]
fn every_array_binop_is_refused_by_operand_class() {
    for op in [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Mod,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
    ] {
        assert_eq!(
            admits(&array_binop(op)).unwrap_err().construct,
            "binop.array_operand",
            "{op:?} on array values must be named by the operand, not the operator"
        );
    }
}

/// `a[i] + b[i]` loads scalars. The array bases are not the binop operands.
#[cfg(feature = "std-surface")]
#[test]
fn element_add_of_arrays_is_admitted() {
    let instrs = vec![fndef(vec![
        Instr::ConstArray {
            dst: ValueId(0),
            name: None,
            values: vec![1, 2, 4],
        },
        Instr::ConstArray {
            dst: ValueId(1),
            name: None,
            values: vec![10, 20, 30],
        },
        Instr::ConstI64(ValueId(2), 0),
        Instr::ArrayLoad {
            dst: ValueId(3),
            base: ValueId(0),
            index: ValueId(2),
        },
        Instr::ArrayLoad {
            dst: ValueId(4),
            base: ValueId(1),
            index: ValueId(2),
        },
        Instr::BinOp {
            dst: ValueId(5),
            op: BinOp::Add,
            lhs: ValueId(3),
            rhs: ValueId(4),
        },
    ])];
    assert_eq!(admits(&instrs), Ok(()));
}

/// One array operand is enough. `a + 1` is the same miscompile class.
#[cfg(feature = "std-surface")]
#[test]
fn array_plus_scalar_is_refused() {
    let instrs = vec![fndef(vec![
        Instr::ConstArray {
            dst: ValueId(0),
            name: None,
            values: vec![7],
        },
        Instr::ConstI64(ValueId(1), 1),
        Instr::BinOp {
            dst: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
        },
    ])];
    assert_eq!(
        admits(&instrs).unwrap_err().construct,
        "binop.array_operand"
    );
}

/// The array is the right operand (`1 + a`, and the comparisons). A predicate
/// that inspects only the left operand admits every one of these: the left
/// value is a scalar constant.
#[cfg(feature = "std-surface")]
#[test]
fn scalar_op_array_is_refused() {
    for op in [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Mod,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
    ] {
        let instrs = vec![fndef(vec![
            Instr::ConstI64(ValueId(0), 1),
            Instr::ConstArray {
                dst: ValueId(1),
                name: None,
                values: vec![7, 8],
            },
            Instr::BinOp {
                dst: ValueId(2),
                op,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ])];
        assert_eq!(
            admits(&instrs).unwrap_err().construct,
            "binop.array_operand",
            "{op:?} with the array on the right"
        );
    }
}

/// `bytes[N]` indexing is a pointer add. The parameter is not an aggregate,
/// so the add stays on the scalar operator rule.
#[cfg(feature = "std-surface")]
#[test]
fn bytes_index_add_is_not_an_array_binop() {
    let mut module = IRModule::new();
    module.fn_signatures.insert(
        "main".into(),
        (
            vec![crate::ast::TypeAnn::Named("bytes[4]".into())],
            Some(crate::ast::TypeAnn::ScalarI64),
        ),
    );
    module.instrs.push(Instr::FnDef {
        name: "main".into(),
        params: vec![("v".into(), ValueId(0))],
        ret_id: None,
        body: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "v".into(),
                index: 0,
            },
            Instr::ConstI64(ValueId(1), 0),
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ],
        reap_threshold: None,
        semantic_types: None,
        value_types: Default::default(),
    });
    assert_eq!(profile_frozen_admits(&module), Ok(()));
}

/// An array parameter has no `ConstArray` instruction. The signature is
/// what makes `a + b` an aggregate binop.
#[cfg(feature = "std-surface")]
#[test]
fn array_param_add_is_refused() {
    let mut module = IRModule::new();
    let arr = crate::ast::TypeAnn::Array {
        element: Box::new(crate::ast::TypeAnn::ScalarI64),
        length: 2,
    };
    module.fn_signatures.insert(
        "main".into(),
        (vec![arr.clone(), arr], Some(crate::ast::TypeAnn::ScalarI64)),
    );
    module.instrs.push(Instr::FnDef {
        name: "main".into(),
        params: vec![("a".into(), ValueId(0)), ("b".into(), ValueId(1))],
        ret_id: None,
        body: vec![Instr::BinOp {
            dst: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
        }],
        reap_threshold: None,
        semantic_types: None,
        value_types: Default::default(),
    });
    assert_eq!(
        profile_frozen_admits(&module).unwrap_err().construct,
        "binop.array_operand"
    );
}

#[cfg(feature = "std-surface")]
fn lowered_construct(src: &str) -> Result<(), &'static str> {
    let module = crate::parser::parse(src).expect("parses");
    let ir = crate::eval::lower_to_ir(&module).expect("lowers");
    profile_frozen_admits(&ir).map_err(|r| r.construct)
}

/// The probe programs lower to a `ConstArray` binop. Element addition does
/// not, and the readiness-corpus loop that adds `a[i]` stays admitted.
#[cfg(feature = "std-surface")]
#[test]
fn lowered_array_add_is_refused_and_element_add_is_admitted() {
    assert_eq!(
        lowered_construct(
            "fn main() -> i64 { let a: [i64; 1] = [7]; let b: [i64; 1] = [8]; \
                 let c = a + b; return c[0]; }"
        ),
        Err("binop.array_operand")
    );
    assert_eq!(
        lowered_construct(
            "fn main() -> i64 { let a = [5, 6]; let b = [7, 8]; \
                 let c = a + b; return c[0]; }"
        ),
        Err("binop.array_operand")
    );
    assert_eq!(
        lowered_construct(
            "fn main() -> i64 { let a: [i64; 3] = [1, 2, 4]; \
                 let b: [i64; 3] = [10, 20, 30]; \
                 if a[0] + b[0] == 11 { if a[2] + b[2] == 34 { return 1; } } return 2; }"
        ),
        Ok(())
    );
    assert_eq!(
        lowered_construct(
            "fn main()->i64{let a=[1,2,3,4]; let mut s:i64=0; \
                 for i in 0..4 {s=s+a[i];} return s;}"
        ),
        Ok(())
    );
}

/// Source form of a scalar operator with the array on the right. Lowering must
/// keep that operand on the right, where the fence sees it.
#[cfg(feature = "std-surface")]
#[test]
fn lowered_scalar_op_array_is_refused() {
    assert_eq!(
        lowered_construct(
            "fn main() -> i64 { let a: [i64; 2] = [8, 6]; \
                 let c = 1 + a; return c[0]; }"
        ),
        Err("binop.array_operand")
    );
    for op in ["<", "<=", ">", ">=", "==", "!="] {
        let src = format!(
            "fn main() -> i64 {{ let a: [i64; 2] = [1, 2]; \
             if 1 {op} a {{ return 1; }} return 0; }}"
        );
        assert_eq!(
            lowered_construct(&src),
            Err("binop.array_operand"),
            "{op} with the array on the right"
        );
    }
}
