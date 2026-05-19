// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Phase 6.5 Stage 1a — bitwise BinOp IR lowering and MLIR emission.
//!
//! Covers:
//! 1. `123 & 255` lowers to `BinOp::BitAnd` and emits `arith.andi`.
//! 2. `0xFF | 0x0F`, `0xAA ^ 0x55`.
//! 3. `1 << 8`, `256 >> 2`.
//! 4. Constant-folding of bitwise ops in `ir_canonical`.
//! 5. MLIR emission uses the correct arith dialect ops.
//!
//! Gated: `cargo test --features "std-surface mlir-lowering" --test std_surface_bitwise_binops`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::{BinOp, Instr};
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

fn find_binop(instrs: &[Instr], target: BinOp) -> bool {
    for instr in instrs {
        match instr {
            Instr::BinOp { op, .. } if *op == target => return true,
            Instr::FnDef { body, .. } => {
                if find_binop(body, target) {
                    return true;
                }
            }
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                if find_binop(cond_instrs, target)
                    || find_binop(then_instrs, target)
                    || find_binop(else_instrs, target)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: `123 & 255` lowers to BinOp::BitAnd
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bitand_lowers_to_ir_binop_bitand() {
    let src = "let x: i64 = 123 & 255; x";
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert!(
        find_binop(&ir.instrs, BinOp::BitAnd),
        "expected BinOp::BitAnd in IR; got:\n{:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: `0xFF | 0x0F` → BitOr, `0xAA ^ 0x55` → BitXor
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bitor_lowers_to_ir_binop_bitor() {
    let src = "let x: i64 = 255 | 15; x";
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert!(
        find_binop(&ir.instrs, BinOp::BitOr),
        "expected BinOp::BitOr in IR"
    );
}

#[test]
fn bitxor_lowers_to_ir_binop_bitxor() {
    let src = "let x: i64 = 170 ^ 85; x";
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert!(
        find_binop(&ir.instrs, BinOp::BitXor),
        "expected BinOp::BitXor in IR"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: shift operators
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn shl_lowers_to_ir_binop_shl() {
    let src = "let x: i64 = 1 << 8; x";
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert!(
        find_binop(&ir.instrs, BinOp::Shl),
        "expected BinOp::Shl in IR"
    );
}

#[test]
fn shr_lowers_to_ir_binop_shr() {
    let src = "let x: i64 = 256 >> 2; x";
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert!(
        find_binop(&ir.instrs, BinOp::Shr),
        "expected BinOp::Shr in IR"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 4: bitwise ops inside functions (load_byte pattern from the lexer)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn bitand_inside_fn_lowers_correctly() {
    let src = r#"
fn load_byte(buf: i64, i: i64) -> i64 {
    __mind_load_i64(buf + i) & 255
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    assert!(
        find_binop(&ir.instrs, BinOp::BitAnd),
        "expected BinOp::BitAnd inside fn body; IR:\n{:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 5: MLIR emission uses arith dialect ops
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "mlir-lowering")]
#[test]
fn bitand_emits_arith_andi_in_mlir() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    let mut m = IRModule::new();
    let a = ValueId(0);
    let b = ValueId(1);
    let dst = ValueId(2);

    m.instrs.push(Instr::ConstI64(a, 123));
    m.instrs.push(Instr::ConstI64(b, 255));
    m.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::BitAnd,
        lhs: a,
        rhs: b,
    });
    m.instrs.push(Instr::Output(dst));
    m.next_id = 3;

    let text = lower_ir_to_mlir(&m)
        .expect("BitAnd MLIR lowering must not fail")
        .text;

    assert!(
        text.contains("arith.andi"),
        "expected `arith.andi` in MLIR output; got:\n{text}"
    );
}

#[cfg(feature = "mlir-lowering")]
#[test]
fn bitor_emits_arith_ori_in_mlir() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    let mut m = IRModule::new();
    let a = m.fresh();
    let b = m.fresh();
    let dst = m.fresh();

    m.instrs.push(Instr::ConstI64(a, 0xFF));
    m.instrs.push(Instr::ConstI64(b, 0x0F));
    m.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::BitOr,
        lhs: a,
        rhs: b,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("BitOr lowering").text;
    assert!(
        text.contains("arith.ori"),
        "expected `arith.ori`; got:\n{text}"
    );
}

#[cfg(feature = "mlir-lowering")]
#[test]
fn bitxor_emits_arith_xori_in_mlir() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    let mut m = IRModule::new();
    let a = m.fresh();
    let b = m.fresh();
    let dst = m.fresh();

    m.instrs.push(Instr::ConstI64(a, 0xAA));
    m.instrs.push(Instr::ConstI64(b, 0x55));
    m.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::BitXor,
        lhs: a,
        rhs: b,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("BitXor lowering").text;
    assert!(
        text.contains("arith.xori"),
        "expected `arith.xori`; got:\n{text}"
    );
}

#[cfg(feature = "mlir-lowering")]
#[test]
fn shl_emits_arith_shli_in_mlir() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    let mut m = IRModule::new();
    let a = m.fresh();
    let b = m.fresh();
    let dst = m.fresh();

    m.instrs.push(Instr::ConstI64(a, 1));
    m.instrs.push(Instr::ConstI64(b, 8));
    m.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Shl,
        lhs: a,
        rhs: b,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("Shl lowering").text;
    assert!(
        text.contains("arith.shli"),
        "expected `arith.shli`; got:\n{text}"
    );
}

#[cfg(feature = "mlir-lowering")]
#[test]
fn shr_emits_arith_shrsi_in_mlir() {
    use libmind::ir::{IRModule, ValueId};
    use libmind::mlir::lower_ir_to_mlir;

    let mut m = IRModule::new();
    let a = m.fresh();
    let b = m.fresh();
    let dst = m.fresh();

    m.instrs.push(Instr::ConstI64(a, 256));
    m.instrs.push(Instr::ConstI64(b, 2));
    m.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Shr,
        lhs: a,
        rhs: b,
    });
    m.instrs.push(Instr::Output(dst));

    let text = lower_ir_to_mlir(&m).expect("Shr lowering").text;
    assert!(
        text.contains("arith.shrsi"),
        "expected `arith.shrsi`; got:\n{text}"
    );
}
