// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Native memory addresses retain their integer base and byte offsets in MLIR.

#![cfg(all(
    feature = "std-surface",
    any(feature = "mlir-lowering", feature = "mlir-build")
))]

use libmind::ast::TypeAnn;
use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use libmind::mlir::lower_ir_to_mlir;
use std::collections::BTreeMap;

fn constant(module: &mut IRModule, value: i64) -> ValueId {
    let dst = module.fresh();
    module.instrs.push(Instr::ConstI64(dst, value));
    dst
}

fn call(module: &mut IRModule, name: &str, args: Vec<ValueId>) -> ValueId {
    let dst = module.fresh();
    module.instrs.push(call_instr(dst, name, args));
    dst
}

fn call_instr(dst: ValueId, name: &str, args: Vec<ValueId>) -> Instr {
    Instr::Call {
        dst,
        name: name.to_string(),
        args,
        resolved_callee: None,
    }
}

fn binary(module: &mut IRModule, op: BinOp, lhs: ValueId, rhs: ValueId) -> ValueId {
    let dst = module.fresh();
    module.instrs.push(Instr::BinOp { dst, op, lhs, rhs });
    dst
}

fn reference(id: ValueId) -> String {
    format!("%{}", id.0)
}

/// Follow a memory operation's pointer to its integer root. Intermediate SSA
/// names are deliberately unconstrained; the dataflow and GEP element type are
/// the contract under test.
fn address_parts(text: &str, pointer: &str) -> (String, Vec<String>) {
    let definitions: BTreeMap<&str, &str> = text
        .lines()
        .filter_map(|line| line.trim().split_once(" = "))
        .collect();
    let mut cursor = pointer;
    let mut offsets = Vec::new();
    for _ in 0..definitions.len() {
        let operation = definitions
            .get(cursor)
            .unwrap_or_else(|| panic!("missing pointer definition {cursor}:\n{text}"));
        if let Some(integer) = operation.strip_prefix("llvm.inttoptr ") {
            assert!(integer.ends_with(" : i64 to !llvm.ptr"), "{operation}");
            offsets.reverse();
            return (
                integer.split_whitespace().next().unwrap().to_string(),
                offsets,
            );
        }
        let gep = operation
            .strip_prefix("llvm.getelementptr ")
            .unwrap_or_else(|| panic!("unexpected address operation {operation}:\n{text}"));
        let (parent, indices) = gep.split_once('[').expect("GEP indices");
        let (offset, signature) = indices.split_once(']').expect("GEP index end");
        assert_eq!(signature, " : (!llvm.ptr, i64) -> !llvm.ptr, i8");
        assert!(
            !offset.contains(','),
            "one byte offset per GEP: {operation}"
        );
        offsets.push(offset.to_string());
        cursor = parent;
    }
    panic!("cyclic or unterminated pointer chain {pointer}:\n{text}");
}

fn assert_address(text: &str, pointer: &str, root: &str, offsets: &[String]) {
    assert_eq!(
        address_parts(text, pointer),
        (root.to_string(), offsets.to_vec()),
        "unexpected address dataflow for {pointer}:\n{text}"
    );
}

fn loop_body_root(text: &str) -> &str {
    text.lines()
        .map(str::trim)
        .find(|line| line.starts_with("^while_body_0("))
        .expect("loop body declaration")
        .split_once('(')
        .unwrap()
        .1
        .split(':')
        .next()
        .unwrap()
}

#[test]
fn memory_addresses_use_byte_geps_for_every_load_and_store_width() {
    for width in [8, 16, 32, 64] {
        for store in [false, true] {
            let mut module = IRModule::new();
            let base = call(&mut module, "address_base", vec![]);
            let offset = call(&mut module, "address_offset", vec![]);
            let negative = constant(&mut module, -7);
            let inner = binary(&mut module, BinOp::Add, base, offset);
            let address = binary(&mut module, BinOp::Add, negative, inner);
            let name = format!("__mind_{}_i{width}", if store { "store" } else { "load" });
            let mut args = vec![address];
            if store {
                args.push(constant(&mut module, -1));
            }
            let result = call(&mut module, &name, args);
            module.instrs.push(Instr::Output(result));
            let text = lower_ir_to_mlir(&module)
                .expect("lower memory address")
                .text;
            let pointer = format!("%{}p{}", if store { "st" } else { "ld" }, result.0);
            assert_address(
                &text,
                &pointer,
                &reference(base),
                &[reference(offset), reference(negative)],
            );
            assert!(text.contains("{alignment = 1 : i64}"), "{text}");
            assert!(!text.contains("inbounds"), "{text}");
            assert!(!text.contains(&format!("func.call @{name}")), "{text}");
            assert!(
                !text.contains(&format!("func.func private @{name}")),
                "{text}"
            );
        }
    }
}

#[test]
fn memory_addresses_choose_nonconstant_base_at_each_add() {
    let mut module = IRModule::new();
    let base = call(&mut module, "address_base", vec![]);
    let low = constant(&mut module, i64::MIN);
    let high = constant(&mut module, i64::MAX);
    let inner = binary(&mut module, BinOp::Add, low, base);
    let address = binary(&mut module, BinOp::Add, high, inner);
    let result = call(&mut module, "__mind_load_i64", vec![address]);
    module.instrs.push(Instr::Output(result));
    let text = lower_ir_to_mlir(&module)
        .expect("lower signed offsets")
        .text;
    assert_address(
        &text,
        &format!("%ldp{}", result.0),
        &reference(base),
        &[reference(low), reference(high)],
    );
    assert!(!text.contains("inbounds"), "{text}");
}

#[test]
fn memory_addresses_preserve_unsigned_64_bit_add_chains() {
    let mut module = IRModule::new();
    for name in ["unsigned_base", "unsigned_offset"] {
        module.fn_signatures.insert(
            name.to_string(),
            (vec![], Some(TypeAnn::Named("u64".to_string()))),
        );
    }
    let base = call(&mut module, "unsigned_base", vec![]);
    let offset = call(&mut module, "unsigned_offset", vec![]);
    let tail = constant(&mut module, 7);
    let inner = binary(&mut module, BinOp::Add, base, offset);
    let address = binary(&mut module, BinOp::Add, tail, inner);
    let result = call(&mut module, "__mind_load_i64", vec![address]);
    module.instrs.push(Instr::Output(result));
    let text = lower_ir_to_mlir(&module)
        .expect("lower unsigned address")
        .text;
    assert_address(
        &text,
        &format!("%ldp{}", result.0),
        &reference(base),
        &[reference(offset), reference(tail)],
    );
}

#[test]
fn memory_addresses_keep_non_add_operations_opaque() {
    let mut module = IRModule::new();
    let base = call(&mut module, "address_base", vec![]);
    let offset = constant(&mut module, 8);
    let address = binary(&mut module, BinOp::Sub, base, offset);
    let result = call(&mut module, "__mind_load_i64", vec![address]);
    module.instrs.push(Instr::Output(result));
    let text = lower_ir_to_mlir(&module)
        .expect("lower opaque address")
        .text;
    assert_address(
        &text,
        &format!("%ldp{}", result.0),
        &reference(address),
        &[],
    );
}

#[test]
fn memory_addresses_do_not_reinterpret_narrow_or_boolean_add_operands() {
    for boolean in [false, true] {
        let mut module = IRModule::new();
        let base = call(&mut module, "address_base", vec![]);
        let first = constant(&mut module, 1);
        let second = constant(&mut module, 2);
        let operand = if boolean {
            binary(&mut module, BinOp::Lt, first, second)
        } else {
            module.fn_signatures.insert(
                "narrow_offset".to_string(),
                (vec![], Some(TypeAnn::ScalarI32)),
            );
            let narrow = call(&mut module, "narrow_offset", vec![]);
            binary(&mut module, BinOp::Add, narrow, second)
        };
        let address = binary(&mut module, BinOp::Add, base, operand);
        let result = call(&mut module, "__mind_load_i64", vec![address]);
        module.instrs.push(Instr::Output(result));
        let text = lower_ir_to_mlir(&module)
            .expect("lower coerced address")
            .text;
        assert_address(
            &text,
            &format!("%ldp{}", result.0),
            &reference(address),
            &[],
        );
    }
}

#[test]
fn memory_addresses_reuse_outer_add_chains_in_branches() {
    let mut module = IRModule::new();
    let base = call(&mut module, "address_base", vec![]);
    let offset = constant(&mut module, 8);
    let address = binary(&mut module, BinOp::Add, offset, base);
    let condition = constant(&mut module, 1);
    let then_result = module.fresh();
    let else_result = module.fresh();
    let result = module.fresh();
    module.instrs.push(Instr::If {
        cond_id: condition,
        cond_instrs: vec![],
        then_instrs: vec![call_instr(then_result, "__mind_load_i64", vec![address])],
        then_result,
        else_instrs: vec![call_instr(else_result, "__mind_load_i64", vec![address])],
        else_result,
        dst: result,
        branch_bindings: vec![],
        merges: vec![],
    });
    module.instrs.push(Instr::Output(result));
    let text = lower_ir_to_mlir(&module)
        .expect("lower branch addresses")
        .text;
    for loaded in [then_result, else_result] {
        assert_address(
            &text,
            &format!("%ldp{}", loaded.0),
            &reference(base),
            &[reference(offset)],
        );
    }
}

#[test]
fn memory_addresses_preserve_loop_carried_roots_and_outer_addresses() {
    let mut module = IRModule::new();
    let base = call(&mut module, "address_base", vec![]);
    let offset = constant(&mut module, 8);
    let outer = binary(&mut module, BinOp::Add, offset, base);
    let initial = binary(&mut module, BinOp::Add, base, offset);
    let fixed = binary(&mut module, BinOp::Add, initial, offset);
    let condition = module.fresh();
    let outer_load = module.fresh();
    let carried_load = module.fresh();
    let fixed_load = module.fresh();
    let advanced_load = module.fresh();
    let advanced = module.fresh();
    let exit = module.fresh();
    module.instrs.push(Instr::While {
        cond_id: condition,
        cond_instrs: vec![call_instr(condition, "__mind_load_i8", vec![outer])],
        body: vec![
            call_instr(outer_load, "__mind_load_i64", vec![outer]),
            call_instr(carried_load, "__mind_load_i64", vec![initial]),
            call_instr(fixed_load, "__mind_load_i64", vec![fixed]),
            Instr::BinOp {
                dst: advanced,
                op: BinOp::Add,
                lhs: initial,
                rhs: offset,
            },
            call_instr(advanced_load, "__mind_load_i64", vec![advanced]),
        ],
        live_vars: vec![("cursor".to_string(), advanced)],
        init_ids: vec![initial],
        exit_ids: vec![exit],
    });
    let text = lower_ir_to_mlir(&module)
        .expect("lower loop addresses")
        .text;
    for loaded in [condition, outer_load] {
        assert_address(
            &text,
            &format!("%ldp{}", loaded.0),
            &reference(base),
            &[reference(offset)],
        );
    }
    let body_root = loop_body_root(&text);
    assert_address(&text, &format!("%ldp{}", carried_load.0), body_root, &[]);
    assert_address(
        &text,
        &format!("%ldp{}", fixed_load.0),
        &reference(fixed),
        &[],
    );
    assert_address(
        &text,
        &format!("%ldp{}", advanced_load.0),
        body_root,
        &[reference(offset)],
    );
}

#[test]
fn memory_addresses_treat_carried_constant_initializers_as_runtime_bases() {
    let mut module = IRModule::new();
    let initial = constant(&mut module, 4096);
    let offset = call(&mut module, "address_offset", vec![]);
    let condition = module.fresh();
    let address = module.fresh();
    let loaded = module.fresh();
    let exit = module.fresh();
    module.instrs.push(Instr::While {
        cond_id: condition,
        cond_instrs: vec![Instr::ConstI64(condition, 0)],
        body: vec![
            Instr::BinOp {
                dst: address,
                op: BinOp::Add,
                lhs: initial,
                rhs: offset,
            },
            call_instr(loaded, "__mind_load_i64", vec![address]),
        ],
        live_vars: vec![("cursor".to_string(), address)],
        init_ids: vec![initial],
        exit_ids: vec![exit],
    });
    let text = lower_ir_to_mlir(&module)
        .expect("lower carried constant address")
        .text;
    assert_address(
        &text,
        &format!("%ldp{}", loaded.0),
        loop_body_root(&text),
        &[reference(offset)],
    );
}
