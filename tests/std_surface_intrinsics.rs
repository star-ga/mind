// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 1+ — the `i64`-signed `__mind_*` intrinsics.
//!
//! Phase 0 (committed `8b2199f`) made `Instr::Call` reach MLIR.
//! Phase 1+ makes those names *known* to the type-checker with
//! fixed `(i64..) -> i64` signatures so the bottom of `std.vec` /
//! `std.string` / `std.map` / `std.io` can compile. Phase 1.5 added
//! `__mind_load_i64` / `__mind_store_i64` for scalar load/store at
//! address; Phase 1.6 (task #306) added `__mind_load_i8` /
//! `__mind_store_i8` for proper one-byte access (closing the
//! 8-byte-store-at-byte-offset heap-OOB landmine). Pure type-checker
//! tests use the AST directly; the end-to-end smoke runs the Phase-0
//! path so the contract is checked through to MLIR text.
//!
//! Gated: `cargo test --features std-surface,mlir-lowering
//! --test std_surface_intrinsics`.

#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]

use std::collections::HashMap;

use libmind::ast::{Literal, Module, Node, Span};
use libmind::ir::{IRModule, Instr, ValueId};
use libmind::mlir::lower_ir_to_mlir;
use libmind::type_checker::check_module_types;

fn lit_int(v: i64) -> Node {
    Node::Lit(Literal::Int(v), Span::new(0, 0))
}

fn call(name: &str, args: Vec<Node>) -> Node {
    Node::Call {
        callee: name.to_string(),
        args,
        span: Span::new(0, 0),
    }
}

fn module_of(stmts: Vec<Node>) -> Module {
    Module { items: stmts }
}

fn scalar(m: &mut IRModule, v: i64) -> ValueId {
    let id = m.fresh();
    m.instrs.push(Instr::ConstI64(id, v));
    id
}

// ── Type-checker: each intrinsic has the documented (i64..) -> i64 signature ──

#[test]
fn alloc_one_i64_arg_typechecks() {
    let module = module_of(vec![call("__mind_alloc", vec![lit_int(64)])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_alloc(64) must type-check; got {diags:#?}"
    );
}

#[test]
fn realloc_two_i64_args_typechecks() {
    let module = module_of(vec![call(
        "__mind_realloc",
        vec![lit_int(0xdead_beef), lit_int(128)],
    )]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_realloc/2 must accept; got {diags:#?}"
    );
}

#[test]
fn free_one_i64_arg_typechecks() {
    let module = module_of(vec![call("__mind_free", vec![lit_int(42)])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_free/1 must accept; got {diags:#?}"
    );
}

#[test]
fn read_four_i64_args_typechecks() {
    let module = module_of(vec![call(
        "__mind_read",
        vec![lit_int(1), lit_int(2), lit_int(3), lit_int(4)],
    )]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_read/4 must accept; got {diags:#?}"
    );
}

#[test]
fn write_four_i64_args_typechecks() {
    let module = module_of(vec![call(
        "__mind_write",
        vec![lit_int(1), lit_int(2), lit_int(3), lit_int(4)],
    )]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_write/4 must accept; got {diags:#?}"
    );
}

// ── Phase 1.5 (P0c) — scalar load/store at address ──

#[test]
fn load_i64_one_i64_arg_typechecks() {
    let module = module_of(vec![call("__mind_load_i64", vec![lit_int(0xdead_beef)])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_load_i64/1 must accept; got {diags:#?}"
    );
}

#[test]
fn store_i64_two_i64_args_typechecks() {
    let module = module_of(vec![call(
        "__mind_store_i64",
        vec![lit_int(0xdead_beef), lit_int(42)],
    )]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_store_i64/2 must accept; got {diags:#?}"
    );
}

// RFC 0005 Phase 1.6 (task #306) — single-byte ABI.

#[test]
fn load_i8_one_i64_arg_typechecks() {
    let module = module_of(vec![call("__mind_load_i8", vec![lit_int(0xdead_beef)])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_load_i8/1 must accept; got {diags:#?}"
    );
}

#[test]
fn store_i8_two_i64_args_typechecks() {
    let module = module_of(vec![call(
        "__mind_store_i8",
        vec![lit_int(0xdead_beef), lit_int(0x42)],
    )]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        diags.is_empty(),
        "__mind_store_i8/2 must accept; got {diags:#?}"
    );
}

// ── Wrong arity is a clear error that names the i64 ABI / phase 2+ ──

#[test]
fn alloc_wrong_arity_is_a_clear_error() {
    // alloc takes one arg — give it two
    let module = module_of(vec![call("__mind_alloc", vec![lit_int(1), lit_int(2)])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(!diags.is_empty(), "wrong arity must produce a diagnostic");
    let msg = &diags[0].message;
    assert!(
        msg.contains("__mind_alloc") && msg.contains("1 i64 argument"),
        "error must name the intrinsic + expected arity; got: {msg}"
    );
}

#[test]
fn write_wrong_arity_is_a_clear_error() {
    // write takes four args — give it three
    let module = module_of(vec![call(
        "__mind_write",
        vec![lit_int(1), lit_int(2), lit_int(3)],
    )]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(!diags.is_empty());
    let msg = &diags[0].message;
    assert!(
        msg.contains("__mind_write") && msg.contains("4 i64 argument"),
        "error must name the intrinsic + expected arity; got: {msg}"
    );
}

// ── Non-i64 arg (a tensor expression) is a clear error pointing to phase 2+ ──

#[test]
fn tensor_arg_to_intrinsic_is_rejected_with_phase_2_pointer() {
    // tensor.zeros(f32, (2,3)) — a Tensor — fed to __mind_alloc must error.
    let tensor_expr = Node::Call {
        callee: "tensor.zeros".to_string(),
        args: vec![
            Node::Lit(Literal::Ident("f32".into()), Span::new(0, 0)),
            Node::Tuple {
                elements: vec![lit_int(2), lit_int(3)],
                span: Span::new(0, 0),
            },
        ],
        span: Span::new(0, 0),
    };
    let module = module_of(vec![call("__mind_alloc", vec![tensor_expr])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(!diags.is_empty(), "tensor arg must be rejected");
    let msg = &diags[0].message;
    assert!(
        msg.contains("must be i64") && msg.contains("RFC 0005 phase 2+"),
        "error must name the i64 ABI + phase 2+ aggregate path; got: {msg}"
    );
}

// ── Unknown __mind_* prefix stays an honest "unsupported call" error ──
// (We do NOT silently accept anything that starts with __mind_ — only
// the documented five names are recognised by the registry.)
#[test]
fn unknown_mind_intrinsic_still_errors() {
    let module = module_of(vec![call("__mind_NOT_REAL", vec![lit_int(1)])]);
    let env = libmind::type_checker::TypeEnv::default();
    let diags = check_module_types(&module, "", &env);
    assert!(
        !diags.is_empty(),
        "unknown __mind_* name must NOT be silently accepted"
    );
    let msg = &diags[0].message;
    assert!(
        msg.contains("__mind_NOT_REAL") && msg.contains("unsupported"),
        "error must say `unsupported call`; got: {msg}"
    );
}

// ── End-to-end: IR `Instr::Call` for each intrinsic lowers via Phase 0 ──

#[test]
fn each_intrinsic_lowers_to_func_call_with_private_decl() {
    for (name, arity) in [
        ("__mind_alloc", 1usize),
        ("__mind_free", 1),
        ("__mind_load_i64", 1),
        ("__mind_load_i8", 1),
        ("__mind_read", 4),
        ("__mind_realloc", 2),
        ("__mind_store_i64", 2),
        ("__mind_store_i8", 2),
        ("__mind_write", 4),
    ] {
        let mut m = IRModule::new();
        let args: Vec<ValueId> = (0..arity).map(|i| scalar(&mut m, (i + 1) as i64)).collect();
        let dst = m.fresh();
        m.instrs.push(Instr::Call {
            dst,
            name: name.to_string(),
            args,
        });
        m.instrs.push(Instr::Output(dst));

        let text = lower_ir_to_mlir(&m).expect("lower").text;
        let call_needle = format!("func.call @{name}(%");
        let sig_tys: Vec<&str> = (0..arity).map(|_| "i64").collect();
        let call_sig = format!(") : ({}) -> i64", sig_tys.join(", "));
        let decl_needle = format!("func.func private @{name}({}) -> i64", sig_tys.join(", "));

        assert!(
            text.contains(&call_needle),
            "{name}: expected `{call_needle}` in:\n{text}"
        );
        assert!(
            text.contains(&call_sig),
            "{name}: expected `{call_sig}` in:\n{text}"
        );
        assert!(
            text.contains(&decl_needle),
            "{name}: expected `{decl_needle}` in:\n{text}"
        );
    }
}
