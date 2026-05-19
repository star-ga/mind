// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 2 — `std/vec.mind` lowers cleanly to the heap-record
//! ABI + load/store intrinsic pattern shipped by P0e + P0f.
//!
//! For each `pub fn` in `std/vec.mind`, asserts:
//!
//! - `vec_new`         emits one `__mind_alloc` + three `__mind_store_i64`
//!                     (the StructLit `Vec { addr: 0, len: 0, cap: 0 }`).
//! - `vec_len/cap/addr` emit one `__mind_load_i64` each (the FieldAccess
//!                     read of the struct-typed parameter — P0f Step 2
//!                     case (3) seeds the side-table from Param.ty).
//! - `vec_get/set`     emit one `__mind_load_i64` / `__mind_store_i64`
//!                     for the explicit intrinsic call PLUS one
//!                     `__mind_load_i64` for `v.addr`. Total per fn:
//!                     2 loads + 1 store for `vec_set`; 2 loads for
//!                     `vec_get`.
//! - `vec_push`        is the integration smoke — its body alone uses
//!                     every primitive `std.vec` needs: StructLit
//!                     allocation, FieldAccess reads on a struct-typed
//!                     parameter, conditional code, explicit intrinsic
//!                     calls, and a returned StructLit.
//!
//! Gated: `cargo test --features std-surface --test std_surface_vec_module`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const VEC_MIND_SRC: &str = include_str!("../std/vec.mind");

fn lower_vec_mind() -> libmind::ir::IRModule {
    let module = parser::parse(VEC_MIND_SRC).expect("std/vec.mind must parse");
    lower_to_ir(&module)
}

fn fn_body<'a>(ir: &'a libmind::ir::IRModule, name: &str) -> &'a [Instr] {
    ir.instrs
        .iter()
        .find_map(|i| match i {
            Instr::FnDef { name: n, body, .. } if n == name => Some(body.as_slice()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected FnDef with name `{name}` in lowered IR"))
}

fn count_calls(body: &[Instr], callee: &str) -> usize {
    let mut n = 0;
    for instr in body {
        match instr {
            Instr::Call { name, .. } if name == callee => n += 1,
            // Recurse into If and While sub-instruction streams so that
            // calls inside conditional/loop code are counted correctly
            // after Phase 6.5 Stage 1a moved them into Instr::If branches.
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                n += count_calls(cond_instrs, callee);
                n += count_calls(then_instrs, callee);
                n += count_calls(else_instrs, callee);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                n += count_calls(cond_instrs, callee);
                n += count_calls(body, callee);
            }
            _ => {}
        }
    }
    n
}

#[test]
fn vec_mind_parses_and_lowers() {
    // First load-bearing assertion: the file compiles end-to-end.
    let ir = lower_vec_mind();
    // And contains the seven expected pub fns.
    for want in [
        "vec_new", "vec_len", "vec_cap", "vec_addr", "vec_get", "vec_set", "vec_push",
    ] {
        assert!(
            ir.instrs.iter().any(|i| matches!(
                i,
                Instr::FnDef { name, .. } if name == want
            )),
            "missing FnDef for `{want}` in lowered IR"
        );
    }
    // And the Vec schema is in the registry.
    assert_eq!(
        ir.struct_defs.get("Vec"),
        Some(&vec![
            "addr".to_string(),
            "len".to_string(),
            "cap".to_string()
        ]),
        "Vec schema must be recorded in canonical field-name order"
    );
}

#[test]
fn vec_new_emits_alloc_and_three_stores() {
    let ir = lower_vec_mind();
    let body = fn_body(&ir, "vec_new");

    assert_eq!(
        count_calls(body, "__mind_alloc"),
        1,
        "vec_new() body should allocate the heap record exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        3,
        "vec_new() body should store three field values (addr/len/cap)"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        0,
        "vec_new() must not load — it constructs, doesn't read"
    );
}

#[test]
fn field_readers_emit_one_load_each() {
    let ir = lower_vec_mind();
    for fn_name in ["vec_len", "vec_cap", "vec_addr"] {
        let body = fn_body(&ir, fn_name);
        assert_eq!(
            count_calls(body, "__mind_load_i64"),
            1,
            "{fn_name}: FieldAccess on struct-typed param must produce exactly one __mind_load_i64"
        );
        assert_eq!(
            count_calls(body, "__mind_alloc"),
            0,
            "{fn_name}: a pure reader must not allocate"
        );
        assert_eq!(
            count_calls(body, "__mind_store_i64"),
            0,
            "{fn_name}: a pure reader must not store"
        );
    }
}

#[test]
fn vec_get_uses_v_addr_plus_one_explicit_load() {
    let ir = lower_vec_mind();
    let body = fn_body(&ir, "vec_get");

    // `v.addr` → one field-load. The explicit `__mind_load_i64(...)`
    // intrinsic call is the same callee, so the total count is 2.
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        2,
        "vec_get body should issue 2 loads (one for v.addr field read, one explicit element load)"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "vec_get must not allocate"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        0,
        "vec_get must not store"
    );
}

#[test]
fn vec_set_uses_v_addr_load_plus_one_explicit_store() {
    let ir = lower_vec_mind();
    let body = fn_body(&ir, "vec_set");

    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        1,
        "vec_set body should issue 1 load (for v.addr field read)"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        1,
        "vec_set body should issue 1 explicit __mind_store_i64"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "vec_set must not allocate"
    );
}

#[test]
fn vec_push_uses_every_primitive() {
    let ir = lower_vec_mind();
    let body = fn_body(&ir, "vec_push");

    // vec_push body uses, at minimum:
    //   - several v.len / v.cap / v.addr reads
    //   - one __mind_alloc on grow
    //   - one explicit __mind_store_i64 for the new element
    //   - one Vec struct literal at the tail (3 more stores via P0e)
    //
    // So we just assert the floor counts — exact numbers depend on
    // common-subexpression behavior in the lowering.
    let loads = count_calls(body, "__mind_load_i64");
    let stores = count_calls(body, "__mind_store_i64");
    let allocs = count_calls(body, "__mind_alloc");

    assert!(
        loads >= 3,
        "vec_push reads v.len / v.cap / v.addr at minimum, got {loads}"
    );
    assert!(
        stores >= 4,
        "vec_push stores element + three StructLit fields at minimum, got {stores}"
    );
    assert!(
        allocs >= 1,
        "vec_push must contain at least one __mind_alloc, got {allocs}"
    );
}
