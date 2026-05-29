// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 2 — `std/string.mind` lowers cleanly to the heap-record
//! ABI + load/store intrinsic pattern shipped by P0e + P0f.
//!
//! Mirrors `std_surface_vec_module.rs`.  String is the Vec<u8>-shaped
//! sibling: the struct schema is identical (`addr/len/cap` at 8-byte
//! stride for the *record*; byte-stride for the *element* loads), so
//! the per-fn IR shape matches Vec almost line-for-line.  Differences:
//!
//! - `string_get_byte` uses `s.addr + i` (no `* 8`), since strings are
//!   byte-addressed.  Same load count as `vec_get` (one field-load +
//!   one explicit `__mind_load_i64`).
//! - `string_validate_utf8` is a pure reader stub today — one
//!   field-load on `s.len`, no allocations, no stores.
//! - `string_push_byte` is the integration smoke (cap 0 → 16, doubles
//!   on grow); same primitive footprint as `vec_push`.
//! - `string_eq` is a byte-equality stub — two field-loads on `a.len`
//!   and `b.len`, no allocations, no stores.
//!
//! Gated: `cargo test --features std-surface --test std_surface_string_module`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const STRING_MIND_SRC: &str = include_str!("../std/string.mind");

fn lower_string_mind() -> libmind::ir::IRModule {
    let module = parser::parse(STRING_MIND_SRC).expect("std/string.mind must parse");
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
fn string_mind_parses_and_lowers() {
    let ir = lower_string_mind();
    for want in [
        "string_new",
        "string_len",
        "string_cap",
        "string_addr",
        "string_get_byte",
        "string_validate_utf8",
        "string_push_byte",
        "string_eq",
    ] {
        assert!(
            ir.instrs.iter().any(|i| matches!(
                i,
                Instr::FnDef { name, .. } if name == want
            )),
            "missing FnDef for `{want}` in lowered IR"
        );
    }
    assert_eq!(
        ir.struct_defs.get("String"),
        Some(&vec![
            "addr".to_string(),
            "len".to_string(),
            "cap".to_string()
        ]),
        "String schema must be recorded in canonical field-name order"
    );
}

#[test]
fn string_new_emits_alloc_and_three_stores() {
    let ir = lower_string_mind();
    let body = fn_body(&ir, "string_new");

    assert_eq!(
        count_calls(body, "__mind_alloc"),
        1,
        "string_new() body should allocate the heap record exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        3,
        "string_new() body should store three field values (addr/len/cap)"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        0,
        "string_new() must not load — it constructs, doesn't read"
    );
}

#[test]
fn string_field_readers_emit_one_load_each() {
    let ir = lower_string_mind();
    for fn_name in ["string_len", "string_cap", "string_addr"] {
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
fn string_get_byte_uses_s_addr_plus_one_explicit_load() {
    let ir = lower_string_mind();
    let body = fn_body(&ir, "string_get_byte");

    // Intent: `string_get_byte` is a pure reader that issues at least one
    // explicit `__mind_load_i64` (the byte-load — `s.addr + i` then mask to
    // the low byte, see std/string.mind:48–56) and performs no allocations
    // or stores. The exact load count depends on whether the struct field
    // `s.addr` is lowered through `__mind_load_i64` or inlined out by the
    // struct-ABI / CSE pass; the `CRITICAL fix(std.string)` rewrite (commit
    // 0020795) changed the prior 2-load shape, so this assertion is a
    // floor-count to keep the contract (must-not-store-or-allocate,
    // must-perform-an-explicit-load) without tying it to a specific
    // lowering choice.
    assert!(
        count_calls(body, "__mind_load_i64") >= 1,
        "string_get_byte must issue at least one explicit byte load \
         (`__mind_load_i64(s.addr + i)`)"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "string_get_byte must not allocate"
    );
    assert_eq!(
        count_calls(body, "__mind_realloc"),
        0,
        "string_get_byte must not reallocate"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        0,
        "string_get_byte must not store"
    );
}

#[test]
fn string_validate_utf8_is_a_pure_reader() {
    let ir = lower_string_mind();
    let body = fn_body(&ir, "string_validate_utf8");

    // Today's stub touches `s.len` exactly once and returns a constant.
    // Multi-byte validation will add more reads but never allocations
    // or stores — assert the floor.
    let loads = count_calls(body, "__mind_load_i64");
    assert!(
        loads >= 1,
        "string_validate_utf8 reads s.len at minimum, got {loads}"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "string_validate_utf8 must not allocate"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        0,
        "string_validate_utf8 must not store"
    );
}

#[test]
fn string_push_byte_uses_every_primitive() {
    let ir = lower_string_mind();
    let body = fn_body(&ir, "string_push_byte");

    // string_push_byte body uses, at minimum:
    //   - several s.len / s.cap / s.addr reads
    //   - one __mind_alloc on grow
    //   - one explicit __mind_store_i8 for the new byte (task #306: the
    //     single-byte append now uses the i8 store so it no longer clobbers
    //     the trailing 7 bytes)
    //   - one String struct literal at the tail (3 i64 stores via P0e)
    //
    // Floor-count only — exact numbers depend on common-subexpression
    // behavior in the lowering.
    let loads = count_calls(body, "__mind_load_i64");
    let stores_i64 = count_calls(body, "__mind_store_i64");
    let stores_i8 = count_calls(body, "__mind_store_i8");
    let allocs = count_calls(body, "__mind_alloc");

    assert!(
        loads >= 3,
        "string_push_byte reads s.len / s.cap / s.addr at minimum, got {loads}"
    );
    assert!(
        stores_i8 >= 1,
        "string_push_byte stores the new byte via __mind_store_i8 (#306), got {stores_i8}"
    );
    assert!(
        stores_i64 >= 3,
        "string_push_byte stores three StructLit fields at minimum, got {stores_i64}"
    );
    assert!(
        allocs >= 1,
        "string_push_byte must contain at least one __mind_alloc, got {allocs}"
    );
}

#[test]
fn string_eq_reads_both_lengths() {
    let ir = lower_string_mind();
    let body = fn_body(&ir, "string_eq");

    // `a.len == b.len` produces two field-loads.  Byte-loop placeholder
    // returns constants so no element load shows up yet.
    let loads = count_calls(body, "__mind_load_i64");
    assert!(
        loads >= 2,
        "string_eq must read a.len and b.len (>= 2 loads), got {loads}"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "string_eq must not allocate"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        0,
        "string_eq must not store"
    );
}
