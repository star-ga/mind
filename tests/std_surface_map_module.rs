// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 2 — `std/map.mind` lowers cleanly to the heap-record
//! ABI + load/store intrinsic pattern shipped by P0e + P0f.
//!
//! Map is the parallel-array sibling to Vec / String — same primitive
//! footprint, but with two backing stores (`keys_addr` + `vals_addr`)
//! instead of one and a 4-field heap record instead of 3.
//!
//! Per-fn shape:
//! - `map_new`        1 __mind_alloc + 4 __mind_store_i64.
//! - `map_len/cap`    1 __mind_load_i64 each (FieldAccess on Map param).
//! - `map_keys_addr / map_vals_addr` same — pure field readers.
//! - `map_key_at / map_value_at`  2 loads (one for the field-load on
//!                                `m.keys_addr` or `m.vals_addr`, one
//!                                for the explicit element load).
//! - `map_insert`     ≥4 loads (m.len/cap/keys_addr/vals_addr), ≥2
//!                    __mind_alloc (one per backing array on grow),
//!                    ≥6 __mind_store_i64 (key + value + 4-field
//!                    StructLit at tail).
//!
//! Gated: `cargo test --features std-surface --test std_surface_map_module`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const MAP_MIND_SRC: &str = include_str!("../std/map.mind");

fn lower_map_mind() -> libmind::ir::IRModule {
    let module = parser::parse(MAP_MIND_SRC).expect("std/map.mind must parse");
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
    body.iter()
        .filter(|i| matches!(i, Instr::Call { name, .. } if name == callee))
        .count()
}

#[test]
fn map_mind_parses_and_lowers() {
    let ir = lower_map_mind();
    for want in [
        "map_new",
        "map_len",
        "map_cap",
        "map_keys_addr",
        "map_vals_addr",
        "map_key_at",
        "map_value_at",
        "map_insert",
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
        ir.struct_defs.get("Map"),
        Some(&vec![
            "keys_addr".to_string(),
            "vals_addr".to_string(),
            "len".to_string(),
            "cap".to_string(),
        ]),
        "Map schema must be recorded in canonical field-name order"
    );
}

#[test]
fn map_new_emits_alloc_and_four_stores() {
    let ir = lower_map_mind();
    let body = fn_body(&ir, "map_new");

    assert_eq!(
        count_calls(body, "__mind_alloc"),
        1,
        "map_new() body should allocate the heap record exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        4,
        "map_new() body should store four field values (keys_addr/vals_addr/len/cap)"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        0,
        "map_new() must not load — it constructs, doesn't read"
    );
}

#[test]
fn map_field_readers_emit_one_load_each() {
    let ir = lower_map_mind();
    for fn_name in ["map_len", "map_cap", "map_keys_addr", "map_vals_addr"] {
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
fn map_key_at_and_value_at_use_field_plus_explicit_load() {
    let ir = lower_map_mind();
    for fn_name in ["map_key_at", "map_value_at"] {
        let body = fn_body(&ir, fn_name);
        // `m.keys_addr` / `m.vals_addr` → one field-load. The explicit
        // `__mind_load_i64(...)` intrinsic call is the same callee, so
        // the total count is 2.
        assert_eq!(
            count_calls(body, "__mind_load_i64"),
            2,
            "{fn_name}: should issue 2 loads (one for field-read, one explicit element load)"
        );
        assert_eq!(
            count_calls(body, "__mind_alloc"),
            0,
            "{fn_name}: pure reader must not allocate"
        );
        assert_eq!(
            count_calls(body, "__mind_store_i64"),
            0,
            "{fn_name}: pure reader must not store"
        );
    }
}

#[test]
fn map_insert_uses_every_primitive_twice() {
    let ir = lower_map_mind();
    let body = fn_body(&ir, "map_insert");

    // map_insert body uses, at minimum:
    //   - reads on m.len / m.cap / m.keys_addr / m.vals_addr (>= 4)
    //   - one __mind_alloc per backing array on grow (>= 2)
    //   - one explicit __mind_store_i64 per side (>= 2)
    //   - four StructLit field stores at the tail (>= 4 more)
    //   - total stores >= 6
    //
    // Floor-count only — exact numbers depend on common-subexpression
    // behavior in the lowering.
    let loads = count_calls(body, "__mind_load_i64");
    let stores = count_calls(body, "__mind_store_i64");
    let allocs = count_calls(body, "__mind_alloc");

    assert!(
        loads >= 4,
        "map_insert reads m.len/cap/keys_addr/vals_addr at minimum, got {loads}"
    );
    assert!(
        stores >= 6,
        "map_insert stores key + value + four StructLit fields at minimum, got {stores}"
    );
    assert!(
        allocs >= 2,
        "map_insert must allocate both backing arrays on grow, got {allocs}"
    );
}
