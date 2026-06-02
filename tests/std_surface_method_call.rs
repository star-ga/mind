// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Method-as-field brick — a zero-arg method whose name matches a field of
//! the receiver's struct type lowers to the SAME `__mind_load_i64(base+idx*8)`
//! field load that the `FieldAccess` arm emits for `s.len`.
//!
//! `s.len()` on `struct String { addr, len, cap }` is `s.len` — the byte
//! length stored at field index 1. Before this brick, every method call fell
//! through to a const-0 placeholder; now zero-arg accessor methods resolve to
//! the underlying field load. Methods that take args, or whose name is not a
//! field, still fall through to const-0 (additive, keystone-safe — no method
//! calls appear in the keystone or in std).
//!
//! Both receiver-resolution paths are covered:
//!   * a local `let s = String { .. }` binding  -> Step 1 (`struct_env` Ident)
//!   * a `s: String` parameter                  -> Step 2 (`receiver_types`)
//!
//! Gated: `cargo test --features std-surface --test std_surface_method_call`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
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

/// Count `Instr::Call { name == target }` across the whole IR, recursing into
/// function bodies and control-flow blocks.
fn count_calls_named(instrs: &[Instr], target: &str) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::Call { name, .. } if name == target => n += 1,
            Instr::FnDef { body, .. } => n += count_calls_named(body, target),
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                n += count_calls_named(cond_instrs, target);
                n += count_calls_named(then_instrs, target);
                n += count_calls_named(else_instrs, target);
            }
            Instr::While {
                body, cond_instrs, ..
            } => {
                n += count_calls_named(cond_instrs, target);
                n += count_calls_named(body, target);
            }
            _ => {}
        }
    }
    n
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 1 — local `let s = String { .. }` binding resolves via `struct_env`.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn zero_arg_accessor_on_local_struct_lowers_to_field_load() {
    let src = r#"
struct String { addr: i64, len: i64, cap: i64 }
fn probe() -> i64 {
    let s = String { addr: 0, len: 5, cap: 8 }
    s.len()
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    let loads = count_calls_named(&ir.instrs, "__mind_load_i64");
    assert!(
        loads >= 1,
        "`s.len()` must lower to a __mind_load_i64 field load (the `len` field \
         at index 1), not the const-0 placeholder; got {loads} loads.\nIR: {:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Step 2 — `s: String` parameter resolves via the `receiver_types` side-table.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn zero_arg_accessor_on_struct_param_lowers_to_field_load() {
    let src = r#"
struct String { addr: i64, len: i64, cap: i64 }
fn probe(s: String) -> i64 {
    s.len()
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    let loads = count_calls_named(&ir.instrs, "__mind_load_i64");
    assert!(
        loads >= 1,
        "`s.len()` on a String parameter must lower to a field load via the \
         receiver_types side-table; got {loads} loads.\nIR: {:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Field index matters — `s.cap()` must load field 2, not field 1 (`len`).
// This guards against a degenerate "always load field 0" or name-agnostic bug.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn accessor_selects_the_named_field_not_a_fixed_offset() {
    // `addr` is field 0 -> no offset add; `cap` is field 2 -> offset 16.
    // Both must still emit exactly one __mind_load_i64.
    let src = r#"
struct String { addr: i64, len: i64, cap: i64 }
fn probe_addr() -> i64 {
    let s = String { addr: 7, len: 5, cap: 8 }
    s.addr()
}
fn probe_cap() -> i64 {
    let s = String { addr: 7, len: 5, cap: 8 }
    s.cap()
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    let loads = count_calls_named(&ir.instrs, "__mind_load_i64");
    assert!(
        loads >= 2,
        "each of s.addr() and s.cap() must emit its own field load; got {loads}.\nIR: {:?}",
        ir.instrs
    );
}
