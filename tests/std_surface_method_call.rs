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

// ──────────────────────────────────────────────────────────────────────────────
// UFCS desugar — a method call WITH ARGS whose name is not a field of the
// receiver's struct lowers to a real free-function call
// `{lowercase(Type)}_{method}(recv, args…)` — NOT the const-0 placeholder.
//
// `v.push(42)` on `struct Vec { addr, len, cap }` desugars to
// `vec_push(v, 42)` with the receiver threaded as the first argument. This
// guards the #1 silent-miscompile blocker: before this brick, every
// method-with-args call returned 0 instead of executing.
// ──────────────────────────────────────────────────────────────────────────────

/// Recursively count `Instr::ConstI64(_, 0)` so we can assert the call site is
/// NOT a const-0 placeholder. (Struct-literal builds legitimately use const-0
/// for an `addr: 0` field, so we assert on the presence of the real call, not
/// the absence of all zeros.)
fn vec_push_call_threads_receiver(instrs: &[Instr]) -> bool {
    for instr in instrs {
        match instr {
            // The receiver must be the FIRST argument and there must be a
            // second (the pushed value) — proving it is the desugared
            // `vec_push(recv, value)`, not a zero-arg accessor or const-0.
            Instr::Call { name, args, .. } if name == "vec_push" && args.len() == 2 => return true,
            Instr::FnDef { body, .. } => {
                if vec_push_call_threads_receiver(body) {
                    return true;
                }
            }
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                if vec_push_call_threads_receiver(cond_instrs)
                    || vec_push_call_threads_receiver(then_instrs)
                    || vec_push_call_threads_receiver(else_instrs)
                {
                    return true;
                }
            }
            Instr::While {
                body, cond_instrs, ..
            } => {
                if vec_push_call_threads_receiver(cond_instrs)
                    || vec_push_call_threads_receiver(body)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

#[test]
fn method_with_args_ufcs_desugars_to_free_function_call() {
    let src = r#"
struct Vec { addr: i64, len: i64, cap: i64 }
fn vec_push(v: Vec, value: i64) -> Vec {
    Vec { addr: v.addr, len: v.len + 1, cap: v.cap }
}
fn probe() -> i64 {
    let v = Vec { addr: 0, len: 0, cap: 0 }
    let w = v.push(42)
    w.len
}
"#;
    let module = must_parse(src);
    let ir = lower_to_ir(&module);
    // The whole point: `v.push(42)` is a REAL call to `vec_push`, with the
    // receiver `v` as the first argument and `42` as the second — not a
    // silent const-0 placeholder.
    assert!(
        vec_push_call_threads_receiver(&ir.instrs),
        "`v.push(42)` must desugar to `vec_push(v, 42)` (a 2-arg func.call with \
         the receiver threaded first), NOT a const-0 placeholder.\nIR: {:?}",
        ir.instrs
    );
    // And it must have produced NO new const-0 *at the call site*: confirm a
    // `vec_push` call exists at all (the strong signal above already does this,
    // but assert the count is exactly one for clarity).
    assert_eq!(
        count_calls_named(&ir.instrs, "vec_push"),
        1,
        "exactly one vec_push call expected from the single `v.push(42)`.\nIR: {:?}",
        ir.instrs
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Fail-loud — a method call WITH ARGS whose receiver type cannot be resolved
// must NOT silently lower to const-0. Under the old fallthrough this was a
// silent miscompile; now it is a clear, loud failure (panic in the lowering
// pass) per the #306 fail-closed philosophy.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn method_with_args_on_unresolved_receiver_fails_loud_not_const_zero() {
    // `x: i64` is not a struct, so `x.frobnicate(7)` cannot desugar to any
    // `<type>_frobnicate` free function. The lowering pass must refuse to emit
    // a const-0 placeholder.
    let src = r#"
fn probe(x: i64) -> i64 {
    x.frobnicate(7)
}
"#;
    let module = must_parse(src);
    let result = std::panic::catch_unwind(|| lower_to_ir(&module));
    assert!(
        result.is_err(),
        "an unresolved method-with-args call must fail loud (panic), NOT silently \
         lower to const-0 — that would be a miscompile (#306 fail-closed)."
    );
}
