// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 2 — `std/io.mind` lowers cleanly on the
//! `__mind_read` / `__mind_write` intrinsic pair plus the P0e+P0f
//! struct ABI.  Verifies the I/O surface bottoms out into the
//! generic `Instr::Call` arm (no special-case lowering needed).
//!
//! Per-fn shape:
//! - `stdin/stdout/stderr` — 1 __mind_alloc + 1 __mind_store_i64
//!   (single-field StructLit), 0 loads.
//! - `file_fd` — 1 __mind_load_i64 (FieldAccess on the
//!   `f: File` parameter), 0 alloc, 0 store.
//! - `file_read` — 1 __mind_load_i64 for `f.fd`, 1 call to
//!   `__mind_read`, 0 stores.
//! - `file_write` — 1 __mind_load_i64 for `f.fd`, 1 call to
//!   `__mind_write`, 0 stores.
//! - `print_bytes / eprint_bytes / read_stdin_bytes` — no struct
//!   accesses, just 1 call to the i64 intrinsic.
//!
//! Gated: `cargo test --features std-surface --test std_surface_io_module`.

#![cfg(feature = "std-surface")]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const IO_MIND_SRC: &str = include_str!("../std/io.mind");

fn lower_io_mind() -> libmind::ir::IRModule {
    let module = parser::parse(IO_MIND_SRC).expect("std/io.mind must parse");
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
fn io_mind_parses_and_lowers() {
    let ir = lower_io_mind();
    for want in [
        "stdin",
        "stdout",
        "stderr",
        "file_fd",
        "file_read",
        "file_write",
        "print_bytes",
        "eprint_bytes",
        "read_stdin_bytes",
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
        ir.struct_defs.get("File"),
        Some(&vec!["fd".to_string()]),
        "File schema must record the single `fd` field"
    );
}

#[test]
fn stream_handles_construct_a_single_field_struct() {
    let ir = lower_io_mind();
    for fn_name in ["stdin", "stdout", "stderr"] {
        let body = fn_body(&ir, fn_name);
        assert_eq!(
            count_calls(body, "__mind_alloc"),
            1,
            "{fn_name}: a stream-handle constructor must allocate one heap record"
        );
        assert_eq!(
            count_calls(body, "__mind_store_i64"),
            1,
            "{fn_name}: a stream-handle constructor must store exactly one field"
        );
        assert_eq!(
            count_calls(body, "__mind_load_i64"),
            0,
            "{fn_name}: a pure constructor must not load"
        );
    }
}

#[test]
fn file_fd_reads_one_field() {
    let ir = lower_io_mind();
    let body = fn_body(&ir, "file_fd");
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        1,
        "file_fd: FieldAccess on struct-typed param must produce exactly one __mind_load_i64"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "file_fd: pure reader must not allocate"
    );
    assert_eq!(
        count_calls(body, "__mind_store_i64"),
        0,
        "file_fd: pure reader must not store"
    );
}

#[test]
fn file_read_routes_to_mind_read_intrinsic() {
    let ir = lower_io_mind();
    let body = fn_body(&ir, "file_read");
    assert_eq!(
        count_calls(body, "__mind_read"),
        1,
        "file_read body must call __mind_read exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        1,
        "file_read should read `f.fd` exactly once via __mind_load_i64"
    );
}

#[test]
fn file_write_routes_to_mind_write_intrinsic() {
    let ir = lower_io_mind();
    let body = fn_body(&ir, "file_write");
    assert_eq!(
        count_calls(body, "__mind_write"),
        1,
        "file_write body must call __mind_write exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        1,
        "file_write should read `f.fd` exactly once via __mind_load_i64"
    );
}

#[test]
fn print_bytes_writes_to_stdout_fd_directly() {
    let ir = lower_io_mind();
    let body = fn_body(&ir, "print_bytes");
    // No FieldAccess — fd is the literal `1`, so we expect zero
    // __mind_load_i64 calls.  Just one __mind_write.
    assert_eq!(
        count_calls(body, "__mind_write"),
        1,
        "print_bytes body must call __mind_write exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        0,
        "print_bytes must not field-load — fd is a constant literal"
    );
    assert_eq!(
        count_calls(body, "__mind_alloc"),
        0,
        "print_bytes must not allocate"
    );
}

#[test]
fn eprint_bytes_writes_to_stderr_fd_directly() {
    let ir = lower_io_mind();
    let body = fn_body(&ir, "eprint_bytes");
    assert_eq!(
        count_calls(body, "__mind_write"),
        1,
        "eprint_bytes body must call __mind_write exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        0,
        "eprint_bytes must not field-load — fd is a constant literal"
    );
}

#[test]
fn read_stdin_bytes_calls_mind_read_directly() {
    let ir = lower_io_mind();
    let body = fn_body(&ir, "read_stdin_bytes");
    assert_eq!(
        count_calls(body, "__mind_read"),
        1,
        "read_stdin_bytes body must call __mind_read exactly once"
    );
    assert_eq!(
        count_calls(body, "__mind_load_i64"),
        0,
        "read_stdin_bytes must not field-load — fd is a constant literal"
    );
}
