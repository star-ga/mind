// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Enum soundness gate (audit finding 20).
//!
//! A function declared with a BARE SCALAR return that returns a payload-carrying
//! enum constructor on some path was a SILENT MISCOMPILE: the enum value is a
//! heap-record handle (an i64 address), so `divide(5, 0)` returning `Res::Err(0)`
//! leaked a raw pointer as the result. The type-checker has no
//! declared-return-vs-body unification, so it compiled and ran wrong. The
//! `abi_gate` fail-closed gate now flags this on the emit path (loud file:line
//! error, no `.so`). It is inert for any module with no `enum` declaration, so
//! the keystone (0 enums) stays byte-identical.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports"
//!                   --test enum_soundness`

#![cfg(feature = "std-surface")]

use libmind::pipeline::{CompileOptions, compile_source_with_name};

const CODE: &str = "lower::enum_handle_in_scalar_return";

/// The source must compile + type-check, then be recorded as a runnable blocker
/// (a loud `--emit-shared` error) for returning an enum handle as a bare scalar.
fn rejects(src: &str) {
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .unwrap_or_else(|e| panic!("should parse + type-check, got {e:?}\nsrc:\n{src}"));
    assert!(
        p.runnable_blockers.iter().any(|d| d.code == CODE),
        "expected `{CODE}` blocker, got: {:?}\nsrc:\n{src}",
        p.runnable_blockers
    );
}

/// The source must compile with NO enum-handle-return blocker (a legitimate
/// program — fieldless return, enum-typed return, match-to-scalar, ctor-in-arg).
fn accepts(src: &str) {
    let p = compile_source_with_name(src, None, &CompileOptions::default())
        .unwrap_or_else(|e| panic!("should parse + type-check, got {e:?}\nsrc:\n{src}"));
    assert!(
        !p.runnable_blockers.iter().any(|d| d.code == CODE),
        "must NOT be gated, got: {:?}\nsrc:\n{src}",
        p.runnable_blockers
    );
}

#[test]
fn enum_handle_in_scalar_return_is_gated() {
    // The audit case: a payload ctor returned (nested in an if) where i64 is
    // declared — leaked a raw heap pointer; now a loud fail-closed error.
    rejects(
        "enum Res { Ok(i64), Err(i64) }\npub fn divide(a: i64, b: i64) -> i64 { if b == 0 { return Res::Err(0) }\n return a / b }",
    );
    // Tail-position enum ctor on one branch, scalar on the other, declared -> i64.
    rejects(
        "enum Res { Ok(i64), Err(i64) }\npub fn g(x: i64) -> i64 { if x > 0 { Res::Ok(x) } else { x } }",
    );
}

#[test]
fn legitimate_enum_programs_still_compile() {
    // Fieldless match -> i64 (bare ordinal tags, not handles).
    accepts(
        "enum Mode { On, Off }\npub fn pick(m: Mode) -> i64 { match m { Mode::On => 1, Mode::Off => 0 } }",
    );
    // Enum-typed return is the CORRECT shape (the must-not-false-positive twin
    // of the rejected case — differs only in `-> Res` vs `-> i64`).
    accepts(
        "enum Res { Ok(i64), Err(i64) }\npub fn divide(a: i64, b: i64) -> Res { if b == 0 { return Res::Err(0) }\n return Res::Ok(a) }",
    );
    // Payload ctor built then MATCHED to a scalar before returning.
    accepts(
        "enum Opt { Some(i64), None }\nfn build(x: i64) -> Opt { Opt::Some(x) }\npub fn probe() -> i64 { let o = build(41)\n match o { Opt::Some(v) => v, Opt::None => 0 } }",
    );
    // Enum ctor in ARGUMENT position (not a return position).
    accepts(
        "enum Mode { On, Off }\nfn use_m(m: Mode) -> i64 { 1 }\npub fn caller() -> i64 { use_m(Mode::On) }",
    );
}
