//! Release-readiness P1.1 — the runnable-artifact ABI gate.
//!
//! `CompileProducts.runnable_blockers` records constructs that parse +
//! type-check but cannot lower to a *correct* runnable artifact in the shipped
//! i64-scalar ABI. The CLI fails loud on these for `--emit-obj`/`--emit-shared`
//! while leaving `mindc check`/`--emit-ir`/`--emit-mlir` (where these are valid
//! *types*) untouched.
//!
//! These tests assert the gate logic directly at the library boundary:
//!   * the silent-miscompile constructs are recorded as blockers, and
//!   * the i64 subset (the self-host/std core) records ZERO blockers, so the
//!     keystone / gap corpus / mic@3 flip are unaffected.

use libmind::pipeline::{CompileOptions, compile_source_with_name};

/// Compile and return the number of runnable-artifact blockers (the source must
/// parse + type-check — these are valid programs that merely don't lower).
fn blockers(src: &str) -> usize {
    let opts = CompileOptions::default();
    let products = compile_source_with_name(src, None, &opts).unwrap_or_else(|e| {
        panic!("expected source to parse + type-check, got {e:?}\nsrc:\n{src}")
    });
    products.runnable_blockers.len()
}

// ---- blocked: the silent sub-i64-ABI miscompiles ----

#[test]
fn i32_param_is_blocked() {
    assert!(blockers("fn f(a: i32) -> i64 { 0 }") >= 1);
}

#[test]
fn u32_return_is_blocked() {
    assert!(blockers("fn f(a: i64) -> u32 { 0 }") >= 1);
}

#[test]
fn sub_i64_struct_field_is_blocked() {
    assert!(blockers("struct P { a: i32, b: i32 }\nfn f() -> i64 { 0 }") >= 1);
}

#[test]
fn bool_struct_field_is_blocked() {
    assert!(blockers("struct Flag { on: bool }\nfn f() -> i64 { 0 }") >= 1);
}

// ---- NOT blocked: the i64 subset + the scalars that lower correctly ----

#[test]
fn i64_program_is_clean() {
    assert_eq!(blockers("fn add(a: i64, b: i64) -> i64 { a + b }"), 0);
}

#[test]
fn f64_signature_is_clean() {
    // f64 lowers correctly (arith.addf) — must NOT be gated.
    assert_eq!(blockers("fn fadd(a: f64, b: f64) -> f64 { a + b }"), 0);
}

#[test]
fn bool_return_is_clean() {
    // bool RETURN lowers via i1->i64 zero-extend — correct, not gated.
    assert_eq!(blockers("fn is_pos(a: i64) -> bool { a > 0 }"), 0);
}

#[test]
fn i64_struct_is_clean() {
    // The all-i64 pointer-handle record the self-host stack depends on.
    assert_eq!(
        blockers("struct Pt { x: i64, y: i64 }\nfn f() -> i64 { 0 }"),
        0
    );
}
