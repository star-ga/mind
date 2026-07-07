// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase B — per-arg signature matching on imported `pub fn`s.
//!
//! Phase A (shipped in v0.4.0) wired `use std.vec` so calls to
//! `vec_new()` / `vec_push(...)` etc. type-check loosely as
//! `ScalarI64`-returning, with no arg-count or arg-type validation.
//! Phase B records each `pub fn`'s full signature in the module
//! table and walks the args against the declaration at the call
//! site.
//!
//! What this suite covers:
//!
//! 1. Correct calls pass — `vec_get(0, 1)` (declared `(Vec, i64) -> i64`)
//!    type-checks because both args resolve to `ScalarI64` under the
//!    Phase-B widening rule (Named structs map to i64 ABI; literals
//!    widen i32 ↔ i64).
//! 2. Wrong arity errors with a specific message.
//! 3. Wrong type errors when the actual arg can't be coerced to the
//!    declared param type.
//! 4. Return-type fidelity — `file_read(...)` returns `i64`, and the
//!    Phase-B path threads that through rather than the Phase-A
//!    fallback `ScalarI64` placeholder.
//! 5. Phase-A fallback survives — when the imported module has an
//!    `export { ... }` block surface (no captured signatures), the
//!    Phase-A loose typing still resolves calls.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_use_import_phase_b`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::parser;
use libmind::project::module_table::{ModuleTable, build_module_table};
use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

const VEC_MIND_SRC: &str = include_str!("../std/vec.mind");
const IO_MIND_SRC: &str = include_str!("../std/io.mind");

fn build_stdlib_table() -> ModuleTable {
    let vec_ast = parser::parse(VEC_MIND_SRC).expect("std/vec.mind must parse");
    let io_ast = parser::parse(IO_MIND_SRC).expect("std/io.mind must parse");
    build_module_table(&[
        ("std.vec".to_string(), &vec_ast),
        ("std.io".to_string(), &io_ast),
    ])
}

#[test]
fn correct_arity_and_types_pass() {
    let table = build_stdlib_table();
    // vec_get(v: Vec, i: i64) -> i64 — under the Phase B widening
    // rule, integer literals (ScalarI32) coerce to ScalarI64 and
    // Named("Vec") maps to ScalarI64 too (Option-C heap ABI).
    let consumer_src = "use std.vec\nlet x = vec_get(0, 1)\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        diags.is_empty(),
        "vec_get(0, 1) must type-check under Phase B signature matching; got {:?}",
        diags
    );
}

#[test]
fn wrong_arity_errors_at_call_site() {
    let table = build_stdlib_table();
    // vec_new() takes zero args; passing one must fail.
    let consumer_src = "use std.vec\nlet x = vec_new(42)\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "vec_new(42) must error under Phase B (declared arity 0)"
    );
    let joined: String = diags.iter().map(|d| format!("{:?}", d)).collect();
    assert!(
        joined.contains("expects 0 argument") || joined.contains("0 argument"),
        "expected an arity-0 error message; got {}",
        joined
    );
}

#[test]
fn too_many_args_errors_with_specific_message() {
    let table = build_stdlib_table();
    // vec_get(v, i) takes 2 args; passing 3 must fail.
    let consumer_src = "use std.vec\nlet x = vec_get(0, 1, 2)\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "vec_get(0, 1, 2) must error under Phase B (declared arity 2)"
    );
}

#[test]
fn return_type_threads_through_to_let_binding() {
    let table = build_stdlib_table();
    // file_read(File, i64, i64, i64) -> i64.  The Phase-B path
    // returns the declared ret_type, so the `let` annotation must
    // accept the imported declaration's return shape.  We test
    // round-trip — if the call succeeds, the return type was used.
    let consumer_src = "use std.io\nlet n: i64 = file_read(0, 0, 64, 0)\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        diags.is_empty(),
        "file_read should type-check with i64 args + i64 annotation; got {:?}",
        diags
    );
}

#[test]
fn explicit_export_block_donor_enforces_arity() {
    // An explicit `export { do_stuff }` block whose module ALSO defines
    // `fn do_stuff` locally now captures that signature into `exported_fns`
    // (RFC 0007 cross-module resolution), so a wrong-arity call across the
    // module boundary is rejected — the same guarantee the `pub fn` path
    // gives, and exactly what lets a sibling module's `q16_sub(a, b)`
    // type-check against the real arity. (Historically this path fell back to
    // Phase-A loose typing and accepted any arity; capturing the local
    // signature is strictly more correct and consistent with the `pub fn`
    // arity tests above.)
    let donor_src = "export { do_stuff }\nfn do_stuff(x: i64) -> i64 { x }\n";
    let donor_ast = parser::parse(donor_src).expect("donor must parse");
    let table = build_module_table(&[("crate.donor".to_string(), &donor_ast)]);
    let consumer_src = "use crate.donor\nlet r = do_stuff(1, 2, 3, 99)\n";
    let consumer_ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&consumer_ast, consumer_src, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "do_stuff(1,2,3,99) must error: the explicit-export donor declares arity 1"
    );
    let joined: String = diags.iter().map(|d| format!("{d:?}")).collect();
    assert!(
        joined.contains("expects 1 argument"),
        "expected an arity-1 error for the explicit-export-block donor; got {joined}"
    );
}

#[test]
fn imported_fn_with_no_return_type_defaults_to_i64() {
    // A pub fn with no `->` declaration has ret_type = None.
    // Phase-B should still accept calls and return ScalarI64.
    let donor_src = "pub fn noisy() { }\n";
    let donor_ast = parser::parse(donor_src).expect("donor must parse");
    let table = build_module_table(&[("crate.donor".to_string(), &donor_ast)]);
    let consumer_src = "use crate.donor\nlet _r = noisy()\n";
    let consumer_ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&consumer_ast, consumer_src, None, &env, &table);
    assert!(
        diags.is_empty(),
        "ret_type=None should yield ScalarI64; got {:?}",
        diags
    );
}

#[test]
fn phase_d2_named_struct_param_named_in_arity_error() {
    // Phase D2 (light): when arity mismatches against a fn whose
    // parameters include a Named struct, the error message must
    // preserve the struct's name rather than collapsing it to the
    // i64 ABI lowering. The compatibility check itself stays
    // permissive (Named structs accept i64 values under Option-C);
    // this is purely an error-message-clarity contract.
    //
    // We trigger a per-arg widening failure by passing a tensor
    // expression where a Vec was declared. The actual ValueType
    // (Tensor) doesn't widen to ScalarI64, so the per-arg check
    // fires and the error message gets exercised.
    let donor_src = "pub struct Vec { addr: i64, len: i64, cap: i64 }\n\
         pub fn vec_set(v: Vec, i: i64, x: i64) -> i64 { x }\n";
    let donor_ast = parser::parse(donor_src).expect("donor must parse");
    let table = build_module_table(&[("crate.donor".to_string(), &donor_ast)]);
    // A tensor literal expression won't widen to i64 under the
    // Phase-B compatibility rule, so the error path fires for
    // argument 0.
    let consumer_src = "use crate.donor\n\
                        let t: tensor<f32[3]> = zeros([3])\n\
                        let r = vec_set(t, 0, 99)\n";
    let consumer_ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&consumer_ast, consumer_src, None, &env, &table);
    let combined = diags
        .iter()
        .map(|d| format!("{:?}", d))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        combined.contains("expects Vec"),
        "Phase D2 error must name the Vec parameter; got:\n{}",
        combined
    );
    assert!(
        combined.contains("heap-record i64 addr"),
        "Phase D2 error must mention the heap-record ABI; got:\n{}",
        combined
    );
}
