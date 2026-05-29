// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 Phase 2 (partial) — equals-form flag parsing
//! `--name=value` + the `string_starts_with` / `string_slice_from`
//! helpers that back it.
//!
//! Tests resolve the new public surface through the bundled-stdlib
//! type-checker path used by the project loader; coverage of the
//! actual byte-level parsing semantics lives in the conformance
//! suite alongside the other std-surface modules.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_cli_equals_form`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::ast::Module;
use libmind::parser;
use libmind::project::module_table::build_module_table;
use libmind::project::stdlib::parsed_stdlib_modules;
use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

fn build_table_with_stdlib() -> libmind::project::module_table::ModuleTable {
    let stdlib: Vec<(String, Module)> = parsed_stdlib_modules();
    let refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    build_module_table(&refs)
}

#[test]
fn string_starts_with_and_slice_from_resolve() {
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    let h = string_new()\n\
                    let n = string_new()\n\
                    let r = string_starts_with(h, n)\n\
                    let sv = string_slice_from(h, 5)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "string_starts_with(String, String) -> i64 and \
         string_slice_from(String, i64) -> String must resolve; got {:?}",
        diags
    );
}

#[test]
fn args_flag_value_equals_form_consumer_resolves() {
    // The realistic agent-CLI shape: query a flag value not knowing
    // whether the user wrote `--out value` or `--out=value`.
    // args_flag_value handles both — a single call site covers both.
    let table = build_table_with_stdlib();
    let consumer = "use std.cli\n\
                    use std.string\n\
                    let a = args_new()\n\
                    let long = string_new()\n\
                    let short = string_new()\n\
                    let v = args_flag_value(a, long, short)\n\
                    let n = string_len(v)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "args_flag_value still resolves after the equals-form extension; got {:?}",
        diags
    );
}

#[test]
fn string_starts_with_phase_b_rejects_wrong_arity() {
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    let s = string_new()\n\
                    let bad = string_starts_with(s)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "Phase B arity check must reject string_starts_with(String)"
    );
}
