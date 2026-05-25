// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 Phase 2 — `args_subcommand` + `args_subcommand_eq`.
//!
//! Tests resolve the two new public functions through the bundled-
//! stdlib type-checker path.  Byte-level dispatch semantics live in
//! the conformance suite.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_cli_subcommand`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::ast::Module;
use libmind::parser;
use libmind::project::module_table::build_module_table;
use libmind::project::stdlib::parsed_stdlib_modules;
use libmind::type_checker::{check_module_types_with_modules, TypeEnv};

fn build_table_with_stdlib() -> libmind::project::module_table::ModuleTable {
    let stdlib: Vec<(String, Module)> = parsed_stdlib_modules();
    let refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    build_module_table(&refs)
}

#[test]
fn args_subcommand_resolves_against_bundled_stdlib() {
    let table = build_table_with_stdlib();
    let consumer = "use std.cli\n\
                    use std.string\n\
                    let a = args_new()\n\
                    let sub = args_subcommand(a)\n\
                    let n = string_len(sub)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "args_subcommand(Args) -> String must resolve; got {:?}",
        diags
    );
}

#[test]
fn args_subcommand_eq_realistic_dispatch_shape() {
    // The shape an agent CLI's main() would use: take an Args,
    // compare its first positional to a known set of verb names.
    let table = build_table_with_stdlib();
    let consumer = "use std.cli\n\
                    use std.string\n\
                    let a = args_new()\n\
                    let v_build = string_new()\n\
                    let v_run = string_new()\n\
                    let v_test = string_new()\n\
                    let is_build = args_subcommand_eq(a, v_build)\n\
                    let is_run = args_subcommand_eq(a, v_run)\n\
                    let is_test = args_subcommand_eq(a, v_test)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "dispatch-shape consumer must resolve; got {:?}",
        diags
    );
}

#[test]
fn args_subcommand_phase_b_rejects_wrong_arity() {
    let table = build_table_with_stdlib();
    let consumer = "use std.cli\nlet sub = args_subcommand()\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "Phase B arity check must reject args_subcommand()"
    );
}
