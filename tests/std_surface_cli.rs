// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 Phase 1 — `std.cli` parse + type-resolution smoke.
//!
//! The Tier-1 std.cli surface lands as a pure-MIND module bundled
//! through `parsed_stdlib_modules`. This test exercises the same
//! library-level path the project loader uses internally, on a real
//! consumer file that drives every public function the module ships:
//!
//!   1. `parsed_stdlib_modules()` must include `std.cli` and its
//!      AST must parse cleanly under `std-surface`.
//!   2. A consumer `use std.cli` reaches every exported function
//!      with the right arity / argument types / return types.
//!   3. The Phase-B signature-aware resolver catches an arity
//!      regression on `args_len` (one-arg only — no zero-arg form).
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_cli`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::ast::Module;
use libmind::parser;
use libmind::project::module_table::build_module_table;
use libmind::project::stdlib::parsed_stdlib_modules;
use libmind::type_checker::{check_module_types_with_modules, TypeEnv};

fn build_table_with_stdlib() -> (Vec<(String, Module)>, libmind::project::module_table::ModuleTable) {
    let stdlib: Vec<(String, Module)> = parsed_stdlib_modules();
    let refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);
    (stdlib, table)
}

#[test]
fn std_cli_is_in_the_bundle_and_parses() {
    let (modules, _) = build_table_with_stdlib();
    let cli = modules
        .iter()
        .find(|(p, _)| p == "std.cli")
        .expect("std.cli must appear in the bundled stdlib registry");
    // A parsed module is non-empty (the file has public functions).
    // The exact AST shape is checked by the consumer test below; here
    // we just gate on "parses + present in the registry".
    let _ = &cli.1;
}

#[test]
fn std_cli_consumer_resolves_full_public_surface() {
    let (_modules, table) = build_table_with_stdlib();

    // Touch every public function the module ships. Each call must
    // resolve through the Phase-B signature-aware resolver — arity
    // and per-arg types are validated.
    let consumer = "use std.cli\n\
                    use std.string\n\
                    let a0 = args_new()\n\
                    let s = string_new()\n\
                    let a1 = args_push(a0, s)\n\
                    let n = args_len(a1)\n\
                    let s2 = args_get(a1, 0)\n\
                    let end = args_flag_end_idx(a1)\n\
                    let empty = string_new()\n\
                    let has = args_has_flag(a1, s, empty)\n\
                    let vidx = args_flag_value_idx(a1, s, empty)\n\
                    let v = args_flag_value(a1, s, empty)\n\
                    let pos = args_first_positional_idx(a1)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "every std.cli public function must resolve via the bundled stdlib; got {:?}",
        diags
    );
}

#[test]
fn std_cli_phase_b_rejects_wrong_arity() {
    let (_modules, table) = build_table_with_stdlib();
    // args_len takes exactly one Args argument. Phase B should
    // reject the zero-arg call even though Phase A would let it
    // resolve by name.
    let consumer = "use std.cli\nlet n = args_len()\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "bundled std.cli must enforce Phase B arity check on args_len"
    );
}
