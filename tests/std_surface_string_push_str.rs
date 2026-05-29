// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 — `string_push_str` parse + resolve.
//!
//! Appending one String to another is the last building block agent-CLI
//! code needs before it can build multi-segment messages without
//! dropping to byte-by-byte loops.  This test drives the resolver
//! through the bundled-stdlib type-checker path.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_string_push_str`.

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
fn string_push_str_resolves_against_bundled_stdlib() {
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    let a = string_new()\n\
                    let b = string_new()\n\
                    let c = string_push_str(a, b)\n\
                    let n = string_len(c)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "string_push_str(String, String) -> String must resolve; got {:?}",
        diags
    );
}

#[test]
fn string_push_str_composes_with_push_i64_and_ansi() {
    // The realistic agent-CLI shape: build an ANSI-coloured status
    // line piece by piece — escape, label, value, reset — using all
    // three appender flavours in sequence.
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    use std.io\n\
                    let line = string_new()\n\
                    let label = string_new()\n\
                    let l1 = string_push_ansi_sgr(line,  sgr_fg_cyan())\n\
                    let l2 = string_push_str(l1, label)\n\
                    let l3 = string_push_i64(l2, 42)\n\
                    let l4 = string_push_ansi_sgr(l3, sgr_reset())\n\
                    let n = string_len(l4)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "agent-CLI status-line shape must resolve end-to-end; got {:?}",
        diags
    );
}

#[test]
fn string_push_str_phase_b_rejects_wrong_arity() {
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    let s = string_new()\n\
                    let bad = string_push_str(s)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "Phase B arity check must reject string_push_str(String)"
    );
}
