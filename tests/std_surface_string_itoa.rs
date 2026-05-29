// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 — `string_push_i64` + `string_push_ansi_sgr`
//! parse + resolve.
//!
//! The integer-to-decimal-ASCII helper in std.string is the bedrock
//! every text-output path needs (ANSI escape composition, log
//! formatting, agent CLI status lines).  The ANSI helper in std.io
//! composes the framing `ESC [ <code> m` on top of it.  This test
//! drives both through the bundled-stdlib type-checker path so a
//! Phase-B arity / per-arg-type regression on either fails CI.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_string_itoa`.

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
fn string_push_i64_resolves_against_bundled_stdlib() {
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    let s = string_new()\n\
                    let s1 = string_push_i64(s, 0)\n\
                    let s2 = string_push_i64(s1, 42)\n\
                    let s3 = string_push_i64(s2, 0 - 7)\n\
                    let n = string_len(s3)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "string_push_i64(String, i64) -> String must resolve; got {:?}",
        diags
    );
}

#[test]
fn string_push_ansi_sgr_composes_itoa_with_escape_framing() {
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    use std.io\n\
                    let s = string_new()\n\
                    let s1 = string_push_ansi_sgr(s,  sgr_fg_red())\n\
                    let s2 = string_push_byte(s1, 88)\n\
                    let s3 = string_push_ansi_sgr(s2, sgr_reset())\n\
                    let n = string_len(s3)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "string_push_ansi_sgr(String, i64) -> String must resolve; got {:?}",
        diags
    );
}

#[test]
fn string_push_i64_phase_b_rejects_wrong_arity() {
    let table = build_table_with_stdlib();
    // string_push_i64 takes (String, i64); the one-arg call must be
    // rejected by the Phase-B arity check.  (Per-argument type
    // strictness across String / i64 surfaces is intentionally
    // lenient at this layer — the resolver enforces arity + named-
    // struct identity, not full primitive-vs-named coercion.)
    let consumer = "use std.string\n\
                    let s = string_new()\n\
                    let bad = string_push_i64(s)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "Phase B arity check must reject string_push_i64(String)"
    );
}
