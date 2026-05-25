// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 — `std.io` TTY + ANSI extensions parse + resolve.
//!
//! Verifies that the new `io_isatty(File) -> i64` libc wrapper and
//! the ECMA-48 SGR vocabulary (`sgr_reset` / `sgr_bold` / `sgr_fg_*`
//! / `sgr_bg_*`) resolve through the bundled-stdlib type-checker
//! path used by the project loader. Higher-level escape-sequence
//! composition lands once std.string ships an integer-to-decimal
//! helper; for now the constants must at minimum *type-check* as
//! `() -> i64` and roundtrip through the resolver.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_io_ansi`.

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
fn io_isatty_resolves_against_bundled_stdlib() {
    let table = build_table_with_stdlib();
    let consumer = "use std.io\n\
                    let out = stdout()\n\
                    let tty = io_isatty(out)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "io_isatty(File) -> i64 must resolve through the bundled std.io; got {:?}",
        diags
    );
}

#[test]
fn sgr_constants_resolve_and_are_i64() {
    let table = build_table_with_stdlib();
    // One name from each constant family — style, foreground, background.
    let consumer = "use std.io\n\
                    let r  = sgr_reset()\n\
                    let b  = sgr_bold()\n\
                    let fr = sgr_fg_red()\n\
                    let bg = sgr_bg_blue()\n\
                    let fd = sgr_fg_default()\n\
                    let bd = sgr_bg_default()\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "every SGR constant must resolve as () -> i64; got {:?}",
        diags
    );
}

#[test]
fn io_isatty_phase_b_rejects_wrong_arity() {
    let table = build_table_with_stdlib();
    let consumer = "use std.io\nlet t = io_isatty()\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "Phase B arity check must reject io_isatty()"
    );
}
