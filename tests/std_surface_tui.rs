// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0013 Tier 1 (c) — `std.tui` minimal surface parse + resolve.
//!
//! Verifies that the cursor/screen control-escape composers
//! (`tui_cursor_to` / `tui_clear_screen` / `tui_enter_alt_screen` /
//! `tui_cursor_hide` / `tui_scroll_up` …) resolve through the
//! bundled-stdlib type-checker path the project loader uses, carry
//! their full Phase B signatures, and reject wrong arity. Each
//! function consumes and returns a `String`, so this also exercises
//! the `use std.string` cross-module dependency from one bundled std
//! module into another.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_tui`.

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
fn tui_cursor_and_screen_composers_resolve() {
    let table = build_table_with_stdlib();
    // One representative from each family: absolute position, relative
    // move, column-absolute, erase, alt-screen, visibility, scroll.
    let consumer = "use std.string\n\
                    use std.tui\n\
                    let s0 = string_new()\n\
                    let s1 = tui_cursor_to(s0, 3, 7)\n\
                    let s2 = tui_cursor_up(s1, 2)\n\
                    let s3 = tui_cursor_col(s2, 1)\n\
                    let s4 = tui_clear_screen(s3)\n\
                    let s5 = tui_clear_line(s4)\n\
                    let s6 = tui_enter_alt_screen(s5)\n\
                    let s7 = tui_cursor_hide(s6)\n\
                    let s8 = tui_scroll_up(s7, 4)\n\
                    let s9 = tui_leave_alt_screen(s8)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "std.tui composers must resolve as String->String through the bundled stdlib; got {:?}",
        diags
    );
}

#[test]
fn tui_phase_b_carries_signatures() {
    let stdlib = parsed_stdlib_modules();
    let refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);

    // tui_cursor_to(String, i64, i64) — three params.
    let cursor_to = table
        .lookup_imported_fn("tui_cursor_to")
        .expect("tui_cursor_to must be in the bundled stdlib's exported_fns");
    assert_eq!(
        cursor_to.param_types.len(),
        3,
        "tui_cursor_to takes (String, row, col)"
    );

    // tui_clear_screen(String) — one param.
    let clear = table
        .lookup_imported_fn("tui_clear_screen")
        .expect("tui_clear_screen must be in the bundled stdlib");
    assert_eq!(clear.param_types.len(), 1, "tui_clear_screen takes (String)");
}

#[test]
fn tui_phase_b_rejects_wrong_arity() {
    let table = build_table_with_stdlib();
    // tui_cursor_to needs (String, row, col); calling with one arg must
    // be rejected by the Phase B arity check.
    let consumer = "use std.string\n\
                    use std.tui\n\
                    let s0 = string_new()\n\
                    let bad = tui_cursor_to(s0)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "Phase B arity check must reject tui_cursor_to(s0)"
    );
}
