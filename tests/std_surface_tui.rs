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

#[test]
fn tui_terminal_size_resolves_and_carries_accessors() {
    // Phase 1 add-on: `tui_terminal_size(fd) -> TermSize` plus the
    // tui_term_rows / tui_term_cols accessors must resolve as a unit
    // through the bundled stdlib's cross-module resolver. The integer
    // `fd` is the same opaque i64 `file_fd(stdout())` returns.
    let table = build_table_with_stdlib();
    let consumer = "use std.io\n\
                    use std.tui\n\
                    let fd = file_fd(stdout())\n\
                    let ts = tui_terminal_size(fd)\n\
                    let r = tui_term_rows(ts)\n\
                    let c = tui_term_cols(ts)\n\
                    let req = tui_tiocgwinsz()\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "tui_terminal_size + accessors must resolve via bundled std.tui; got {:?}",
        diags
    );

    // Phase B signatures: 1 param for tui_terminal_size, 0 for the constant.
    let term_size = table
        .lookup_imported_fn("tui_terminal_size")
        .expect("tui_terminal_size must be in the bundled stdlib");
    assert_eq!(
        term_size.param_types.len(),
        1,
        "tui_terminal_size takes (fd: i64)"
    );
    let tiocgwinsz = table
        .lookup_imported_fn("tui_tiocgwinsz")
        .expect("tui_tiocgwinsz must be in the bundled stdlib");
    assert!(
        tiocgwinsz.param_types.is_empty(),
        "tui_tiocgwinsz is a () -> i64 constant"
    );
}

#[test]
fn tui_box_widget_constructs_and_renders() {
    // Phase 1 add-on: the Box widget (struct + tui_box_new constructor +
    // tui_box_render composer) must round-trip through the resolver and
    // compose with the existing cursor/escape surface so a caller can
    // build a frame as one String value.
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    use std.tui\n\
                    let buf = string_new()\n\
                    let b = tui_box_new(2, 4, 20, 8)\n\
                    let frame = tui_box_render(buf, b)\n\
                    let w = tui_box_width(b)\n\
                    let h = tui_box_height(b)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "Box widget + render must resolve through bundled std.tui; got {:?}",
        diags
    );

    // Phase B signature: tui_box_new takes (row, col, width, height).
    let box_new = table
        .lookup_imported_fn("tui_box_new")
        .expect("tui_box_new must be in the bundled stdlib");
    assert_eq!(
        box_new.param_types.len(),
        4,
        "tui_box_new takes (row, col, width, height)"
    );

    // tui_box_render takes (s: String, b: Box) — two params.
    let box_render = table
        .lookup_imported_fn("tui_box_render")
        .expect("tui_box_render must be in the bundled stdlib");
    assert_eq!(
        box_render.param_types.len(),
        2,
        "tui_box_render takes (String, Box)"
    );
}

#[test]
fn tui_text_widget_renders_with_and_without_style() {
    // Phase 1 add-on: the Text widget composes a cursor-position + (optional)
    // SGR-wrapped payload. Verifies the four-arg shape resolves against the
    // bundled stdlib and that the unstyled (sgr_code == 0) and styled paths
    // both pass the cross-module resolver.
    let table = build_table_with_stdlib();
    let consumer = "use std.string\n\
                    use std.io\n\
                    use std.tui\n\
                    let buf = string_new()\n\
                    let payload = string_new()\n\
                    let plain = tui_text_new(1, 1, 0)\n\
                    let red = tui_text_new(2, 5, sgr_fg_red())\n\
                    let frame0 = tui_text_render(buf, plain, payload.addr, payload.len)\n\
                    let frame1 = tui_text_render(frame0, red, payload.addr, payload.len)\n\
                    let sgr = tui_text_sgr(red)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "Text widget + render must resolve through bundled std.tui; got {:?}",
        diags
    );

    // Phase B signature: tui_text_render takes (String, Text, addr, len).
    let text_render = table
        .lookup_imported_fn("tui_text_render")
        .expect("tui_text_render must be in the bundled stdlib");
    assert_eq!(
        text_render.param_types.len(),
        4,
        "tui_text_render takes (String, Text, payload_addr, payload_len)"
    );
}
