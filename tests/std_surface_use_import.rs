// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase 2 — `use std.vec` resolution.
//!
//! End-to-end check that the three pieces wired earlier compose:
//!
//! 1. `std/*.mind` auto-exports its top-level `pub fn` and `struct`
//!    names via `collect_module_exports` (module_table.rs).
//! 2. The project loader / programmatic `ModuleTable` builder records
//!    those names under their dotted module path (e.g. `std.vec`).
//! 3. The type-checker's `cross-module-imports` resolver injects the
//!    names into the consumer's type-env at the `Node::Import` site,
//!    and the `infer_call` catch-all accepts calls to those names as
//!    `ScalarI64`-returning callables (no full signature check yet —
//!    Phase B will add per-arg type matching against the imported
//!    `pub fn` declarations).
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_use_import`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::parser;
use libmind::project::module_table::{
    ModuleExports, ModuleTable, build_module_table, collect_module_exports,
};
use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

const VEC_MIND_SRC: &str = include_str!("../std/vec.mind");
const STRING_MIND_SRC: &str = include_str!("../std/string.mind");
const MAP_MIND_SRC: &str = include_str!("../std/map.mind");
const IO_MIND_SRC: &str = include_str!("../std/io.mind");

fn build_stdlib_table() -> ModuleTable {
    let vec_ast = parser::parse(VEC_MIND_SRC).expect("std/vec.mind must parse");
    let string_ast = parser::parse(STRING_MIND_SRC).expect("std/string.mind must parse");
    let map_ast = parser::parse(MAP_MIND_SRC).expect("std/map.mind must parse");
    let io_ast = parser::parse(IO_MIND_SRC).expect("std/io.mind must parse");
    build_module_table(&[
        ("std.vec".to_string(), &vec_ast),
        ("std.string".to_string(), &string_ast),
        ("std.map".to_string(), &map_ast),
        ("std.io".to_string(), &io_ast),
    ])
}

#[test]
fn std_vec_module_auto_exports_its_pub_fns() {
    let vec_ast = parser::parse(VEC_MIND_SRC).expect("std/vec.mind must parse");
    let ex: ModuleExports = collect_module_exports("std.vec", &vec_ast);
    assert_eq!(ex.module_path, "std.vec");
    for want in [
        "Vec", "vec_addr", "vec_cap", "vec_get", "vec_len", "vec_new", "vec_push", "vec_set",
    ] {
        assert!(
            ex.exported.iter().any(|s| s == want),
            "std.vec must auto-export `{want}`; got {:?}",
            ex.exported
        );
    }
}

#[test]
fn std_string_module_auto_exports_struct_and_fns() {
    let ast = parser::parse(STRING_MIND_SRC).expect("std/string.mind must parse");
    let ex = collect_module_exports("std.string", &ast);
    for want in [
        "String",
        "string_addr",
        "string_cap",
        "string_eq",
        "string_get_byte",
        "string_len",
        "string_new",
        "string_push_byte",
        "string_validate_utf8",
    ] {
        assert!(
            ex.exported.iter().any(|s| s == want),
            "std.string must auto-export `{want}`; got {:?}",
            ex.exported
        );
    }
}

#[test]
fn std_io_module_auto_exports_struct_and_fns() {
    let ast = parser::parse(IO_MIND_SRC).expect("std/io.mind must parse");
    let ex = collect_module_exports("std.io", &ast);
    for want in [
        "File",
        "eprint_bytes",
        "file_fd",
        "file_read",
        "file_write",
        "print_bytes",
        "read_stdin_bytes",
        "stderr",
        "stdin",
        "stdout",
    ] {
        assert!(
            ex.exported.iter().any(|s| s == want),
            "std.io must auto-export `{want}`; got {:?}",
            ex.exported
        );
    }
}

#[test]
fn use_std_vec_resolves_vec_new_call() {
    let table = build_stdlib_table();
    // A consumer file that imports `std.vec` and calls one of its
    // exported names.  Type-check must succeed with the table in
    // place — under cross-module-imports, the resolver injects
    // `vec_new` into the local env and `infer_call`'s catch-all
    // accepts it.
    let consumer_src = "use std.vec\nlet x = vec_new()\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::default();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        diags.is_empty(),
        "`use std.vec` should make `vec_new` callable; got {:?}",
        diags
    );
}

#[test]
fn use_std_io_resolves_print_bytes_call() {
    let table = build_stdlib_table();
    let consumer_src = "use std.io\nlet n = print_bytes(0, 0)\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::default();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        diags.is_empty(),
        "`use std.io` should make `print_bytes` callable; got {:?}",
        diags
    );
}

#[test]
fn unimported_module_does_not_pollute_consumer_env() {
    let table = build_stdlib_table();
    // No `use std.vec` here — `vec_new` must NOT resolve, even though
    // it's in the module table.  The injection is gated on the
    // `Node::Import` arm running.
    let consumer_src = "let x = vec_new()\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::default();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "without `use std.vec`, calling `vec_new` must be a type error"
    );
}

#[test]
fn wrong_module_path_does_not_resolve() {
    let table = build_stdlib_table();
    // `use std.does_not_exist` is a no-op (the resolver simply finds
    // no entry in the table); calls to `vec_new` must still fail.
    let consumer_src = "use std.does_not_exist\nlet x = vec_new()\n";
    let ast = parser::parse(consumer_src).expect("consumer must parse");
    let env = TypeEnv::default();
    let diags = check_module_types_with_modules(&ast, consumer_src, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "`use` of an unknown path must not silently resolve foreign names"
    );
}

#[test]
fn export_block_still_overrides_auto_export() {
    // A module with an explicit `export { ... }` keeps its explicit
    // contract — auto-export only kicks in when no export block is
    // present.  This protects the RFC 0002 contract from drift.
    let src = "export { just_this }\nfn just_this() {}\nfn hidden() {}\n";
    let ast = parser::parse(src).expect("module must parse");
    let ex = collect_module_exports("crate.x", &ast);
    assert_eq!(
        ex.exported,
        vec!["just_this".to_string()],
        "explicit `export {{...}}` must hide non-exported fns; got {:?}",
        ex.exported
    );
}
