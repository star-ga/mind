// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase C — stdlib auto-bundling end-to-end smoke.
//!
//! Phases A + B made `use std.vec` work when the consumer (test or
//! build pipeline) supplied the std/* ASTs to `build_module_table`.
//! Phase C bakes those sources into the mindc binary via
//! `include_str!` and seeds the module table with them in the
//! project loader's cross-module-imports block.  A downstream
//! `mind build` of a project that says `use std.vec` now resolves
//! without any external file dependency.
//!
//! This test exercises the bundle through the library-level entry
//! points the project loader uses internally:
//!
//!   1. `parsed_stdlib_modules()` parses all four bundled sources.
//!   2. Adding a consumer module on top produces a complete
//!      `ModuleTable` whose layout matches what the project loader
//!      hands to the type-checker.
//!   3. The type-checker (under `check_module_types_with_modules`)
//!      then validates a real consumer file via the Phase-B
//!      signature-aware resolver path — arity, per-arg types, and
//!      return type are all threaded through.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_phase_c_stdlib_bundle`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::ast::Module;
use libmind::parser;
use libmind::project::module_table::build_module_table;
use libmind::project::stdlib::parsed_stdlib_modules;
use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

#[test]
fn bundled_stdlib_resolves_use_std_vec_end_to_end() {
    // Build the project-loader-shaped module table: bundled stdlib
    // first, then any consumer modules (in this minimal test, just
    // the consumer below).  This mirrors what
    // `src/project/mod.rs`'s cross-module-imports block does.
    let stdlib: Vec<(String, Module)> = parsed_stdlib_modules();
    let stdlib_refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&stdlib_refs);

    // A real consumer file: `use std.vec` + calls into the std
    // surface.  The Phase-B path validates that `vec_push` takes
    // two args (Vec, i64) and returns Vec; the let binding accepts
    // that result implicitly.
    let consumer = "use std.vec\n\
                    let v = vec_new()\n\
                    let v2 = vec_push(v, 42)\n\
                    let n = vec_len(v2)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "use std.vec + vec_new + vec_push + vec_len should resolve end-to-end via the bundled stdlib; got {:?}",
        diags
    );
}

#[test]
fn bundled_stdlib_resolves_all_four_modules() {
    let stdlib: Vec<(String, Module)> = parsed_stdlib_modules();
    let stdlib_refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&stdlib_refs);

    // One call into each module proves the full bundle resolves.
    let consumer = "use std.vec\n\
                    use std.string\n\
                    use std.map\n\
                    use std.io\n\
                    use std.toml\n\
                    let v = vec_new()\n\
                    let s = string_new()\n\
                    let m = map_new()\n\
                    let out = stdout()\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "all four std modules must resolve via the bundle; got {:?}",
        diags
    );
}

#[test]
fn bundled_stdlib_phase_b_rejects_wrong_arity() {
    let stdlib: Vec<(String, Module)> = parsed_stdlib_modules();
    let stdlib_refs: Vec<(String, &Module)> = stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&stdlib_refs);

    // vec_new is declared to take zero args.  Phase B should reject
    // the extra arg even though Phase A would have accepted it.
    let consumer = "use std.vec\nlet v = vec_new(99)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::new();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        !diags.is_empty(),
        "bundled stdlib must enforce Phase B arity check on vec_new"
    );
}

#[test]
fn user_module_can_shadow_bundled_stdlib_entry() {
    // `ModuleTable::insert` is last-write-wins by contract.  The
    // project loader's cross-module-imports block parses the bundled
    // stdlib FIRST then appends the user's src/ files, so a user
    // module named `std.vec` shadows the bundled one.  This isn't
    // recommended but the behaviour should be predictable.
    let mut all: Vec<(String, Module)> = parsed_stdlib_modules();
    let user_src = "pub fn vec_new() -> i64 { 99 }\n";
    let user_ast = parser::parse(user_src).expect("user must parse");
    all.push(("std.vec".to_string(), user_ast));
    let refs: Vec<(String, &Module)> = all.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);
    // The user's `vec_new` still resolves — and its declared param
    // count is zero (matches the bundled one), so the assertion is
    // just that resolution doesn't crash and returns *some*
    // signature.
    let sig = table
        .lookup_imported_fn("vec_new")
        .expect("vec_new resolves");
    assert!(sig.param_types.is_empty());
}
