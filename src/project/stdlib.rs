// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0005 Phase C — bundle `std/*.mind` into the mindc binary.
//!
//! Phase A wired the type-checker resolver for `use std.foo`.
//! Phase B added per-arg signature matching on imported `pub fn`s.
//! Both relied on the consumer (test or build pipeline) supplying the
//! std module ASTs to `build_module_table`. A real downstream `mind
//! build` running on a project that says `use std.vec` had no way to
//! find the std/*.mind files unless they were vendored into the
//! project — which defeats the point of having a shared standard
//! library.
//!
//! This module bundles the four pure-MIND std/*.mind files
//! (`vec.mind`, `string.mind`, `map.mind`, `io.mind`) into the mindc
//! binary at compile time via `include_str!`. The project loader
//! consumes `parsed_stdlib_modules()` and prepends them to the
//! `(module_path, AST)` pairs it feeds into `build_module_table` so
//! the cross-module resolver sees `std.vec` / `std.string` /
//! `std.map` / `std.io` before walking the project's own src tree.
//!
//! Gated entirely behind `feature = "cross-module-imports"`. Default
//! build never compiles this file; the std/*.mind strings are only
//! linked into binaries that already opted into the cross-module
//! resolver. The compile-speed moat is held — module-level gate, no
//! per-statement cfg, zero runtime dispatch on the default hot path.

use crate::ast::Module;

/// Compile-time bundle of the four RFC 0005 Phase 2 std/*.mind files,
/// laid out as `(module_path, source_text)` pairs. Module paths use
/// the dotted notation `Node::Import` already speaks (`use std.vec`
/// → `["std", "vec"]` → joined `"std.vec"`).
///
/// The list is ordered alphabetically by module path so the module
/// table's deterministic-insertion contract is preserved when the
/// project loader prepends these to the user's own modules.
pub const STDLIB_MIND_SOURCES: &[(&str, &str)] = &[
    ("std.io", include_str!("../../std/io.mind")),
    ("std.map", include_str!("../../std/map.mind")),
    ("std.string", include_str!("../../std/string.mind")),
    ("std.vec", include_str!("../../std/vec.mind")),
];

/// Parse every bundled std/*.mind source and return the
/// `(module_path, AST)` pairs the project loader feeds into
/// `build_module_table`. Each source is parsed exactly once per
/// call.
///
/// Parse failures are silently dropped: an unparseable std/*.mind
/// file is a bug in the mindc release that should be caught by the
/// existing per-module `std_surface_*_module` test suite, not a
/// failure of the project loader. Falling back to "no std on the
/// path" is the safer behaviour — the user gets an "unknown
/// identifier" at the call site rather than a build crash.
pub fn parsed_stdlib_modules() -> Vec<(String, Module)> {
    let mut out = Vec::with_capacity(STDLIB_MIND_SOURCES.len());
    for (path, src) in STDLIB_MIND_SOURCES {
        if let Ok(ast) = crate::parser::parse(src) {
            out.push(((*path).to_string(), ast));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_source_parses() {
        // The std/*.mind shipped in v0.4.0 + v0.4.1 must parse under
        // the same parser the project loader uses. If any of them
        // breaks, the corresponding `tests/std_surface_*_module.rs`
        // suite already catches it — but assert here too so a
        // dropped `include_str!` path or a half-renamed module is
        // flagged at lib-test time.
        let mods = parsed_stdlib_modules();
        assert_eq!(
            mods.len(),
            STDLIB_MIND_SOURCES.len(),
            "every STDLIB_MIND_SOURCES entry must parse cleanly; got {} of {}",
            mods.len(),
            STDLIB_MIND_SOURCES.len()
        );
        let names: Vec<&str> = mods.iter().map(|(p, _)| p.as_str()).collect();
        assert!(names.contains(&"std.vec"));
        assert!(names.contains(&"std.string"));
        assert!(names.contains(&"std.map"));
        assert!(names.contains(&"std.io"));
    }

    #[test]
    fn stdlib_table_resolves_use_std_vec() {
        use crate::project::module_table::build_module_table;
        // The whole point: build a table from the bundled stdlib and
        // verify the resolver can find `vec_new` at `std.vec`.
        let mods = parsed_stdlib_modules();
        let refs: Vec<(String, &Module)> = mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);
        assert!(table.resolves(&["std".into(), "vec".into()], "vec_new"));
        assert!(table.resolves(&["std".into(), "string".into()], "string_new"));
        assert!(table.resolves(&["std".into(), "map".into()], "map_new"));
        assert!(table.resolves(&["std".into(), "io".into()], "stdout"));
    }

    #[test]
    fn bundled_stdlib_carries_phase_b_signatures() {
        use crate::project::module_table::build_module_table;
        // Phase B contract: every bundled std fn has its full
        // signature recorded in `exported_fns`, not just its name.
        let mods = parsed_stdlib_modules();
        let refs: Vec<(String, &Module)> = mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);
        let vec_new = table
            .lookup_imported_fn("vec_new")
            .expect("vec_new must be in the bundled stdlib's exported_fns");
        assert!(vec_new.param_types.is_empty(), "vec_new takes zero params");
        let vec_push = table
            .lookup_imported_fn("vec_push")
            .expect("vec_push must be in the bundled stdlib");
        assert_eq!(vec_push.param_types.len(), 2);
    }
}
