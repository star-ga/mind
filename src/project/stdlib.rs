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
    ("std.blas", include_str!("../../std/blas.mind")),
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
    // Phase D — env-var override.
    //
    // If `MIND_STDLIB_PATH` is set, treat it as a directory containing
    // `vec.mind`, `string.mind`, `map.mind`, `io.mind` and parse those
    // instead of the bundled blobs. This lets a downstream user fork
    // the stdlib without rebuilding mindc — e.g. a stricter
    // string-validation variant for a regulated deployment.
    //
    // Falls back silently to the bundled blobs on any error (missing
    // dir, missing file, unreadable, parse failure). The principle
    // matches `parsed_stdlib_modules`'s own behaviour: a broken
    // override is the user's problem to surface, not ours to crash on.
    if let Some(modules) = parsed_stdlib_modules_from_env() {
        return modules;
    }

    let mut out = Vec::with_capacity(STDLIB_MIND_SOURCES.len());
    for (path, src) in STDLIB_MIND_SOURCES {
        if let Ok(ast) = crate::parser::parse(src) {
            out.push(((*path).to_string(), ast));
        }
    }
    out
}

/// Reads `MIND_STDLIB_PATH` from the environment. If set and pointing
/// at a directory that contains all four `.mind` source files, returns
/// the parsed modules; otherwise returns `None` so the caller can fall
/// back to the bundled blobs.
///
/// "All four files present and all four parse" is the bar — partial
/// overrides (e.g. supply your own `vec.mind`, fall back to bundled
/// for the rest) would be useful but introduce surprising
/// last-write-wins precedence between bundled and override; better to
/// require a full set for now and revisit if there's user demand.
fn parsed_stdlib_modules_from_env() -> Option<Vec<(String, Module)>> {
    let root = std::env::var_os("MIND_STDLIB_PATH")?;
    let root = std::path::PathBuf::from(root);
    if !root.is_dir() {
        return None;
    }

    let mut out = Vec::with_capacity(STDLIB_MIND_SOURCES.len());
    for (module_path, _) in STDLIB_MIND_SOURCES {
        // "std.vec" -> "vec.mind"; works for the 4 current modules and
        // matches the on-disk layout under mind/std/.
        let file_name = module_path.trim_start_matches("std.").to_string() + ".mind";
        let candidate = root.join(&file_name);
        let src = std::fs::read_to_string(&candidate).ok()?;
        let ast = crate::parser::parse(&src).ok()?;
        out.push(((*module_path).to_string(), ast));
    }
    Some(out)
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

    #[test]
    fn env_override_falls_back_when_unset() {
        // Phase D: when MIND_STDLIB_PATH is not set, parsed_stdlib_modules
        // must transparently return the bundled set. This is the default
        // path every existing consumer relies on; the override must not
        // change behaviour when absent.
        //
        // SAFETY: We deliberately remove the env var inside the test.
        // Cargo runs tests in parallel by default; a co-running test
        // that *did* set MIND_STDLIB_PATH could observe the unset. We
        // gate this against any concurrent setter by only asserting on
        // the bundled-paths invariant, not on the env var's state
        // post-test.
        unsafe {
            std::env::remove_var("MIND_STDLIB_PATH");
        }
        let mods = parsed_stdlib_modules();
        assert_eq!(mods.len(), STDLIB_MIND_SOURCES.len());
    }

    #[test]
    fn env_override_loads_directory_when_set() {
        // Phase D: pointing MIND_STDLIB_PATH at the repo's own std/
        // directory must round-trip every module through the override
        // path (file-system read + parse) instead of the bundled blobs.
        // We assert on count + names rather than on internal AST
        // identity since the bundled and on-disk sources are the same
        // file (verified by the include_str! pointing at ../../std/*).
        let std_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("std");
        assert!(
            std_dir.is_dir(),
            "test prerequisite: {} exists",
            std_dir.display()
        );

        unsafe {
            std::env::set_var("MIND_STDLIB_PATH", &std_dir);
        }
        let mods = parsed_stdlib_modules();
        unsafe {
            std::env::remove_var("MIND_STDLIB_PATH");
        }

        let names: Vec<&str> = mods.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(mods.len(), STDLIB_MIND_SOURCES.len());
        assert!(names.contains(&"std.vec"));
        assert!(names.contains(&"std.string"));
        assert!(names.contains(&"std.map"));
        assert!(names.contains(&"std.io"));
    }

    #[test]
    fn env_override_falls_back_on_missing_dir() {
        // A pointed-at dir that doesn't exist must NOT crash and must
        // NOT silently produce an empty stdlib — the override is
        // honoured "best effort" and falls back to the bundled set.
        unsafe {
            std::env::set_var("MIND_STDLIB_PATH", "/nonexistent/mind/stdlib/path");
        }
        let mods = parsed_stdlib_modules();
        unsafe {
            std::env::remove_var("MIND_STDLIB_PATH");
        }
        assert_eq!(
            mods.len(),
            STDLIB_MIND_SOURCES.len(),
            "missing override dir must fall back to bundled stdlib"
        );
    }
}
