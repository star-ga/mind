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
//! This module bundles every `std/*.mind` file into the mindc binary at
//! compile time via `include_str!`. The project loader consumes
//! [`parsed_stdlib_modules_for`] and prepends the std modules a compile
//! loads (the `load` submodule) to the `(module_path, AST)` pairs it feeds into
//! `build_module_table`, so the cross-module resolver sees each imported
//! `std.<m>` before walking the project's own src tree.
//!
//! The table is exhaustive over `std/*.mind`: a shipped module without a
//! row was accepted at its import and refused at every call, so
//! `tests/std_bundled_module_table.rs` enumerates the directory and fails
//! on a missing row. An import of a `std` path that names no shipped module
//! is refused at the import ([`unknown_std_import`]).
//!
//! Gated entirely behind `feature = "cross-module-imports"`. Default
//! build never compiles this file; the std/*.mind strings are only
//! linked into binaries that already opted into the cross-module
//! resolver. The compile-speed moat is held — module-level gate, no
//! per-statement cfg, zero runtime dispatch on the default hot path.

use crate::ast::Module;

mod load;
pub use load::{
    IMPLICIT_STD_MODULES, bundled_module_ast, imported_std_extras, is_loaded, misplaced_std_import,
    misplaced_std_imports, parsed_std_modules, parsed_stdlib_modules_for, std_import_roots,
    std_source, std_sources,
};

/// Compile-time bundle of every std/*.mind file, laid out as
/// `(module_path, source_text)` pairs. Module paths use the dotted
/// notation `Node::Import` already speaks (`use std.vec` → `["std", "vec"]`
/// → joined `"std.vec"`).
///
/// The list is ordered alphabetically by module path so the module
/// table's deterministic-insertion contract is preserved when the
/// project loader prepends these to the user's own modules.
// One row per line, unwrapped: the manifest lint
// (examples/mindc_mind/stdlib_manifest_lint.py) reads each row as
// `("std.<m>", include_str!` on one line.
#[rustfmt::skip]
pub const STDLIB_MIND_SOURCES: &[(&str, &str)] = &[
    ("std.aes_gcm", include_str!("../../std/aes_gcm.mind")),
    ("std.arena", include_str!("../../std/arena.mind")),
    ("std.async", include_str!("../../std/async.mind")),
    ("std.blas", include_str!("../../std/blas.mind")),
    ("std.chacha20_poly1305", include_str!("../../std/chacha20_poly1305.mind")),
    ("std.cli", include_str!("../../std/cli.mind")),
    ("std.detmath", include_str!("../../std/detmath.mind")),
    ("std.dns", include_str!("../../std/dns.mind")),
    ("std.ecdsa_p256", include_str!("../../std/ecdsa_p256.mind")),
    ("std.entropy", include_str!("../../std/entropy.mind")),
    ("std.fs", include_str!("../../std/fs.mind")),
    ("std.hkdf", include_str!("../../std/hkdf.mind")),
    ("std.hpack", include_str!("../../std/hpack.mind")),
    ("std.http", include_str!("../../std/http.mind")),
    ("std.http2_frame", include_str!("../../std/http2_frame.mind")),
    ("std.https", include_str!("../../std/https.mind")),
    ("std.https_http1", include_str!("../../std/https_http1.mind")),
    ("std.https_io", include_str!("../../std/https_io.mind")),
    ("std.https_probe", include_str!("../../std/https_probe.mind")),
    ("std.https_record", include_str!("../../std/https_record.mind")),
    ("std.https_url", include_str!("../../std/https_url.mind")),
    ("std.io", include_str!("../../std/io.mind")),
    ("std.io_canon", include_str!("../../std/io_canon.mind")),
    ("std.iouring", include_str!("../../std/iouring.mind")),
    ("std.json", include_str!("../../std/json.mind")),
    ("std.json_canon", include_str!("../../std/json_canon.mind")),
    ("std.keccak", include_str!("../../std/keccak.mind")),
    ("std.llvm", include_str!("../../std/llvm.mind")),
    ("std.map", include_str!("../../std/map.mind")),
    ("std.mic_record", include_str!("../../std/mic_record.mind")),
    ("std.mic_results", include_str!("../../std/mic_results.mind")),
    ("std.mlir", include_str!("../../std/mlir.mind")),
    ("std.mlkem768", include_str!("../../std/mlkem768.mind")),
    ("std.net", include_str!("../../std/net.mind")),
    ("std.process", include_str!("../../std/process.mind")),
    ("std.reactor", include_str!("../../std/reactor.mind")),
    ("std.regex", include_str!("../../std/regex.mind")),
    ("std.ring", include_str!("../../std/ring.mind")),
    ("std.rsa_pss", include_str!("../../std/rsa_pss.mind")),
    ("std.sha256", include_str!("../../std/sha256.mind")),
    ("std.sha512", include_str!("../../std/sha512.mind")),
    ("std.string", include_str!("../../std/string.mind")),
    ("std.time", include_str!("../../std/time.mind")),
    ("std.tls13_client_api", include_str!("../../std/tls13_client_api.mind")),
    ("std.tls13_client_core", include_str!("../../std/tls13_client_core.mind")),
    ("std.tls13_client_hello", include_str!("../../std/tls13_client_hello.mind")),
    ("std.tls13_finished", include_str!("../../std/tls13_finished.mind")),
    ("std.tls13_handshake", include_str!("../../std/tls13_handshake.mind")),
    ("std.tls13_keyschedule", include_str!("../../std/tls13_keyschedule.mind")),
    ("std.tls13_record", include_str!("../../std/tls13_record.mind")),
    ("std.toml", include_str!("../../std/toml.mind")),
    ("std.tui", include_str!("../../std/tui.mind")),
    ("std.vec", include_str!("../../std/vec.mind")),
    ("std.x25519", include_str!("../../std/x25519.mind")),
    ("std.x25519mlkem768", include_str!("../../std/x25519mlkem768.mind")),
    ("std.x509", include_str!("../../std/x509.mind")),
    ("std.x509_bignum", include_str!("../../std/x509_bignum.mind")),
    ("std.x509_bundle", include_str!("../../std/x509_bundle.mind")),
    ("std.x509_der", include_str!("../../std/x509_der.mind")),
    ("std.x509_ext", include_str!("../../std/x509_ext.mind")),
    ("std.x509_name", include_str!("../../std/x509_name.mind")),
    ("std.x509_p384", include_str!("../../std/x509_p384.mind")),
    ("std.x509_path", include_str!("../../std/x509_path.mind")),
    ("std.x509_sig", include_str!("../../std/x509_sig.mind")),
];

/// The standard-library modules the Core v1 specification names whose
/// entities are compiler builtins, available without an import
/// (spec/v1.0/stdlib.md, "Module organization"). Importing one is valid and
/// needs no bundled source; `std.io` is also bundled.
pub const BUILTIN_STD_MODULES: &[&str] =
    &["std.core", "std.diff", "std.io", "std.math", "std.tensor"];

/// True when `module` (dotted, e.g. `std.keccak`) is a standard-library
/// module this compiler ships: a bundled `std/*.mind` source or a builtin
/// module of the specification.
pub fn is_shipped_std_module(module: &str) -> bool {
    STDLIB_MIND_SOURCES.iter().any(|(name, _)| *name == module)
        || BUILTIN_STD_MODULES.contains(&module)
}

/// The refusal for an import path that starts with `std` but names no
/// shipped standard-library module; `None` for any other import.
///
/// Such an import used to be accepted and resolve nothing, so the program
/// failed later at each call (`unsupported call`, or `unsupported
/// operation` under `mindc test`). Every consumer that admits an import
/// asks this one function.
pub fn unknown_std_import(path: &[String]) -> Option<String> {
    if path.first().map(String::as_str) != Some("std") {
        return None;
    }
    let module = path.join(".");
    if is_shipped_std_module(&module) {
        return None;
    }
    Some(format!(
        "unknown standard-library module `{module}`: no std/*.mind module and no builtin \
         std module ({}) has that path, so this import cannot resolve",
        BUILTIN_STD_MODULES.join(", ")
    ))
}

/// Parse every bundled std/*.mind source (as [`std_sources`] sees it, so a
/// `MIND_STDLIB_PATH` override applies) into `(module_path, AST)` pairs in
/// table order. A compile loads only what it uses
/// ([`parsed_stdlib_modules_for`]); this whole-table form is for checks that
/// enumerate the bundle.
///
/// Parse failures are silently dropped: an unparseable std/*.mind
/// file is a bug in the mindc release that should be caught by the
/// existing per-module `std_surface_*_module` test suite, not a
/// failure of the project loader. Falling back to "no std on the
/// path" is the safer behaviour — the user gets an "unknown
/// identifier" at the call site rather than a build crash.
pub fn parsed_stdlib_modules() -> Vec<(String, Module)> {
    std_sources()
        .iter()
        .filter_map(|(path, src)| Some(((*path).to_string(), crate::parser::parse(src).ok()?)))
        .collect()
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
        assert!(names.contains(&"std.async"));
        assert!(names.contains(&"std.vec"));
        assert!(names.contains(&"std.string"));
        assert!(names.contains(&"std.map"));
        assert!(names.contains(&"std.io"));
        assert!(names.contains(&"std.toml"));
        assert!(names.contains(&"std.json"));
        assert!(names.contains(&"std.regex"));
        assert!(names.contains(&"std.fs"));
        assert!(names.contains(&"std.http"));
        assert!(names.contains(&"std.net"));
        assert!(names.contains(&"std.process"));
        assert!(names.contains(&"std.sha256"));
        assert!(names.contains(&"std.sha512"));
        assert!(names.contains(&"std.tui"));
    }

    #[cfg(feature = "cross-module-imports")]
    #[test]
    fn stdlib_table_resolves_use_std_vec() {
        use crate::project::module_table::build_module_table;
        // The whole point: build a table from the bundled stdlib and
        // verify the resolver can find `vec_new` at `std.vec`.
        let mods = parsed_stdlib_modules();
        let refs: Vec<(String, &Module)> = mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);
        assert!(table.resolves(&["std".into(), "async".into()], "sync_scheduler"));
        assert!(table.resolves(&["std".into(), "async".into()], "replay_scheduler"));
        assert!(table.resolves(&["std".into(), "async".into()], "submit"));
        assert!(table.resolves(&["std".into(), "async".into()], "run"));
        assert!(table.resolves(&["std".into(), "async".into()], "trace_hash"));
        assert!(table.resolves(&["std".into(), "vec".into()], "vec_new"));
        assert!(table.resolves(&["std".into(), "string".into()], "string_new"));
        assert!(table.resolves(&["std".into(), "map".into()], "map_new"));
        assert!(table.resolves(&["std".into(), "io".into()], "stdout"));
        assert!(table.resolves(&["std".into(), "toml".into()], "toml_parse"));
        assert!(table.resolves(&["std".into(), "json".into()], "jv_parse"));
        assert!(table.resolves(&["std".into(), "regex".into()], "rx_compile"));
        assert!(table.resolves(&["std".into(), "fs".into()], "read_to_string"));
        assert!(table.resolves(&["std".into(), "http".into()], "http_get"));
        assert!(table.resolves(&["std".into(), "http".into()], "http_post"));
        assert!(table.resolves(&["std".into(), "net".into()], "tcp_listen"));
        assert!(table.resolves(&["std".into(), "process".into()], "spawn"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_cursor_to"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_clear_screen"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_terminal_size"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_box_new"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_box_render"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_text_new"));
        assert!(table.resolves(&["std".into(), "tui".into()], "tui_text_render"));
        assert!(table.resolves(&["std".into(), "sha256".into()], "sha256"));
    }

    #[cfg(feature = "cross-module-imports")]
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
        assert!(names.contains(&"std.async"));
        assert!(names.contains(&"std.vec"));
        assert!(names.contains(&"std.string"));
        assert!(names.contains(&"std.map"));
        assert!(names.contains(&"std.io"));
        assert!(names.contains(&"std.toml"));
        assert!(names.contains(&"std.json"));
        assert!(names.contains(&"std.regex"));
        assert!(names.contains(&"std.fs"));
        assert!(names.contains(&"std.http"));
        assert!(names.contains(&"std.net"));
        assert!(names.contains(&"std.sha256"));
        assert!(names.contains(&"std.process"));
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

    #[test]
    fn unknown_std_import_refuses_only_std_paths_that_do_not_ship() {
        let path = |p: &str| p.split('.').map(str::to_string).collect::<Vec<_>>();
        assert!(unknown_std_import(&path("std.no_such_module")).is_some());
        assert!(unknown_std_import(&path("std.keccak.keccak_sha3_512")).is_some());
        assert!(unknown_std_import(&path("std")).is_some());
        assert_eq!(unknown_std_import(&path("std.keccak")), None);
        assert_eq!(unknown_std_import(&path("std.math")), None);
        assert_eq!(unknown_std_import(&path("crate.helper")), None);
        assert_eq!(unknown_std_import(&path("helper")), None);
    }
}
