// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase D₁ — `MIND_STDLIB_PATH` end-to-end smoke.
//!
//! The unit tests in `src/project/stdlib.rs` cover the loader
//! helper in isolation: env-unset falls back to bundled, env-set
//! pointing at a real directory parses the on-disk files, env-set
//! pointing at a non-existent path falls back. Those run in the
//! library-test binary and exercise `parsed_stdlib_modules_from_env`
//! directly.
//!
//! This integration test exercises the same paths from the test-
//! binary side, mirroring how a downstream `mind build` would
//! invoke the project loader's cross-module-imports block. The
//! point is to lock the contract at the consumer surface — anyone
//! who breaks `parsed_stdlib_modules()`'s env-var preamble will
//! see this test fail, even if the unit tests inside the source
//! tree get accidentally moved around.
//!
//! Gated: `cargo test --features std-surface,cross-module-imports
//!                  --test std_surface_phase_d_env_override`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

use libmind::project::module_table::build_module_table;
use libmind::project::stdlib::parsed_stdlib_modules;

/// Sets `MIND_STDLIB_PATH` for the duration of the closure, then
/// clears it. The wrapper centralises the unsafe blocks the
/// test needs to mutate the process environment — Rust 2024 makes
/// `set_var` / `remove_var` unsafe to discourage racy concurrent
/// access. The integration tests run in a single binary so we
/// serialise these by convention (each test mutates and then
/// clears before returning).
fn with_env<F: FnOnce()>(key: &str, value: Option<&std::path::Path>, body: F) {
    unsafe {
        match value {
            Some(p) => std::env::set_var(key, p),
            None => std::env::remove_var(key),
        }
    }
    body();
    unsafe {
        std::env::remove_var(key);
    }
}

#[test]
fn env_unset_uses_bundled_stdlib() {
    with_env("MIND_STDLIB_PATH", None, || {
        let mods = parsed_stdlib_modules();
        let names: Vec<&str> = mods.iter().map(|(p, _)| p.as_str()).collect();
        // The bundled set is exactly the six canonical modules
        // (std.blas joined in RFC 0006; std.toml joined in task #258),
        // in deterministic alphabetical order.
        assert_eq!(mods.len(), 6);
        assert!(names.contains(&"std.blas"));
        assert!(names.contains(&"std.io"));
        assert!(names.contains(&"std.map"));
        assert!(names.contains(&"std.string"));
        assert!(names.contains(&"std.toml"));
        assert!(names.contains(&"std.vec"));
    });
}

#[test]
fn env_set_to_repo_std_dir_round_trips() {
    // Point the env var at the on-disk std/ directory shipped in
    // the repo. The four files there are the *same* sources the
    // bundle is built from (`include_str!("../../std/foo.mind")`),
    // so the resulting module table is structurally identical to
    // the bundled path.
    let std_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("std");
    assert!(
        std_dir.is_dir(),
        "test prerequisite: {} must exist",
        std_dir.display()
    );

    with_env("MIND_STDLIB_PATH", Some(&std_dir), || {
        let mods = parsed_stdlib_modules();
        assert_eq!(mods.len(), 6);

        // Build the same project-loader-shaped table and confirm
        // every canonical public fn still resolves through the
        // override path.
        let refs: Vec<(String, &libmind::ast::Module)> =
            mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);

        assert!(table.resolves(&["std".into(), "vec".into()], "vec_new"));
        assert!(table.resolves(&["std".into(), "vec".into()], "vec_push"));
        assert!(table.resolves(&["std".into(), "string".into()], "string_new"));
        assert!(table.resolves(&["std".into(), "map".into()], "map_new"));
        assert!(table.resolves(&["std".into(), "io".into()], "stdout"));
        assert!(table.resolves(&["std".into(), "toml".into()], "toml_parse"));
    });
}

#[test]
fn env_set_to_missing_dir_falls_back_to_bundled() {
    // A pointed-at dir that doesn't exist must NOT crash and must
    // produce the bundled set, not an empty one. This is the
    // "best effort" contract documented in `stdlib.rs` —
    // user-side breakage of the override surfaces as the override
    // simply not taking effect, never as a compiler crash.
    let nonexistent = std::path::Path::new("/nonexistent/mind/stdlib/path/that/does/not/exist");
    with_env("MIND_STDLIB_PATH", Some(nonexistent), || {
        let mods = parsed_stdlib_modules();
        assert_eq!(
            mods.len(),
            6,
            "missing override dir must fall back to bundled stdlib"
        );
    });
}

#[test]
fn env_set_to_partial_dir_falls_back_to_bundled() {
    // A dir that exists but only has *some* of the .mind files
    // (e.g. user only wanted to fork std.string, forgot to copy
    // the others) is the most likely real-world misuse. The
    // contract says "all four or none" — a partial set falls back
    // to bundled, so the user gets a working build even if their
    // fork is half-done. The bundled-then-override precedence
    // doesn't trigger here because the override returns None on
    // any missing file.
    let tmp = std::env::temp_dir().join("mind-stdlib-partial-d-test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("setup: create partial override dir");
    std::fs::write(tmp.join("vec.mind"), "// only vec, missing the other three")
        .expect("setup: write partial vec.mind");

    with_env("MIND_STDLIB_PATH", Some(&tmp), || {
        let mods = parsed_stdlib_modules();
        assert_eq!(
            mods.len(),
            6,
            "partial override dir must fall back to bundled stdlib"
        );
    });

    let _ = std::fs::remove_dir_all(&tmp);
}
