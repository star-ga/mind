// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0005 Phase D₁ — `MIND_STDLIB_PATH` end-to-end smoke.
//!
//! The unit tests in `src/project/stdlib.rs` cover the loader
//! helper in isolation: env-unset uses the bundled set, env-set
//! pointing at a real directory parses the on-disk files, an unusable
//! directory is an error that names the path and the missing file. Those run in the
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

mod common;

use libmind::project::module_table::build_module_table;
use libmind::project::stdlib::{STDLIB_MIND_SOURCES, parsed_stdlib_modules, stdlib_override_error};

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
        // Every bundled module must parse cleanly and round-trip through the
        // loader. Assert against the source-of-truth count rather than a magic
        // number so adding a std module never silently leaves this stale.
        assert_eq!(mods.len(), STDLIB_MIND_SOURCES.len());
        assert!(names.contains(&"std.async"));
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
        assert_eq!(mods.len(), STDLIB_MIND_SOURCES.len());

        // Build the same project-loader-shaped table and confirm
        // every canonical public fn still resolves through the
        // override path.
        let refs: Vec<(String, &libmind::ast::Module)> =
            mods.iter().map(|(p, m)| (p.clone(), m)).collect();
        let table = build_module_table(&refs);

        assert!(table.resolves(&["std".into(), "async".into()], "sync_scheduler"));
        assert!(table.resolves(&["std".into(), "vec".into()], "vec_new"));
        assert!(table.resolves(&["std".into(), "vec".into()], "vec_push"));
        assert!(table.resolves(&["std".into(), "string".into()], "string_new"));
        assert!(table.resolves(&["std".into(), "map".into()], "map_new"));
        assert!(table.resolves(&["std".into(), "io".into()], "stdout"));
        assert!(table.resolves(&["std".into(), "toml".into()], "toml_parse"));
    });
}

#[test]
fn a_missing_override_dir_is_an_error_naming_the_path() {
    // spec/v1.0/stdlib.md, "Environment override": a set but unusable
    // MIND_STDLIB_PATH is an error, never a silent build against the bundled
    // set. Checked on the path, not through the process environment, so a
    // parallel test in this binary never sees the bad value.
    let nonexistent = std::path::Path::new("/nonexistent/mind/stdlib/path/that/does/not/exist");
    let err = stdlib_override_error(nonexistent).expect("a missing dir is refused");
    assert!(
        err.contains("/nonexistent/mind/stdlib/path/that/does/not/exist")
            && err.contains("is not a directory"),
        "{err}"
    );
}

#[test]
fn a_partial_override_dir_is_an_error_naming_the_missing_file() {
    // A dir that holds only some of the four specified files (a fork of
    // std.vec without the other three) is the most likely misuse. It names
    // the first required file it lacks instead of using the bundled set.
    let tmp = common::scratch_dir("std_surface_phase_d_env_override").join("mind-stdlib-partial-d");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("setup: create partial override dir");
    std::fs::write(tmp.join("vec.mind"), "// only vec, missing the other three")
        .expect("setup: write partial vec.mind");
    let err = stdlib_override_error(&tmp).expect("a partial dir is refused");
    assert!(
        err.contains(&tmp.display().to_string()) && err.contains("lacks `io.mind` (std.io)"),
        "{err}"
    );
    let repo_io = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("std/io.mind");
    std::fs::copy(repo_io, tmp.join("io.mind")).expect("setup: copy io.mind");
    let err = stdlib_override_error(&tmp).expect("still partial");
    assert!(err.contains("lacks `map.mind` (std.map)"), "{err}");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `mindc check` with a set but unusable MIND_STDLIB_PATH exits 2 and names
/// the path and the missing file or the parse error; it never checks against
/// the bundled set. Unset, the same program checks clean (the control).
#[test]
fn mindc_refuses_an_unusable_override_and_names_it() {
    let base = common::scratch_dir("std_surface_phase_d_env_override").join("mind-stdlib-cli-d");
    let _ = std::fs::remove_dir_all(&base);
    let prog = base.join("prog");
    std::fs::create_dir_all(prog.join(".git")).expect("setup: project dir");
    std::fs::write(
        prog.join("Mind.toml"),
        "[package]\nname = \"p\"\nversion = \"0.0.1\"\n",
    )
    .expect("setup: Mind.toml");
    std::fs::write(
        prog.join("main.mind"),
        "import std.vec;\n\nfn main() -> i64 {\n    return vec_len(vec_new());\n}\n",
    )
    .expect("setup: main.mind");
    let repo_std = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("std");
    let lacking = base.join("lacking");
    std::fs::create_dir_all(&lacking).expect("setup: lacking dir");
    for m in ["map", "string", "vec"] {
        std::fs::copy(
            repo_std.join(format!("{m}.mind")),
            lacking.join(format!("{m}.mind")),
        )
        .expect("setup: copy module");
    }
    let broken = base.join("broken");
    std::fs::create_dir_all(&broken).expect("setup: broken dir");
    for m in ["io", "map", "string", "vec"] {
        std::fs::copy(
            repo_std.join(format!("{m}.mind")),
            broken.join(format!("{m}.mind")),
        )
        .expect("setup: copy module");
    }
    std::fs::write(broken.join("json.mind"), "fn (").expect("setup: broken json.mind");
    let check = |root: Option<&std::path::Path>| {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_mindc"));
        cmd.arg("check").arg("main.mind").current_dir(&prog);
        match root {
            Some(root) => cmd.env("MIND_STDLIB_PATH", root),
            None => cmd.env_remove("MIND_STDLIB_PATH"),
        };
        let out = cmd.output().expect("spawn mindc");
        let all = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), all)
    };
    let missing = base.join("no_such_dir");
    for (root, item) in [
        (&missing, "is not a directory".to_string()),
        (&lacking, "lacks `io.mind` (std.io)".to_string()),
        (&broken, "`json.mind` (std.json) does not parse".to_string()),
    ] {
        let (code, all) = check(Some(root));
        assert!(
            code == Some(2) && all.contains(&root.display().to_string()) && all.contains(&item),
            "MIND_STDLIB_PATH={} must exit 2 naming the path and `{item}`, got {code:?}:\n{all}",
            root.display()
        );
    }
    let (code, all) = check(None);
    assert_eq!(code, Some(0), "unset uses the bundled std:\n{all}");
    let _ = std::fs::remove_dir_all(&base);
}
