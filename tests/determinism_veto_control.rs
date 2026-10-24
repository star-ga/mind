// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Positive control for the determinism-by-default veto (`error[determinism]`).
//!
//! # Why this file exists
//!
//! Several std-surface integration tests legitimately pass
//! `--allow-nondeterministic` because the surface they exercise genuinely reads
//! the world (`std/net.mind` calls `getsockname()`, `std/iouring.mind` issues a
//! raw `syscall()`, `std/fs.mind` calls `read()`, `std/process.mind` calls
//! `fork()`). Those call sites are correct — but on their own they are only a
//! NEGATIVE control: every one of them would stay green if the veto in
//! `src/bin/mindc.rs` were removed outright, because passing a flag to a gate
//! that no longer exists is silently harmless. The whole
//! determinism-by-default claim would then be unasserted anywhere in `tests/`.
//!
//! This file is the missing positive control: it compiles the SAME sources
//! WITHOUT the flag and requires the compile to FAIL with `error[determinism]`
//! naming the offending intrinsic. Delete the veto and this test goes red.
//!
//! The check under test runs on the lowered IR before any emission backend is
//! consulted, so this gate needs neither the `mlir-build` feature nor an MLIR
//! toolchain: it only needs a `mindc` that can parse the std surface. It
//! therefore runs in the default (`std-surface`) `Build & Test` matrix on every
//! OS, not only in the feature-gated std-surface jobs.
//!
//! Fail-closed: the binary is resolved via `CARGO_BIN_EXE_mindc`, so there is
//! no skip path — a missing binary is a failure, and a run that asserted
//! nothing (`ran=0`) is a failure too.
//!
//! Run with:
//!   cargo test --test determinism_veto_control

#![cfg(feature = "std-surface")]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// `(source under `std/`, a world-reading intrinsic the source is known to call,
/// flag site defended)`.
///
/// The diagnostic is NOT required to name that particular intrinsic: the veto
/// names the FIRST hard-nondeterministic call in the lowered IR, so an edit that
/// reorders a module (std/process.mind's checked `wait` now reads its exec-error
/// pipe before it forks) moves which call is named without weakening anything.
/// What `veto_fires_without_allow_nondeterministic` requires instead is that the
/// named call is the one the library's own classifier picks from its classified
/// set for that same source, and that the source really calls it.
const CASES: &[(&str, &str, &str)] = &[
    // tests/std_surface_http.rs (compiles std/net.mind + std/http.mind)
    ("net.mind", "getsockname", "tests/std_surface_http.rs"),
    // tests/std_surface_iouring.rs (two sites)
    ("iouring.mind", "syscall", "tests/std_surface_iouring.rs"),
    // tests/std_surface_net_fs_process.rs (world-reading fixtures)
    ("fs.mind", "read", "tests/std_surface_net_fs_process.rs"),
    (
        "process.mind",
        "fork",
        "tests/std_surface_net_fs_process.rs",
    ),
];

/// The first call the library's determinism classifier treats as HARD
/// nondeterministic in `src` (the same function `mindc`'s veto consults). Only
/// names in the classified set can be returned.
fn classified_offender(src: &str) -> Option<String> {
    let module = libmind::parser::parse(src).expect("std source must parse");
    let ir = libmind::eval::lower::lower_to_ir(&module).expect("std source must lower");
    libmind::ir::ir_first_hard_nondeterministic_call(&ir)
}

fn std_source(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("std")
        .join(name)
}

/// Run `mindc <src> --emit-shared <tmp> [--allow-nondeterministic]`.
/// Returns `(success, stderr)`.
fn compile(src: &Path, tag: &str, allow: bool) -> (bool, String) {
    let mindc = common::mindc_bin();
    assert!(
        mindc.exists(),
        "mindc not built at {mindc:?}; this gate is fail-closed and must not skip"
    );
    // Unique per case AND per leg: the two tests run concurrently and the
    // authorised leg really does emit under `mlir-build`, so a shared output
    // path would be a write race between processes.
    let out_so = std::env::temp_dir().join(format!("mind_determinism_veto_{tag}.so"));
    let mut args = vec![
        src.to_str().expect("utf-8 source path").to_string(),
        "--emit-shared".to_string(),
        out_so.to_string_lossy().into_owned(),
    ];
    if allow {
        args.push("--allow-nondeterministic".to_string());
    }
    let out = Command::new(&mindc)
        .args(&args)
        .output()
        .expect("spawn mindc");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Without `--allow-nondeterministic`, a world-reading source must be REFUSED
/// when a runnable artifact is requested, and the diagnostic must name the
/// intrinsic that caused the refusal.
#[test]
fn veto_fires_without_allow_nondeterministic() {
    let mut ran = 0usize;
    for (name, known, site) in CASES {
        let src = std_source(name);
        assert!(src.exists(), "missing std source {src:?}");
        let text = std::fs::read_to_string(&src).expect("read std source");
        // Positive control: the case really is a world-reading source.
        assert!(
            text.contains(&format!("{known}(")),
            "std/{name}: no longer calls `{known}()`; update this case's known offender"
        );
        let offender = classified_offender(&text).unwrap_or_else(|| {
            panic!("std/{name}: the classifier found no hard-nondeterministic call at all")
        });
        assert!(
            text.contains(&format!("{offender}(")),
            "std/{name}: classifier named `{offender}()`, which the source does not call"
        );
        let (ok, stderr) = compile(&src, &format!("refusal_{name}"), false);
        assert!(
            !ok,
            "std/{name}: --emit-shared succeeded WITHOUT --allow-nondeterministic; \
             the determinism veto did not fire (flag at {site} is now decorative)\nstderr: {stderr}"
        );
        assert!(
            stderr.contains("error[determinism]"),
            "std/{name}: compile failed but not with error[determinism]; \
             the determinism veto did not fire (flag at {site} is now decorative)\nstderr: {stderr}"
        );
        assert!(
            stderr.contains(&format!("`{offender}()`")),
            "std/{name}: error[determinism] did not name `{offender}()`\nstderr: {stderr}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(refusal): ran={ran} fail=0");
    assert_eq!(ran, CASES.len(), "vacuous run: ran={ran}");
}

/// Complement: WITH the flag the veto must NOT fire — the flag is what
/// authorises the build. (The compile may still fail for an unrelated reason,
/// e.g. `--emit-shared` without the `mlir-build` feature; what must never
/// appear is `error[determinism]`.) This pins the veto to the flag rather than
/// to "this source never compiles".
#[test]
fn veto_is_silent_with_allow_nondeterministic() {
    let mut ran = 0usize;
    for (name, _offender, _site) in CASES {
        let (_ok, stderr) = compile(&std_source(name), &format!("authorised_{name}"), true);
        assert!(
            !stderr.contains("error[determinism]"),
            "std/{name}: --allow-nondeterministic did not authorise the build\nstderr: {stderr}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(authorised): ran={ran} fail=0");
    assert_eq!(ran, CASES.len(), "vacuous run: ran={ran}");
}
