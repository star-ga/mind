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
//! It also covers the CALLERS of std functions that read the world
//! (`std.entropy`, `std.dns`): their bodies are compiled into a separate
//! object, so the caller's IR holds only the call, and the classifier must know
//! the function by name. That case resolves a std import, so it needs
//! `cross-module-imports`.
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

/// `(source under `std/`, intrinsic the diagnostic must name, flag site defended)`
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
    let out_so = common::scratch_dir("determinism_veto_control")
        .join(format!("mind_determinism_veto_{tag}.so"));
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
    for (name, offender, site) in CASES {
        let src = std_source(name);
        assert!(src.exists(), "missing std source {src:?}");
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

/// `(module, call, the function the refusal must name)`: a CALLER of a std
/// function whose body reads the world. The body is compiled into a separate
/// substrate object, so the caller's IR — what the veto, the evidence label and
/// `mindc verify` classify — holds only the call.
#[cfg(feature = "cross-module-imports")]
const STD_WORLD_CALLERS: &[(&str, &str, &str)] = &[
    ("entropy", "entropy.entropy_fill(p, 8)", "entropy_fill"),
    ("dns", "dns.dns_resolve(p, 9)", "dns_resolve"),
    ("dns", "dns.tcp_connect_host(p, 9, 80)", "tcp_connect_host"),
    (
        "dns",
        "dns.tcp_connect_host_timeout(p, 9, 80, 100)",
        "tcp_connect_host_timeout",
    ),
    ("https", "https.https_get(p, 9)", "https_get"),
    ("https", "https.https_get_ca(p, 9, p)", "https_get_ca"),
    ("https", "https.https_post(p, 9, p, 1, p, 1)", "https_post"),
    (
        "https",
        "https.https_post_ca(p, 9, p, 1, p, 1, p)",
        "https_post_ca",
    ),
    (
        "https",
        "https.https_get_with_headers(p, 9, p, 1, p)",
        "https_get_with_headers",
    ),
    (
        "https",
        "https.https_post_with_headers(p, 9, p, 1, p, 1, p, 1, p)",
        "https_post_with_headers",
    ),
];

/// Pure functions of the same modules: the positive control.
#[cfg(feature = "cross-module-imports")]
const STD_PURE_CALLERS: &[(&str, &str)] = &[
    ("entropy", "entropy.entropy_max_len()"),
    ("dns", "dns.dns_status(p)"),
    ("https", "https.https_deadline_s()"),
];

/// Write `import std.<module>; pub fn f(p: i64) -> i64 { <call> }` to
/// `dir/<tag>.mind`; returns `(source, evidence path)`.
#[cfg(feature = "cross-module-imports")]
fn std_caller(dir: &Path, tag: &str, module: &str, call: &str) -> (String, String) {
    let text = format!("import std.{module};\n\npub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
    let src = common::scratch_src_in(dir, &format!("{tag}.mind"), &text);
    let ev = dir.join(format!("{tag}.ev"));
    (
        src.to_string_lossy().into_owned(),
        ev.to_string_lossy().into_owned(),
    )
}

/// `mindc <args>`: `(success, stdout + stderr)`.
#[cfg(feature = "cross-module-imports")]
fn mindc(args: &[&str]) -> (bool, String) {
    let out = Command::new(common::mindc_bin())
        .args(args)
        .output()
        .expect("spawn mindc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// A program that calls `entropy.entropy_fill` or a `std.dns` resolver must be
/// treated like one that calls `getrandom` / `getaddrinfo` itself: refused
/// without `--allow-nondeterministic`, and with it attested `nondeterministic`,
/// which `mindc verify --require-deterministic` rejects. Before these were
/// classified, the caller emitted, attested `deterministic` and passed that
/// verify. The pure accessors run the same three steps and must stay
/// `deterministic`, so a refusal for an unrelated reason cannot pass as a veto.
#[cfg(feature = "cross-module-imports")]
#[test]
fn callers_of_std_world_functions_are_refused_and_attested_nondeterministic() {
    let dir = common::scratch_dir("determinism_std_callers");
    let mut ran = 0usize;
    for (i, (module, call, offender)) in STD_WORLD_CALLERS.iter().enumerate() {
        let (src, ev) = std_caller(&dir, &format!("world{i}"), module, call);
        let (ok, text) = mindc(&[&src, "--emit-evidence", &ev]);
        assert!(
            !ok && text.contains("error[determinism]") && text.contains(&format!("{offender}()`")),
            "a caller of `{call}` was not refused naming `{offender}()`\n{text}"
        );
        let (ok, text) = mindc(&[&src, "--emit-evidence", &ev, "--allow-nondeterministic"]);
        assert!(
            ok,
            "`{call}` with --allow-nondeterministic did not emit\n{text}"
        );
        let (ok, text) = mindc(&["verify", "--require-deterministic", &ev]);
        assert!(
            !ok && text.contains("determinism:      nondeterministic"),
            "the evidence for a caller of `{call}` passed --require-deterministic\n{text}"
        );
        ran += 1;
    }
    for (i, (module, call)) in STD_PURE_CALLERS.iter().enumerate() {
        let (src, ev) = std_caller(&dir, &format!("pure{i}"), module, call);
        let (ok, text) = mindc(&[&src, "--emit-evidence", &ev]);
        assert!(ok, "a caller of the pure `{call}` was refused\n{text}");
        let (ok, text) = mindc(&["verify", "--require-deterministic", &ev]);
        assert!(
            ok && text.contains("determinism:      deterministic"),
            "the evidence for a caller of the pure `{call}` is not deterministic\n{text}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(std callers): ran={ran} fail=0");
    assert_eq!(
        ran,
        STD_WORLD_CALLERS.len() + STD_PURE_CALLERS.len(),
        "vacuous run: ran={ran}"
    );
}
