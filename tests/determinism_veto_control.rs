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
//! `cross-module-imports`. So does the last case: a program whose OWN project
//! modules make such a call, one or more calls away from the entry, which the
//! veto and the evidence label reach by following the calls across modules.
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
    // The `::` and full-path spellings name the same std function.
    (
        "entropy",
        "std::entropy::entropy_fill(p, 8)",
        "entropy_fill",
    ),
    ("dns", "std.dns.dns_resolve(p, 9)", "dns_resolve"),
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
    (
        "https_probe",
        "https_probe.https_probe(p, 9, 100)",
        "https_probe",
    ),
];

/// Pure functions of the same modules: the positive control.
#[cfg(feature = "cross-module-imports")]
const STD_PURE_CALLERS: &[(&str, &str)] = &[
    ("entropy", "entropy.entropy_max_len()"),
    ("dns", "dns.dns_status(p)"),
    ("https", "https.https_deadline_s()"),
    ("https_http1", "https_http1.https_error_class(p)"),
    ("https_probe", "https_probe.https_probe_ok()"),
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

/// Copy the evidence at `src` to `dst` with only its stored determinism label
/// rewritten from `nondeterministic` to `deterministic` (its length byte too).
#[cfg(feature = "cross-module-imports")]
fn forge_deterministic(src: &Path, dst: &Path) {
    let bytes = std::fs::read(src).expect("read evidence");
    let marker = b"nondeterministic";
    let at = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("stored label");
    assert!(at > 0 && bytes[at - 1] == marker.len() as u8);
    let parts: [&[u8]; 4] = [
        &bytes[..at - 1],
        &[b"deterministic".len() as u8],
        b"deterministic",
        &bytes[at + marker.len()..],
    ];
    std::fs::write(dst, parts.concat()).expect("write forged evidence");
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
        let forged = format!("{ev}.forged");
        forge_deterministic(Path::new(&ev), Path::new(&forged));
        let (ok, text) = mindc(&["verify", "--require-deterministic", &forged]);
        assert!(
            !ok && text.contains("TAMPERED"),
            "`{call}` relabelled deterministic was not refused as TAMPERED\n{text}"
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

/// A project (`.git` boundary, `Mind.toml`, `main.mind` and `files`) in a fresh
/// scratch directory; returns that directory. `main.mind` imports `w` and only
/// calls `w.g`, so whatever the program does beyond that, it does in `w` (and
/// in the modules `w` imports).
#[cfg(feature = "cross-module-imports")]
fn sibling_project(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = common::scratch_dir(&format!("determinism_sibling_{tag}"));
    std::fs::create_dir_all(dir.join(".git")).expect("create project boundary");
    let sources = std::iter::once("main.mind")
        .chain(files.iter().map(|(name, _)| *name))
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let manifest = format!(
        "[package]\nname = \"pw\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"pw\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [{sources}]\n"
    );
    common::scratch_src_in(&dir, "Mind.toml", &manifest);
    common::scratch_src_in(
        &dir,
        "main.mind",
        "import w;\n\npub fn f(p: i64) -> i64 {\n    w.g(p)\n}\n",
    );
    for (name, text) in files {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create source parent");
        }
        common::scratch_src_in(&dir, name, text);
    }
    dir
}

/// `mindc <args>` run in `dir`: `(success, stdout + stderr)`.
#[cfg(feature = "cross-module-imports")]
fn mindc_in(dir: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(common::mindc_bin())
        .args(args)
        .current_dir(dir)
        .output()
        .expect("spawn mindc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// One project whose world read sits one or more calls away from the entry:
/// in `w.g`, behind a private helper of `w`, or in a third module `v` that `w`
/// imports.
#[cfg(feature = "cross-module-imports")]
struct SiblingCase {
    tag: &'static str,
    /// Project files besides `main.mind`: `(file name, source)`.
    files: &'static [(&'static str, &'static str)],
    /// The callee the refusal names.
    offender: &'static str,
    /// The tail of the project function whose body calls it.
    via: &'static str,
}

#[cfg(feature = "cross-module-imports")]
const SIBLING_WORLD: &[SiblingCase] = &[
    SiblingCase {
        tag: "entropy",
        files: &[(
            "w.mind",
            "import std.entropy;\n\npub fn g(p: i64) -> i64 {\n    entropy.entropy_fill(p, 8)\n}\n",
        )],
        offender: "entropy_fill",
        via: "::g`",
    },
    SiblingCase {
        tag: "dns",
        files: &[(
            "w.mind",
            "import std.dns;\n\npub fn g(p: i64) -> i64 {\n    dns.dns_resolve(p, 9)\n}\n",
        )],
        offender: "dns_resolve",
        via: "::g`",
    },
    SiblingCase {
        tag: "now_ns",
        files: &[(
            "w.mind",
            "import std.time;\n\npub fn g(p: i64) -> i64 {\n    time.now_ns() + p\n}\n",
        )],
        offender: "now_ns",
        via: "::g`",
    },
    SiblingCase {
        tag: "helper",
        files: &[(
            "w.mind",
            "import std.entropy;\n\nfn h(p: i64) -> i64 {\n    entropy.entropy_fill(p, 8)\n}\n\n\
             pub fn g(p: i64) -> i64 {\n    h(p) + 1\n}\n",
        )],
        offender: "entropy_fill",
        via: "::h`",
    },
    SiblingCase {
        tag: "third_module",
        files: &[
            (
                "w.mind",
                "import v;\n\npub fn g(p: i64) -> i64 {\n    v.k(p) + 1\n}\n",
            ),
            (
                "v.mind",
                "import std.dns;\n\npub fn k(p: i64) -> i64 {\n    dns.tcp_connect_host(p, 9, 80)\n}\n",
            ),
        ],
        offender: "tcp_connect_host",
        via: "::k`",
    },
];

/// A program whose sibling module reads the world is that program: refused
/// without `--allow-nondeterministic`, naming the callee and the project
/// function that makes the call; with it, attested `nondeterministic`, which
/// `mindc verify --require-deterministic` rejects while plain `verify` accepts
/// the label (the body alone cannot show the sibling, and the stricter label
/// is not a tamper). Before the calls were followed, every case here emitted,
/// attested `deterministic` and passed `--require-deterministic`.
#[cfg(feature = "cross-module-imports")]
#[test]
fn sibling_module_world_calls_are_refused_and_attested_nondeterministic() {
    let mut ran = 0usize;
    for SiblingCase {
        tag,
        files,
        offender,
        via,
    } in SIBLING_WORLD
    {
        let dir = sibling_project(tag, files);
        let (ok, text) = mindc_in(&dir, &["main.mind", "--emit-evidence", "x.ev"]);
        assert!(
            !ok && text.contains("error[determinism]") && text.contains(&format!("`{offender}()`")),
            "{tag}: a program whose sibling calls `{offender}` was not refused naming it\n{text}"
        );
        assert!(
            text.contains("called by `") && text.contains(via),
            "{tag}: the refusal does not name the project function `{via}`\n{text}"
        );
        let (ok, text) = mindc_in(
            &dir,
            &[
                "main.mind",
                "--emit-evidence",
                "x.ev",
                "--allow-nondeterministic",
            ],
        );
        assert!(ok, "{tag}: --allow-nondeterministic did not emit\n{text}");
        let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
        assert!(
            !ok && text.contains("determinism:      nondeterministic"),
            "{tag}: the evidence passed --require-deterministic\n{text}"
        );
        let (ok, text) = mindc_in(&dir, &["verify", "x.ev"]);
        assert!(
            ok && !text.contains("TAMPERED") && text.contains("determinism:      nondeterministic"),
            "{tag}: plain verify did not accept the nondeterministic label\n{text}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(sibling world): ran={ran} fail=0");
    assert_eq!(ran, SIBLING_WORLD.len(), "vacuous run: ran={ran}");
}

/// The control: a sibling whose called function is pure stays `deterministic`,
/// even though the same module holds a world-reading function the program never
/// calls, so a refusal for any other reason cannot pass as the veto.
#[cfg(feature = "cross-module-imports")]
#[test]
fn a_pure_sibling_call_is_still_attested_deterministic() {
    let w = "import std.entropy;\n\npub fn g(p: i64) -> i64 {\n    entropy.entropy_max_len() + p\n}\n\n\
             pub fn unused(p: i64) -> i64 {\n    entropy.entropy_fill(p, 8)\n}\n";
    let dir = sibling_project("pure", &[("w.mind", w)]);
    let (ok, text) = mindc_in(&dir, &["main.mind", "--emit-evidence", "x.ev"]);
    assert!(
        ok,
        "a program calling only a pure sibling function was refused\n{text}"
    );
    let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
    assert!(
        ok && text.contains("determinism:      deterministic"),
        "the evidence of a pure sibling call is not deterministic\n{text}"
    );
    println!("determinism_veto_control(sibling pure): ran=1 fail=0");
}

/// Overwrite the entry `sibling_project` wrote, keeping its manifest.
#[cfg(feature = "cross-module-imports")]
fn set_main(dir: &Path, text: &str) {
    common::scratch_src_in(dir, "main.mind", text);
}

/// A user function spelled like a std world function is deterministic. The
/// reproduction is a project callee `w.dns_resolve`, a `#[deterministic]`
/// callee `w.tcp_connect_host`, and a same-file bare `entropy_fill`. Each
/// emitted and passed `--require-deterministic` before these names were
/// matched as bare builtins. The positive control is a real `dns.dns_resolve`
/// inside `#[deterministic]`, which stays a type error with the allow flag.
#[cfg(feature = "cross-module-imports")]
#[test]
fn a_user_function_named_like_a_std_world_fn_is_deterministic() {
    let dir = sibling_project(
        "user_dns",
        &[(
            "w.mind",
            "pub fn dns_resolve(p: i64, n: i64) -> i64 {\n    p + n\n}\n",
        )],
    );
    set_main(
        &dir,
        "import w;\n\npub fn f(p: i64) -> i64 {\n    w.dns_resolve(p, 9)\n}\n",
    );
    let (ok, text) = mindc_in(&dir, &["main.mind", "--emit-evidence", "x.ev"]);
    assert!(ok, "w.dns_resolve was refused\n{text}");
    let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
    assert!(
        ok && text.contains("determinism:      deterministic"),
        "w.dns_resolve was not attested deterministic\n{text}"
    );

    let dir = sibling_project(
        "user_tcp",
        &[(
            "w.mind",
            "#[deterministic]\npub fn tcp_connect_host(a: i64, b: i64, c: i64) -> i64 {\n    \
             a + b + c\n}\n",
        )],
    );
    set_main(
        &dir,
        "import w;\n\n#[deterministic]\npub fn f(p: i64) -> i64 {\n    \
         w.tcp_connect_host(p, 9, 80)\n}\n",
    );
    let legs: [&[&str]; 2] = [&[], &["--allow-nondeterministic"]];
    for flags in legs {
        let base: &[&str] = &["main.mind", "--emit-evidence", "x.ev"];
        let args = [base, flags].concat();
        let (ok, text) = mindc_in(&dir, &args);
        assert!(
            ok && !text.contains("determinism::nondeterministic_in_deterministic"),
            "#[deterministic] w.tcp_connect_host was refused {flags:?}\n{text}"
        );
    }
    let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
    assert!(
        ok && text.contains("determinism:      deterministic"),
        "#[deterministic] w.tcp_connect_host was not attested deterministic\n{text}"
    );

    let dir = common::scratch_dir("determinism_bare_entropy_fill");
    let src = common::scratch_src_in(
        &dir,
        "bare.mind",
        "fn entropy_fill(p: i64, n: i64) -> i64 {\n    p + n\n}\n\n\
         pub fn f(p: i64) -> i64 {\n    entropy_fill(p, 8)\n}\n",
    );
    let ev = dir.join("bare.ev");
    let (ok, text) = mindc(&[
        src.to_str().unwrap(),
        "--emit-evidence",
        ev.to_str().unwrap(),
    ]);
    assert!(ok, "a bare entropy_fill was refused\n{text}");
    let (ok, text) = mindc(&["verify", "--require-deterministic", ev.to_str().unwrap()]);
    assert!(
        ok && text.contains("determinism:      deterministic"),
        "a bare entropy_fill was not attested deterministic\n{text}"
    );

    let dir = common::scratch_dir("determinism_qualified_dns_in_deterministic");
    let src = common::scratch_src_in(
        &dir,
        "qual.mind",
        "import std.dns;\n\n#[deterministic]\npub fn f(p: i64) -> i64 {\n    \
         dns.dns_resolve(p, 9)\n}\n",
    );
    let ev = dir.join("qual.ev");
    let (ok, text) = mindc(&[
        src.to_str().unwrap(),
        "--emit-evidence",
        ev.to_str().unwrap(),
        "--allow-nondeterministic",
    ]);
    assert!(
        !ok && text.contains("determinism::nondeterministic_in_deterministic"),
        "#[deterministic] dns.dns_resolve was accepted\n{text}"
    );
    println!("determinism_veto_control(user names): ran=4 fail=0");
}

/// A sibling that does not lower is refused, not merely labelled. P4b's world
/// read is `time.now_ns` beside a function that does not resolve. P13b's world
/// read is `entropy.entropy_fill` in a project module the sibling imports as
/// `std.ownrng`, which that sibling cannot lower with, and its own source shows
/// no qualified std call: the refusal and the label both come from the
/// unclassifiable-export arm. Before, both programs emitted without the flag.
#[cfg(feature = "cross-module-imports")]
#[test]
fn an_unlowerable_sibling_world_read_is_refused() {
    let dir = sibling_project(
        "p4b",
        &[(
            "w.mind",
            "import std.time;\n\npub fn g(p: i64) -> i64 {\n    time.now_ns() + p\n}\n\n\
             pub fn broken(p: i64) -> i64 {\n    no_such_fn(p)\n}\n",
        )],
    );
    let (ok, text) = mindc_in(&dir, &["main.mind", "--emit-evidence", "x.ev"]);
    assert!(
        !ok && text.contains("error[determinism]: `g()`")
            && text.contains("::g`, in a project module"),
        "P4b was not refused naming the unclassifiable export\n{text}"
    );
    let (ok, text) = mindc_in(
        &dir,
        &[
            "main.mind",
            "--emit-evidence",
            "x.ev",
            "--allow-nondeterministic",
        ],
    );
    assert!(ok, "P4b with --allow-nondeterministic did not emit\n{text}");
    let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
    assert!(
        !ok && text.contains("determinism:      nondeterministic"),
        "P4b evidence passed --require-deterministic\n{text}"
    );
    let (ok, text) = mindc_in(&dir, &["verify", "x.ev"]);
    assert!(
        ok && !text.contains("TAMPERED"),
        "P4b plain verify rejected the honest label\n{text}"
    );

    // P13b as the review wrote it (`import std.ownrng`): `w` does not lower,
    // so `g` itself is refused. Named without its `std.` prefix the module
    // resolves, and the call is followed to the world read in `k`.
    let mut ran = 1usize;
    let p13b = [
        ("p13b", "std.ownrng", "`g()`", "::g`, in a project module"),
        (
            "p13b_bare",
            "ownrng",
            "`entropy_fill()`",
            "ownrng::k`, in a project module",
        ),
    ];
    for (tag, import, offender, via) in p13b {
        let w =
            format!("import {import};\n\npub fn g(p: i64) -> i64 {{\n    ownrng.k(p) + 1\n}}\n");
        let dir = sibling_project(
            tag,
            &[
                ("w.mind", w.as_str()),
                (
                    "std/ownrng.mind",
                    "import std.entropy;\n\npub fn k(p: i64) -> i64 {\n    \
                     entropy.entropy_fill(p, 8)\n}\n",
                ),
            ],
        );
        let (ok, text) = mindc_in(&dir, &["main.mind", "--emit-evidence", "x.ev"]);
        assert!(
            !ok && text.contains(&format!("error[determinism]: {offender}")) && text.contains(via),
            "{tag} was not refused naming {offender} via {via}\n{text}"
        );
        let args = [
            "main.mind",
            "--emit-evidence",
            "x.ev",
            "--allow-nondeterministic",
        ];
        let (ok, text) = mindc_in(&dir, &args);
        assert!(
            ok,
            "{tag} with --allow-nondeterministic did not emit\n{text}"
        );
        let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
        assert!(
            !ok && text.contains("determinism:      nondeterministic"),
            "{tag} evidence passed --require-deterministic\n{text}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(unlowerable sibling): ran={ran} fail=0");
    assert_eq!(ran, 3, "vacuous run: ran={ran}");
}

/// Rewriting only the stored determinism label of a sibling world read must
/// not pass `verify --require-deterministic`. The entry body does not hold the
/// sibling, so the evidence copy carries a witness the verifier re-derives.
#[cfg(feature = "cross-module-imports")]
#[test]
fn rewriting_the_sibling_label_to_deterministic_fails_verify() {
    let dir = sibling_project(
        "relabel",
        &[(
            "w.mind",
            "import std.time;\n\npub fn g(p: i64) -> i64 {\n    time.now_ns() + p\n}\n",
        )],
    );
    let (ok, text) = mindc_in(
        &dir,
        &[
            "main.mind",
            "--emit-evidence",
            "x.ev",
            "--allow-nondeterministic",
        ],
    );
    assert!(ok, "sibling now_ns did not emit\n{text}");
    forge_deterministic(&dir.join("x.ev"), &dir.join("y.ev"));
    let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "y.ev"]);
    assert!(
        !ok && text.contains("TAMPERED"),
        "a label rewritten to deterministic passed --require-deterministic\n{text}"
    );
    let (ok, text) = mindc_in(&dir, &["verify", "y.ev"]);
    assert!(
        !ok && text.contains("TAMPERED"),
        "plain verify did not report the rewritten label as TAMPERED\n{text}"
    );
    println!("determinism_veto_control(relabel): ran=1 fail=0");
}
