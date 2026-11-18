// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The CPU affinity wrappers `cpu_affinity_sys_set`, `cpu_affinity_sys_get` and
//! `cpu_affinity_sys_errno` of `std.cpu_affinity_sys`, and `cpu_affinity_pin_ev`
//! of `std.os_evidence`, are World calls like the names of `std.cpu_affinity`
//! (rows of `src/std_qualified.rs`).
//!
//! Companion to `determinism_veto_bare_std` and `determinism_veto_std_callers`,
//! which hold the bare, qualified, shadowing and local-binding cases of the same
//! names. This file holds what they do not: the three emit routes (`--emit-obj`,
//! `--emit-shared`, `--emit-evidence`) for each name, a call through a sibling
//! project module, and the entry compile of `std/cpu_affinity.mind`, whose calls
//! into the wrapper module no longer sit beside an `extern` block.
//!
//! The veto runs on the lowered entry before any emission backend is consulted,
//! so this gate needs no MLIR tool. With the flag the object and shared-library
//! routes may still fail for a toolchain reason: what must never appear there is
//! `error[determinism]`. The evidence route always emits, and its label is read
//! back with `mindc verify --require-deterministic`.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// `(import line, call, the name the refusal must name)`. The qualified, the `::`
/// and the bare spellings, with the import and (for the implicit wrapper module)
/// without it.
const NEW_WORLD_CALLS: &[(&str, &str, &str)] = &[
    (
        "import std.cpu_affinity_sys;\n\n",
        "cpu_affinity_sys.cpu_affinity_sys_set(8, p)",
        "cpu_affinity_sys_set",
    ),
    (
        "import std.cpu_affinity_sys;\n\n",
        "std::cpu_affinity_sys::cpu_affinity_sys_get(8, p)",
        "cpu_affinity_sys_get",
    ),
    (
        "import std.cpu_affinity_sys;\n\n",
        "cpu_affinity_sys.cpu_affinity_sys_errno()",
        "cpu_affinity_sys_errno",
    ),
    ("", "cpu_affinity_sys_set(8, p)", "cpu_affinity_sys_set"),
    ("", "cpu_affinity_sys_get(8, p)", "cpu_affinity_sys_get"),
    ("", "cpu_affinity_sys_errno()", "cpu_affinity_sys_errno"),
    (
        "import std.os_evidence;\n\n",
        "os_evidence.cpu_affinity_pin_ev(p, 0)",
        "cpu_affinity_pin_ev",
    ),
    (
        "import std.os_evidence;\n\n",
        "cpu_affinity_pin_ev(p, 0)",
        "cpu_affinity_pin_ev",
    ),
];

/// The three artifact routes of `mindc`.
const ROUTES: &[&str] = &["--emit-obj", "--emit-shared", "--emit-evidence"];

/// `mindc <args>` in `dir`: `(success, stdout + stderr)`.
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

/// A source file `dir/<tag>.mind` holding `import; pub fn f(p: i64) -> i64 { call }`.
fn entry(dir: &Path, tag: &str, import: &str, call: &str) -> String {
    let text = format!("{import}pub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
    let path = common::scratch_src_in(dir, &format!("{tag}.mind"), &text);
    path.to_string_lossy().into_owned()
}

/// The refusal text of the determinism veto naming `offender`.
fn refused_naming(ok: bool, text: &str, offender: &str) -> bool {
    !ok && text.contains("error[determinism]") && text.contains(&format!("`{offender}()`"))
}

/// Each new name, on each route: refused without the flag, naming itself. With the
/// flag the veto is silent, and the evidence route emits a record that
/// `verify --require-deterministic` rejects (the label is `nondeterministic`).
#[test]
fn the_new_affinity_names_are_refused_on_every_emit_route() {
    let dir = common::scratch_dir("determinism_affinity_sys_routes");
    let mut ran = 0usize;
    for (i, (import, call, offender)) in NEW_WORLD_CALLS.iter().enumerate() {
        let src = entry(&dir, &format!("r{i}"), import, call);
        for route in ROUTES {
            let out = dir.join(format!("r{i}{}", route.trim_start_matches('-')));
            let out = out.to_string_lossy().into_owned();
            let (ok, text) = mindc_in(&dir, &[&src, route, &out]);
            assert!(
                refused_naming(ok, &text, offender),
                "`{call}` on {route} was not refused naming `{offender}()`\n{text}"
            );
            let (ok, text) = mindc_in(&dir, &[&src, route, &out, "--allow-nondeterministic"]);
            assert!(
                !text.contains("error[determinism]"),
                "`{call}` on {route}: --allow-nondeterministic did not authorise it\n{text}"
            );
            if *route == "--emit-evidence" {
                assert!(ok, "`{call}` with the flag did not emit evidence\n{text}");
                let (vok, vtext) = mindc_in(&dir, &["verify", "--require-deterministic", &out]);
                assert!(
                    !vok && vtext.contains("determinism:      nondeterministic"),
                    "the evidence of `{call}` passed --require-deterministic\n{vtext}"
                );
            }
            ran += 1;
        }
    }
    println!("determinism_veto_control(affinity sys routes): ran={ran} fail=0");
    assert_eq!(ran, NEW_WORLD_CALLS.len() * ROUTES.len(), "vacuous run");
}

/// Positive control for the routes above: the same program with a local binding
/// spelled like each wrapper (a value, not a call) emits evidence with no flag and
/// is attested `deterministic`, so a refusal for any other reason cannot pass.
#[test]
fn a_local_binding_named_like_a_wrapper_is_still_deterministic() {
    let dir = common::scratch_dir("determinism_affinity_sys_local");
    let cases = [
        ("import std.cpu_affinity_sys;\n\n", "cpu_affinity_sys_set"),
        ("", "cpu_affinity_sys_get"),
        ("import std.cpu_affinity_sys;\n\n", "cpu_affinity_sys_errno"),
        ("import std.os_evidence;\n\n", "cpu_affinity_pin_ev"),
    ];
    for (i, (import, name)) in cases.iter().enumerate() {
        let call = format!("let {name}: i64 = p;\n    {name} + 1");
        let src = entry(&dir, &format!("l{i}"), import, &call);
        let ev = dir.join(format!("l{i}.ev")).to_string_lossy().into_owned();
        let (ok, text) = mindc_in(&dir, &[&src, "--emit-evidence", &ev]);
        assert!(ok, "a local `{name}` was refused\n{text}");
        let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", &ev]);
        assert!(
            ok && text.contains("determinism:      deterministic"),
            "the evidence of a local `{name}` is not deterministic\n{text}"
        );
    }
}

/// A project (`.git` boundary, `Mind.toml`, `main.mind` and `files`) in a fresh
/// scratch directory. `main.mind` imports `w` and only calls `w.g`, so whatever
/// the program does beyond that, it does in `w` (and in the modules `w` imports).
fn sibling_project(tag: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = common::scratch_dir(&format!("determinism_affinity_sys_{tag}"));
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
        common::scratch_src_in(&dir, name, text);
    }
    dir
}

/// `(tag, files, the name the refusal names, the tail of the project function that
/// makes the call)`: the call directly in `w.g`, behind a private helper of `w`
/// that calls it bare with no import, and in a third module `v`.
type Sibling = (
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static str,
    &'static str,
);

const SIBLINGS: &[Sibling] = &[
    (
        "set_qualified",
        &[(
            "w.mind",
            "import std.cpu_affinity_sys;\n\npub fn g(p: i64) -> i64 {\n    \
             cpu_affinity_sys.cpu_affinity_sys_set(8, p)\n}\n",
        )],
        "cpu_affinity_sys_set",
        "::g`",
    ),
    (
        "errno_helper_bare",
        &[(
            "w.mind",
            "fn h(p: i64) -> i64 {\n    cpu_affinity_sys_errno() + p\n}\n\n\
             pub fn g(p: i64) -> i64 {\n    h(p) + 1\n}\n",
        )],
        "cpu_affinity_sys_errno",
        "::h`",
    ),
    (
        "pin_ev_bare",
        &[(
            "w.mind",
            "import std.os_evidence;\n\npub fn g(p: i64) -> i64 {\n    cpu_affinity_pin_ev(p, 0)\n}\n",
        )],
        "cpu_affinity_pin_ev",
        "::g`",
    ),
    (
        "get_third_module",
        &[
            (
                "w.mind",
                "import v;\n\npub fn g(p: i64) -> i64 {\n    v.k(p) + 1\n}\n",
            ),
            (
                "v.mind",
                "import std.cpu_affinity_sys;\n\npub fn k(p: i64) -> i64 {\n    \
                 cpu_affinity_sys_get(8, p)\n}\n",
            ),
        ],
        "cpu_affinity_sys_get",
        "::k`",
    ),
];

/// A program whose sibling module makes the call is that program: refused without
/// the flag, naming the callee and the project function; with it, attested
/// `nondeterministic`.
#[test]
fn a_sibling_module_calling_the_new_names_is_refused_and_attested_nondeterministic() {
    let mut ran = 0usize;
    for (tag, files, offender, via) in SIBLINGS {
        let dir = sibling_project(tag, files);
        let (ok, text) = mindc_in(&dir, &["main.mind", "--emit-evidence", "x.ev"]);
        assert!(
            refused_naming(ok, &text, offender),
            "{tag}: a sibling that calls `{offender}` was not refused naming it\n{text}"
        );
        assert!(
            text.contains("called by `") && text.contains(via),
            "{tag}: the refusal does not name the project function `{via}`\n{text}"
        );
        let args = [
            "main.mind",
            "--emit-evidence",
            "x.ev",
            "--allow-nondeterministic",
        ];
        let (ok, text) = mindc_in(&dir, &args);
        assert!(ok, "{tag}: --allow-nondeterministic did not emit\n{text}");
        let (ok, text) = mindc_in(&dir, &["verify", "--require-deterministic", "x.ev"]);
        assert!(
            !ok && text.contains("determinism:      nondeterministic"),
            "{tag}: the evidence passed --require-deterministic\n{text}"
        );
        ran += 1;
    }
    println!("determinism_veto_control(affinity sys siblings): ran={ran} fail=0");
    assert_eq!(ran, SIBLINGS.len(), "vacuous run");
}

/// `std/cpu_affinity.mind` makes its system calls through the wrapper module, so
/// its entry compile holds no `sched_setaffinity` call of its own. The veto still
/// refuses it, through the wrapper row, and the flag still authorises it.
#[test]
fn the_mask_module_entry_is_refused_through_its_wrappers() {
    let dir = common::scratch_dir("determinism_affinity_sys_entry");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("std/cpu_affinity.mind");
    let src = src.to_string_lossy().into_owned();
    let out = dir.join("mask.so").to_string_lossy().into_owned();
    let (ok, text) = mindc_in(&dir, &[&src, "--emit-shared", &out]);
    assert!(
        !ok && text.contains("error[determinism]") && text.contains("`cpu_affinity_sys_"),
        "std/cpu_affinity.mind was not refused through a wrapper name\n{text}"
    );
    let (_, text) = mindc_in(
        &dir,
        &[&src, "--emit-shared", &out, "--allow-nondeterministic"],
    );
    assert!(
        !text.contains("error[determinism]"),
        "--allow-nondeterministic did not authorise std/cpu_affinity.mind\n{text}"
    );
}
