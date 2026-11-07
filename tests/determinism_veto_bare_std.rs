// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A bare call of an imported `std.dns` / `std.entropy` world function.
//!
//! Companion to `determinism_veto_control`, which holds the qualified spellings
//! and the sibling-module cases (the review's A, B and D).
//! This file holds the `#[deterministic]` case (C) and the shadowing controls.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;

use std::process::Command;

/// `mindc <args>`: `(success, stdout + stderr)`.
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

/// A bare call is a world call by name resolution, the way the build resolves
/// it: the module imports the std module that defines the row and defines no
/// function of that name. Case C: inside `#[deterministic]` it is a type error
/// even with `--allow-nondeterministic`. The controls: a module function of
/// the same name shadows the std one (deterministic, flag or not), and so does
/// a bare name the module does not import from `std.dns` / `std.entropy`.
#[test]
fn a_bare_imported_std_world_call_is_resolved_by_name() {
    let dir = common::scratch_dir("determinism_bare_std_world");
    for (i, (import, call, offender)) in [
        ("std.dns", "dns_resolve(p, 9)", "dns_resolve"),
        ("std.entropy", "entropy_fill(p, 8)", "entropy_fill"),
        ("std.dns", "tcp_connect_host(p, 9, 80)", "tcp_connect_host"),
    ]
    .iter()
    .enumerate()
    {
        let text = format!(
            "import {import};\n\n#[deterministic]\npub fn f(p: i64) -> i64 {{\n    {call}\n}}\n"
        );
        let src = common::scratch_src_in(&dir, &format!("c{i}.mind"), &text);
        let ev = dir.join(format!("c{i}.ev"));
        let (ok, text) = mindc(&[
            src.to_str().unwrap(),
            "--emit-evidence",
            ev.to_str().unwrap(),
            "--allow-nondeterministic",
        ]);
        assert!(
            !ok && text.contains("determinism::nondeterministic_in_deterministic")
                && text.contains(&format!("`{offender}`")),
            "#[deterministic] bare `{call}` was accepted\n{text}"
        );
    }
    // Shadowed by a function of the module: deterministic, bare.
    for (i, (import, def, call)) in [
        (
            "std.entropy",
            "fn entropy_fill(p: i64, n: i64) -> i64 {\n    p + n\n}",
            "entropy_fill(p, 8)",
        ),
        (
            "std.dns",
            "fn dns_resolve(p: i64, n: i64) -> i64 {\n    p + n\n}",
            "dns_resolve(p, 9)",
        ),
    ]
    .iter()
    .enumerate()
    {
        let text =
            format!("import {import};\n\n{def}\n\npub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
        let src = common::scratch_src_in(&dir, &format!("s{i}.mind"), &text);
        let ev = dir.join(format!("s{i}.ev"));
        let (ok, text) = mindc(&[
            src.to_str().unwrap(),
            "--emit-evidence",
            ev.to_str().unwrap(),
        ]);
        assert!(
            ok,
            "a module function shadowing `{call}` was refused\n{text}"
        );
        let (ok, text) = mindc(&["verify", "--require-deterministic", ev.to_str().unwrap()]);
        assert!(
            ok && text.contains("determinism:      deterministic"),
            "a module function shadowing `{call}` was not attested deterministic\n{text}"
        );
    }
    println!("determinism_veto_control(bare resolution): ran=5 fail=0");
}
