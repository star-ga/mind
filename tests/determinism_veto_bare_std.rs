// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A bare call of an imported `std.dns` / `std.entropy` world function, and of
//! the CPU affinity names of `std.iouring` / `std.cpu_affinity`, which are in
//! the implicit std set and so resolve with no import at all.
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

/// `(call, the row it names)`: the CPU affinity names of `std.iouring` and of
/// `std.cpu_affinity`. Both modules are in the implicit std set, so a bare call
/// resolves to the std function with no import.
const AFFINITY_CALLS: &[(&str, &str)] = &[
    ("io_reactor_pin_cpu(p)", "io_reactor_pin_cpu"),
    ("io_reactor_affinity_mask()", "io_reactor_affinity_mask"),
    ("io_reactor_affinity_has(p)", "io_reactor_affinity_has"),
    ("io_reactor_pin_demo(p)", "io_reactor_pin_demo"),
    ("cpu_affinity_pin(p)", "cpu_affinity_pin"),
    ("cpu_affinity_mask()", "cpu_affinity_mask"),
    ("cpu_affinity_has(p)", "cpu_affinity_has"),
    ("cpu_affinity_pin_demo(p)", "cpu_affinity_pin_demo"),
];

/// A bare affinity call with NO import is the std function: refused without
/// the flag, attested `nondeterministic` with it (which `verify
/// --require-deterministic` rejects), and a type error inside
/// `#[deterministic]` even with the flag. The program makes the real affinity
/// system call, so it is not a deterministic program.
#[test]
fn a_bare_implicit_std_affinity_call_is_a_world_call() {
    let dir = common::scratch_dir("determinism_bare_affinity_world");
    for (i, (call, offender)) in AFFINITY_CALLS.iter().enumerate() {
        let text = format!("pub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
        let src = common::scratch_src_in(&dir, &format!("w{i}.mind"), &text);
        let ev = dir.join(format!("w{i}.ev"));
        let (src, ev) = (src.to_str().unwrap(), ev.to_str().unwrap());
        let (ok, out) = mindc(&[src, "--emit-evidence", ev]);
        assert!(
            !ok && out.contains("error[determinism]") && out.contains(&format!("`{offender}()`")),
            "a bare `{call}` was not refused naming `{offender}()`\n{out}"
        );
        let (ok, out) = mindc(&[src, "--emit-evidence", ev, "--allow-nondeterministic"]);
        assert!(
            ok,
            "`{call}` with --allow-nondeterministic did not emit\n{out}"
        );
        let (ok, out) = mindc(&["verify", "--require-deterministic", ev]);
        assert!(
            !ok && out.contains("determinism:      nondeterministic"),
            "the evidence of a bare `{call}` passed --require-deterministic\n{out}"
        );
        let text = format!("#[deterministic]\npub fn f(p: i64) -> i64 {{\n    {call}\n}}\n");
        let src = common::scratch_src_in(&dir, &format!("d{i}.mind"), &text);
        let ev = dir.join(format!("d{i}.ev"));
        let (ok, out) = mindc(&[
            src.to_str().unwrap(),
            "--emit-evidence",
            ev.to_str().unwrap(),
            "--allow-nondeterministic",
        ]);
        assert!(
            !ok && out.contains("determinism::nondeterministic_in_deterministic")
                && out.contains(&format!("`{offender}`")),
            "#[deterministic] bare `{call}` was accepted\n{out}"
        );
    }
    println!("determinism_veto_control(bare affinity): ran=16 fail=0");
}

/// The reach through a helper function of the same file is the same call, with
/// the std module imported and without. A pure function of `std.iouring` called
/// the same way stays deterministic, so a refusal for any other reason cannot
/// pass as the veto.
#[test]
fn an_affinity_call_behind_a_helper_function_is_a_world_call() {
    let dir = common::scratch_dir("determinism_helper_affinity_world");
    let cases = [
        (
            "import std.iouring;\n\n",
            "io_reactor_affinity_mask()",
            "io_reactor_affinity_mask",
        ),
        (
            "import std.cpu_affinity;\n\n",
            "cpu_affinity_pin(p)",
            "cpu_affinity_pin",
        ),
        ("", "io_reactor_pin_demo(p)", "io_reactor_pin_demo"),
        ("", "cpu_affinity_has(p)", "cpu_affinity_has"),
    ];
    for (i, (import, call, offender)) in cases.iter().enumerate() {
        let text = format!(
            "{import}fn helper(p: i64) -> i64 {{\n    {call}\n}}\n\n\
             pub fn f(p: i64) -> i64 {{\n    helper(p) + 1\n}}\n"
        );
        let src = common::scratch_src_in(&dir, &format!("h{i}.mind"), &text);
        let ev = dir.join(format!("h{i}.ev"));
        let (ok, out) = mindc(&[
            src.to_str().unwrap(),
            "--emit-evidence",
            ev.to_str().unwrap(),
        ]);
        assert!(
            !ok && out.contains("error[determinism]") && out.contains(&format!("`{offender}()`")),
            "`{call}` behind a helper was not refused naming `{offender}()`\n{out}"
        );
    }
    let pure = "import std.iouring;\n\nfn helper(p: i64) -> i64 {\n    iouring_cqe_f_more()\n}\n\n\
                pub fn f(p: i64) -> i64 {\n    helper(p) + 1\n}\n";
    let src = common::scratch_src_in(&dir, "pure.mind", pure);
    let ev = dir.join("pure.ev");
    let (ok, out) = mindc(&[
        src.to_str().unwrap(),
        "--emit-evidence",
        ev.to_str().unwrap(),
    ]);
    assert!(
        ok,
        "a pure std.iouring call behind a helper was refused\n{out}"
    );
    let (ok, out) = mindc(&["verify", "--require-deterministic", ev.to_str().unwrap()]);
    assert!(
        ok && out.contains("determinism:      deterministic"),
        "a pure std.iouring call behind a helper is not attested deterministic\n{out}"
    );
}

/// A function of the module spelled like an affinity name is the user's: it
/// stays deterministic with no import. The std calls above are the positive
/// control for the same spellings.
#[test]
fn a_user_function_named_like_an_affinity_name_is_deterministic() {
    let dir = common::scratch_dir("determinism_shadow_affinity");
    for (i, (_, name)) in AFFINITY_CALLS.iter().enumerate() {
        let text = format!(
            "fn {name}(p: i64) -> i64 {{\n    p + 1\n}}\n\n\
             pub fn f(p: i64) -> i64 {{\n    {name}(p)\n}}\n"
        );
        let src = common::scratch_src_in(&dir, &format!("s{i}.mind"), &text);
        let ev = dir.join(format!("s{i}.ev"));
        let (ok, out) = mindc(&[
            src.to_str().unwrap(),
            "--emit-evidence",
            ev.to_str().unwrap(),
        ]);
        assert!(ok, "a module function `{name}` was refused\n{out}");
        let (ok, out) = mindc(&["verify", "--require-deterministic", ev.to_str().unwrap()]);
        assert!(
            ok && out.contains("determinism:      deterministic"),
            "a module function `{name}` was not attested deterministic\n{out}"
        );
    }
}
