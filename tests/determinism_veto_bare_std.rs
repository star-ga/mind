// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A bare call of an imported `std.dns` / `std.entropy` world function, and of
//! the CPU affinity names of `std.iouring` / `std.cpu_affinity`, which are in
//! the implicit std set and so resolve with no import at all.
//!
//! Companion to `determinism_veto_control`, which holds the qualified spellings
//! and the sibling-module cases (the review's A, B and D).
//! This file holds the `#[deterministic]` case (C) and the shadowing controls,
//! for a user function and for a local binding.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;

use std::path::Path;
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
        (
            "std.os_evidence",
            "cpu_affinity_pin_ev(p, 0)",
            "cpu_affinity_pin_ev",
        ),
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
        (
            "std.os_evidence",
            "fn cpu_affinity_pin_ev(p: i64, n: i64) -> i64 {\n    p + n\n}",
            "cpu_affinity_pin_ev(p, 0)",
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
    println!("determinism_veto_control(bare resolution): ran=7 fail=0");
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
    // The wrappers `std.cpu_affinity` makes its system calls through: the module
    // is in the implicit std set too.
    ("cpu_affinity_sys_set(8, p)", "cpu_affinity_sys_set"),
    ("cpu_affinity_sys_get(8, p)", "cpu_affinity_sys_get"),
    ("cpu_affinity_sys_errno()", "cpu_affinity_sys_errno"),
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
    println!(
        "determinism_veto_control(bare affinity): ran={} fail=0",
        AFFINITY_CALLS.len() * 2
    );
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

/// Local-binding shapes: `@N` is the binding, read as a value. A `let`, a
/// `let mut`, a parameter, a loop variable, a match arm binding and a module `const`.
const LOCAL_SHAPES: &[&str] = &[
    "pub fn f(p: i64) -> i64 {\n    let @N: i64 = p;\n    @N + 1\n}\n",
    "pub fn f(p: i64) -> i64 {\n    let mut @N: i64 = p;\n    @N = @N + 1;\n    @N\n}\n",
    "pub fn f(@N: i64) -> i64 {\n    @N + 1\n}\n",
    "pub fn f(p: i64) -> i64 {\n    let mut s: i64 = 0;\n    for @N in 0..p {\n        \
     s = s + @N;\n    }\n    s\n}\n",
    "pub fn f(p: i64) -> i64 {\n    match p {\n        0 => 7,\n        @N => @N + 1,\n    }\n}\n",
    "const @N: i64 = 3\n\npub fn f(p: i64) -> i64 {\n    @N + p\n}\n",
];

/// `text` built to evidence with no flag as `dir/<tag>.mind`: `(ok, output, evidence)`.
fn emit_evidence(dir: &Path, tag: &str, text: &str) -> (bool, String, String) {
    let src = common::scratch_src_in(dir, &format!("{tag}.mind"), text);
    let ev = dir.join(format!("{tag}.ev")).to_string_lossy().into_owned();
    let (ok, out) = mindc(&[src.to_str().unwrap(), "--emit-evidence", &ev]);
    (ok, out, ev)
}

/// The first error line of `out`, for a one-line failure summary.
fn first_error(out: &str) -> String {
    let line = out
        .lines()
        .find(|l| l.contains("error"))
        .unwrap_or("no error");
    line.chars().take(110).collect()
}

/// A local variable, parameter, loop variable or match binding spelled like an
/// affinity name is a value, not a call: the program makes no affinity system call,
/// so it builds with no flag and is attested `deterministic`, with the std module
/// imported and with none. A bare function value is a type error (E2037), so a
/// bare read of the name is always a local. Two-sided: the same program plus a
/// real call of the same name in another function is refused naming that call,
/// and so is a real call of another name in the same function.
#[test]
fn a_local_binding_named_like_an_affinity_name_is_deterministic() {
    let dir = common::scratch_dir("determinism_local_affinity");
    let (mut ran, mut failed) = (0usize, Vec::<String>::new());
    for (i, (call, name)) in AFFINITY_CALLS.iter().enumerate() {
        let module = if name.starts_with("io_reactor") {
            "std.iouring"
        } else if name.starts_with("cpu_affinity_sys_") {
            "std.cpu_affinity_sys"
        } else {
            "std.cpu_affinity"
        };
        let imports = ["".to_string(), format!("import {module};\n\n")];
        let (other_call, other) = AFFINITY_CALLS[(i + 1) % AFFINITY_CALLS.len()];
        for (j, shape) in LOCAL_SHAPES.iter().enumerate() {
            for (k, import) in imports.iter().enumerate() {
                let tag = format!("l{i}_{j}_{k}");
                let local = format!("{import}{}", shape.replace("@N", name));
                let (ok, out, ev) = emit_evidence(&dir, &tag, &local);
                let (vok, vout) = if ok {
                    mindc(&["verify", "--require-deterministic", &ev])
                } else {
                    (false, String::new())
                };
                if !(ok && vok && vout.contains("determinism:      deterministic")) {
                    failed.push(format!("{tag}: local `{name}`: {}", first_error(&out)));
                }
                // A module `const` is visible to every function, so calling its own
                // name there is a type error (E2012): that twin calls another name.
                let (twin_call, twin) = if shape.starts_with("const") {
                    (other_call, other)
                } else {
                    (*call, *name)
                };
                let world = format!("{local}\nfn h(p: i64) -> i64 {{\n    {twin_call}\n}}\n");
                let (ok, out, _) = emit_evidence(&dir, &format!("{tag}w"), &world);
                if ok
                    || !out.contains("error[determinism]")
                    || !out.contains(&format!("`{twin}()`"))
                {
                    failed.push(format!("{tag}w: call `{twin}`: {}", first_error(&out)));
                }
                ran += 2;
                if j == 0 {
                    let call_it = shape.replace("@N + 1", &format!("@N + {other_call}"));
                    let same = format!("{import}{}", call_it.replace("@N", name));
                    let (ok, out, _) = emit_evidence(&dir, &format!("{tag}s"), &same);
                    if ok || !out.contains(&format!("`{other}()`")) {
                        failed.push(format!("{tag}s: call `{other}`: {}", first_error(&out)));
                    }
                    ran += 1;
                }
            }
        }
    }
    println!(
        "determinism_veto_control(local affinity): ran={ran} fail={}",
        failed.len()
    );
    assert!(
        failed.is_empty(),
        "{} of {ran} failed:\n{}",
        failed.len(),
        failed.join("\n")
    );
    assert_eq!(
        ran,
        AFFINITY_CALLS.len() * (LOCAL_SHAPES.len() * 2 * 2 + 2),
        "vacuous run"
    );
}
