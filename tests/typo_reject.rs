// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Type-checker silent-miscompile regression gate.
//!
//! Two confirmed silent miscompiles a one-char typo used to produce:
//!   E2008 — a nonexistent qualified enum variant (`Color::Rde`) used as a
//!           value passed `mindc check` and silently lowered to tag 0 (the
//!           first variant), so `c == Color::Red` matched a typo.
//!   E2009 — an assignment to an undeclared variable (`conter = ...`) was
//!           silently auto-created as a fresh binding, so the intended
//!           variable never updated.
//! Both now fail `mindc check`. This gate asserts the typo shapes are REJECTED
//! and the corresponding CORRECT programs still pass (no false positives — the
//! undeclared-assign check must still accept forward-declared module state,
//! which the self-host relies on).
//!
//! We assert on the presence/absence of the diagnostic CODE in stderr rather
//! than the process exit status: `mindc check` also emits a `fmt::drift`
//! diagnostic for an unformatted file, and the exit code / `--no-fmt` auto-fix
//! semantics muddy a pure pass/fail read. Whether E2008/E2009 fires is the
//! behavior under test.

use std::io::Write;
use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn write_tmp(name: &str, src: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&p).expect("create tmp");
    f.write_all(src.as_bytes()).expect("write tmp");
    p
}

/// Returns the combined stdout+stderr of `mindc check <path>`. The default
/// `human` reporter prints diagnostics to stdout, so both streams are read.
fn check_stderr(path: &std::path::Path) -> String {
    let out = mindc()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("spawn mindc");
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn nonexistent_enum_variant_rejected() {
    let bad = write_tmp(
        "mind_typo_enum_bad.mind",
        "enum Color { Red, Green, Blue }\n\
         pub fn pick() -> i64 {\n\
         \x20   let c = Color::Rde\n\
         \x20   if c == Color::Red { return 111 }\n\
         \x20   return 999\n\
         }\n",
    );
    let stderr = check_stderr(&bad);
    assert!(stderr.contains("E2008"), "typo enum variant not rejected (silent miscompile); stderr: {stderr}");
}

#[test]
fn valid_enum_variant_accepted() {
    let good = write_tmp(
        "mind_typo_enum_good.mind",
        "enum Color { Red, Green, Blue }\n\
         pub fn pick() -> i64 {\n\
         \x20   let c = Color::Green\n\
         \x20   if c == Color::Green { return 222 }\n\
         \x20   return 999\n\
         }\n",
    );
    let stderr = check_stderr(&good);
    assert!(!stderr.contains("E2008"), "valid enum variant falsely rejected: {stderr}");
}

#[test]
fn undeclared_assign_target_rejected() {
    let bad = write_tmp(
        "mind_typo_assign_bad.mind",
        "pub fn count_up() -> i64 {\n\
         \x20   let mut counter = 0\n\
         \x20   let mut i = 0\n\
         \x20   while i < 10 {\n\
         \x20       conter = counter + 1\n\
         \x20       i = i + 1\n\
         \x20   }\n\
         \x20   return counter\n\
         }\n",
    );
    let stderr = check_stderr(&bad);
    assert!(stderr.contains("E2009"), "undeclared assign not rejected (silent miscompile); stderr: {stderr}");
}

#[test]
fn declared_reassignment_accepted() {
    let good = write_tmp(
        "mind_typo_assign_good.mind",
        "pub fn count_up() -> i64 {\n\
         \x20   let mut counter = 0\n\
         \x20   let mut i = 0\n\
         \x20   while i < 10 {\n\
         \x20       counter = counter + 1\n\
         \x20       i = i + 1\n\
         \x20   }\n\
         \x20   return counter\n\
         }\n",
    );
    let stderr = check_stderr(&good);
    assert!(!stderr.contains("E2009"), "valid reassignment falsely rejected: {stderr}");
}
