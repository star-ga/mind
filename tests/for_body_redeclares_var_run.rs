// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A range-`for` body that re-declares the loop variable terminates.
//!
//! `for i in 0..3 { let i: i64 = i; … }` built and then never returned: the
//! byte-neutral desugar `let i = 0; while i < 3 { BODY; i = i + 1 }` put the
//! step in the body's own scope, where the body's `let i` had shadowed the
//! counter, so the step advanced the shadow and the loop condition read the
//! counter's start value forever. The interpreter binds `VAR` fresh each
//! iteration, so a re-declaration only shadows it until the iteration ends.
//! The `For` arm in `src/eval/lower.rs` now takes its hygienic form (a hidden
//! counter, `let mut VAR = counter` per iteration) when a top-level body
//! statement re-declares `VAR`.
//!
//! Each case is an executable whose `main` returns the hand-computed sum (kept
//! under 256, the exit status width), run under `timeout` so a regression
//! fails as rc 124 instead of stalling.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::process::Command;

/// Build `main` (the body of `fn main() -> i64`) as a one-file project and
/// return its exit code, or `None` when the build was routed through the
/// capability-skip gate.
fn run(target: &str, main: &str) -> Option<i32> {
    let dir = common::scratch_dir(target);
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"fr\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"fr\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        dir.join("main.mind"),
        format!("fn main() -> i64 {{\n{main}}}\n"),
    )
    .expect("write main.mind");
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .output()
        .expect("run mindc build");
    if !common::gate::compiled(target, &out) {
        return None;
    }
    let run = Command::new("timeout")
        .arg("20")
        .arg(dir.join("target").join("debug").join("fr"))
        .output()
        .expect("run the built executable under timeout");
    Some(run.status.code().expect("exit code"))
}

fn check(target: &str, main: &str, want: i32) {
    if let Some(code) = run(target, main) {
        assert_ne!(code, 124, "{target}: the loop did not terminate");
        assert_eq!(code, want, "{target}");
    }
}

#[test]
fn a_let_copying_the_loop_variable_terminates() {
    // 0 + 1 + 2
    check(
        "for_redeclare_copy",
        "    let mut s: i64 = 0;\n    for i in 0..3 {\n        let i: i64 = i;\n        s = s + i;\n    }\n    return s;\n",
        3,
    );
}

#[test]
fn a_let_rebinding_the_loop_variable_to_a_new_value_terminates() {
    // 0 + 10 + 20
    check(
        "for_redeclare_scaled",
        "    let mut s: i64 = 0;\n    for i in 0..3 {\n        let i: i64 = i * 10;\n        s = s + i;\n    }\n    return s;\n",
        30,
    );
}

#[test]
fn a_re_declaration_with_a_variable_bound_terminates() {
    // 0 + 1 + 2 + 3
    check(
        "for_redeclare_var_bound",
        "    let n: i64 = 4;\n    let mut s: i64 = 0;\n    for k in 0..n {\n        let k: i64 = k + 0;\n        s = s + k;\n    }\n    return s;\n",
        6,
    );
}

#[test]
fn a_re_declaration_mid_body_shadows_only_the_rest_of_the_iteration() {
    // Each iteration adds i, then the shadow's 50: (0+1+2+3) + 4 * 50.
    check(
        "for_redeclare_mid_body",
        "    let mut s: i64 = 0;\n    for i in 0..4 {\n        s = s + i;\n        let i: i64 = 50;\n        s = s + i;\n    }\n    return s;\n",
        206,
    );
}

#[test]
fn a_tuple_binding_that_re_declares_the_loop_variable_terminates() {
    // (1 + 2) + (2 + 2) + (3 + 2)
    check(
        "for_redeclare_tuple",
        "    let mut s: i64 = 0;\n    for i in 0..3 {\n        let (i, j) = (i + 1, 2);\n        s = s + i + j;\n    }\n    return s;\n",
        12,
    );
}

#[test]
fn continue_after_a_re_declaration_still_steps_the_counter() {
    // Shadows 0, 2, 4, 6; the iteration whose shadow is 2 is skipped.
    check(
        "for_redeclare_continue",
        "    let mut s: i64 = 0;\n    for i in 0..4 {\n        let i: i64 = i * 2;\n        if i == 2 {\n            continue;\n        }\n        s = s + i;\n    }\n    return s;\n",
        10,
    );
}

#[test]
fn a_re_declaration_in_a_nested_scope_was_already_scoped() {
    // Controls on the byte-neutral desugar: a `let` inside a branch or an
    // inner loop never shadowed the counter. 1 + (7 + 1) + (7 + 1) and
    // 3 * (0 + 1 + 10).
    check(
        "for_redeclare_in_branch",
        "    let mut s: i64 = 0;\n    for i in 0..3 {\n        if i > 0 {\n            let i: i64 = 7;\n            s = s + i;\n        }\n        s = s + 1;\n    }\n    return s;\n",
        17,
    );
    check(
        "for_redeclare_in_inner_loop",
        "    let mut s: i64 = 0;\n    for i in 0..3 {\n        for j in 0..2 {\n            let i: i64 = j;\n            s = s + i;\n        }\n        s = s + 10;\n    }\n    return s;\n",
        33,
    );
}
