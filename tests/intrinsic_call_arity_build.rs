// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc build` refuses an intrinsic call with the wrong argument count.
//!
//! `__mind_write(fd, buf, count, offset)` called with three arguments used to
//! build and run, so the C runtime read an undefined offset and the write was
//! lost. `tests/intrinsic_call_arity.rs` checks the IR rule on hand-built
//! modules; this gate drives the `mindc` binary end to end, so un-wiring the
//! rule from the build path fails here even when the IR unit tests stay green.
//! The correct four-argument call is the positive control: it must build and
//! print, which proves the refusals below come from the arity rule.

#![cfg(all(target_os = "linux", feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::path::PathBuf;
use std::process::{Command, Output};

const TARGET: &str = "intrinsic_call_arity_build";
const ARITY_MESSAGE: &str = "intrinsic `__mind_write` expects 4 i64 argument(s)";

/// Build `body` as `main.mind` into a fresh scratch project and return the
/// build output with the path the executable would be written to.
fn build(name: &str, body: &str) -> (Output, PathBuf) {
    let project_dir = common::scratch_dir(TARGET).join(name);
    let _ = std::fs::remove_dir_all(&project_dir);
    std::fs::create_dir_all(&project_dir).expect("create project dir");
    std::fs::write(
        project_dir.join("Mind.toml"),
        "[package]\nname = \"intrinsic_arity\"\nversion = \"0.0.1\"\n\n[build]\nentry = \"main.mind\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(project_dir.join("main.mind"), body).expect("write main.mind");

    let out = Command::new(common::mindc_bin())
        .current_dir(&project_dir)
        .env("MINDC_RUNTIME_OBJ_CACHE", "0")
        .args([
            "build",
            "--backend",
            "mlir",
            "--target",
            "cpu",
            "--emit",
            "binary",
            "--no-cache",
            "--out",
            "program",
        ])
        .output()
        .expect("spawn mindc build");
    (out, project_dir.join("program"))
}

/// Assert `body` is refused by `mindc build` with the arity message naming
/// `got`, and that no executable was written.
fn assert_refused(name: &str, body: &str, got: usize) {
    let (out, program) = build(name, body);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "{name}: mindc build accepted a call with {got} argument(s): {text}"
    );
    assert!(
        text.contains(ARITY_MESSAGE) && text.contains(&format!("got {got}")),
        "{name}: the refusal must name the arity rule and `got {got}`: {text}"
    );
    assert!(
        !program.exists(),
        "{name}: a refused build must not write {}",
        program.display()
    );
}

#[test]
fn the_correct_call_builds_and_prints() {
    let (out, program) = build(
        "correct",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    __mind_write(1, s.addr, s.len, 0 - 1);\n    return 0\n}\n",
    );
    if !common::gate::compiled(TARGET, &out) {
        return;
    }
    let run = Command::new(&program).output().expect("run the program");
    assert_eq!(run.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "ok\n",
        "the four-argument write reached stdout"
    );
}

#[test]
fn a_call_with_too_few_arguments_is_refused() {
    assert_refused(
        "short",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    __mind_write(1, s.addr, s.len);\n    return 0\n}\n",
        3,
    );
}

#[test]
fn a_call_with_too_many_arguments_is_refused() {
    assert_refused(
        "long",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    __mind_write(1, s.addr, s.len, 0 - 1, 0);\n    return 0\n}\n",
        5,
    );
}

#[test]
fn a_short_call_inside_an_if_branch_is_refused() {
    assert_refused(
        "short_in_if",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    if s.len > 0 {\n        __mind_write(1, s.addr, s.len);\n    }\n    return 0\n}\n",
        3,
    );
}

#[test]
fn a_short_call_inside_a_while_body_is_refused() {
    // The type checker does not type-check while bodies (RFC 0005 Gap 1), so
    // `mindc check` and the build's own type-check pass both miss this call;
    // the IR verifier is the layer that refuses it.
    assert_refused(
        "short_in_while",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    let mut i: i64 = 0;\n    while i < 1 {\n        __mind_write(1, s.addr, s.len);\n        i = i + 1;\n    }\n    return 0\n}\n",
        3,
    );
}

#[test]
fn a_short_call_inside_an_else_if_branch_is_refused() {
    assert_refused(
        "short_in_else_if",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    if s.len > 5 {\n        return 1\n    } else if s.len > 0 {\n        __mind_write(1, s.addr, s.len);\n    }\n    return 0\n}\n",
        3,
    );
}

#[test]
fn a_short_call_in_a_while_condition_is_refused() {
    assert_refused(
        "short_in_while_cond",
        "fn main() -> i64 {\n    let s = \"ok\\n\";\n    let mut i: i64 = 0;\n    while __mind_write(1, s.addr, s.len) > 100 {\n        i = i + 1;\n    }\n    return i\n}\n",
        3,
    );
}

#[test]
fn a_short_call_nested_as_an_argument_is_refused() {
    // The type checker reports this shape as well; the build refuses it either
    // way.
    assert_refused(
        "short_as_argument",
        "fn id(x: i64) -> i64 {\n    return x\n}\n\nfn main() -> i64 {\n    let s = \"ok\\n\";\n    let n = id(__mind_write(1, s.addr, s.len));\n    return n\n}\n",
        3,
    );
}
