// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc build` refuses an intrinsic call with the wrong argument count.
//!
//! `__mind_write(fd, buf, count, offset)` called with three arguments used to
//! build and run, so the C runtime read an undefined offset and the write was
//! lost. Two layers refuse it on the build path: the type checker (E2005, in
//! every body shape) and, behind it, the IR verifier (E3001), which
//! `tests/intrinsic_call_arity.rs` pins on hand-built modules. The build tests
//! below pass while either layer refuses, so they prove that no wrong-arity
//! call builds, not which layer refused it; the `mindc check` tests at the end
//! pin the type checker alone. The correct four-argument call is the positive
//! control: it must build and print, which proves the refusals below come from
//! the arity rule.

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
    // `infer_expr` does not type a while body (RFC 0005 Gap 1), but the
    // intrinsic-arity pass walks it, so the build's type-check pass refuses
    // this call (E2005) before the IR verifier sees it.
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
    // The type checker reports this shape in every resolution of `id` (see
    // the `mindc check` tests below); the build refuses it either way.
    assert_refused(
        "short_as_argument",
        "fn id(x: i64) -> i64 {\n    return x\n}\n\nfn main() -> i64 {\n    let s = \"ok\\n\";\n    let n = id(__mind_write(1, s.addr, s.len));\n    return n\n}\n",
        3,
    );
}

// ── `mindc check` reports E2005 in every body shape ─────────────────────
//
// `mindc check` runs the type checker alone. That checker used to report the
// rule only where its `infer_expr` walk reached the call, so `while` bodies and
// conditions (RFC 0005 Gap 1), `else if` branches, a non-final statement of a
// body, and a call nested as an argument of a call whose arguments it never
// re-infers all passed `check` and were refused only at build time, by the IR
// verifier (E3001). These drive the `check` subcommand over every shape the
// build tests above refuse and every shape `tests/intrinsic_call_arity.rs`
// refuses in IR, so `check` and `build` agree. Closures and impl methods reach
// the checker as the functions they desugar to, exactly as in the build.

const SHORT_CALL: &str = "__mind_write(1, 0, 0)";
const LONG_CALL: &str = "__mind_write(1, 0, 0, 0 - 1, 0)";
const DECLARED_CALL: &str = "__mind_write(1, 0, 0, 0 - 1)";

/// Every body shape, as a program with `CALL` where the intrinsic call goes.
const CHECK_SHAPES: &[(&str, &str)] = &[
    (
        "as a statement in a fn body",
        "fn main() -> i64 {\n    CALL;\n    return 0\n}\n",
    ),
    (
        "in a let initializer",
        "fn main() -> i64 {\n    let n = CALL;\n    return n\n}\n",
    ),
    (
        "in a return expression",
        "fn main() -> i64 {\n    return CALL\n}\n",
    ),
    (
        "in an if condition",
        "fn main() -> i64 {\n    if CALL > 0 {\n        return 1\n    }\n    return 0\n}\n",
    ),
    (
        "in an if then branch",
        "fn main() -> i64 {\n    if 1 > 0 {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "before another statement in an if branch",
        "fn main() -> i64 {\n    let mut t: i64 = 0;\n    if 1 > 0 {\n        CALL;\n        t = 1;\n    }\n    return t\n}\n",
    ),
    (
        "before another statement in a for body",
        "fn main() -> i64 {\n    let mut t: i64 = 0;\n    for i in 0..2 {\n        CALL;\n        t = t + i;\n    }\n    return t\n}\n",
    ),
    (
        "in an else branch",
        "fn main() -> i64 {\n    if 1 > 5 {\n        return 1\n    } else {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "in an else if branch",
        "fn main() -> i64 {\n    if 1 > 5 {\n        return 1\n    } else if 1 > 0 {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "in a while body",
        "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 1 {\n        CALL;\n        i = i + 1;\n    }\n    return 0\n}\n",
    ),
    (
        "in a while condition",
        "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while CALL > 100 {\n        i = i + 1;\n    }\n    return i\n}\n",
    ),
    (
        "nested as an argument of a same-file call",
        "fn id(x: i64) -> i64 {\n    return x\n}\n\nfn main() -> i64 {\n    let n = id(CALL);\n    return n\n}\n",
    ),
    // `export { main }` keeps `id` off the project module table, so the call
    // resolves as an intra-module one: the path whose arguments the checker
    // never re-infers (`check_intra_fn_call`).
    (
        "nested as an argument of an intra-module call",
        "export { main };\n\nfn id(x: i64) -> i64 {\n    return x\n}\n\nfn main() -> i64 {\n    let n = id(CALL);\n    return n\n}\n",
    ),
    (
        "in a for body",
        "fn main() -> i64 {\n    for i in 0..2 {\n        CALL;\n    }\n    return 0\n}\n",
    ),
    (
        "in a match arm",
        "fn pick(n: i64) -> i64 {\n    match n { 0 => CALL, _ => 0 }\n}\n\nfn main() -> i64 {\n    return pick(1)\n}\n",
    ),
    (
        "in a region",
        "fn main() -> i64 {\n    let r = region {\n        let w = CALL\n        0\n    }\n    return r\n}\n",
    ),
    (
        "in a while inside an if inside a region",
        "fn main() -> i64 {\n    let r = region {\n        let mut i: i64 = 0\n        if i < 1 {\n            while i < 1 {\n                CALL;\n                i = i + 1\n            }\n        }\n        i\n    }\n    return r\n}\n",
    ),
    (
        "in a closure body",
        "fn main() -> i64 {\n    let k: i64 = 1\n    let f = |k; x: i64| -> i64 {\n        let w = CALL\n        x + k\n    }\n    return f(5)\n}\n",
    ),
    (
        "in an impl method body",
        "struct Foo { x: i64 }\n\ntrait Speak {\n    fn val(self) -> i64\n}\n\nimpl Speak for Foo {\n    fn val(self) -> i64 {\n        let w = CALL\n        return self.x\n    }\n}\n\nfn main() -> i64 {\n    let foo = Foo { x: 1 }\n    return foo.val()\n}\n",
    ),
];

/// `mindc check` on `body` written as `main.mind` in a fresh scratch dir.
fn check(name: &str, body: &str) -> Output {
    let dir = common::scratch_dir(TARGET).join("check").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create check dir");
    std::fs::write(dir.join("main.mind"), body).expect("write main.mind");
    Command::new(common::mindc_bin())
        .arg("check")
        .arg("--no-fmt")
        .arg(dir.join("main.mind"))
        .output()
        .expect("spawn mindc check")
}

fn check_text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn check_reports_e2005_in_every_body_shape() {
    let mut missed = Vec::new();
    for (call, got) in [(SHORT_CALL, 3), (LONG_CALL, 5)] {
        for (i, (where_, shape)) in CHECK_SHAPES.iter().enumerate() {
            let out = check(&format!("got{got}_{i}"), &shape.replace("CALL", call));
            let text = check_text(&out);
            if out.status.success()
                || !text.contains(ARITY_MESSAGE)
                || !text.contains(&format!("got {got}"))
                || !text.contains("E2005")
            {
                missed.push(format!(
                    "{got} arguments {where_} (exit {:?}): {text}",
                    out.status.code()
                ));
            }
        }
    }
    assert!(
        missed.is_empty(),
        "mindc check must refuse each wrong-arity call with the E2005 intrinsic arity rule:\n{}",
        missed.join("\n")
    );
}

#[test]
fn check_accepts_the_declared_arity_in_every_body_shape() {
    // Positive control: the same programs with the declared four-argument call
    // pass `mindc check`, so each refusal above comes from the argument count.
    let mut refused = Vec::new();
    for (i, (where_, shape)) in CHECK_SHAPES.iter().enumerate() {
        let out = check(
            &format!("declared_{i}"),
            &shape.replace("CALL", DECLARED_CALL),
        );
        if !out.status.success() {
            refused.push(format!("{where_}: {}", check_text(&out)));
        }
    }
    assert!(
        refused.is_empty(),
        "the declared arity must pass mindc check:\n{}",
        refused.join("\n")
    );
}
