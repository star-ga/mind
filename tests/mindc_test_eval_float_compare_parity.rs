// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A float comparison is a `bool` under `mindc test`, exactly as in the build.
//!
//! Every comparison (`<` `<=` `>` `>=` `==` `!=`) on `f32` and on `f64`, over
//! an ordered pair, the reversed pair and an equal pair, is read through each
//! way a program consumes a condition: `if`, `!`, `&&`, `||`, a `bool`
//! returned from a function (with and without `!`), a `bool` binding, and a
//! `while` condition. Each case compares what the program observed with the
//! truth Rust computes for the same operands, so a comparison that yields
//! `Float(0.0)` (read as true by `if`, refused by `!`) turns the case into a
//! counted miss. The program returns `checked` (every case ran and agreed) or
//! `100 + misses`; the interpreter leg and the native leg are held to the
//! same score, and the interpreter leg also runs with score + 1 and must FAIL.
//!
//! Gate: `cargo test --release --no-default-features
//!        --features mlir-build,std-surface,cross-module-imports
//!        --test mindc_test_eval_float_compare_parity`

#![cfg(feature = "std-surface")]

mod common;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const OPS: [(&str, &str, fn(f64, f64) -> bool); 6] = [
    ("lt", "<", |a, b| a < b),
    ("le", "<=", |a, b| a <= b),
    ("gt", ">", |a, b| a > b),
    ("ge", ">=", |a, b| a >= b),
    ("eq", "==", |a, b| a == b),
    ("ne", "!=", |a, b| a != b),
];

const PAIRS: [(&str, &str, f64, f64); 3] = [
    ("1.0", "0.05", 1.0, 0.05),
    ("0.05", "1.0", 0.05, 1.0),
    ("2.0", "2.0", 2.0, 2.0),
];

/// One context: the statement that sets `seen` to 1 when the condition built
/// from comparison `cmp` (and its helper-fn name `f`) holds, and whether the
/// observed truth is the negation of the comparison.
fn context_stmt(ctx: &str, cmp: &str, name: &str, n: usize) -> (String, bool) {
    let (f, g, h) = (
        format!("f_{name}"),
        format!("g_{name}"),
        format!("h_{name}"),
    );
    match ctx {
        "if" => (format!("if {cmp} {{ seen{n} = 1; }}"), false),
        "not" => (format!("if !({cmp}) {{ seen{n} = 1; }}"), true),
        "and" => (
            format!("if ({cmp}) && (a{n} == a{n}) {{ seen{n} = 1; }}"),
            false,
        ),
        "or" => (
            format!("if ({cmp}) || (a{n} != a{n}) {{ seen{n} = 1; }}"),
            false,
        ),
        "fn" => (format!("if {f}(a{n}, b{n}) {{ seen{n} = 1; }}"), false),
        "notfn" => (format!("if !{f}(a{n}, b{n}) {{ seen{n} = 1; }}"), true),
        "paren" => (format!("if ({cmp}) {{ seen{n} = 1; }}"), false),
        "fnparen" => (format!("if {g}(a{n}, b{n}) {{ seen{n} = 1; }}"), false),
        "fntail" => (format!("if {h}(a{n}, b{n}) {{ seen{n} = 1; }}"), false),
        "bind" => (
            format!("let r{n}: bool = {cmp};\n    if r{n} {{ seen{n} = 1; }}"),
            false,
        ),
        "while" => (
            format!(
                "let mut w{n}: i64 = 0;\n    while ({cmp}) && w{n} < 3 {{ w{n} = w{n} + 1; }}\n    \
                 if w{n} == 3 {{ seen{n} = 1; }}"
            ),
            false,
        ),
        _ => unreachable!("unknown context"),
    }
}

const CONTEXTS: [&str; 11] = [
    "if", "not", "and", "or", "paren", "fn", "notfn", "fnparen", "fntail", "bind", "while",
];

/// The probe program for one `(type, context)`; returns `(source, expected)`.
fn program(ty: &str, ctx: &str) -> (String, i64) {
    let mut helpers = String::new();
    for (name, sym, _) in OPS {
        helpers.push_str(&format!(
            "fn f_{name}(x: {ty}, y: {ty}) -> bool {{\n    return x {sym} y;\n}}\n\n\
             fn g_{name}(x: {ty}, y: {ty}) -> bool {{\n    return (x {sym} y);\n}}\n\n\
             fn h_{name}(x: {ty}, y: {ty}) -> bool {{\n    x {sym} y\n}}\n\n"
        ));
    }
    let mut body = String::from(
        "fn score() -> i64 {\n    let mut checked: i64 = 0;\n    let mut bad: i64 = 0;\n",
    );
    let mut n = 0usize;
    for (a, b, av, bv) in PAIRS {
        for (name, sym, truth) in OPS {
            n += 1;
            let cmp = format!("a{n} {sym} b{n}");
            let (stmt, negated) = context_stmt(ctx, &cmp, name, n);
            let want = i64::from(truth(av, bv) != negated);
            body.push_str(&format!(
                "    let a{n}: {ty} = {a};\n    let b{n}: {ty} = {b};\n    let mut seen{n}: i64 = 0;\n    \
                 {stmt}\n    if seen{n} == {want} {{ checked = checked + 1; }} else {{ bad = bad + 1; }}\n"
            ));
        }
    }
    body.push_str("    if bad > 0 {\n        return 100 + bad;\n    }\n    return checked;\n}\n");
    (format!("{helpers}{body}"), n as i64)
}

fn scratch(label: &str) -> PathBuf {
    let dir = common::scratch_dir("mindc_test_eval_float_compare_parity").join(label);
    fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn mindc_test(stem: &str, program: &str, expected: i64) -> (i32, String) {
    let path = scratch(&format!("{stem}-expect{expected}")).join(format!("{stem}.mind"));
    let src = format!(
        "{program}\n#[test]\nfn parity() {{\n    let got: i64 = score();\n    \
         assert got == {expected}, \"score\";\n}}\n"
    );
    fs::write(&path, src).expect("write test source");
    let out = Command::new(common::require_mindc())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    (out.status.code().unwrap_or(-1), text(&out))
}

fn assert_interpreter(stem: &str, program: &str, expected: i64) {
    let (code, out) = mindc_test(stem, program, expected);
    assert_eq!(
        code, 0,
        "mindc test must pass with score {expected}:\n{out}"
    );
    assert!(
        out.contains(&format!("test {stem}::parity ... ok")) && out.contains("1 passed; 0 failed"),
        "expected the parity test to run and pass:\n{out}"
    );
    let (code, out) = mindc_test(stem, program, expected + 1);
    assert_ne!(code, 0, "a wrong expected score must fail:\n{out}");
    assert!(
        out.contains(&format!("test {stem}::parity ... FAILED")),
        "expected the wrong-score control to fail its assert:\n{out}"
    );
}

fn assert_native(stem: &str, program: &str, expected: i64) {
    let dir = scratch(&format!("{stem}-native"));
    fs::create_dir_all(dir.join(".git")).expect("project boundary");
    fs::write(
        dir.join("Mind.toml"),
        format!(
            "[package]\nname = \"{stem}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("write manifest");
    fs::write(
        dir.join("main.mind"),
        format!("{program}\nfn main() -> i64 {{\n    return score();\n}}\n"),
    )
    .expect("write entry");
    let built = common::run_build_captured(&common::require_mindc(), &dir, &[]);
    if !common::gate::compiled(
        &format!("mindc_test_eval_float_compare_parity::{stem}"),
        &built,
    ) {
        return;
    }
    let artifact = common::reported_artifact(&built);
    let ran = Command::new(&artifact)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e}", artifact.display()));
    assert_eq!(
        ran.status.code(),
        Some(expected as i32),
        "the compiled {stem} program must exit with score {expected}:\n{}",
        text(&ran)
    );
}

fn run_type(ty: &str) {
    for ctx in CONTEXTS {
        let (src, expected) = program(ty, ctx);
        assert_interpreter(&format!("cmp_{ty}_{ctx}"), &src, expected);
    }
    for ctx in CONTEXTS {
        let (src, expected) = program(ty, ctx);
        assert_native(&format!("cmp_{ty}_{ctx}"), &src, expected);
    }
}

#[test]
fn f64_comparisons_are_booleans_in_every_condition_context() {
    run_type("f64");
}

#[test]
fn f32_comparisons_are_booleans_in_every_condition_context() {
    run_type("f32");
}

/// The exact shapes a downstream port reported: a false `<=` as an `if`
/// condition, and `!` over a function returning a float comparison.
#[test]
fn reported_shapes_take_the_right_branch() {
    let src = "fn le(a: f32, b: f32) -> bool {\n    return a <= b;\n}\n\n\
               fn score() -> i64 {\n    let mut code: i64 = 0;\n    \
               if 1.0 <= 0.05 {\n        code = code + 100;\n    } else {\n        code = code + 1;\n    }\n    \
               if !le(1.0, 0.05) {\n        code = code + 2;\n    }\n    return code;\n}\n";
    assert_interpreter("reported_shapes", src, 3);
    assert_native("reported_shapes", src, 3);
}
