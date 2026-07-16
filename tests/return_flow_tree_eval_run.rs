// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Early-`return` control-flow propagation in the TREE (const-eval) evaluator
//! — Salov C3 / task #179 differential gate.
//!
//! Before the fix, `eval::eval_value_expr_mode` evaluated a function body
//! straight-line: a `return X` mid-body produced `X` but did NOT stop
//! evaluation, so a later statement's value clobbered it (an early return in an
//! `if` was ignored; a return in a loop did not stop the loop OR the fn). The
//! compiled/native path handles `return` correctly, so the interpreter oracle
//! silently DISAGREED with the artifact — this is exactly what `#[bimap]`'s
//! generated `if … { return … }` dispatch chains hit.
//!
//! This test is a DIFFERENTIAL: for each probe it compiles the module to a
//! `.so` (native truth) and evaluates the SAME call through the tree walker,
//! asserting `tree-eval == native` on four shapes:
//!   [A] return inside an `if` branch (dispatch chain)
//!   [B] return inside a `while` loop (must stop the loop AND the fn)
//!   [C] return as the last statement (baseline — must not regress)
//!   [D] return BEFORE a side-effecting statement (short-circuit)
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test return_flow_tree_eval_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::collections::HashMap;
use std::process::Command;

use libmind::eval;
use libmind::parser;

const SRC: &str = r#"
// [A] return inside an `if` branch — a bimap-style dispatch chain. A regression
// (no early exit) falls through to the last `return 1`, so classify(-5) yields
// 1 instead of -1 and classify(0) yields 1 instead of 0.
pub fn classify(x: i64) -> i64 {
    if x < 0 {
        return -1
    }
    if x == 0 {
        return 0
    }
    return 1
}

// [B] return inside a `while` loop — must stop the loop AND the function. A
// regression runs the loop to its cap and returns the trailing `-1`.
pub fn first_ge(limit: i64) -> i64 {
    let mut i: i64 = 0
    while i < 1000 {
        if i * i >= limit {
            return i
        }
        i = i + 1
    }
    return -1
}

// [C] return as the last statement — the normal (implicit-return) path. Must
// stay correct under the fix (no regression on straight-line bodies).
pub fn double(x: i64) -> i64 {
    let y: i64 = x * 2
    return y
}

// [D] return BEFORE a side-effecting statement — the early exit must skip the
// later `acc = acc + 100`. A regression lets that mutation run and returns 120
// for guarded(20) instead of 20.
pub fn guarded(x: i64) -> i64 {
    let mut acc: i64 = x
    if x > 10 {
        return acc
    }
    acc = acc + 100
    return acc
}
"#;

/// Tree-walk const-eval of `<fn>(<arg>)` by appending a top-level binding and
/// reading the module's last value (Preview / const-eval mode).
fn tree_eval_call(func: &str, arg: i64) -> i64 {
    let src = format!("{SRC}\nlet __probe: i64 = {func}({arg})\n");
    let module = parser::parse(&src).expect("probe parses");
    let mut env: HashMap<String, i64> = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(&src))
        .unwrap_or_else(|e| panic!("tree-eval of {func}({arg}) failed: {e}"));
    match value {
        eval::Value::Int(n) => n,
        other => panic!("tree-eval of {func}({arg}) gave non-int {other:?}"),
    }
}

#[test]
fn early_return_tree_eval_matches_native() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("return-flow-tree-eval-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src_path = dir.join("mind_return_flow_tree_eval_run.mind");
    let so = dir.join("mind_return_flow_tree_eval_run.so");
    std::fs::write(&src_path, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([
            src_path.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("return-flow-tree-eval-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("return-flow-tree-eval-run: mindc --emit-shared failed:\n{stderr}");
    }

    // (function, argument) probes spanning all four shapes and both branch sides.
    let cases: &[(&str, i64)] = &[
        ("classify", -5),
        ("classify", 0),
        ("classify", 7),
        ("first_ge", 50),
        ("first_ge", 0),
        ("first_ge", 1),
        ("double", 21),
        ("guarded", 20),
        ("guarded", 5),
    ];

    // Native truth via ctypes, one call per case; compare to the tree-eval value.
    let mut lines = String::new();
    for (func, arg) in cases {
        let tree = tree_eval_call(func, *arg);
        lines.push_str(&format!(
            "fn = lib.{func}; fn.restype = ctypes.c_int64; fn.argtypes = [ctypes.c_int64]\n\
             r = fn({arg})\n\
             assert r == {tree}, '{func}({arg}): native=' + str(r) + ' tree={tree}'\n\
             print('{func}({arg}) = ' + str(r) + '  (tree-eval == native)')\n",
        ));
    }
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\n{lines}",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    // Surface the verbatim differential lines on the console for the gate report.
    print!("{}", String::from_utf8_lossy(&out.stdout));
    assert!(
        out.status.success(),
        "return-flow differential FAILED (tree-eval != native):\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
