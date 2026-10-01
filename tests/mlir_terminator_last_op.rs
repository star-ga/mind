// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Nothing may follow a terminator in an MLIR block.
//!
//! A `return` directly in a `while` body, or any statement written after a
//! `return` / `break` / `continue`, used to lower to MLIR with ops after the
//! terminator in the same block. mlir-opt rejects that ("'func.return' op must
//! be the last operation in the parent block", or "operation with block
//! successors must terminate its parent block"), so every such program failed
//! to build. The `while` shape needs no dead code in the source at all: the IR
//! lowering appends the loop body's unit value after the `return`.
//!
//! The MLIR emitter now stops a straight-line instruction list at its first
//! terminator. The structural tests check the emitted text block by block (the
//! 'lowering' and 'exec' tiers); the run test ('exec' tier) builds the same
//! programs through mlir-opt and clang and checks the values they compute.

#![cfg(all(
    feature = "std-surface",
    any(feature = "mlir-lowering", feature = "mlir-build")
))]

mod common;

/// Programs with code after a terminator, and the value each `f` returns.
const CASES: &[(&str, &str, i64)] = &[
    (
        "return directly in a while body",
        "pub fn f(n: i64) -> i64 {\n    let mut i: i64 = 0;\n    while i < n {\n        return i + 7;\n    }\n    return 99;\n}\n",
        7,
    ),
    (
        "return after loop-carried updates",
        "pub fn f(n: i64) -> i64 {\n    let mut i: i64 = 0;\n    let mut s: i64 = 0;\n    while i < n {\n        s = s + i;\n        i = i + 1;\n        return s + 100;\n    }\n    return s;\n}\n",
        100,
    ),
    (
        "statement after a return in a fn body",
        "pub fn f(n: i64) -> i64 {\n    return n + 1;\n    let y: i64 = 2;\n    return y;\n}\n",
        4,
    ),
    (
        "statement after a return in an if branch",
        "pub fn f(n: i64) -> i64 {\n    if n > 0 {\n        return 4;\n        let z: i64 = 3;\n    }\n    return 9;\n}\n",
        4,
    ),
    (
        "statement after a break",
        "pub fn f(n: i64) -> i64 {\n    let mut i: i64 = 0;\n    while i < n {\n        i = i + 1;\n        if i == 2 {\n            break;\n            i = 50;\n        }\n    }\n    return i;\n}\n",
        2,
    ),
];

/// Every line after a terminator must open a new block or close the function.
fn assert_terminators_are_last(name: &str, text: &str) {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    for (i, line) in lines.iter().enumerate() {
        let terminator = line.starts_with("return")
            || line.starts_with("llvm.return")
            || line.starts_with("cf.br ")
            || line.starts_with("cf.cond_br ");
        if !terminator {
            continue;
        }
        if let Some(next) = lines.get(i + 1) {
            assert!(
                next.starts_with('^') || next.starts_with('}'),
                "{name}: `{line}` is followed by `{next}` in the same block:\n{text}"
            );
        }
    }
}

fn lower(src: &str) -> String {
    let module = libmind::parser::parse(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    let ir = libmind::eval::lower::lower_to_ir(&module).expect("lower to IR");
    libmind::mlir::lower_ir_to_mlir(&ir)
        .expect("lower to MLIR")
        .text
}

#[test]
fn no_op_follows_a_terminator_in_its_block() {
    for (name, src, _) in CASES {
        assert_terminators_are_last(name, &lower(src));
    }
}

#[test]
fn a_loop_without_a_terminator_keeps_its_back_edge() {
    // Positive control: stopping at a terminator must not drop the back-edge of
    // a body that falls through.
    let text = lower(
        "pub fn f(n: i64) -> i64 {\n    let mut i: i64 = 0;\n    while i < n {\n        i = i + 1;\n    }\n    return i;\n}\n",
    );
    assert_terminators_are_last("plain loop", &text);
    assert!(
        text.contains("cf.br ^while_header_0("),
        "the loop must branch back to its header:\n{text}"
    );
}

#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
#[test]
fn programs_with_code_after_a_terminator_build_and_run() {
    use std::process::Command;

    let dir = common::scratch_dir("mlir_terminator_last_op");
    for (i, (name, src, expected)) in CASES.iter().enumerate() {
        let src_path = dir.join(format!("case{i}.mind"));
        let so = dir.join(format!("case{i}.so"));
        std::fs::write(&src_path, src).expect("write source");
        let out = Command::new(common::mindc_bin())
            .arg(&src_path)
            .arg("--emit-shared")
            .arg(&so)
            .output()
            .expect("run mindc");
        if !common::gate::compiled("mlir_terminator_last_op", &out) {
            return;
        }
        let py = format!(
            "import ctypes\n\
             f = ctypes.CDLL(r'{}').f\n\
             f.restype = ctypes.c_int64\n\
             f.argtypes = [ctypes.c_int64]\n\
             print(f(3))\n",
            so.to_string_lossy()
        );
        let run = Command::new("python3")
            .args(["-c", &py])
            .output()
            .expect("python3");
        let got = String::from_utf8_lossy(&run.stdout).trim().to_string();
        assert_eq!(
            got,
            expected.to_string(),
            "{name}: f(3) returned {got:?}; stderr: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
