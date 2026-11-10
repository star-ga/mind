// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The time one `std.https` socket read or write may block is the smaller of
//! the time left to the request deadline and `https_io_timeout_s()`.
//!
//! `https_io_budget_ms` is what `https_arm` sets as the socket timeout before
//! every read and write of the session, so a single read cannot run past the
//! deadline however slowly the peer sends. A built executable checks it and
//! exits with the number of failed checks; a second run flips one expected
//! value and must fail exactly that check, so the program cannot pass by
//! checking nothing.

#![cfg(all(
    target_os = "linux",
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

#[path = "support/https_loopback_harness.rs"]
#[allow(dead_code)]
mod harness;

use harness::{IMPORTS, build_program};
use std::process::Command;

/// `(remaining_ms, expected budget)`; the io limit is 30 s.
const CASES: &[(i64, i64)] = &[
    (-5, 0),
    (0, 0),
    (1, 1),
    (29_999, 29_999),
    (30_000, 30_000),
    (30_001, 30_000),
    (120_000, 30_000),
];

fn program(flip_last: bool) -> String {
    let mut body = String::from("fn main() -> i64 {\n    let mut f: i64 = 0;\n");
    for (i, (remaining, expected)) in CASES.iter().enumerate() {
        let want = if flip_last && i + 1 == CASES.len() {
            expected + 1
        } else {
            *expected
        };
        body.push_str(&format!(
            "    if https_io_budget_ms({remaining}) != {want} {{\n        f = f + 1;\n    }}\n"
        ));
    }
    body.push_str("    f\n}\n");
    let mut src: String = IMPORTS
        .iter()
        .map(|m| format!("import std.{m};\n"))
        .collect();
    src.push('\n');
    src.push_str(&body);
    src
}

fn exit_code(case: &str, main: &str) -> Option<i32> {
    let exe = build_program(case, main)?;
    Some(
        Command::new(exe)
            .status()
            .expect("run the budget program")
            .code()
            .expect("exit code"),
    )
}

#[test]
fn the_socket_budget_is_bounded_by_the_time_left() {
    let Some(code) = exit_code("deadline_budget", &program(false)) else {
        return;
    };
    assert_eq!(code, 0, "every budget case must hold");
    let mutated = exit_code("deadline_budget_mutated", &program(true)).expect("built once");
    assert_eq!(
        mutated, 1,
        "the flipped expectation is the one failed check"
    );
}
