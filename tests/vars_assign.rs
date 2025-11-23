// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

use mind::eval;
use mind::parser;

#[test]
fn let_and_use_variable() {
    let m = parser::parse("let x = 2; x * 3 + 1").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 7);
}

#[test]
fn assign_updates_value() {
    let m = parser::parse("let x = 1; x = x + 4; x * 2").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 10);
}
