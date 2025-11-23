// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

use mind::parser;

#[test]
fn parses_scalar_annotation() {
    let m = parser::parse("let n: i32 = 3; n + 1").unwrap();
    assert_eq!(m.items.len(), 2);
}

#[test]
fn parses_tensor_annotation() {
    let m = parser::parse("let x: Tensor[f32,(B,3,224,224)] = 0;").unwrap();
    assert_eq!(m.items.len(), 1);
}
