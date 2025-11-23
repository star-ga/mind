// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

#[cfg(feature = "mlir")]
#[test]
fn lower_placeholder_contains_mlir_module() {
    let mlir = mind::ir::lower_placeholder("x 123");
    assert!(mlir.contains("mlir.module"));
    assert!(mlir.contains("x 123"));
}
