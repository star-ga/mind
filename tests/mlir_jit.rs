#![cfg(feature = "mlir-jit")]

// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

use std::collections::HashMap;

use mind::eval;

use mind::parser;
#[test]
fn jit_mode_falls_back_cleanly() {
    let src = "let x: Tensor[f32,(1,1)] = 0; x + 1";
    let module = parser::parse(src).expect("parser failure");
    let mut env = HashMap::new();
    let result = eval::eval_module_value_with_env_mode(
        &module,
        &mut env,
        Some(src),
        eval::ExecMode::MlirJitCpu,
    );
    assert!(result.is_ok());
}
