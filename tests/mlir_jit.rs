#![cfg(feature = "mlir-jit")]

use mind::{eval, parser};
use std::collections::HashMap;

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
