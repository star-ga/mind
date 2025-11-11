#![cfg(feature = "mlir-gpu")]

use std::collections::HashMap;

use mind::eval;

use mind::parser;
#[test]
fn gpu_mode_falls_back_cleanly() {
    let src = "let x: Tensor[f32,(1,1)] = 0; x + 1";
    let module = parser::parse(src).expect("parser failure");
    let mut env = HashMap::new();
    let result = eval::eval_module_value_with_env_mode(
        &module,
        &mut env,
        Some(src),
        eval::ExecMode::MlirGpu {
            backend: eval::GpuBackend::Cuda,
            blocks: (1, 1, 1),
            threads: (1, 1, 1),
        },
    );
    assert!(result.is_ok());
}
