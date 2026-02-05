#![cfg(feature = "mlir-jit")]

// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::collections::HashMap;

use libmind::eval;

use libmind::parser;
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
