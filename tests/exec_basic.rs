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

#[cfg(feature = "cpu-exec")]
mod cpu {
    use mind::eval;
    use mind::parser;

    use std::collections::HashMap;

    #[test]
    fn add_scalar_exec() {
        let src = "let x: Tensor[f32,(2,2)] = 1; x + 2";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let result = eval::eval_module_value_with_env_mode(
            &module,
            &mut env,
            Some(src),
            eval::ExecMode::CpuExec,
        );

        // In open-core build, runtime stubs return Unsupported. With proprietary
        // runtime, the operation materializes.
        match result {
            Ok(value) => {
                let text = eval::format_value_human(&value);
                assert!(text.contains("(2,2)"));
                // Either materialized (with runtime) or fill= (preview fallback)
                assert!(
                    text.contains("materialized") || text.contains("fill="),
                    "expected materialized or preview: {text}"
                );
            }
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("Unsupported") || msg.contains("proprietary"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn matmul_exec() {
        let src = "let a: Tensor[f32,(2,2)] = 1; let b: Tensor[f32,(2,2)] = 1; tensor.matmul(a,b)";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let result = eval::eval_module_value_with_env_mode(
            &module,
            &mut env,
            Some(src),
            eval::ExecMode::CpuExec,
        );

        // In open-core build, runtime stubs return Unsupported. With proprietary
        // runtime, the operation materializes.
        match result {
            Ok(value) => {
                let text = eval::format_value_human(&value);
                assert!(text.contains("(2,2)"));
                assert!(
                    text.contains("materialized") || text.contains("fill="),
                    "expected materialized or preview: {text}"
                );
            }
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("Unsupported") || msg.contains("proprietary"),
                    "unexpected error: {msg}"
                );
            }
        }
    }
}
