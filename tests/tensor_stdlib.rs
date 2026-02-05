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

fn eval_source(src: &str) -> eval::Value {
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap()
}

#[test]
fn zeros_and_ones_create_expected_previews() {
    let src = "tensor.zeros(f32, (2,3))";
    let value = eval_source(src);
    let preview = eval::format_value_human(&value);
    assert!(preview.contains("Tensor"));
    assert!(preview.contains("(2,3)"));
    assert!(preview.contains("fill=0"));

    let src2 = "tensor.ones(f32, (4,1))";
    let value2 = eval_source(src2);
    let preview2 = eval::format_value_human(&value2);
    assert!(preview2.contains("fill=1"));
}

#[test]
fn shape_and_dtype_return_preview_values() {
    let src = "let t = tensor.ones(f32, (2,3)); tensor.shape(t)";
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
    let preview = eval::format_value_human(&value);
    assert!(preview.contains("(2,3)"));

    let src_dtype = "let t = tensor.zeros(bf16, (5,)); tensor.dtype(t)";
    let module_dtype = parser::parse(src_dtype).unwrap();
    let mut env_dtype = HashMap::new();
    let value_dtype =
        eval::eval_module_value_with_env(&module_dtype, &mut env_dtype, Some(src_dtype)).unwrap();
    let preview_dtype = eval::format_value_human(&value_dtype);
    assert_eq!(preview_dtype, "bf16");
}
