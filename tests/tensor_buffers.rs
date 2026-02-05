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

#[cfg(feature = "cpu-buffers")]
#[test]
fn materializes_small_filled_tensor() {
    let src = "let x: Tensor[f32,(2,3)] = 1; tensor.materialize(x)";
    let m = libmind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = libmind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = libmind::eval::format_value_human(&v);
    assert!(s.contains("materialized"));
    assert!(s.contains("(2,3)"));
}

#[cfg(feature = "cpu-buffers")]
#[test]
fn tensor_sample_uses_materialized_data() {
    let src = "let x: Tensor[f32,(2,3)] = 1; tensor.sample(tensor.materialize(x), 4)";
    let m = libmind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = libmind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = libmind::eval::format_value_human(&v);
    assert!(s.contains("materialized"));
    assert!(s.contains("(4)"));
}

#[cfg(not(feature = "cpu-buffers"))]
#[test]
fn preview_only_without_buffers() {
    let src = "let x: Tensor[f32,(2,3)] = 1; x";
    let m = libmind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = libmind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = libmind::eval::format_value_human(&v);
    assert!(s.contains("fill=1"));
    assert!(!s.contains("materialized"));
}
