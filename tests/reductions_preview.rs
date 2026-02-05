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
fn sum_all_axes_preview() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 1;
        tensor.sum(x)
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("Tensor[F32,()]"));
    assert!(s.contains("fill=6"));
}

#[test]
fn mean_keepdims_preview() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 2;
        tensor.mean(x, axes=[1], keepdims=true)
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("Tensor[F32,(2,1)]"));
    assert!(s.contains("fill=2"));
}
