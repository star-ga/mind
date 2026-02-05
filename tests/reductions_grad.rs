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
fn grad_sum_is_ones() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        grad(tensor.sum(x), wrt=[x])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("x: Tensor[F32,(2,3)]"));
    assert!(s.contains("fill=1"));
}

#[test]
fn grad_mean_is_1_over_n() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        grad(tensor.mean(x), wrt=[x])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("fill=0.166666") || s.contains("fill=0.166667"));
}
