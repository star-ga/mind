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

use mind::eval;
use mind::parser;
#[test]
fn grad_sum_matmul_is_ones() {
    let src = r#"
        let A: Tensor[f32,(2,3)] = 0;
        let B: Tensor[f32,(3,4)] = 0;
        grad(tensor.sum(tensor.matmul(A,B)), wrt=[A,B])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("A: Tensor[F32,(2,3)]"));
    assert!(s.contains("B: Tensor[F32,(3,4)]"));
    assert!(s.contains("fill=1"));
}
