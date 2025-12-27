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

#[test]
fn gather_inserts_idx_shape() {
    let src = r#"
        let x: Tensor[f32,(3,4)] = 5;
        let idx: Tensor[i32,(2)] = 0;
        tensor.gather(x, axis=1, idx)
    "#;
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("(3,2)"));
    assert!(s.contains("fill=5"));
}
