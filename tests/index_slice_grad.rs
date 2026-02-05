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
fn grad_through_slice_shapes_ok() {
    let src = r#"
        let X: Tensor[f32,(3,6)] = 0;
        let y = tensor.sum(tensor.slice(X, axis=1, start=1, end=4));
        grad(y, wrt=[X])
    "#;
    let m = libmind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = libmind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = libmind::eval::format_value_human(&v);
    assert!(s.contains("X: Tensor[F32,(3,6)]"));
}
