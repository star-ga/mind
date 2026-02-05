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

use libmind::eval;
use libmind::parser;

#[test]
fn scalar_annotation_matches() {
    let src = "let n: i32 = 3; n + 1";
    let m = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let out = eval::eval_module_with_env(&m, &mut env, Some(src)).unwrap();
    assert_eq!(out, 4);
}

#[test]
fn tensor_ann_blocks_scalar_ops() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let m = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let err = eval::eval_module_with_env(&m, &mut env, Some(src)).unwrap_err();
    assert!(matches!(err, eval::EvalError::Unsupported), "got: {err}");
}
