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

use libmind::parser;
use libmind::type_checker;
use libmind::types::DType;
use libmind::types::ShapeDim;
use libmind::types::TensorType;
use libmind::types::ValueType;
#[test]
fn broadcast_with_symbols_equal_symbols_ok() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(1)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(diags.is_empty(), "{:?}", diags);
}

#[test]
fn broadcast_with_symbols_mismatch_fails() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("C"), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(!diags.is_empty(), "expected symbol mismatch");
}
