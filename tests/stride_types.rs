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
fn zero_step_is_error() {
    let src = r#" let x: Tensor[f32,(2,10)] = 0; tensor.slice_stride(x, axis=1, start=0, end=10, step=0) "#;
    let m = libmind::parser::parse(src).unwrap();
    let diags = libmind::type_checker::check_module_types(&m, src, &std::collections::HashMap::new());
    assert!(!diags.is_empty());
}
