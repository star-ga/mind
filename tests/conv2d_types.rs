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
fn conv2d_channel_mismatch_errors() {
    let src = r#"
        let x: Tensor[f32,(1,2,2,2)] = 0;
        let w: Tensor[f32,(1,1,3,1)] = 0;
        tensor.conv2d(x, w)
    "#;
    let module = libmind::parser::parse(src).unwrap();
    let diags = libmind::type_checker::check_module_types(&module, src, &HashMap::new());
    assert!(!diags.is_empty());
}

#[test]
fn conv2d_same_padding_symbolic_shapes() {
    let src = r#"
        let x: Tensor[f32,(N,H,W,C)] = 0;
        let w: Tensor[f32,(3,3,C,F)] = 0;
        tensor.conv2d(x, w, stride_h=2, stride_w=2, padding="same")
    "#;
    let module = libmind::parser::parse(src).unwrap();
    let diags = libmind::type_checker::check_module_types(&module, src, &HashMap::new());
    assert!(diags.is_empty());
}
use std::collections::HashMap;
