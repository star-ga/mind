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
fn mlir_export_reductions_cover_sum_and_mean() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 1;
        let s = tensor.sum(x, axes=[1], keepdims=false);
        tensor.mean(s, axes=[0], keepdims=false)
    "#;
    let module = parser::parse(src).expect("parse reductions module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");

    assert!(
        mlir.contains("tensor.reduce"),
        "expected tensor.reduce in {mlir}"
    );
    assert!(mlir.contains("arith.addf"), "expected arith.addf in {mlir}");
    assert!(mlir.contains("arith.divf"), "expected arith.divf in {mlir}");
}
