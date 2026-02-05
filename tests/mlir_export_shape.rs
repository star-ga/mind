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
fn mlir_export_covers_shape_ops() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        let reshaped = tensor.reshape(x, (3,2));
        let expanded = tensor.expand_dims(reshaped, axis=1);
        let squeezed = tensor.squeeze(expanded, axes=[1]);
        tensor.transpose(squeezed, axes=[1,0])
    "#;
    let module = parser::parse(src).expect("parse shape module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");

    assert!(
        mlir.contains("tensor.reshape"),
        "expected tensor.reshape in {mlir}"
    );
    assert!(
        mlir.contains("linalg.transpose"),
        "expected linalg.transpose in {mlir}"
    );
}
