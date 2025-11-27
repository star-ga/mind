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

use mind::eval;
use mind::parser;

#[test]
fn mlir_export_basic() {
    let src = "1 + 2 * 3";
    let m = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&m);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(mlir.contains("func.func @main"));
    assert!(mlir.contains("arith.addi"));
    assert!(mlir.contains("return"));
}

#[test]
fn mlir_export_tensor_const() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let m = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&m);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(mlir.contains("tensor.empty"));
    assert!(mlir.contains("linalg.fill"));
}
