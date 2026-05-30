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
fn mlir_export_emits_linalg_dot_and_matmul() {
    let src = r#"
        let a: Tensor[f32,(2)] = 1;
        let b: Tensor[f32,(2)] = 1;
        let dot_val = tensor.dot(a, b);
        let m: Tensor[f32,(2,3)] = 1;
        let n: Tensor[f32,(3,4)] = 1;
        let mat = tensor.matmul(m, n);
        mat
    "#;
    let module = parser::parse(src).expect("parse linalg module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");

    assert!(mlir.contains("linalg.dot"), "expected linalg.dot in {mlir}");
    assert!(
        mlir.contains("linalg.matmul"),
        "expected linalg.matmul in {mlir}"
    );
}

/// Eval-path emitter must lower `tensor.relu` to a `linalg.generic` with
/// `arith.maximumf` — not silently drop it (audit H2). Mirrors the canonical
/// pipeline emitter.
#[test]
fn mlir_export_emits_relu_maximumf() {
    let src = r#"
        let a: Tensor[f32,(4,8)] = 1;
        tensor.relu(a)
    "#;
    let module = parser::parse(src).expect("parse relu module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");
    eprintln!("=== eval relu MLIR ===\n{mlir}\n======================");
    assert!(
        mlir.contains("linalg.generic"),
        "expected linalg.generic in {mlir}"
    );
    assert!(
        mlir.contains("arith.maximumf"),
        "expected arith.maximumf in {mlir}"
    );
}
