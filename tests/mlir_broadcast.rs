#![cfg(feature = "mlir-lowering")]
// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0012 §4.2 broadcast lowering on the canonical (`pipeline::lower_to_mlir`)
//! path — the emitter `mindc` and the criterion `compiler` bench use. The
//! front-end already validates the broadcast (`shapes::broadcast_shapes`); these
//! tests pin that the backend honours it instead of demanding matching shapes.

use libmind::{CompileOptions, compile_source, lower_to_mlir};

fn lower(src: &str) -> String {
    let products = compile_source(src, &CompileOptions::default()).expect("compilation failed");
    #[cfg(feature = "autodiff")]
    let products = lower_to_mlir(&products.ir, None).expect("MLIR lowering failed");
    #[cfg(not(feature = "autodiff"))]
    let products = lower_to_mlir(&products.ir).expect("MLIR lowering failed");
    products.primal_mlir
}

/// Bias-add MLP fragment — the exact shape the criterion `medium_mlp` bench hit.
/// `(128,128) + (128)` broadcasts the bias along rows; the rhs operand reads only
/// the trailing iteration dim.
#[test]
fn bias_add_broadcasts_via_linalg_generic() {
    let mlir = lower(
        r#"
        let input: Tensor[f32,(128,256)] = 0;
        let weight: Tensor[f32,(256,128)] = 1;
        let bias: Tensor[f32,(128)] = 0;
        let matmul_out = tensor.matmul(input, weight);
        let biased = matmul_out + bias;
        tensor.relu(biased)
    "#,
    );
    eprintln!("=== bias-add MLIR ===\n{mlir}\n=====================");
    assert!(
        mlir.contains("linalg.generic"),
        "expected linalg.generic for broadcast in:\n{mlir}"
    );
    assert!(
        mlir.contains("arith.addf"),
        "expected arith.addf in broadcast body:\n{mlir}"
    );
    assert!(
        mlir.contains("linalg.yield"),
        "expected linalg.yield in broadcast body:\n{mlir}"
    );
    assert!(
        mlir.contains("affine_map<(d0, d1) -> (d1)>"),
        "expected rhs bias broadcast map `affine_map<(d0, d1) -> (d1)>` in:\n{mlir}"
    );
}

/// Multi-dimensional size-1 stretch on *both* operands:
/// `(4,1,3) .* (1,5,3) -> (4,5,3)` (mind-spec shapes.md broadcasting example 4).
/// Each stretched axis maps to the constant `0`.
#[test]
fn multidim_broadcast_emits_const_zero_maps() {
    let mlir = lower(
        r#"
        let a: Tensor[f32,(4,1,3)] = 0;
        let b: Tensor[f32,(1,5,3)] = 1;
        a * b
    "#,
    );
    assert!(
        mlir.contains("linalg.generic"),
        "expected linalg.generic in:\n{mlir}"
    );
    assert!(
        mlir.contains("(d0, 0, d2)"),
        "expected lhs map `(d0, 0, d2)` in:\n{mlir}"
    );
    assert!(
        mlir.contains("(0, d1, d2)"),
        "expected rhs map `(0, d1, d2)` in:\n{mlir}"
    );
    assert!(
        mlir.contains("tensor<4x5x3xf32>"),
        "expected broadcast result type tensor<4x5x3xf32> in:\n{mlir}"
    );
}

/// Regression guard: equal shapes must keep the original single-line `arith`
/// emit (byte-identity for the self-host bootstrap) — never a `linalg.generic`.
#[test]
fn equal_shape_add_stays_plain_arith() {
    let mlir = lower(
        r#"
        let a: Tensor[f32,(8,8)] = 1;
        let b: Tensor[f32,(8,8)] = 1;
        a + b
    "#,
    );
    assert!(
        mlir.contains("arith.addf"),
        "expected arith.addf in:\n{mlir}"
    );
    assert!(
        !mlir.contains("linalg.generic"),
        "equal-shape add must NOT use linalg.generic:\n{mlir}"
    );
}

/// The bench `large_network` fixture (3 bias-add layers feeding `tensor.relu`
/// into the next `matmul`). Exercises relu lowering + broadcast end-to-end on
/// the canonical pipeline emitter.
#[test]
fn large_network_lowers() {
    let mlir = lower(
        r#"
        let input: Tensor[f32,(128,784)] = 0;
        let w1: Tensor[f32,(784,512)] = 1;
        let b1: Tensor[f32,(512)] = 0;
        let w2: Tensor[f32,(512,256)] = 1;
        let b2: Tensor[f32,(256)] = 0;
        let w3: Tensor[f32,(256,10)] = 1;
        let b3: Tensor[f32,(10)] = 0;
        let matmul1 = tensor.matmul(input, w1);
        let h1 = tensor.relu(matmul1 + b1);
        let matmul2 = tensor.matmul(h1, w2);
        let h2 = tensor.relu(matmul2 + b2);
        let matmul3 = tensor.matmul(h2, w3);
        matmul3 + b3
    "#,
    );
    eprintln!(
        "=== large_network MLIR ===
{mlir}
=========================="
    );
    assert!(
        mlir.contains("linalg.generic"),
        "expected broadcast/relu generic in:
{mlir}"
    );
    assert!(
        mlir.contains("arith.maximumf"),
        "expected relu arith.maximumf in:
{mlir}"
    );
    assert!(
        mlir.contains("linalg.matmul"),
        "expected matmul in:
{mlir}"
    );
}

/// `tensor.relu` lowers to a shape-preserving `linalg.generic` whose body is a
/// single `arith.maximumf` against `0.0` (RFC 0012 elementwise activation).
#[test]
fn relu_lowers_to_maximumf_generic() {
    let mlir = lower(
        r#"
        let a: Tensor[f32,(4,8)] = 1;
        tensor.relu(a)
    "#,
    );
    assert!(
        mlir.contains("linalg.generic"),
        "expected linalg.generic in:\n{mlir}"
    );
    assert!(
        mlir.contains("arith.maximumf"),
        "expected arith.maximumf in:\n{mlir}"
    );
    assert!(
        mlir.contains("arith.constant 0.0"),
        "expected zero constant in:\n{mlir}"
    );
}
