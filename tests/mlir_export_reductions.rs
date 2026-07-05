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

// Static f32/f64 reductions under the element cap take the STRICT pinned
// canonical-order fold (a fixed left-to-right `arith.addf` chain rebuilt with
// `tensor.from_elements`, NO `tensor.reduce` / `vector.reduction` / fastmath),
// so the result is byte-identical across substrates and run-to-run. The
// tree-shaped `tensor.reduce` tier is retained only for dynamic / over-cap /
// non-float reductions (see `over_cap_reduction_uses_treeshaped_reduce`).
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

    // Pinned fold: unrolled scalar adds rebuilt with `tensor.from_elements`,
    // never a reassociable `tensor.reduce`.
    assert!(
        !mlir.contains("tensor.reduce"),
        "static f32 reduction must use the pinned fold, not tensor.reduce, in {mlir}"
    );
    assert!(
        mlir.contains("tensor.from_elements"),
        "expected tensor.from_elements (pinned fold) in {mlir}"
    );
    assert!(
        mlir.contains("tensor.extract"),
        "expected tensor.extract in {mlir}"
    );
    assert!(mlir.contains("arith.addf"), "expected arith.addf in {mlir}");
    assert!(
        mlir.contains("arith.divf"),
        "expected arith.divf (mean) in {mlir}"
    );
}

/// Regression (Phase B.2 `.sum()`): empty axes must reduce over ALL axes to a
/// scalar. Under the pinned fold the whole tensor collapses to a single
/// `tensor.from_elements` of `tensor<f32>` (rank-0), NOT the full `2x3` shape.
#[test]
fn reduce_all_axes_when_axes_empty_yields_scalar() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 1;
        tensor.sum(x, axes=[], keepdims=false)
    "#;
    let module = parser::parse(src).expect("parse empty-axes reduction");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(
        !mlir.contains("tensor.reduce"),
        "static f32 reduction must use the pinned fold, not tensor.reduce, in {mlir}"
    );
    assert!(
        mlir.contains("tensor.from_elements") && mlir.contains(": tensor<f32>"),
        "empty axes must fold to a scalar tensor<f32> via from_elements; got:\n{mlir}"
    );
    assert!(
        !mlir.contains("from_elements") || !mlir.contains("from_elements %0 : tensor<2x3xf32>"),
        "pinned scalar result must be tensor<f32>, not the unreduced 2x3 shape; got:\n{mlir}"
    );
}

/// Tree-tier coverage: a reduction whose source exceeds the pinned-fold unroll
/// cap (4096 elements) must FALL BACK to the tree-shaped `tensor.reduce`
/// (integer/associative-float approximate tier), never unroll into thousands
/// of scalar adds.
#[test]
fn over_cap_reduction_uses_treeshaped_reduce() {
    let src = r#"
        let x: Tensor[f32,(100,100)] = 1;
        tensor.sum(x, axes=[], keepdims=false)
    "#;
    let module = parser::parse(src).expect("parse over-cap reduction");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(
        mlir.contains("tensor.reduce"),
        "over-cap (10000-elem) reduction must use tree-shaped tensor.reduce, got:\n{}",
        &mlir[..mlir.len().min(800)]
    );
    assert!(
        !mlir.contains("tensor.from_elements"),
        "over-cap reduction must not unroll into a pinned fold; got tree tier expected"
    );
}
