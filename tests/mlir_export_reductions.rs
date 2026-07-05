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
// tree-shaped `tensor.reduce` tier is retained only for associative integer
// reductions, or — for float — the explicit non-deterministic fast opt-in (see
// `over_cap_float_reduction_gates_on_fast_optin`).
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

/// Determinism gate (determinism.md §4, todo #75): an f32 reduction whose source
/// exceeds the pinned-fold unroll cap (4096 elements) can only lower to a
/// tree-shaped `tensor.reduce`, which REORDERS float additions (~1e-4 tolerance,
/// not bit-identical). By DEFAULT the interp backend must FAIL LOUD on this —
/// matching the native lowering, which refuses the same over-cap/dynamic float
/// case with `MlirLowerError::UnsupportedOp`. Only the explicit
/// `MIND_FLOAT_REDUCE_FAST=1` opt-in unlocks the non-deterministic tree tier.
///
/// Both behaviours live in ONE test so the env-var mutation stays serial (the
/// other tests in this file use small static tensors that take the pinned fold
/// and never read the env var, so there is no cross-test race).
#[test]
fn over_cap_float_reduction_gates_on_fast_optin() {
    let src = r#"
        let x: Tensor[f32,(100,100)] = 1;
        tensor.sum(x, axes=[], keepdims=false)
    "#;
    let module = parser::parse(src).expect("parse over-cap reduction");
    let ir = eval::lower_to_ir(&module);

    // Default (no opt-in): the non-deterministic float tree path is REFUSED.
    // This is the interp analogue of native's UnsupportedOp — both back ends
    // agree by failing loud rather than silently returning a tolerance number.
    // SAFETY: env mutation is confined to this single (serial) test; no other
    // test in this binary reads MIND_FLOAT_REDUCE_FAST.
    unsafe { std::env::remove_var("MIND_FLOAT_REDUCE_FAST") };
    let strict =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval::to_mlir(&ir, "main")));
    assert!(
        strict.is_err(),
        "over-cap (10000-elem) f32 reduction must FAIL LOUD by default (determinism.md §4), \
         not silently emit a reordering tensor.reduce"
    );

    // Explicit opt-in: the operator knowingly selects the fast, non-deterministic
    // tier, so the tree-shaped `tensor.reduce` is emitted (never the pinned fold).
    unsafe { std::env::set_var("MIND_FLOAT_REDUCE_FAST", "1") };
    let mlir = eval::to_mlir(&ir, "main");
    unsafe { std::env::remove_var("MIND_FLOAT_REDUCE_FAST") };
    assert!(
        mlir.contains("tensor.reduce"),
        "with MIND_FLOAT_REDUCE_FAST=1 the over-cap f32 reduction must emit the tree tier, got:\n{}",
        &mlir[..mlir.len().min(800)]
    );
    assert!(
        !mlir.contains("tensor.from_elements"),
        "opt-in fast tier is the tree reduce, not the pinned unrolled fold"
    );
}

/// Integer reductions are associative (any grouping is bit-identical), so an
/// over-cap INTEGER reduction is NOT gated — it stays on the tree tier
/// unconditionally, no opt-in required.
#[test]
fn over_cap_integer_reduction_stays_treeshaped_ungated() {
    // SAFETY: single serial test; no concurrent reader of this env var.
    unsafe { std::env::remove_var("MIND_FLOAT_REDUCE_FAST") };
    let src = r#"
        let x: Tensor[i32,(100,100)] = 1;
        tensor.sum(x, axes=[], keepdims=false)
    "#;
    let module = parser::parse(src).expect("parse over-cap integer reduction");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(
        mlir.contains("tensor.reduce"),
        "over-cap integer reduction must use tree-shaped tensor.reduce (associative), got:\n{}",
        &mlir[..mlir.len().min(800)]
    );
}
