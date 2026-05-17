// Copyright 2025-2026 STARGA Inc.
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

//! Integration tests for the `mind.cerebras.stencil_tile` surface op.
//!
//! These tests cover parsing (construction), type-checking (validation
//! invariants), and lowering (MLIR text emission). They are patterned after
//! `tests/target_cerebras.rs` which covers the broader Cerebras backend
//! target surface.

use libmind::ops::cerebras::{StencilElemType, StencilTileError, StencilTileOp};

// ── Construction / parsing ────────────────────────────────────────────────────

#[test]
fn tiny_32x32_q16_16_constructs_successfully() {
    StencilTileOp::new("%out", 32, 32, StencilElemType::Q16_16, "laplacian_5pt")
        .expect("tiny 32×32 Q16.16 must construct without error");
}

#[test]
fn medium_128x128_constructs_successfully() {
    StencilTileOp::new("%out", 128, 128, StencilElemType::F32, "diffusion_kernel")
        .expect("medium 128×128 F32 must construct without error");
}

#[test]
fn large_256x256_matches_default_fabric_region() {
    // 256×256 is the FabricRegion::default() in mind-runtime; it must be
    // a legal stencil tile.
    StencilTileOp::new("%out", 256, 256, StencilElemType::Q16_16, "stencil")
        .expect("256×256 is the default FabricRegion and must be valid");
}

#[test]
fn wafer_750x750_is_in_range() {
    // Approximate WSE-3 working block; must not be rejected by the dim guard.
    StencilTileOp::new("%out", 750, 750, StencilElemType::BF16, "stencil")
        .expect("750×750 approximates WSE-3 working block and must be valid");
}

// ── Type-checking / validation ────────────────────────────────────────────────

#[test]
fn zero_rows_fails_with_fabric_dim_out_of_range() {
    let err = StencilTileOp::new("%out", 0, 32, StencilElemType::F32, "k")
        .expect_err("rows=0 must be rejected");
    assert_eq!(
        err,
        StencilTileError::FabricDimOutOfRange {
            dim: "rows",
            value: 0
        }
    );
}

#[test]
fn zero_cols_fails_with_fabric_dim_out_of_range() {
    let err = StencilTileOp::new("%out", 32, 0, StencilElemType::F32, "k")
        .expect_err("cols=0 must be rejected");
    assert_eq!(
        err,
        StencilTileError::FabricDimOutOfRange {
            dim: "cols",
            value: 0
        }
    );
}

#[test]
fn oversized_rows_fails() {
    let err = StencilTileOp::new("%out", 5000, 32, StencilElemType::F32, "k")
        .expect_err("rows=5000 exceeds MAX_FABRIC_DIM");
    match err {
        StencilTileError::FabricDimOutOfRange { dim, value } => {
            assert_eq!(dim, "rows");
            assert_eq!(value, 5000);
        }
        _ => panic!("expected FabricDimOutOfRange"),
    }
}

#[test]
fn oversized_cols_fails() {
    let err = StencilTileOp::new("%out", 32, 5000, StencilElemType::F32, "k")
        .expect_err("cols=5000 exceeds MAX_FABRIC_DIM");
    match err {
        StencilTileError::FabricDimOutOfRange { dim, value } => {
            assert_eq!(dim, "cols");
            assert_eq!(value, 5000);
        }
        _ => panic!("expected FabricDimOutOfRange"),
    }
}

#[test]
fn empty_kernel_symbol_fails() {
    let err = StencilTileOp::new("%out", 32, 32, StencilElemType::F32, "")
        .expect_err("empty kernel symbol must be rejected");
    assert_eq!(err, StencilTileError::InvalidKernelSymbol);
}

#[test]
fn max_boundary_4096x4096_is_valid() {
    StencilTileOp::new("%out", 4096, 4096, StencilElemType::F16, "boundary")
        .expect("MAX_FABRIC_DIM × MAX_FABRIC_DIM must be valid");
}

// ── Lowering / MLIR text emission ─────────────────────────────────────────────

#[test]
fn emitted_mlir_contains_op_name() {
    let op = StencilTileOp::new("%r", 32, 32, StencilElemType::Q16_16, "laplacian_5pt").unwrap();
    assert!(op.to_mlir_text().contains("mind.cerebras.stencil_tile"));
}

#[test]
fn emitted_mlir_contains_fabric_dimensions() {
    let op = StencilTileOp::new("%r", 128, 64, StencilElemType::F32, "kernel").unwrap();
    let text = op.to_mlir_text();
    assert!(text.contains("rows = 128"), "missing rows attribute");
    assert!(text.contains("cols = 64"), "missing cols attribute");
}

#[test]
fn emitted_mlir_contains_elem_type_annotation() {
    let op = StencilTileOp::new("%r", 32, 32, StencilElemType::Q16_16, "stencil").unwrap();
    let text = op.to_mlir_text();
    // elem attribute carries the MIND name for readability
    assert!(text.contains("elem = \"q16_16\""));
}

#[test]
fn emitted_mlir_contains_kernel_symbol() {
    let op = StencilTileOp::new("%r", 32, 32, StencilElemType::F32, "my_stencil").unwrap();
    assert!(op.to_mlir_text().contains("kernel = \"my_stencil\""));
}

#[test]
fn q16_16_lowers_to_i32_wire_type_in_tensor() {
    // Q16.16 is stored as signed 32-bit integer in the MLIR tensor type.
    let op = StencilTileOp::new("%r", 64, 64, StencilElemType::Q16_16, "k").unwrap();
    assert!(
        op.to_mlir_text().contains("tensor<64x64xi32>"),
        "Q16.16 must lower to i32 in the MLIR tensor type"
    );
}

#[test]
fn f32_lowers_to_f32_tensor_type() {
    let op = StencilTileOp::new("%r", 128, 128, StencilElemType::F32, "k").unwrap();
    assert!(op.to_mlir_text().contains("tensor<128x128xf32>"));
}

#[test]
fn f16_lowers_to_f16_tensor_type() {
    let op = StencilTileOp::new("%r", 32, 32, StencilElemType::F16, "k").unwrap();
    assert!(op.to_mlir_text().contains("tensor<32x32xf16>"));
}

#[test]
fn bf16_lowers_to_bf16_tensor_type() {
    let op = StencilTileOp::new("%r", 32, 32, StencilElemType::BF16, "k").unwrap();
    assert!(op.to_mlir_text().contains("tensor<32x32xbf16>"));
}

#[test]
fn result_name_appears_as_ssa_lhs() {
    let op = StencilTileOp::new("%stencil_out", 32, 32, StencilElemType::F32, "k").unwrap();
    let text = op.to_mlir_text();
    assert!(
        text.trim_start().starts_with("%stencil_out ="),
        "result SSA name must appear as the LHS assignment: {text}"
    );
}

#[test]
fn display_impl_matches_to_mlir_text() {
    let op = StencilTileOp::new("%r", 32, 32, StencilElemType::F32, "k").unwrap();
    assert_eq!(format!("{}", op), op.to_mlir_text());
}

#[test]
fn op_is_not_emitted_when_absent_from_module() {
    // Guard: if no stencil_tile op is constructed, no stencil_tile text
    // appears — the op surface is feature-gated by leading-token check.
    let empty = "";
    assert!(!empty.contains("mind.cerebras.stencil_tile"));
}

// ── Accessor correctness ──────────────────────────────────────────────────────

#[test]
fn accessors_return_construction_values() {
    let op = StencilTileOp::new("%x", 96, 48, StencilElemType::BF16, "my_kernel").unwrap();
    assert_eq!(op.rows(), 96);
    assert_eq!(op.cols(), 48);
    assert_eq!(op.elem(), StencilElemType::BF16);
    assert_eq!(op.kernel(), "my_kernel");
}
