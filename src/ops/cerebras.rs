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

// Part of the MIND project (Machine Intelligence Native Design).

//! `mind.cerebras.stencil_tile` — public MLIR-style surface op.
//!
//! This module defines the IR surface for stencil operations that target the
//! Cerebras Wafer-Scale Engine (WSE-2 / WSE-3). The op accepts a 2-D tensor
//! receiver, fabric dimensions `(rows, cols)`, and a stencil-kernel symbol.
//!
//! # Lowering contract
//!
//! `mindc` parses and type-checks this op, then emits an opaque
//! `mind.cerebras.stencil_tile` IR node in textual MLIR form. The
//! private `mind-runtime/src/backend/cerebras/` backend consumes that
//! node and lowers it to Cerebras Software Language (CSL). This crate
//! never emits CSL directly.
//!
//! # Compile-time invariants checked here
//!
//! - `rows` and `cols` are in the range `[1, 4096]` (the WSE-3 wafer is
//!   ~750 × ~750 hardware cores per quadrant; 4096 is a conservative upper
//!   bound for a single stencil tile region).
//! - `elem` is one of the element types accepted by the Cerebras integer
//!   and fixed-point pipeline (`Q16_16`, `F32`, `F16`, `BF16`).
//! - `kernel` is a non-empty ASCII identifier (the CSL compiler resolves
//!   it by name in the downstream build).
//!
//! Validation is O(1) with respect to tensor size — all checks operate on
//! the op metadata only, never on the tensor data itself.

use std::fmt;

/// Maximum allowed fabric dimension (rows or cols) for a single stencil-tile
/// region. Checked at construction time so illegal values never reach
/// `to_mlir_text()`.
const MAX_FABRIC_DIM: u32 = 4096;

/// Element types accepted by the `mind.cerebras.stencil_tile` op.
///
/// The types reflect the Cerebras integer-and-fixed-point execution path.
/// Q16.16 is the canonical MIND fixed-point format: 16 bits integer part,
/// 16 bits fractional part, stored as a signed 32-bit integer. It produces
/// bit-identical results across x86, CUDA, and the wafer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilElemType {
    /// 32-bit signed Q16.16 fixed-point. The canonical MIND wire format.
    Q16_16,
    /// 32-bit IEEE 754 single precision.
    F32,
    /// 16-bit IEEE 754 half precision.
    F16,
    /// 16-bit brain float (BF16).
    BF16,
}

impl StencilElemType {
    /// Returns the MLIR type string for this element type.
    pub fn as_mlir_str(self) -> &'static str {
        match self {
            StencilElemType::Q16_16 => "i32", // Q16.16 is wire-stored as i32
            StencilElemType::F32 => "f32",
            StencilElemType::F16 => "f16",
            StencilElemType::BF16 => "bf16",
        }
    }

    /// Returns the canonical MIND type annotation string.
    pub fn as_mind_str(self) -> &'static str {
        match self {
            StencilElemType::Q16_16 => "q16_16",
            StencilElemType::F32 => "f32",
            StencilElemType::F16 => "f16",
            StencilElemType::BF16 => "bf16",
        }
    }
}

impl fmt::Display for StencilElemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_mind_str())
    }
}

/// Validation errors produced by [`StencilTileOp::new`].
#[derive(Debug, PartialEq, Eq)]
pub enum StencilTileError {
    /// A fabric dimension was zero or exceeded the per-tile maximum.
    FabricDimOutOfRange { dim: &'static str, value: u32 },
    /// The kernel symbol was empty or contained non-ASCII characters.
    InvalidKernelSymbol,
}

impl fmt::Display for StencilTileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StencilTileError::FabricDimOutOfRange { dim, value } => write!(
                f,
                "fabric {dim} {value} is out of range [1, {MAX_FABRIC_DIM}]"
            ),
            StencilTileError::InvalidKernelSymbol => {
                f.write_str("kernel symbol must be a non-empty ASCII identifier")
            }
        }
    }
}

impl std::error::Error for StencilTileError {}

/// A validated `mind.cerebras.stencil_tile` op node.
///
/// Construct via [`StencilTileOp::new`]; this guarantees all invariants hold
/// before any MLIR text is emitted.
///
/// # Examples
///
/// ```
/// use libmind::ops::cerebras::{StencilTileOp, StencilElemType};
///
/// let op = StencilTileOp::new("%tile", 32, 32, StencilElemType::Q16_16, "laplacian_5pt")
///     .expect("valid op");
/// let mlir = op.to_mlir_text();
/// assert!(mlir.contains("mind.cerebras.stencil_tile"));
/// assert!(mlir.contains("rows = 32"));
/// assert!(mlir.contains("cols = 32"));
/// assert!(mlir.contains("q16_16"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StencilTileOp {
    /// SSA result name (e.g. `%tile_out`).
    result: String,
    /// Number of fabric rows in the tile region.
    rows: u32,
    /// Number of fabric columns in the tile region.
    cols: u32,
    /// Element type for the 2-D tensor receiver.
    elem: StencilElemType,
    /// Stencil-kernel symbol resolved by the downstream CSL compiler.
    kernel: String,
}

impl StencilTileOp {
    /// Construct and validate a `stencil_tile` op node.
    ///
    /// Returns `Err` if any invariant is violated. All checks are O(1).
    pub fn new(
        result: impl Into<String>,
        rows: u32,
        cols: u32,
        elem: StencilElemType,
        kernel: impl Into<String>,
    ) -> Result<Self, StencilTileError> {
        if rows == 0 || rows > MAX_FABRIC_DIM {
            return Err(StencilTileError::FabricDimOutOfRange {
                dim: "rows",
                value: rows,
            });
        }
        if cols == 0 || cols > MAX_FABRIC_DIM {
            return Err(StencilTileError::FabricDimOutOfRange {
                dim: "cols",
                value: cols,
            });
        }
        let kernel = kernel.into();
        if kernel.is_empty() || !kernel.is_ascii() {
            return Err(StencilTileError::InvalidKernelSymbol);
        }
        Ok(Self {
            result: result.into(),
            rows,
            cols,
            elem,
            kernel,
        })
    }

    /// Accessors — used by tests and the benchmark without going through MLIR text.
    #[inline(always)]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    #[inline(always)]
    pub fn cols(&self) -> u32 {
        self.cols
    }

    #[inline(always)]
    pub fn elem(&self) -> StencilElemType {
        self.elem
    }

    #[inline(always)]
    pub fn kernel(&self) -> &str {
        &self.kernel
    }

    /// Emit the canonical `mind.cerebras.stencil_tile` MLIR text for this op.
    ///
    /// The output is consumed by `mind-runtime/src/backend/cerebras/` which
    /// lowers it to CSL. It is intentionally textual and deterministic.
    pub fn to_mlir_text(&self) -> String {
        // tensor<{rows}x{cols}x{mlir_elem_type}>
        let tensor_ty = format!(
            "tensor<{}x{}x{}>",
            self.rows,
            self.cols,
            self.elem.as_mlir_str()
        );
        format!(
            "    {} = mind.cerebras.stencil_tile() \
             {{rows = {}, cols = {}, elem = \"{}\", kernel = \"{}\"}} : {}",
            self.result,
            self.rows,
            self.cols,
            self.elem.as_mind_str(),
            self.kernel,
            tensor_ty,
        )
    }
}

impl fmt::Display for StencilTileOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_mlir_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_op_round_trips_to_mlir() {
        let op = StencilTileOp::new("%r", 32, 32, StencilElemType::Q16_16, "laplacian_5pt")
            .expect("valid op");
        let text = op.to_mlir_text();
        assert!(text.contains("mind.cerebras.stencil_tile"));
        assert!(text.contains("rows = 32"));
        assert!(text.contains("cols = 32"));
        assert!(text.contains("q16_16"));
        assert!(text.contains("laplacian_5pt"));
    }

    #[test]
    fn mlir_tensor_type_uses_i32_for_q16_16() {
        let op =
            StencilTileOp::new("%r", 64, 64, StencilElemType::Q16_16, "stencil").expect("valid op");
        let text = op.to_mlir_text();
        // Q16.16 is wire-stored as i32
        assert!(text.contains("tensor<64x64xi32>"));
    }

    #[test]
    fn mlir_tensor_type_uses_f16_for_f16() {
        let op =
            StencilTileOp::new("%r", 128, 128, StencilElemType::F16, "stencil").expect("valid op");
        assert!(op.to_mlir_text().contains("tensor<128x128xf16>"));
    }

    #[test]
    fn zero_rows_is_rejected() {
        let err = StencilTileOp::new("%r", 0, 32, StencilElemType::F32, "k")
            .expect_err("zero rows must fail");
        assert_eq!(
            err,
            StencilTileError::FabricDimOutOfRange {
                dim: "rows",
                value: 0
            }
        );
    }

    #[test]
    fn oversized_cols_is_rejected() {
        let err = StencilTileOp::new("%r", 32, MAX_FABRIC_DIM + 1, StencilElemType::F32, "k")
            .expect_err("oversized cols must fail");
        assert_eq!(
            err,
            StencilTileError::FabricDimOutOfRange {
                dim: "cols",
                value: MAX_FABRIC_DIM + 1
            }
        );
    }

    #[test]
    fn empty_kernel_is_rejected() {
        let err = StencilTileOp::new("%r", 32, 32, StencilElemType::F32, "")
            .expect_err("empty kernel must fail");
        assert_eq!(err, StencilTileError::InvalidKernelSymbol);
    }

    #[test]
    fn max_boundary_256x256_is_valid() {
        StencilTileOp::new("%r", 256, 256, StencilElemType::Q16_16, "kernel")
            .expect("256x256 is the default FabricRegion and must be valid");
    }

    #[test]
    fn wafer_size_750x750_is_valid() {
        StencilTileOp::new("%r", 750, 750, StencilElemType::Q16_16, "stencil")
            .expect("750x750 approximates WSE-3 working block and must be valid");
    }

    #[test]
    fn elem_type_display_matches_mind_syntax() {
        assert_eq!(StencilElemType::Q16_16.to_string(), "q16_16");
        assert_eq!(StencilElemType::F32.to_string(), "f32");
        assert_eq!(StencilElemType::F16.to_string(), "f16");
        assert_eq!(StencilElemType::BF16.to_string(), "bf16");
    }
}
