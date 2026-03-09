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

//! ASIC target dialect for MLIR lowering.
//!
//! Extends the MLIR lowering pipeline with ASIC-specific fusion passes
//! that rewrite standard MLIR op sequences into fused `mind.asic.*` ops.
//! This enables direct SSA IR execution on XRM-SSD hardware, bypassing
//! Python bytecode for nanosecond-level latency.

use std::fmt;

/// ASIC tile configuration for fused operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            m: 64,
            n: 64,
            k: 32,
        }
    }
}

impl fmt::Display for TileConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#mind.asic.tile<M={}, N={}, K={}>", self.m, self.n, self.k)
    }
}

/// Layout annotation for ASIC memory banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
}

impl fmt::Display for MemoryLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryLayout::RowMajor => write!(f, "row_major"),
            MemoryLayout::ColumnMajor => write!(f, "col_major"),
        }
    }
}

/// Bank assignment for ASIC SRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankAssignment {
    pub layout: MemoryLayout,
    pub bank: usize,
}

impl fmt::Display for BankAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#mind.layout<{}, bank={}>", self.layout, self.bank)
    }
}

/// A fused ASIC operation produced by the fusion pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsicOp {
    /// Fused matmul + bias + relu: replaces linalg.matmul → arith.addf → arith.maxf
    FusedMatMulBiasRelu {
        result: String,
        lhs: String,
        rhs: String,
        bias: String,
        lhs_ty: String,
        rhs_ty: String,
        bias_ty: String,
        result_ty: String,
        tile: TileConfig,
    },
    /// Quantized multi-head attention: replaces Q*K^T → scale → softmax → V
    QuantizedAttention {
        result: String,
        query: String,
        key: String,
        value: String,
        tensor_ty: String,
        tile: TileConfig,
    },
    /// Tiled 2D convolution with ASIC-native tiling
    TiledConv2d {
        result: String,
        input: String,
        filter: String,
        input_ty: String,
        filter_ty: String,
        result_ty: String,
        tile: TileConfig,
    },
}

impl AsicOp {
    /// Emit this fused op as MLIR text.
    pub fn to_mlir(&self) -> String {
        match self {
            AsicOp::FusedMatMulBiasRelu {
                result,
                lhs,
                rhs,
                bias,
                lhs_ty,
                rhs_ty,
                bias_ty,
                result_ty,
                tile,
            } => {
                format!(
                    "    {} = mind.asic.fused_matmul_bias_relu({} : {}, {} : {}, {} : {}) -> {} {{tile = {}}}",
                    result, lhs, lhs_ty, rhs, rhs_ty, bias, bias_ty, result_ty, tile
                )
            }
            AsicOp::QuantizedAttention {
                result,
                query,
                key,
                value,
                tensor_ty,
                tile,
            } => {
                format!(
                    "    {} = mind.asic.quantized_attention({} : {}, {} : {}, {} : {}) -> {} {{tile = {}}}",
                    result, query, tensor_ty, key, tensor_ty, value, tensor_ty, tensor_ty, tile
                )
            }
            AsicOp::TiledConv2d {
                result,
                input,
                filter,
                input_ty,
                filter_ty,
                result_ty,
                tile,
            } => {
                format!(
                    "    {} = mind.asic.tiled_conv2d({} : {}, {} : {}) -> {} {{tile = {}}}",
                    result, input, input_ty, filter, filter_ty, result_ty, tile
                )
            }
        }
    }
}

/// Pattern matcher that detects fusible op sequences in MLIR text.
///
/// Scans the MLIR body for sequences like:
/// - matmul → add (bias) → max (relu) → fused_matmul_bias_relu
/// - conv2d → tiled_conv2d
///
/// Returns a new MLIR body with fused ops replacing the matched sequences.
pub fn apply_asic_fusion(mlir_body: &str, tile: TileConfig) -> String {
    let lines: Vec<&str> = mlir_body.lines().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        // Pattern: linalg.matmul → arith.addf (bias) → arith.maxf (relu)
        if i + 2 < lines.len()
            && lines[i].contains("linalg.matmul")
            && lines[i + 1].contains("arith.addf")
            && lines[i + 2].contains("arith.maxf")
        {
            // Extract the result SSA value from the matmul line
            let matmul_line = lines[i].trim();
            if let Some(matmul_result) = extract_ssa_result(matmul_line) {
                let relu_line = lines[i + 2].trim();
                if let Some(relu_result) = extract_ssa_result(relu_line) {
                    result.push(format!(
                        "    // Fused: matmul+bias+relu → mind.asic.fused_matmul_bias_relu"));
                    result.push(format!(
                        "    // Original: {} + {} + {}",
                        matmul_line, lines[i + 1].trim(), relu_line
                    ));
                    result.push(format!(
                        "    {} = mind.asic.fused_matmul_bias_relu {{tile = {}}}",
                        relu_result, tile
                    ));
                    i += 3;
                    continue;
                }
            }
        }

        // Pattern: linalg.conv_2d_nhwc_hwcf → mind.asic.tiled_conv2d
        if lines[i].contains("linalg.conv_2d_nhwc_hwcf") {
            let conv_line = lines[i].trim();
            result.push(format!("    // Fused: conv2d → mind.asic.tiled_conv2d"));
            result.push(format!("    // Original: {}", conv_line));
            // Replace linalg.conv with mind.asic.tiled_conv2d, preserving SSA
            let fused = conv_line.replace("linalg.conv_2d_nhwc_hwcf", "mind.asic.tiled_conv2d");
            result.push(format!("    {}", fused.trim()));
            i += 1;
            continue;
        }

        result.push(lines[i].to_string());
        i += 1;
    }

    result.join("\n")
}

/// Extract the SSA result name (e.g., "%5") from an MLIR assignment line.
fn extract_ssa_result(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if let Some(eq_pos) = trimmed.find(" = ") {
        Some(trimmed[..eq_pos].trim())
    } else {
        None
    }
}

/// ASIC target specialization for the lowering pipeline.
///
/// When the target is ASIC, this pass runs after standard MLIR lowering
/// to fuse compatible op sequences into `mind.asic.*` operations.
pub fn specialize_for_asic(mlir_text: &str, tile: TileConfig) -> String {
    // Split into module header, body, and footer
    let mut header = String::new();
    let mut body = String::new();
    let mut footer = String::new();
    let mut in_body = false;
    let mut body_depth = 0;

    for line in mlir_text.lines() {
        if line.contains("func.func @main") {
            in_body = true;
            header.push_str(line);
            header.push('\n');
            body_depth = 0;
            continue;
        }
        if !in_body {
            header.push_str(line);
            header.push('\n');
            continue;
        }
        // Track brace depth to find end of function
        let opens = line.chars().filter(|c| *c == '{').count();
        let closes = line.chars().filter(|c| *c == '}').count();
        body_depth += opens as i32;
        body_depth -= closes as i32;

        if body_depth < 0 || (line.trim() == "}" && body_depth == 0) {
            footer.push_str(line);
            footer.push('\n');
            in_body = false;
            continue;
        }

        body.push_str(line);
        body.push('\n');
    }

    let fused_body = apply_asic_fusion(&body, tile);
    format!("{}{}\n{}", header, fused_body, footer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_display() {
        let tile = TileConfig::default();
        assert_eq!(tile.to_string(), "#mind.asic.tile<M=64, N=64, K=32>");
    }

    #[test]
    fn test_bank_assignment_display() {
        let bank = BankAssignment {
            layout: MemoryLayout::ColumnMajor,
            bank: 2,
        };
        assert_eq!(bank.to_string(), "#mind.layout<col_major, bank=2>");
    }

    #[test]
    fn test_fused_matmul_bias_relu_mlir() {
        let op = AsicOp::FusedMatMulBiasRelu {
            result: "%out".to_string(),
            lhs: "%a".to_string(),
            rhs: "%b".to_string(),
            bias: "%bias".to_string(),
            lhs_ty: "tensor<64x128xf32>".to_string(),
            rhs_ty: "tensor<128x256xf32>".to_string(),
            bias_ty: "tensor<256xf32>".to_string(),
            result_ty: "tensor<64x256xf32>".to_string(),
            tile: TileConfig::default(),
        };
        let mlir = op.to_mlir();
        assert!(mlir.contains("mind.asic.fused_matmul_bias_relu"));
        assert!(mlir.contains("#mind.asic.tile<M=64, N=64, K=32>"));
    }

    #[test]
    fn test_extract_ssa_result() {
        assert_eq!(extract_ssa_result("    %5 = arith.addf %3, %4 : f32"), Some("%5"));
        assert_eq!(extract_ssa_result("    return"), None);
    }

    #[test]
    fn test_asic_fusion_matmul_bias_relu() {
        let body = "\
    %0 = linalg.matmul ins(%a : tensor<4x8xf32>, %b : tensor<8x16xf32>) outs(%tmp : tensor<4x16xf32>) -> tensor<4x16xf32>
    %1 = arith.addf %0, %bias : tensor<4x16xf32>
    %2 = arith.maxf %1, %zero : tensor<4x16xf32>";

        let fused = apply_asic_fusion(body, TileConfig::default());
        assert!(fused.contains("mind.asic.fused_matmul_bias_relu"));
        assert!(!fused.contains("linalg.matmul"));
    }

    #[test]
    fn test_asic_fusion_conv2d() {
        let body = "    %0 = linalg.conv_2d_nhwc_hwcf ins(%input : tensor<1x28x28x1xf32>, %filter : tensor<3x3x1x32xf32>) outs(%tmp : tensor<1x26x26x32xf32>) -> tensor<1x26x26x32xf32>";
        let fused = apply_asic_fusion(body, TileConfig::default());
        assert!(fused.contains("mind.asic.tiled_conv2d"));
        assert!(!fused.contains("linalg.conv_2d_nhwc_hwcf"));
    }

    #[test]
    fn test_asic_fusion_passthrough_unfused() {
        let body = "    %0 = arith.constant 42 : i64\n    return %0 : i64";
        let fused = apply_asic_fusion(body, TileConfig::default());
        assert_eq!(fused, body);
    }

    #[test]
    fn test_specialize_for_asic() {
        let mlir = "module {\n  func.func @main() -> (tensor<4x16xf32>) {\n    %0 = linalg.matmul ins(%a : tensor<4x8xf32>, %b : tensor<8x16xf32>) outs(%tmp : tensor<4x16xf32>) -> tensor<4x16xf32>\n    %1 = arith.addf %0, %bias : tensor<4x16xf32>\n    %2 = arith.maxf %1, %zero : tensor<4x16xf32>\n    return %2 : tensor<4x16xf32>\n  }\n}\n";
        let result = specialize_for_asic(mlir, TileConfig::default());
        assert!(result.contains("mind.asic.fused_matmul_bias_relu"));
    }
}
