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

//! Memory Access Pattern (MAP) optimization for VRAM.
//!
//! Analyzes tensor access patterns to determine optimal memory layouts
//! (row-major vs column-major) and performs tensor coloring to reduce
//! peak VRAM usage through buffer sharing.
//!
//! Target: MBU 91.3% → 96.8%, VRAM 6.31 GB → 5.85 GB for 8B model.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ir::{IRModule, Instr, ValueId};

/// Access pattern for a single tensor in the IR.
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// The tensor's SSA value ID.
    pub tensor_id: ValueId,
    /// Whether the tensor is accessed in row-major order.
    pub row_major_accesses: usize,
    /// Whether the tensor is accessed in column-major order (e.g., transposed matmul).
    pub col_major_accesses: usize,
    /// Estimated reuse distance (instructions between uses).
    pub reuse_distance: usize,
    /// First use (instruction index).
    pub first_use: usize,
    /// Last use (instruction index).
    pub last_use: usize,
    /// Estimated size in bytes (0 if unknown/symbolic).
    pub estimated_bytes: u64,
}

impl AccessPattern {
    /// Returns the optimal layout based on access patterns.
    pub fn optimal_layout(&self) -> LayoutDecision {
        if self.col_major_accesses > self.row_major_accesses {
            LayoutDecision::ColumnMajor
        } else {
            LayoutDecision::RowMajor
        }
    }

    /// Lifetime span in instruction indices.
    pub fn lifetime(&self) -> usize {
        self.last_use.saturating_sub(self.first_use)
    }
}

/// Compiler-decided memory layout for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutDecision {
    RowMajor,
    ColumnMajor,
}

impl fmt::Display for LayoutDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutDecision::RowMajor => write!(f, "row_major"),
            LayoutDecision::ColumnMajor => write!(f, "col_major"),
        }
    }
}

/// Result of tensor coloring: which tensors can share buffers.
#[derive(Debug, Clone)]
pub struct ColorAssignment {
    /// Maps tensor ValueId to a color (buffer group).
    pub colors: BTreeMap<ValueId, usize>,
    /// Total number of colors (distinct buffers needed).
    pub num_colors: usize,
    /// Estimated peak memory with coloring.
    pub peak_bytes_colored: u64,
    /// Estimated peak memory without coloring.
    pub peak_bytes_uncolored: u64,
}

impl ColorAssignment {
    /// Memory savings as a fraction (0.0 to 1.0).
    pub fn savings_ratio(&self) -> f64 {
        if self.peak_bytes_uncolored == 0 {
            return 0.0;
        }
        1.0 - (self.peak_bytes_colored as f64 / self.peak_bytes_uncolored as f64)
    }
}

/// Full MAP analysis result for an IR module.
#[derive(Debug, Clone)]
pub struct MapAnalysis {
    /// Per-tensor access patterns.
    pub patterns: Vec<AccessPattern>,
    /// Layout decisions.
    pub layouts: BTreeMap<ValueId, LayoutDecision>,
    /// Bank assignments for ASIC SRAM.
    pub bank_assignments: BTreeMap<ValueId, usize>,
    /// Tensor coloring result.
    pub coloring: ColorAssignment,
}

impl fmt::Display for MapAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Memory Access Pattern (MAP) Analysis ===")?;
        writeln!(f)?;
        for pat in &self.patterns {
            let layout = self.layouts.get(&pat.tensor_id).unwrap_or(&LayoutDecision::RowMajor);
            let bank = self.bank_assignments.get(&pat.tensor_id).unwrap_or(&0);
            let color = self.coloring.colors.get(&pat.tensor_id).unwrap_or(&0);
            writeln!(
                f,
                "  %{}: layout={}, bank={}, color={}, lifetime=[{}..{}], size={}B, reuse={}",
                pat.tensor_id.0,
                layout,
                bank,
                color,
                pat.first_use,
                pat.last_use,
                pat.estimated_bytes,
                pat.reuse_distance,
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "  Peak VRAM: {} → {} ({:.1}% reduction, {:.1}% MBU)",
            format_bytes(self.coloring.peak_bytes_uncolored),
            format_bytes(self.coloring.peak_bytes_colored),
            self.coloring.savings_ratio() * 100.0,
            if self.coloring.peak_bytes_uncolored > 0 {
                (1.0 - self.coloring.savings_ratio()) * 100.0
            } else {
                100.0
            },
        )?;
        Ok(())
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Analyze access patterns for all tensors in an IR module.
pub fn analyze_access_patterns(module: &IRModule) -> Vec<AccessPattern> {
    // Track definitions and uses of each ValueId
    let mut defs: BTreeMap<ValueId, usize> = BTreeMap::new();
    let mut uses: BTreeMap<ValueId, Vec<usize>> = BTreeMap::new();
    let mut is_tensor: BTreeSet<ValueId> = BTreeSet::new();
    let mut col_major: BTreeSet<ValueId> = BTreeSet::new();

    for (idx, instr) in module.instrs.iter().enumerate() {
        match instr {
            Instr::ConstTensor(id, _, _, _) => {
                defs.insert(*id, idx);
                is_tensor.insert(*id);
            }
            Instr::MatMul { dst, a, b } => {
                defs.insert(*dst, idx);
                is_tensor.insert(*dst);
                uses.entry(*a).or_default().push(idx);
                uses.entry(*b).or_default().push(idx);
                // RHS of matmul is accessed column-major
                col_major.insert(*b);
            }
            Instr::Conv2d { dst, input, filter, .. } => {
                defs.insert(*dst, idx);
                is_tensor.insert(*dst);
                uses.entry(*input).or_default().push(idx);
                uses.entry(*filter).or_default().push(idx);
            }
            Instr::BinOp { dst, lhs, rhs, .. } => {
                defs.insert(*dst, idx);
                uses.entry(*lhs).or_default().push(idx);
                uses.entry(*rhs).or_default().push(idx);
            }
            Instr::Transpose { dst, src, .. } => {
                defs.insert(*dst, idx);
                is_tensor.insert(*dst);
                uses.entry(*src).or_default().push(idx);
                col_major.insert(*src);
            }
            Instr::Output(id) => {
                uses.entry(*id).or_default().push(idx);
            }
            _ => {}
        }
    }

    is_tensor
        .iter()
        .map(|id| {
            let def_idx = defs.get(id).copied().unwrap_or(0);
            let use_indices = uses.get(id).cloned().unwrap_or_default();
            let first_use = use_indices.iter().copied().min().unwrap_or(def_idx);
            let last_use = use_indices.iter().copied().max().unwrap_or(def_idx);
            let reuse_distance = if use_indices.len() > 1 {
                use_indices.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(0)
            } else {
                0
            };
            let col_accesses = if col_major.contains(id) { 1 } else { 0 };
            let row_accesses = use_indices.len().saturating_sub(col_accesses);

            AccessPattern {
                tensor_id: *id,
                row_major_accesses: row_accesses,
                col_major_accesses: col_accesses,
                reuse_distance,
                first_use,
                last_use,
                estimated_bytes: 0, // Requires shape resolution
            }
        })
        .collect()
}

/// Perform tensor coloring: assign non-overlapping tensors to shared buffers.
///
/// Uses a greedy graph-coloring approach on the interference graph
/// (two tensors interfere if their lifetimes overlap).
pub fn color_tensors(patterns: &[AccessPattern]) -> ColorAssignment {
    if patterns.is_empty() {
        return ColorAssignment {
            colors: BTreeMap::new(),
            num_colors: 0,
            peak_bytes_colored: 0,
            peak_bytes_uncolored: 0,
        };
    }

    // Build interference graph
    let n = patterns.len();
    let mut interferes = vec![vec![false; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            // Two tensors interfere if their lifetimes overlap
            let a = &patterns[i];
            let b = &patterns[j];
            if a.first_use <= b.last_use && b.first_use <= a.last_use {
                interferes[i][j] = true;
                interferes[j][i] = true;
            }
        }
    }

    // Greedy coloring
    let mut colors: BTreeMap<ValueId, usize> = BTreeMap::new();
    let mut max_color: usize = 0;

    for i in 0..n {
        let mut used_colors: BTreeSet<usize> = BTreeSet::new();
        for j in 0..n {
            if interferes[i][j] {
                if let Some(&c) = colors.get(&patterns[j].tensor_id) {
                    used_colors.insert(c);
                }
            }
        }
        // Find smallest unused color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }
        colors.insert(patterns[i].tensor_id, color);
        if color > max_color {
            max_color = color;
        }
    }

    let peak_uncolored: u64 = patterns.iter().map(|p| p.estimated_bytes).sum();
    // With coloring, each color group only needs max(sizes) not sum(sizes)
    let mut color_max: BTreeMap<usize, u64> = BTreeMap::new();
    for pat in patterns {
        let c = colors.get(&pat.tensor_id).copied().unwrap_or(0);
        let entry = color_max.entry(c).or_insert(0);
        if pat.estimated_bytes > *entry {
            *entry = pat.estimated_bytes;
        }
    }
    let peak_colored: u64 = color_max.values().sum();

    ColorAssignment {
        colors,
        num_colors: max_color + 1,
        peak_bytes_colored: peak_colored,
        peak_bytes_uncolored: peak_uncolored,
    }
}

/// Run full MAP optimization on an IR module.
pub fn optimize_layout(module: &IRModule) -> MapAnalysis {
    let patterns = analyze_access_patterns(module);

    let mut layouts = BTreeMap::new();
    let mut bank_assignments = BTreeMap::new();
    let mut bank_counter = 0;

    for pat in &patterns {
        let layout = pat.optimal_layout();
        layouts.insert(pat.tensor_id, layout);
        bank_assignments.insert(pat.tensor_id, bank_counter % 4); // 4 SRAM banks
        bank_counter += 1;
    }

    let coloring = color_tensors(&patterns);

    MapAnalysis {
        patterns,
        layouts,
        bank_assignments,
        coloring,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IRModule;
    use crate::types::{DType, ShapeDim};

    fn make_test_module() -> IRModule {
        // matmul: A(4x8) * B(8x16) = C(4x16)
        IRModule {
            instrs: vec![
                Instr::ConstTensor(
                    ValueId(0),
                    DType::F32,
                    vec![ShapeDim::Known(4), ShapeDim::Known(8)],
                    Some(1.0),
                ),
                Instr::ConstTensor(
                    ValueId(1),
                    DType::F32,
                    vec![ShapeDim::Known(8), ShapeDim::Known(16)],
                    Some(1.0),
                ),
                Instr::MatMul {
                    dst: ValueId(2),
                    a: ValueId(0),
                    b: ValueId(1),
                },
                Instr::Output(ValueId(2)),
            ],
            next_id: 3,
        }
    }

    #[test]
    fn test_analyze_access_patterns() {
        let module = make_test_module();
        let patterns = analyze_access_patterns(&module);
        assert_eq!(patterns.len(), 3); // A, B, C
    }

    #[test]
    fn test_matmul_rhs_column_major() {
        let module = make_test_module();
        let patterns = analyze_access_patterns(&module);
        // ValueId(1) is the RHS of matmul — should prefer column-major
        let rhs = patterns.iter().find(|p| p.tensor_id == ValueId(1)).unwrap();
        assert_eq!(rhs.optimal_layout(), LayoutDecision::ColumnMajor);
    }

    #[test]
    fn test_matmul_lhs_row_major() {
        let module = make_test_module();
        let patterns = analyze_access_patterns(&module);
        let lhs = patterns.iter().find(|p| p.tensor_id == ValueId(0)).unwrap();
        assert_eq!(lhs.optimal_layout(), LayoutDecision::RowMajor);
    }

    #[test]
    fn test_tensor_coloring_non_overlapping() {
        // Two tensors with non-overlapping lifetimes should share a buffer
        let patterns = vec![
            AccessPattern {
                tensor_id: ValueId(0),
                row_major_accesses: 1,
                col_major_accesses: 0,
                reuse_distance: 0,
                first_use: 0,
                last_use: 2,
                estimated_bytes: 1024,
            },
            AccessPattern {
                tensor_id: ValueId(1),
                row_major_accesses: 1,
                col_major_accesses: 0,
                reuse_distance: 0,
                first_use: 5,
                last_use: 8,
                estimated_bytes: 2048,
            },
        ];
        let coloring = color_tensors(&patterns);
        // Non-overlapping: should get the same color
        assert_eq!(coloring.num_colors, 1);
        assert_eq!(coloring.colors[&ValueId(0)], coloring.colors[&ValueId(1)]);
    }

    #[test]
    fn test_tensor_coloring_overlapping() {
        // Two tensors with overlapping lifetimes need separate buffers
        let patterns = vec![
            AccessPattern {
                tensor_id: ValueId(0),
                row_major_accesses: 1,
                col_major_accesses: 0,
                reuse_distance: 0,
                first_use: 0,
                last_use: 5,
                estimated_bytes: 1024,
            },
            AccessPattern {
                tensor_id: ValueId(1),
                row_major_accesses: 1,
                col_major_accesses: 0,
                reuse_distance: 0,
                first_use: 3,
                last_use: 8,
                estimated_bytes: 2048,
            },
        ];
        let coloring = color_tensors(&patterns);
        assert_eq!(coloring.num_colors, 2);
        assert_ne!(coloring.colors[&ValueId(0)], coloring.colors[&ValueId(1)]);
    }

    #[test]
    fn test_optimize_layout_full() {
        let module = make_test_module();
        let analysis = optimize_layout(&module);
        assert!(!analysis.patterns.is_empty());
        assert!(!analysis.layouts.is_empty());
        // K matrix (RHS of matmul) should be column-major
        assert_eq!(
            analysis.layouts.get(&ValueId(1)),
            Some(&LayoutDecision::ColumnMajor)
        );
    }

    #[test]
    fn test_map_display() {
        let module = make_test_module();
        let analysis = optimize_layout(&module);
        let display = format!("{}", analysis);
        assert!(display.contains("Memory Access Pattern"));
        assert!(display.contains("col_major"));
    }

    #[test]
    fn test_empty_module() {
        let module = IRModule { instrs: vec![], next_id: 0 };
        let analysis = optimize_layout(&module);
        assert!(analysis.patterns.is_empty());
        assert_eq!(analysis.coloring.num_colors, 0);
    }
}
