// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-device all-reduce — every shard ends up with the same combined
//! tensor, with the reduction order fixed at compile time so two runs
//! produce bit-identical output.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Sum,
    Min,
    Max,
    Mean,
}

impl fmt::Display for ReductionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReductionKind::Sum => write!(f, "sum"),
            ReductionKind::Min => write!(f, "min"),
            ReductionKind::Max => write!(f, "max"),
            ReductionKind::Mean => write!(f, "mean"),
        }
    }
}

/// Order in which shard contributions are folded together.
///
/// `Lexicographic` is the only deterministic option; the others are
/// included for completeness but `mindc` refuses to lower them when
/// the `deterministic_all_reduce` invariant is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOrder {
    /// Shards folded in shard-id order: 0, 1, 2, ...
    Lexicographic,
    /// Tree reduction (deterministic ordering by tree shape, but
    /// requires per-tree-shape stability proof — refused under
    /// `deterministic_all_reduce` unless the tree shape is also
    /// pinned at compile time).
    Tree,
    /// Whichever shard arrives first — non-deterministic, refused under
    /// `deterministic_all_reduce`.
    Arrival,
}

impl ReductionOrder {
    pub fn is_bit_identical(self) -> bool {
        matches!(self, ReductionOrder::Lexicographic)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllReduceOp {
    pub operand: String,
    pub result: String,
    pub kind: ReductionKind,
    pub order: ReductionOrder,
    pub world: u32,
}

impl AllReduceOp {
    pub fn lexicographic_sum(operand: &str, result: &str, world: u32) -> Self {
        Self {
            operand: operand.to_string(),
            result: result.to_string(),
            kind: ReductionKind::Sum,
            order: ReductionOrder::Lexicographic,
            world,
        }
    }

    pub fn to_mlir(&self) -> String {
        format!(
            "    {} = mind.distributed.all_reduce({}) {{kind = \"{}\", order = \"{:?}\", world = {}}}",
            self.result, self.operand, self.kind, self.order, self.world
        )
    }

    pub fn is_deterministic(&self) -> bool {
        self.order.is_bit_identical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_renders() {
        assert_eq!(ReductionKind::Sum.to_string(), "sum");
        assert_eq!(ReductionKind::Mean.to_string(), "mean");
    }

    #[test]
    fn only_lexicographic_is_bit_identical() {
        assert!(ReductionOrder::Lexicographic.is_bit_identical());
        assert!(!ReductionOrder::Tree.is_bit_identical());
        assert!(!ReductionOrder::Arrival.is_bit_identical());
    }

    #[test]
    fn lexicographic_sum_constructor() {
        let op = AllReduceOp::lexicographic_sum("%g", "%G", 3);
        assert_eq!(op.kind, ReductionKind::Sum);
        assert_eq!(op.order, ReductionOrder::Lexicographic);
        assert_eq!(op.world, 3);
        assert!(op.is_deterministic());
    }

    #[test]
    fn mlir_emission_includes_all_attrs() {
        let op = AllReduceOp::lexicographic_sum("%a", "%b", 4);
        let mlir = op.to_mlir();
        assert!(mlir.contains("mind.distributed.all_reduce"));
        assert!(mlir.contains("kind = \"sum\""));
        assert!(mlir.contains("Lexicographic"));
        assert!(mlir.contains("world = 4"));
    }

    #[test]
    fn arrival_order_flagged_nondeterministic() {
        let op = AllReduceOp {
            operand: "%a".into(),
            result: "%b".into(),
            kind: ReductionKind::Sum,
            order: ReductionOrder::Arrival,
            world: 2,
        };
        assert!(!op.is_deterministic());
    }
}
