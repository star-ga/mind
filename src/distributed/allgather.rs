// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-device all-gather — every shard contributes its slice and the
//! concatenation order is fixed at compile time.

// no `std::fmt` import needed — this module's `Display` lives on
// `GatherOrder`-internal helpers tested via `Debug` rendering.

/// Order in which shards' contributions are concatenated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatherOrder {
    /// Shards concatenated in shard-id order: 0, 1, 2, …
    Lexicographic,
    /// Concatenated in arrival order — non-deterministic, refused under
    /// `gather_order_lexicographic`.
    Arrival,
}

impl GatherOrder {
    pub fn is_bit_identical(self) -> bool {
        matches!(self, GatherOrder::Lexicographic)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllGatherOp {
    pub operand: String,
    pub result: String,
    pub axis: u8,
    pub order: GatherOrder,
    pub world: u32,
}

impl AllGatherOp {
    pub fn lexicographic(operand: &str, result: &str, axis: u8, world: u32) -> Self {
        Self {
            operand: operand.to_string(),
            result: result.to_string(),
            axis,
            order: GatherOrder::Lexicographic,
            world,
        }
    }

    pub fn to_mlir(&self) -> String {
        format!(
            "    {} = mind.distributed.all_gather({}) {{axis = {}, order = \"{:?}\", world = {}}}",
            self.result, self.operand, self.axis, self.order, self.world
        )
    }

    pub fn is_deterministic(&self) -> bool {
        self.order.is_bit_identical()
    }

    /// Compute the post-gather shape: the gathered axis grows by `world`.
    pub fn output_shape(&self, local: &[u32]) -> Result<Vec<u32>, &'static str> {
        let a = self.axis as usize;
        if a >= local.len() {
            return Err("gather axis out of range");
        }
        let mut out = local.to_vec();
        out[a] = out[a].saturating_mul(self.world);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_helpers() {
        assert!(GatherOrder::Lexicographic.is_bit_identical());
        assert!(!GatherOrder::Arrival.is_bit_identical());
    }

    #[test]
    fn lexicographic_constructor() {
        let op = AllGatherOp::lexicographic("%local", "%full", 0, 4);
        assert_eq!(op.world, 4);
        assert!(op.is_deterministic());
    }

    #[test]
    fn output_shape_grows_along_axis() {
        let op = AllGatherOp::lexicographic("%a", "%b", 1, 3);
        assert_eq!(op.output_shape(&[8, 16]).unwrap(), vec![8, 48]);
    }

    #[test]
    fn output_shape_rejects_invalid_axis() {
        let op = AllGatherOp::lexicographic("%a", "%b", 5, 3);
        assert!(op.output_shape(&[8, 16]).is_err());
    }

    #[test]
    fn mlir_emission_includes_attrs() {
        let op = AllGatherOp::lexicographic("%local", "%full", 0, 4);
        let mlir = op.to_mlir();
        assert!(mlir.contains("mind.distributed.all_gather"));
        assert!(mlir.contains("axis = 0"));
        assert!(mlir.contains("Lexicographic"));
        assert!(mlir.contains("world = 4"));
    }
}
