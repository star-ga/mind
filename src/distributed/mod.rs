// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Distributed primitives for tensor and pipeline parallelism.
//!
//! These are the modules referenced by
//! `bitnet-mind-governance/docs/parallel_pipeline.md` (TP+PP for ternary
//! BitNet). All primitives in this module enforce **deterministic
//! reduction order at compile time** — every collective is keyed by a
//! lexicographic shard ID schedule, never timestamp-arrival order, so
//! cross-shard runs produce bit-identical outputs.
//!
//! Surface in `.mind` source (the names mindc resolves to ops in this
//! module):
//!
//! ```text
//! use mind.distributed.shard
//! use mind.distributed.allreduce
//! use mind.distributed.allgather
//! use mind.distributed.pipeline
//! ```
//!
//! Each submodule below is the compiler-side IR for one primitive. The
//! invariants in `invariants` are the compile-time gates `mindc`
//! evaluates before emitting MLIR.
//!
//! ## Speed-preservation discipline
//!
//! These primitives are gated by their module-level imports. Source
//! files that don't `use mind.distributed.*` pay zero analysis cost —
//! the existing 1.8–15.5 µs frontend latency stays bit-identical for
//! single-device compiles. See `docs/roadmap.md` Phase 13.6 for the
//! full discipline.

pub mod allgather;
pub mod allreduce;
pub mod invariants;
pub mod pipeline;
pub mod shard;

pub use allgather::{AllGatherOp, GatherOrder};
pub use allreduce::{AllReduceOp, ReductionKind, ReductionOrder};
pub use invariants::{DistributedInvariant, InvariantViolation};
pub use pipeline::{PipelineGraph, PipelineStage, StageBoundary};
pub use shard::{ShardLayout, ShardSpec, ShardingError};

/// Compile-time-fixed cluster topology used by every primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldSize(pub u32);

impl WorldSize {
    pub fn shards(&self) -> u32 {
        self.0
    }
    pub fn requires_collective(&self) -> bool {
        self.0 > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_size_helpers() {
        assert_eq!(WorldSize(3).shards(), 3);
        assert!(WorldSize(2).requires_collective());
        assert!(!WorldSize(1).requires_collective());
    }
}
