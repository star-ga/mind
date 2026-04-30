// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Compile-time invariants enforced over distributed primitives.

use super::{
    allgather::{AllGatherOp, GatherOrder},
    allreduce::{AllReduceOp, ReductionOrder},
    pipeline::{PipelineError, PipelineGraph},
};
use std::fmt;

/// Invariants `mindc` evaluates before lowering distributed ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributedInvariant {
    DeterministicAllReduce,
    ReductionOrderLexicographic,
    GatherOrderLexicographic,
    EvidenceChainContinuous,
}

impl DistributedInvariant {
    pub fn name(self) -> &'static str {
        match self {
            DistributedInvariant::DeterministicAllReduce => "deterministic_all_reduce",
            DistributedInvariant::ReductionOrderLexicographic => "reduction_order_lexicographic",
            DistributedInvariant::GatherOrderLexicographic => "gather_order_lexicographic",
            DistributedInvariant::EvidenceChainContinuous => "evidence_chain_continuous",
        }
    }
}

impl fmt::Display for DistributedInvariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvariantViolation {
    NonLexicographicAllReduce { observed: ReductionOrder },
    NonLexicographicAllGather { observed: GatherOrder },
    BrokenEvidenceChain(String),
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvariantViolation::NonLexicographicAllReduce { observed } => write!(
                f,
                "all-reduce uses non-lexicographic order ({:?}); deterministic_all_reduce refuses to lower",
                observed
            ),
            InvariantViolation::NonLexicographicAllGather { observed } => write!(
                f,
                "all-gather uses non-lexicographic order ({:?}); gather_order_lexicographic refuses to lower",
                observed
            ),
            InvariantViolation::BrokenEvidenceChain(detail) => {
                write!(f, "evidence_chain_continuous violation: {}", detail)
            }
        }
    }
}

impl std::error::Error for InvariantViolation {}

/// Verify `deterministic_all_reduce` against a single op.
pub fn check_deterministic_all_reduce(op: &AllReduceOp) -> Result<(), InvariantViolation> {
    if !op.order.is_bit_identical() {
        return Err(InvariantViolation::NonLexicographicAllReduce { observed: op.order });
    }
    Ok(())
}

/// Verify `gather_order_lexicographic` against a single op.
pub fn check_gather_order_lexicographic(op: &AllGatherOp) -> Result<(), InvariantViolation> {
    if !op.order.is_bit_identical() {
        return Err(InvariantViolation::NonLexicographicAllGather { observed: op.order });
    }
    Ok(())
}

/// Verify `evidence_chain_continuous` against a pipeline graph.
pub fn check_evidence_chain_continuous(graph: &PipelineGraph) -> Result<(), InvariantViolation> {
    graph.validate_chain_continuous().map_err(|e: PipelineError| {
        InvariantViolation::BrokenEvidenceChain(e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        allgather::AllGatherOp,
        allreduce::{AllReduceOp, ReductionKind},
        pipeline::{PipelineGraph, PipelineStage},
    };

    #[test]
    fn invariant_names_render() {
        assert_eq!(
            DistributedInvariant::DeterministicAllReduce.to_string(),
            "deterministic_all_reduce"
        );
        assert_eq!(
            DistributedInvariant::EvidenceChainContinuous.to_string(),
            "evidence_chain_continuous"
        );
    }

    #[test]
    fn lexicographic_all_reduce_passes() {
        let op = AllReduceOp::lexicographic_sum("%g", "%G", 3);
        check_deterministic_all_reduce(&op).unwrap();
    }

    #[test]
    fn arrival_order_all_reduce_violates() {
        let op = AllReduceOp {
            operand: "%g".into(),
            result: "%G".into(),
            kind: ReductionKind::Sum,
            order: ReductionOrder::Arrival,
            world: 3,
        };
        assert!(matches!(
            check_deterministic_all_reduce(&op),
            Err(InvariantViolation::NonLexicographicAllReduce { .. })
        ));
    }

    #[test]
    fn lexicographic_all_gather_passes() {
        let op = AllGatherOp::lexicographic("%a", "%b", 0, 4);
        check_gather_order_lexicographic(&op).unwrap();
    }

    #[test]
    fn arrival_order_all_gather_violates() {
        let op = AllGatherOp {
            operand: "%a".into(),
            result: "%b".into(),
            axis: 0,
            order: GatherOrder::Arrival,
            world: 4,
        };
        assert!(matches!(
            check_gather_order_lexicographic(&op),
            Err(InvariantViolation::NonLexicographicAllGather { .. })
        ));
    }

    #[test]
    fn three_stage_pipeline_passes_evidence_chain() {
        let mut g = PipelineGraph::new();
        g.add_stage(PipelineStage::new(0, "s0", 0, 8)).unwrap();
        g.add_stage(PipelineStage::new(1, "s1", 8, 16)).unwrap();
        g.add_stage(PipelineStage::new(2, "s2", 16, 24)).unwrap();
        check_evidence_chain_continuous(&g).unwrap();
    }

    #[test]
    fn empty_pipeline_violates_evidence_chain() {
        let g = PipelineGraph::new();
        assert!(matches!(
            check_evidence_chain_continuous(&g),
            Err(InvariantViolation::BrokenEvidenceChain(_))
        ));
    }
}
