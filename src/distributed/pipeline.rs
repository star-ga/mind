// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Pipeline parallelism — split a model's layers across stages.
//!
//! Each `[pipeline_stage(N)]` annotated function becomes a `PipelineStage`.
//! Boundaries between stages must verify the previous stage's evidence
//! emit before processing — the `evidence_chain_continuous` invariant
//! refuses to compile if any boundary bypasses the chain.

use std::fmt;

/// One stage of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    pub stage_id: u32,
    pub name: String,
    pub layers_first: u32,
    pub layers_last: u32,
}

impl PipelineStage {
    pub fn new(
        stage_id: u32,
        name: impl Into<String>,
        layers_first: u32,
        layers_last: u32,
    ) -> Self {
        Self {
            stage_id,
            name: name.into(),
            layers_first,
            layers_last,
        }
    }

    pub fn layer_count(&self) -> u32 {
        self.layers_last.saturating_sub(self.layers_first)
    }
}

/// Boundary between two stages — the audit anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageBoundary {
    pub from: u32,
    pub to: u32,
    /// Compile-time-fixed evidence digest function name (e.g.
    /// "sha256"). The same function must be referenced at both sides.
    pub digest: String,
}

impl StageBoundary {
    pub fn new(from: u32, to: u32, digest: impl Into<String>) -> Self {
        Self {
            from,
            to,
            digest: digest.into(),
        }
    }
}

/// Static-DAG pipeline graph. Stages are stored in monotonic id order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineGraph {
    pub stages: Vec<PipelineStage>,
    pub boundaries: Vec<StageBoundary>,
}

impl Default for PipelineGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineGraph {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            boundaries: Vec::new(),
        }
    }

    /// Append a stage. Stages must be added in monotonic stage-id order.
    pub fn add_stage(&mut self, stage: PipelineStage) -> Result<(), PipelineError> {
        if let Some(last) = self.stages.last() {
            if stage.stage_id <= last.stage_id {
                return Err(PipelineError::NonMonotonicStageId {
                    last: last.stage_id,
                    next: stage.stage_id,
                });
            }
            // Auto-insert the boundary between the previous and the new stage.
            self.boundaries
                .push(StageBoundary::new(last.stage_id, stage.stage_id, "sha256"));
        }
        self.stages.push(stage);
        Ok(())
    }

    /// Validate that the pipeline forms a continuous chain — no gaps,
    /// no cycles, no orphan boundaries.
    pub fn validate_chain_continuous(&self) -> Result<(), PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::Empty);
        }
        // Boundary count must equal stages-1 before we can index pairwise.
        if self.boundaries.len() + 1 != self.stages.len() {
            return Err(PipelineError::BoundaryCountMismatch {
                stages: self.stages.len(),
                boundaries: self.boundaries.len(),
            });
        }
        for (i, b) in self.boundaries.iter().enumerate() {
            if b.from >= b.to {
                return Err(PipelineError::CyclicBoundary {
                    from: b.from,
                    to: b.to,
                });
            }
            let expected_from = self.stages[i].stage_id;
            let expected_to = self.stages[i + 1].stage_id;
            if b.from != expected_from || b.to != expected_to {
                return Err(PipelineError::BrokenChain {
                    boundary_index: i,
                    expected_from,
                    expected_to,
                });
            }
        }
        Ok(())
    }

    pub fn total_layers(&self) -> u32 {
        self.stages.iter().map(|s| s.layer_count()).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    Empty,
    NonMonotonicStageId {
        last: u32,
        next: u32,
    },
    CyclicBoundary {
        from: u32,
        to: u32,
    },
    BrokenChain {
        boundary_index: usize,
        expected_from: u32,
        expected_to: u32,
    },
    BoundaryCountMismatch {
        stages: usize,
        boundaries: usize,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Empty => write!(f, "pipeline graph is empty"),
            PipelineError::NonMonotonicStageId { last, next } => {
                write!(f, "non-monotonic stage id: last={}, next={}", last, next)
            }
            PipelineError::CyclicBoundary { from, to } => {
                write!(f, "boundary {}->{} is cyclic or self-referencing", from, to)
            }
            PipelineError::BrokenChain {
                boundary_index,
                expected_from,
                expected_to,
            } => {
                write!(
                    f,
                    "boundary at index {} should connect stage {} -> {}",
                    boundary_index, expected_from, expected_to
                )
            }
            PipelineError::BoundaryCountMismatch { stages, boundaries } => write!(
                f,
                "boundary count {} does not match expected {} for {} stages",
                boundaries,
                stages.saturating_sub(1),
                stages
            ),
        }
    }
}

impl std::error::Error for PipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pipeline_fails_validation() {
        assert!(matches!(
            PipelineGraph::new().validate_chain_continuous(),
            Err(PipelineError::Empty)
        ));
    }

    #[test]
    fn three_stage_pipeline_validates() {
        let mut g = PipelineGraph::new();
        g.add_stage(PipelineStage::new(0, "stage0", 0, 8)).unwrap();
        g.add_stage(PipelineStage::new(1, "stage1", 8, 16)).unwrap();
        g.add_stage(PipelineStage::new(2, "stage2", 16, 24))
            .unwrap();
        g.validate_chain_continuous().unwrap();
        assert_eq!(g.total_layers(), 24);
        assert_eq!(g.boundaries.len(), 2);
    }

    #[test]
    fn rejects_non_monotonic_stage_ids() {
        let mut g = PipelineGraph::new();
        g.add_stage(PipelineStage::new(2, "stage2", 0, 8)).unwrap();
        let r = g.add_stage(PipelineStage::new(1, "stage1", 8, 16));
        assert!(matches!(r, Err(PipelineError::NonMonotonicStageId { .. })));
    }

    #[test]
    fn auto_inserts_boundary_between_stages() {
        let mut g = PipelineGraph::new();
        g.add_stage(PipelineStage::new(0, "s0", 0, 4)).unwrap();
        g.add_stage(PipelineStage::new(1, "s1", 4, 8)).unwrap();
        assert_eq!(g.boundaries[0].from, 0);
        assert_eq!(g.boundaries[0].to, 1);
        assert_eq!(g.boundaries[0].digest, "sha256");
    }

    #[test]
    fn stage_layer_count_is_inclusive_exclusive() {
        let s = PipelineStage::new(0, "s", 5, 13);
        assert_eq!(s.layer_count(), 8);
    }

    #[test]
    fn boundary_count_must_match_stage_count_minus_one() {
        let mut g = PipelineGraph::new();
        g.add_stage(PipelineStage::new(0, "s0", 0, 4)).unwrap();
        g.add_stage(PipelineStage::new(1, "s1", 4, 8)).unwrap();
        // Tamper with boundary list to simulate corruption.
        g.boundaries.push(StageBoundary::new(1, 2, "sha256"));
        assert!(matches!(
            g.validate_chain_continuous(),
            Err(PipelineError::BoundaryCountMismatch { .. })
        ));
    }
}
