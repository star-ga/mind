// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// Part of the MIND project (Machine Intelligence Native Design).

//! Tensor sharding spec — how a tensor is distributed across the world.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardLayout {
    Replicated,
    Split { axis: u8 },
    Split2D { row_axis: u8, col_axis: u8 },
}

impl fmt::Display for ShardLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardLayout::Replicated => write!(f, "replicated"),
            ShardLayout::Split { axis } => write!(f, "split(axis={})", axis),
            ShardLayout::Split2D { row_axis, col_axis } => {
                write!(f, "split2d(row={}, col={})", row_axis, col_axis)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSpec {
    pub layout: ShardLayout,
    pub world: u32,
    pub deterministic: bool,
}

impl ShardSpec {
    pub fn replicated(world: u32) -> Self {
        Self {
            layout: ShardLayout::Replicated,
            world,
            deterministic: true,
        }
    }

    pub fn split(axis: u8, world: u32) -> Self {
        Self {
            layout: ShardLayout::Split { axis },
            world,
            deterministic: true,
        }
    }

    pub fn split2d(row_axis: u8, col_axis: u8, world: u32) -> Self {
        Self {
            layout: ShardLayout::Split2D { row_axis, col_axis },
            world,
            deterministic: true,
        }
    }

    pub fn local_shape(&self, global: &[u32]) -> Result<Vec<u32>, ShardingError> {
        match self.layout {
            ShardLayout::Replicated => Ok(global.to_vec()),
            ShardLayout::Split { axis } => {
                let mut local = global.to_vec();
                let a = axis as usize;
                if a >= local.len() {
                    return Err(ShardingError::AxisOutOfRange {
                        axis,
                        ndim: local.len(),
                    });
                }
                if local[a] % self.world != 0 {
                    return Err(ShardingError::NotEvenlyDivisible {
                        dim: local[a],
                        world: self.world,
                    });
                }
                local[a] /= self.world;
                Ok(local)
            }
            ShardLayout::Split2D { row_axis, col_axis } => {
                let row_world = (self.world as f32).sqrt() as u32;
                let col_world = if row_world == 0 {
                    0
                } else {
                    self.world / row_world
                };
                if row_world * col_world != self.world {
                    return Err(ShardingError::Not2DDivisible(self.world));
                }
                let mut local = global.to_vec();
                if (row_axis as usize) >= local.len() || (col_axis as usize) >= local.len() {
                    return Err(ShardingError::AxisOutOfRange {
                        axis: row_axis.max(col_axis),
                        ndim: local.len(),
                    });
                }
                if local[row_axis as usize] % row_world != 0 {
                    return Err(ShardingError::NotEvenlyDivisible {
                        dim: local[row_axis as usize],
                        world: row_world,
                    });
                }
                if local[col_axis as usize] % col_world != 0 {
                    return Err(ShardingError::NotEvenlyDivisible {
                        dim: local[col_axis as usize],
                        world: col_world,
                    });
                }
                local[row_axis as usize] /= row_world;
                local[col_axis as usize] /= col_world;
                Ok(local)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardingError {
    AxisOutOfRange { axis: u8, ndim: usize },
    NotEvenlyDivisible { dim: u32, world: u32 },
    Not2DDivisible(u32),
}

impl fmt::Display for ShardingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardingError::AxisOutOfRange { axis, ndim } => {
                write!(
                    f,
                    "shard axis {} out of range for tensor with {} dims",
                    axis, ndim
                )
            }
            ShardingError::NotEvenlyDivisible { dim, world } => {
                write!(
                    f,
                    "dimension {} not evenly divisible by world size {}",
                    dim, world
                )
            }
            ShardingError::Not2DDivisible(world) => {
                write!(
                    f,
                    "world size {} is not a perfect square for 2D split",
                    world
                )
            }
        }
    }
}

impl std::error::Error for ShardingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicated_returns_global_shape() {
        let spec = ShardSpec::replicated(3);
        assert_eq!(spec.local_shape(&[1024, 768]).unwrap(), vec![1024, 768]);
    }

    #[test]
    fn split_evenly_divides() {
        let spec = ShardSpec::split(0, 3);
        assert_eq!(spec.local_shape(&[24, 768]).unwrap(), vec![8, 768]);
    }

    #[test]
    fn split_rejects_non_divisible() {
        let spec = ShardSpec::split(0, 3);
        assert!(matches!(
            spec.local_shape(&[25, 768]),
            Err(ShardingError::NotEvenlyDivisible { .. })
        ));
    }

    #[test]
    fn split_rejects_axis_out_of_range() {
        let spec = ShardSpec::split(5, 3);
        assert!(matches!(
            spec.local_shape(&[24, 768]),
            Err(ShardingError::AxisOutOfRange { .. })
        ));
    }

    #[test]
    fn split2d_divides_both_axes() {
        let spec = ShardSpec::split2d(0, 1, 4);
        assert_eq!(spec.local_shape(&[16, 16]).unwrap(), vec![8, 8]);
    }

    #[test]
    fn shard_layout_renders() {
        assert_eq!(ShardLayout::Replicated.to_string(), "replicated");
        assert_eq!(ShardLayout::Split { axis: 1 }.to_string(), "split(axis=1)");
        assert_eq!(
            ShardLayout::Split2D {
                row_axis: 0,
                col_axis: 1
            }
            .to_string(),
            "split2d(row=0, col=1)"
        );
    }

    #[test]
    fn deterministic_default_is_true() {
        assert!(ShardSpec::replicated(2).deterministic);
        assert!(ShardSpec::split(0, 4).deterministic);
    }
}
