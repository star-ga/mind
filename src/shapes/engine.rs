use std::fmt;

/// A rank-N tensor shape represented as a list of extents.
///
/// Core v1 treats shapes as ordered lists of non-negative extents. The
/// engine does not attempt to encode symbolic dimensions; it is purely
/// numeric and intended for concrete validation and tests.
pub type Shape = Vec<usize>;

/// High-level shape rule categories for Core v1 operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeRuleKind {
    /// Unary elementwise op: output shape equals input shape.
    ElementwiseUnary,
    /// Binary elementwise op: broadcasting is applied to operands.
    ElementwiseBinary,
    /// Full reduction to a scalar (rank-0) value.
    ReduceAll,
    /// Matrix multiplication of two rank-2 tensors.
    MatMul2D,
}

/// Error kinds produced by the shape engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeErrorKind {
    /// Operator is unknown to the Core v1 shape engine.
    UnknownOp,
    /// Rank or size mismatch for the given rule.
    RankMismatch {
        expected: String,
        actual_lhs: Vec<usize>,
        actual_rhs: Option<Vec<usize>>,
    },
    /// Broadcasting failed for the given input shapes.
    BroadcastError {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
}

/// Rich shape error containing the operator name and a structured kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    pub op: String,
    pub kind: ShapeErrorKind,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ShapeErrorKind::UnknownOp => {
                write!(f, "shape rule not defined for op `{}`", self.op)
            }
            ShapeErrorKind::RankMismatch {
                expected,
                actual_lhs,
                actual_rhs,
            } => {
                if let Some(rhs) = actual_rhs {
                    write!(
                        f,
                        "rank mismatch for op `{}`: expected {}, got lhs={:?}, rhs={:?}",
                        self.op, expected, actual_lhs, rhs
                    )
                } else {
                    write!(
                        f,
                        "rank mismatch for op `{}`: expected {}, got lhs={:?}",
                        self.op, expected, actual_lhs
                    )
                }
            }
            ShapeErrorKind::BroadcastError { lhs, rhs } => write!(
                f,
                "cannot broadcast shapes {:?} and {:?} for op `{}`",
                lhs, rhs, self.op
            ),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Returns the coarse shape rule kind for a Core v1 operator name.
///
/// The mapping intentionally focuses on the small Core v1 surface and
/// uses the same string identifiers as the operator registry.
pub fn rule_for_op(op: &str) -> Option<ShapeRuleKind> {
    match op {
        // Unary elementwise.
        "tensor.relu" | "tensor.neg" | "tensor.exp" | "tensor.log" => {
            Some(ShapeRuleKind::ElementwiseUnary)
        }

        // Binary elementwise.
        "tensor.add" | "tensor.sub" | "tensor.mul" | "tensor.div" => {
            Some(ShapeRuleKind::ElementwiseBinary)
        }

        // Full reduction to scalar.
        "tensor.sum_all" => Some(ShapeRuleKind::ReduceAll),

        // 2D matrix multiplication.
        "tensor.matmul" => Some(ShapeRuleKind::MatMul2D),

        _ => None,
    }
}

/// Convenience helper: true if the given op is treated as Core v1
/// elementwise (unary or binary).
pub fn is_elementwise(op: &str) -> bool {
    matches!(
        rule_for_op(op),
        Some(ShapeRuleKind::ElementwiseUnary | ShapeRuleKind::ElementwiseBinary)
    )
}

/// Compute the broadcasted shape for two input shapes following the
/// standard "numpy-style" broadcasting rules.
///
/// Shapes are aligned from the right; dimensions must be equal or 1,
/// otherwise broadcasting fails.
pub fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<Shape, ShapeErrorKind> {
    let mut result = Vec::new();

    let max_rank = lhs.len().max(rhs.len());
    for i in 0..max_rank {
        let a = lhs.get(lhs.len().wrapping_sub(1).wrapping_sub(i)).copied().unwrap_or(1);
        let b = rhs.get(rhs.len().wrapping_sub(1).wrapping_sub(i)).copied().unwrap_or(1);

        let dim = if a == b || a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err(ShapeErrorKind::BroadcastError {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        };

        result.push(dim);
    }

    result.reverse();
    Ok(result)
}

/// Infer the output shape for a Core v1 operator given its input shapes.
///
/// This helper is deliberately minimal and does not yet attempt to
/// model axis parameters or partial reductions; it focuses on the
/// common, fully-determined cases that are easy to validate in tests.
pub fn infer_output_shape(op: &str, inputs: &[&[usize]]) -> Result<Shape, ShapeError> {
    let rule = match rule_for_op(op) {
        Some(rule) => rule,
        None => {
            return Err(ShapeError {
                op: op.to_string(),
                kind: ShapeErrorKind::UnknownOp,
            })
        }
    };

    match rule {
        ShapeRuleKind::ElementwiseUnary => {
            let lhs = inputs
                .get(0)
                .ok_or_else(|| ShapeError {
                    op: op.to_string(),
                    kind: ShapeErrorKind::RankMismatch {
                        expected: "one input tensor".to_string(),
                        actual_lhs: Vec::new(),
                        actual_rhs: None,
                    },
                })?;
            Ok(lhs.to_vec())
        }
        ShapeRuleKind::ElementwiseBinary => {
            let lhs = inputs.get(0).copied().unwrap_or(&[]);
            let rhs = inputs.get(1).copied().unwrap_or(&[]);
            broadcast_shapes(lhs, rhs).map_err(|kind| ShapeError {
                op: op.to_string(),
                kind,
            })
        }
        ShapeRuleKind::ReduceAll => {
            let lhs = inputs.get(0).copied().unwrap_or(&[]);
            if lhs.is_empty() {
                // Reducing a scalar stays scalar.
                Ok(Vec::new())
            } else {
                // Full reduction → rank-0 scalar.
                Ok(Vec::new())
            }
        }
        ShapeRuleKind::MatMul2D => {
            let lhs = inputs.get(0).copied().unwrap_or(&[]);
            let rhs = inputs.get(1).copied().unwrap_or(&[]);

            if lhs.len() != 2 || rhs.len() != 2 {
                return Err(ShapeError {
                    op: op.to_string(),
                    kind: ShapeErrorKind::RankMismatch {
                        expected: "two rank-2 tensors".to_string(),
                        actual_lhs: lhs.to_vec(),
                        actual_rhs: Some(rhs.to_vec()),
                    },
                });
            }

            if lhs[1] != rhs[0] {
                return Err(ShapeError {
                    op: op.to_string(),
                    kind: ShapeErrorKind::RankMismatch {
                        expected: "lhs.shape[1] == rhs.shape[0]".to_string(),
                        actual_lhs: lhs.to_vec(),
                        actual_rhs: Some(rhs.to_vec()),
                    },
                });
            }

            Ok(vec![lhs[0], rhs[1]])
        }
    }
}
