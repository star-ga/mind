// Copyright 2025 STARGA Inc.
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

use crate::types::DType;

/// Fixed-function metadata for a Core v1 operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSignature {
    /// Canonical operator name as it appears in the surface language or IR.
    pub name: &'static str,
    /// Number of inputs expected by the op.
    pub arity: Arity,
    /// Dtypes accepted by the op. An empty slice means "type dependent" and
    /// should be validated elsewhere.
    pub allowed_dtypes: &'static [DType],
    /// Whether the op is differentiable under the Core v1 autodiff contract.
    pub differentiable: bool,
    /// Short description of the op contract.
    pub summary: &'static str,
}

/// Arity description for ops that accept a fixed or variadic input count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    Fixed(usize),
    Variadic { min: usize },
}

/// The curated, auditable list of Core v1 ops.
///
/// The set intentionally mirrors the IR and surface language. Keep the
/// ordering stable so CLI output and documentation stay deterministic.
pub const fn core_v1_ops() -> &'static [OpSignature] {
    use Arity::*;
    &[
        OpSignature {
            name: "add",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::I32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Elementwise addition with standard broadcasting.",
        },
        OpSignature {
            name: "sub",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::I32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Elementwise subtraction with standard broadcasting.",
        },
        OpSignature {
            name: "mul",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::I32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Elementwise multiplication with standard broadcasting.",
        },
        OpSignature {
            name: "div",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Elementwise division with standard broadcasting.",
        },
        OpSignature {
            name: "tensor.sum",
            arity: Variadic { min: 1 },
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Reduction over axes with optional keepdims.",
        },
        OpSignature {
            name: "tensor.mean",
            arity: Variadic { min: 1 },
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Mean reduction over axes with optional keepdims.",
        },
        OpSignature {
            name: "tensor.reshape",
            arity: Fixed(2),
            allowed_dtypes: &[],
            differentiable: true,
            summary: "Reshape tensor to a new compatible shape.",
        },
        OpSignature {
            name: "tensor.expand_dims",
            arity: Fixed(2),
            allowed_dtypes: &[],
            differentiable: true,
            summary: "Insert a length-1 dimension at the given axis.",
        },
        OpSignature {
            name: "tensor.squeeze",
            arity: Variadic { min: 1 },
            allowed_dtypes: &[],
            differentiable: true,
            summary: "Remove length-1 dimensions.",
        },
        OpSignature {
            name: "tensor.transpose",
            arity: Fixed(2),
            allowed_dtypes: &[],
            differentiable: true,
            summary: "Permute tensor axes.",
        },
        OpSignature {
            name: "tensor.dot",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "1D dot product.",
        },
        OpSignature {
            name: "tensor.matmul",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Matrix multiplication for rank ≥ 2 tensors.",
        },
        OpSignature {
            name: "tensor.conv2d",
            arity: Fixed(2),
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "2D convolution with stride/padding parameters.",
        },
        OpSignature {
            name: "tensor.index",
            arity: Variadic { min: 2 },
            allowed_dtypes: &[],
            differentiable: false,
            summary: "Basic integer indexing.",
        },
        OpSignature {
            name: "tensor.slice",
            arity: Variadic { min: 2 },
            allowed_dtypes: &[],
            differentiable: false,
            summary: "Half-open slicing per axis.",
        },
        OpSignature {
            name: "tensor.gather",
            arity: Fixed(3),
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Gather elements along an axis using integer indices.",
        },
        OpSignature {
            name: "tensor.relu",
            arity: Fixed(1),
            allowed_dtypes: &[DType::F32, DType::BF16, DType::F16],
            differentiable: true,
            summary: "Elementwise ReLU activation.",
        },
    ]
}

/// Returns true if the provided name is a Core v1 op.
pub fn is_core_v1_op(name: &str) -> bool {
    core_v1_ops().iter().any(|op| op.name == name)
}

/// Looks up the Core v1 metadata for an op.
pub fn core_v1_op_signature(name: &str) -> Option<&'static OpSignature> {
    core_v1_ops().iter().find(|op| op.name == name)
}
