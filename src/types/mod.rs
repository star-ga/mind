// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Basic tensor type definitions.
//!
//! # Example
//! ```
//! use libmind::types::{TensorType, DType, ShapeDim};
//! let ty = TensorType::new(DType::F32, vec![ShapeDim::Known(2), ShapeDim::Known(3)]);
//! assert_eq!(ty.shape.len(), 2);
//! ```

pub mod infer;
pub mod intern;
pub mod value;

pub use infer::*;
pub use value::ValueType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    I32,
    F32,
    BF16,
    F16,
}

impl DType {
    fn parse_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "i32" => Some(DType::I32),
            "f32" => Some(DType::F32),
            "bf16" => Some(DType::BF16),
            "f16" => Some(DType::F16),
            _ => None,
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::parse_name(name)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DType::I32 => "i32",
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
        }
    }
}

impl std::str::FromStr for DType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DType::parse_name(s).ok_or(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    Known(usize),
    Sym(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvPadding {
    Valid,
    Same,
}

impl ConvPadding {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "valid" => Some(ConvPadding::Valid),
            "same" => Some(ConvPadding::Same),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ConvPadding::Valid => "valid",
            ConvPadding::Same => "same",
        }
    }
}

impl std::str::FromStr for ConvPadding {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ConvPadding::parse(s).ok_or(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<ShapeDim>,
}

impl TensorType {
    pub fn new(dtype: DType, shape: Vec<ShapeDim>) -> Self {
        Self { dtype, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;
    use super::ShapeDim;
    use super::TensorType;

    #[test]
    fn tensor_type_new_covers_constructor() {
        let t = TensorType::new(DType::F32, vec![ShapeDim::Known(2), ShapeDim::Known(3)]);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![ShapeDim::Known(2), ShapeDim::Known(3)]);
    }

    #[test]
    fn tensor_type_with_symbolic_dim() {
        let t = TensorType::new(DType::I32, vec![ShapeDim::Sym("B"), ShapeDim::Known(128)]);
        assert!(matches!(t.shape[0], ShapeDim::Sym("B")));
        assert!(matches!(t.shape[1], ShapeDim::Known(128)));
    }
}
