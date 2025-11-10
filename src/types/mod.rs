//! Basic tensor type definitions.
//!
//! # Example
//! ```
//! use mind::types::{TensorType, DType, ShapeDim};
//! let ty = TensorType::new(DType::F32, vec![ShapeDim::Known(2), ShapeDim::Known(3)]);
//! assert_eq!(ty.shape.len(), 2);
//! ```

pub mod infer;
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
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "i32" => Some(DType::I32),
            "f32" => Some(DType::F32),
            "bf16" => Some(DType::BF16),
            "f16" => Some(DType::F16),
            _ => None,
        }
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
    pub fn from_str(s: &str) -> Option<Self> {
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
    use super::{DType, ShapeDim, TensorType};

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
