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

pub use value::ValueType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    I32,
    F32,
    BF16,
    F16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    Known(usize),
    Sym(&'static str),
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
