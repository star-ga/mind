#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType { I32, F32, BF16, F16 }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim { Known(usize), Sym(&'static str) }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<ShapeDim>,
}

impl TensorType {
    pub fn new(dtype: DType, shape: Vec<ShapeDim>) -> Self { Self { dtype, shape } }
}
