use super::TensorType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    ScalarI32,
    Tensor(TensorType),
}

impl ValueType {
    pub fn is_scalar(&self) -> bool {
        matches!(self, ValueType::ScalarI32)
    }
}
