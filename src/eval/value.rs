use crate::types::{DType, ShapeDim, TensorType};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorVal {
    pub dtype: DType,
    pub shape: Vec<ShapeDim>,
    /// Optional constant fill value for all elements (preview only).
    pub fill: Option<f64>,
}

impl TensorVal {
    pub fn from_type(t: &TensorType, fill: Option<f64>) -> Self {
        Self { dtype: t.dtype.clone(), shape: t.shape.clone(), fill }
    }

    pub fn new(dtype: DType, shape: Vec<ShapeDim>, fill: Option<f64>) -> Self {
        Self { dtype, shape, fill }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Tensor(TensorVal),
}

impl Value {
    pub fn is_int(&self) -> bool {
        matches!(self, Value::Int(_))
    }

    pub fn as_int(&self) -> Option<i64> {
        if let Value::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

pub fn format_value_human(v: &Value) -> String {
    match v {
        Value::Int(n) => format!("{n}"),
        Value::Tensor(t) => {
            let mut shape = String::from("(");
            for (i, d) in t.shape.iter().enumerate() {
                if i > 0 {
                    shape.push(',');
                }
                match d {
                    ShapeDim::Known(n) => shape.push_str(&n.to_string()),
                    ShapeDim::Sym(sym) => shape.push_str(sym),
                }
            }
            shape.push(')');
            match t.fill {
                Some(f) => format!(
                    "Tensor[{dtype:?},{shape}] fill={fill}",
                    dtype = t.dtype,
                    shape = shape,
                    fill = trim_float(f)
                ),
                None => format!("Tensor[{dtype:?},{shape}]", dtype = t.dtype, shape = shape),
            }
        }
    }
}

fn trim_float(x: f64) -> String {
    let s = format!("{:.6}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DType, ShapeDim};

    #[test]
    fn format_tensor_fill_trim() {
        let t = TensorVal::new(DType::F32, vec![ShapeDim::Known(2)], Some(1.2300001));
        let s = format_value_human(&Value::Tensor(t));
        assert!(s.contains("1.23"));
    }

    #[test]
    fn format_tensor_shape_symbols() {
        let t = TensorVal::new(DType::I32, vec![ShapeDim::Sym("B"), ShapeDim::Known(4)], None);
        let s = format_value_human(&Value::Tensor(t));
        assert!(s.contains("(B,4)"));
    }
}
