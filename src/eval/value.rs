use std::collections::BTreeMap;

use crate::types::DType;

use crate::types::ShapeDim;

use crate::types::TensorType;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub String);

#[cfg(feature = "cpu-buffers")]
#[derive(Debug, Clone, PartialEq)]
pub enum Buffer {
    I32(Vec<i32>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorVal {
    pub dtype: DType,
    pub shape: Vec<ShapeDim>,
    /// Optional constant fill value for all elements (preview only).
    pub fill: Option<f64>,
    #[cfg(feature = "cpu-buffers")]
    pub buf: Option<Buffer>,
}

impl TensorVal {
    pub fn from_type(t: &TensorType, fill: Option<f64>) -> Self {
        Self {
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            fill,
            #[cfg(feature = "cpu-buffers")]
            buf: None,
        }
    }

    pub fn new(dtype: DType, shape: Vec<ShapeDim>, fill: Option<f64>) -> Self {
        Self {
            dtype,
            shape,
            fill,
            #[cfg(feature = "cpu-buffers")]
            buf: None,
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn from_materialized_f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            dtype: DType::F32,
            shape: shape.into_iter().map(ShapeDim::Known).collect(),
            fill: None,
            buf: Some(Buffer::F32(data)),
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn is_materialized_f32(&self) -> bool {
        matches!(self.buf, Some(Buffer::F32(_)))
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.buf {
            Some(Buffer::F32(values)) => Some(values.as_slice()),
            _ => None,
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.buf {
            Some(Buffer::F32(values)) => Some(values.as_mut_slice()),
            _ => None,
        }
    }

    #[cfg(feature = "cpu-buffers")]
    pub fn shape_as_usize(&self) -> Option<Vec<usize>> {
        let mut out = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            match dim {
                ShapeDim::Known(n) => out.push(*n),
                ShapeDim::Sym(_) => return None,
            }
        }
        Some(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    Tensor(TensorVal),
    GradMap(BTreeMap<VarId, TensorVal>),
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
        Value::Str(s) => s.clone(),
        Value::Tuple(items) => {
            let mut out = String::from("(");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format_value_human(item));
            }
            out.push(')');
            out
        }
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
            #[cfg(feature = "cpu-buffers")]
            if let Some(buf) = &t.buf {
                let sample = format_buffer_sample(buf, 8);
                return format!(
                    "Tensor[{dtype:?},{shape}] materialized (sample=[{sample}])",
                    dtype = t.dtype,
                    shape = shape,
                    sample = sample
                );
            }
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
        Value::GradMap(m) => {
            let mut parts = Vec::new();
            for (var, tensor) in m {
                let tensor_str = format_value_human(&Value::Tensor(tensor.clone()));
                parts.push(format!("{}: {}", var.0, tensor_str));
            }
            format!("grad{{ {} }}", parts.join(", "))
        }
    }
}

fn trim_float(x: f64) -> String {
    let s = format!("{:.6}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(feature = "cpu-buffers")]
fn format_buffer_sample(buf: &Buffer, limit: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    match buf {
        Buffer::I32(values) => {
            let max = values.len().min(limit);
            for v in &values[..max] {
                parts.push(v.to_string());
            }
            if values.len() > limit {
                parts.push("...".to_string());
            }
        }
        Buffer::F32(values) => {
            let max = values.len().min(limit);
            for v in &values[..max] {
                parts.push(trim_float(*v as f64));
            }
            if values.len() > limit {
                parts.push("...".to_string());
            }
        }
    }
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DType;
    use crate::types::ShapeDim;

    #[test]
    fn format_tensor_fill_trim() {
        let t = TensorVal::new(DType::F32, vec![ShapeDim::Known(2)], Some(1.2300001));
        let s = format_value_human(&Value::Tensor(t));
        assert!(s.contains("1.23"));
    }

    #[test]
    fn format_tensor_shape_symbols() {
        let t = TensorVal::new(
            DType::I32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(4)],
            None,
        );
        let s = format_value_human(&Value::Tensor(t));
        assert!(s.contains("(B,4)"));
    }
}
