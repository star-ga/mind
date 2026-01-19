// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! Core types for MIC v2 IR representation.

use std::fmt;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    F32,
    F64,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    /// Parse dtype from string token.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f16" => Some(Self::F16),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            "bf16" => Some(Self::BF16),
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }

    /// Convert to canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        }
    }

    /// Convert to binary encoding byte.
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::F16 => 0,
            Self::F32 => 1,
            Self::F64 => 2,
            Self::BF16 => 3,
            Self::I8 => 4,
            Self::I16 => 5,
            Self::I32 => 6,
            Self::I64 => 7,
            Self::U8 => 8,
            Self::U16 => 9,
            Self::U32 => 10,
            Self::U64 => 11,
            Self::Bool => 12,
        }
    }

    /// Parse from binary encoding byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::F16),
            1 => Some(Self::F32),
            2 => Some(Self::F64),
            3 => Some(Self::BF16),
            4 => Some(Self::I8),
            5 => Some(Self::I16),
            6 => Some(Self::I32),
            7 => Some(Self::I64),
            8 => Some(Self::U8),
            9 => Some(Self::U16),
            10 => Some(Self::U32),
            11 => Some(Self::U64),
            12 => Some(Self::Bool),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Tensor type with dtype and shape dimensions.
///
/// Shape dimensions are stored as strings to support:
/// - Fixed dimensions: "128", "256"
/// - Symbolic dimensions: "B", "seq", "hidden"
/// - Wildcard: "?"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<String>,
}

impl TensorType {
    /// Create a new tensor type.
    pub fn new(dtype: DType, shape: Vec<String>) -> Self {
        Self { dtype, shape }
    }

    /// Create a scalar (rank-0) tensor type.
    pub fn scalar(dtype: DType) -> Self {
        Self {
            dtype,
            shape: vec![],
        }
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// Operation codes for compute nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Opcode {
    /// Matrix multiplication (2 inputs)
    Matmul,
    /// Element-wise addition (2 inputs)
    Add,
    /// Element-wise subtraction (2 inputs)
    Sub,
    /// Element-wise multiplication (2 inputs)
    Mul,
    /// Element-wise division (2 inputs)
    Div,
    /// ReLU activation (1 input)
    Relu,
    /// Softmax with axis (1 input)
    Softmax(i64),
    /// Sigmoid activation (1 input)
    Sigmoid,
    /// Tanh activation (1 input)
    Tanh,
    /// GELU activation (1 input)
    Gelu,
    /// Layer normalization (1 input)
    LayerNorm,
    /// Transpose with optional permutation
    Transpose(Vec<i64>),
    /// Reshape
    Reshape,
    /// Sum reduction with axes
    Sum(Vec<i64>),
    /// Mean reduction with axes
    Mean(Vec<i64>),
    /// Max reduction with axes
    Max(Vec<i64>),
    /// Concatenate along axis
    Concat(i64),
    /// Split along axis
    Split(i64, usize),
    /// Gather along axis
    Gather(i64),
    /// Custom operation with name
    Custom(String),
}

impl Opcode {
    /// Parse opcode from mic@2 token.
    pub fn parse(tok: &str, params: &[&str]) -> Option<Self> {
        match tok {
            "m" => Some(Self::Matmul),
            "+" => Some(Self::Add),
            "-" => Some(Self::Sub),
            "*" => Some(Self::Mul),
            "/" => Some(Self::Div),
            "r" => Some(Self::Relu),
            "s" => {
                let axis = params.first().and_then(|s| s.parse().ok()).unwrap_or(-1);
                Some(Self::Softmax(axis))
            }
            "sig" => Some(Self::Sigmoid),
            "th" => Some(Self::Tanh),
            "gelu" => Some(Self::Gelu),
            "ln" => Some(Self::LayerNorm),
            "t" => {
                let perm: Vec<i64> = params.iter().filter_map(|s| s.parse().ok()).collect();
                Some(Self::Transpose(perm))
            }
            "rshp" => Some(Self::Reshape),
            "sum" => {
                let axes: Vec<i64> = params.iter().filter_map(|s| s.parse().ok()).collect();
                Some(Self::Sum(axes))
            }
            "mean" => {
                let axes: Vec<i64> = params.iter().filter_map(|s| s.parse().ok()).collect();
                Some(Self::Mean(axes))
            }
            "max" => {
                let axes: Vec<i64> = params.iter().filter_map(|s| s.parse().ok()).collect();
                Some(Self::Max(axes))
            }
            "cat" => {
                let axis = params.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                Some(Self::Concat(axis))
            }
            "split" => {
                let axis = params.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                let n = params.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
                Some(Self::Split(axis, n))
            }
            "gth" => {
                let axis = params.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                Some(Self::Gather(axis))
            }
            _ => None,
        }
    }

    /// Convert to mic@2 token.
    pub fn as_token(&self) -> &str {
        match self {
            Self::Matmul => "m",
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Relu => "r",
            Self::Softmax(_) => "s",
            Self::Sigmoid => "sig",
            Self::Tanh => "th",
            Self::Gelu => "gelu",
            Self::LayerNorm => "ln",
            Self::Transpose(_) => "t",
            Self::Reshape => "rshp",
            Self::Sum(_) => "sum",
            Self::Mean(_) => "mean",
            Self::Max(_) => "max",
            Self::Concat(_) => "cat",
            Self::Split(_, _) => "split",
            Self::Gather(_) => "gth",
            Self::Custom(name) => name,
        }
    }

    /// Get expected number of inputs (None for variadic).
    pub fn arity(&self) -> Option<usize> {
        match self {
            Self::Matmul => Some(2),
            Self::Add | Self::Sub | Self::Mul | Self::Div => Some(2),
            Self::Relu | Self::Sigmoid | Self::Tanh | Self::Gelu => Some(1),
            Self::Softmax(_) => Some(1),
            Self::LayerNorm => Some(1),
            Self::Transpose(_) => Some(1),
            Self::Reshape => Some(1),
            Self::Sum(_) | Self::Mean(_) | Self::Max(_) => Some(1),
            Self::Concat(_) => None, // variadic
            Self::Split(_, _) => Some(1),
            Self::Gather(_) => Some(2),
            Self::Custom(_) => None,
        }
    }

    /// Convert to binary encoding byte.
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::Matmul => 0,
            Self::Add => 1,
            Self::Sub => 2,
            Self::Mul => 3,
            Self::Div => 4,
            Self::Relu => 5,
            Self::Softmax(_) => 6,
            Self::Sigmoid => 7,
            Self::Tanh => 8,
            Self::Gelu => 9,
            Self::LayerNorm => 10,
            Self::Transpose(_) => 11,
            Self::Reshape => 12,
            Self::Sum(_) => 13,
            Self::Mean(_) => 14,
            Self::Max(_) => 15,
            Self::Concat(_) => 16,
            Self::Split(_, _) => 17,
            Self::Gather(_) => 18,
            Self::Custom(_) => 255,
        }
    }
}

/// Value in the computation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// Function argument: name, type_idx
    Arg(String, usize),
    /// Learned parameter: name, type_idx
    Param(String, usize),
    /// Compute node: opcode, input_ids
    Node(Opcode, Vec<usize>),
}

impl Value {
    /// Create an argument value.
    pub fn arg(name: impl Into<String>, type_idx: usize) -> Self {
        Self::Arg(name.into(), type_idx)
    }

    /// Create a parameter value.
    pub fn param(name: impl Into<String>, type_idx: usize) -> Self {
        Self::Param(name.into(), type_idx)
    }

    /// Create a node value.
    pub fn node(opcode: Opcode, inputs: Vec<usize>) -> Self {
        Self::Node(opcode, inputs)
    }

    /// Get the binary tag for this value type.
    pub fn tag(&self) -> u8 {
        match self {
            Self::Arg(_, _) => 0,
            Self::Param(_, _) => 1,
            Self::Node(_, _) => 2,
        }
    }
}

/// Computation graph.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Optional symbol declarations (e.g., "B", "seq")
    pub symbols: Vec<String>,
    /// Type definitions
    pub types: Vec<TensorType>,
    /// Values (implicit IDs by index)
    pub values: Vec<Value>,
    /// Output value ID
    pub output: usize,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            types: Vec::new(),
            values: Vec::new(),
            output: 0,
        }
    }

    /// Add a symbol declaration.
    pub fn add_symbol(&mut self, name: impl Into<String>) {
        self.symbols.push(name.into());
    }

    /// Add a type definition and return its index.
    pub fn add_type(&mut self, tensor_type: TensorType) -> usize {
        let idx = self.types.len();
        self.types.push(tensor_type);
        idx
    }

    /// Add a value and return its implicit ID.
    pub fn add_value(&mut self, value: Value) -> usize {
        let id = self.values.len();
        self.values.push(value);
        id
    }

    /// Set the output value ID.
    pub fn set_output(&mut self, id: usize) {
        self.output = id;
    }

    /// Validate the graph structure.
    pub fn validate(&self) -> Result<(), String> {
        // Check type indices in values
        for (vid, value) in self.values.iter().enumerate() {
            match value {
                Value::Arg(_, tid) | Value::Param(_, tid) => {
                    if *tid >= self.types.len() {
                        return Err(format!("Value {} references invalid type {}", vid, tid));
                    }
                }
                Value::Node(_, inputs) => {
                    for &inp in inputs {
                        if inp >= vid {
                            return Err(format!("Value {} has forward reference to {}", vid, inp));
                        }
                    }
                }
            }
        }

        // Check output
        if self.output >= self.values.len() {
            return Err(format!(
                "Output {} references invalid value (max {})",
                self.output,
                self.values.len().saturating_sub(1)
            ));
        }

        Ok(())
    }

    /// Canonical residual block fixture: Y = relu(XW + b) + X
    pub fn residual_block() -> Self {
        let mut g = Self::new();

        // Types: T0=[128,128], T1=[128]
        g.add_type(TensorType::new(
            DType::F16,
            vec!["128".into(), "128".into()],
        ));
        g.add_type(TensorType::new(DType::F16, vec!["128".into()]));

        // Values (implicit IDs: 0, 1, 2, ...)
        g.add_value(Value::arg("X", 0)); // id=0
        g.add_value(Value::param("W", 0)); // id=1
        g.add_value(Value::param("b", 1)); // id=2
        g.add_value(Value::node(Opcode::Matmul, vec![0, 1])); // id=3: X @ W
        g.add_value(Value::node(Opcode::Add, vec![3, 2])); // id=4: (X@W) + b
        g.add_value(Value::node(Opcode::Relu, vec![4])); // id=5: relu(...)
        g.add_value(Value::node(Opcode::Add, vec![5, 0])); // id=6: relu(...) + X

        g.set_output(6);
        g
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for deep equality comparison of graphs.
pub trait GraphEq {
    fn eq(&self, other: &Self) -> bool;
}

impl GraphEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        // Compare symbols (order matters)
        if self.symbols != other.symbols {
            return false;
        }

        // Compare types
        if self.types.len() != other.types.len() {
            return false;
        }
        for (a, b) in self.types.iter().zip(other.types.iter()) {
            if a != b {
                return false;
            }
        }

        // Compare values
        if self.values.len() != other.values.len() {
            return false;
        }
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            if a != b {
                return false;
            }
        }

        // Compare output
        self.output == other.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_roundtrip() {
        for dtype in [DType::F16, DType::F32, DType::I32, DType::I64] {
            let s = dtype.as_str();
            let parsed = DType::parse(s).expect("parse failed");
            assert_eq!(dtype, parsed);
        }
    }

    #[test]
    fn test_dtype_byte_roundtrip() {
        for i in 0..=12 {
            if let Some(dtype) = DType::from_byte(i) {
                assert_eq!(dtype.to_byte(), i);
            }
        }
    }

    #[test]
    fn test_residual_block_valid() {
        let g = Graph::residual_block();
        assert!(g.validate().is_ok());
        assert_eq!(g.values.len(), 7);
        assert_eq!(g.output, 6);
    }

    #[test]
    fn test_opcode_arity() {
        assert_eq!(Opcode::Matmul.arity(), Some(2));
        assert_eq!(Opcode::Add.arity(), Some(2));
        assert_eq!(Opcode::Relu.arity(), Some(1));
        assert_eq!(Opcode::Concat(0).arity(), None); // variadic
    }
}
