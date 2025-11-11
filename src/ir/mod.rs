use crate::types::{DType, ShapeDim};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceSpec {
    pub axis: i64,
    pub start: i64,
    pub end: Option<i64>,
    pub stride: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSpec {
    pub axis: i64,
    pub index: i64,
}

#[derive(Debug, Clone)]
pub enum Instr {
    ConstI64(ValueId, i64),
    ConstTensor(ValueId, DType, Vec<ShapeDim>, Option<f64>),
    BinOp { dst: ValueId, op: BinOp, lhs: ValueId, rhs: ValueId },
    Sum { dst: ValueId, src: ValueId, axes: Vec<i64>, keepdims: bool },
    Mean { dst: ValueId, src: ValueId, axes: Vec<i64>, keepdims: bool },
    Reshape { dst: ValueId, src: ValueId, new_shape: Vec<ShapeDim> },
    ExpandDims { dst: ValueId, src: ValueId, axis: i64 },
    Squeeze { dst: ValueId, src: ValueId, axes: Vec<i64> },
    Transpose { dst: ValueId, src: ValueId, perm: Vec<i64> },
    Dot { dst: ValueId, a: ValueId, b: ValueId },
    MatMul { dst: ValueId, a: ValueId, b: ValueId },
    Index { dst: ValueId, src: ValueId, indices: Vec<IndexSpec> },
    Slice { dst: ValueId, src: ValueId, dims: Vec<SliceSpec> },
    Gather { dst: ValueId, src: ValueId, indices: ValueId, axis: i64 },
    Output(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub struct IRModule {
    pub instrs: Vec<Instr>,
    pub next_id: usize,
}

impl IRModule {
    pub fn new() -> Self {
        Self { instrs: Vec::new(), next_id: 0 }
    }

    pub fn fresh(&mut self) -> ValueId {
        let id = self.next_id;
        self.next_id += 1;
        ValueId(id)
    }
}

impl Default for IRModule {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for IRModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instr in &self.instrs {
            writeln!(f, "{:?}", instr)?;
        }
        Ok(())
    }
}
