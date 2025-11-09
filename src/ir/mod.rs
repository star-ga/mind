use crate::types::{DType, ShapeDim};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone)]
pub enum Instr {
    ConstI64(ValueId, i64),
    ConstTensor(ValueId, DType, Vec<ShapeDim>, Option<f64>),
    BinOp { dst: ValueId, op: BinOp, lhs: ValueId, rhs: ValueId },
    Sum { dst: ValueId, src: ValueId },
    Reshape { dst: ValueId, src: ValueId, new_shape: Vec<ShapeDim> },
    MatMul { dst: ValueId, a: ValueId, b: ValueId },
    Slice { dst: ValueId, src: ValueId, axis: usize, start: usize, end: usize, stride: usize },
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

impl fmt::Display for IRModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instr in &self.instrs {
            writeln!(f, "{:?}", instr)?;
        }
        Ok(())
    }
}
