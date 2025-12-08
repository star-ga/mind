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

use crate::types::ConvPadding;
use crate::types::DType;
use crate::types::ShapeDim;

use std::fmt;

mod print;
mod verify;

pub use crate::opt::ir_canonical::canonicalize_module;
pub use print::format_ir_module;
pub use verify::{verify_module, IrVerifyError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub usize);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

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
    BinOp {
        dst: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
    },
    Sum {
        dst: ValueId,
        src: ValueId,
        axes: Vec<i64>,
        keepdims: bool,
    },
    Mean {
        dst: ValueId,
        src: ValueId,
        axes: Vec<i64>,
        keepdims: bool,
    },
    Reshape {
        dst: ValueId,
        src: ValueId,
        new_shape: Vec<ShapeDim>,
    },
    ExpandDims {
        dst: ValueId,
        src: ValueId,
        axis: i64,
    },
    Squeeze {
        dst: ValueId,
        src: ValueId,
        axes: Vec<i64>,
    },
    Transpose {
        dst: ValueId,
        src: ValueId,
        perm: Vec<i64>,
    },
    Dot {
        dst: ValueId,
        a: ValueId,
        b: ValueId,
    },
    MatMul {
        dst: ValueId,
        a: ValueId,
        b: ValueId,
    },
    Conv2d {
        dst: ValueId,
        input: ValueId,
        filter: ValueId,
        stride_h: usize,
        stride_w: usize,
        padding: ConvPadding,
    },
    Index {
        dst: ValueId,
        src: ValueId,
        indices: Vec<IndexSpec>,
    },
    Slice {
        dst: ValueId,
        src: ValueId,
        dims: Vec<SliceSpec>,
    },
    Gather {
        dst: ValueId,
        src: ValueId,
        indices: ValueId,
        axis: i64,
    },
    Output(ValueId),
}

pub(crate) fn instruction_dst(instr: &Instr) -> Option<ValueId> {
    match instr {
        Instr::ConstI64(dst, ..)
        | Instr::ConstTensor(dst, ..)
        | Instr::BinOp { dst, .. }
        | Instr::Sum { dst, .. }
        | Instr::Mean { dst, .. }
        | Instr::Reshape { dst, .. }
        | Instr::ExpandDims { dst, .. }
        | Instr::Squeeze { dst, .. }
        | Instr::Transpose { dst, .. }
        | Instr::Dot { dst, .. }
        | Instr::MatMul { dst, .. }
        | Instr::Conv2d { dst, .. }
        | Instr::Index { dst, .. }
        | Instr::Slice { dst, .. }
        | Instr::Gather { dst, .. } => Some(*dst),
        Instr::Output(_) => None,
    }
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
        Self {
            instrs: Vec::new(),
            next_id: 0,
        }
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
        write!(f, "{}", print::format_ir_module(self))
    }
}

/// Run verification and canonicalization on the module before handing it to a
/// backend.
pub fn prepare_ir_for_backend(module: &mut IRModule) -> Result<(), IrVerifyError> {
    verify::verify_module(module)?;
    crate::opt::ir_canonical::canonicalize_module(module);
    verify::verify_module(module)
}
