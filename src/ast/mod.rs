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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    start: usize,
    end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Int(i64),
    Ident(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeAnn {
    ScalarI32,
    ScalarI64,
    ScalarF32,
    ScalarF64,
    ScalarBool,
    Tensor {
        dtype: String,
        dims: Vec<String>,
    },
    /// Differentiable tensor: `diff tensor<f32[N, M]>`
    DiffTensor {
        dtype: String,
        dims: Vec<String>,
    },
}

/// Function parameter: `name: type`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: TypeAnn,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Lit(Literal, Span),
    Binary {
        op: BinOp,
        left: Box<Node>,
        right: Box<Node>,
        span: Span,
    },
    Paren(Box<Node>, Span),
    Tuple {
        elements: Vec<Node>,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<Node>,
        span: Span,
    },
    CallGrad {
        loss: Box<Node>,
        wrt: Vec<String>,
        span: Span,
    },
    CallTensorSum {
        x: Box<Node>,
        axes: Vec<i32>,
        keepdims: bool,
        span: Span,
    },
    CallTensorMean {
        x: Box<Node>,
        axes: Vec<i32>,
        keepdims: bool,
        span: Span,
    },
    CallReshape {
        x: Box<Node>,
        dims: Vec<String>,
        span: Span,
    },
    CallExpandDims {
        x: Box<Node>,
        axis: i32,
        span: Span,
    },
    CallSqueeze {
        x: Box<Node>,
        axes: Vec<i32>,
        span: Span,
    },
    CallTranspose {
        x: Box<Node>,
        axes: Option<Vec<i32>>,
        span: Span,
    },
    CallIndex {
        x: Box<Node>,
        axis: i32,
        i: i32,
        span: Span,
    },
    CallSlice {
        x: Box<Node>,
        axis: i32,
        start: i32,
        end: i32,
        span: Span,
    },
    CallSliceStride {
        x: Box<Node>,
        axis: i32,
        start: i32,
        end: i32,
        step: i32,
        span: Span,
    },
    CallGather {
        x: Box<Node>,
        axis: i32,
        idx: Box<Node>,
        span: Span,
    },
    CallDot {
        a: Box<Node>,
        b: Box<Node>,
        span: Span,
    },
    CallMatMul {
        a: Box<Node>,
        b: Box<Node>,
        span: Span,
    },
    CallTensorRelu {
        x: Box<Node>,
        span: Span,
    },
    CallTensorConv2d {
        x: Box<Node>,
        w: Box<Node>,
        stride_h: usize,
        stride_w: usize,
        padding: ConvPadding,
        span: Span,
    },
    Let {
        name: String,
        ann: Option<TypeAnn>,
        value: Box<Node>,
        span: Span,
    },
    Assign {
        name: String,
        value: Box<Node>,
        span: Span,
    },
    /// Function definition: `fn name(params) -> ret_type { body }`
    FnDef {
        name: String,
        params: Vec<Param>,
        ret_type: Option<TypeAnn>,
        body: Vec<Node>,
        span: Span,
    },
    /// Return statement: `return expr`
    Return {
        value: Option<Box<Node>>,
        span: Span,
    },
    /// Block of statements: `{ stmts }`
    Block {
        stmts: Vec<Node>,
        span: Span,
    },
    /// If expression: `if cond { then } else { else }`
    If {
        cond: Box<Node>,
        then_branch: Vec<Node>,
        else_branch: Option<Vec<Node>>,
        span: Span,
    },
    /// Import statement: `import std.io;`
    Import {
        path: Vec<String>,
        span: Span,
    },
}

impl Node {
    pub fn span(&self) -> Span {
        match self {
            Node::Lit(_, span)
            | Node::Binary { span, .. }
            | Node::Paren(_, span)
            | Node::Tuple { span, .. }
            | Node::Call { span, .. }
            | Node::CallGrad { span, .. }
            | Node::CallTensorSum { span, .. }
            | Node::CallTensorMean { span, .. }
            | Node::CallReshape { span, .. }
            | Node::CallExpandDims { span, .. }
            | Node::CallSqueeze { span, .. }
            | Node::CallTranspose { span, .. }
            | Node::CallIndex { span, .. }
            | Node::CallSlice { span, .. }
            | Node::CallSliceStride { span, .. }
            | Node::CallGather { span, .. }
            | Node::CallDot { span, .. }
            | Node::CallMatMul { span, .. }
            | Node::CallTensorRelu { span, .. }
            | Node::CallTensorConv2d { span, .. }
            | Node::Let { span, .. }
            | Node::Assign { span, .. }
            | Node::FnDef { span, .. }
            | Node::Return { span, .. }
            | Node::Block { span, .. }
            | Node::If { span, .. }
            | Node::Import { span, .. } => *span,
        }
    }

    pub fn span_start(&self) -> usize {
        self.span().start()
    }

    pub fn span_end(&self) -> usize {
        self.span().end()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Module {
    pub items: Vec<Node>,
}
