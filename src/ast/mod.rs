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

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Modulo (remainder). Phase 10.6 — needed by rfn-mind/src/groupnorm.mind
    /// to validate channel count is divisible by group count.
    Mod,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Logical binary operator (Phase 10.5 Tier-1).
/// Kept separate from `BinOp` so the existing arithmetic/comparison
/// match arms across the compiler stay closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
}

/// Bitwise binary operator (Phase 10.5 Tier-1).
/// Held separate from `BinOp` for the same matching-stability reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOp {
    /// `|` bitwise or
    Or,
    /// `&` bitwise and
    And,
    /// `^` xor
    Xor,
    /// `<<` shift left
    Shl,
    /// `>>` shift right
    Shr,
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
    /// User-defined type name (alias, struct, enum). Resolved later in typecheck.
    /// Phase 10.5 Tier-1.
    Named(String),
    /// Unsigned 32-bit integer (Phase 10.5 Tier-2).
    ScalarU32,
    /// Borrowed slice `&[T]` or `&mut [T]` (Phase 10.6). Used heavily in
    /// rfn-mind fn signatures to pass weight buffers without copying.
    /// The type checker treats this as a sized contiguous run of T.
    Slice {
        mutable: bool,
        element: Box<TypeAnn>,
    },
    /// Fixed-size array `[T; N]` (Phase 10.6). Used in rfn-mind for LUT
    /// tables (TANH_TABLE: [Q16_16; 256]) where the count is part of
    /// the type and known at compile time.
    Array {
        element: Box<TypeAnn>,
        length: u32,
    },
    /// Borrowed reference to a single value `&T` or `&mut T` (Phase 10.6).
    /// Distinct from `Slice` (which is `&[T]`); used to pass structs by
    /// reference without copying (e.g. `&memory.MemoryBank`,
    /// `&ExitController`).
    Ref {
        mutable: bool,
        target: Box<TypeAnn>,
    },
}

/// Function parameter: `name: type`
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: TypeAnn,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
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
    CallTensorRand {
        shape: Vec<usize>,
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
    /// Array literal: `[1.0, 2.0, 3.0]`
    ArrayLit {
        elements: Vec<Node>,
        span: Span,
    },
    /// For loop: `for i in 0..N { body }`
    For {
        var: String,
        start: Box<Node>,
        end: Box<Node>,
        body: Vec<Node>,
        span: Span,
    },
    /// Print statement: `print("msg", expr)`
    Print {
        args: Vec<Node>,
        span: Span,
    },
    /// Unary negation: `-expr`
    Neg {
        operand: Box<Node>,
        span: Span,
    },
    /// Method call
    MethodCall {
        receiver: Box<Node>,
        method: String,
        args: Vec<Node>,
        span: Span,
    },
    /// Field access
    FieldAccess {
        receiver: Box<Node>,
        field: String,
        span: Span,
    },
    /// Compile-time constant: `const NAME: type = expr`
    /// Phase 10.5 Tier-1.
    Const {
        name: String,
        ty: Option<TypeAnn>,
        value: Box<Node>,
        attrs: Vec<Attribute>,
        span: Span,
    },
    /// Type alias: `type X = Y`
    /// Phase 10.5 Tier-1.
    TypeAlias {
        name: String,
        target: TypeAnn,
        attrs: Vec<Attribute>,
        span: Span,
    },
    /// Export block: `export { name1, name2 }`
    /// Phase 10.5 Tier-1.
    Export {
        names: Vec<String>,
        span: Span,
    },
    /// Struct declaration: `struct Name { f: T, g: U }`
    /// Phase 10.5 Tier-2.
    StructDef {
        name: String,
        fields: Vec<Field>,
        attrs: Vec<Attribute>,
        span: Span,
    },
    /// Enum declaration: `enum Name { Variant, Variant(T) }`
    /// Phase 10.5 Tier-2.
    EnumDef {
        name: String,
        variants: Vec<EnumVariant>,
        attrs: Vec<Attribute>,
        span: Span,
    },
    /// `assert cond[, "msg"]` — runtime check, no return value.
    /// Phase 10.5 stretch.
    Assert {
        cond: Box<Node>,
        msg: Option<String>,
        span: Span,
    },
    /// `expr as type` — explicit cast.
    /// Phase 10.5 stretch.
    As {
        expr: Box<Node>,
        ty: TypeAnn,
        span: Span,
    },
    /// Logical binary expression: `a && b`, `a || b`.
    /// Phase 10.5 Tier-1. Held separate from `Node::Binary` so existing
    /// numeric/tensor binop matches remain exhaustive without churn.
    Logical {
        op: LogicalOp,
        left: Box<Node>,
        right: Box<Node>,
        span: Span,
    },
    /// Bitwise binary expression: `a | b`, `a & b`, `a ^ b`, `a << b`, `a >> b`.
    /// Phase 10.5 Tier-1.
    Bitwise {
        op: BitOp,
        left: Box<Node>,
        right: Box<Node>,
        span: Span,
    },
    /// Struct literal expression: `Name { field: value, field: value }`.
    /// Phase 10.6 — used by rfn-mind to return aggregate values
    /// (`PartialPair { da: dy, db: dy }` etc.). Type-checker resolves
    /// the name against a StructDef in scope.
    StructLit {
        name: String,
        fields: Vec<StructLitField>,
        span: Span,
    },
}

/// A `field: value` pair inside a struct literal expression.
/// Phase 10.6.
#[derive(Debug, Clone, PartialEq)]
pub struct StructLitField {
    pub name: String,
    pub value: Node,
    pub span: Span,
}

/// Attribute metadata, e.g. `[protection]`, `[test]`, `[bench]`.
/// Public mindc records but does not interpret these (Phase 10.5).
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<String>,
    pub span: Span,
}

/// Field of a struct declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: TypeAnn,
    pub span: Span,
}

/// Variant of an enum declaration. Tier-2 ships unit + tuple variants.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub payload: Vec<TypeAnn>,
    pub span: Span,
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
            | Node::CallTensorRand { span, .. }
            | Node::CallTensorConv2d { span, .. }
            | Node::Let { span, .. }
            | Node::Assign { span, .. }
            | Node::FnDef { span, .. }
            | Node::Return { span, .. }
            | Node::Block { span, .. }
            | Node::If { span, .. }
            | Node::Import { span, .. }
            | Node::ArrayLit { span, .. }
            | Node::For { span, .. }
            | Node::Print { span, .. }
            | Node::Neg { span, .. }
            | Node::MethodCall { span, .. }
            | Node::FieldAccess { span, .. }
            | Node::Const { span, .. }
            | Node::TypeAlias { span, .. }
            | Node::Export { span, .. }
            | Node::StructDef { span, .. }
            | Node::StructLit { span, .. }
            | Node::EnumDef { span, .. }
            | Node::Assert { span, .. }
            | Node::As { span, .. }
            | Node::Logical { span, .. }
            | Node::Bitwise { span, .. } => *span,
        }
    }

    pub fn span_start(&self) -> usize {
        self.span().start()
    }

    pub fn span_end(&self) -> usize {
        self.span().end()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Module {
    pub items: Vec<Node>,
}
