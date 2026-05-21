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
use crate::types::ShapeDim;

/// Storage layout for a sparse tensor.
///
/// v1 ships CSR as the only concrete layout; the remaining variants are
/// syntactically accepted and round-trip through the AST but the runtime
/// resolves them to the appropriate storage format at load time.
///
/// Reference: arXiv:2202.04305 (MLIR sparse_tensor dialect taxonomy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseLayout {
    /// Compressed Sparse Row — row-major, two-array (indptr + indices).
    Csr,
    /// Compressed Sparse Column — column-major variant of CSR.
    Csc,
    /// Coordinate format — explicit (row, col, val) triples.
    Coo,
    /// Block Sparse Row — tiled extension of CSR for structured sparsity.
    Bsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Modulo (remainder). Phase 10.6 — needed by numeric kernels that
    /// validate divisibility (e.g. channel count vs. group count in
    /// normalization layers).
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
    /// Borrowed slice `&[T]` or `&mut [T]` (Phase 10.6). Used in fn
    /// signatures to pass contiguous buffers without copying. The type
    /// checker treats this as a sized run of T.
    Slice {
        mutable: bool,
        element: Box<TypeAnn>,
    },
    /// Fixed-size array `[T; N]` (Phase 10.6). Used for compile-time
    /// LUT tables (e.g. `TANH_TABLE: [Q16_16; 256]`) where the count is
    /// part of the type.
    Array {
        element: Box<TypeAnn>,
        length: u32,
    },
    /// Borrowed reference to a single value `&T` or `&mut T` (Phase 10.6).
    /// Distinct from `Slice` (which is `&[T]`); used to pass structs by
    /// reference without copying.
    Ref {
        mutable: bool,
        target: Box<TypeAnn>,
    },
    /// Generic type application `Name<A, B, ...>` (Phase 10.6).
    /// Examples: `Vec<i32>`, `Result<T, E>`, `Option<u32>`. mindc records
    /// the head identifier and the type arguments; structural resolution
    /// defers to the type checker.
    Generic {
        name: String,
        args: Vec<TypeAnn>,
    },
    /// Tuple type `(T, U)` (Phase 10.6). Used in fn signatures that
    /// return multiple values, e.g. `fn defaults() -> (i32, u32)`.
    Tuple {
        elements: Vec<TypeAnn>,
    },
    /// Sparse tensor type surface: `tensor<sparse[csr], q16_16>`.
    ///
    /// `layout` names the storage format; `element` is the scalar element
    /// type; `shape` uses the existing `ShapeDim` encoding (symbolic or
    /// concrete). The runtime resolves physical layout at load time —
    /// `valuetype_from_ann` intentionally returns `None` for this variant
    /// in v1 so callers fall through to the runtime resolver.
    ///
    /// Reference: arXiv:2202.04305 (MLIR sparse_tensor dialect).
    SparseTensor {
        layout: SparseLayout,
        element: Box<TypeAnn>,
        shape: Vec<ShapeDim>,
    },
    /// Raw C pointer: `*const T` or `*mut T`.
    ///
    /// RFC 0010 Phase A — extern "C" ABI surface. Raw pointers are only
    /// constructible inside `unsafe` blocks; in Phase A they are accepted
    /// in `extern "C"` function signatures and lowered to `!llvm.ptr`.
    /// The pointee type is recorded for documentation/future phases but is
    /// not used in Phase A lowering (all pointers lower to opaque `!llvm.ptr`).
    RawPtr {
        mutable: bool,
        pointee: Box<TypeAnn>,
    },
    /// Callback function pointer: `extern "C" fn(T, U) -> R`.
    ///
    /// RFC 0010 Phase B — used to declare callback parameters in `extern "C"`
    /// function signatures, e.g. the `compar` parameter of `qsort`.
    /// Lowered to `!llvm.ptr` (opaque function pointer) in the MLIR emission.
    /// The parameter types and return type are stored for type-checking that
    /// callback signatures obey Phase B rules (all-Copy params + return).
    FnPtr {
        params: Vec<TypeAnn>,
        ret: Option<Box<TypeAnn>>,
    },
}

/// Calling convention tag for an `extern "C"` block (RFC 0010 Phase A).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallConv {
    /// Platform-default C ABI — resolves to `.sysv` on Linux/macOS x86_64
    /// and `.win64` on Windows x86_64 when no explicit tag is present.
    C,
    /// System V AMD64 ABI.
    SysV,
    /// Microsoft x64 calling convention.
    Win64,
    /// ARM Architecture Procedure Call Standard (AArch64).
    Aapcs,
}

/// A single function declaration inside an `extern "C"` block.
///
/// RFC 0010 Phase A. The body is absent — extern declarations have no
/// body. `is_unsafe` mirrors the `unsafe fn` vs `safe fn` annotation;
/// `is_varargs` is set when `...` appears after the last concrete parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternFn {
    pub is_unsafe: bool,
    pub name: String,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeAnn>,
    pub is_varargs: bool,
    pub span: Span,
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
    ///
    /// `reap_threshold` carries the value from a preceding
    /// `[reap_threshold(0.5)]` attribute (REAP MoE pruning, arXiv:2510.13999).
    /// `None` means the attribute was absent; `Some(t)` means the function is
    /// annotated for compile-time expert pruning with threshold `t ∈ [0.0, 1.0)`.
    ///
    /// `is_test` is set when the function carries a `[test]` attribute
    /// (RFC 0008 Phase B). A test function must have zero parameters and return
    /// type `()` or `bool`. The test runner collects all `is_test = true` fns.
    FnDef {
        /// Whether the `pub` visibility modifier was present in source.
        is_pub: bool,
        /// Whether the `[test]` attribute was present (RFC 0008 Phase B).
        is_test: bool,
        name: String,
        params: Vec<Param>,
        ret_type: Option<TypeAnn>,
        body: Vec<Node>,
        /// `[reap_threshold(t)]` annotation — `None` when absent.
        reap_threshold: Option<f64>,
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
    /// While loop: `while cond { body }` (RFC 0005 Gap 1).
    ///
    /// Lowers to a header-block + body-block basic-block loop in MLIR with
    /// zero per-iteration stack allocation. `break` / `continue` are
    /// follow-on items (Gap 1 ships the loop primitive only).
    #[cfg(feature = "std-surface")]
    While {
        cond: Box<Node>,
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
        /// Whether the `pub` visibility modifier was present in source.
        is_pub: bool,
        name: String,
        fields: Vec<Field>,
        attrs: Vec<Attribute>,
        span: Span,
    },
    /// Enum declaration: `enum Name { Variant, Variant(T) }`
    /// Phase 10.5 Tier-2.
    EnumDef {
        /// Whether the `pub` visibility modifier was present in source.
        is_pub: bool,
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
    /// Phase 10.6 — used to construct aggregate values. Type-checker
    /// resolves the name against a StructDef in scope.
    StructLit {
        name: String,
        fields: Vec<StructLitField>,
        span: Span,
    },
    /// Index access expression: `receiver[index]` (Phase 10.6). Distinct
    /// from `Node::CallIndex` (which is for `tensor.index(...)` builtin
    /// dispatch); this is the generic indexing operator used on slices,
    /// arrays, and Vec values.
    IndexAccess {
        receiver: Box<Node>,
        index: Box<Node>,
        span: Span,
    },
    /// Indexed assignment statement: `receiver[index] = value` (Phase 10.6).
    /// Used in backward / gradient kernels to write through a mutable
    /// slice receiver.
    IndexAssign {
        receiver: Box<Node>,
        index: Box<Node>,
        value: Box<Node>,
        span: Span,
    },
    /// Field assignment statement: `receiver.field = value` (Phase 10.6).
    /// Used to mutate struct fields through a `&mut Foo` receiver.
    FieldAssign {
        receiver: Box<Node>,
        field: String,
        value: Box<Node>,
        span: Span,
    },
    /// Match expression: `match value { Pat => body, ... }` (Phase 10.7).
    ///
    /// Lowered to a chain of if-else in v1 IR — no new IR opcode required.
    /// Exhaustiveness checking is deferred; a non-blocking hint is emitted
    /// when the scrutinee is a known-finite enum and no wildcard arm is
    /// present.
    Match {
        scrutinee: Box<Node>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    /// Reference-taking expression: `&expr` or `&mut expr` (Phase 10.7).
    ///
    /// Symmetric with the `&T` / `&mut T` type annotations already in
    /// `TypeAnn::Ref`. The type-checker annotates the result with `&T` /
    /// `&mut T`. Lifetime tracking is out of scope for v1; the compiler
    /// propagates the `&` / `&mut` tag as metadata only.
    Ref {
        mutable: bool,
        inner: Box<Node>,
        span: Span,
    },
    /// `extern "C" [callconv(.x)] { fn decls... }` block (RFC 0010 Phase A).
    ///
    /// Phase A accepts the syntax, stores the AST, type-checks signatures
    /// for Copy-only types, and lowers calls to `llvm.call`. Win64 and
    /// AAPCS parse but reuse the default `.c` lowering until Phase C/D.
    ExternBlock {
        callconv: CallConv,
        fns: Vec<ExternFn>,
        span: Span,
    },
}

/// One arm of a `match` expression: `pattern => body`.
/// Phase 10.7.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Node,
    pub span: Span,
}

/// Pattern for a `match` arm (Phase 10.7).
///
/// v1 ships four pattern kinds:
/// - `EnumVariant` — `Mode::On`, `Result::Ok(x)`.
/// - `Literal`     — `0`, `true`, `"hello"`.
/// - `Ident`       — bare binding `x` (always matches, binds the name).
/// - `Wildcard`    — `_` (always matches, binds nothing).
///
/// Exhaustiveness checking is advisory only in v1.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Qualified enum variant, e.g. `Mode::On` or `Result::Ok(x)`.
    EnumVariant {
        /// Full dotted+`::` path, e.g. `"Mode::On"` or `"config.Mode::On"`.
        path: String,
        /// Payload sub-patterns inside `( ... )`, if present.
        args: Vec<Pattern>,
    },
    /// Literal constant: integer, float, bool, or string.
    Literal(Literal),
    /// Bare identifier binding — matches anything, binds the name.
    Ident(String),
    /// Wildcard `_` — matches anything, binds nothing.
    Wildcard,
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
    /// Whether the `pub` visibility modifier was present in source.
    pub is_pub: bool,
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
            | Node::IndexAccess { span, .. }
            | Node::IndexAssign { span, .. }
            | Node::FieldAssign { span, .. }
            | Node::EnumDef { span, .. }
            | Node::Assert { span, .. }
            | Node::As { span, .. }
            | Node::Logical { span, .. }
            | Node::Bitwise { span, .. }
            | Node::Match { span, .. }
            | Node::Ref { span, .. }
            | Node::ExternBlock { span, .. } => *span,
            #[cfg(feature = "std-surface")]
            Node::While { span, .. } => *span,
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
