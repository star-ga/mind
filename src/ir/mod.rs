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

pub mod compact;
mod print;
mod verify;

pub use crate::opt::ir_canonical::canonicalize_module;
pub use print::format_ir_module;
pub use verify::{verify_module, IrVerifyError};

/// Errors surfaced by [`load`].
#[derive(Debug)]
pub enum LoadError {
    /// Input bytes are not valid UTF-8 (mic@1/mic@2 are text formats).
    InvalidUtf8(std::str::Utf8Error),
    /// Input did not match a recognised MIC format header.
    UnknownFormat,
    /// mic@1 parser failed.
    Mic1(compact::MicParseError),
    /// mic@2 was detected; use `compact::v2::parse_mic2` directly — that
    /// format produces `Graph`, not [`IRModule`].
    Mic2NotSupportedByLoad,
    /// Binary MIC-B was detected; use `compact::v2::parse_micb` directly.
    MicBNotSupportedByLoad,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::InvalidUtf8(e) => write!(f, "invalid UTF-8: {}", e),
            LoadError::UnknownFormat => f.write_str("unrecognised MIC format header"),
            LoadError::Mic1(e) => write!(f, "mic@1 parse error: {}", e.message),
            LoadError::Mic2NotSupportedByLoad => {
                f.write_str("mic@2 detected — use compact::v2::parse_mic2 directly")
            }
            LoadError::MicBNotSupportedByLoad => {
                f.write_str("MIC-B detected — use compact::v2::parse_micb directly")
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Load an [`IRModule`] from MIC text bytes.
///
/// This is the stable runtime-facing entry point used by `mind-runtime` and
/// other backends to consume pre-compiled IR without re-running the surface
/// parser. The accepted format is **mic@1** (textual IR with explicit node IDs).
///
/// `mic@2` and `MIC-B` are also detected, but those formats produce a
/// different [`Graph`](compact::v2::Graph) type and must be loaded through
/// [`compact::v2::parse_mic2`] / [`compact::v2::parse_micb`] directly.
///
/// # Stability
/// The mic@1 textual form is part of the v0.2.x stability surface — see
/// `docs/versioning.md` and `docs/ir-stability.md`.
pub fn load(data: &[u8]) -> Result<IRModule, LoadError> {
    let text = std::str::from_utf8(data).map_err(LoadError::InvalidUtf8)?;
    match compact::detect_format(data) {
        compact::MicFormat::Mic1 => compact::parse_mic(text).map_err(LoadError::Mic1),
        compact::MicFormat::Mic2 => Err(LoadError::Mic2NotSupportedByLoad),
        compact::MicFormat::MicB => Err(LoadError::MicBNotSupportedByLoad),
        compact::MicFormat::Unknown => Err(LoadError::UnknownFormat),
    }
}

/// Serialize an [`IRModule`] to mic@1 text.
///
/// This is the stable counterpart to [`load`]. Output is deterministic
/// (RFC-0001): the same module always produces byte-identical text.
pub fn save(module: &IRModule) -> String {
    compact::emit_mic(module)
}

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
    ConstF64(ValueId, f64),
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
    /// Backward pass for Conv2d: compute gradient with respect to input.
    /// Given upstream gradient dy (NHWC) and filter (HWIO), computes dx (NHWC).
    Conv2dGradInput {
        dst: ValueId,
        dy: ValueId,             // upstream gradient, NHWC
        filter: ValueId,         // HWIO
        input_shape: [usize; 4], // N, H, W, C - needed to allocate dst
        stride_h: usize,
        stride_w: usize,
        padding: ConvPadding,
    },
    /// Backward pass for Conv2d: compute gradient with respect to filter.
    /// Given upstream gradient dy (NHWC) and input (NHWC), computes dw (HWIO).
    Conv2dGradFilter {
        dst: ValueId,
        input: ValueId,           // NHWC
        dy: ValueId,              // upstream gradient, NHWC
        filter_shape: [usize; 4], // KH, KW, C, O - needed to allocate dst
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
    /// Sparse-known metadata annotation on a value.
    ///
    /// This is a **metadata-only** instruction: it carries sparsity information
    /// through canonicalization without changing evaluation semantics.  The
    /// runtime ignores `SparseAttr` nodes unless the sparse execution path is
    /// explicitly requested.  Actual sparse autodiff lowering is a follow-up.
    ///
    /// Reference: arXiv:2202.04305 (MLIR sparse_tensor dialect approach).
    SparseAttr {
        /// The value being annotated.
        src: ValueId,
        /// Destination SSA id that the annotation shadows.
        dst: ValueId,
        /// Storage layout hint (CSR for v1; others accepted but treated as CSR).
        layout: crate::ast::SparseLayout,
    },
    /// Function definition
    FnDef {
        name: String,
        params: Vec<(String, ValueId)>,
        ret_id: Option<ValueId>,
        body: Vec<Instr>,
        /// Threshold from `[reap_threshold(t)]` attribute, if present.
        reap_threshold: Option<f64>,
    },
    /// Function call
    Call {
        dst: ValueId,
        name: String,
        args: Vec<ValueId>,
    },
    /// Return from function
    Return {
        value: Option<ValueId>,
    },
    /// Function parameter
    Param {
        dst: ValueId,
        name: String,
        index: usize,
    },
    /// Fixed-size array constant: `const NAME: [i64; N] = [v0, v1, …]`
    /// (RFC 0005 Phase 6.2b Gap 2).
    ///
    /// `dst` is the SSA id that holds the base address (i64) of the
    /// read-only constant blob.  `values` carries the literal element
    /// values in declaration order; they are stamped into a const section
    /// by the backend rather than emitted as N individual store instructions.
    ///
    /// `name` is `Some(ident)` for module-level named constants (so that
    /// `IndexAccess` on the identifier can be emitted as a direct load from
    /// the named blob) and `None` for anonymous literals.
    ///
    /// Gated to `std-surface` — default builds never construct this variant.
    #[cfg(feature = "std-surface")]
    ConstArray {
        /// SSA destination: the i64 base-address value of the const blob.
        dst: ValueId,
        /// Symbolic name, present for named `const` declarations.
        name: Option<String>,
        /// Packed element values in declaration order.
        values: Vec<i64>,
    },
    /// Load one element from a fixed-size array constant at a runtime index
    /// (RFC 0005 Phase 6.2b Gap 2).
    ///
    /// Emitted for `arr[idx]` when `arr` is a `[i64; N]` value.  The
    /// backend lowers this to a bounds-checked (or unchecked, Phase 6.2b)
    /// load from the base address.
    ///
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    ArrayLoad {
        /// SSA destination: the loaded i64 element.
        dst: ValueId,
        /// SSA id of the base-address value (produced by `ConstArray`).
        base: ValueId,
        /// SSA id of the runtime index value.
        index: ValueId,
    },
    /// While loop (RFC 0005 Gap 1).
    ///
    /// Carries the SSA id of the condition value and the sequence of IR
    /// instructions that form the loop body. The body is re-lowered on every
    /// iteration; mutable variables are threaded as re-bound SSA values.
    /// Gated to `std-surface` — default builds never construct this variant.
    #[cfg(feature = "std-surface")]
    While {
        /// SSA id of the boolean (i1 / i64) condition value. Re-evaluated by
        /// the header block on each iteration.
        cond_id: ValueId,
        /// Instructions that produce the condition value (re-emitted into the
        /// header block on every MLIR lowering pass).
        cond_instrs: Vec<Instr>,
        /// Instructions forming the loop body (emitted into the body block).
        body: Vec<Instr>,
        /// SSA ids of variables that are mutated in the body and must be
        /// live across the back-edge (threaded as block arguments).
        live_vars: Vec<(String, ValueId)>,
    },
    /// Conditional branch (Phase 6.5 Stage 1a).
    ///
    /// Lowers `if cond { then } else { else }` into separate sub-instruction
    /// streams so the MLIR backend can emit `scf.if` / `cf.cond_br` without
    /// placing a `func.return` in the middle of a basic block.
    ///
    /// `dst` receives the result value of whichever branch was taken.
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    If {
        /// SSA id of the boolean (i1 / i64) condition value.
        cond_id: ValueId,
        /// Instructions that produce the condition value.
        cond_instrs: Vec<Instr>,
        /// Instructions forming the then-branch.
        then_instrs: Vec<Instr>,
        /// SSA id of the then-branch result value (last value produced).
        then_result: ValueId,
        /// Instructions forming the else-branch (may be empty — unit i64=0).
        else_instrs: Vec<Instr>,
        /// SSA id of the else-branch result value.
        else_result: ValueId,
        /// SSA id that receives the selected branch result in the outer scope.
        dst: ValueId,
        /// Variable bindings produced in either branch and visible after the
        /// if (Gap C: let bindings threaded back to outer fn_env).
        branch_bindings: Vec<(String, ValueId)>,
    },
    /// RFC 0006 Track B (increment 1) — load `lanes` contiguous f32 values
    /// from a heap address into a SIMD vector value.
    ///
    /// `base` is an i64 value holding the opaque base address (Option-C
    /// ABI from RFC 0005 P0a). `offset` is an i64 value holding the byte
    /// offset added to `base` before the load. `lanes` is the statically
    /// known vector width (8 for the AVX2-class default).
    ///
    /// Lowers to MLIR `vector`-dialect memory access: an `llvm.inttoptr`
    /// of the address plus an `llvm.getelementptr` byte offset, then a
    /// vector-typed `llvm.load` of `vector<lanes x f32>`. LLVM's target
    /// legalisation maps this to the host SIMD width (AVX2 / AVX-512 /
    /// NEON / SVE2 / NVPTX) with no per-target code in mindc and no
    /// runtime-support C shim — the Track B thesis-pure property.
    ///
    /// Gated to `std-surface` — default builds never construct this variant.
    #[cfg(feature = "std-surface")]
    VecLoad {
        /// SSA destination: the loaded `vector<lanes x f32>` value.
        dst: ValueId,
        /// SSA id of the i64 base-address value.
        base: ValueId,
        /// SSA id of the i64 byte-offset value.
        offset: ValueId,
        /// Statically known SIMD lane count.
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 1) — fused multiply-add across lanes:
    /// `dst = a * b + acc`, element-wise on `vector<lanes x f32>`.
    ///
    /// Lowers to the MLIR `vector.fma` op, which `convert-vector-to-llvm`
    /// turns into the `llvm.intr.fmuladd` intrinsic — a single hardware
    /// FMA per lane group on targets that have one.
    ///
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    VecFma {
        /// SSA destination: the `vector<lanes x f32>` result.
        dst: ValueId,
        /// SSA id of the first `vector<lanes x f32>` multiplicand.
        a: ValueId,
        /// SSA id of the second `vector<lanes x f32>` multiplicand.
        b: ValueId,
        /// SSA id of the `vector<lanes x f32>` accumulator addend.
        acc: ValueId,
        /// Statically known SIMD lane count (must match the operands).
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 1) — horizontal sum of a SIMD vector
    /// down to a single f32 scalar.
    ///
    /// Lowers to MLIR `vector.reduction <add>`, which becomes the
    /// `llvm.intr.vector.reduce.fadd` intrinsic. The reduction is the
    /// tree-shaped pairwise sum LLVM emits for the target; it is *not*
    /// bit-identical to a sequential scalar accumulation (documented in
    /// the numerical contract — f32 stays within 1e-4 relative of the
    /// f64 oracle, exactly as Track A's AVX2 path does).
    ///
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    VecReduceAdd {
        /// SSA destination: the reduced f32 scalar (as an i64-packed value).
        dst: ValueId,
        /// SSA id of the `vector<lanes x f32>` source.
        src: ValueId,
        /// Statically known SIMD lane count of `src`.
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 2) — store `lanes` contiguous f32 values
    /// from a SIMD vector value back to a heap address.
    ///
    /// Symmetric with [`Instr::VecLoad`]: `base` is an i64 opaque base
    /// address (Option-C ABI), `offset` is an i64 byte offset, `src` is the
    /// `vector<lanes x f32>` SSA value to store. Lowers to an
    /// `llvm.inttoptr` + byte `llvm.getelementptr` + vector-typed
    /// `llvm.store`. Enables vectorised output kernels (e.g. a future
    /// vectorised `matmul_rmajor` writing its row result back through the
    /// vector path). Produces no SSA value.
    ///
    /// Gated to `std-surface` — default builds never construct this variant.
    #[cfg(feature = "std-surface")]
    VecStore {
        /// SSA id of the `vector<lanes x f32>` value to store.
        src: ValueId,
        /// SSA id of the i64 base-address value.
        base: ValueId,
        /// SSA id of the i64 byte-offset value.
        offset: ValueId,
        /// Statically known SIMD lane count.
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 2) — load `lanes` contiguous i32 Q16.16
    /// fixed-point values from a heap address into a SIMD vector value.
    ///
    /// The i32 sibling of [`Instr::VecLoad`]; same opaque-i64-address
    /// Option-C ABI. Used by the Q16.16 vector dot path. Lowers to
    /// `llvm.inttoptr` + byte `llvm.getelementptr` + vector-typed
    /// `llvm.load` of `vector<lanes x i32>`.
    ///
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    VecLoadI32 {
        /// SSA destination: the loaded `vector<lanes x i32>` value.
        dst: ValueId,
        /// SSA id of the i64 base-address value.
        base: ValueId,
        /// SSA id of the i64 byte-offset value.
        offset: ValueId,
        /// Statically known SIMD lane count.
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 2) — Q16.16 fused widening
    /// multiply-shift-accumulate across lanes.
    ///
    /// `dst[i] = acc[i] + ((sext_i64(a[i]) * sext_i64(b[i])) >> 16)`,
    /// element-wise, with an *arithmetic* (sign-preserving) right shift —
    /// the exact per-element operation the Track A scalar oracle
    /// `mind_blas_dot_q16_scalar` performs (`acc += prod >> 16`). `a` and
    /// `b` are `vector<lanes x i32>`; `acc` and `dst` are
    /// `vector<lanes x i64>`.
    ///
    /// Q16.16 integer reduction is associative, so accumulating into i64
    /// lanes and summing the lanes afterwards (`VecReduceAddI64`) yields a
    /// result **byte-identical** to the sequential scalar oracle at every
    /// length. This is the cross-arch bit-identity contract (task #57)
    /// extended to the native vector path — unlike the f32
    /// `VecReduceAdd`, this path is NOT lossy.
    ///
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    VecMulAddQ16 {
        /// SSA destination: the `vector<lanes x i64>` accumulator result.
        dst: ValueId,
        /// SSA id of the first `vector<lanes x i32>` Q16.16 multiplicand.
        a: ValueId,
        /// SSA id of the second `vector<lanes x i32>` Q16.16 multiplicand.
        b: ValueId,
        /// SSA id of the `vector<lanes x i64>` accumulator addend.
        acc: ValueId,
        /// Statically known SIMD lane count (must match the operands).
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 2) — horizontal sum of an i64 SIMD
    /// vector down to a single i64 scalar.
    ///
    /// Lowers to MLIR `vector.reduction <add>` over `vector<lanes x i64>`,
    /// which becomes `llvm.intr.vector.reduce.add`. Integer addition is
    /// associative, so the reduction is bit-identical to a sequential
    /// scalar accumulation regardless of the lane-grouping LLVM chooses —
    /// this is what makes the Q16.16 vector path satisfy the task-#57
    /// cross-arch bit-identity gate.
    ///
    /// Gated to `std-surface`.
    #[cfg(feature = "std-surface")]
    VecReduceAddI64 {
        /// SSA destination: the reduced i64 scalar.
        dst: ValueId,
        /// SSA id of the `vector<lanes x i64>` source.
        src: ValueId,
        /// Statically known SIMD lane count of `src`.
        lanes: usize,
    },
    /// RFC 0010 Phase A — declare an `extern "C"` function symbol.
    ///
    /// Carries the function signature for MLIR lowering: the name, arity
    /// (number of concrete parameters), whether the last argument is a
    /// varargs sentinel, and a simple type-tag vector for the parameters
    /// and return type. Phase A supports only i64 (covers all integers and
    /// pointer-as-i64) and f64 scalars in the MLIR emission path; the
    /// caller is responsible for inserting any necessary bitcasts.
    ///
    /// This instruction carries no SSA `dst`; it is a pure declaration.
    /// Gated to `std-surface` — default builds never construct this variant.
    #[cfg(feature = "std-surface")]
    ExternFnDecl {
        /// The C symbol name (used verbatim in `llvm.func @name`).
        name: String,
        /// MLIR type strings for each concrete parameter.
        /// Phase A uses `"i64"` for all integer/pointer types and `"f64"` for f64.
        /// Phase B uses `"!llvm.ptr"` for raw pointer types and per-SysV struct types.
        /// Phase C uses Win64-classified types when `callconv` is `Win64`.
        param_types: Vec<String>,
        /// MLIR type string for the return type, or `None` for void.
        ret_type: Option<String>,
        /// `true` when `...` varargs were declared.
        is_varargs: bool,
        /// RFC 0010 Phase B — optional type hints for variadic arguments beyond
        /// the declared concrete parameters. When a call site passes more
        /// arguments than `param_types.len()`, the extra argument types are
        /// looked up from this list (by index offset from `param_types.len()`),
        /// falling back to `"i64"` for positions beyond `vararg_hints.len()`.
        ///
        /// This supports the common printf use-cases: passing `!llvm.ptr`
        /// for `*const i8` string args, `i64` for integer args, `f64` for
        /// floating-point args. Phase A always used `"i64"` for all extras;
        /// Phase B makes the per-position type precise.
        vararg_hints: Vec<String>,
        /// RFC 0010 Phase C — calling convention for this declaration.
        ///
        /// Defaults to `CallConv::SysV` for all Phase A/B declarations.
        /// `CallConv::Win64` causes the MLIR lowerer to emit
        /// `cconv = #llvm.cconv<win64cc>` on `llvm.func` and `llvm.call`.
        callconv: crate::ast::CallConv,
    },
}

pub(crate) fn instruction_dst(instr: &Instr) -> Option<ValueId> {
    match instr {
        Instr::ConstI64(dst, ..)
        | Instr::ConstF64(dst, ..)
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
        | Instr::Conv2dGradInput { dst, .. }
        | Instr::Conv2dGradFilter { dst, .. }
        | Instr::Index { dst, .. }
        | Instr::Slice { dst, .. }
        | Instr::Gather { dst, .. }
        | Instr::Call { dst, .. }
        | Instr::Param { dst, .. }
        | Instr::SparseAttr { dst, .. } => Some(*dst),
        Instr::Output(_) | Instr::FnDef { .. } | Instr::Return { .. } => None,
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl { .. } => None,
        #[cfg(feature = "std-surface")]
        Instr::While { .. } => None,
        #[cfg(feature = "std-surface")]
        Instr::ConstArray { dst, .. } | Instr::ArrayLoad { dst, .. } => Some(*dst),
        #[cfg(feature = "std-surface")]
        Instr::If { dst, .. } => Some(*dst),
        #[cfg(feature = "std-surface")]
        Instr::VecLoad { dst, .. }
        | Instr::VecFma { dst, .. }
        | Instr::VecReduceAdd { dst, .. } => Some(*dst),
        #[cfg(feature = "std-surface")]
        Instr::VecStore { .. } => None,
        #[cfg(feature = "std-surface")]
        Instr::VecLoadI32 { dst, .. }
        | Instr::VecMulAddQ16 { dst, .. }
        | Instr::VecReduceAddI64 { dst, .. } => Some(*dst),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Modulo (remainder). Phase 10.6.
    Mod,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    /// Bitwise AND (`&`). Phase 6.5 Stage 1a.
    #[cfg(feature = "std-surface")]
    BitAnd,
    /// Bitwise OR (`|`). Phase 6.5 Stage 1a.
    #[cfg(feature = "std-surface")]
    BitOr,
    /// Bitwise XOR (`^`). Phase 6.5 Stage 1a.
    #[cfg(feature = "std-surface")]
    BitXor,
    /// Left shift (`<<`). Phase 6.5 Stage 1a.
    #[cfg(feature = "std-surface")]
    Shl,
    /// Arithmetic right shift (`>>`). Phase 6.5 Stage 1a.
    #[cfg(feature = "std-surface")]
    Shr,
}

#[derive(Debug, Clone)]
pub struct IRModule {
    pub instrs: Vec<Instr>,
    pub next_id: usize,
    /// Names declared in an `export { ... }` block, populated by the
    /// AST -> IR lowering pass. Consumed by the C-ABI codegen pass under
    /// `feature = "ffi-c-user"` (RFC 0002, deliverable 2). Empty in the
    /// default code path; readers must never assume membership without
    /// the feature flag enabled. Kept on `IRModule` rather than in a
    /// side-table so mic@1 round-trip preserves the export set.
    pub exports: std::collections::HashSet<String>,
    /// RFC 0005 P0e Step 1 — struct schema registry. Maps a struct name
    /// to its canonical field-name order (as declared in `Node::StructDef`).
    /// Populated by the lowering pass when it visits a `StructDef` and
    /// read by the `StructLit` arm to reorder literal fields into canonical
    /// order before emitting the heap-record stores. `BTreeMap` keeps
    /// iteration deterministic so the mic@1 round-trip + the model_hash
    /// stay stable. Gated; the default build never populates this.
    #[cfg(feature = "std-surface")]
    pub struct_defs: std::collections::BTreeMap<String, Vec<String>>,
    /// RFC 0005 Phase 6.2b Gap 2 — const-array data registry.
    /// Maps a const-array name to its element values so that fn bodies can
    /// re-emit a `ConstArray` node when they reference the name.  This is
    /// required because fn bodies lower into a fresh `IRModule` with an
    /// independent SSA counter, so outer module ValueIds cannot be reused.
    /// `BTreeMap` for deterministic iteration. Gated.
    #[cfg(feature = "std-surface")]
    pub const_array_defs: std::collections::BTreeMap<String, Vec<i64>>,
    /// RFC 0010 Phase B — `#[repr(C)]` struct registry.
    ///
    /// Maps a struct name to its field types (as `crate::ast::TypeAnn`),
    /// in declaration order, for structs annotated with `#[repr(C)]`.
    /// Populated by the AST→IR lowering pass when it sees a `StructDef`
    /// with a `repr(C)` attribute.  Consumed by `extern_type_to_mlir` and
    /// `check_extern_type` to classify Named types that appear in `extern "C"`
    /// signatures as legal C-ABI types and to derive SysV passing convention.
    ///
    /// `BTreeMap` for deterministic ordering. Gated.
    #[cfg(feature = "std-surface")]
    pub repr_c_structs: std::collections::BTreeMap<String, Vec<crate::ast::TypeAnn>>,
}

impl IRModule {
    pub fn new() -> Self {
        Self {
            instrs: Vec::new(),
            next_id: 0,
            exports: std::collections::HashSet::new(),
            #[cfg(feature = "std-surface")]
            struct_defs: std::collections::BTreeMap::new(),
            #[cfg(feature = "std-surface")]
            const_array_defs: std::collections::BTreeMap::new(),
            #[cfg(feature = "std-surface")]
            repr_c_structs: std::collections::BTreeMap::new(),
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

/// Placeholder MLIR lowering stub for testing.
///
/// This is a stub that produces a minimal MLIR module skeleton. Real MLIR
/// lowering is provided by the `mlir-lowering` feature and `mind-runtime`.
#[cfg(feature = "mlir")]
pub fn lower_placeholder(input: &str) -> String {
    format!(
        r#"mlir.module {{
  // placeholder IR for: {}
}}"#,
        input
    )
}
