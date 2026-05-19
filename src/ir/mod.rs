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
        Instr::While { .. } => None,
        #[cfg(feature = "std-surface")]
        Instr::ConstArray { dst, .. } | Instr::ArrayLoad { dst, .. } => Some(*dst),
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
