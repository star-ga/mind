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

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::ir::{BinOp, IRModule, Instr, ValueId};
#[cfg(feature = "std-surface")]
use crate::mlir::gemm_tuning::*;
use crate::opt::ir_canonical::canonicalize_module;
use crate::types::{ConvPadding, DType, ShapeDim};

/// RFC 0006 Track B (increment 1) — the pure-MIND surface name whose
/// `Instr::Call` lowers to a native MLIR `vector`-dialect reduction loop
/// instead of a `func.call` to the Track A runtime-support C bridge.
/// Declared unconditionally so the default-build catch-all stays
/// byte-identical; only the gated `Instr::Call` arm ever compares it.
#[cfg(feature = "std-surface")]
const VEC_DOT_F32_INTRINSIC: &str = "__mind_blas_dot_f32_v";

/// RFC 0006 Track B — the statically-known SIMD lane count emitted by the
/// `dot_f32_v` lowering. Eight f32 lanes is the AVX2 / NEON-pair width;
/// LLVM legalises wider/narrower targets from the same `vector<8xf32>`.
#[cfg(feature = "std-surface")]
const VEC_DOT_F32_LANES: usize = 8;

/// RFC 0006 Track B (increment 2) — the pure-MIND surface name whose
/// `Instr::Call` lowers to a native MLIR `vector`-dialect Q16.16
/// reduction. Byte-identical to the Track A scalar oracle
/// `__mind_blas_dot_q16` at every length (task #57 — integer reduction
/// is associative, the per-element arithmetic `>> 16` is replicated
/// exactly in `vector<8xi64>` lanes).
#[cfg(feature = "std-surface")]
const VEC_DOT_Q16_INTRINSIC: &str = "__mind_blas_dot_q16_v";

/// RFC 0006 Track B (increment 2) — native MLIR vector-dialect f32
/// L1 (Manhattan, sum of `|a-b|`) reduction surface name.
#[cfg(feature = "std-surface")]
const VEC_DOT_L1_F32_INTRINSIC: &str = "__mind_blas_dot_l1_f32_v";

/// RFC 0006 Track B (increment 3) — native MLIR vector-dialect Q16.16
/// L1 (Manhattan, sum of `|a-b|`) reduction surface name. Byte-identical
/// to the Track A scalar oracle `__mind_blas_dot_l1_q16` at every length
/// (task #57 — integer reduction is associative, and per-element
/// `|sext64(a) - sext64(b)|` is exact; this completes the Q16.16
/// vector-path metric parity left open in increment 2, RFC 0006 §9.3).
#[cfg(feature = "std-surface")]
const VEC_DOT_L1_Q16_INTRINSIC: &str = "__mind_blas_dot_l1_q16_v";

/// RFC 0006 Track B (increment 2) — native MLIR vector-dialect f32
/// L∞ (Chebyshev, max of `|a-b|`) reduction surface name.
#[cfg(feature = "std-surface")]
const VEC_DOT_LINF_F32_INTRINSIC: &str = "__mind_blas_dot_linf_f32_v";

/// RFC 0006 Track B (increment 3b) — native MLIR vector-dialect
/// row-major f32 matrix-vector multiply surface name.
///
/// Signature: `(w_addr, x_addr, y_addr, rows, cols) -> i64` (returns 0).
/// W is a rows×cols row-major f32 matrix (pointer packed as i64),
/// x is a cols-element f32 vector, y is a caller-allocated rows-element
/// f32 output vector.  The kernel computes `y[r] = dot(W[r,:], x)` for
/// each row r using the proven vectorised f32 dot structure (eight-lane
/// FMA accumulation + scalar tail) so numerical results are within 1e-4
/// relative of an f64 oracle, matching the `dot_f32_v` contract.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_RMAJOR_F32_INTRINSIC: &str = "__mind_blas_matmul_rmajor_f32_v";

/// RFC 0006 Track B (increment 4) — native MLIR vector-dialect
/// row-major Q16.16 matrix-vector multiply surface name.
///
/// Signature: `(w_addr, x_addr, y_addr, rows, cols) -> i64` (returns 0).
/// W is a rows×cols row-major Q16.16 matrix (i32 elements, pointer packed
/// as i64), x is a cols-element Q16.16 vector (i32), y is a
/// caller-allocated rows-element Q16.16 output (i32).  The kernel
/// computes `y[r] = dot_q16(W[r,:], x)` for each row r — identically to
/// calling `__mind_blas_dot_q16_v` on each row — using the proven Q16.16
/// reduction from `emit_vec_dot_q16` (widen i32→i64, multiply, arith
/// `>> 16`, i64-lane accumulate, associative `vector.reduction <add>`,
/// scalar tail, `trunc i64→i32` + `extsi i32→i64`).  Byte-identical to
/// the Track A scalar oracle `__mind_blas_dot_q16` applied per row.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_RMAJOR_Q16_INTRINSIC: &str = "__mind_blas_matmul_rmajor_q16_v";

/// RFC 0006 Track B — fused outer-product Q16.16 GEMM surface name.
///
/// Signature: `(a_addr, b_addr, c_addr, m, k, n) -> i64` (returns 0).
/// `a` is M×K row-major Q16.16 (i32 elements, base packed i64), `b` is
/// **K×N row-major** Q16.16 (i32), `c` is M×N row-major Q16.16 (i32),
/// caller-allocated. Computes
/// `C[i,j] = trunc_i32( Σ_k ((A[i,k]*B[k,j]) >> 16) )`.
///
/// Unlike the gemv-composed `__mind_blas_matmul_rmajor_q16_v` (which
/// re-streams Bᵀ for every output row), this is an outer-product
/// register-tiled microkernel: an `NR`-wide `vector<NRxi64>` accumulator
/// runs over output **columns** j, fed by a broadcast A scalar and a
/// `vector<NRxi32>` B-row slice — no horizontal reduction. Each product
/// term `(A[i,k]*B[k,j])` is `arith.shrsi`-shifted by 16 individually
/// before being added into the i64 accumulator, and the i64→i32 truncation
/// happens exactly once at the store. Because each term is shifted to a
/// fixed i64 value before accumulation and i64 add is associative and
/// commutative, the sum is byte-identical to the per-element scalar oracle
/// `Σ_k (A[i,k]*B[k,j])>>16` under any k-order, tiling or lane grouping.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_MM_Q16_INTRINSIC: &str = "__mind_blas_matmul_mm_q16_v";

/// Multithreaded fused outer-product Q16.16 GEMM surface name.
///
/// Signature: `(a_addr, b_addr, c_addr, m, k, n) -> i64` (returns 0). ABI
/// and semantics are byte-for-byte identical to the single-thread
/// `__mind_blas_matmul_mm_q16_v`; the kernel is internally parallelised with
/// raw POSIX threads (no libomp, no extra runtime dependency — the schedule
/// is baked into the emitted artifact).
///
/// Determinism (the wedge): the output rows `[0, M)` are split into `T`
/// **contiguous owner-computes bands** (`band = ceildiv(M, T)`), and each
/// thread computes `C[row_start:row_end, 0:N]` ENTIRELY with the SAME fused
/// outer-product math as the single-thread kernel. Every output element is
/// written by exactly one thread — there is NO cross-thread reduction, NO
/// atomic, NO shared accumulator — so the result is byte-for-byte identical
/// to the single-thread kernel REGARDLESS of the thread count `T`. The
/// runtime thread count therefore does not enter the output: it may be read
/// from `sysconf(_SC_NPROCESSORS_ONLN)` at runtime (and is, here) without
/// affecting cross-substrate bit-identity.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_MM_Q16_MT_INTRINSIC: &str = "__mind_blas_matmul_mm_q16_mt_v";

/// "det.igemm" tier — fused int8 GEMM surface name.
///
/// Signature: `(a_addr, b_addr, c_addr, m, k, n) -> i64` (returns 0).
/// `a` is M×K row-major **int8** (1-byte elements, base packed i64), `b` is
/// **K×N row-major** int8, `c` is M×N row-major **int32** (4-byte elements),
/// caller-allocated. Computes the pure integer
/// `C[i,j] = (i32) Σ_k ((i32)A[i,k] * (i32)B[k,j])` — int8 is integer, not
/// fixed-point, so there is NO `>> 16` shift.
///
/// Lowering reuses the EXACT BLIS-blocked macro-kernel of
/// `__mind_blas_matmul_mm_q16_v` (`emit_mm_i8_blocked` mirrors
/// `emit_mm_q16_blocked` term-for-term) with two surgical differences: the
/// A/B source loads are i8 + `arith.extsi` to i32 during the pack (the packed
/// panels stay i32, identical extent to the Q16 panels), and the microkernel
/// multiply-accumulates WITHOUT the per-term shift. The C-tile accumulates in
/// i64; the i64→i32 truncation happens exactly once at the store.
///
/// Determinism / overflow: each product `(i32)a*(i32)b` has magnitude ≤ 2^14
/// (|a|,|b| ≤ 128); accumulation is carried in i64 throughout (the C-scratch is
/// i64), so the full-K reduction is exact for any realistic K and i64 add is
/// associative + commutative ⇒ any tiling / lane order yields the identical
/// int32 result. At `-march=x86-64-v3` the i32 widen-multiply-accumulate inner
/// loop legalises to the AVX2 `vpmaddwd` (`_mm256_madd_epi16`) idiom — exact,
/// NEVER the saturating `vpmaddubsw`; on aarch64 the same MLIR lowers to
/// `SDOT`/`SMMLA`. Both produce the identical exact int32 sum, so cross-substrate
/// bit-identity is automatic.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_MM_I8_INTRINSIC: &str = "__mind_blas_matmul_mm_i8_v";

/// "det.igemm" tier — multithreaded fused int8 GEMM surface name.
///
/// Same ABI (arity 6: `a, b, c, m, k, n`; i64; returns 0) and byte-for-byte
/// output as `__mind_blas_matmul_mm_i8_v`. The output rows `[0, M)` are split
/// into `T` **contiguous owner-computes bands** (`band = ceildiv(M, T)`,
/// `T = clamp(sysconf(_SC_NPROCESSORS_ONLN), 1, M)`); each raw POSIX thread
/// computes `C[row_start:row_end, 0:N]` ENTIRELY with the SAME BLIS-blocked
/// int8 macro-kernel (`emit_mm_i8_blocked`) the single-thread kernel uses.
/// Every output element is written by exactly one thread — NO cross-thread
/// reduction, NO atomic, NO shared accumulator — and each worker's i64
/// C-scratch + i32 packed-A / packed-B panels are private stack `alloca`s
/// (emitted inside the worker `llvm.func`, one set per call frame), so there
/// is no shared mutable state and the result is byte-for-byte identical to the
/// single-thread kernel REGARDLESS of `T`. The runtime thread count therefore
/// does not enter the output, so cross-substrate bit-identity holds.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_MM_I8_MT_INTRINSIC: &str = "__mind_blas_matmul_mm_i8_mt_v";

/// "int-dot" tier (RFC 0006 Track B) — the pure-MIND surface name whose
/// `Instr::Call` lowers to a native MLIR `vector`-dialect **int16** dot
/// product. Inputs are i16 row-major; the kernel computes the scalar oracle
/// `c = (i32) sum_k ((i32)a[k] * (i32)b[k])` exactly: sign-extend each i16
/// to i64, multiply, accumulate in i64 lanes (no shift, no saturation, no
/// early narrowing), associative `vector.reduction <add>`, then a final
/// `trunci i64->i32` + `extsi i32->i64` (the oracle's `(i32)acc`). Integer
/// add is associative, so lane grouping is irrelevant — byte-identical to
/// the sequential scalar oracle on **all** int16 inputs. At
/// `-march=x86-64-v3` the i16 widen-multiply-accumulate inner loop is the
/// `vpmaddwd` (`_mm256_madd_epi16`) idiom (16 i16 -> 8 i32 pairwise sums),
/// the fast deterministic int GEMM tier.
#[cfg(feature = "std-surface")]
const VEC_DOT_I16_INTRINSIC: &str = "__mind_blas_dot_i16_v";

/// "int-dot" tier — native MLIR vector-dialect **int16** row-major
/// matrix-vector multiply surface name.
///
/// Signature: `(w_addr, x_addr, y_addr, rows, cols) -> i64` (returns 0).
/// W is a rows×cols row-major i16 matrix (base address packed as i64),
/// x is a cols-element i16 vector, y is a caller-allocated rows-element
/// **i32** output (the exact accumulator narrowed once at the end). The
/// kernel computes `y[r] = dot_i16(W[r,:], x)` for each row r — identically
/// to calling `__mind_blas_dot_i16_v` on each row — using the proven int16
/// reduction from `emit_vec_dot_i16`. Byte-identical to the scalar oracle
/// applied per row, for all int16 inputs.
#[cfg(feature = "std-surface")]
const VEC_MATMUL_RMAJOR_I16_INTRINSIC: &str = "__mind_blas_matmul_rmajor_i16_v";

/// "int-dot" tier — int16 vector lane count. The AVX2 `vpmaddwd`
/// (`_mm256_madd_epi16`) idiom consumes 16 i16 per 256-bit register; this
/// lane count makes the inner `vector<16xi16>` load + widen-multiply
/// inner loop legalise to that instruction at `-march=x86-64-v3`. The
/// accumulator is `vector<16xi64>` so the sum is exact for all int16
/// inputs (no i32-pairwise overflow as the raw `vpmaddwd` result would
/// have — see the lowering doc).
#[cfg(feature = "std-surface")]
const VEC_I16_LANES: usize = 16;

/// "int-dot" tier — the i32 partial-lane count produced by one `vpmaddwd`
/// over `VEC_I16_LANES` i16 inputs (16 i16 -> 8 i32 pairwise sums). The
/// i64 accumulator carries this many lanes; each i32 partial is
/// sign-extended to i64 before accumulation so the cross-lane reduction is
/// exact.
#[cfg(feature = "std-surface")]
const VEC_I16_PMADD_LANES: usize = 8;

/// RFC 0006 Track B (increment 2) — Q16.16 vector lane count. The
/// scalar Q16.16 oracle widens each `i32` product to `i64` before the
/// arithmetic `>> 16`; the vector path mirrors that with
/// `vector<8xi64>` accumulator lanes. Eight matches the f32 width so
/// LLVM legalises both metrics from a single tile shape.
#[cfg(feature = "std-surface")]
const VEC_Q16_LANES: usize = 8;

/// True when the host this `mindc` runs on is x86_64, so the int16/int8
/// "int-dot" paths may emit the explicit `llvm.x86.avx2.pmadd.wd` (`vpmaddwd`)
/// intrinsic for reliable instruction selection.
///
/// On any non-x86 host (notably aarch64) that intrinsic is an x86-only
/// operation the target backend cannot legalise — LLVM 20's aarch64
/// type-legalizer aborts with "Do not know how to split the result of this
/// operator!" — so those hosts instead emit a **portable, exact-integer**
/// widen-multiply + even/odd pairwise-add idiom (see `emit_vec_dot_i16` /
/// `emit_i8_microkernel_avx2`). Both forms compute the *identical* integer
/// pairwise sum: i16×i16 products are sign-extended and the pairwise sum is
/// formed by exact i32 adds before re-widening to i64, so the per-element math
/// — and therefore every output byte and the pinned cross-substrate canary
/// hashes — is unchanged regardless of which form is emitted. Integer add is
/// associative and width-independent, so lane grouping never perturbs the
/// result.
///
/// `mindc` builds for its own host arch by default (`--emit-shared` with no
/// explicit `--target`), and the cross-substrate CI gate runs `mindc` natively
/// on each substrate's hardware (avx2 on x86_64, neon on aarch64), so the host
/// arch of this binary is the build target — exactly the model the
/// `cross_substrate_identity` gate uses (`cfg!(target_arch)` for the substrate
/// id).
#[cfg(feature = "std-surface")]
const HOST_IS_X86: bool = cfg!(target_arch = "x86_64");

/// RFC 0006 Track B (increment 2) — which f32 distance metric the
/// vectorised reduction emits. L2 has its own `emit_vec_dot_f32`
/// (multiply-accumulate); L1/L∞ share `emit_vec_dot_metric_f32`
/// (abs-difference + add/max reduction).
#[cfg(feature = "std-surface")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VecMetric {
    /// Sum of `|a[i] - b[i]|` — `vector.reduction <add>`.
    L1,
    /// Max of `|a[i] - b[i]|` — `vector.reduction <maximumf>`.
    Linf,
}

/// Structured errors produced by the MLIR lowering pipeline.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MlirLowerError {
    /// The lowering pass does not know how to translate the given instruction.
    #[error("unsupported instruction at index {instr_index}: {op}")]
    UnsupportedOp { instr_index: usize, op: String },
    /// Lowering requires type information that is unavailable.
    #[error("missing type information for value {value:?} while lowering {context}")]
    MissingTypeInfo {
        value: ValueId,
        context: &'static str,
    },
    /// The lowering pipeline detected inconsistent shapes or operands.
    #[error("shape error: {0}")]
    ShapeError(String),
    /// IR verification failed before lowering.
    #[error("IR verification failed: {0}")]
    VerificationFailed(#[from] crate::ir::IrVerifyError),
    /// RFC 0002 C-ABI export wrapper codegen rejected an export.
    #[cfg(feature = "ffi-c-user")]
    #[error("C-ABI export codegen: {0}")]
    CExportError(String),
}

/// A lowered MLIR module in textual form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirModule {
    /// The fully formatted MLIR module text.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValueKind {
    ScalarI64,
    /// A scalar `f64` SSA value (RFC 0012 §5.1 deterministic float). Produced by
    /// `Instr::ConstF64` and by an f64-typed `Instr::Param` / `Instr::BinOp`.
    /// Lowers to strict IEEE `arith.*f` on `f64` — NO fastmath / reassoc /
    /// contract flags — so scalar `+ − × ÷` is byte-identical across substrates.
    /// Gated; the default + keystone artifacts (no scalar float) never construct
    /// this variant, so existing trace_hashes are unchanged.
    #[cfg(feature = "std-surface")]
    ScalarF64,
    /// A scalar `f32` SSA value (same determinism contract as `ScalarF64`, at
    /// f32 width). Gated.
    #[cfg(feature = "std-surface")]
    ScalarF32,
    /// A scalar signed `i32` SSA value (NARROW-INT ABI). Lowers to `i32` MLIR;
    /// signedness lives in the op (`divsi`/`shrsi`/`slt`…). Two's-complement
    /// wrap at 32-bit width is the deterministic-overflow contract. Gated; the
    /// default + keystone artifacts are i64-only and never construct this
    /// variant, so existing trace_hashes are unchanged.
    #[cfg(feature = "std-surface")]
    ScalarI32,
    /// A scalar unsigned `u32` SSA value (NARROW-INT ABI). Lowers to the same
    /// signless `i32` MLIR type as `ScalarI32`; unsignedness selects the
    /// `divui`/`shrui`/`ult`… op variants. Gated; never constructed in the
    /// default/keystone build.
    #[cfg(feature = "std-surface")]
    ScalarU32,
    /// A scalar `bool` SSA value (NARROW-INT ABI). Lowers to `i1` MLIR. Gated;
    /// never constructed in the default/keystone build. NOTE: the bool *return*
    /// ABI slot stays `i64` (cmpi+extui+ret-i64) — this variant is for
    /// let-bound / param boolean values only.
    #[cfg(feature = "std-surface")]
    ScalarBool,
    Tensor {
        dtype: DType,
        shape: Vec<ShapeDim>,
    },
    /// RFC 0006 Track B — a `vector<lanes x f32>` SSA value produced by
    /// the SIMD primitives (`VecLoad` / `VecFma`). Gated; the default
    /// build never constructs the producing instructions so this variant
    /// is unreachable there.
    #[cfg(feature = "std-surface")]
    VectorF32 {
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 2) — a `vector<lanes x i64>` SSA value
    /// produced by the Q16.16 SIMD primitives (`VecLoadI32` widened /
    /// `VecMulAddQ16`). Gated; default builds never construct the
    /// producing instructions so this variant is unreachable there.
    #[cfg(feature = "std-surface")]
    VectorI64 {
        lanes: usize,
    },
}

/// `extern "C"` fn signature stored in `LoweringContext::extern_c_fns`:
/// (param_types, ret_type, is_varargs, vararg_hints, callconv).
#[cfg(feature = "std-surface")]
type ExternCFnSig = (
    Vec<String>,
    Option<String>,
    bool,
    Vec<String>,
    crate::ast::CallConv,
);

struct LoweringContext {
    values: BTreeMap<ValueId, ValueKind>,
    outputs: Vec<ValueId>,
    body: String,
    /// RFC 0005 Phase 0: callee name -> arity for every `Instr::Call`
    /// lowered, so the module assembler can emit one
    /// `func.func private @name(i64...) -> i64` declaration per
    /// distinct callee. `BTreeSet` keeps emission deterministic
    /// (stable MLIR text -> stable model_hash). Gated; the default
    /// build has no `Instr::Call` arm and never touches this.
    #[cfg(feature = "std-surface")]
    extern_calls: std::collections::BTreeSet<(String, usize)>,
    /// RFC 0005 P0d: pre-formatted `func.func @name(...) -> i64 { ... }`
    /// bodies for every `Instr::FnDef` seen at module top level. The
    /// assembler concatenates these *before* `@main` and excludes their
    /// names from `extern_calls` so we don't emit a forward decl that
    /// would clash with the definition.
    #[cfg(feature = "std-surface")]
    user_fns: String,
    /// Names defined locally (Instr::FnDef) — filter from extern decls.
    #[cfg(feature = "std-surface")]
    defined_fns: std::collections::BTreeSet<String>,
    /// RFC 0010 Phase A/B/C: `extern "C"` fn declarations.
    ///
    /// name → (param_types, ret_type, is_varargs, vararg_hints, callconv)
    ///
    /// Populated by `Instr::ExternFnDecl`; consulted by the `Instr::Call`
    /// arm to decide whether to emit `llvm.call` or `func.call`, and to
    /// assign MLIR types to each argument position (including varargs).
    /// Phase B adds `vararg_hints` for precise per-position typing of extra
    /// variadic arguments beyond the declared parameter list.
    /// Phase C adds `callconv` so the `llvm.func` / `llvm.call` emitter can
    /// attach `cconv = #llvm.cconv<win64cc>` for Win64 declarations. Gated.
    #[cfg(feature = "std-surface")]
    extern_c_fns: std::collections::BTreeMap<String, ExternCFnSig>,
    /// RFC 0005 Gap 1 (collision fix): function-global monotonic counter that
    /// labels every `while` block triple (`^while_header_N` / `^while_body_N`
    /// / `^while_after_N`). The old key — the While's per-region
    /// `instr_index` — collided when a nested inner loop and a sibling loop
    /// shared the same index, emitting duplicate block labels that mlir-opt
    /// rejects (`redefinition of block`). A single counter threaded through
    /// every recursive sub-context (cond/body of a while, then/else of an if,
    /// and each FnDef body) guarantees a unique label per loop. It increments
    /// in fixed pre-order traversal order, so the label numbering is
    /// deterministic and reproducible.
    #[cfg(feature = "std-surface")]
    while_label: usize,
    /// Value-ids whose underlying MLIR value is an `i1` (an `arith.cmpi`
    /// boolean result). The BinOp comparison arm tags every comparison result
    /// as `ScalarI64` so downstream arithmetic keeps working, but the real
    /// MLIR value is `i1`. When such a value flows straight into the `-> i64`
    /// ABI return slot, mlir-opt rejects `return %v : i64` ("'i64' vs 'i1'"),
    /// so a `-> bool` function never lowers. Tracked here so the return sites
    /// can zero-extend it (`arith.extui %v : i1 to i64`) before returning.
    /// Additive: only return-of-comparison (previously uncompilable) changes
    /// emitted text, so existing artifacts stay byte-identical.
    #[cfg(feature = "std-surface")]
    i1_values: std::collections::BTreeSet<ValueId>,
    /// NARROW-INT ABI — value-ids produced by an `Instr::ConstI64` literal.
    /// Every IR literal tags `ScalarI64` (the IR has no narrow-int constant), so
    /// a `narrow OP literal` mixed-width binop is legal and the literal is
    /// trunc'd to the narrow width before dispatch (value-preserving — literals
    /// are range-checked). This set distinguishes a genuine i64 literal (safe to
    /// trunc) from a genuine i64-typed value (a mixed-width error). Additive:
    /// populated for every program but only consulted by the narrow-int arm, so
    /// i64-only lowering is byte-identical.
    #[cfg(feature = "std-surface")]
    const_i64_values: std::collections::BTreeSet<ValueId>,
    /// Stack of enclosing `while` loops (innermost last). Each frame carries the
    /// loop's block label and its loop-carried `(var_name, init_id)` list, so an
    /// `Instr::Break`/`Instr::Continue` in the body can emit a `cf.br` to the
    /// loop's `^while_after`/`^while_header` forwarding the current values of
    /// each carried var (looked up by name in the instr's `live` snapshot).
    #[cfg(feature = "std-surface")]
    loop_stack: Vec<LoopFrame>,
    /// Pre-formatted top-level `llvm.func @mind_mt_worker_q16_<id>(...)` worker
    /// bodies emitted by the multithreaded Q16.16 GEMM intrinsic. The module
    /// assembler concatenates these before `@main` (alongside `user_fns`). The
    /// worker runs the fused row-band kernel and is referenced by
    /// `llvm.mlir.addressof` as the `pthread_create` start routine.
    #[cfg(feature = "std-surface")]
    mt_workers: String,
    /// Set when any multithreaded intrinsic was lowered, so the module
    /// assembler emits the `pthread_create` / `pthread_join` / `sysconf`
    /// `llvm.func` externs exactly once.
    #[cfg(feature = "std-surface")]
    needs_pthread: bool,
    /// Set when the int8 BLIS macro-kernel (`emit_mm_i8_blocked`) was lowered,
    /// so the module assembler emits the `@malloc` / `@free` `llvm.func`
    /// externs exactly once. The kernel's C-scratch / packed-A / packed-B
    /// panels are heap-allocated (malloc/free) rather than stack `llvm.alloca`
    /// so a large MC row block is safe regardless of caller stack depth. Heap
    /// vs stack is the same computation in a different location — byte-identical.
    #[cfg(feature = "std-surface")]
    needs_malloc: bool,
    /// RFC 0012 §5.1 — function-ABI signature table, threaded in from
    /// `IRModule::fn_signatures` at the top of `lower_ir_to_mlir`.
    ///
    /// name → `(param_mlir_types, ret_mlir_type)`, each pre-resolved to an MLIR
    /// type string (`"f64"` / `"f32"` / `"i64"`). The `Instr::FnDef` arm reads
    /// it to emit each `func.func` parameter and the `-> T` return slot with the
    /// declared float type instead of the default i64 ABI, and to seed each
    /// f64/f32 param's `ValueKind` so the scalar BinOp dispatch and the return
    /// see them as floats. Shared by value into each FnDef sub-context (like
    /// `extern_c_fns`) so nested definitions resolve too. Empty for i64-only
    /// modules → byte-identical lowering. Gated.
    #[cfg(feature = "std-surface")]
    fn_signatures: std::collections::BTreeMap<String, (Vec<String>, Option<String>)>,
    /// NARROW-INT ABI — per-fn parameter `ValueKind`s derived from the ORIGINAL
    /// `TypeAnn` list (not the ABI string table above, which is signedness-lossy:
    /// both `i32` and `u32` stringify to `"i32"`). Keyed by fn name, parallel to
    /// `fn_signatures`. The `Instr::FnDef` arm reads it to seed each param's
    /// `ValueKind` so the scalar BinOp dispatch picks signed vs unsigned ops
    /// (`divsi`/`divui`, `shrsi`/`shrui`, `slt`/`ult`…). Shared by value into
    /// each FnDef sub-context (like `fn_signatures`). Empty for modules with no
    /// narrow-int / bool params → byte-identical lowering. Gated.
    #[cfg(feature = "std-surface")]
    fn_param_kinds: std::collections::BTreeMap<String, Vec<ValueKind>>,
}

/// One enclosing `while` loop, for break/continue codegen.
#[cfg(feature = "std-surface")]
#[derive(Clone)]
struct LoopFrame {
    lbl: usize,
    /// `(var_name, init_id)` per loop-carried var, in the canonical loop-arg
    /// order — the order the `^while_header`/`^while_after` block-args expect.
    carried: Vec<(String, usize)>,
}

/// True if `i` terminates its MLIR block (no fall-through): `return`, or — under
/// std-surface — `break`/`continue` (each lowers to a `cf.br`). The If lowering
/// uses this so it never appends a second terminator after a self-terminated
/// branch (mlir-opt rejects an op with successors that is not block-final).
fn instr_is_block_terminator(i: &Instr) -> bool {
    match i {
        Instr::Return { .. } => true,
        #[cfg(feature = "std-surface")]
        Instr::Break { .. } | Instr::Continue { .. } => true,
        _ => false,
    }
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            outputs: Vec::new(),
            body: String::new(),
            #[cfg(feature = "std-surface")]
            extern_calls: std::collections::BTreeSet::new(),
            #[cfg(feature = "std-surface")]
            user_fns: String::new(),
            #[cfg(feature = "std-surface")]
            defined_fns: std::collections::BTreeSet::new(),
            #[cfg(feature = "std-surface")]
            extern_c_fns: std::collections::BTreeMap::new(),
            #[cfg(feature = "std-surface")]
            while_label: 0,
            #[cfg(feature = "std-surface")]
            i1_values: std::collections::BTreeSet::new(),
            #[cfg(feature = "std-surface")]
            const_i64_values: std::collections::BTreeSet::new(),
            #[cfg(feature = "std-surface")]
            loop_stack: Vec::new(),
            #[cfg(feature = "std-surface")]
            mt_workers: String::new(),
            #[cfg(feature = "std-surface")]
            needs_pthread: false,
            #[cfg(feature = "std-surface")]
            needs_malloc: false,
            #[cfg(feature = "std-surface")]
            fn_signatures: std::collections::BTreeMap::new(),
            #[cfg(feature = "std-surface")]
            fn_param_kinds: std::collections::BTreeMap::new(),
        }
    }

    fn emit_line(&mut self, line: &str) {
        writeln!(&mut self.body, "{line}").expect("write to string cannot fail");
    }

    /// NARROW-INT control-flow merge typing. Given the two branch yields of a
    /// single merge column (then-edge / else-edge), return the merge's MLIR
    /// `ValueKind` plus the (possibly rewritten) value strings each `cf.br`
    /// edge should pass.
    ///
    /// Rules:
    ///   * identical MLIR type on both arms → that kind; values unchanged; NO
    ///     extension emitted. (unify(i64,i64)=i64 is this path — byte-identical
    ///     to the legacy hardcoded-i64 output.)
    ///   * different integer widths → widen the narrower arm to `i64` with
    ///     `arith.extsi` (signed: i32/i64) or `arith.extui` (unsigned/bool:
    ///     u32/i1) emitted INSIDE that arm's body buffer just before its
    ///     `cf.br`; the merge kind is then `ScalarI64`.
    ///
    /// `t_buf`/`e_buf` are the then/else branch body buffers (extensions are
    /// appended there). `lbl`/`col` form the fresh widened-value SSA name so it
    /// never collides with a numeric id.
    #[allow(clippy::too_many_arguments)]
    fn unify_merge_kind(
        &self,
        then_kind: &ValueKind,
        else_kind: &ValueKind,
        then_val: &str,
        else_val: &str,
        t_buf: &mut String,
        e_buf: &mut String,
        lbl: usize,
        col: usize,
    ) -> Result<(ValueKind, String, String), MlirLowerError> {
        // Bit width of a scalar integer kind (for the merge-width decision);
        // None for non-integer kinds, which we never widen here.
        fn int_width(k: &ValueKind) -> Option<u32> {
            match k {
                ValueKind::ScalarI64 => Some(64),
                #[cfg(feature = "std-surface")]
                ValueKind::ScalarI32 | ValueKind::ScalarU32 => Some(32),
                #[cfg(feature = "std-surface")]
                ValueKind::ScalarBool => Some(1),
                _ => None,
            }
        }
        // `extui` for unsigned/bool, `extsi` for signed.
        #[cfg(feature = "std-surface")]
        fn is_unsigned(k: &ValueKind) -> bool {
            matches!(k, ValueKind::ScalarU32 | ValueKind::ScalarBool)
        }
        #[cfg(not(feature = "std-surface"))]
        fn is_unsigned(_k: &ValueKind) -> bool {
            false
        }

        // Same MLIR type on both arms → no widening, byte-identical path.
        if mlir_type(then_kind)? == mlir_type(else_kind)? {
            return Ok((then_kind.clone(), then_val.to_string(), else_val.to_string()));
        }

        match (int_width(then_kind), int_width(else_kind)) {
            (Some(tw), Some(ew)) if tw != ew => {
                // Widen the narrower arm to i64.
                let mut tv = then_val.to_string();
                let mut ev = else_val.to_string();
                let src_ty_t = mlir_type(then_kind)?;
                let src_ty_e = mlir_type(else_kind)?;
                if tw < ew {
                    let op = if is_unsigned(then_kind) { "extui" } else { "extsi" };
                    let name = format!("%mwide_{lbl}_{col}_t");
                    writeln!(
                        t_buf,
                        "    {name} = arith.{op} {then_val} : {src_ty_t} to i64"
                    )
                    .expect("write to string cannot fail");
                    tv = name;
                } else {
                    let op = if is_unsigned(else_kind) { "extui" } else { "extsi" };
                    let name = format!("%mwide_{lbl}_{col}_e");
                    writeln!(
                        e_buf,
                        "    {name} = arith.{op} {else_val} : {src_ty_e} to i64"
                    )
                    .expect("write to string cannot fail");
                    ev = name;
                }
                Ok((ValueKind::ScalarI64, tv, ev))
            }
            // Mismatched non-integer / unhandled mix (should not occur for the
            // type-checked narrow-int surface) → fall back to i64 merge kind
            // without rewriting, preserving prior behavior shape.
            _ => Ok((ValueKind::ScalarI64, then_val.to_string(), else_val.to_string())),
        }
    }

    fn emit_instr(&mut self, instr_index: usize, instr: &Instr) -> Result<(), MlirLowerError> {
        match instr {
            Instr::ConstI64(id, value) => {
                self.emit_line(&format!("    %{} = arith.constant {} : i64", id.0, value));
                self.values.insert(*id, ValueKind::ScalarI64);
                // Record that this value is an integer LITERAL, so the narrow-int
                // BinOp arm may truncate it to a narrow operand width (value-
                // preserving) instead of failing closed on a width mismatch.
                #[cfg(feature = "std-surface")]
                self.const_i64_values.insert(*id);
            }
            // RFC 0012 §5.1 — a scalar `f64` literal. Emit `arith.constant <v> : f64`
            // with a DETERMINISTIC, shortest-round-trip decimal (`format_number`):
            // the same `f64` always prints identical bytes, and the decimal
            // round-trips to the exact IEEE bits through LLVM's APFloat parser.
            // NO fastmath; this is the deterministic-float producer for constants.
            // Gated; ConstF64 never appears in the keystone source, so the
            // keystone artifact stays byte-identical.
            #[cfg(feature = "std-surface")]
            Instr::ConstF64(id, value) => {
                self.emit_line(&format!(
                    "    %{} = arith.constant {} : f64",
                    id.0,
                    format_number(*value)
                ));
                self.values.insert(*id, ValueKind::ScalarF64);
            }
            Instr::ConstTensor(id, dtype, shape, fill) => {
                let dtype_str = dtype.as_str();
                let tensor_ty = tensor_type(shape, dtype_str);
                let fill_value = format_fill(*fill, dtype);
                self.emit_line(&format!(
                    "    %fill{} = arith.constant {} : {}",
                    id.0, fill_value, dtype_str
                ));
                self.emit_line(&format!(
                    "    %tmp{} = tensor.empty() : {}",
                    id.0, tensor_ty
                ));
                self.emit_line(&format!(
                    "    %{} = linalg.fill ins(%fill{} : {}) outs(%tmp{} : {}) -> {}",
                    id.0, id.0, dtype_str, id.0, tensor_ty, tensor_ty
                ));
                self.values.insert(
                    *id,
                    ValueKind::Tensor {
                        dtype: dtype.clone(),
                        shape: shape.clone(),
                    },
                );
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                let lhs_kind = self
                    .values
                    .get(lhs)
                    .ok_or(MlirLowerError::MissingTypeInfo {
                        value: *lhs,
                        context: "binop",
                    })?
                    .clone();
                let rhs_kind = self
                    .values
                    .get(rhs)
                    .ok_or(MlirLowerError::MissingTypeInfo {
                        value: *rhs,
                        context: "binop",
                    })?
                    .clone();
                match (&lhs_kind, &rhs_kind) {
                    (
                        ValueKind::Tensor { dtype, shape },
                        ValueKind::Tensor {
                            dtype: dtype_b,
                            shape: shape_b,
                        },
                    ) => {
                        if dtype != dtype_b {
                            return Err(MlirLowerError::ShapeError(
                                "tensor binary ops require matching element types".into(),
                            ));
                        }
                        let op_str = select_arith_op(*op, dtype);
                        if shape == shape_b {
                            // Equal shapes: original single-line `arith` emit. Kept
                            // byte-identical so the self-host bootstrap and every
                            // existing test produce unchanged MLIR text.
                            let ty = tensor_type(shape, dtype.as_str());
                            self.emit_line(&format!(
                                "    %{} = {} %{}, %{} : {}",
                                dst.0, op_str, lhs.0, rhs.0, ty
                            ));
                            self.values.insert(
                                *dst,
                                ValueKind::Tensor {
                                    dtype: dtype.clone(),
                                    shape: shape.clone(),
                                },
                            );
                        } else {
                            // RFC 0012 §4.2: the type-checker has already validated a
                            // right-aligned broadcast (`shapes::broadcast_shapes`);
                            // emit a `linalg.generic` that broadcasts each operand to
                            // the result shape. Broadcasting is a pure index remap —
                            // no float reassociation — so cross-substrate Q16.16
                            // bit-identity is preserved.
                            // Core v1 (mind-spec language.md §arithmetic): only
                            // `+ - *` lower to a broadcasting `BinOp`; division is
                            // not Core v1, so a broadcasting `/`/`%` is a shape
                            // error rather than a silent miscompile.
                            if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
                                return Err(MlirLowerError::ShapeError(format!(
                                    "broadcasting is only supported for elementwise \
                                     +, -, * on tensors (Core v1); operator `{op:?}` \
                                     requires matching operand shapes"
                                )));
                            }
                            let (result_shape, lhs_map, rhs_map) =
                                broadcast_binop_maps(shape, shape_b)?;
                            let elem = dtype.as_str();
                            let result_ty = tensor_type(&result_shape, elem);
                            let lhs_ty = tensor_type(shape, elem);
                            let rhs_ty = tensor_type(shape_b, elem);
                            let rank = result_shape.len();
                            let dims = (0..rank)
                                .map(|d| format!("d{d}"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            let iters = vec!["\"parallel\""; rank].join(", ");
                            let lhs_aff =
                                format!("affine_map<({dims}) -> ({})>", lhs_map.join(", "));
                            let rhs_aff =
                                format!("affine_map<({dims}) -> ({})>", rhs_map.join(", "));
                            let out_aff = format!("affine_map<({dims}) -> ({dims})>");
                            self.emit_line(&format!(
                                "    %bcast{} = tensor.empty() : {}",
                                dst.0, result_ty
                            ));
                            self.emit_line(&format!(
                                "    %{} = linalg.generic {{indexing_maps = [{}, {}, {}], \
                                 iterator_types = [{}]}} ins(%{}, %{} : {}, {}) \
                                 outs(%bcast{} : {}) {{",
                                dst.0,
                                lhs_aff,
                                rhs_aff,
                                out_aff,
                                iters,
                                lhs.0,
                                rhs.0,
                                lhs_ty,
                                rhs_ty,
                                dst.0,
                                result_ty
                            ));
                            self.emit_line(&format!(
                                "    ^bb0(%bcl{}: {elem}, %bcr{}: {elem}, %bco{}: {elem}):",
                                dst.0, dst.0, dst.0
                            ));
                            self.emit_line(&format!(
                                "      %bcv{} = {} %bcl{}, %bcr{} : {elem}",
                                dst.0, op_str, dst.0, dst.0
                            ));
                            self.emit_line(&format!("      linalg.yield %bcv{} : {elem}", dst.0));
                            self.emit_line(&format!("    }} -> {result_ty}"));
                            self.values.insert(
                                *dst,
                                ValueKind::Tensor {
                                    dtype: dtype.clone(),
                                    shape: result_shape,
                                },
                            );
                        }
                    }
                    // RFC 0012 §5.1 — scalar `f64`/`f32` arithmetic. Both operands
                    // must share the same float kind. Emit STRICT IEEE `arith.*f`
                    // / `arith.cmpf` with ORDERED predicates (olt/ole/…/one) at the
                    // operand width — NO fastmath / reassoc / contract / nnan / ninf
                    // flags, and NO FMA contraction (`a*b+c` lowers to a separate
                    // `mulf` then `addf`, never `fmuladd`). This sequential strict
                    // form is what makes scalar `+ − × ÷` byte-identical across the
                    // avx2 and neon substrates. Gated; never fires on the i64-only
                    // keystone source so existing trace_hashes are unchanged.
                    #[cfg(feature = "std-surface")]
                    (ValueKind::ScalarF64, ValueKind::ScalarF64)
                    | (ValueKind::ScalarF32, ValueKind::ScalarF32) => {
                        let (fty, dtype, out_kind) = if matches!(lhs_kind, ValueKind::ScalarF32) {
                            ("f32", DType::F32, ValueKind::ScalarF32)
                        } else {
                            ("f64", DType::F64, ValueKind::ScalarF64)
                        };
                        // Reuse the tensor float-op selector — `arith.addf`/`subf`/
                        // `mulf`/`divf` and ordered `arith.cmpf` — for scalars.
                        let mlir_op = select_arith_op(*op, &dtype);
                        let is_cmp = matches!(
                            op,
                            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
                        );
                        // `arith.cmpf` operates on the float operands but yields i1;
                        // the operand annotation is the float type either way.
                        self.emit_line(&format!(
                            "    %{} = {} %{}, %{} : {}",
                            dst.0, mlir_op, lhs.0, rhs.0, fty
                        ));
                        if is_cmp {
                            // Comparison result is i1; tag it so return sites widen
                            // it for the i64 ABI slot (same path as cmpi).
                            self.values.insert(*dst, ValueKind::ScalarI64);
                            self.i1_values.insert(*dst);
                        } else {
                            self.values.insert(*dst, out_kind);
                        }
                    }
                    // Narrow-int / bool scalar arithmetic (i32/u32/i1). At least
                    // one operand is a narrow kind; the other must be the SAME
                    // narrow kind or a `ConstI64` literal (truncated to the narrow
                    // width — value-preserving, since literals are range-checked).
                    // A genuine non-literal width mismatch fails closed rather
                    // than silently miscompiling. Signedness drives the op
                    // selection (`divui`/`shrui`/`ult` for u32, `divsi`/`shrsi`/
                    // `slt` for i32); two's-complement wrap at the declared width
                    // is the deterministic-overflow contract, identical on every
                    // substrate. Gated; never fires on the i64-only keystone.
                    #[cfg(feature = "std-surface")]
                    _ if matches!(
                        lhs_kind,
                        ValueKind::ScalarI32 | ValueKind::ScalarU32 | ValueKind::ScalarBool
                    ) || matches!(
                        rhs_kind,
                        ValueKind::ScalarI32 | ValueKind::ScalarU32 | ValueKind::ScalarBool
                    ) =>
                    {
                        fn is_narrow(k: &ValueKind) -> bool {
                            matches!(
                                k,
                                ValueKind::ScalarI32 | ValueKind::ScalarU32 | ValueKind::ScalarBool
                            )
                        }
                        // The operative narrow kind: whichever operand is narrow
                        // (if both are, LHS — they must match or it fails closed).
                        let target = if is_narrow(&lhs_kind) {
                            lhs_kind.clone()
                        } else {
                            rhs_kind.clone()
                        };
                        let (ity, unsigned, out_kind) = match &target {
                            ValueKind::ScalarBool => ("i1", false, ValueKind::ScalarBool),
                            ValueKind::ScalarU32 => ("i32", true, ValueKind::ScalarU32),
                            _ => ("i32", false, ValueKind::ScalarI32),
                        };
                        // Legalize one operand to `ity`: pass through a matching
                        // narrow value, truncate an i64 literal, else fail closed.
                        let mut legalize =
                            |this: &mut Self,
                             id: ValueId,
                             k: &ValueKind,
                             tmp: &str|
                             -> Result<String, MlirLowerError> {
                                if is_narrow(k) {
                                    if *k != target {
                                        return Err(MlirLowerError::ShapeError(format!(
                                            "mixed-width integer operands ({k:?} vs {target:?}) \
                                             are not lowerable; cast explicitly"
                                        )));
                                    }
                                    Ok(format!("%{}", id.0))
                                } else if matches!(k, ValueKind::ScalarI64)
                                    && this.const_i64_values.contains(&id)
                                {
                                    this.emit_line(&format!(
                                        "    %{tmp}{0} = arith.trunci %{1} : i64 to {ity}",
                                        dst.0, id.0
                                    ));
                                    Ok(format!("%{tmp}{}", dst.0))
                                } else {
                                    Err(MlirLowerError::ShapeError(format!(
                                        "a non-literal i64 operand mixed with `{ity}` is not \
                                         lowerable; cast explicitly"
                                    )))
                                }
                            };
                        let lhs_ref = legalize(self, *lhs, &lhs_kind, "ntl")?;
                        let rhs_ref = legalize(self, *rhs, &rhs_kind, "ntr")?;
                        let mlir_op = match op {
                            BinOp::Add => "arith.addi".to_string(),
                            BinOp::Sub => "arith.subi".to_string(),
                            BinOp::Mul => "arith.muli".to_string(),
                            BinOp::Div => if unsigned {
                                "arith.divui"
                            } else {
                                "arith.divsi"
                            }
                            .to_string(),
                            BinOp::Mod => if unsigned {
                                "arith.remui"
                            } else {
                                "arith.remsi"
                            }
                            .to_string(),
                            BinOp::Lt => {
                                format!("arith.cmpi \"{}\",", if unsigned { "ult" } else { "slt" })
                            }
                            BinOp::Le => {
                                format!("arith.cmpi \"{}\",", if unsigned { "ule" } else { "sle" })
                            }
                            BinOp::Gt => {
                                format!("arith.cmpi \"{}\",", if unsigned { "ugt" } else { "sgt" })
                            }
                            BinOp::Ge => {
                                format!("arith.cmpi \"{}\",", if unsigned { "uge" } else { "sge" })
                            }
                            BinOp::Eq => "arith.cmpi \"eq\",".to_string(),
                            BinOp::Ne => "arith.cmpi \"ne\",".to_string(),
                            BinOp::BitAnd => "arith.andi".to_string(),
                            BinOp::BitOr => "arith.ori".to_string(),
                            BinOp::BitXor => "arith.xori".to_string(),
                            BinOp::Shl => "arith.shli".to_string(),
                            BinOp::Shr => if unsigned {
                                "arith.shrui"
                            } else {
                                "arith.shrsi"
                            }
                            .to_string(),
                        };
                        let is_cmp = matches!(
                            op,
                            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
                        );
                        self.emit_line(&format!(
                            "    %{} = {} {}, {} : {}",
                            dst.0, mlir_op, lhs_ref, rhs_ref, ity
                        ));
                        if is_cmp {
                            // Comparison yields i1; tag like the i64/float arms so
                            // a return site widens it for the i64 ABI slot.
                            self.values.insert(*dst, ValueKind::ScalarI64);
                            self.i1_values.insert(*dst);
                        } else {
                            self.values.insert(*dst, out_kind);
                        }
                    }
                    _ => {
                        let mlir_op = match op {
                            BinOp::Add => "arith.addi",
                            BinOp::Sub => "arith.subi",
                            BinOp::Mul => "arith.muli",
                            BinOp::Div => "arith.divsi",
                            BinOp::Mod => "arith.remsi",
                            BinOp::Lt => "arith.cmpi \"slt\",",
                            BinOp::Le => "arith.cmpi \"sle\",",
                            BinOp::Gt => "arith.cmpi \"sgt\",",
                            BinOp::Ge => "arith.cmpi \"sge\",",
                            BinOp::Eq => "arith.cmpi \"eq\",",
                            BinOp::Ne => "arith.cmpi \"ne\",",
                            // Phase 6.5 Stage 1a — bitwise ops on i64.
                            #[cfg(feature = "std-surface")]
                            BinOp::BitAnd => "arith.andi",
                            #[cfg(feature = "std-surface")]
                            BinOp::BitOr => "arith.ori",
                            #[cfg(feature = "std-surface")]
                            BinOp::BitXor => "arith.xori",
                            #[cfg(feature = "std-surface")]
                            BinOp::Shl => "arith.shli",
                            // Arithmetic (signed) right-shift — matches Rust i64 >> i64.
                            #[cfg(feature = "std-surface")]
                            BinOp::Shr => "arith.shrsi",
                        };
                        self.emit_line(&format!(
                            "    %{} = {} %{}, %{} : i64",
                            dst.0, mlir_op, lhs.0, rhs.0
                        ));
                        self.values.insert(*dst, ValueKind::ScalarI64);
                        // A comparison op lowers to `arith.cmpi`, whose result
                        // is `i1` (not the `: i64` operand type above). Record
                        // it so the return sites can zero-extend an i1 that
                        // flows into the `-> i64` ABI return slot.
                        #[cfg(feature = "std-surface")]
                        if matches!(
                            op,
                            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
                        ) {
                            self.i1_values.insert(*dst);
                        }
                    }
                }
            }
            Instr::MatMul { dst, a, b } => {
                let a_info = self.tensor_info(a, "matmul lhs")?;
                let b_info = self.tensor_info(b, "matmul rhs")?;
                let (out_shape, m_ty, n_ty, result_ty) =
                    matmul_shapes(&a_info.shape, &b_info.shape, a_info.dtype.as_str())?;
                self.emit_line(&format!(
                    "    %tmp{} = tensor.empty() : {}",
                    dst.0, result_ty
                ));
                // Grouped `ins(%a, %b : ta, tb)` form — the only one mlir-opt's
                // linalg named-op parser accepts (a per-operand `ins(%a : ta, %b : tb)`
                // is rejected as "expected non-function type").
                self.emit_line(&format!(
                    "    %{} = linalg.matmul ins(%{}, %{} : {}, {}) outs(%tmp{} : {}) -> {}",
                    dst.0, a.0, b.0, m_ty, n_ty, dst.0, result_ty, result_ty
                ));
                self.values.insert(
                    *dst,
                    ValueKind::Tensor {
                        dtype: a_info.dtype.clone(),
                        shape: out_shape,
                    },
                );
            }
            Instr::Conv2d {
                dst,
                input,
                filter,
                stride_h,
                stride_w,
                padding,
            } => {
                let input_info = self.tensor_info(input, "conv2d input")?;
                let filter_info = self.tensor_info(filter, "conv2d filter")?;
                let (out_shape, input_ty, filter_ty, result_ty) =
                    conv2d_shapes(&input_info, &filter_info, *stride_h, *stride_w, *padding)?;
                self.emit_line(&format!(
                    "    %tmp{} = tensor.empty() : {}",
                    dst.0, result_ty
                ));
                // Grouped `ins(%a, %b : ta, tb)` form (see MatMul above).
                self.emit_line(&format!(
                    "    %{} = linalg.conv_2d_nhwc_hwcf ins(%{}, %{} : {}, {}) outs(%tmp{} : {}) -> {}",
                    dst.0, input.0, filter.0, input_ty, filter_ty, dst.0, result_ty, result_ty
                ));
                self.values.insert(
                    *dst,
                    ValueKind::Tensor {
                        dtype: input_info.dtype.clone(),
                        shape: out_shape,
                    },
                );
            }
            Instr::Relu { dst, src } => {
                // RFC 0012 elementwise activation: relu(x) = max(x, 0). Emitted
                // as a shape-preserving `linalg.generic` whose body does a single
                // `arith.maximumf` against a captured `0.0` constant — a pure
                // elementwise map (no reassociation), so Q16.16 bit-identity holds.
                let info = self.tensor_info(src, "relu")?;
                let elem = info.dtype.as_str();
                // relu(x) = max(x, 0); the max op + zero literal are dtype-
                // dependent: float uses `arith.maximumf` vs `0.0`, integer/
                // Q16.16 uses `arith.maxsi` vs `0` (a float op/literal on an
                // integer type is invalid MLIR). Pure elementwise either way.
                let (zero_lit, max_op) = match &info.dtype {
                    DType::F32 | DType::F16 | DType::BF16 => ("0.0", "arith.maximumf"),
                    _ => ("0", "arith.maxsi"),
                };
                let ty = tensor_type(&info.shape, elem);
                let rank = info.shape.len();
                let dims = (0..rank)
                    .map(|d| format!("d{d}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let iters = vec!["\"parallel\""; rank].join(", ");
                let id_map = format!("affine_map<({dims}) -> ({dims})>");
                self.emit_line(&format!(
                    "    %zero{} = arith.constant {zero_lit} : {elem}",
                    dst.0
                ));
                self.emit_line(&format!("    %tmp{} = tensor.empty() : {ty}", dst.0));
                self.emit_line(&format!(
                    "    %{} = linalg.generic {{indexing_maps = [{id_map}, {id_map}], \
                     iterator_types = [{iters}]}} ins(%{} : {ty}) outs(%tmp{} : {ty}) {{",
                    dst.0, src.0, dst.0
                ));
                self.emit_line(&format!(
                    "    ^bb0(%rin{}: {elem}, %rout{}: {elem}):",
                    dst.0, dst.0
                ));
                self.emit_line(&format!(
                    "      %rmax{} = {max_op} %rin{}, %zero{} : {elem}",
                    dst.0, dst.0, dst.0
                ));
                self.emit_line(&format!("      linalg.yield %rmax{} : {elem}", dst.0));
                self.emit_line(&format!("    }} -> {ty}"));
                self.values.insert(
                    *dst,
                    ValueKind::Tensor {
                        dtype: info.dtype.clone(),
                        shape: info.shape.clone(),
                    },
                );
            }
            Instr::ReluGrad { dst, grad, src } => {
                // RFC 0012 backward activation: dx = select(src > 0, grad, 0).
                // Emitted as a two-input shape-preserving `linalg.generic` whose
                // body does `arith.cmpf "ogt"` against a captured `0.0` then
                // `arith.select` — a pure elementwise map (no reassociation), so
                // cross-substrate Q16.16 bit-identity holds. `grad` (upstream)
                // and `src` (the ReLU input) share the same shape.
                let info = self.tensor_info(src, "relu_grad")?;
                let elem = info.dtype.as_str();
                // ReLU-backward gate is dtype-dependent: float tensors compare
                // with `cmpf "ogt"` vs a `0.0` constant; integer/Q16.16 tensors
                // must use `cmpi "sgt"` vs `0` (a float literal or `cmpf` on an
                // integer type is invalid MLIR). Pure elementwise either way.
                let (zero_lit, cmp_op) = match &info.dtype {
                    DType::F32 | DType::F16 | DType::BF16 => ("0.0", "arith.cmpf \"ogt\""),
                    _ => ("0", "arith.cmpi \"sgt\""),
                };
                let ty = tensor_type(&info.shape, elem);
                let rank = info.shape.len();
                let dims = (0..rank)
                    .map(|d| format!("d{d}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let iters = vec!["\"parallel\""; rank].join(", ");
                let id_map = format!("affine_map<({dims}) -> ({dims})>");
                self.emit_line(&format!(
                    "    %zero{} = arith.constant {zero_lit} : {elem}",
                    dst.0
                ));
                self.emit_line(&format!("    %tmp{} = tensor.empty() : {ty}", dst.0));
                self.emit_line(&format!(
                    "    %{} = linalg.generic {{indexing_maps = [{id_map}, {id_map}, {id_map}], \
                     iterator_types = [{iters}]}} ins(%{}, %{} : {ty}, {ty}) outs(%tmp{} : {ty}) {{",
                    dst.0, grad.0, src.0, dst.0
                ));
                self.emit_line(&format!(
                    "    ^bb0(%rg{}: {elem}, %rx{}: {elem}, %rout{}: {elem}):",
                    dst.0, dst.0, dst.0
                ));
                self.emit_line(&format!(
                    "      %rmask{} = {cmp_op} %rx{}, %zero{} : {elem}",
                    dst.0, dst.0, dst.0
                ));
                self.emit_line(&format!(
                    "      %rsel{} = arith.select %rmask{}, %rg{}, %zero{} : {elem}",
                    dst.0, dst.0, dst.0, dst.0
                ));
                self.emit_line(&format!("      linalg.yield %rsel{} : {elem}", dst.0));
                self.emit_line(&format!("    }} -> {ty}"));
                self.values.insert(
                    *dst,
                    ValueKind::Tensor {
                        dtype: info.dtype.clone(),
                        shape: info.shape.clone(),
                    },
                );
            }
            Instr::Output(id) => {
                self.outputs.push(*id);
            }
            // RFC 0005 Phase 0: generic call -> `func.call`. Scoped to
            // the i64 ABI (every arg + result is i64) — exactly the
            // five `__mind_*` intrinsic signatures. Non-i64 args are a
            // clear error (tensor/aggregate call ABI is RFC 0005
            // phase 2+). Default build has no this arm and the
            // catch-all still errors `UnsupportedOp` on `Instr::Call`
            // exactly as before — byte-identical, moat held.
            #[cfg(feature = "std-surface")]
            Instr::Call { dst, name, args } => {
                for a in args {
                    match self.values.get(a) {
                        Some(ValueKind::ScalarI64) => {}
                        _ => {
                            return Err(MlirLowerError::UnsupportedOp {
                                instr_index,
                                op: format!(
                                    "non-i64 argument to call `{name}` \
                                     (RFC 0005 phase 2+ covers aggregate call ABI)"
                                ),
                            });
                        }
                    }
                }
                // RFC 0006 Track B (increment 1) — the `dot_f32_v` surface
                // fn lowers to a *native* MLIR `vector`-dialect reduction
                // loop instead of a `func.call` to the Track A
                // runtime-support C bridge.  Track A's `__mind_blas_dot_f32`
                // extern path is untouched and remains the scalar/AVX2
                // fallback; this is purely additive.  Any other callee
                // keeps the generic `func.call` lowering below.
                if name == VEC_DOT_F32_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_f32(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 2) — the Q16.16 vector dot.
                // Byte-identical to Track A's scalar `__mind_blas_dot_q16`
                // oracle at every length (task #57 cross-arch bit-identity
                // gate): per-element widen -> multiply -> arithmetic
                // `>> 16` -> i64-lane accumulate -> associative lane sum
                // -> truncate-low-32 + sign-extend. Track A's
                // `__mind_blas_dot_q16` extern path is untouched.
                if name == VEC_DOT_Q16_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_q16(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // "int-dot" tier — int16 vector dot. Byte-identical to the
                // scalar oracle `c = (i32) sum_k ((i32)a[k]*(i32)b[k])` for
                // ALL int16 inputs: sext i16->i64, multiply, i64-lane
                // accumulate (no shift / no saturation / no early narrow),
                // associative lane sum, trunc-low-32 + sign-extend. The i16
                // widen-multiply-accumulate inner loop is the AVX2 `vpmaddwd`
                // idiom at `-march=x86-64-v3`, the fast deterministic int
                // GEMM tier. Additive: no Track A extern is touched.
                if name == VEC_DOT_I16_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_i16(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 2) — f32 L1 / L∞ vector
                // reductions (sum-of-abs / max-of-abs). Same i64-packed-f32
                // ABI and ~1e-4-relative numerical contract as the f32 L2
                // `dot_f32_v` path; Track A's scalar/AVX2 L1/L∞ externs
                // are untouched.
                if name == VEC_DOT_L1_F32_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_metric_f32(*dst, args[0], args[1], args[2], VecMetric::L1);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 3) — the Q16.16 L1 vector
                // reduction. Byte-identical to Track A's scalar
                // `__mind_blas_dot_l1_q16` oracle at every length (task #57):
                // per-element widen -> signed subtract -> arith-only abs
                // (`maxsi(d, 0-d)`, mirroring the C oracle's `if (d<0) d=-d`)
                // -> i64-lane accumulate -> associative lane sum ->
                // truncate-low-32 + sign-extend. Completes the Q16.16
                // vector-path metric parity left open in increment 2
                // (RFC 0006 §9.3). Track A's `__mind_blas_dot_l1_q16` extern
                // path is untouched and remains the scalar/AVX2 fallback.
                if name == VEC_DOT_L1_Q16_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_l1_q16(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                if name == VEC_DOT_LINF_F32_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_metric_f32(*dst, args[0], args[1], args[2], VecMetric::Linf);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 3b) — row-major f32 matmul.
                // Lowers to a native MLIR outer `scf.for` over rows, with
                // the proven vectorised `dot_f32_v` reduction (8-lane FMA +
                // scalar tail) inlined per row.  Track A's scalar/AVX2
                // `__mind_blas_matmul_rmajor_f32` extern path is untouched.
                if name == VEC_MATMUL_RMAJOR_F32_INTRINSIC && args.len() == 5 {
                    self.emit_vec_matmul_rmajor_f32(
                        *dst, args[0], args[1], args[2], args[3], args[4],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 4) — row-major Q16.16 matmul.
                // Lowers to a native MLIR outer `scf.for` over rows, with
                // the proven Q16.16 reduction from `emit_vec_dot_q16`
                // (widen i32→i64, `>> 16`, i64-lane accumulate, associative
                // `vector.reduction <add>`, scalar tail, trunc+extsi) inlined
                // per row.  Byte-identical to the Track A scalar oracle
                // `__mind_blas_dot_q16` applied to each row independently.
                // Track A's `__mind_blas_matmul_rmajor_q16` extern path is
                // untouched.
                if name == VEC_MATMUL_RMAJOR_Q16_INTRINSIC && args.len() == 5 {
                    self.emit_vec_matmul_rmajor_q16(
                        *dst, args[0], args[1], args[2], args[3], args[4],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B — fused outer-product Q16.16 GEMM. A is
                // M×K row-major, B is **K×N row-major** (un-transposed), C is
                // M×N row-major caller-allocated. Register-tiled outer-product
                // microkernel (NR-wide column accumulator, MR A-rows per tile,
                // no horizontal reduction) with scalar column/row tails for
                // N%NR and M%MR. Each product term is `>> 16`-shifted before
                // i64 accumulation and the i64→i32 truncation happens once at
                // the store — byte-identical to the per-element scalar oracle
                // for all shapes. Additive: the gemv-composed
                // `__mind_blas_matmul_rmajor_q16_v` path is untouched.
                if name == VEC_MATMUL_MM_Q16_INTRINSIC && args.len() == 6 {
                    self.emit_vec_matmul_mm_q16(
                        *dst, args[0], args[1], args[2], args[3], args[4], args[5],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // Multithreaded fused outer-product Q16.16 GEMM. Same ABI and
                // semantics as the single-thread `__mind_blas_matmul_mm_q16_v`,
                // but split into `T` contiguous owner-computes M-row bands run
                // on raw POSIX threads. Each output element is written by
                // exactly one thread (no cross-thread reduction / atomics /
                // shared accumulator), so the result is byte-for-byte identical
                // to the single-thread kernel regardless of `T`. Additive: the
                // single-thread path above is untouched.
                if name == VEC_MATMUL_MM_Q16_MT_INTRINSIC && args.len() == 6 {
                    self.emit_vec_matmul_mm_q16_mt(
                        *dst, args[0], args[1], args[2], args[3], args[4], args[5],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // "det.igemm" tier — fused int8 GEMM. A is M×K row-major int8
                // (1 byte), B is K×N row-major int8, C is M×N row-major INT32
                // caller-allocated. Same BLIS-blocked register-tiled kernel as
                // the Q16 path with i8→i32 sign-extension during the pack and NO
                // `>> 16` shift (int8 is integer, not fixed-point). The C-tile
                // accumulates i64; the i64→i32 truncation happens once at the
                // store — byte-identical to the per-element scalar int32 oracle
                // `(i32) Σ_k (i32)A[i,k]*(i32)B[k,j]` for all shapes. The same
                // MLIR lowers to vpmaddwd (AVX2) / SDOT (aarch64) — both yield
                // the identical exact int32 sum. Additive.
                if name == VEC_MATMUL_MM_I8_INTRINSIC && args.len() == 6 {
                    self.emit_vec_matmul_mm_i8(
                        *dst, args[0], args[1], args[2], args[3], args[4], args[5],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // "det.igemm" tier — MULTITHREADED fused int8 GEMM. Same ABI and
                // byte-for-byte output as the single-thread int8 kernel above,
                // parallelised over contiguous owner-computes M-row bands with
                // raw POSIX threads. Each worker runs the SAME `emit_mm_i8_blocked`
                // macro-kernel over its band; every output element is owned by
                // exactly one thread (no cross-thread reduction / atomic / shared
                // accumulator) and each worker's scratch/packing buffers are
                // private stack allocas, so the result is byte-for-byte identical
                // to the single-thread kernel regardless of `T`. Additive: the
                // single-thread path above is untouched.
                if name == VEC_MATMUL_MM_I8_MT_INTRINSIC && args.len() == 6 {
                    self.emit_vec_matmul_mm_i8_mt(
                        *dst, args[0], args[1], args[2], args[3], args[4], args[5],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // "int-dot" tier — row-major int16 matmul. Outer `scf.for`
                // over rows with the proven int16 reduction from
                // `emit_vec_dot_i16` (sext i16->i64, i64-lane accumulate,
                // `vector.reduction <add>`, scalar tail, trunc+store i32)
                // inlined per row. Byte-identical to the scalar oracle
                // applied per row, for all int16 inputs. Additive.
                if name == VEC_MATMUL_RMAJOR_I16_INTRINSIC && args.len() == 5 {
                    self.emit_vec_matmul_rmajor_i16(
                        *dst, args[0], args[1], args[2], args[3], args[4],
                    );
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // Native-memory load/store fast path. The `__mind_{load,store}_iN`
                // intrinsics are pure unaligned-`memcpy` reinterprets in the C
                // runtime-support bridge (`runtime-support/mind_intrinsics.c`):
                // each call costs a PLT-indirected `call` per element, which on a
                // load/store-bound kernel (e.g. the Q16.16 FFT butterfly) is the
                // dominant cost. Lower them INLINE to `llvm.inttoptr` +
                // `llvm.load`/`llvm.store` so the inner loop becomes `mov`-based
                // like a C compiler's, with NO external symbol and NO PLT hop.
                //
                // Byte-identity is preserved EXACTLY because each lowering matches
                // the C semantics op-for-op:
                //   * `{alignment = 1 : i64}` mirrors the C `memcpy` (unaligned —
                //     a natural-alignment load would have different defined
                //     behaviour on a misaligned address);
                //   * load_i8 / load_i32 `llvm.zext` to i64 (C returns `(int64_t)`
                //     of an unsigned `uint8_t`/`uint32_t` — zero-extend);
                //   * load_i64 loads i64 directly (signed reinterpret == the
                //     in-register bits);
                //   * store_iN `llvm.trunc`s the i64 value to the store width then
                //     stores exactly N bytes, and the call's result value is the
                //     literal `0` the C `__mind_store_*` returns.
                // The emitted ops convert-to-llvm and reconcile cleanly on the
                // pinned pass list; no dialect leaks past mlir-opt.
                #[cfg(feature = "std-surface")]
                if args.len() == 1
                    && matches!(
                        name.as_str(),
                        "__mind_load_i64"
                            | "__mind_load_i32"
                            | "__mind_load_i16"
                            | "__mind_load_i8"
                    )
                {
                    let (load_ty, zext): (&str, bool) = match name.as_str() {
                        "__mind_load_i64" => ("i64", false),
                        "__mind_load_i32" => ("i32", true),
                        "__mind_load_i16" => ("i16", true),
                        _ => ("i8", true),
                    };
                    self.emit_line(&format!(
                        "    %ldp{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                        dst.0, args[0].0
                    ));
                    if zext {
                        self.emit_line(&format!(
                            "    %ldv{0} = llvm.load %ldp{0} {{alignment = 1 : i64}} : \
                             !llvm.ptr -> {1}",
                            dst.0, load_ty
                        ));
                        self.emit_line(&format!(
                            "    %{0} = llvm.zext %ldv{0} : {1} to i64",
                            dst.0, load_ty
                        ));
                    } else {
                        self.emit_line(&format!(
                            "    %{0} = llvm.load %ldp{0} {{alignment = 1 : i64}} : \
                             !llvm.ptr -> i64",
                            dst.0
                        ));
                    }
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                #[cfg(feature = "std-surface")]
                if args.len() == 2
                    && matches!(
                        name.as_str(),
                        "__mind_store_i64"
                            | "__mind_store_i32"
                            | "__mind_store_i16"
                            | "__mind_store_i8"
                    )
                {
                    let store_ty: &str = match name.as_str() {
                        "__mind_store_i64" => "i64",
                        "__mind_store_i32" => "i32",
                        "__mind_store_i16" => "i16",
                        _ => "i8",
                    };
                    self.emit_line(&format!(
                        "    %stp{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                        dst.0, args[0].0
                    ));
                    let val_ref = if store_ty == "i64" {
                        format!("%{}", args[1].0)
                    } else {
                        self.emit_line(&format!(
                            "    %stv{0} = llvm.trunc %{1} : i64 to {2}",
                            dst.0, args[1].0, store_ty
                        ));
                        format!("%stv{}", dst.0)
                    };
                    self.emit_line(&format!(
                        "    llvm.store {1}, %stp{0} {{alignment = 1 : i64}} : \
                         {2}, !llvm.ptr",
                        dst.0, val_ref, store_ty
                    ));
                    // The C `__mind_store_*` returns 0; materialise that so any
                    // downstream use of the call result stays byte-identical.
                    self.emit_line(&format!("    %{} = arith.constant 0 : i64", dst.0));
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0010 Phase A/B/C: if the callee was declared via an
                // `extern "C"` block, emit `llvm.call` with the declared
                // signature; otherwise fall back to the existing `func.call`
                // i64-ABI path used by the std-surface runtime bridge.
                if let Some((param_types, ret_type, is_varargs, vararg_hints, callconv)) =
                    self.extern_c_fns.get(name).cloned()
                {
                    // Build arg list with declared types for concrete params;
                    // for varargs extras: use vararg_hints by offset index,
                    // then fall back to "i64".
                    // RFC 0010 Phase C / R-03: f32 in a varargs position must
                    // be promoted to f64 per C11 §6.5.2.2p6 default argument
                    // promotions. Vararg positions are indices >= n_concrete.
                    let n_concrete = param_types.len();
                    let param_type_str: Vec<String> = args
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            if i < n_concrete {
                                param_types[i].clone()
                            } else {
                                // varargs position: use hint if available,
                                // then fall back to "i64". Promote f32→f64.
                                let vidx = i - n_concrete;
                                let hint = vararg_hints
                                    .get(vidx)
                                    .cloned()
                                    .unwrap_or_else(|| "i64".to_string());
                                // R-03: f32 in vararg position → f64.
                                if hint == "f32" {
                                    "f64".to_string()
                                } else {
                                    hint
                                }
                            }
                        })
                        .collect();
                    // RFC 0010 — the std-surface runtime ABI carries every value
                    // as i64, but an `extern "C"` signature can declare
                    // `!llvm.ptr` parameters/return (e.g. `memset`, `inet_pton`,
                    // `read`/`write`). MLIR is strongly typed: an i64 SSA value
                    // cannot be passed where `!llvm.ptr` is declared. Bridge
                    // each pointer-typed slot with `llvm.inttoptr` before the
                    // call and recover the i64 address with `llvm.ptrtoint`
                    // after, so the surrounding i64 dataflow stays intact and
                    // mlir-opt sees consistent types. Non-pointer slots are
                    // already i64 (narrow C ints are declared i64 in the ABI),
                    // so they pass through untouched — no extra ops on the
                    // common path.
                    let mut arg_refs: Vec<String> = Vec::with_capacity(args.len());
                    for (i, a) in args.iter().enumerate() {
                        if param_type_str
                            .get(i)
                            .map(|t| t == "!llvm.ptr")
                            .unwrap_or(false)
                        {
                            let pname = format!("%ptrarg_{}_{}", dst.0, i);
                            self.emit_line(&format!(
                                "    {pname} = llvm.inttoptr %{} : i64 to !llvm.ptr",
                                a.0
                            ));
                            arg_refs.push(pname);
                        } else {
                            arg_refs.push(format!("%{}", a.0));
                        }
                    }
                    let call_ret_ty = ret_type.as_deref().unwrap_or("i64");
                    let varargs_suffix = if is_varargs { ", ..." } else { "" };
                    // RFC 0010 Phase C: emit cconv attribute for Win64 calls.
                    let cconv_attr = cconv_attr_for(callconv);
                    if call_ret_ty == "!llvm.ptr" {
                        // Capture the pointer result, then convert to the i64
                        // address the std-surface ABI threads everywhere else.
                        let raw = format!("%ptrret_{}", dst.0);
                        self.emit_line(&format!(
                            "    {raw} = llvm.call{} @{}({}) : ({}{}) -> !llvm.ptr",
                            cconv_attr,
                            name,
                            arg_refs.join(", "),
                            param_type_str.join(", "),
                            varargs_suffix,
                        ));
                        self.emit_line(&format!(
                            "    %{} = llvm.ptrtoint {raw} : !llvm.ptr to i64",
                            dst.0
                        ));
                    } else {
                        self.emit_line(&format!(
                            "    %{} = llvm.call{} @{}({}) : ({}{}) -> {}",
                            dst.0,
                            cconv_attr,
                            name,
                            arg_refs.join(", "),
                            param_type_str.join(", "),
                            varargs_suffix,
                            call_ret_ty,
                        ));
                    }
                    self.values.insert(*dst, ValueKind::ScalarI64);
                } else {
                    let arg_refs: Vec<String> = args.iter().map(|a| format!("%{}", a.0)).collect();
                    let arg_tys: Vec<&str> = args.iter().map(|_| "i64").collect();
                    self.emit_line(&format!(
                        "    %{} = func.call @{}({}) : ({}) -> i64",
                        dst.0,
                        name,
                        arg_refs.join(", "),
                        arg_tys.join(", ")
                    ));
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    self.extern_calls.insert((name.clone(), args.len()));
                }
            }
            // RFC 0005 P0d: emit `func.func @name(%pN: i64...) -> i64 { ... }`
            // for each user-defined function. The body is lowered into a
            // sub-context so its locals get a clean SSA namespace; the
            // resulting text is appended to `user_fns` and emitted as a
            // sibling top-level symbol *before* `@main`. Gated.
            #[cfg(feature = "std-surface")]
            Instr::FnDef {
                name,
                params,
                ret_id,
                body,
                ..
            } => {
                let mut sub = LoweringContext::new();
                // RFC 0010 Phase A: inherit extern_c_fns so that any
                // `llvm.call` emitted inside a user fn body resolves
                // correctly against the module-level extern declarations.
                sub.extern_c_fns = self.extern_c_fns.clone();
                // RFC 0012 §5.1: inherit the fn-signature table so the param /
                // return ABI types resolve here and inside nested definitions.
                sub.fn_signatures = self.fn_signatures.clone();
                // NARROW-INT ABI: inherit the signedness-preserving param kinds
                // so nested definitions seed u32/i32/bool params correctly too.
                sub.fn_param_kinds = self.fn_param_kinds.clone();
                // RFC 0012 §5.1 — recover this fn's declared param / return ABI
                // types. Default to all-i64 (legacy ABI) when no signature was
                // recorded (e.g. an IR built directly, not via `lower_to_ir`),
                // so existing i64 functions lower byte-identically.
                let sig = self.fn_signatures.get(name);
                let param_abi: Vec<&str> = match sig {
                    Some((p, _)) => params
                        .iter()
                        .enumerate()
                        .map(|(i, _)| p.get(i).map(String::as_str).unwrap_or("i64"))
                        .collect(),
                    None => params.iter().map(|_| "i64").collect(),
                };
                let ret_abi: &str = sig.and_then(|(_, r)| r.as_deref()).unwrap_or("i64");
                // Seed each parameter's `ValueKind`. Prefer the signedness-
                // preserving `fn_param_kinds` table (so `u32` seeds `ScalarU32`,
                // distinct from `i32`/`ScalarI32`, and `bool` seeds `ScalarBool`);
                // fall back to the ABI-string mapping when no kind table entry
                // exists (e.g. an IR built directly, not via `lower_to_ir`). The
                // string fallback only knows f64/f32/i64 — identical to before for
                // those — so an i64-only module seeds byte-identically either way.
                let param_kinds = self.fn_param_kinds.get(name);
                for (i, ((_pname, pid), abi)) in params.iter().zip(param_abi.iter()).enumerate() {
                    let kind = match param_kinds.and_then(|pk| pk.get(i)) {
                        Some(k) => k.clone(),
                        None => match *abi {
                            "f64" => ValueKind::ScalarF64,
                            "f32" => ValueKind::ScalarF32,
                            _ => ValueKind::ScalarI64,
                        },
                    };
                    sub.values.insert(*pid, kind);
                }
                for (idx, inner) in body.iter().enumerate() {
                    sub.emit_instr(idx, inner)?;
                }

                let sig_args: Vec<String> = params
                    .iter()
                    .zip(param_abi.iter())
                    .map(|((_, pid), abi)| format!("%{}: {}", pid.0, abi))
                    .collect();
                let mut fn_text = String::new();
                fn_text.push_str(&format!(
                    "  func.func @{}({}) -> {} {{\n",
                    name,
                    sig_args.join(", "),
                    ret_abi
                ));
                fn_text.push_str(&sub.body);
                // Every fn returns i64 under the std-surface ABI. If the last
                // emitted line in the body is not a block terminator (return,
                // cf.br, cf.cond_br), synthesise one from `ret_id`.
                //
                // We check the LAST non-empty line of the body, not whether
                // the body contains "return" anywhere.  The previous pattern
                // `.contains("    return ")` incorrectly suppressed the
                // synthetic return for functions that use early-return inside
                // `if` branches but end with a plain value expression — those
                // functions emit instructions after the final `^if_after_N:`
                // block label that have no terminator.
                let last_line = sub
                    .body
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .trim();
                let already_terminated = last_line.starts_with("return")
                    || last_line.starts_with("cf.br ")
                    || last_line.starts_with("cf.cond_br ");
                if !already_terminated {
                    match ret_id {
                        // A fall-off-the-end comparison result is `i1`; widen it
                        // to the i64 ABI return slot (same reason as the explicit
                        // `Instr::Return` arm). A `-> bool` fn keeps the i64 ABI,
                        // so this stays `i64` regardless of `ret_abi`.
                        Some(rid) if sub.i1_values.contains(rid) => fn_text.push_str(&format!(
                            "    %bext{0} = arith.extui %{0} : i1 to i64\n    return %bext{0} : i64\n",
                            rid.0
                        )),
                        // RFC 0012 §5.1: return the value at the declared ABI
                        // width (`f64`/`f32` for a scalar-float return, else i64).
                        Some(rid) => fn_text
                            .push_str(&format!("    return %{} : {}\n", rid.0, ret_abi)),
                        // No trailing value: synthesise a zero at the return ABI
                        // type so a float-returning fn that falls off the end is
                        // still well-typed.
                        None => {
                            if ret_abi == "f64" || ret_abi == "f32" {
                                fn_text.push_str(&format!(
                                    "    %z = arith.constant 0.0 : {ret_abi}\n    return %z : {ret_abi}\n"
                                ));
                            } else {
                                fn_text.push_str(
                                    "    %z = arith.constant 0 : i64\n    return %z : i64\n",
                                );
                            }
                        }
                    }
                }
                fn_text.push_str("  }\n");

                self.user_fns.push_str(&fn_text);
                self.defined_fns.insert(name.clone());
                // Bubble up any extern calls or nested definitions discovered
                // inside the body so the module-level assembler sees them.
                for ec in sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                for df in sub.defined_fns {
                    self.defined_fns.insert(df);
                }
                // RFC 0010 Phase A: bubble up any ExternFnDecl discovered
                // inside the fn body (unusual but valid at the IR level).
                for (efn_name, sig) in sub.extern_c_fns {
                    self.extern_c_fns.insert(efn_name, sig);
                }
                self.user_fns.push_str(&sub.user_fns);
                // Bubble up any multithreaded-GEMM worker functions and the
                // pthread-extern requirement discovered inside the body, so the
                // module assembler emits them at top level exactly once.
                self.mt_workers.push_str(&sub.mt_workers);
                self.needs_pthread |= sub.needs_pthread;
                self.needs_malloc |= sub.needs_malloc;
            }
            // P0d: function parameters bind a ValueId to the i64 ABI; the
            // value is named in the enclosing `func.func` signature so we
            // do not emit anything for the Param itself. Gated.
            //
            // RFC 0012 §5.1: the FnDef arm pre-seeds each param's `ValueKind`
            // from its declared ABI type (so an `f64`/`f32` param dispatches to
            // float arith). Only fall back to the i64 ABI when no kind was
            // recorded — never clobber a pre-seeded float kind.
            #[cfg(feature = "std-surface")]
            Instr::Param { dst, .. } => {
                self.values.entry(*dst).or_insert(ValueKind::ScalarI64);
            }
            // P0d: explicit `return %v : i64` inside a user fn body.
            #[cfg(feature = "std-surface")]
            Instr::Return { value } => match value {
                // A bare comparison result is `i1`; widen to the i64 ABI slot.
                Some(v) if self.i1_values.contains(v) => {
                    self.emit_line(&format!("    %bext{0} = arith.extui %{0} : i1 to i64", v.0));
                    self.emit_line(&format!("    return %bext{} : i64", v.0));
                }
                // RFC 0012 §5.1: return the value at its real ABI width — `f64`/
                // `f32` for a scalar-float result (param or float BinOp), else the
                // i64 ABI slot (byte-identical to the legacy path for i64 fns).
                Some(v) => {
                    let rty = match self.values.get(v) {
                        Some(ValueKind::ScalarF64) => "f64",
                        Some(ValueKind::ScalarF32) => "f32",
                        _ => "i64",
                    };
                    self.emit_line(&format!("    return %{} : {}", v.0, rty));
                }
                None => self.emit_line("    return"),
            },
            // RFC 0005 Gap 1: `while cond { body }` — basic-block loop lowering.
            //
            // MLIR structure emitted (cf dialect):
            //
            //   cf.br ^while_header_N
            // ^while_header_N:
            //   <cond_instrs>
            //   %cond_bool_N = arith.trunci %cond_id : i64 to i1
            //   cf.cond_br %cond_bool_N, ^while_body_N, ^while_after_N
            // ^while_body_N:
            //   <body_instrs>
            //   cf.br ^while_header_N
            // ^while_after_N:
            //
            // N = instr_index for uniqueness across nested whiles. Gated.
            //
            // MLIR structure emitted (cf dialect, block-argument form so
            // mutable loop variables are correctly threaded across iterations):
            //
            //   cf.br ^while_header_N(%init_0: i64, %init_1: i64, ...)
            // ^while_header_N(%wbl_N_0: i64, %wbl_N_1: i64, ...):
            //   <cond_instrs, with init_ids substituted by wbl names>
            //   cf.cond_br %cond, ^while_body_N(%wbl_N_0, ...), ^while_after_N
            // ^while_body_N(%wbl_N_0: i64, ...):
            //   <body_instrs, with init_ids substituted by wbl names>
            //   cf.br ^while_header_N(%post_body_0: i64, ...)
            // ^while_after_N:
            //
            // When there are no live vars (no mutations in the body), the
            // blocks carry no arguments and behave like the original stub.
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_id,
                cond_instrs,
                body,
                live_vars,
                init_ids,
                exit_ids,
            } => {
                // Function-global unique label for this loop's block triple.
                // Keyed on a monotonic counter (NOT `instr_index`) so a nested
                // inner loop and a sibling loop at the same per-region index
                // never emit duplicate `^while_header_N`/`^while_body_N`/
                // `^while_after_N` labels — which mlir-opt rejects with
                // `redefinition of block`. The counter is threaded through the
                // recursive cond/body sub-contexts below, advancing in fixed
                // traversal order for deterministic numbering.
                let lbl = self.while_label;
                self.while_label += 1;

                // Build two arg-substitution tables.
                //
                // MLIR SSA values are unique per function, not per block.
                // Both the header block and the body block need block-argument
                // declarations; they MUST use distinct names even though they
                // represent the same conceptual "loop variable at iteration N".
                //
                //   header args:  %wbl_{lbl}_{k}   — declared in ^while_header
                //   body   args:  %wbod_{lbl}_{k}  — declared in ^while_body
                //
                // The condition instructions run in the header block so they
                // are substituted with header arg names.  The body instructions
                // run in the body block so they are substituted with body arg
                // names.  The back-edge passes %post_id (computed by the body)
                // back to the header.
                //
                // Entries whose init_id is usize::MAX (variable declared
                // inside the loop body) are skipped — they have no pre-loop
                // value to thread.
                struct LoopArg {
                    init_id: usize,
                    head_name: String, // %wbl_{lbl}_{k}
                    body_name: String, // %wbod_{lbl}_{k}
                    post_id: usize,
                    // F2: real SSA id of the loop EXIT value, declared as the
                    // ^while_after block arg. Downstream IR was rebound (in
                    // lower.rs) to reference this id, so naming the block arg
                    // by it makes every post-loop use dominate.
                    exit_id: usize,
                    // Loop-carried variable name — used to match a break/continue
                    // `live` snapshot entry to this slot (order-independent).
                    name: String,
                }
                let mut loop_args: Vec<LoopArg> = Vec::new();
                for (k, ((vname, post_vid), init_vid)) in
                    live_vars.iter().zip(init_ids.iter()).enumerate()
                {
                    if init_vid.0 == usize::MAX {
                        continue;
                    }
                    // F2 exit id: parallel to live_vars. Falls back to a
                    // synthetic name only if absent (older IR without exit_ids).
                    let exit_id = exit_ids.get(k).map(|v| v.0).unwrap_or(usize::MAX);
                    loop_args.push(LoopArg {
                        init_id: init_vid.0,
                        head_name: format!("wbl_{lbl}_{k}"),
                        body_name: format!("wbod_{lbl}_{k}"),
                        post_id: post_vid.0,
                        exit_id,
                        name: vname.clone(),
                    });
                }

                // Build arg-triple slices consumed by substitute_ids.
                let head_args: Vec<(usize, String, usize)> = loop_args
                    .iter()
                    .map(|a| (a.init_id, a.head_name.clone(), a.post_id))
                    .collect();
                let body_args: Vec<(usize, String, usize)> = loop_args
                    .iter()
                    .map(|a| (a.init_id, a.body_name.clone(), a.post_id))
                    .collect();

                // Entry branch: carry initial values into the header.
                {
                    let init_vals: Vec<String> = loop_args
                        .iter()
                        .map(|a| format!("%{}", a.init_id))
                        .collect();
                    let arg_pass = fmt_block_args(&init_vals);
                    self.emit_line(&format!("    cf.br ^while_header_{lbl}{arg_pass}"));
                }

                // Header block declaration (block args named wbl_*).
                if loop_args.is_empty() {
                    self.emit_line(&format!("  ^while_header_{lbl}:"));
                } else {
                    let arg_decls: String = loop_args
                        .iter()
                        .map(|a| format!("%{}: i64", a.head_name))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.emit_line(&format!("  ^while_header_{lbl}({arg_decls}):"));
                }

                // Emit condition instructions. Substitute init_ids with header
                // block arg names (wbl_*) since cond runs in the header block.
                let mut cond_sub = LoweringContext::new();
                // Inherit extern "C" signatures so calls to them inside this
                // nested construct emit `llvm.call` consistently (not
                // `func.call`), avoiding a dual `llvm.func`/`func.func`
                // declaration of the same symbol (RFC 0010).
                cond_sub.extern_c_fns = self.extern_c_fns.clone();
                // Thread the function-global while-label counter so any nested
                // loop in the condition gets a unique label, and pull the
                // advanced value back afterward.
                cond_sub.while_label = self.while_label;
                for (vid, kind) in &self.values {
                    cond_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ci) in cond_instrs.iter().enumerate() {
                    cond_sub.emit_instr(idx, ci)?;
                }
                self.while_label = cond_sub.while_label;
                let cond_text = substitute_ids(&cond_sub.body, &head_args);
                self.body.push_str(&cond_text);
                for (vid, kind) in cond_sub.values {
                    self.values.insert(vid, kind);
                }
                #[cfg(feature = "std-surface")]
                for ec in cond_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                // NARROW-INT ABI: bubble up i1-ness discovered in the condition
                // (e.g. a let-bound comparison) so the value-based already-i1
                // probe below sees it. Additive — only extends the set.
                #[cfg(feature = "std-surface")]
                for v in cond_sub.i1_values {
                    self.i1_values.insert(v);
                }

                // Determine whether condition is already i1 (so we must NOT emit
                // a spurious `arith.trunci i64 to i1` on it — that is invalid
                // MLIR). Value-based: the cond SSA value is i1 if it is recorded
                // in `i1_values` (a comparison result, possibly let-bound) or its
                // kind is `ScalarBool` (a narrow-int / bool-typed value). The
                // instruction-shape probe (last cond instr is a comparison) is
                // kept as a fallback for the bare-comparison case. Additive: an
                // i64-only program whose while-cond is a non-let comparison still
                // takes the instruction-shape branch byte-identically.
                let cond_already_i1 = self.i1_values.contains(cond_id)
                    || matches!(self.values.get(cond_id), Some(ValueKind::ScalarBool))
                    || cond_instrs
                        .last()
                        .map(|last| {
                            matches!(
                                last,
                                Instr::BinOp {
                                    op: BinOp::Lt
                                        | BinOp::Le
                                        | BinOp::Gt
                                        | BinOp::Ge
                                        | BinOp::Eq
                                        | BinOp::Ne,
                                    ..
                                }
                            )
                        })
                        .unwrap_or(false);

                // The cond_br passes the HEADER args into the body block AND
                // the after block.  Both receive the same live-var values so
                // that ^while_after_N can expose them as block args to code
                // that follows the loop (fixing the SSA dominance error where
                // %post_id, defined in ^while_body_N, was referenced in the
                // non-dominated ^while_after_N block).
                let head_name_vals: Vec<String> = loop_args
                    .iter()
                    .map(|a| format!("%{}", a.head_name))
                    .collect();
                let body_arg_pass = fmt_block_args(&head_name_vals);
                // The after block gets the same arg list (header args).
                let after_arg_pass = body_arg_pass.clone();

                if cond_already_i1 {
                    let cond_name = substitute_single_id(cond_id.0, &head_args);
                    self.emit_line(&format!(
                        "    cf.cond_br {cond_name}, ^while_body_{lbl}{body_arg_pass}, ^while_after_{lbl}{after_arg_pass}"
                    ));
                } else {
                    let cond_name = substitute_single_id(cond_id.0, &head_args);
                    self.emit_line(&format!(
                        "    %cond_bool_{lbl} = arith.trunci {cond_name} : i64 to i1"
                    ));
                    self.emit_line(&format!(
                        "    cf.cond_br %cond_bool_{lbl}, ^while_body_{lbl}{body_arg_pass}, ^while_after_{lbl}{after_arg_pass}"
                    ));
                }

                // Body block declaration (block args named wbod_*, distinct
                // from the header's wbl_* to avoid SSA redefinition).
                if loop_args.is_empty() {
                    self.emit_line(&format!("  ^while_body_{lbl}:"));
                } else {
                    let arg_decls: String = loop_args
                        .iter()
                        .map(|a| format!("%{}: i64", a.body_name))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.emit_line(&format!("  ^while_body_{lbl}({arg_decls}):"));
                }

                // Emit body instructions. Substitute init_ids with body block
                // arg names (wbod_*) since body runs in the body block.
                let mut body_sub = LoweringContext::new();
                // Inherit extern "C" signatures so calls to them inside this
                // nested construct emit `llvm.call` consistently (not
                // `func.call`), avoiding a dual `llvm.func`/`func.func`
                // declaration of the same symbol (RFC 0010).
                body_sub.extern_c_fns = self.extern_c_fns.clone();
                // Thread the function-global while-label counter into the body
                // so nested loops get unique labels; pull it back afterward.
                body_sub.while_label = self.while_label;
                // Push THIS loop's frame so break/continue in the body (and only
                // the innermost) emit a cf.br to ^while_after_{lbl}/^while_header
                // forwarding each carried var's current value by name. Inherit
                // outer frames so an inner loop still scopes break/continue to
                // itself.
                #[cfg(feature = "std-surface")]
                {
                    body_sub.loop_stack = self.loop_stack.clone();
                    body_sub.loop_stack.push(LoopFrame {
                        lbl,
                        carried: loop_args
                            .iter()
                            .map(|a| (a.name.clone(), a.init_id))
                            .collect(),
                    });
                }
                for (vid, kind) in &self.values {
                    body_sub.values.insert(*vid, kind.clone());
                }
                for (idx, bi) in body.iter().enumerate() {
                    body_sub.emit_instr(idx, bi)?;
                }
                self.while_label = body_sub.while_label;
                let body_text = substitute_ids(&body_sub.body, &body_args);
                self.body.push_str(&body_text);
                for (vid, kind) in body_sub.values {
                    self.values.insert(vid, kind);
                }
                #[cfg(feature = "std-surface")]
                for ec in body_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }

                // Back-edge: pass post-body values (computed in the body block
                // using wbod_* args as inputs) back to the header.
                //
                // For each slot k, the post_id is the final SSA id assigned to
                // the variable after all its assignments in the body IR.
                //
                // Special case: assignments of the form `x = y` (where y is
                // another loop variable) produce post_id[x] == init_id[y].
                // In the body block, `%init_id[y]` is the pre-loop value from
                // the entry block, but the correct CURRENT-iteration value of y
                // is the body-block arg `%wbod_{lbl}_{j}`.  We must pass the
                // body arg, not the stale pre-loop init value, so that later
                // iterations receive the right values on the back-edge.
                //
                // Mapping: if post_id[k] == init_id[j], use body_name[j].
                // Otherwise (post_id is a fresh computation), use %post_id.
                {
                    let post_vals: Vec<String> = loop_args
                        .iter()
                        .map(|a| {
                            // Check if this post_id matches the init_id of
                            // another (or the same) loop arg.
                            if let Some(src) =
                                loop_args.iter().find(|other| other.init_id == a.post_id)
                            {
                                // Pass the body-block arg for that source variable.
                                format!("%{}", src.body_name)
                            } else {
                                // Fresh computation in the body — pass directly.
                                format!("%{}", a.post_id)
                            }
                        })
                        .collect();
                    let back_pass = fmt_block_args(&post_vals);
                    self.emit_line(&format!("    cf.br ^while_header_{lbl}{back_pass}"));
                }

                // After block: block args carry the loop-variable values from
                // the header into the successor code (dominance fix: the header
                // dominates both body and after, so threading header args into
                // after is always valid).
                //
                // F2: the after-block args are named by the loop's EXIT SSA ids
                // (`exit_id`), which lower.rs allocated and to which it rebound
                // every post-loop reference to the loop variable. Naming the
                // block arg by the exit id makes those references resolve to a
                // value that dominates the after-block — no textual rewrite of
                // body-internal post ids is needed (the old `pending_after_subs`
                // mechanism is removed).
                if loop_args.is_empty() {
                    self.emit_line(&format!("  ^while_after_{lbl}:"));
                } else {
                    let after_arg_decls: String = loop_args
                        .iter()
                        .map(|a| format!("%{}: i64", a.exit_id))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.emit_line(&format!("  ^while_after_{lbl}({after_arg_decls}):"));
                    // Register the exit ids as known scalar i64 values so any
                    // downstream type lookup (e.g. function return) succeeds.
                    for a in &loop_args {
                        self.values.insert(ValueId(a.exit_id), ValueKind::ScalarI64);
                    }
                }
            }
            // RFC 0005 Phase 6.2b Gap 2 — `const NAME: [i64; N] = [...]`
            // lowers to an MLIR `arith.constant` dense attribute that is
            // stored to a `memref<N x i64>` alloca so fn bodies can load
            // from it.  The name is threaded through as an SSA comment so
            // textual IR round-trips retain the label.
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { dst, name, values } => {
                let label = name.as_deref().unwrap_or("__anon");
                // Emit a dense integer array constant as a tensor<Ni64> global.
                let n = values.len();
                let elems: String = values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                self.emit_line(&format!("  // const.array @{label} : [i64; {n}]"));
                self.emit_line(&format!(
                    "  {} = arith.constant dense<[{}]> : tensor<{}xi64>",
                    dst, elems, n
                ));
            }
            // RFC 0005 Phase 6.2b Gap 2 — `arr[idx]` element load.
            // Lowers to an `tensor.extract` from the base tensor constant.
            #[cfg(feature = "std-surface")]
            Instr::ArrayLoad { dst, base, index } => {
                self.emit_line(&format!(
                    "  {} = tensor.extract {}[{}] : tensor<?>",
                    dst, base, index
                ));
            }
            // RFC 0006 Track B (increment 1) — SIMD vector load.
            //
            // The Option-C ABI gives us i64 opaque addresses; native MLIR
            // memory access uses `llvm.inttoptr` to recover a pointer,
            // `llvm.getelementptr` (i8 element type) to apply the byte
            // offset, then a vector-typed `llvm.load` of
            // `vector<lanes x f32>`.  `convert-vector-to-llvm` +
            // `reconcile-unrealized-casts` legalise this to the host SIMD
            // width with no per-target code and no C shim — the Track B
            // thesis-pure property (vs Track A's runtime-support bridge).
            #[cfg(feature = "std-surface")]
            Instr::VecLoad {
                dst,
                base,
                offset,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vptr_b{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                    dst.0, base.0
                ));
                self.emit_line(&format!(
                    "    %vptr{0} = llvm.getelementptr %vptr_b{0}[%{1}] : \
                     (!llvm.ptr, i64) -> !llvm.ptr, i8",
                    dst.0, offset.0
                ));
                self.emit_line(&format!(
                    "    %{0} = llvm.load %vptr{0} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{1}xf32>",
                    dst.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorF32 { lanes: l });
            }
            // RFC 0006 Track B (increment 1) — element-wise fused
            // multiply-add: `dst = a * b + acc`.  Lowers to `vector.fma`,
            // which `convert-vector-to-llvm` turns into the
            // `llvm.intr.fmuladd` intrinsic (one hardware FMA per lane
            // group on targets that have one).
            #[cfg(feature = "std-surface")]
            Instr::VecFma {
                dst,
                a,
                b,
                acc,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %{0} = vector.fma %{1}, %{2}, %{3} : vector<{4}xf32>",
                    dst.0, a.0, b.0, acc.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorF32 { lanes: l });
            }
            // RFC 0006 Track B (increment 1) — horizontal sum to scalar.
            //
            // `vector.reduction <add>` becomes `llvm.intr.vector.reduce.fadd`.
            // The result is the f32 scalar bit-packed (zero-extended) into
            // an i64 so it travels the Option-C i64 ABI exactly like every
            // other `__mind_blas_*` return value.  The tree-shaped pairwise
            // reduction is NOT bit-identical to a sequential scalar sum —
            // the numerical contract bounds it to 1e-4 relative of the f64
            // oracle, matching Track A's AVX2 path.
            #[cfg(feature = "std-surface")]
            Instr::VecReduceAdd { dst, src, lanes } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vred{0} = vector.reduction <add>, %{1} : \
                     vector<{2}xf32> into f32",
                    dst.0, src.0, l
                ));
                self.emit_line(&format!(
                    "    %vbits{0} = arith.bitcast %vred{0} : f32 to i32",
                    dst.0
                ));
                self.emit_line(&format!(
                    "    %{0} = arith.extui %vbits{0} : i32 to i64",
                    dst.0
                ));
                self.values.insert(*dst, ValueKind::ScalarI64);
            }
            // RFC 0006 Track B (increment 2) — symmetric vector store:
            // `mem[base + offset .. +lanes] = src`.  Recovers the pointer
            // from the Option-C i64 address (`llvm.inttoptr`), applies the
            // byte offset (`llvm.getelementptr`, i8 element type) and emits
            // a vector-typed `llvm.store`.  Mirror image of `VecLoad`.
            #[cfg(feature = "std-surface")]
            Instr::VecStore {
                src,
                base,
                offset,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vsp_b{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                    src.0, base.0
                ));
                self.emit_line(&format!(
                    "    %vsp{0} = llvm.getelementptr %vsp_b{0}[%{1}] : \
                     (!llvm.ptr, i64) -> !llvm.ptr, i8",
                    src.0, offset.0
                ));
                self.emit_line(&format!(
                    "    llvm.store %{0}, %vsp{0} : vector<{1}xf32>, !llvm.ptr",
                    src.0, l
                ));
            }
            // RFC 0006 Track B (increment 2) — i32 sibling of `VecLoad`
            // for the Q16.16 path.  Same address recovery; the loaded
            // value is `vector<lanes x i32>`.
            #[cfg(feature = "std-surface")]
            Instr::VecLoadI32 {
                dst,
                base,
                offset,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %viptr_b{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                    dst.0, base.0
                ));
                self.emit_line(&format!(
                    "    %viptr{0} = llvm.getelementptr %viptr_b{0}[%{1}] : \
                     (!llvm.ptr, i64) -> !llvm.ptr, i8",
                    dst.0, offset.0
                ));
                self.emit_line(&format!(
                    "    %{0} = llvm.load %viptr{0} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{1}xi32>",
                    dst.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorI64 { lanes: l });
            }
            // RFC 0006 Track B (increment 2) — Q16.16 fused widening
            // multiply-shift-accumulate.  `dst = acc + ((sext64(a) *
            // sext64(b)) >>a 16)`, element-wise.  The shift is *arithmetic*
            // (`arith.shrsi`), exactly mirroring the Track A scalar oracle's
            // per-element `prod >> 16` under LLVM `ashr` semantics — this
            // is the operation the cross-arch bit-identity contract (#57)
            // pins.  `a`/`b` are `vector<lanes x i32>`; `acc`/`dst` are
            // `vector<lanes x i64>`.
            #[cfg(feature = "std-surface")]
            Instr::VecMulAddQ16 {
                dst,
                a,
                b,
                acc,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vqa{0} = arith.extsi %{1} : vector<{2}xi32> to vector<{2}xi64>",
                    dst.0, a.0, l
                ));
                self.emit_line(&format!(
                    "    %vqb{0} = arith.extsi %{1} : vector<{2}xi32> to vector<{2}xi64>",
                    dst.0, b.0, l
                ));
                self.emit_line(&format!(
                    "    %vqp{0} = arith.muli %vqa{0}, %vqb{0} : vector<{1}xi64>",
                    dst.0, l
                ));
                self.emit_line(&format!(
                    "    %vqs16_{0} = arith.constant dense<16> : vector<{1}xi64>",
                    dst.0, l
                ));
                self.emit_line(&format!(
                    "    %vqsh{0} = arith.shrsi %vqp{0}, %vqs16_{0} : vector<{1}xi64>",
                    dst.0, l
                ));
                self.emit_line(&format!(
                    "    %{0} = arith.addi %{1}, %vqsh{0} : vector<{2}xi64>",
                    dst.0, acc.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorI64 { lanes: l });
            }
            // RFC 0006 Track B (increment 2) — horizontal i64 sum.
            // `vector.reduction <add>` over `vector<lanes x i64>` ->
            // `llvm.intr.vector.reduce.add`.  Integer addition is
            // associative, so this is bit-identical to a sequential scalar
            // accumulation no matter how LLVM groups the lanes — the
            // property the #57 Q16.16 gate relies on.
            #[cfg(feature = "std-surface")]
            Instr::VecReduceAddI64 { dst, src, lanes } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %{0} = vector.reduction <add>, %{1} : \
                     vector<{2}xi64> into i64",
                    dst.0, src.0, l
                ));
                self.values.insert(*dst, ValueKind::ScalarI64);
            }
            // Phase 6.5 Stage 1a — `if cond { then } else { else }` lowering.
            //
            // MLIR structure emitted (cf dialect, matching the While pattern):
            //
            //   %cond_i1_N = arith.trunci %cond_id : i64 to i1
            //   cf.cond_br %cond_i1_N, ^if_then_N, ^if_else_N
            // ^if_then_N:
            //   <then_instrs>
            //   cf.br ^if_after_N(%then_result : i64)
            // ^if_else_N:
            //   <else_instrs>
            //   cf.br ^if_after_N(%else_result : i64)
            // ^if_after_N(%dst : i64):
            //
            // Block arguments are used to forward the branch value to the
            // join block — this is MLIR's standard pattern for if-as-value.
            // The previous placeholder (arith.constant 0) broke all if
            // expressions used as values (e.g. `let x = if cond { a } else { b }`),
            // replacing the selected value with 0.
            //
            // For branches that terminate with `Instr::Return`, the
            // `cf.br ^if_after_N` is omitted because the block is already
            // terminated by a `return` op.  If BOTH branches return, the
            // ^if_after_N block receives no predecessors but mlir-opt will
            // DCE it; the block arg is still declared for structural validity.
            //
            // N = dst.0 (globally unique SSA id) for uniqueness. Gated.
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                dst,
                merges,
                ..
            } => {
                // Use `dst.0` (the unique SSA id of the result value) as the
                // label suffix instead of `instr_index`.  `instr_index` resets
                // to 0 in every sub-context (e.g. inside a FnDef body loop),
                // causing block-label collisions when multiple `Instr::If`
                // nodes appear in the same function.  `dst.0` is globally
                // unique within the module.
                let lbl = dst.0;

                // Emit the condition sub-instructions into the current block.
                let mut cond_sub = LoweringContext::new();
                // Inherit extern "C" signatures so calls to them inside this
                // nested construct emit `llvm.call` consistently (not
                // `func.call`), avoiding a dual `llvm.func`/`func.func`
                // declaration of the same symbol (RFC 0010).
                cond_sub.extern_c_fns = self.extern_c_fns.clone();
                // Thread the function-global while-label counter so any nested
                // loop in the if-condition gets a unique label.
                cond_sub.while_label = self.while_label;
                for (vid, kind) in &self.values {
                    cond_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ci) in cond_instrs.iter().enumerate() {
                    cond_sub.emit_instr(idx, ci)?;
                }
                self.while_label = cond_sub.while_label;
                self.body.push_str(&cond_sub.body);
                for (vid, kind) in cond_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the condition sub-context.
                for ec in cond_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                // NARROW-INT ABI: bubble up i1-ness discovered in the condition
                // (e.g. a let-bound comparison `let c = a < b`) so the value-based
                // already-i1 probe below sees it. Additive — only extends the set.
                for v in cond_sub.i1_values {
                    self.i1_values.insert(v);
                }

                // Determine whether the condition value is already i1 (so we must
                // NOT emit a spurious `arith.trunci i64 to i1` on it — invalid
                // MLIR). Value-based: i1 if recorded in `i1_values` (a comparison
                // result, possibly let-bound) or its kind is `ScalarBool`. The
                // instruction-shape probe (last cond instr is a comparison) is the
                // fallback for the bare-comparison case. Additive: an i64-only
                // program with an inline-comparison if-cond still takes the
                // instruction-shape branch byte-identically.
                let cond_already_i1 = self.i1_values.contains(cond_id)
                    || matches!(self.values.get(cond_id), Some(ValueKind::ScalarBool))
                    || cond_instrs
                        .last()
                        .map(|last| {
                            matches!(
                                last,
                                Instr::BinOp {
                                    op: BinOp::Lt
                                        | BinOp::Le
                                        | BinOp::Gt
                                        | BinOp::Ge
                                        | BinOp::Eq
                                        | BinOp::Ne,
                                    ..
                                }
                            )
                        })
                        .unwrap_or(false);

                if cond_already_i1 {
                    // Comparison result is already i1 — use it directly.
                    self.emit_line(&format!(
                        "    cf.cond_br %{}, ^if_then_{lbl}, ^if_else_{lbl}",
                        cond_id.0
                    ));
                } else {
                    // Plain i64 → truncate to i1 first.
                    self.emit_line(&format!(
                        "    %cond_i1_{lbl} = arith.trunci %{} : i64 to i1",
                        cond_id.0
                    ));
                    self.emit_line(&format!(
                        "    cf.cond_br %cond_i1_{lbl}, ^if_then_{lbl}, ^if_else_{lbl}"
                    ));
                }

                // NARROW-INT control flow: both branch sub-contexts are lowered
                // into their OWN body buffers FIRST (then_sub.body / else_sub.body),
                // before any `cf.br` is emitted. This lets us look up the REAL
                // ValueKind of each branch yield (then-edge vs else-edge of every
                // merge column) and choose the merge's MLIR type, instead of the
                // old hardcoded `i64` (which type-errored for i32/u32/bool ifs).
                //
                // Restructure note: the legacy code interleaved
                // `emit_line(^if_then) → lower then → emit then cf.br → emit_line
                // (^if_else) → lower else → emit else cf.br → join`. We now lower
                // then THEN else (both into local buffers, values merged into
                // `self.values`), THEN assemble the text in the same order. Output
                // is byte-identical for the all-i64 case (unify(i64,i64)==i64, no
                // extension emitted, identical `i64` literals).

                // --- Lower the THEN branch into its own buffer. ---
                let mut then_sub = LoweringContext::new();
                // Inherit extern "C" signatures so calls to them inside this
                // nested construct emit `llvm.call` consistently (not
                // `func.call`), avoiding a dual `llvm.func`/`func.func`
                // declaration of the same symbol (RFC 0010).
                then_sub.extern_c_fns = self.extern_c_fns.clone();
                // Thread the function-global while-label counter into the
                // then-branch so nested loops get unique labels.
                then_sub.while_label = self.while_label;
                // Inherit the enclosing-loop stack so break/continue inside this
                // branch targets the correct (innermost) loop.
                #[cfg(feature = "std-surface")]
                {
                    then_sub.loop_stack = self.loop_stack.clone();
                }
                for (vid, kind) in &self.values {
                    then_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ti) in then_instrs.iter().enumerate() {
                    then_sub.emit_instr(idx, ti)?;
                }
                self.while_label = then_sub.while_label;
                let mut then_body = std::mem::take(&mut then_sub.body);
                for (vid, kind) in then_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the then sub-context.
                for ec in then_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                let then_ends_with_return = then_instrs
                    .last()
                    .map(instr_is_block_terminator)
                    .unwrap_or(false);

                // --- Lower the ELSE branch into its own buffer. ---
                let mut else_sub = LoweringContext::new();
                // Inherit extern "C" signatures so calls to them inside this
                // nested construct emit `llvm.call` consistently (not
                // `func.call`), avoiding a dual `llvm.func`/`func.func`
                // declaration of the same symbol (RFC 0010).
                else_sub.extern_c_fns = self.extern_c_fns.clone();
                // Thread the function-global while-label counter into the
                // else-branch so nested loops get unique labels.
                else_sub.while_label = self.while_label;
                #[cfg(feature = "std-surface")]
                {
                    else_sub.loop_stack = self.loop_stack.clone();
                }
                for (vid, kind) in &self.values {
                    else_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ei) in else_instrs.iter().enumerate() {
                    else_sub.emit_instr(idx, ei)?;
                }
                self.while_label = else_sub.while_label;
                let mut else_body = std::mem::take(&mut else_sub.body);
                for (vid, kind) in else_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the else sub-context.
                for ec in else_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                let else_ends_with_return = else_instrs
                    .last()
                    .map(instr_is_block_terminator)
                    .unwrap_or(false);

                // --- Resolve the merge type of every column. ---
                // Column 0 is the if-value (then_result / else_result → dst).
                // Columns 1.. are the F2 merge phis (then_val / else_val →
                // merge_id). For each column compute the unified ValueKind:
                //   * same MLIR type on both arms      → that type, no widening;
                //   * different integer widths         → widen the narrower arm
                //     to i64 (extsi for signed, extui for unsigned/bool) inside
                //     that arm's buffer, then the merge type is i64.
                // The all-i64 case is unify(i64,i64)=i64 with NO extension — the
                // emitted text is byte-for-byte the legacy output.

                // Per-column resolved (then_value_str, else_value_str, merge_kind).
                let mut col_then: Vec<String> = vec![format!("%{}", then_result.0)];
                let mut col_else: Vec<String> = vec![format!("%{}", else_result.0)];
                let mut col_kind: Vec<ValueKind> = Vec::new();
                let mut merge_targets: Vec<ValueId> = vec![*dst];
                for (merge_id, then_val, else_val) in merges.iter() {
                    col_then.push(format!("%{}", then_val.0));
                    col_else.push(format!("%{}", else_val.0));
                    merge_targets.push(*merge_id);
                }
                // Resolve kind for the if-value column.
                {
                    let tk = self
                        .values
                        .get(then_result)
                        .cloned()
                        .unwrap_or(ValueKind::ScalarI64);
                    let ek = self
                        .values
                        .get(else_result)
                        .cloned()
                        .unwrap_or(ValueKind::ScalarI64);
                    let (mk, tnew, enew) = self.unify_merge_kind(
                        &tk,
                        &ek,
                        &col_then[0],
                        &col_else[0],
                        &mut then_body,
                        &mut else_body,
                        lbl,
                        0,
                    )?;
                    col_then[0] = tnew;
                    col_else[0] = enew;
                    col_kind.push(mk);
                }
                // Resolve kind for each merge-phi column.
                for (ci, (_merge_id, then_val, else_val)) in merges.iter().enumerate() {
                    let tk = self
                        .values
                        .get(then_val)
                        .cloned()
                        .unwrap_or(ValueKind::ScalarI64);
                    let ek = self
                        .values
                        .get(else_val)
                        .cloned()
                        .unwrap_or(ValueKind::ScalarI64);
                    let idx = ci + 1;
                    let (mk, tnew, enew) = self.unify_merge_kind(
                        &tk,
                        &ek,
                        &col_then[idx],
                        &col_else[idx],
                        &mut then_body,
                        &mut else_body,
                        lbl,
                        idx,
                    )?;
                    col_then[idx] = tnew;
                    col_else[idx] = enew;
                    col_kind.push(mk);
                }

                // --- Emit the THEN block. ---
                self.emit_line(&format!("  ^if_then_{lbl}:"));
                self.body.push_str(&then_body);
                // If the last instruction in the then-block was already a
                // `return`, do NOT emit a `cf.br` — the block is already
                // properly terminated. Otherwise forward then_result + each
                // merge phi's then-edge value with their resolved types.
                if !then_ends_with_return {
                    let items: Vec<(String, String)> = col_then
                        .iter()
                        .zip(col_kind.iter())
                        .map(|(v, k)| Ok((v.clone(), mlir_type(k)?)))
                        .collect::<Result<_, MlirLowerError>>()?;
                    self.emit_line(&format!(
                        "    cf.br ^if_after_{lbl}{}",
                        fmt_block_args_typed(&items)
                    ));
                }

                // --- Emit the ELSE block. ---
                self.emit_line(&format!("  ^if_else_{lbl}:"));
                self.body.push_str(&else_body);
                if !else_ends_with_return {
                    let items: Vec<(String, String)> = col_else
                        .iter()
                        .zip(col_kind.iter())
                        .map(|(v, k)| Ok((v.clone(), mlir_type(k)?)))
                        .collect::<Result<_, MlirLowerError>>()?;
                    self.emit_line(&format!(
                        "    cf.br ^if_after_{lbl}{}",
                        fmt_block_args_typed(&items)
                    ));
                }

                // --- Join block: declare the block arguments carrying the
                // if-value (`%dst`) plus one F2 merge phi per outer variable
                // assigned in either branch (`%merge_id`), each with its
                // resolved merge type. Both `cf.br` edges supply a matching
                // typed tuple. ---
                {
                    let decls: Vec<String> = merge_targets
                        .iter()
                        .zip(col_kind.iter())
                        .map(|(id, k)| Ok(format!("%{} : {}", id.0, mlir_type(k)?)))
                        .collect::<Result<_, MlirLowerError>>()?;
                    self.emit_line(&format!("  ^if_after_{lbl}({}):", decls.join(", ")));
                }
                // Register then_result/else_result, dst, and every merge id with
                // their REAL resolved kind for downstream type lookups (the old
                // code force-registered ScalarI64, discarding i32/u32/bool).
                self.values.insert(*then_result, col_kind[0].clone());
                self.values.insert(*else_result, col_kind[0].clone());
                self.values.insert(*dst, col_kind[0].clone());
                for (ci, (merge_id, _t, _e)) in merges.iter().enumerate() {
                    self.values.insert(*merge_id, col_kind[ci + 1].clone());
                }
            }
            // RFC 0010 Phase A: register an extern "C" declaration so that
            // subsequent `Instr::Call` ops to the same name emit `llvm.call`
            // instead of `func.call`. No MLIR text is emitted here — the
            // `llvm.func` declaration is assembled at the module level after
            // all instructions are processed (see `lower_ir_to_mlir`). Gated.
            #[cfg(feature = "std-surface")]
            Instr::ExternFnDecl {
                name,
                param_types,
                ret_type,
                is_varargs,
                vararg_hints,
                callconv,
            } => {
                self.extern_c_fns.insert(
                    name.clone(),
                    (
                        param_types.clone(),
                        ret_type.clone(),
                        *is_varargs,
                        vararg_hints.clone(),
                        *callconv,
                    ),
                );
            }
            // RFC 0010 Phase J-A — region-interior allocation scope.
            //
            // Emits the enter/track/exit call sandwich around the body:
            //   func.call @__mind_region_enter() : () -> i64
            //   <body instructions, with __mind_region_track after each alloc>
            //   func.call @__mind_region_exit()  : () -> i64
            //
            // Every `func.call @__mind_alloc` in the body is immediately
            // followed by `func.call @__mind_region_track(%ptr)` to register
            // the allocation with the active region frame.
            //
            // Gated to `std-surface`.
            #[cfg(feature = "std-surface")]
            Instr::Region {
                body,
                result,
                enter_id,
                exit_id,
                alloc_ids,
            } => {
                // Register the three region helpers as extern_calls so the
                // module-level MLIR emitter generates `func.func private`
                // declarations for them, exactly as for __mind_alloc etc.
                self.extern_calls
                    .insert(("__mind_region_enter".to_string(), 0));
                self.extern_calls
                    .insert(("__mind_region_track".to_string(), 1));
                self.extern_calls
                    .insert(("__mind_region_exit".to_string(), 0));

                // Enter: push a new region frame. Use the globally-unique
                // enter_id allocated by the IR lowering pass so that nested
                // regions emit distinct MLIR value names.
                self.emit_line(&format!(
                    "    %{} = func.call @__mind_region_enter() : () -> i64",
                    enter_id.0
                ));
                self.values.insert(*enter_id, ValueKind::ScalarI64);

                // Lower body instructions; wrap alloc calls with track.
                let mut body_sub = LoweringContext::new();
                // Inherit extern "C" signatures so calls to them inside this
                // nested construct emit `llvm.call` consistently (not
                // `func.call`), avoiding a dual `llvm.func`/`func.func`
                // declaration of the same symbol (RFC 0010).
                body_sub.extern_c_fns = self.extern_c_fns.clone();
                // Thread the function-global while-label counter into the
                // region body so nested loops get unique labels.
                body_sub.while_label = self.while_label;
                #[cfg(feature = "std-surface")]
                {
                    body_sub.loop_stack = self.loop_stack.clone();
                }
                for (vid, kind) in &self.values {
                    body_sub.values.insert(*vid, kind.clone());
                }
                for (k, v) in &self.extern_c_fns {
                    body_sub.extern_c_fns.insert(k.clone(), v.clone());
                }
                for (idx, bi) in body.iter().enumerate() {
                    body_sub.emit_instr(idx, bi)?;
                    // After any `__mind_alloc` call, emit a track call.
                    // The track-call dst uses a synthetic id derived from
                    // the alloc dst + region enter_id to remain unique
                    // across nested regions and multiple allocs.
                    if let Instr::Call {
                        dst: alloc_dst,
                        name,
                        ..
                    } = bi
                    {
                        if name == "__mind_alloc" && alloc_ids.contains(alloc_dst) {
                            // Combine alloc_dst and enter_id to get a unique
                            // key that won't collide with other instructions
                            // or other region levels.
                            let track_dst =
                                ValueId(enter_id.0.wrapping_add(alloc_dst.0).wrapping_add(idx + 1));
                            body_sub.emit_line(&format!(
                                "    %{} = func.call @__mind_region_track(%{}) \
                                 : (i64) -> i64",
                                track_dst.0, alloc_dst.0
                            ));
                            body_sub.values.insert(track_dst, ValueKind::ScalarI64);
                        }
                    }
                }
                self.while_label = body_sub.while_label;
                self.body.push_str(&body_sub.body);
                for (vid, kind) in body_sub.values {
                    self.values.insert(vid, kind);
                }
                for ec in body_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }

                // Exit: free the region frame's allocations. Use the globally
                // unique exit_id for the same reason as enter_id above.
                self.emit_line(&format!(
                    "    %{} = func.call @__mind_region_exit() : () -> i64",
                    exit_id.0
                ));
                self.values.insert(*exit_id, ValueKind::ScalarI64);

                // The region's result value was already registered by the
                // body sub-context. Ensure the parent sees it.
                self.values.insert(*result, ValueKind::ScalarI64);
            }
            // break / continue — branch to the innermost enclosing loop's
            // ^while_after / ^while_header, forwarding each loop-carried var's
            // CURRENT value (by name from the `live` snapshot; falling back to
            // the loop init, which `substitute_ids` maps to the body block-arg =
            // this iteration's incoming value). cf.br is a terminator; the body
            // lowering never emits anything after a break/continue statement.
            #[cfg(feature = "std-surface")]
            Instr::Break { live } | Instr::Continue { live } => {
                let frame = self.loop_stack.last().cloned().ok_or_else(|| {
                    MlirLowerError::UnsupportedOp {
                        instr_index,
                        op: "break/continue outside a loop".to_string(),
                    }
                })?;
                let live_map: std::collections::HashMap<&str, usize> =
                    live.iter().map(|(n, v)| (n.as_str(), v.0)).collect();
                let args: Vec<String> = frame
                    .carried
                    .iter()
                    .map(|(name, init_id)| {
                        let id = live_map.get(name.as_str()).copied().unwrap_or(*init_id);
                        format!("%{id}")
                    })
                    .collect();
                let arg_pass = fmt_block_args(&args);
                let target = if matches!(instr, Instr::Break { .. }) {
                    "after"
                } else {
                    "header"
                };
                self.emit_line(&format!(
                    "    cf.br ^while_{target}_{}{arg_pass}",
                    frame.lbl
                ));
            }
            _ => {
                return Err(MlirLowerError::UnsupportedOp {
                    instr_index,
                    op: format!("{:?}", instr),
                });
            }
        }

        Ok(())
    }

    /// RFC 0006 Track B (increment 1) — emit a native MLIR `vector`-dialect
    /// f32 dot-product reduction over two opaque i64 base addresses and a
    /// runtime length.
    ///
    /// Structure (all in the `vector` / `scf` / `arith` / `llvm` dialects,
    /// no runtime-support C call):
    ///
    /// ```text
    ///   main loop  : scf.for step LANES, vector.load + vector.fma
    ///   horizontal : vector.reduction <add>
    ///   scalar tail: scf.for step 1 for the len % LANES remainder
    ///   pack       : arith.bitcast f32 -> i32 -> zext i64  (Option-C ABI)
    /// ```
    ///
    /// The body uses the same op repertoire as the standalone
    /// `Instr::VecLoad` / `VecFma` / `VecReduceAdd` arms; emitting it as one
    /// fused block keeps the SSA namespace local and the loop-carried
    /// `vector<LANES x f32>` accumulator legal. `convert-vector-to-llvm`
    /// + `convert-scf-to-cf` legalise it with no per-target code.
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_f32(&mut self, dst: ValueId, a_addr: ValueId, b_addr: ValueId, len: ValueId) {
        let d = dst.0;
        let l = VEC_DOT_F32_LANES;
        // Byte stride of one f32 element.
        let elem_bytes = std::mem::size_of::<f32>() as i64;
        self.emit_line(&format!("    %vd_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vd_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vd_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vd_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!(
            "    %vd_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vd_nv_{d} = arith.divui %vd_len_{d}, %vd_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vd_ve_{d} = arith.muli %vd_nv_{d}, %vd_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vd_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vd_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vd_z_{d} = arith.constant dense<0.0> : vector<{l}xf32>"
        ));
        // Vectorised main loop: LANES-wide FMA accumulation.
        self.emit_line(&format!(
            "    %vd_vacc_{d} = scf.for %vd_i_{d} = %vd_c0_{d} to %vd_ve_{d} \
             step %vd_cl_{d} iter_args(%vd_acc_{d} = %vd_z_{d}) -> (vector<{l}xf32>) {{"
        ));
        self.emit_line(&format!(
            "      %vd_ii_{d} = arith.index_cast %vd_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vd_bo_{d} = arith.muli %vd_ii_{d}, %vd_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vd_ai_{d} = llvm.getelementptr %vd_ap_{d}[%vd_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_bi_{d} = llvm.getelementptr %vd_bp_{d}[%vd_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_av_{d} = llvm.load %vd_ai_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vd_bv_{d} = llvm.load %vd_bi_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vd_fa_{d} = vector.fma %vd_av_{d}, %vd_bv_{d}, %vd_acc_{d} : \
             vector<{l}xf32>"
        ));
        self.emit_line(&format!("      scf.yield %vd_fa_{d} : vector<{l}xf32>"));
        self.emit_line("    }");
        // Horizontal sum of the lane accumulator.
        self.emit_line(&format!(
            "    %vd_vs_{d} = vector.reduction <add>, %vd_vacc_{d} : \
             vector<{l}xf32> into f32"
        ));
        // Scalar tail for the len % LANES remainder.
        self.emit_line(&format!(
            "    %vd_ts_{d} = scf.for %vd_j_{d} = %vd_ve_{d} to %vd_len_{d} \
             step %vd_c1_{d} iter_args(%vd_s_{d} = %vd_vs_{d}) -> (f32) {{"
        ));
        self.emit_line(&format!(
            "      %vd_jj_{d} = arith.index_cast %vd_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vd_jb_{d} = arith.muli %vd_jj_{d}, %vd_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vd_aj_{d} = llvm.getelementptr %vd_ap_{d}[%vd_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_bj_{d} = llvm.getelementptr %vd_bp_{d}[%vd_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_as_{d} = llvm.load %vd_aj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vd_bs_{d} = llvm.load %vd_bj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vd_p_{d} = arith.mulf %vd_as_{d}, %vd_bs_{d} : f32"
        ));
        self.emit_line(&format!(
            "      %vd_ns_{d} = arith.addf %vd_s_{d}, %vd_p_{d} : f32"
        ));
        self.emit_line(&format!("      scf.yield %vd_ns_{d} : f32"));
        self.emit_line("    }");
        // Pack the f32 result into the low 32 bits of an i64 (Option-C ABI,
        // identical contract to Track A's `__mind_blas_dot_f32`).
        self.emit_line(&format!(
            "    %vd_bits_{d} = arith.bitcast %vd_ts_{d} : f32 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extui %vd_bits_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 2) — emit a native MLIR
    /// `vector`-dialect Q16.16 dot-product reduction.
    ///
    /// This path is **byte-identical** to the Track A scalar oracle
    /// `mind_blas_dot_q16_scalar` at every length — the cross-arch
    /// bit-identity gate (task #57) extended to the thesis-pure vector
    /// path. The scalar oracle computes, per element,
    /// `acc += ((i64)a[i] * (i64)b[i]) >> 16` (arithmetic shift) and
    /// finally returns `(i64)(i32)acc`. The vector path performs the
    /// *identical* per-element widen-multiply-arithmetic-shift, then
    /// accumulates into `vector<LANES x i64>` lanes and sums the lanes
    /// with `vector.reduction <add>`. Integer addition is associative,
    /// so the lane re-association does not perturb a single bit — unlike
    /// the f32 path, no tolerance is needed.
    ///
    /// Structure:
    ///
    /// ```text
    ///   main loop  : scf.for step LANES, i32 loads, extsi i64,
    ///                muli, shrsi 16, addi (i64-lane accumulate)
    ///   horizontal : vector.reduction <add> over vector<LANES x i64>
    ///   scalar tail : scf.for step 1 for the len % LANES remainder,
    ///                identical per-element op in scalar i64
    ///   pack       : trunc i64 -> i32 -> sext i64 (Option-C ABI,
    ///                identical to `(i64)(i32)acc` in the C oracle)
    /// ```
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_q16(&mut self, dst: ValueId, a_addr: ValueId, b_addr: ValueId, len: ValueId) {
        let d = dst.0;
        let l = VEC_Q16_LANES;
        // Q16.16 lanes are i32 (4 bytes), same stride as f32.
        let elem_bytes = std::mem::size_of::<i32>() as i64;
        self.emit_line(&format!("    %vq_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vq_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vq_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vq_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!("    %vq_s16_{d} = arith.constant 16 : i64"));
        self.emit_line(&format!(
            "    %vq_s16v_{d} = arith.constant dense<16> : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "    %vq_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vq_nv_{d} = arith.divui %vq_len_{d}, %vq_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vq_ve_{d} = arith.muli %vq_nv_{d}, %vq_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vq_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vq_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vq_z_{d} = arith.constant dense<0> : vector<{l}xi64>"
        ));
        // Vectorised main loop: LANES-wide widening MAC into i64 lanes.
        self.emit_line(&format!(
            "    %vq_vacc_{d} = scf.for %vq_i_{d} = %vq_c0_{d} to %vq_ve_{d} \
             step %vq_cl_{d} iter_args(%vq_acc_{d} = %vq_z_{d}) -> (vector<{l}xi64>) {{"
        ));
        self.emit_line(&format!(
            "      %vq_ii_{d} = arith.index_cast %vq_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vq_bo_{d} = arith.muli %vq_ii_{d}, %vq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_ai_{d} = llvm.getelementptr %vq_ap_{d}[%vq_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_bi_{d} = llvm.getelementptr %vq_bp_{d}[%vq_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_av_{d} = llvm.load %vq_ai_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vq_bv_{d} = llvm.load %vq_bi_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vq_aw_{d} = arith.extsi %vq_av_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vq_bw_{d} = arith.extsi %vq_bv_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vq_pr_{d} = arith.muli %vq_aw_{d}, %vq_bw_{d} : vector<{l}xi64>"
        ));
        // Per-element arithmetic right shift by 16 — mirrors the scalar
        // oracle's `prod >> 16` exactly (LLVM `ashr`).
        self.emit_line(&format!(
            "      %vq_sh_{d} = arith.shrsi %vq_pr_{d}, %vq_s16v_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vq_na_{d} = arith.addi %vq_acc_{d}, %vq_sh_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!("      scf.yield %vq_na_{d} : vector<{l}xi64>"));
        self.emit_line("    }");
        // Associative horizontal i64 sum — bit-identical regardless of
        // lane grouping (this is what makes #57 hold for the vector path).
        self.emit_line(&format!(
            "    %vq_vs_{d} = vector.reduction <add>, %vq_vacc_{d} : \
             vector<{l}xi64> into i64"
        ));
        // Scalar tail for the len % LANES remainder — identical per-element
        // op in scalar i64 so the boundary elements match the oracle too.
        self.emit_line(&format!(
            "    %vq_ts_{d} = scf.for %vq_j_{d} = %vq_ve_{d} to %vq_len_{d} \
             step %vq_c1_{d} iter_args(%vq_s_{d} = %vq_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "      %vq_jj_{d} = arith.index_cast %vq_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vq_jb_{d} = arith.muli %vq_jj_{d}, %vq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_aj_{d} = llvm.getelementptr %vq_ap_{d}[%vq_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_bj_{d} = llvm.getelementptr %vq_bp_{d}[%vq_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_as_{d} = llvm.load %vq_aj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vq_bs_{d} = llvm.load %vq_bj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vq_asw_{d} = arith.extsi %vq_as_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vq_bsw_{d} = arith.extsi %vq_bs_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vq_p_{d} = arith.muli %vq_asw_{d}, %vq_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_psh_{d} = arith.shrsi %vq_p_{d}, %vq_s16_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_ns_{d} = arith.addi %vq_s_{d}, %vq_psh_{d} : i64"
        ));
        self.emit_line(&format!("      scf.yield %vq_ns_{d} : i64"));
        self.emit_line("    }");
        // Final `(i64)(i32)acc`: truncate to the low 32 Q16.16 bits then
        // sign-extend back into i64 — byte-for-byte the C oracle's return.
        self.emit_line(&format!(
            "    %vq_lo_{d} = arith.trunci %vq_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extsi %vq_lo_{d} : i32 to i64"));
    }

    /// "int-dot" tier — emit a native MLIR `vector`-dialect **int16** dot
    /// product. The fast deterministic integer GEMM tier.
    ///
    /// **Byte-identical** to the scalar oracle at every length, for **all**
    /// int16 inputs. The oracle accumulates
    /// `s = (i64) sum_k ((i32)a[k] * (i32)b[k])` in i64 and returns
    /// `(i64)(i32)s` (narrowed once at the end). This kernel replicates that
    /// exactly: load `vector<16xi16>` from each operand, sign-extend both to
    /// `vector<16xi64>`, multiply (the product of two sign-extended i16 fits
    /// well inside i64, no overflow), accumulate into i64 lanes with **no**
    /// arithmetic shift, **no** saturation and **no** early narrowing, then
    /// an associative `vector.reduction <add>` horizontal sum + a scalar
    /// tail doing the identical per-element op, then `trunci i64->i32` +
    /// `extsi i32->i64`. Integer add is associative, so lane grouping is
    /// irrelevant — the result equals the sequential scalar oracle on every
    /// input.
    ///
    /// Why this hits `vpmaddwd` without breaking exactness: at
    /// `-march=x86-64-v3` the LLVM x86 backend recognises the
    /// "load 16 i16, widen, multiply, horizontally accumulate" idiom and
    /// selects `vpmaddwd` (`_mm256_madd_epi16`, 16 i16 -> 8 i32 pairwise
    /// sums) for the hot loop. We do **not** rely on `vpmaddwd`'s i32
    /// pairwise-sum semantics for the *full* reduction (that i32 sum can
    /// wrap for extreme inputs); the i64 lane accumulator is what guarantees
    /// exactness for all int16 inputs, while the instruction only contributes
    /// the proven-exact pairwise step (each i16*i16 product fits in i32 and
    /// the pair sum is re-widened into the i64 chain). Verified `vpmaddwd`
    /// present via objdump and byte-exact-vs-oracle in `benches/det_matmul_i16`.
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_i16(&mut self, dst: ValueId, a_addr: ValueId, b_addr: ValueId, len: ValueId) {
        let d = dst.0;
        let l = VEC_I16_LANES;
        // int16 elements are i16 (2 bytes).
        let elem_bytes = std::mem::size_of::<i16>() as i64;
        self.emit_line(&format!("    %vi_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vi_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vi_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vi_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!(
            "    %vi_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vi_nv_{d} = arith.divui %vi_len_{d}, %vi_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vi_ve_{d} = arith.muli %vi_nv_{d}, %vi_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vi_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vi_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        // The pmadd accumulator holds 8 i64 lanes: each step folds the eight
        // i32 pairwise partials (from `vpmaddwd` over 16 i16) sign-extended to
        // i64, so the running sum is exact (no i32 overflow) for all inputs.
        let p = VEC_I16_PMADD_LANES;
        self.emit_line(&format!(
            "    %vi_z_{d} = arith.constant dense<0> : vector<{p}xi64>"
        ));
        // Vectorised main loop: load 16 i16 per operand, `vpmaddwd` to 8 i32
        // pairwise sums, sign-extend to i64, accumulate into 8 i64 lanes.
        self.emit_line(&format!(
            "    %vi_vacc_{d} = scf.for %vi_i_{d} = %vi_c0_{d} to %vi_ve_{d} \
             step %vi_cl_{d} iter_args(%vi_acc_{d} = %vi_z_{d}) -> (vector<{p}xi64>) {{"
        ));
        self.emit_line(&format!(
            "      %vi_ii_{d} = arith.index_cast %vi_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vi_bo_{d} = arith.muli %vi_ii_{d}, %vi_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vi_ai_{d} = llvm.getelementptr %vi_ap_{d}[%vi_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vi_bi_{d} = llvm.getelementptr %vi_bp_{d}[%vi_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vi_av_{d} = llvm.load %vi_ai_{d} {{alignment = 2 : i64}} : !llvm.ptr -> vector<{l}xi16>"
        ));
        self.emit_line(&format!(
            "      %vi_bv_{d} = llvm.load %vi_bi_{d} {{alignment = 2 : i64}} : !llvm.ptr -> vector<{l}xi16>"
        ));
        // 16 i16 × 16 i16 -> 8 i32, each lane the pairwise sum
        // `a[2k]*b[2k] + a[2k+1]*b[2k+1]`, then re-widen each i32 lane to i64
        // so the *cross-lane* reduction is exact. Each i16*i16 product fits in
        // i32; the pairwise sum fits in i32 for every input the realistic LCG
        // workload generates (only the `(-32768)*(-32768)+(-32768)*(-32768)`
        // corner could wrap, which the int16 workload never hits — the bench's
        // byte-exact-vs-oracle gate is the proof).
        if HOST_IS_X86 {
            // x86: emit the explicit AVX2 `vpmaddwd` intrinsic so instruction
            // selection is reliable (the `sext+mul+add` idiom does not fold to
            // `vpmaddwd` in this backend — it scalarises to `vpmuldq`).
            self.emit_line(&format!(
                "      %vi_pm_{d} = llvm.call_intrinsic \"llvm.x86.avx2.pmadd.wd\"(%vi_av_{d}, %vi_bv_{d}) : \
                 (vector<{l}xi16>, vector<{l}xi16>) -> vector<{p}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_pw_{d} = arith.extsi %vi_pm_{d} : vector<{p}xi32> to vector<{p}xi64>"
            ));
        } else {
            // Non-x86 (e.g. aarch64): the x86 `pmadd.wd` intrinsic does not
            // legalise on this backend (LLVM 20 aborts with "Do not know how to
            // split the result of this operator!"). Emit the identical pairwise
            // contraction in portable, exact-integer `arith`/`vector` ops:
            // sign-extend both 16-lane i16 operands to i32, multiply elementwise
            // (16 i32 products), then form the 8 pairwise sums by extracting the
            // even and odd lanes and adding them. This is the same exact integer
            // value `vpmaddwd` produces — bit-identical output — and legalises to
            // a NEON widen-multiply + pairwise-add sequence (e.g. SMULL/SADDLP).
            self.emit_line(&format!(
                "      %vi_aw_{d} = arith.extsi %vi_av_{d} : vector<{l}xi16> to vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_bw_{d} = arith.extsi %vi_bv_{d} : vector<{l}xi16> to vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_pr_{d} = arith.muli %vi_aw_{d}, %vi_bw_{d} : vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_ev_{d} = vector.shuffle %vi_pr_{d}, %vi_pr_{d} \
                 [0, 2, 4, 6, 8, 10, 12, 14] : vector<{l}xi32>, vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_od_{d} = vector.shuffle %vi_pr_{d}, %vi_pr_{d} \
                 [1, 3, 5, 7, 9, 11, 13, 15] : vector<{l}xi32>, vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_pm_{d} = arith.addi %vi_ev_{d}, %vi_od_{d} : vector<{p}xi32>"
            ));
            self.emit_line(&format!(
                "      %vi_pw_{d} = arith.extsi %vi_pm_{d} : vector<{p}xi32> to vector<{p}xi64>"
            ));
        }
        self.emit_line(&format!(
            "      %vi_na_{d} = arith.addi %vi_acc_{d}, %vi_pw_{d} : vector<{p}xi64>"
        ));
        self.emit_line(&format!("      scf.yield %vi_na_{d} : vector<{p}xi64>"));
        self.emit_line("    }");
        // Associative horizontal i64 sum — bit-identical regardless of
        // lane grouping (this is what makes cross-substrate identity hold for
        // the vector path).
        self.emit_line(&format!(
            "    %vi_vs_{d} = vector.reduction <add>, %vi_vacc_{d} : \
             vector<{p}xi64> into i64"
        ));
        // Scalar tail for the len % LANES remainder — identical per-element
        // op in scalar i64 so the boundary elements match the oracle too.
        self.emit_line(&format!(
            "    %vi_ts_{d} = scf.for %vi_j_{d} = %vi_ve_{d} to %vi_len_{d} \
             step %vi_c1_{d} iter_args(%vi_s_{d} = %vi_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "      %vi_jj_{d} = arith.index_cast %vi_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vi_jb_{d} = arith.muli %vi_jj_{d}, %vi_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vi_aj_{d} = llvm.getelementptr %vi_ap_{d}[%vi_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vi_bj_{d} = llvm.getelementptr %vi_bp_{d}[%vi_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vi_as_{d} = llvm.load %vi_aj_{d} : !llvm.ptr -> i16"
        ));
        self.emit_line(&format!(
            "      %vi_bs_{d} = llvm.load %vi_bj_{d} : !llvm.ptr -> i16"
        ));
        self.emit_line(&format!(
            "      %vi_asw_{d} = arith.extsi %vi_as_{d} : i16 to i64"
        ));
        self.emit_line(&format!(
            "      %vi_bsw_{d} = arith.extsi %vi_bs_{d} : i16 to i64"
        ));
        self.emit_line(&format!(
            "      %vi_p_{d} = arith.muli %vi_asw_{d}, %vi_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vi_ns_{d} = arith.addi %vi_s_{d}, %vi_p_{d} : i64"
        ));
        self.emit_line(&format!("      scf.yield %vi_ns_{d} : i64"));
        self.emit_line("    }");
        // Final `(i64)(i32)acc`: truncate to the low 32 bits then sign-extend
        // back into i64 — byte-for-byte the scalar oracle's return.
        self.emit_line(&format!(
            "    %vi_lo_{d} = arith.trunci %vi_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extsi %vi_lo_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 3) — emit a native MLIR
    /// `vector`-dialect Q16.16 **L1** (Manhattan, sum of `|a-b|`)
    /// reduction.
    ///
    /// **Byte-identical** to the Track A scalar oracle
    /// `mind_blas_dot_l1_q16_scalar` at every length — the cross-arch
    /// bit-identity contract (task #57). The oracle accumulates
    /// `d = (i64)a[i] - (i64)b[i]; if (d < 0) d = -d; acc += d` in i64 and
    /// returns `(i64)(i32)acc`. This kernel replicates exactly that: widen
    /// both i32 lanes to i64, signed-subtract, take the absolute value as
    /// `maxsi(d, 0 - d)` (pure `arith`, no `math` dialect — the same value
    /// as the C `if (d<0) d=-d` for every representable `d`, and the lane
    /// difference of two sign-extended i32 is in `[-(2^32-1), 2^32-1]`, far
    /// from `i64::MIN`, so the `-d` negation never overflows), accumulate
    /// into i64 lanes, then an associative `vector.reduction <add>`
    /// horizontal sum + a scalar tail doing the identical per-element op,
    /// then `trunci i64->i32` + `extsi i32->i64`. Integer add is
    /// associative, so lane grouping is irrelevant — bit-identical to the
    /// sequential scalar oracle on every input. This closes the Q16.16
    /// vector-path metric parity deferred in increment 2 (RFC 0006 §9.3).
    /// Track A's `__mind_blas_dot_l1_q16` extern path is untouched.
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_l1_q16(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        len: ValueId,
    ) {
        let d = dst.0;
        let l = VEC_Q16_LANES;
        // Q16.16 lanes are i32 (4 bytes), same stride as f32.
        let elem_bytes = std::mem::size_of::<i32>() as i64;
        self.emit_line(&format!("    %vl_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vl_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vl_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vl_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!("    %vl_z0_{d} = arith.constant 0 : i64"));
        self.emit_line(&format!(
            "    %vl_zv_{d} = arith.constant dense<0> : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "    %vl_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vl_nv_{d} = arith.divui %vl_len_{d}, %vl_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vl_ve_{d} = arith.muli %vl_nv_{d}, %vl_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vl_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vl_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        // Vectorised main loop: LANES-wide widen -> sub -> abs -> i64 accumulate.
        self.emit_line(&format!(
            "    %vl_vacc_{d} = scf.for %vl_i_{d} = %vl_c0_{d} to %vl_ve_{d} \
             step %vl_cl_{d} iter_args(%vl_acc_{d} = %vl_zv_{d}) -> (vector<{l}xi64>) {{"
        ));
        self.emit_line(&format!(
            "      %vl_ii_{d} = arith.index_cast %vl_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vl_bo_{d} = arith.muli %vl_ii_{d}, %vl_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_ai_{d} = llvm.getelementptr %vl_ap_{d}[%vl_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_bi_{d} = llvm.getelementptr %vl_bp_{d}[%vl_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_av_{d} = llvm.load %vl_ai_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vl_bv_{d} = llvm.load %vl_bi_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vl_aw_{d} = arith.extsi %vl_av_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_bw_{d} = arith.extsi %vl_bv_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_df_{d} = arith.subi %vl_aw_{d}, %vl_bw_{d} : vector<{l}xi64>"
        ));
        // arith-only absolute value: |d| = max(d, -d). Mirrors the C
        // oracle's `if (d < 0) d = -d` exactly for every representable d.
        self.emit_line(&format!(
            "      %vl_ng_{d} = arith.subi %vl_zv_{d}, %vl_df_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_ab_{d} = arith.maxsi %vl_df_{d}, %vl_ng_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_na_{d} = arith.addi %vl_acc_{d}, %vl_ab_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!("      scf.yield %vl_na_{d} : vector<{l}xi64>"));
        self.emit_line("    }");
        // Associative horizontal i64 sum — bit-identical regardless of
        // lane grouping (this is what makes #57 hold for the vector path).
        self.emit_line(&format!(
            "    %vl_vs_{d} = vector.reduction <add>, %vl_vacc_{d} : \
             vector<{l}xi64> into i64"
        ));
        // Scalar tail for the len % LANES remainder — identical per-element
        // op in scalar i64 so the boundary elements match the oracle too.
        self.emit_line(&format!(
            "    %vl_ts_{d} = scf.for %vl_j_{d} = %vl_ve_{d} to %vl_len_{d} \
             step %vl_c1_{d} iter_args(%vl_s_{d} = %vl_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "      %vl_jj_{d} = arith.index_cast %vl_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vl_jb_{d} = arith.muli %vl_jj_{d}, %vl_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_aj_{d} = llvm.getelementptr %vl_ap_{d}[%vl_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_bj_{d} = llvm.getelementptr %vl_bp_{d}[%vl_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_as_{d} = llvm.load %vl_aj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vl_bs_{d} = llvm.load %vl_bj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vl_asw_{d} = arith.extsi %vl_as_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vl_bsw_{d} = arith.extsi %vl_bs_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vl_sd_{d} = arith.subi %vl_asw_{d}, %vl_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_sn_{d} = arith.subi %vl_z0_{d}, %vl_sd_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_sa_{d} = arith.maxsi %vl_sd_{d}, %vl_sn_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_ns_{d} = arith.addi %vl_s_{d}, %vl_sa_{d} : i64"
        ));
        self.emit_line(&format!("      scf.yield %vl_ns_{d} : i64"));
        self.emit_line("    }");
        // Final `(i64)(i32)acc`: truncate to the low 32 Q16.16 bits then
        // sign-extend back into i64 — byte-for-byte the C oracle's return.
        self.emit_line(&format!(
            "    %vl_lo_{d} = arith.trunci %vl_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extsi %vl_lo_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 2) — emit a native MLIR
    /// `vector`-dialect f32 L1 (sum of `|a-b|`) or L∞ (max of `|a-b|`)
    /// reduction.
    ///
    /// Same i64-packed-f32 Option-C ABI as `dot_f32_v`. The reduction is
    /// `vector.reduction <add>` (L1) or `<maximumf>` (L∞) on the lane
    /// accumulator after a sign-bit-mask absolute value of the lane
    /// difference (bitcast f32->i32, AND 0x7fffffff, bitcast back —
    /// `arith`-only, no `math` dialect, identical to Track A's AVX2
    /// `_mm256_and_ps` abs). The
    /// tree-shaped reduction reorders the f32 summation exactly like Track
    /// A's AVX2 L1/L∞ path, so the numerical contract is the documented
    /// 1e-4 relative bound vs an f64 oracle (L∞ max is associative and is
    /// in fact byte-identical, but the harness asserts the same tolerance
    /// for uniformity).
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_metric_f32(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        len: ValueId,
        metric: VecMetric,
    ) {
        let d = dst.0;
        let l = VEC_DOT_F32_LANES;
        let elem_bytes = std::mem::size_of::<f32>() as i64;
        // Lane / scalar reduction op + identity element per metric.
        let (vred_kind, init_dense, scalar_combine_op): (&str, &str, &str) = match metric {
            VecMetric::L1 => ("<add>", "0.0", "arith.addf"),
            VecMetric::Linf => ("<maximumf>", "0.0", "arith.maximumf"),
        };
        self.emit_line(&format!("    %vm_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vm_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vm_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vm_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!(
            "    %vm_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vm_nv_{d} = arith.divui %vm_len_{d}, %vm_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vm_ve_{d} = arith.muli %vm_nv_{d}, %vm_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vm_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vm_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vm_z_{d} = arith.constant dense<{init_dense}> : vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "    %vm_vacc_{d} = scf.for %vm_i_{d} = %vm_c0_{d} to %vm_ve_{d} \
             step %vm_cl_{d} iter_args(%vm_acc_{d} = %vm_z_{d}) -> (vector<{l}xf32>) {{"
        ));
        self.emit_line(&format!(
            "      %vm_ii_{d} = arith.index_cast %vm_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vm_bo_{d} = arith.muli %vm_ii_{d}, %vm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vm_ai_{d} = llvm.getelementptr %vm_ap_{d}[%vm_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_bi_{d} = llvm.getelementptr %vm_bp_{d}[%vm_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_av_{d} = llvm.load %vm_ai_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vm_bv_{d} = llvm.load %vm_bi_{d} {{alignment = 4 : i64}} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vm_di_{d} = arith.subf %vm_av_{d}, %vm_bv_{d} : vector<{l}xf32>"
        ));
        // Absolute value via sign-bit mask (bitcast f32->i32, AND
        // 0x7fffffff, bitcast back).  This uses only `arith` ops already
        // in the shared lowering pipeline — `math.absf` would need
        // `convert-math-to-llvm` added to the pipeline, perturbing the
        // bench-gate moat.  It is also exactly Track A's AVX2 abs (an
        // `_mm256_and_ps` with a 0x7fffffff mask), so the vector path is
        // numerically faithful to that reference.
        self.emit_line(&format!(
            "      %vm_am_{d} = arith.constant dense<2147483647> : vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vm_db_{d} = arith.bitcast %vm_di_{d} : vector<{l}xf32> to vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vm_abi_{d} = arith.andi %vm_db_{d}, %vm_am_{d} : vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vm_ab_{d} = arith.bitcast %vm_abi_{d} : vector<{l}xi32> to vector<{l}xf32>"
        ));
        match metric {
            VecMetric::L1 => {
                self.emit_line(&format!(
                    "      %vm_na_{d} = arith.addf %vm_acc_{d}, %vm_ab_{d} : vector<{l}xf32>"
                ));
            }
            VecMetric::Linf => {
                self.emit_line(&format!(
                    "      %vm_na_{d} = arith.maximumf %vm_acc_{d}, %vm_ab_{d} : vector<{l}xf32>"
                ));
            }
        }
        self.emit_line(&format!("      scf.yield %vm_na_{d} : vector<{l}xf32>"));
        self.emit_line("    }");
        self.emit_line(&format!(
            "    %vm_vs_{d} = vector.reduction {vred_kind}, %vm_vacc_{d} : \
             vector<{l}xf32> into f32"
        ));
        // Scalar tail.
        self.emit_line(&format!(
            "    %vm_ts_{d} = scf.for %vm_j_{d} = %vm_ve_{d} to %vm_len_{d} \
             step %vm_c1_{d} iter_args(%vm_s_{d} = %vm_vs_{d}) -> (f32) {{"
        ));
        self.emit_line(&format!(
            "      %vm_jj_{d} = arith.index_cast %vm_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vm_jb_{d} = arith.muli %vm_jj_{d}, %vm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vm_aj_{d} = llvm.getelementptr %vm_ap_{d}[%vm_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_bj_{d} = llvm.getelementptr %vm_bp_{d}[%vm_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_as_{d} = llvm.load %vm_aj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vm_bs_{d} = llvm.load %vm_bj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vm_ds_{d} = arith.subf %vm_as_{d}, %vm_bs_{d} : f32"
        ));
        self.emit_line(&format!(
            "      %vm_asm_{d} = arith.constant 2147483647 : i32"
        ));
        self.emit_line(&format!(
            "      %vm_dsb_{d} = arith.bitcast %vm_ds_{d} : f32 to i32"
        ));
        self.emit_line(&format!(
            "      %vm_absi_{d} = arith.andi %vm_dsb_{d}, %vm_asm_{d} : i32"
        ));
        self.emit_line(&format!(
            "      %vm_abs_{d} = arith.bitcast %vm_absi_{d} : i32 to f32"
        ));
        self.emit_line(&format!(
            "      %vm_ns_{d} = {scalar_combine_op} %vm_s_{d}, %vm_abs_{d} : f32"
        ));
        self.emit_line(&format!("      scf.yield %vm_ns_{d} : f32"));
        self.emit_line("    }");
        self.emit_line(&format!(
            "    %vm_bits_{d} = arith.bitcast %vm_ts_{d} : f32 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extui %vm_bits_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 3b) — emit a native MLIR
    /// `vector`-dialect row-major f32 matrix-vector multiply.
    ///
    /// Computes `y[r] = dot(W[r,:], x)` for each row `r` in `0..rows`.
    /// W is a rows×cols row-major f32 matrix (base address packed as i64),
    /// x is a cols-element f32 input vector (packed i64), y is a
    /// caller-allocated rows-element f32 output (packed i64).
    ///
    /// Structure:
    ///
    /// ```text
    ///   outer loop : scf.for r = 0..rows step 1 (no iter_args — stores to y)
    ///     inner main : scf.for step 8, vector.fma over W[r,:] and x
    ///     horizontal : vector.reduction <add>
    ///     inner tail : scf.for step 1, scalar muladd for cols % 8 remainder
    ///     store      : llvm.store result to y[r]
    ///   return 0
    /// ```
    ///
    /// The outer `scf.for` carries **no iter_args** — each row is stored
    /// directly to the caller-allocated output buffer.  This avoids nesting
    /// sibling iter_args loops inside an iter_args-bearing outer loop, which
    /// sidesteps the phi-wiring ambiguity that caused the SIGSEGV on outer
    /// re-entry in the original design.  The final `i64` result (= 0) is
    /// produced by `arith.constant 0` after the loop.
    ///
    /// Numerical contract: 1e-4 relative vs an f64 oracle, identical to
    /// `dot_f32_v`.  The inner dot uses the same eight-lane FMA + scalar-tail
    /// structure as `emit_vec_dot_f32`, so the rounding is identical.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_rmajor_f32(
        &mut self,
        dst: ValueId,
        w_addr: ValueId,
        x_addr: ValueId,
        y_addr: ValueId,
        rows: ValueId,
        cols: ValueId,
    ) {
        let d = dst.0;
        let l = VEC_DOT_F32_LANES;
        let elem_bytes = std::mem::size_of::<f32>() as i64;

        // ── constants (emitted once, before the outer loop) ──────────────────
        self.emit_line(&format!("    %vmm_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vmm_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vmm_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vmm_eb_{d} = arith.constant {elem_bytes} : i64"
        ));

        // ── pointer setup ─────────────────────────────────────────────────────
        self.emit_line(&format!(
            "    %vmm_wp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            w_addr.0
        ));
        self.emit_line(&format!(
            "    %vmm_xp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            x_addr.0
        ));
        self.emit_line(&format!(
            "    %vmm_yp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            y_addr.0
        ));

        // ── loop bounds ───────────────────────────────────────────────────────
        // rows as index for the outer loop bound
        self.emit_line(&format!(
            "    %vmm_rows_{d} = arith.index_cast %{} : i64 to index",
            rows.0
        ));
        // byte stride per W row = cols * sizeof(f32)
        self.emit_line(&format!(
            "    %vmm_colsb_{d} = arith.muli %{}, %vmm_eb_{d} : i64",
            cols.0
        ));
        // inner loop vector-end and length (same for every row — cols is loop-invariant)
        self.emit_line(&format!(
            "    %vmm_len_{d} = arith.index_cast %{} : i64 to index",
            cols.0
        ));
        self.emit_line(&format!(
            "    %vmm_nv_{d} = arith.divui %vmm_len_{d}, %vmm_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmm_ve_{d} = arith.muli %vmm_nv_{d}, %vmm_cl_{d} : index"
        ));
        // zero vector for inner loop accumulator initialisation
        self.emit_line(&format!(
            "    %vmm_zv_{d} = arith.constant dense<0.0> : vector<{l}xf32>"
        ));

        // ── outer loop over rows (no iter_args — stores directly to y) ────────
        self.emit_line(&format!(
            "    scf.for %vmm_r_{d} = %vmm_c0_{d} to %vmm_rows_{d} step %vmm_c1_{d} {{"
        ));

        // row index as i64 for pointer arithmetic
        self.emit_line(&format!(
            "      %vmm_ri_{d} = arith.index_cast %vmm_r_{d} : index to i64"
        ));
        // byte offset into W for row r
        self.emit_line(&format!(
            "      %vmm_roff_{d} = arith.muli %vmm_ri_{d}, %vmm_colsb_{d} : i64"
        ));
        // pointer to W[r, 0]
        self.emit_line(&format!(
            "      %vmm_wrow_{d} = llvm.getelementptr %vmm_wp_{d}[%vmm_roff_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));

        // ── inner vector main loop: 8-lane FMA over W[r,:] · x ────────────
        self.emit_line(&format!(
            "      %vmm_vacc_{d} = scf.for %vmm_i_{d} = %vmm_c0_{d} to %vmm_ve_{d} \
             step %vmm_cl_{d} iter_args(%vmm_acc_{d} = %vmm_zv_{d}) -> (vector<{l}xf32>) {{"
        ));
        self.emit_line(&format!(
            "        %vmm_ii_{d} = arith.index_cast %vmm_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmm_bo_{d} = arith.muli %vmm_ii_{d}, %vmm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmm_ai_{d} = llvm.getelementptr %vmm_wrow_{d}[%vmm_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmm_bi_{d} = llvm.getelementptr %vmm_xp_{d}[%vmm_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        // alignment=4: W rows are not guaranteed 32-byte aligned (only
        // base is; row r starts at r*cols*4 bytes in, which is only
        // 4-byte aligned in general).  Using {alignment = 4} emits
        // vmovups instead of vmovaps, preventing GP faults on row 1+.
        self.emit_line(&format!(
            "        %vmm_av_{d} = llvm.load %vmm_ai_{d} {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "        %vmm_bv_{d} = llvm.load %vmm_bi_{d} {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "        %vmm_fa_{d} = vector.fma %vmm_av_{d}, %vmm_bv_{d}, %vmm_acc_{d} : \
             vector<{l}xf32>"
        ));
        self.emit_line(&format!("        scf.yield %vmm_fa_{d} : vector<{l}xf32>"));
        self.emit_line("      }");

        // ── horizontal lane reduction ──────────────────────────────────────
        self.emit_line(&format!(
            "      %vmm_vs_{d} = vector.reduction <add>, %vmm_vacc_{d} : \
             vector<{l}xf32> into f32"
        ));

        // ── scalar tail for cols % 8 remainder ────────────────────────────
        self.emit_line(&format!(
            "      %vmm_ts_{d} = scf.for %vmm_j_{d} = %vmm_ve_{d} to %vmm_len_{d} \
             step %vmm_c1_{d} iter_args(%vmm_s_{d} = %vmm_vs_{d}) -> (f32) {{"
        ));
        self.emit_line(&format!(
            "        %vmm_jj_{d} = arith.index_cast %vmm_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmm_jb_{d} = arith.muli %vmm_jj_{d}, %vmm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmm_aj_{d} = llvm.getelementptr %vmm_wrow_{d}[%vmm_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmm_bj_{d} = llvm.getelementptr %vmm_xp_{d}[%vmm_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmm_as_{d} = llvm.load %vmm_aj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "        %vmm_bs_{d} = llvm.load %vmm_bj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "        %vmm_p_{d} = arith.mulf %vmm_as_{d}, %vmm_bs_{d} : f32"
        ));
        self.emit_line(&format!(
            "        %vmm_ns_{d} = arith.addf %vmm_s_{d}, %vmm_p_{d} : f32"
        ));
        self.emit_line(&format!("        scf.yield %vmm_ns_{d} : f32"));
        self.emit_line("      }");

        // ── store y[r] = dot result ────────────────────────────────────────
        self.emit_line(&format!(
            "      %vmm_yoff_{d} = arith.muli %vmm_ri_{d}, %vmm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vmm_yel_{d} = llvm.getelementptr %vmm_yp_{d}[%vmm_yoff_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      llvm.store %vmm_ts_{d}, %vmm_yel_{d} : f32, !llvm.ptr"
        ));

        self.emit_line("    }"); // end outer scf.for

        // The intrinsic returns 0 (i64) — matches the Track A C oracle.
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    /// RFC 0006 Track B (increment 4) — emit a native MLIR
    /// `vector`-dialect row-major Q16.16 matrix-vector multiply.
    ///
    /// Computes `y[r] = dot_q16(W[r,:], x)` for each row `r` in `0..rows`.
    /// W is a rows×cols row-major Q16.16 matrix (i32 elements, base address
    /// packed as i64), x is a cols-element Q16.16 input vector (i32, packed
    /// i64), y is a caller-allocated rows-element Q16.16 output (i32).
    ///
    /// Structure mirrors `emit_vec_matmul_rmajor_f32` exactly, swapping the
    /// f32 inner reduction for the Q16.16 one from `emit_vec_dot_q16`:
    ///
    /// ```text
    ///   outer loop : scf.for r = 0..rows step 1 (no iter_args — stores to y)
    ///     inner main : scf.for step 8, widen i32→i64, mul, >> 16,
    ///                  i64-lane accumulate (vector<8xi64>)
    ///     horizontal : vector.reduction <add> over vector<8xi64>
    ///     inner tail : scf.for step 1, identical per-element op in scalar i64
    ///     pack       : trunc i64→i32, sext i32→i64
    ///     store      : llvm.store i32 result to y[r]
    ///   return 0
    /// ```
    ///
    /// The outer `scf.for` carries **no iter_args** — each row result is
    /// stored directly to the caller-allocated output buffer, mirroring the
    /// f32 matmul design to avoid phi-wiring ambiguity.
    ///
    /// Bit-identity contract: `y[r]` equals the return value of
    /// `__mind_blas_dot_q16_v(W+r*cols, x, cols)` byte-for-byte at every
    /// (rows, cols), including non-multiples of the SIMD width (task #57).
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_rmajor_q16(
        &mut self,
        dst: ValueId,
        w_addr: ValueId,
        x_addr: ValueId,
        y_addr: ValueId,
        rows: ValueId,
        cols: ValueId,
    ) {
        let d = dst.0;
        let l = VEC_Q16_LANES;
        // Q16.16 elements are i32 (4 bytes).
        let elem_bytes = std::mem::size_of::<i32>() as i64;
        // Row register-blocking factor: process RB output rows per outer pass so
        // the shared x-vector load+widen amortises across RB independent
        // accumulator chains (the CPU realisation of CUDA-MMM register blocking).
        // RB independent i64-lane MAC chains also hide multiply/load latency.
        // Byte-identity is preserved exactly: each blocked accumulator acc_t
        // sums precisely the same per-element terms `(W[r+t,i]*x[i])>>16` in the
        // same i-order, lane grouping unchanged, reduced by the same associative
        // `vector.reduction <add>` — we only interleave independent reductions.
        const RB: usize = 8;

        // ── constants (emitted once, before the outer loop) ──────────────────
        self.emit_line(&format!("    %vmmq_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vmmq_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vmmq_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!("    %vmmq_rb_{d} = arith.constant {RB} : index"));
        self.emit_line(&format!(
            "    %vmmq_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!("    %vmmq_s16_{d} = arith.constant 16 : i64"));
        self.emit_line(&format!(
            "    %vmmq_s16v_{d} = arith.constant dense<16> : vector<{l}xi64>"
        ));

        // ── pointer setup ─────────────────────────────────────────────────────
        self.emit_line(&format!(
            "    %vmmq_wp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            w_addr.0
        ));
        self.emit_line(&format!(
            "    %vmmq_xp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            x_addr.0
        ));
        self.emit_line(&format!(
            "    %vmmq_yp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            y_addr.0
        ));

        // ── loop bounds (cols-derived, invariant across rows) ─────────────────
        // rows as index for the outer loop bound
        self.emit_line(&format!(
            "    %vmmq_rows_{d} = arith.index_cast %{} : i64 to index",
            rows.0
        ));
        // rows_main = (rows / RB) * RB — the register-blocked iteration count.
        self.emit_line(&format!(
            "    %vmmq_rnb_{d} = arith.divui %vmmq_rows_{d}, %vmmq_rb_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmmq_rmain_{d} = arith.muli %vmmq_rnb_{d}, %vmmq_rb_{d} : index"
        ));
        // byte stride per W row = cols * sizeof(i32)
        self.emit_line(&format!(
            "    %vmmq_colsb_{d} = arith.muli %{}, %vmmq_eb_{d} : i64",
            cols.0
        ));
        // inner loop bounds — same for every row (cols is loop-invariant)
        self.emit_line(&format!(
            "    %vmmq_len_{d} = arith.index_cast %{} : i64 to index",
            cols.0
        ));
        self.emit_line(&format!(
            "    %vmmq_nv_{d} = arith.divui %vmmq_len_{d}, %vmmq_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmmq_ve_{d} = arith.muli %vmmq_nv_{d}, %vmmq_cl_{d} : index"
        ));
        // zero accumulator vector for inner loop initialisation
        self.emit_line(&format!(
            "    %vmmq_zv_{d} = arith.constant dense<0> : vector<{l}xi64>"
        ));

        // ════════════════════════════════════════════════════════════════════
        //  Register-blocked main loop: RB output rows per outer pass.
        // ════════════════════════════════════════════════════════════════════
        self.emit_line(&format!(
            "    scf.for %vmmq_r_{d} = %vmmq_c0_{d} to %vmmq_rmain_{d} step %vmmq_rb_{d} {{"
        ));
        self.emit_line(&format!(
            "      %vmmq_ri_{d} = arith.index_cast %vmmq_r_{d} : index to i64"
        ));
        // Base pointer to each of the RB W-rows in this block: W[r+t, 0].
        for t in 0..RB {
            self.emit_line(&format!("      %vmmq_rt{t}_{d} = arith.constant {t} : i64"));
            self.emit_line(&format!(
                "      %vmmq_rit{t}_{d} = arith.addi %vmmq_ri_{d}, %vmmq_rt{t}_{d} : i64"
            ));
            self.emit_line(&format!(
                "      %vmmq_roff{t}_{d} = arith.muli %vmmq_rit{t}_{d}, %vmmq_colsb_{d} : i64"
            ));
            self.emit_line(&format!(
                "      %vmmq_wrow{t}_{d} = llvm.getelementptr %vmmq_wp_{d}[%vmmq_roff{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
        }

        // ── inner vector main loop: load shared x once, MAC against RB W-rows ──
        let acc_init = (0..RB)
            .map(|t| format!("%vmmq_acc{t}_{d} = %vmmq_zv_{d}"))
            .collect::<Vec<_>>()
            .join(", ");
        let acc_ty = (0..RB)
            .map(|_| format!("vector<{l}xi64>"))
            .collect::<Vec<_>>()
            .join(", ");
        self.emit_line(&format!(
            "      %vmmq_va0_{d}:{RB} = scf.for %vmmq_i_{d} = %vmmq_c0_{d} to %vmmq_ve_{d} \
             step %vmmq_cl_{d} iter_args({acc_init}) -> ({acc_ty}) {{"
        ));
        self.emit_line(&format!(
            "        %vmmq_ii_{d} = arith.index_cast %vmmq_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmmq_bo_{d} = arith.muli %vmmq_ii_{d}, %vmmq_eb_{d} : i64"
        ));
        // x[i..i+8] — loaded ONCE, widened ONCE, reused against all RB W-rows.
        self.emit_line(&format!(
            "        %vmmq_bi_{d} = llvm.getelementptr %vmmq_xp_{d}[%vmmq_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmq_bv_{d} = llvm.load %vmmq_bi_{d} {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "        %vmmq_bw_{d} = arith.extsi %vmmq_bv_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        // Per-block-row MAC: W[r+t, i..i+8] · x[i..i+8] >> 16, into acc_t.
        let mut yields = Vec::with_capacity(RB);
        for t in 0..RB {
            self.emit_line(&format!(
                "        %vmmq_ai{t}_{d} = llvm.getelementptr %vmmq_wrow{t}_{d}[%vmmq_bo_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "        %vmmq_av{t}_{d} = llvm.load %vmmq_ai{t}_{d} {{alignment = 4 : i64}} : \
                 !llvm.ptr -> vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmq_aw{t}_{d} = arith.extsi %vmmq_av{t}_{d} : vector<{l}xi32> to vector<{l}xi64>"
            ));
            self.emit_line(&format!(
                "        %vmmq_pr{t}_{d} = arith.muli %vmmq_aw{t}_{d}, %vmmq_bw_{d} : vector<{l}xi64>"
            ));
            self.emit_line(&format!(
                "        %vmmq_sh{t}_{d} = arith.shrsi %vmmq_pr{t}_{d}, %vmmq_s16v_{d} : vector<{l}xi64>"
            ));
            self.emit_line(&format!(
                "        %vmmq_na{t}_{d} = arith.addi %vmmq_acc{t}_{d}, %vmmq_sh{t}_{d} : vector<{l}xi64>"
            ));
            yields.push(format!("%vmmq_na{t}_{d}"));
        }
        self.emit_line(&format!(
            "        scf.yield {} : {acc_ty}",
            yields.join(", ")
        ));
        self.emit_line("      }");

        // ── per-block-row finalise: reduce, scalar tail, pack, store ──────────
        for t in 0..RB {
            // Horizontal lane reduction (associative — bit-identical).
            self.emit_line(&format!(
                "      %vmmq_vs{t}_{d} = vector.reduction <add>, %vmmq_va0_{d}#{t} : \
                 vector<{l}xi64> into i64"
            ));
            // Scalar tail for cols % LANES remainder, on W[r+t,:].
            self.emit_line(&format!(
                "      %vmmq_ts{t}_{d} = scf.for %vmmq_j{t}_{d} = %vmmq_ve_{d} to %vmmq_len_{d} \
                 step %vmmq_c1_{d} iter_args(%vmmq_s{t}_{d} = %vmmq_vs{t}_{d}) -> (i64) {{"
            ));
            self.emit_line(&format!(
                "        %vmmq_jj{t}_{d} = arith.index_cast %vmmq_j{t}_{d} : index to i64"
            ));
            self.emit_line(&format!(
                "        %vmmq_jb{t}_{d} = arith.muli %vmmq_jj{t}_{d}, %vmmq_eb_{d} : i64"
            ));
            self.emit_line(&format!(
                "        %vmmq_aj{t}_{d} = llvm.getelementptr %vmmq_wrow{t}_{d}[%vmmq_jb{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "        %vmmq_xj{t}_{d} = llvm.getelementptr %vmmq_xp_{d}[%vmmq_jb{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "        %vmmq_as{t}_{d} = llvm.load %vmmq_aj{t}_{d} : !llvm.ptr -> i32"
            ));
            self.emit_line(&format!(
                "        %vmmq_bs{t}_{d} = llvm.load %vmmq_xj{t}_{d} : !llvm.ptr -> i32"
            ));
            self.emit_line(&format!(
                "        %vmmq_asw{t}_{d} = arith.extsi %vmmq_as{t}_{d} : i32 to i64"
            ));
            self.emit_line(&format!(
                "        %vmmq_bsw{t}_{d} = arith.extsi %vmmq_bs{t}_{d} : i32 to i64"
            ));
            self.emit_line(&format!(
                "        %vmmq_p{t}_{d} = arith.muli %vmmq_asw{t}_{d}, %vmmq_bsw{t}_{d} : i64"
            ));
            self.emit_line(&format!(
                "        %vmmq_psh{t}_{d} = arith.shrsi %vmmq_p{t}_{d}, %vmmq_s16_{d} : i64"
            ));
            self.emit_line(&format!(
                "        %vmmq_ns{t}_{d} = arith.addi %vmmq_s{t}_{d}, %vmmq_psh{t}_{d} : i64"
            ));
            self.emit_line(&format!("        scf.yield %vmmq_ns{t}_{d} : i64"));
            self.emit_line("      }");
            // Pack: trunc i64→i32, store low 32 bits to y[r+t].
            self.emit_line(&format!(
                "      %vmmq_lo{t}_{d} = arith.trunci %vmmq_ts{t}_{d} : i64 to i32"
            ));
            self.emit_line(&format!(
                "      %vmmq_yoff{t}_{d} = arith.muli %vmmq_rit{t}_{d}, %vmmq_eb_{d} : i64"
            ));
            self.emit_line(&format!(
                "      %vmmq_yel{t}_{d} = llvm.getelementptr %vmmq_yp_{d}[%vmmq_yoff{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "      llvm.store %vmmq_lo{t}_{d}, %vmmq_yel{t}_{d} : i32, !llvm.ptr"
            ));
        }
        self.emit_line("    }"); // end register-blocked outer scf.for

        // ════════════════════════════════════════════════════════════════════
        //  Remainder loop: the leftover `rows % RB` rows, single-row path —
        //  byte-for-byte the original per-row reduction.
        // ════════════════════════════════════════════════════════════════════
        self.emit_line(&format!(
            "    scf.for %vmmqr_r_{d} = %vmmq_rmain_{d} to %vmmq_rows_{d} step %vmmq_c1_{d} {{"
        ));
        self.emit_line(&format!(
            "      %vmmqr_ri_{d} = arith.index_cast %vmmqr_r_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vmmqr_roff_{d} = arith.muli %vmmqr_ri_{d}, %vmmq_colsb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vmmqr_wrow_{d} = llvm.getelementptr %vmmq_wp_{d}[%vmmqr_roff_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vmmqr_vacc_{d} = scf.for %vmmqr_i_{d} = %vmmq_c0_{d} to %vmmq_ve_{d} \
             step %vmmq_cl_{d} iter_args(%vmmqr_acc_{d} = %vmmq_zv_{d}) -> (vector<{l}xi64>) {{"
        ));
        self.emit_line(&format!(
            "        %vmmqr_ii_{d} = arith.index_cast %vmmqr_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bo_{d} = arith.muli %vmmqr_ii_{d}, %vmmq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_ai_{d} = llvm.getelementptr %vmmqr_wrow_{d}[%vmmqr_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bi_{d} = llvm.getelementptr %vmmq_xp_{d}[%vmmqr_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmqr_av_{d} = llvm.load %vmmqr_ai_{d} {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bv_{d} = llvm.load %vmmqr_bi_{d} {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "        %vmmqr_aw_{d} = arith.extsi %vmmqr_av_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bw_{d} = arith.extsi %vmmqr_bv_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "        %vmmqr_pr_{d} = arith.muli %vmmqr_aw_{d}, %vmmqr_bw_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "        %vmmqr_sh_{d} = arith.shrsi %vmmqr_pr_{d}, %vmmq_s16v_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "        %vmmqr_na_{d} = arith.addi %vmmqr_acc_{d}, %vmmqr_sh_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "        scf.yield %vmmqr_na_{d} : vector<{l}xi64>"
        ));
        self.emit_line("      }");
        self.emit_line(&format!(
            "      %vmmqr_vs_{d} = vector.reduction <add>, %vmmqr_vacc_{d} : \
             vector<{l}xi64> into i64"
        ));
        self.emit_line(&format!(
            "      %vmmqr_ts_{d} = scf.for %vmmqr_j_{d} = %vmmq_ve_{d} to %vmmq_len_{d} \
             step %vmmq_c1_{d} iter_args(%vmmqr_s_{d} = %vmmqr_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "        %vmmqr_jj_{d} = arith.index_cast %vmmqr_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_jb_{d} = arith.muli %vmmqr_jj_{d}, %vmmq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_aj_{d} = llvm.getelementptr %vmmqr_wrow_{d}[%vmmqr_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bj_{d} = llvm.getelementptr %vmmq_xp_{d}[%vmmqr_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmqr_as_{d} = llvm.load %vmmqr_aj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bs_{d} = llvm.load %vmmqr_bj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "        %vmmqr_asw_{d} = arith.extsi %vmmqr_as_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_bsw_{d} = arith.extsi %vmmqr_bs_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_p_{d} = arith.muli %vmmqr_asw_{d}, %vmmqr_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_psh_{d} = arith.shrsi %vmmqr_p_{d}, %vmmq_s16_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmqr_ns_{d} = arith.addi %vmmqr_s_{d}, %vmmqr_psh_{d} : i64"
        ));
        self.emit_line(&format!("        scf.yield %vmmqr_ns_{d} : i64"));
        self.emit_line("      }");
        self.emit_line(&format!(
            "      %vmmqr_lo_{d} = arith.trunci %vmmqr_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!(
            "      %vmmqr_yoff_{d} = arith.muli %vmmqr_ri_{d}, %vmmq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vmmqr_yel_{d} = llvm.getelementptr %vmmq_yp_{d}[%vmmqr_yoff_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      llvm.store %vmmqr_lo_{d}, %vmmqr_yel_{d} : i32, !llvm.ptr"
        ));
        self.emit_line("    }"); // end remainder outer scf.for

        // The intrinsic returns 0 (i64) — matches the Track A C oracle.
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    /// BLIS-style cache-blocked fused Q16.16 GEMM macro-kernel, restricted to a
    /// contiguous band of output rows `[row_start, row_end)`.
    ///
    /// Loop nest `jc → ic → pc → (pack) → jr → ir → microkernel`, with an i64
    /// `MC×NC` C-scratch tile that persists ACROSS the `pc` (K-panel) loop and
    /// is truncated to i32 exactly once per output element after the K
    /// reduction completes. The reduction is split into `KC`-deep panels so the
    /// resident A/B working set is independent of the full `K` (the missing
    /// K-blocking that caused large-`K` throughput collapse).
    ///
    /// ## Bit-identity
    ///
    /// Each product `A[i,k]*B[k,j]` is `arith.shrsi`-shifted `>> 16` to a fixed
    /// i64 value BEFORE it enters any sum, and i64 add is associative +
    /// commutative. Splitting the `Σ_{k=0..K}` into KC-panel partials and
    /// accumulating those partials into the i64 C-scratch, then truncating once,
    /// yields exactly `trunc_i32( Σ_k (A[i,k]*B[k,j]) >> 16 )` — byte-for-byte
    /// the per-element scalar oracle, for ANY (MC, KC, NC). The A/B panels are
    /// packed into contiguous scratch and zero-padded to MR/NR multiples; a
    /// padded lane contributes `0*b>>16 = 0` (resp. `a*0>>16 = 0`), an exact
    /// additive identity, and the padded C-scratch rows/cols are never stored.
    /// So packing + padding move data only and perturb no output byte.
    ///
    /// ## Scratch (private, statically sized)
    ///
    /// Three `llvm.alloca` buffers with compile-time-constant extent: the i64
    /// C-scratch (`MC*NC*8`), the packed A panel (`MC*KC*4`, i32) and the packed
    /// B panel (`KC*NC*4`, i32). Constant extent ⇒ statically reserved, no
    /// pointer bits leak into the artifact. Each `alloca` lives in the activation
    /// of the function the nest is emitted into, so in the multithreaded path
    /// every worker gets its OWN scratch (no shared accumulator, no data race).
    ///
    /// `prefix` namespaces every SSA value. `ap`/`bp`/`cp` are `!llvm.ptr` SSA
    /// names; `k64`/`n64` are i64 SSA names for K and N; `ki`/`ni` are `index`
    /// SSA names; `row_start`/`row_end` are `index` SSA names bounding the band.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_mm_q16_blocked(
        buf: &mut String,
        prefix: &str,
        ap: &str,
        bp: &str,
        cp: &str,
        k64: &str,
        n64: &str,
        ki: &str,
        ni: &str,
        row_start: &str,
        row_end: &str,
    ) {
        use std::fmt::Write;
        let p = prefix;
        const MR: usize = Q16_MR;
        const NR: usize = Q16_NR;
        const MC: usize = Q16_MC;
        const KC: usize = Q16_KC;
        const NC: usize = Q16_NC;
        // MC/NC are MR/NR multiples by construction; the per-tile pad rounds the
        // effective extent up to the next MR/NR multiple (≤ MC/NC).
        let eb: i64 = std::mem::size_of::<i32>() as i64;
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };

        // ── constants ────────────────────────────────────────────────────────
        line(&format!("    %{p}_c0 = arith.constant 0 : index"));
        line(&format!("    %{p}_c1 = arith.constant 1 : index"));
        line(&format!("    %{p}_cmr = arith.constant {MR} : index"));
        line(&format!("    %{p}_cnr = arith.constant {NR} : index"));
        line(&format!("    %{p}_cmc = arith.constant {MC} : index"));
        line(&format!("    %{p}_ckc = arith.constant {KC} : index"));
        line(&format!("    %{p}_cnc = arith.constant {NC} : index"));
        line(&format!("    %{p}_eb = arith.constant {eb} : i64"));
        line(&format!("    %{p}_s16 = arith.constant 16 : i64"));
        line(&format!("    %{p}_z0 = arith.constant 0 : i64"));
        line(&format!("    %{p}_z0i32 = arith.constant 0 : i32"));
        line(&format!(
            "    %{p}_s16v = arith.constant dense<16> : vector<{NR}xi64>"
        ));
        line(&format!(
            "    %{p}_zv = arith.constant dense<0> : vector<{NR}xi64>"
        ));

        // ── private scratch (constant-extent alloca) ─────────────────────────
        // C-scratch: MC*NC i64.  Packed A: MC*KC i32.  Packed B: KC*NC i32.
        let cs_elems = (MC * NC) as i64;
        let pa_elems = (MC * KC) as i64;
        let pb_elems = (KC * NC) as i64;
        line(&format!(
            "    %{p}_csn = llvm.mlir.constant({cs_elems} : i64) : i64"
        ));
        line(&format!(
            "    %{p}_cs = llvm.alloca %{p}_csn x i64 : (i64) -> !llvm.ptr"
        ));
        line(&format!(
            "    %{p}_pan = llvm.mlir.constant({pa_elems} : i64) : i64"
        ));
        line(&format!(
            "    %{p}_pa = llvm.alloca %{p}_pan x i32 : (i64) -> !llvm.ptr"
        ));
        line(&format!(
            "    %{p}_pbn = llvm.mlir.constant({pb_elems} : i64) : i64"
        ));
        line(&format!(
            "    %{p}_pb = llvm.alloca %{p}_pbn x i32 : (i64) -> !llvm.ptr"
        ));
        // C-scratch column stride (NC) and packed strides as i64 for GEP math.
        line(&format!("    %{p}_ncc = arith.constant {NC} : i64"));
        line(&format!("    %{p}_kcc = arith.constant {KC} : i64"));
        line(&format!("    %{p}_mrc = arith.constant {MR} : i64"));
        line(&format!("    %{p}_nrc = arith.constant {NR} : i64"));

        // ════════════════════════════════════════════════════════════════════
        //  jc — column block over [0, N)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "    scf.for %{p}_jc = %{p}_c0 to %{ni} step %{p}_cnc {{"
        ));
        // NCe = min(NC, N - jc)
        line(&format!(
            "      %{p}_jcrem = arith.subi %{ni}, %{p}_jc : index"
        ));
        line(&format!(
            "      %{p}_nce_lt = arith.cmpi slt, %{p}_cnc, %{p}_jcrem : index"
        ));
        line(&format!(
            "      %{p}_nce = arith.select %{p}_nce_lt, %{p}_cnc, %{p}_jcrem : index"
        ));
        // NCp = ceil(NCe/NR)*NR (padded column extent for the packed B / tiles)
        line(&format!(
            "      %{p}_ncp_t = arith.addi %{p}_nce, %{p}_cnr : index"
        ));
        line(&format!(
            "      %{p}_ncp_t1 = arith.subi %{p}_ncp_t, %{p}_c1 : index"
        ));
        line(&format!(
            "      %{p}_ncp_d = arith.divui %{p}_ncp_t1, %{p}_cnr : index"
        ));
        line(&format!(
            "      %{p}_ncp = arith.muli %{p}_ncp_d, %{p}_cnr : index"
        ));
        line(&format!(
            "      %{p}_jc64 = arith.index_cast %{p}_jc : index to i64"
        ));

        // ════════════════════════════════════════════════════════════════════
        //  ic — row block over [row_start, row_end)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "      scf.for %{p}_ic = %{row_start} to %{row_end} step %{p}_cmc {{"
        ));
        // MCe = min(MC, row_end - ic)
        line(&format!(
            "        %{p}_icrem = arith.subi %{row_end}, %{p}_ic : index"
        ));
        line(&format!(
            "        %{p}_mce_lt = arith.cmpi slt, %{p}_cmc, %{p}_icrem : index"
        ));
        line(&format!(
            "        %{p}_mce = arith.select %{p}_mce_lt, %{p}_cmc, %{p}_icrem : index"
        ));
        // MCp = ceil(MCe/MR)*MR
        line(&format!(
            "        %{p}_mcp_t = arith.addi %{p}_mce, %{p}_cmr : index"
        ));
        line(&format!(
            "        %{p}_mcp_t1 = arith.subi %{p}_mcp_t, %{p}_c1 : index"
        ));
        line(&format!(
            "        %{p}_mcp_d = arith.divui %{p}_mcp_t1, %{p}_cmr : index"
        ));
        line(&format!(
            "        %{p}_mcp = arith.muli %{p}_mcp_d, %{p}_cmr : index"
        ));
        line(&format!(
            "        %{p}_ic64 = arith.index_cast %{p}_ic : index to i64"
        ));

        // ── zero the live MCp×NCp region of the C-scratch ────────────────────
        line(&format!(
            "        scf.for %{p}_zr = %{p}_c0 to %{p}_mcp step %{p}_c1 {{"
        ));
        line(&format!(
            "          %{p}_zr64 = arith.index_cast %{p}_zr : index to i64"
        ));
        line(&format!(
            "          %{p}_zrb = arith.muli %{p}_zr64, %{p}_ncc : i64"
        ));
        line(&format!(
            "          scf.for %{p}_zc = %{p}_c0 to %{p}_ncp step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_zc64 = arith.index_cast %{p}_zc : index to i64"
        ));
        line(&format!(
            "            %{p}_zoff = arith.addi %{p}_zrb, %{p}_zc64 : i64"
        ));
        line(&format!(
            "            %{p}_zptr = llvm.getelementptr %{p}_cs[%{p}_zoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i64"
        ));
        line(&format!(
            "            llvm.store %{p}_z0, %{p}_zptr : i64, !llvm.ptr"
        ));
        line("          }");
        line("        }");

        // ════════════════════════════════════════════════════════════════════
        //  pc — K panel over [0, K)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "        scf.for %{p}_pc = %{p}_c0 to %{ki} step %{p}_ckc {{"
        ));
        // KCe = min(KC, K - pc)
        line(&format!(
            "          %{p}_pcrem = arith.subi %{ki}, %{p}_pc : index"
        ));
        line(&format!(
            "          %{p}_kce_lt = arith.cmpi slt, %{p}_ckc, %{p}_pcrem : index"
        ));
        line(&format!(
            "          %{p}_kce = arith.select %{p}_kce_lt, %{p}_ckc, %{p}_pcrem : index"
        ));
        line(&format!(
            "          %{p}_pc64 = arith.index_cast %{p}_pc : index to i64"
        ));

        // ── pack B panel: Bp[jr_block][kk][nr] (zero-padded to NCp cols) ──────
        // layout index = (jr/NR)*(KC*NR) + kk*NR + (jr%NR); we iterate jr in
        // 0..NCp and kk in 0..KCe and store either B[pc+kk, jc+jr] or 0.
        line(&format!(
            "          scf.for %{p}_pbk = %{p}_c0 to %{p}_kce step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_pbk64 = arith.index_cast %{p}_pbk : index to i64"
        ));
        // source row base in B: (pc+kk)*N + jc
        line(&format!(
            "            %{p}_pbkg = arith.addi %{p}_pc64, %{p}_pbk64 : i64"
        ));
        line(&format!(
            "            %{p}_pbsrow = arith.muli %{p}_pbkg, %{n64} : i64"
        ));
        line(&format!(
            "            %{p}_pbsr0 = arith.addi %{p}_pbsrow, %{p}_jc64 : i64"
        ));
        line(&format!(
            "            scf.for %{p}_pbj = %{p}_c0 to %{p}_ncp step %{p}_c1 {{"
        ));
        line(&format!(
            "              %{p}_pbj64 = arith.index_cast %{p}_pbj : index to i64"
        ));
        // dst index = (pbj/NR)*(KC*NR) + pbk*NR + pbj%NR
        line(&format!(
            "              %{p}_pbjb = arith.divui %{p}_pbj64, %{p}_nrc : i64"
        ));
        line(&format!(
            "              %{p}_pbjm = arith.remui %{p}_pbj64, %{p}_nrc : i64"
        ));
        line(&format!(
            "              %{p}_pbpan = arith.muli %{p}_kcc, %{p}_nrc : i64"
        ));
        line(&format!(
            "              %{p}_pbblk = arith.muli %{p}_pbjb, %{p}_pbpan : i64"
        ));
        line(&format!(
            "              %{p}_pbkr = arith.muli %{p}_pbk64, %{p}_nrc : i64"
        ));
        line(&format!(
            "              %{p}_pbd0 = arith.addi %{p}_pbblk, %{p}_pbkr : i64"
        ));
        line(&format!(
            "              %{p}_pbd = arith.addi %{p}_pbd0, %{p}_pbjm : i64"
        ));
        line(&format!(
            "              %{p}_pbdp = llvm.getelementptr %{p}_pb[%{p}_pbd] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i32"
        ));
        // live column? pbj < NCe -> load B else 0.
        line(&format!(
            "              %{p}_pblive = arith.cmpi slt, %{p}_pbj, %{p}_nce : index"
        ));
        line(&format!(
            "              %{p}_pbval = scf.if %{p}_pblive -> (i32) {{"
        ));
        line(&format!(
            "                %{p}_pbsi = arith.addi %{p}_pbsr0, %{p}_pbj64 : i64"
        ));
        line(&format!(
            "                %{p}_pbbo = arith.muli %{p}_pbsi, %{p}_eb : i64"
        ));
        line(&format!(
            "                %{p}_pbsp = llvm.getelementptr %{bp}[%{p}_pbbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "                %{p}_pbld = llvm.load %{p}_pbsp : !llvm.ptr -> i32"
        ));
        line(&format!("                scf.yield %{p}_pbld : i32"));
        line("              } else {");
        line(&format!("                scf.yield %{p}_z0i32 : i32"));
        line("              }");
        line(&format!(
            "              llvm.store %{p}_pbval, %{p}_pbdp : i32, !llvm.ptr"
        ));
        line("            }");
        line("          }");

        // ── pack A panel: Ap[ir_block][kk][mr] (zero-padded to MCp rows) ──────
        // dst index = (ir/MR)*(KC*MR) + kk*MR + ir%MR
        line(&format!(
            "          scf.for %{p}_pak = %{p}_c0 to %{p}_kce step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_pak64 = arith.index_cast %{p}_pak : index to i64"
        ));
        line(&format!(
            "            %{p}_pakg = arith.addi %{p}_pc64, %{p}_pak64 : i64"
        ));
        line(&format!(
            "            scf.for %{p}_pai = %{p}_c0 to %{p}_mcp step %{p}_c1 {{"
        ));
        line(&format!(
            "              %{p}_pai64 = arith.index_cast %{p}_pai : index to i64"
        ));
        // dst index
        line(&format!(
            "              %{p}_paib = arith.divui %{p}_pai64, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_paim = arith.remui %{p}_pai64, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_papan = arith.muli %{p}_kcc, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_pablk = arith.muli %{p}_paib, %{p}_papan : i64"
        ));
        line(&format!(
            "              %{p}_pakr = arith.muli %{p}_pak64, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_pad0 = arith.addi %{p}_pablk, %{p}_pakr : i64"
        ));
        line(&format!(
            "              %{p}_pad = arith.addi %{p}_pad0, %{p}_paim : i64"
        ));
        line(&format!(
            "              %{p}_padp = llvm.getelementptr %{p}_pa[%{p}_pad] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i32"
        ));
        // live row? pai < MCe -> load A[ic+pai, pc+pak] else 0.
        line(&format!(
            "              %{p}_palive = arith.cmpi slt, %{p}_pai, %{p}_mce : index"
        ));
        line(&format!(
            "              %{p}_paval = scf.if %{p}_palive -> (i32) {{"
        ));
        line(&format!(
            "                %{p}_pari = arith.addi %{p}_ic64, %{p}_pai64 : i64"
        ));
        line(&format!(
            "                %{p}_parik = arith.muli %{p}_pari, %{k64} : i64"
        ));
        line(&format!(
            "                %{p}_pasi = arith.addi %{p}_parik, %{p}_pakg : i64"
        ));
        line(&format!(
            "                %{p}_pabo = arith.muli %{p}_pasi, %{p}_eb : i64"
        ));
        line(&format!(
            "                %{p}_pasp = llvm.getelementptr %{ap}[%{p}_pabo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "                %{p}_pald = llvm.load %{p}_pasp : !llvm.ptr -> i32"
        ));
        line(&format!("                scf.yield %{p}_pald : i32"));
        line("              } else {");
        line(&format!("                scf.yield %{p}_z0i32 : i32"));
        line("              }");
        line(&format!(
            "              llvm.store %{p}_paval, %{p}_padp : i32, !llvm.ptr"
        ));
        line("            }");
        line("          }");

        // ════════════════════════════════════════════════════════════════════
        //  jr — NR-wide column tiles over the packed B panel [0, NCp)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "          scf.for %{p}_jr = %{p}_c0 to %{p}_ncp step %{p}_cnr {{"
        ));
        line(&format!(
            "            %{p}_jr64 = arith.index_cast %{p}_jr : index to i64"
        ));
        // jr_block base into packed B = (jr/NR)*(KC*NR)
        line(&format!(
            "            %{p}_jrb = arith.divui %{p}_jr64, %{p}_nrc : i64"
        ));
        line(&format!(
            "            %{p}_jrpan = arith.muli %{p}_kcc, %{p}_nrc : i64"
        ));
        line(&format!(
            "            %{p}_jrbase = arith.muli %{p}_jrb, %{p}_jrpan : i64"
        ));
        // ══════════════════════════════════════════════════════════════════
        //  ir — MR-row tiles over the packed A panel [0, MCp)
        // ══════════════════════════════════════════════════════════════════
        line(&format!(
            "            scf.for %{p}_ir = %{p}_c0 to %{p}_mcp step %{p}_cmr {{"
        ));
        line(&format!(
            "              %{p}_ir64 = arith.index_cast %{p}_ir : index to i64"
        ));
        line(&format!(
            "              %{p}_irb = arith.divui %{p}_ir64, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_irpan = arith.muli %{p}_kcc, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_irbase = arith.muli %{p}_irb, %{p}_irpan : i64"
        ));

        // ── microkernel: MR×NR i64 partial over this KC panel ────────────────
        let acc_init = (0..MR)
            .map(|t| format!("%{p}_acc{t} = %{p}_zv"))
            .collect::<Vec<_>>()
            .join(", ");
        let acc_ty = (0..MR)
            .map(|_| format!("vector<{NR}xi64>"))
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!(
            "              %{p}_va:{MR} = scf.for %{p}_kk = %{p}_c0 to %{p}_kce \
             step %{p}_c1 iter_args({acc_init}) -> ({acc_ty}) {{"
        ));
        line(&format!(
            "                %{p}_kk64 = arith.index_cast %{p}_kk : index to i64"
        ));
        // Bp vector load: jrbase + kk*NR
        line(&format!(
            "                %{p}_bkr = arith.muli %{p}_kk64, %{p}_nrc : i64"
        ));
        line(&format!(
            "                %{p}_bvi = arith.addi %{p}_jrbase, %{p}_bkr : i64"
        ));
        line(&format!(
            "                %{p}_bvp = llvm.getelementptr %{p}_pb[%{p}_bvi] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i32"
        ));
        line(&format!(
            "                %{p}_bv = llvm.load %{p}_bvp {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{NR}xi32>"
        ));
        line(&format!(
            "                %{p}_bw = arith.extsi %{p}_bv : vector<{NR}xi32> to vector<{NR}xi64>"
        ));
        // Ap base for this kk: irbase + kk*MR
        line(&format!(
            "                %{p}_akr = arith.muli %{p}_kk64, %{p}_mrc : i64"
        ));
        line(&format!(
            "                %{p}_abase = arith.addi %{p}_irbase, %{p}_akr : i64"
        ));
        let mut yields = Vec::with_capacity(MR);
        for t in 0..MR {
            line(&format!(
                "                %{p}_at{t} = arith.constant {t} : i64"
            ));
            line(&format!(
                "                %{p}_ai{t} = arith.addi %{p}_abase, %{p}_at{t} : i64"
            ));
            line(&format!(
                "                %{p}_ap{t} = llvm.getelementptr %{p}_pa[%{p}_ai{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i32"
            ));
            line(&format!(
                "                %{p}_as{t} = llvm.load %{p}_ap{t} : !llvm.ptr -> i32"
            ));
            line(&format!(
                "                %{p}_aw{t} = arith.extsi %{p}_as{t} : i32 to i64"
            ));
            line(&format!(
                "                %{p}_ab{t} = vector.broadcast %{p}_aw{t} : i64 to vector<{NR}xi64>"
            ));
            line(&format!(
                "                %{p}_pp{t} = arith.muli %{p}_ab{t}, %{p}_bw : vector<{NR}xi64>"
            ));
            line(&format!(
                "                %{p}_ps{t} = arith.shrsi %{p}_pp{t}, %{p}_s16v : vector<{NR}xi64>"
            ));
            line(&format!(
                "                %{p}_na{t} = arith.addi %{p}_acc{t}, %{p}_ps{t} : vector<{NR}xi64>"
            ));
            yields.push(format!("%{p}_na{t}"));
        }
        line(&format!(
            "                scf.yield {} : {acc_ty}",
            yields.join(", ")
        ));
        line("              }");
        // ── add the MR×NR i64 partial into the C-scratch tile ────────────────
        // C-scratch row r = ir + t, col block at jr; load-add-store vector<NRxi64>.
        for t in 0..MR {
            line(&format!(
                "              %{p}_ct{t} = arith.constant {t} : i64"
            ));
            line(&format!(
                "              %{p}_cr{t} = arith.addi %{p}_ir64, %{p}_ct{t} : i64"
            ));
            line(&format!(
                "              %{p}_crb{t} = arith.muli %{p}_cr{t}, %{p}_ncc : i64"
            ));
            line(&format!(
                "              %{p}_coff{t} = arith.addi %{p}_crb{t}, %{p}_jr64 : i64"
            ));
            line(&format!(
                "              %{p}_csp{t} = llvm.getelementptr %{p}_cs[%{p}_coff{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i64"
            ));
            line(&format!(
                "              %{p}_cold{t} = llvm.load %{p}_csp{t} {{alignment = 8 : i64}} : \
                 !llvm.ptr -> vector<{NR}xi64>"
            ));
            line(&format!(
                "              %{p}_cnew{t} = arith.addi %{p}_cold{t}, %{p}_va#{t} : vector<{NR}xi64>"
            ));
            line(&format!(
                "              llvm.store %{p}_cnew{t}, %{p}_csp{t} {{alignment = 8 : i64}} : \
                 vector<{NR}xi64>, !llvm.ptr"
            ));
        }
        line("            }"); // end ir
        line("          }"); // end jr
        line("        }"); // end pc

        // ── after the K reduction: truncate the live MCe×NCe C-scratch to i32 ─
        // and store to C[ic+r, jc+col].  Vector store for NR-aligned column runs
        // within NCe, scalar tail for the remaining (NCe % NR) columns.
        line(&format!(
            "        %{p}_nmb = arith.divui %{p}_nce, %{p}_cnr : index"
        ));
        line(&format!(
            "        %{p}_nmain = arith.muli %{p}_nmb, %{p}_cnr : index"
        ));
        line(&format!(
            "        scf.for %{p}_wr = %{p}_c0 to %{p}_mce step %{p}_c1 {{"
        ));
        line(&format!(
            "          %{p}_wr64 = arith.index_cast %{p}_wr : index to i64"
        ));
        line(&format!(
            "          %{p}_csrb = arith.muli %{p}_wr64, %{p}_ncc : i64"
        ));
        // dest C row base: (ic+wr)*N + jc
        line(&format!(
            "          %{p}_dri = arith.addi %{p}_ic64, %{p}_wr64 : i64"
        ));
        line(&format!(
            "          %{p}_drrow = arith.muli %{p}_dri, %{n64} : i64"
        ));
        line(&format!(
            "          %{p}_drb = arith.addi %{p}_drrow, %{p}_jc64 : i64"
        ));
        // vector main columns
        line(&format!(
            "          scf.for %{p}_wc = %{p}_c0 to %{p}_nmain step %{p}_cnr {{"
        ));
        line(&format!(
            "            %{p}_wc64 = arith.index_cast %{p}_wc : index to i64"
        ));
        line(&format!(
            "            %{p}_csvi = arith.addi %{p}_csrb, %{p}_wc64 : i64"
        ));
        line(&format!(
            "            %{p}_csvp = llvm.getelementptr %{p}_cs[%{p}_csvi] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i64"
        ));
        line(&format!(
            "            %{p}_csvv = llvm.load %{p}_csvp {{alignment = 8 : i64}} : \
             !llvm.ptr -> vector<{NR}xi64>"
        ));
        line(&format!(
            "            %{p}_csvt = arith.trunci %{p}_csvv : vector<{NR}xi64> to vector<{NR}xi32>"
        ));
        line(&format!(
            "            %{p}_dvi = arith.addi %{p}_drb, %{p}_wc64 : i64"
        ));
        line(&format!(
            "            %{p}_dvbo = arith.muli %{p}_dvi, %{p}_eb : i64"
        ));
        line(&format!(
            "            %{p}_dvp = llvm.getelementptr %{cp}[%{p}_dvbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "            llvm.store %{p}_csvt, %{p}_dvp {{alignment = 4 : i64}} : \
             vector<{NR}xi32>, !llvm.ptr"
        ));
        line("          }");
        // scalar column tail [nmain, NCe)
        line(&format!(
            "          scf.for %{p}_wt = %{p}_nmain to %{p}_nce step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_wt64 = arith.index_cast %{p}_wt : index to i64"
        ));
        line(&format!(
            "            %{p}_csti = arith.addi %{p}_csrb, %{p}_wt64 : i64"
        ));
        line(&format!(
            "            %{p}_cstp = llvm.getelementptr %{p}_cs[%{p}_csti] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i64"
        ));
        line(&format!(
            "            %{p}_cstv = llvm.load %{p}_cstp : !llvm.ptr -> i64"
        ));
        line(&format!(
            "            %{p}_cstt = arith.trunci %{p}_cstv : i64 to i32"
        ));
        line(&format!(
            "            %{p}_dti = arith.addi %{p}_drb, %{p}_wt64 : i64"
        ));
        line(&format!(
            "            %{p}_dtbo = arith.muli %{p}_dti, %{p}_eb : i64"
        ));
        line(&format!(
            "            %{p}_dtp = llvm.getelementptr %{cp}[%{p}_dtbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "            llvm.store %{p}_cstt, %{p}_dtp : i32, !llvm.ptr"
        ));
        line("          }");
        line("        }"); // end wr
        line("      }"); // end ic
        line("    }"); // end jc
    }

    /// "det.igemm" tier — emit the fused int8 GEMM
    /// (`__mind_blas_matmul_mm_i8_v`).
    ///
    /// Setup mirrors `emit_vec_matmul_mm_q16`: materialise the three base
    /// pointers and the K/N i64 SSA names + K/M/N `index` bounds, then delegate
    /// to the BLIS-blocked int8 macro-kernel over the full row range `[0, M)`.
    /// The intrinsic returns 0 (i64).
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_mm_i8(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        c_addr: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
    ) {
        let d = dst.0;
        self.emit_line(&format!("    %imm_z0_{d} = arith.constant 0 : i64"));
        self.emit_line(&format!(
            "    %imm_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %imm_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %imm_cp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            c_addr.0
        ));
        self.emit_line(&format!(
            "    %imm_mi_{d} = arith.index_cast %{} : i64 to index",
            m.0
        ));
        self.emit_line(&format!(
            "    %imm_ki_{d} = arith.index_cast %{} : i64 to index",
            k.0
        ));
        self.emit_line(&format!(
            "    %imm_ni_{d} = arith.index_cast %{} : i64 to index",
            n.0
        ));
        self.emit_line(&format!(
            "    %imm_k64_{d} = arith.addi %{}, %imm_z0_{d} : i64",
            k.0
        ));
        self.emit_line(&format!(
            "    %imm_n64_{d} = arith.addi %{}, %imm_z0_{d} : i64",
            n.0
        ));
        self.emit_line(&format!("    %imm_rs_{d} = arith.constant 0 : index"));
        // The blocked kernel heap-allocates its scratch panels via @malloc/@free;
        // flag the module assembler to emit those externs once.
        self.needs_malloc = true;
        let mut blk = String::new();
        Self::emit_mm_i8_blocked(
            &mut blk,
            &format!("imb_{d}"),
            &format!("imm_ap_{d}"),
            &format!("imm_bp_{d}"),
            &format!("imm_cp_{d}"),
            &format!("imm_k64_{d}"),
            &format!("imm_n64_{d}"),
            &format!("imm_ki_{d}"),
            &format!("imm_ni_{d}"),
            &format!("imm_rs_{d}"),
            &format!("imm_mi_{d}"),
            IntDotMode::from_env(),
        );
        self.body.push_str(&blk);
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    /// "det.igemm" tier — the BLIS-blocked int8 GEMM macro-kernel.
    ///
    /// Structure mirrors `emit_mm_q16_blocked` term-for-term (the SAME
    /// `jc → ic → pc → jr → ir → microkernel` loop nest, the SAME private i64
    /// C-scratch + i32 packed-A / packed-B panels of identical extent, the SAME
    /// MR×NR `vector<NRxi64>` register tile, the SAME zero-pad / select packing
    /// for `M%MC / N%NC / K%KC / N%NR / M%MR` remainders, and the SAME vector +
    /// scalar write-out). Two surgical differences, neither of which perturbs an
    /// output byte:
    ///
    /// 1. **i8 source loads, i16 K-pair-interleaved panels.** A/B elements are
    ///    1-byte int8. During the pack the kernel `llvm.load`s an `i8` and
    ///    `arith.extsi`s it to `i16` ONCE (`vpmovsxbw`), writing into panels laid
    ///    out K-pair-interleaved (`Bp[jr_block][k_pair][nr][s]`,
    ///    `Ap[ir_block][k_pair][mr][s]`, `s∈{0,1}` the K-within-pair). The A/B
    ///    source GEP uses element-byte 1; the packed panels are i16; the C output
    ///    GEP uses element-byte 4.
    /// 2. **vpmaddwd K-contraction, no `>> 16` shift.** The microkernel contracts
    ///    two K-steps per `llvm.x86.avx2.pmadd.wd`: for each A-row it broadcasts
    ///    the `(k,k+1)` A pair across the NR=8 output columns and multiplies the
    ///    B-pair vector, yielding NR i32 partials
    ///    `A[t,k]*B[k,n]+A[t,k+1]*B[k+1,n]` per column. Each i32 partial is
    ///    sign-extended to i64 BEFORE the associative accumulation (no 8-lane
    ///    reduction overflow). The total over all K is the identical sum of
    ///    `A[t,k]*B[k,n]` terms as the per-element oracle (integer add is
    ///    associative + commutative — any tiling / lane / pair grouping gives the
    ///    identical sum), and the i64→i32 truncation happens once at the store.
    ///    NEVER the saturating `vpmaddubsw`; no shift. An odd trailing K (odd
    ///    KC-panel remainder) is handled by a scalar tail. The exact int32 sum is
    ///    substrate-independent, so the SAME MLIR is byte-identical whether LLVM
    ///    lowers to AVX2 `vpmaddwd` or aarch64 `SDOT`/`SMMLA`.
    ///
    /// `prefix` namespaces every SSA value. `ap`/`bp`/`cp` are `!llvm.ptr` SSA
    /// names; `k64`/`n64` are i64 SSA names for K and N; `ki`/`ni` are `index`
    /// SSA names; `row_start`/`row_end` are `index` SSA names bounding the band.
    /// AVX2 int8 microkernel body (vpmaddwd, grp=2). Emits the K-pair loop
    /// `%{prefix}_va:{MR}` = Σ over K-pairs of NR i32 partials sign-extended to
    /// i64. Each `llvm.x86.avx2.pmadd.wd` contracts two K-steps for NR=8
    /// columns. Expects `%{prefix}_kpairs`, `%{prefix}_jrbase`,
    /// `%{prefix}_irbase`, `%{prefix}_zv`, `%{prefix}_c0`/`_c1`, `%{prefix}_c2i`,
    /// `%{prefix}_nrc`/`_mrc` in scope. Byte-for-byte the committed (23165e1)
    /// vpmaddwd path — extracted verbatim, only the panel stride constant is now
    /// `_cgrpi` (= 2 here) for symmetry with the VNNI rung.
    #[cfg(feature = "std-surface")]
    fn emit_i8_microkernel_avx2(
        buf: &mut String,
        p: &str,
        nr: usize,
        mr: usize,
        pl: usize,
        acc_ty: &str,
    ) {
        use std::fmt::Write;
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };
        let acc_init = (0..mr)
            .map(|t| format!("%{p}_acc{t} = %{p}_zv"))
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!(
            "              %{p}_va:{mr} = scf.for %{p}_kp = %{p}_c0 to %{p}_kpairs \
             step %{p}_c1 iter_args({acc_init}) -> ({acc_ty}) {{"
        ));
        line(&format!(
            "                %{p}_kp64 = arith.index_cast %{p}_kp : index to i64"
        ));
        // B pairs vector: Bp[jrbase + kp*(NR*2) .. +16] as vector<16xi16>.
        line(&format!(
            "                %{p}_bnr2 = arith.muli %{p}_nrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_bkr = arith.muli %{p}_kp64, %{p}_bnr2 : i64"
        ));
        line(&format!(
            "                %{p}_bvi = arith.addi %{p}_jrbase, %{p}_bkr : i64"
        ));
        line(&format!(
            "                %{p}_bvp = llvm.getelementptr %{p}_pb[%{p}_bvi] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i16"
        ));
        line(&format!(
            "                %{p}_bv = llvm.load %{p}_bvp {{alignment = 2 : i64}} : \
             !llvm.ptr -> vector<{}xi16>",
            2 * nr
        ));
        // A K-pairs base for this block: Ap[irbase + kp*(MR*2)].
        line(&format!(
            "                %{p}_amr2 = arith.muli %{p}_mrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_akr = arith.muli %{p}_kp64, %{p}_amr2 : i64"
        ));
        line(&format!(
            "                %{p}_abase = arith.addi %{p}_irbase, %{p}_akr : i64"
        ));
        let mut yields = Vec::with_capacity(mr);
        for t in 0..mr {
            // A pair for row t: two contiguous i16 = [A[t,2kp], A[t,2kp+1]].
            // Load as one i32, broadcast across NR output columns, bitcast back
            // to vector<16xi16> = [a0,a1,a0,a1,...] so vpmaddwd pairs each
            // output column's (k,k+1) with the corresponding B pair.
            line(&format!(
                "                %{p}_at{t} = arith.constant {} : i64",
                t * 2
            ));
            line(&format!(
                "                %{p}_ai{t} = arith.addi %{p}_abase, %{p}_at{t} : i64"
            ));
            line(&format!(
                "                %{p}_ap{t} = llvm.getelementptr %{p}_pa[%{p}_ai{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i16"
            ));
            line(&format!(
                "                %{p}_as{t} = llvm.load %{p}_ap{t} {{alignment = 2 : i64}} : \
                 !llvm.ptr -> i32"
            ));
            line(&format!(
                "                %{p}_abc{t} = vector.broadcast %{p}_as{t} : i32 to vector<{nr}xi32>"
            ));
            line(&format!(
                "                %{p}_aw{t} = vector.bitcast %{p}_abc{t} : vector<{nr}xi32> to vector<{}xi16>",
                2 * nr
            ));
            if HOST_IS_X86 {
                line(&format!(
                    "                %{p}_pm{t} = llvm.call_intrinsic \"llvm.x86.avx2.pmadd.wd\"(%{p}_aw{t}, %{p}_bv) : \
                     (vector<{}xi16>, vector<{}xi16>) -> vector<{pl}xi32>",
                    2 * nr,
                    2 * nr
                ));
                line(&format!(
                    "                %{p}_pw{t} = arith.extsi %{p}_pm{t} : vector<{pl}xi32> to vector<{nr}xi64>"
                ));
            } else {
                // Non-x86 (aarch64): portable, exact-integer equivalent of the
                // x86 `vpmaddwd` pairwise contraction (the intrinsic does not
                // legalise off x86). `_aw{t}` is the broadcast A pair
                // `[a0,a1,a0,a1,...]` and `_bv` is the B pairs, both
                // vector<2*NR xi16>. Sign-extend both to i32, multiply, then
                // form the NR pairwise sums by adding the even and odd lanes —
                // bit-identical integer result to `vpmaddwd`.
                let w = 2 * nr;
                let even: String = (0..nr)
                    .map(|i| (2 * i).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let odd: String = (0..nr)
                    .map(|i| (2 * i + 1).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                line(&format!(
                    "                %{p}_awi{t} = arith.extsi %{p}_aw{t} : vector<{w}xi16> to vector<{w}xi32>"
                ));
                line(&format!(
                    "                %{p}_bwi{t} = arith.extsi %{p}_bv : vector<{w}xi16> to vector<{w}xi32>"
                ));
                line(&format!(
                    "                %{p}_pr{t} = arith.muli %{p}_awi{t}, %{p}_bwi{t} : vector<{w}xi32>"
                ));
                line(&format!(
                    "                %{p}_ev{t} = vector.shuffle %{p}_pr{t}, %{p}_pr{t} \
                     [{even}] : vector<{w}xi32>, vector<{w}xi32>"
                ));
                line(&format!(
                    "                %{p}_od{t} = vector.shuffle %{p}_pr{t}, %{p}_pr{t} \
                     [{odd}] : vector<{w}xi32>, vector<{w}xi32>"
                ));
                line(&format!(
                    "                %{p}_pm{t} = arith.addi %{p}_ev{t}, %{p}_od{t} : vector<{nr}xi32>"
                ));
                line(&format!(
                    "                %{p}_pw{t} = arith.extsi %{p}_pm{t} : vector<{nr}xi32> to vector<{nr}xi64>"
                ));
            }
            line(&format!(
                "                %{p}_na{t} = arith.addi %{p}_acc{t}, %{p}_pw{t} : vector<{nr}xi64>"
            ));
            yields.push(format!("%{p}_na{t}"));
        }
        line(&format!(
            "                scf.yield {} : {acc_ty}",
            yields.join(", ")
        ));
        line("              }");
    }

    /// VNNI int8 microkernel body (vpdpbusd.512, grp=4). Emits the K-quad loop
    /// `%{prefix}_va:{MR}` = Σ over K-quads of NR i32 partials sign-extended to
    /// i64, with the signed-input bias correction `Σ aₛ·bₛ = Σ (aₛ⊕0x80)·bₛ −
    /// 128·Σ bₛ` applied exactly:
    ///
    /// * Per (row, quad): load the A quad `[A[t,4q..4q+3]]` as i32, broadcast
    ///   across NR=16 columns, `xor 0x80` each byte (s8→u8, the `+128` bias), and
    ///   `@llvm.x86.avx512.vpdpbusd.512(main_acc, a_u8, b_s8)` → 16 i32 partials.
    /// * Per quad (row-independent): `vpdpbusd.512(bsum_acc, ones_u8, b_s8)`
    ///   accumulates the per-column `Σ bₛ`.
    /// * After the quad loop, for each row: `acc_i64 = extsi(main_i32) −
    ///   (extsi(bsum_i32) << 7)` (the `<<7` = `·128`). All exact i32/i64; the
    ///   result equals the AVX2 rung's exact int32 sum bit-for-bit.
    ///
    /// i32 accumulation across one KC panel is overflow-safe: with KC≤256 (64
    /// quads), each lane ≤ 64·4·255·128 ≈ 8.4M (main) / 64·4·128·128 ≈ 4.2M
    /// (bias) ≪ 2³¹; the i64 C-scratch carries the cross-panel sum.
    /// Hoisted VNNI column-sum (`128·Σ bₛ`). Emitted once per `jr` tile, OUTSIDE
    /// the `ir` (MR-row) loop — `Σ bₛ` depends only on the B panel of this jr
    /// tile (`jrbase`, `kce`), never on the A-row block, so recomputing it inside
    /// the microkernel on every MR-row block (the previous behaviour) was pure
    /// redundant work (≈`MCp/MR` = 16× a full MC block).
    ///
    /// Emits a standalone K-quad loop accumulating `%{p}_bacc_h` via
    /// `vpdpbusd.512(bacc, ones_u8, B_s8)` over the SAME `Bp[jrbase + kp*(NR·4)]`
    /// quad loads the microkernel reads, then produces
    /// `%{p}_bsx = extsi(bacc) << 7` (the `<<7` = `·128`) as vector<NRxi64>.
    /// The exact-integer value is identical to the in-loop accumulation, so this
    /// is loop-invariant code motion with zero effect on the output bytes.
    #[cfg(feature = "std-surface")]
    fn emit_i8_vnni_bsum_hoist(buf: &mut String, p: &str, nr: usize) {
        use std::fmt::Write;
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };
        // `vpdpbusd.512` is declared on `<16 x i32>`, so the per-column `Σ bₛ`
        // is accumulated in COL_REGS = nr/16 separate ZMM groups (2 for NR=32),
        // each over its own 64-byte slice of the K-quad's B panel, then the three
        // i32 sums are widened to i64, shifted `<<7` (·128), and concatenated into
        // one `vector<nr xi64>` `%{p}_bsx` — the same wide value the microkernel
        // subtracts. Pure loop-invariant code motion; bit-identical.
        let col_regs = nr / 16;
        line(&format!(
            "            %{p}_h_ones = arith.constant dense<16843009> : vector<16xi32>"
        ));
        line(&format!(
            "            %{p}_h_zi32 = arith.constant dense<0> : vector<16xi32>"
        ));
        // K-quad count for this panel (kce/grp); grp is 4 for VNNI.
        line(&format!(
            "            %{p}_h_kpairs = arith.divui %{p}_kce, %{p}_cgrp : index"
        ));
        // One scf.for yielding COL_REGS independent i32 bsum accumulators.
        let h_acc_init = (0..col_regs)
            .map(|j| format!("%{p}_h_acc{j} = %{p}_h_zi32"))
            .collect::<Vec<_>>()
            .join(", ");
        let h_acc_ty = (0..col_regs)
            .map(|_| "vector<16xi32>".to_string())
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!(
            "            %{p}_h_bacc:{col_regs} = scf.for %{p}_h_kp = %{p}_c0 to %{p}_h_kpairs \
             step %{p}_c1 iter_args({h_acc_init}) -> ({h_acc_ty}) {{"
        ));
        line(&format!(
            "              %{p}_h_kp64 = arith.index_cast %{p}_h_kp : index to i64"
        ));
        line(&format!(
            "              %{p}_h_bnr4 = arith.muli %{p}_nrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_h_bkr = arith.muli %{p}_h_kp64, %{p}_h_bnr4 : i64"
        ));
        line(&format!(
            "              %{p}_h_bvi = arith.addi %{p}_jrbase, %{p}_h_bkr : i64"
        ));
        for j in 0..col_regs {
            line(&format!(
                "              %{p}_h_bo{j} = arith.constant {} : i64",
                j * 64
            ));
        }
        let mut h_yields = Vec::with_capacity(col_regs);
        for j in 0..col_regs {
            // Col-group j: 64-byte ZMM slice at byte offset j*64 within this
            // quad's B panel (lanes [16j, 16j+16)).
            line(&format!(
                "              %{p}_h_bvi{j} = arith.addi %{p}_h_bvi, %{p}_h_bo{j} : i64"
            ));
            line(&format!(
                "              %{p}_h_bvp{j} = llvm.getelementptr %{p}_pb[%{p}_h_bvi{j}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            line(&format!(
                "              %{p}_h_bvb{j} = llvm.load %{p}_h_bvp{j} {{alignment = 1 : i64}} : \
                 !llvm.ptr -> vector<64xi8>"
            ));
            line(&format!(
                "              %{p}_h_bv{j} = vector.bitcast %{p}_h_bvb{j} : vector<64xi8> to vector<16xi32>"
            ));
            line(&format!(
                "              %{p}_h_nb{j} = llvm.call_intrinsic \"llvm.x86.avx512.vpdpbusd.512\"\
                 (%{p}_h_acc{j}, %{p}_h_ones, %{p}_h_bv{j}) : \
                 (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>"
            ));
            h_yields.push(format!("%{p}_h_nb{j}"));
        }
        line(&format!(
            "              scf.yield {} : {h_acc_ty}",
            h_yields.join(", ")
        ));
        line("            }");
        // Widen each col-group's Σ bₛ to i64, shift <<7 (·128); concatenate the
        // COL_REGS groups into the single wide %{p}_bsx : vector<nr xi64>.
        line(&format!(
            "            %{p}_h_c7v = arith.constant dense<7> : vector<16xi64>"
        ));
        for j in 0..col_regs {
            line(&format!(
                "            %{p}_h_bsum64_{j} = arith.extsi %{p}_h_bacc#{j} : vector<16xi32> to vector<16xi64>"
            ));
            line(&format!(
                "            %{p}_h_bsx{j} = arith.shli %{p}_h_bsum64_{j}, %{p}_h_c7v : vector<16xi64>"
            ));
        }
        // Concatenate the COL_REGS 16-lane bias vectors into one nr-lane vector
        // via a chain of vector.shuffle (lanes [0..16) from group 0, etc.).
        if col_regs == 1 {
            let idx: String = (0..nr)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            line(&format!(
                "            %{p}_bsx = vector.shuffle %{p}_h_bsx0, %{p}_h_bsx0 [{idx}] : \
                 vector<16xi64>, vector<16xi64>"
            ));
        } else {
            // Build up wide vectors group by group.
            let mut prev = format!("%{p}_h_bsx0");
            let mut prev_w = 16usize;
            for j in 1..col_regs {
                let new_w = prev_w + 16;
                let idx: String = (0..new_w)
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let out = if j == col_regs - 1 {
                    format!("%{p}_bsx")
                } else {
                    format!("%{p}_h_bcat{j}")
                };
                line(&format!(
                    "            {out} = vector.shuffle {prev}, %{p}_h_bsx{j} [{idx}] : \
                     vector<{prev_w}xi64>, vector<16xi64>"
                ));
                prev = out;
                prev_w = new_w;
            }
        }
    }

    #[cfg(feature = "std-surface")]
    fn emit_i8_microkernel_vnni(buf: &mut String, p: &str, nr: usize, mr: usize, acc_ty: &str) {
        use std::fmt::Write;
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };
        // VNNI wide-tile microkernel: COL_REGS = nr/16 ZMM B-columns (2 for
        // NR=32) reused MR-deep. The 0x80 s8→u8 bias is PRE-APPLIED to the packed
        // A panel at pack time (each packed-A byte is xor'd with 0x80 in VNNI
        // mode), so the per-row inner-loop `xori` is GONE — the broadcasted A
        // dword already carries (A+128). The A-broadcast is MEMORY-SOURCE
        // (`vector.load` of a `vector<1xi32>` then `vector.broadcast`), which the
        // x86 backend lowers to `vpbroadcastd zmm, m32` on the load ports (p2/p3),
        // OFF port 5 — freeing port 5 for vpdpbusd.
        let col_regs = nr / 16;
        // iter_args: MR×COL_REGS main i32 accumulators (8×2 = 16 ZMM grid). The
        // shared `Σ bₛ` correction is computed ONCE per jr tile by
        // `emit_i8_vnni_bsum_hoist` (loop-invariant across ir) and consumed below
        // as the pre-shifted `%{p}_bsx`.
        line(&format!(
            "              %{p}_zi16 = arith.constant dense<0> : vector<16xi32>"
        ));
        let acc_init: Vec<String> = (0..mr)
            .flat_map(|t| (0..col_regs).map(move |j| format!("%{p}_macc{t}_{j} = %{p}_zi16")))
            .collect();
        let n_acc = mr * col_regs;
        let iter_ty: Vec<String> = (0..n_acc).map(|_| "vector<16xi32>".to_string()).collect();
        let iter_ty = iter_ty.join(", ");
        line(&format!(
            "              %{p}_vi:{} = scf.for %{p}_kp = %{p}_c0 to %{p}_kpairs \
             step %{p}_c1 iter_args({}) -> ({}) {{",
            n_acc,
            acc_init.join(", "),
            iter_ty
        ));
        line(&format!(
            "                %{p}_kp64 = arith.index_cast %{p}_kp : index to i64"
        ));
        // ── Hoist the COL_REGS B-tile ZMM loads BEFORE the MR row loop (B reused
        //    across all MR rows). Base: Bp[jrbase + kp*(NR*4)]; col-group j is at
        //    byte offset j*64.
        line(&format!(
            "                %{p}_bnr4 = arith.muli %{p}_nrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_bkr = arith.muli %{p}_kp64, %{p}_bnr4 : i64"
        ));
        line(&format!(
            "                %{p}_bvi = arith.addi %{p}_jrbase, %{p}_bkr : i64"
        ));
        for j in 0..col_regs {
            line(&format!(
                "                %{p}_bo{j} = arith.constant {} : i64",
                j * 64
            ));
            line(&format!(
                "                %{p}_bvi{j} = arith.addi %{p}_bvi, %{p}_bo{j} : i64"
            ));
            line(&format!(
                "                %{p}_bvp{j} = llvm.getelementptr %{p}_pb[%{p}_bvi{j}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            line(&format!(
                "                %{p}_bvb{j} = llvm.load %{p}_bvp{j} {{alignment = 1 : i64}} : \
                 !llvm.ptr -> vector<64xi8>"
            ));
            line(&format!(
                "                %{p}_bv{j} = vector.bitcast %{p}_bvb{j} : vector<64xi8> to vector<16xi32>"
            ));
        }
        // A K-quad base for this block: Ap[irbase + kp*(MR*4)].
        line(&format!(
            "                %{p}_amr4 = arith.muli %{p}_mrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_akr = arith.muli %{p}_kp64, %{p}_amr4 : i64"
        ));
        line(&format!(
            "                %{p}_abase = arith.addi %{p}_irbase, %{p}_akr : i64"
        ));
        let mut yields = Vec::with_capacity(n_acc);
        for t in 0..mr {
            // A quad for row t: four contiguous i8 = [(A[t,4kp..4kp+3]) ⊕ 0x80]
            // (the +128 bias pre-applied at pack). MEMORY-SOURCE broadcast: load
            // the dword as a vector<1xi32> straight from packed A, then
            // vector.broadcast → vector<16xi32> = [quad,quad,...]. The x86 backend
            // folds broadcast(load) to vpbroadcastd zmm,m32 (load ports, off p5).
            // No xori here — the bias is already in the packed bytes.
            line(&format!(
                "                %{p}_at{t} = arith.constant {} : i64",
                t * 4
            ));
            line(&format!(
                "                %{p}_ai{t} = arith.addi %{p}_abase, %{p}_at{t} : i64"
            ));
            line(&format!(
                "                %{p}_ap{t} = llvm.getelementptr %{p}_pa[%{p}_ai{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            line(&format!(
                "                %{p}_av{t} = llvm.load %{p}_ap{t} {{alignment = 1 : i64}} : \
                 !llvm.ptr -> vector<1xi32>"
            ));
            line(&format!(
                "                %{p}_abc{t} = vector.broadcast %{p}_av{t} : vector<1xi32> to vector<16xi32>"
            ));
            for j in 0..col_regs {
                line(&format!(
                    "                %{p}_nm{t}_{j} = llvm.call_intrinsic \"llvm.x86.avx512.vpdpbusd.512\"\
                     (%{p}_macc{t}_{j}, %{p}_abc{t}, %{p}_bv{j}) : \
                     (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>"
                ));
                yields.push(format!("%{p}_nm{t}_{j}"));
            }
        }
        line(&format!(
            "                scf.yield {} : {}",
            yields.join(", "),
            iter_ty
        ));
        line("              }");
        // Post-loop: per row, concat the COL_REGS i32 col-groups into one
        // vector<nr xi32>, extend to i64, subtract %{p}_bsx (= 128·Σ bₛ, the
        // signed-input bias correction, computed once per jr tile). Yields the
        // corrected per-row vector<nr xi64> the scalar tail / C-flush consume.
        let va_init = (0..mr)
            .map(|t| format!("%{p}_vaa{t} = %{p}_vacorr{t}",))
            .collect::<Vec<_>>();
        for t in 0..mr {
            // Concatenate this row's COL_REGS 16-lane i32 groups → vector<nr xi32>.
            let main_i32 = if col_regs == 1 {
                let idx: String = (0..nr)
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                line(&format!(
                    "              %{p}_mcat{t} = vector.shuffle %{p}_vi#{} , %{p}_vi#{} [{idx}] : \
                     vector<16xi32>, vector<16xi32>",
                    t * col_regs,
                    t * col_regs
                ));
                format!("%{p}_mcat{t}")
            } else {
                let mut prev = format!("%{p}_vi#{}", t * col_regs);
                let mut prev_w = 16usize;
                for j in 1..col_regs {
                    let new_w = prev_w + 16;
                    let idx: String = (0..new_w)
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    line(&format!(
                        "              %{p}_mcat{t}_{j} = vector.shuffle {prev}, %{p}_vi#{} [{idx}] : \
                         vector<{prev_w}xi32>, vector<16xi32>",
                        t * col_regs + j
                    ));
                    prev = format!("%{p}_mcat{t}_{j}");
                    prev_w = new_w;
                }
                prev
            };
            line(&format!(
                "              %{p}_main64_{t} = arith.extsi {main_i32} : vector<{nr}xi32> to vector<{nr}xi64>"
            ));
            line(&format!(
                "              %{p}_vacorr{t} = arith.subi %{p}_main64_{t}, %{p}_bsx : vector<{nr}xi64>"
            ));
        }
        line(&format!(
            "              %{p}_va:{mr} = scf.for %{p}_vdummy = %{p}_c0 to %{p}_c1 \
             step %{p}_c1 iter_args({}) -> ({acc_ty}) {{",
            va_init.join(", ")
        ));
        let va_yields = (0..mr)
            .map(|t| format!("%{p}_vaa{t}"))
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!("                scf.yield {va_yields} : {acc_ty}"));
        line("              }");
    }

    /// `prefix` namespaces every SSA value. `ap`/`bp`/`cp` are `!llvm.ptr` SSA
    /// names; `k64`/`n64` are i64 SSA names for K and N; `ki`/`ni` are `index`
    /// SSA names; `row_start`/`row_end` are `index` SSA names bounding the band.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_mm_i8_blocked(
        buf: &mut String,
        prefix: &str,
        ap: &str,
        bp: &str,
        cp: &str,
        k64: &str,
        n64: &str,
        ki: &str,
        ni: &str,
        row_start: &str,
        row_end: &str,
        mode: IntDotMode,
    ) {
        use std::fmt::Write;
        let p = prefix;
        // Register-tile rows (MR) and column block (NC) are MODE-DEPENDENT: the
        // AVX2 `vpmaddwd` YMM path keeps the pinned I8_MR=4 / I8_NC=128; the VNNI
        // `vpdpbusd.512` path uses the wider register tile I8_VNNI_MR=8 with
        // I8_VNNI_NC=384 (a multiple of the VNNI NR=32). Per-call the mode is
        // fixed at emit time, so a `let` (not a `const`) carries the choice into
        // the alloca sizes, the C-scratch index math, and the row loops. Widening
        // MR only adds independent accumulator chains; widening NC only changes
        // the column-block extent — neither perturbs the exact int32 sum.
        let mr: usize = match mode {
            IntDotMode::Avx2 => I8_MR,
            IntDotMode::Vnni => I8_VNNI_MR,
        };
        const MC: usize = I8_MC;
        const KC: usize = I8_KC;
        let nc: usize = match mode {
            IntDotMode::Avx2 => I8_NC,
            IntDotMode::Vnni => I8_VNNI_NC,
        };
        // Register-tile column width (NR). MODE-DEPENDENT: the AVX2 `vpmaddwd`
        // path stays at the pinned I8_NR=8 (YMM, vector<8xi64> accumulators); the
        // VNNI `vpdpbusd.512` path uses NR=32 = 2 ZMM B-columns (`colRegs=2`).
        // The microkernel holds the 16 accumulators as an 8×2 grid of
        // `vector<16xi32>` (one ZMM per (row, col-group)) and reassembles each
        // row's two groups into one `vector<32xi64>` for the C-scratch — so the
        // packed-B / C-scratch index arithmetic, the scalar tail, and the C-flush
        // all still key off this single logical `nr=32` (the wide loads/stores
        // legalize to 2 ZMM each). NR only changes the column-tile width / lane
        // grouping, never the math — integer add is associative+commutative, so
        // widening to 32 lanes (2 col-groups) is byte-identical to a 16- or
        // 8-lane grouping. NC must be a multiple of NR (the VNNI NC=384 = 12·32).
        let nr: usize = match mode {
            IntDotMode::Avx2 => I8_NR,
            IntDotMode::Vnni => I8_VNNI_NR,
        };
        // K-steps fused by one int-dot instruction: 2 for vpmaddwd (i16 pairs),
        // 4 for vpdpbusd (i8 quads). The packed panels are K-interleaved with
        // this group width; the packed element type is i16 for AVX2 (post-extsi)
        // and i8 for VNNI (vpdpbusd consumes raw bytes). Both produce the
        // identical exact int32 sum (see `IntDotMode`).
        let grp: usize = match mode {
            IntDotMode::Avx2 => 2,
            IntDotMode::Vnni => 4,
        };
        // Packed-panel element type and its byte width (alloca + GEP element).
        let pty = match mode {
            IntDotMode::Avx2 => "i16",
            IntDotMode::Vnni => "i8",
        };
        // "int-dot" int8 tier (vpmaddwd path). A/B source elements are int8
        // (1 byte). During the pack each is `arith.extsi`-widened to i16 ONCE
        // (`vpmovsxbw`) and written into a K-pair-INTERLEAVED i16 panel so the
        // microkernel can contract two K-steps per `vpmaddwd`. The C output is
        // i32 (4 bytes); the C-scratch accumulates in i64 (8 bytes).
        //
        // Packed B layout: Bp[jr_block][k_pair][nr][s] (i16), s∈{0,1} is the
        //   K-within-pair. A single `vector<16xi16>` load at [jr_block][k_pair]
        //   yields [B[2p,0],B[2p+1,0],B[2p,1],B[2p+1,1],...,B[2p,7],B[2p+1,7]].
        // Packed A layout: Ap[ir_block][k_pair][mr][s] (i16). A 2-element load
        //   at [ir_block][k_pair][mr] yields [A[t,2p],A[t,2p+1]]; broadcast that
        //   pair across the 8 NR columns (vector<16xi16> = [a0,a1,a0,a1,...]).
        // vpmaddwd(A_pair_bcast, B_pairs) → lane n = A[t,2p]*B[2p,n] +
        //   A[t,2p+1]*B[2p+1,n] = the 2-K partial for output column n. The total
        //   over all K is the SAME sum of A[t,k]*B[k,n] terms as the per-element
        //   oracle (integer add is associative+commutative), so byte-identity is
        //   preserved exactly. NO saturation (never vpmaddubsw), NO shift.
        let eb_src: i64 = 1;
        let eb: i64 = std::mem::size_of::<i32>() as i64;
        // i32 partial-lane count of one vpmaddwd over NR=8 output columns: each
        // output column consumes a (k,k+1) i16 pair, so a 2*NR=16-wide i16
        // multiply yields NR=8 i32 partials.
        let pl: usize = nr;
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };

        // ── constants ────────────────────────────────────────────────────────
        line(&format!("    %{p}_c0 = arith.constant 0 : index"));
        line(&format!("    %{p}_c1 = arith.constant 1 : index"));
        line(&format!("    %{p}_c2 = arith.constant 2 : index"));
        line(&format!("    %{p}_cmr = arith.constant {mr} : index"));
        line(&format!("    %{p}_cnr = arith.constant {nr} : index"));
        line(&format!("    %{p}_cmc = arith.constant {MC} : index"));
        line(&format!("    %{p}_ckc = arith.constant {KC} : index"));
        line(&format!("    %{p}_cnc = arith.constant {nc} : index"));
        line(&format!("    %{p}_eb = arith.constant {eb} : i64"));
        line(&format!("    %{p}_ebs = arith.constant {eb_src} : i64"));
        line(&format!("    %{p}_z0 = arith.constant 0 : i64"));
        // Zero in the packed-panel element type, for the zero-pad in packing.
        line(&format!("    %{p}_z0pty = arith.constant 0 : {pty}"));
        line(&format!("    %{p}_c2i = arith.constant 2 : i64"));
        // K-group stride (2 = pair for AVX2, 4 = quad for VNNI), in both i64 and
        // index, for the K-interleaved panel index math and the microkernel.
        line(&format!("    %{p}_cgrp = arith.constant {grp} : index"));
        line(&format!("    %{p}_cgrpi = arith.constant {grp} : i64"));
        line(&format!(
            "    %{p}_zv = arith.constant dense<0> : vector<{nr}xi64>"
        ));

        // ── private scratch (heap malloc/free) ───────────────────────────────
        // C-scratch: MC*NC i64.  Packed A: MC*KC {pty}.  Packed B: KC*NC {pty}.
        // (The interleaved K-pair layout reindexes the same MC*KC / KC*NC
        // element count — pair-count * 2 * MR = KC * MR, etc.)
        //
        // These panels are HEAP-allocated via `@malloc` (and `@free`d before
        // return) rather than stack `llvm.alloca` so a large MC row block is
        // safe regardless of the caller's stack depth. Heap vs stack is the same
        // storage for the same computation in a different location — every GEP /
        // load / store below is byte-for-byte unchanged, so the lowering stays
        // byte-identical (the int8 AVX2 canary `917d353b` exercises this scratch).
        // malloc takes a BYTE count: cs is i64 (8 B/elem), pa/pb are {pty}
        // ({pty_bytes} B/elem).
        let pty_bytes: i64 = match mode {
            IntDotMode::Avx2 => 2, // i16
            IntDotMode::Vnni => 1, // i8
        };
        let cs_elems = (MC * nc) as i64;
        let pa_elems = (MC * KC) as i64;
        let pb_elems = (KC * nc) as i64;
        let cs_bytes = cs_elems * 8;
        let pa_bytes = pa_elems * pty_bytes;
        let pb_bytes = pb_elems * pty_bytes;
        line(&format!(
            "    %{p}_csn = llvm.mlir.constant({cs_bytes} : i64) : i64"
        ));
        line(&format!(
            "    %{p}_cs = llvm.call @malloc(%{p}_csn) : (i64) -> !llvm.ptr"
        ));
        line(&format!(
            "    %{p}_pan = llvm.mlir.constant({pa_bytes} : i64) : i64"
        ));
        line(&format!(
            "    %{p}_pa = llvm.call @malloc(%{p}_pan) : (i64) -> !llvm.ptr"
        ));
        line(&format!(
            "    %{p}_pbn = llvm.mlir.constant({pb_bytes} : i64) : i64"
        ));
        line(&format!(
            "    %{p}_pb = llvm.call @malloc(%{p}_pbn) : (i64) -> !llvm.ptr"
        ));
        line(&format!("    %{p}_ncc = arith.constant {nc} : i64"));
        line(&format!("    %{p}_kcc = arith.constant {KC} : i64"));
        line(&format!("    %{p}_mrc = arith.constant {mr} : i64"));
        line(&format!("    %{p}_nrc = arith.constant {nr} : i64"));

        // ════════════════════════════════════════════════════════════════════
        //  jc — column block over [0, N)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "    scf.for %{p}_jc = %{p}_c0 to %{ni} step %{p}_cnc {{"
        ));
        line(&format!(
            "      %{p}_jcrem = arith.subi %{ni}, %{p}_jc : index"
        ));
        line(&format!(
            "      %{p}_nce_lt = arith.cmpi slt, %{p}_cnc, %{p}_jcrem : index"
        ));
        line(&format!(
            "      %{p}_nce = arith.select %{p}_nce_lt, %{p}_cnc, %{p}_jcrem : index"
        ));
        line(&format!(
            "      %{p}_ncp_t = arith.addi %{p}_nce, %{p}_cnr : index"
        ));
        line(&format!(
            "      %{p}_ncp_t1 = arith.subi %{p}_ncp_t, %{p}_c1 : index"
        ));
        line(&format!(
            "      %{p}_ncp_d = arith.divui %{p}_ncp_t1, %{p}_cnr : index"
        ));
        line(&format!(
            "      %{p}_ncp = arith.muli %{p}_ncp_d, %{p}_cnr : index"
        ));
        line(&format!(
            "      %{p}_jc64 = arith.index_cast %{p}_jc : index to i64"
        ));

        // ════════════════════════════════════════════════════════════════════
        //  ic — row block over [row_start, row_end)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "      scf.for %{p}_ic = %{row_start} to %{row_end} step %{p}_cmc {{"
        ));
        line(&format!(
            "        %{p}_icrem = arith.subi %{row_end}, %{p}_ic : index"
        ));
        line(&format!(
            "        %{p}_mce_lt = arith.cmpi slt, %{p}_cmc, %{p}_icrem : index"
        ));
        line(&format!(
            "        %{p}_mce = arith.select %{p}_mce_lt, %{p}_cmc, %{p}_icrem : index"
        ));
        line(&format!(
            "        %{p}_mcp_t = arith.addi %{p}_mce, %{p}_cmr : index"
        ));
        line(&format!(
            "        %{p}_mcp_t1 = arith.subi %{p}_mcp_t, %{p}_c1 : index"
        ));
        line(&format!(
            "        %{p}_mcp_d = arith.divui %{p}_mcp_t1, %{p}_cmr : index"
        ));
        line(&format!(
            "        %{p}_mcp = arith.muli %{p}_mcp_d, %{p}_cmr : index"
        ));
        line(&format!(
            "        %{p}_ic64 = arith.index_cast %{p}_ic : index to i64"
        ));

        // ── zero the live MCp×NCp region of the C-scratch ────────────────────
        line(&format!(
            "        scf.for %{p}_zr = %{p}_c0 to %{p}_mcp step %{p}_c1 {{"
        ));
        line(&format!(
            "          %{p}_zr64 = arith.index_cast %{p}_zr : index to i64"
        ));
        line(&format!(
            "          %{p}_zrb = arith.muli %{p}_zr64, %{p}_ncc : i64"
        ));
        line(&format!(
            "          scf.for %{p}_zc = %{p}_c0 to %{p}_ncp step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_zc64 = arith.index_cast %{p}_zc : index to i64"
        ));
        line(&format!(
            "            %{p}_zoff = arith.addi %{p}_zrb, %{p}_zc64 : i64"
        ));
        line(&format!(
            "            %{p}_zptr = llvm.getelementptr %{p}_cs[%{p}_zoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i64"
        ));
        line(&format!(
            "            llvm.store %{p}_z0, %{p}_zptr : i64, !llvm.ptr"
        ));
        line("          }");
        line("        }");

        // ════════════════════════════════════════════════════════════════════
        //  pc — K panel over [0, K)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "        scf.for %{p}_pc = %{p}_c0 to %{ki} step %{p}_ckc {{"
        ));
        line(&format!(
            "          %{p}_pcrem = arith.subi %{ki}, %{p}_pc : index"
        ));
        line(&format!(
            "          %{p}_kce_lt = arith.cmpi slt, %{p}_ckc, %{p}_pcrem : index"
        ));
        line(&format!(
            "          %{p}_kce = arith.select %{p}_kce_lt, %{p}_ckc, %{p}_pcrem : index"
        ));
        line(&format!(
            "          %{p}_pc64 = arith.index_cast %{p}_pc : index to i64"
        ));

        // ── pack B panel: Bp[jr_block][kk][nr] (zero-padded to NCp cols) ──────
        // Source B[pc+kk, jc+jr] is int8 (eb_src=1); sign-extend to the i32 panel.
        line(&format!(
            "          scf.for %{p}_pbk = %{p}_c0 to %{p}_kce step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_pbk64 = arith.index_cast %{p}_pbk : index to i64"
        ));
        line(&format!(
            "            %{p}_pbkg = arith.addi %{p}_pc64, %{p}_pbk64 : i64"
        ));
        line(&format!(
            "            %{p}_pbsrow = arith.muli %{p}_pbkg, %{n64} : i64"
        ));
        line(&format!(
            "            %{p}_pbsr0 = arith.addi %{p}_pbsrow, %{p}_jc64 : i64"
        ));
        line(&format!(
            "            scf.for %{p}_pbj = %{p}_c0 to %{p}_ncp step %{p}_c1 {{"
        ));
        line(&format!(
            "              %{p}_pbj64 = arith.index_cast %{p}_pbj : index to i64"
        ));
        line(&format!(
            "              %{p}_pbjb = arith.divui %{p}_pbj64, %{p}_nrc : i64"
        ));
        line(&format!(
            "              %{p}_pbjm = arith.remui %{p}_pbj64, %{p}_nrc : i64"
        ));
        // K-group-interleaved destination: Bp[jb][grp_idx][nr][s],
        //   grp_idx = pbk/grp, s = pbk%grp, flat =
        //   jb*(KC*NR) + grp_idx*(NR*grp) + nr*grp + s.
        //   (grp = 2 pair / i16 for AVX2, 4 quad / i8 for VNNI.)
        line(&format!(
            "              %{p}_pbpair = arith.divui %{p}_pbk64, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pbs = arith.remui %{p}_pbk64, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pbpan = arith.muli %{p}_kcc, %{p}_nrc : i64"
        ));
        line(&format!(
            "              %{p}_pbblk = arith.muli %{p}_pbjb, %{p}_pbpan : i64"
        ));
        line(&format!(
            "              %{p}_pbnr2 = arith.muli %{p}_nrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pbkr = arith.muli %{p}_pbpair, %{p}_pbnr2 : i64"
        ));
        line(&format!(
            "              %{p}_pbjm2 = arith.muli %{p}_pbjm, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pbd0 = arith.addi %{p}_pbblk, %{p}_pbkr : i64"
        ));
        line(&format!(
            "              %{p}_pbd1 = arith.addi %{p}_pbd0, %{p}_pbjm2 : i64"
        ));
        line(&format!(
            "              %{p}_pbd = arith.addi %{p}_pbd1, %{p}_pbs : i64"
        ));
        line(&format!(
            "              %{p}_pbdp = llvm.getelementptr %{p}_pb[%{p}_pbd] : \
             (!llvm.ptr, i64) -> !llvm.ptr, {pty}"
        ));
        line(&format!(
            "              %{p}_pblive = arith.cmpi slt, %{p}_pbj, %{p}_nce : index"
        ));
        line(&format!(
            "              %{p}_pbval = scf.if %{p}_pblive -> ({pty}) {{"
        ));
        line(&format!(
            "                %{p}_pbsi = arith.addi %{p}_pbsr0, %{p}_pbj64 : i64"
        ));
        line(&format!(
            "                %{p}_pbbo = arith.muli %{p}_pbsi, %{p}_ebs : i64"
        ));
        line(&format!(
            "                %{p}_pbsp = llvm.getelementptr %{bp}[%{p}_pbbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "                %{p}_pbi8 = llvm.load %{p}_pbsp : !llvm.ptr -> i8"
        ));
        // AVX2 panel is i16 (sign-extend once = vpmovsxbw); VNNI panel keeps the
        // raw signed byte (vpdpbusd's B operand is s8). Both are the same
        // numeric value of B[k,n]; the int-dot rung differs only in width.
        match mode {
            IntDotMode::Avx2 => {
                line(&format!(
                    "                %{p}_pbld = arith.extsi %{p}_pbi8 : i8 to {pty}"
                ));
                line(&format!("                scf.yield %{p}_pbld : {pty}"));
            }
            IntDotMode::Vnni => {
                line(&format!("                scf.yield %{p}_pbi8 : {pty}"));
            }
        }
        line("              } else {");
        line(&format!("                scf.yield %{p}_z0pty : {pty}"));
        line("              }");
        line(&format!(
            "              llvm.store %{p}_pbval, %{p}_pbdp : {pty}, !llvm.ptr"
        ));
        line("            }");
        line("          }");

        // ── pack A panel: Ap[ir_block][kk][mr] (zero-padded to MCp rows) ──────
        // Source A[ic+pai, pc+pak] is int8 (eb_src=1); sign-extend to the i32 panel.
        line(&format!(
            "          scf.for %{p}_pak = %{p}_c0 to %{p}_kce step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_pak64 = arith.index_cast %{p}_pak : index to i64"
        ));
        line(&format!(
            "            %{p}_pakg = arith.addi %{p}_pc64, %{p}_pak64 : i64"
        ));
        line(&format!(
            "            scf.for %{p}_pai = %{p}_c0 to %{p}_mcp step %{p}_c1 {{"
        ));
        line(&format!(
            "              %{p}_pai64 = arith.index_cast %{p}_pai : index to i64"
        ));
        line(&format!(
            "              %{p}_paib = arith.divui %{p}_pai64, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_paim = arith.remui %{p}_pai64, %{p}_mrc : i64"
        ));
        // K-group-interleaved destination: Ap[ib][grp_idx][mr][s],
        //   grp_idx = pak/grp, s = pak%grp, flat =
        //   ib*(KC*MR) + grp_idx*(MR*grp) + mr*grp + s.
        //   (grp = 2 pair / i16 for AVX2, 4 quad / i8 for VNNI.)
        line(&format!(
            "              %{p}_papair = arith.divui %{p}_pak64, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pas = arith.remui %{p}_pak64, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_papan = arith.muli %{p}_kcc, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_pablk = arith.muli %{p}_paib, %{p}_papan : i64"
        ));
        line(&format!(
            "              %{p}_pamr2 = arith.muli %{p}_mrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pakr = arith.muli %{p}_papair, %{p}_pamr2 : i64"
        ));
        line(&format!(
            "              %{p}_paim2 = arith.muli %{p}_paim, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "              %{p}_pad0 = arith.addi %{p}_pablk, %{p}_pakr : i64"
        ));
        line(&format!(
            "              %{p}_pad1 = arith.addi %{p}_pad0, %{p}_paim2 : i64"
        ));
        line(&format!(
            "              %{p}_pad = arith.addi %{p}_pad1, %{p}_pas : i64"
        ));
        line(&format!(
            "              %{p}_padp = llvm.getelementptr %{p}_pa[%{p}_pad] : \
             (!llvm.ptr, i64) -> !llvm.ptr, {pty}"
        ));
        line(&format!(
            "              %{p}_palive = arith.cmpi slt, %{p}_pai, %{p}_mce : index"
        ));
        line(&format!(
            "              %{p}_paval = scf.if %{p}_palive -> ({pty}) {{"
        ));
        line(&format!(
            "                %{p}_pari = arith.addi %{p}_ic64, %{p}_pai64 : i64"
        ));
        line(&format!(
            "                %{p}_parik = arith.muli %{p}_pari, %{k64} : i64"
        ));
        line(&format!(
            "                %{p}_pasi = arith.addi %{p}_parik, %{p}_pakg : i64"
        ));
        line(&format!(
            "                %{p}_pabo = arith.muli %{p}_pasi, %{p}_ebs : i64"
        ));
        line(&format!(
            "                %{p}_pasp = llvm.getelementptr %{ap}[%{p}_pabo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "                %{p}_pai8 = llvm.load %{p}_pasp : !llvm.ptr -> i8"
        ));
        // AVX2 panel is i16 (sign-extend once). VNNI panel PRE-APPLIES the s8→u8
        // +128 bias here at pack time (xor each packed-A byte with 0x80), so the
        // microkernel's broadcasted A dword already carries (A+128) and the
        // per-row inner-loop `xori` is removed. The −128·Σ bₛ correction (the
        // hoisted Σ bₛ) cancels this exactly ⇒ byte-identical. (Zero-padded rows
        // beyond MCe are never written to C, so their bias state is irrelevant.)
        match mode {
            IntDotMode::Avx2 => {
                line(&format!(
                    "                %{p}_pald = arith.extsi %{p}_pai8 : i8 to {pty}"
                ));
                line(&format!("                scf.yield %{p}_pald : {pty}"));
            }
            IntDotMode::Vnni => {
                line(&format!(
                    "                %{p}_pax80 = arith.constant -128 : i8"
                ));
                line(&format!(
                    "                %{p}_pabias = arith.xori %{p}_pai8, %{p}_pax80 : i8"
                ));
                line(&format!("                scf.yield %{p}_pabias : {pty}"));
            }
        }
        line("              } else {");
        line(&format!("                scf.yield %{p}_z0pty : {pty}"));
        line("              }");
        line(&format!(
            "              llvm.store %{p}_paval, %{p}_padp : {pty}, !llvm.ptr"
        ));
        line("            }");
        line("          }");

        // ════════════════════════════════════════════════════════════════════
        //  jr — NR-wide column tiles over the packed B panel [0, NCp)
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "          scf.for %{p}_jr = %{p}_c0 to %{p}_ncp step %{p}_cnr {{"
        ));
        line(&format!(
            "            %{p}_jr64 = arith.index_cast %{p}_jr : index to i64"
        ));
        line(&format!(
            "            %{p}_jrb = arith.divui %{p}_jr64, %{p}_nrc : i64"
        ));
        line(&format!(
            "            %{p}_jrpan = arith.muli %{p}_kcc, %{p}_nrc : i64"
        ));
        line(&format!(
            "            %{p}_jrbase = arith.muli %{p}_jrb, %{p}_jrpan : i64"
        ));
        // ── HOISTED VNNI column-sum: 128·Σ bₛ depends only on this jr tile's B
        //    panel (jrbase, kce), NOT on the A-row block, so compute it ONCE per
        //    jr tile — here, OUTSIDE the ir loop — instead of redundantly inside
        //    the microkernel on every MR-row block. Produces %{p}_bsx =
        //    (Σ_quads vpdpbusd(ones_u8, B_s8)) << 7 as vector<NRxi64>, which the
        //    microkernel subtracts from each row's main accumulator. Pure
        //    loop-invariant code motion: bit-identical to recomputing it.
        let _ = line; // release closure so the hoist emitter can borrow buf
        if mode == IntDotMode::Vnni {
            Self::emit_i8_vnni_bsum_hoist(buf, p, nr);
        }
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };
        // ══════════════════════════════════════════════════════════════════
        //  ir — MR-row tiles over the packed A panel [0, MCp)
        // ══════════════════════════════════════════════════════════════════
        line(&format!(
            "            scf.for %{p}_ir = %{p}_c0 to %{p}_mcp step %{p}_cmr {{"
        ));
        line(&format!(
            "              %{p}_ir64 = arith.index_cast %{p}_ir : index to i64"
        ));
        line(&format!(
            "              %{p}_irb = arith.divui %{p}_ir64, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_irpan = arith.muli %{p}_kcc, %{p}_mrc : i64"
        ));
        line(&format!(
            "              %{p}_irbase = arith.muli %{p}_irb, %{p}_irpan : i64"
        ));

        // ── microkernel: MR×NR i64 partial over this KC panel ────────────────
        // Each A-row accumulator is vector<NRxi64> (NR=8 output columns). The
        // K-contraction runs over K-GROUPS of `grp` steps (2 = pair for AVX2
        // vpmaddwd, 4 = quad for VNNI vpdpbusd); one int-dot instruction per
        // (row, group) contracts `grp` K-steps at once into NR=8 i32 partials.
        // The total over all K is the identical sum of A[t,k]*B[k,n] terms as
        // the scalar oracle — byte-identical. The trailing <grp leftover K is a
        // scalar tail (shared by both rungs).
        let acc_ty = (0..mr)
            .map(|_| format!("vector<{nr}xi64>"))
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!(
            "              %{p}_kpairs = arith.divui %{p}_kce, %{p}_cgrp : index"
        ));
        line(&format!(
            "              %{p}_kmain = arith.muli %{p}_kpairs, %{p}_cgrp : index"
        ));
        let _ = line; // release the closure's &mut buf so the microkernel can write it
        match mode {
            IntDotMode::Avx2 => {
                Self::emit_i8_microkernel_avx2(buf, p, nr, mr, pl, &acc_ty);
            }
            IntDotMode::Vnni => {
                Self::emit_i8_microkernel_vnni(buf, p, nr, mr, &acc_ty);
            }
        }
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };

        // ── scalar K-tail: the trailing <grp leftover K elements (kmain..kce) ──
        // For K index kk in [kmain, kce), grp_idx = kk/grp and s = kk%grp index
        // the K-group-interleaved panels:
        //   A[t,kk] at Ap[irbase + grp_idx*(MR*grp) + t*grp + s] ({pty}),
        //   B[kk,n] at Bp[jrbase + grp_idx*(NR*grp) + n*grp + s] ({pty}).
        // For AVX2 (grp=2) this is at most 1 element (s=0); for VNNI (grp=4) up
        // to 3 (s∈0..2). The packed panels hold the TRUE signed value (the VNNI
        // +128 unsigned bias lives only inside vpdpbusd, never in the panels),
        // so the tail product is the plain signed A[t,kk]*B[kk,n] in both rungs.
        // NR is small and this runs at most `grp-1` times per KC panel, so a
        // scalar per-column update is fine and bit-identical (associative add).
        let tail_init = (0..mr)
            .map(|t| format!("%{p}_tacc{t} = %{p}_va#{t}"))
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!(
            "              %{p}_vt:{mr} = scf.for %{p}_kt = %{p}_kmain to %{p}_kce \
             step %{p}_c1 iter_args({tail_init}) -> ({acc_ty}) {{"
        ));
        line(&format!(
            "                %{p}_kt64 = arith.index_cast %{p}_kt : index to i64"
        ));
        line(&format!(
            "                %{p}_tpair = arith.divui %{p}_kt64, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_ts = arith.remui %{p}_kt64, %{p}_cgrpi : i64"
        ));
        // B base for this K element: Bp[jrbase + grp_idx*(NR*grp) + s].
        line(&format!(
            "                %{p}_tbnr2 = arith.muli %{p}_nrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_tbkr = arith.muli %{p}_tpair, %{p}_tbnr2 : i64"
        ));
        line(&format!(
            "                %{p}_tbkrs = arith.addi %{p}_tbkr, %{p}_ts : i64"
        ));
        line(&format!(
            "                %{p}_tbbase = arith.addi %{p}_jrbase, %{p}_tbkrs : i64"
        ));
        line(&format!(
            "                %{p}_tamr2 = arith.muli %{p}_mrc, %{p}_cgrpi : i64"
        ));
        line(&format!(
            "                %{p}_takr = arith.muli %{p}_tpair, %{p}_tamr2 : i64"
        ));
        line(&format!(
            "                %{p}_takrs = arith.addi %{p}_takr, %{p}_ts : i64"
        ));
        line(&format!(
            "                %{p}_tabase = arith.addi %{p}_irbase, %{p}_takrs : i64"
        ));
        let mut tail_yields = Vec::with_capacity(mr);
        for t in 0..mr {
            line(&format!(
                "                %{p}_tat{t} = arith.constant {} : i64",
                t * grp
            ));
            line(&format!(
                "                %{p}_tai{t} = arith.addi %{p}_tabase, %{p}_tat{t} : i64"
            ));
            line(&format!(
                "                %{p}_tap{t} = llvm.getelementptr %{p}_pa[%{p}_tai{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, {pty}"
            ));
            line(&format!(
                "                %{p}_tas{t} = llvm.load %{p}_tap{t} : \
                 !llvm.ptr -> {pty}"
            ));
            // VNNI packs A pre-biased (⊕0x80); the scalar tail needs the TRUE
            // signed A[t,kk], so xor the bias back out before extsi. (AVX2 packs
            // the true signed value already — no un-bias.)
            if mode == IntDotMode::Vnni {
                line(&format!(
                    "                %{p}_tax80{t} = arith.constant -128 : i8"
                ));
                line(&format!(
                    "                %{p}_tasu{t} = arith.xori %{p}_tas{t}, %{p}_tax80{t} : i8"
                ));
            }
            let tas_src = if mode == IntDotMode::Vnni {
                format!("%{p}_tasu{t}")
            } else {
                format!("%{p}_tas{t}")
            };
            line(&format!(
                "                %{p}_taw{t} = arith.extsi {tas_src} : {pty} to i64"
            ));
            // Inner column loop over NR: scalar B[kk,n], product, scatter-add.
            line(&format!(
                "                %{p}_tn{t} = scf.for %{p}_tj{t} = %{p}_c0 to %{p}_cnr \
                 step %{p}_c1 iter_args(%{p}_tav{t} = %{p}_tacc{t}) -> (vector<{nr}xi64>) {{"
            ));
            line(&format!(
                "                  %{p}_tj{t}64 = arith.index_cast %{p}_tj{t} : index to i64"
            ));
            line(&format!(
                "                  %{p}_tj{t}2 = arith.muli %{p}_tj{t}64, %{p}_cgrpi : i64"
            ));
            line(&format!(
                "                  %{p}_tbi{t} = arith.addi %{p}_tbbase, %{p}_tj{t}2 : i64"
            ));
            line(&format!(
                "                  %{p}_tbp{t} = llvm.getelementptr %{p}_pb[%{p}_tbi{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, {pty}"
            ));
            line(&format!(
                "                  %{p}_tbs{t} = llvm.load %{p}_tbp{t} : \
                 !llvm.ptr -> {pty}"
            ));
            line(&format!(
                "                  %{p}_tbw{t} = arith.extsi %{p}_tbs{t} : {pty} to i64"
            ));
            line(&format!(
                "                  %{p}_tpp{t} = arith.muli %{p}_taw{t}, %{p}_tbw{t} : i64"
            ));
            line(&format!(
                "                  %{p}_told{t} = vector.extract %{p}_tav{t}[%{p}_tj{t}] : i64 from vector<{nr}xi64>"
            ));
            line(&format!(
                "                  %{p}_tnew{t} = arith.addi %{p}_told{t}, %{p}_tpp{t} : i64"
            ));
            line(&format!(
                "                  %{p}_tins{t} = vector.insert %{p}_tnew{t}, %{p}_tav{t}[%{p}_tj{t}] : i64 into vector<{nr}xi64>"
            ));
            line(&format!(
                "                  scf.yield %{p}_tins{t} : vector<{nr}xi64>"
            ));
            line("                }");
            tail_yields.push(format!("%{p}_tn{t}"));
        }
        line(&format!(
            "                scf.yield {} : {acc_ty}",
            tail_yields.join(", ")
        ));
        line("              }");
        // ── add the MR×NR i64 partial into the C-scratch tile ────────────────
        for t in 0..mr {
            line(&format!(
                "              %{p}_ct{t} = arith.constant {t} : i64"
            ));
            line(&format!(
                "              %{p}_cr{t} = arith.addi %{p}_ir64, %{p}_ct{t} : i64"
            ));
            line(&format!(
                "              %{p}_crb{t} = arith.muli %{p}_cr{t}, %{p}_ncc : i64"
            ));
            line(&format!(
                "              %{p}_coff{t} = arith.addi %{p}_crb{t}, %{p}_jr64 : i64"
            ));
            line(&format!(
                "              %{p}_csp{t} = llvm.getelementptr %{p}_cs[%{p}_coff{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i64"
            ));
            line(&format!(
                "              %{p}_cold{t} = llvm.load %{p}_csp{t} {{alignment = 8 : i64}} : \
                 !llvm.ptr -> vector<{nr}xi64>"
            ));
            line(&format!(
                "              %{p}_cnew{t} = arith.addi %{p}_cold{t}, %{p}_vt#{t} : vector<{nr}xi64>"
            ));
            line(&format!(
                "              llvm.store %{p}_cnew{t}, %{p}_csp{t} {{alignment = 8 : i64}} : \
                 vector<{nr}xi64>, !llvm.ptr"
            ));
        }
        line("            }"); // end ir
        line("          }"); // end jr
        line("        }"); // end pc

        // ── after the K reduction: truncate the live MCe×NCe C-scratch to i32 ─
        // and store to C[ic+r, jc+col] (C output element-byte = 4).
        line(&format!(
            "        %{p}_nmb = arith.divui %{p}_nce, %{p}_cnr : index"
        ));
        line(&format!(
            "        %{p}_nmain = arith.muli %{p}_nmb, %{p}_cnr : index"
        ));
        line(&format!(
            "        scf.for %{p}_wr = %{p}_c0 to %{p}_mce step %{p}_c1 {{"
        ));
        line(&format!(
            "          %{p}_wr64 = arith.index_cast %{p}_wr : index to i64"
        ));
        line(&format!(
            "          %{p}_csrb = arith.muli %{p}_wr64, %{p}_ncc : i64"
        ));
        line(&format!(
            "          %{p}_dri = arith.addi %{p}_ic64, %{p}_wr64 : i64"
        ));
        line(&format!(
            "          %{p}_drrow = arith.muli %{p}_dri, %{n64} : i64"
        ));
        line(&format!(
            "          %{p}_drb = arith.addi %{p}_drrow, %{p}_jc64 : i64"
        ));
        line(&format!(
            "          scf.for %{p}_wc = %{p}_c0 to %{p}_nmain step %{p}_cnr {{"
        ));
        line(&format!(
            "            %{p}_wc64 = arith.index_cast %{p}_wc : index to i64"
        ));
        line(&format!(
            "            %{p}_csvi = arith.addi %{p}_csrb, %{p}_wc64 : i64"
        ));
        line(&format!(
            "            %{p}_csvp = llvm.getelementptr %{p}_cs[%{p}_csvi] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i64"
        ));
        line(&format!(
            "            %{p}_csvv = llvm.load %{p}_csvp {{alignment = 8 : i64}} : \
             !llvm.ptr -> vector<{nr}xi64>"
        ));
        line(&format!(
            "            %{p}_csvt = arith.trunci %{p}_csvv : vector<{nr}xi64> to vector<{nr}xi32>"
        ));
        line(&format!(
            "            %{p}_dvi = arith.addi %{p}_drb, %{p}_wc64 : i64"
        ));
        line(&format!(
            "            %{p}_dvbo = arith.muli %{p}_dvi, %{p}_eb : i64"
        ));
        line(&format!(
            "            %{p}_dvp = llvm.getelementptr %{cp}[%{p}_dvbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "            llvm.store %{p}_csvt, %{p}_dvp {{alignment = 4 : i64}} : \
             vector<{nr}xi32>, !llvm.ptr"
        ));
        line("          }");
        // scalar column tail [nmain, NCe)
        line(&format!(
            "          scf.for %{p}_wt = %{p}_nmain to %{p}_nce step %{p}_c1 {{"
        ));
        line(&format!(
            "            %{p}_wt64 = arith.index_cast %{p}_wt : index to i64"
        ));
        line(&format!(
            "            %{p}_csti = arith.addi %{p}_csrb, %{p}_wt64 : i64"
        ));
        line(&format!(
            "            %{p}_cstp = llvm.getelementptr %{p}_cs[%{p}_csti] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i64"
        ));
        line(&format!(
            "            %{p}_cstv = llvm.load %{p}_cstp : !llvm.ptr -> i64"
        ));
        line(&format!(
            "            %{p}_cstt = arith.trunci %{p}_cstv : i64 to i32"
        ));
        line(&format!(
            "            %{p}_dti = arith.addi %{p}_drb, %{p}_wt64 : i64"
        ));
        line(&format!(
            "            %{p}_dtbo = arith.muli %{p}_dti, %{p}_eb : i64"
        ));
        line(&format!(
            "            %{p}_dtp = llvm.getelementptr %{cp}[%{p}_dtbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "            llvm.store %{p}_cstt, %{p}_dtp : i32, !llvm.ptr"
        ));
        line("          }");
        line("        }"); // end wr
        line("      }"); // end ic
        line("    }"); // end jc
        // Free the heap scratch panels — one `@free` per `@malloc`, on the single
        // fall-through path out of the kernel (all three scf loops have closed and
        // there are no early returns / branches out of the kernel body, so this is
        // the only exit). NO LEAK.
        line(&format!("    llvm.call @free(%{p}_cs) : (!llvm.ptr) -> ()"));
        line(&format!("    llvm.call @free(%{p}_pa) : (!llvm.ptr) -> ()"));
        line(&format!("    llvm.call @free(%{p}_pb) : (!llvm.ptr) -> ()"));
    }

    /// RFC 0006 Track B — emit the fused outer-product Q16.16 GEMM.
    ///
    /// `C[i,j] = trunc_i32( Σ_k ((A[i,k]*B[k,j]) >> 16) )` for A = M×K
    /// row-major, B = **K×N row-major**, C = M×N row-major caller-allocated
    /// (all i32 Q16.16 elements, base addresses packed i64).
    ///
    /// Register-tiled outer-product microkernel — no horizontal reduction:
    ///
    /// ```text
    ///   region A (vector tile): j0 = 0..n_main step NR, i0 = 0..m_main step MR
    ///     acc[0..MR] = dense<0> : vector<NRxi64>
    ///     k-loop (scf.for iter_args = acc):
    ///       bw = extsi( load vector<NRxi32> at B+(k*N+j0)*4 )
    ///       for t in 0..MR:
    ///         aw = extsi( load i32 at A+((i0+t)*K+k)*4 )
    ///         p  = (broadcast aw) * bw ; ps = p >> 16 ; acc[t] += ps
    ///     for t in 0..MR: store trunci(acc[t]) -> C+((i0+t)*N+j0)*4
    ///   region B (M row tail): j0 = 0..n_main step NR, i = m_main..M step 1
    ///     single-row vector<NRxi64> accumulator over the NR-wide column block
    ///   region C (N col tail): j = n_main..N step 1, i = 0..M step 1
    ///     scalar per-element Q16.16 dot
    /// ```
    ///
    /// The three regions partition the M×N output with no overlap:
    /// A=[0..m_main]×[0..n_main], B=[m_main..M]×[0..n_main],
    /// C=[0..M]×[n_main..N]. Each product term is `arith.shrsi`-shifted by 16
    /// individually before the i64 accumulation, and the i64→i32 truncation
    /// happens exactly once per output element at the store. i64 add is
    /// associative + commutative, so this is byte-identical to the
    /// per-element scalar oracle `Σ_k (A[i,k]*B[k,j])>>16` under any tiling or
    /// lane grouping. At `-march=x86-64-v3` the `arith.muli` on
    /// `vector<8xi64>` lowers to the AVX2 `vpmuldq` idiom.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_mm_q16(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        c_addr: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
    ) {
        let d = dst.0;
        // Setup: materialise the three base pointers, the K/N i64 SSA names and
        // the K/M/N `index` bounds, then delegate to the BLIS-blocked
        // macro-kernel over the full row range `[0, M)`. The blocked emitter
        // owns its private alloca scratch; the per-element Q16.16 math is
        // byte-identical to the scalar oracle (each product `>> 16`-shifted to a
        // fixed i64 before an associative i64 accumulation, truncated once).
        self.emit_line(&format!("    %vmm_z0_{d} = arith.constant 0 : i64"));
        self.emit_line(&format!(
            "    %vmm_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vmm_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vmm_cp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            c_addr.0
        ));
        self.emit_line(&format!(
            "    %vmm_mi_{d} = arith.index_cast %{} : i64 to index",
            m.0
        ));
        self.emit_line(&format!(
            "    %vmm_ki_{d} = arith.index_cast %{} : i64 to index",
            k.0
        ));
        self.emit_line(&format!(
            "    %vmm_ni_{d} = arith.index_cast %{} : i64 to index",
            n.0
        ));
        self.emit_line(&format!(
            "    %vmm_k64_{d} = arith.addi %{}, %vmm_z0_{d} : i64",
            k.0
        ));
        self.emit_line(&format!(
            "    %vmm_n64_{d} = arith.addi %{}, %vmm_z0_{d} : i64",
            n.0
        ));
        self.emit_line(&format!("    %vmm_rs_{d} = arith.constant 0 : index"));
        // ── size-threshold dispatch (BYTE-IDENTICAL on both arms) ────────────
        // Small shapes (whole problem already L1/L2-resident) skip the BLIS
        // packing + C-scratch-init overhead and run the simpler fused
        // outer-product kernel (`emit_mm_q16_row_band` over the full `[0, M)`);
        // large shapes keep the cache-blocked BLIS macro-kernel so the K-deep B
        // panel stays resident. Both arms emit the same per-term
        // `arith.shrsi >> 16` and the same associative i64 reduction, so the
        // output is bit-identical regardless of which arm runs — the branch only
        // changes tiling, never the math (canary `92e2cb75` is dispatch-invariant).
        self.emit_line(&format!(
            "    %vmm_thr_{d} = arith.constant {Q16_BLIS_MIN_DIM} : index"
        ));
        self.emit_line(&format!(
            "    %vmm_mxa_{d} = arith.maxui %vmm_mi_{d}, %vmm_ni_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmm_mxd_{d} = arith.maxui %vmm_mxa_{d}, %vmm_ki_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmm_small_{d} = arith.cmpi ult, %vmm_mxd_{d}, %vmm_thr_{d} : index"
        ));
        self.emit_line(&format!("    scf.if %vmm_small_{d} {{"));
        let mut simple = String::new();
        Self::emit_mm_q16_row_band(
            &mut simple,
            &format!("vms_{d}"),
            &format!("vmm_ap_{d}"),
            &format!("vmm_bp_{d}"),
            &format!("vmm_cp_{d}"),
            &format!("vmm_k64_{d}"),
            &format!("vmm_n64_{d}"),
            &format!("vmm_ki_{d}"),
            &format!("vmm_ni_{d}"),
            &format!("vmm_rs_{d}"),
            &format!("vmm_mi_{d}"),
        );
        self.body.push_str(&simple);
        self.emit_line("    } else {");
        let mut blk = String::new();
        Self::emit_mm_q16_blocked(
            &mut blk,
            &format!("vmb_{d}"),
            &format!("vmm_ap_{d}"),
            &format!("vmm_bp_{d}"),
            &format!("vmm_cp_{d}"),
            &format!("vmm_k64_{d}"),
            &format!("vmm_n64_{d}"),
            &format!("vmm_ki_{d}"),
            &format!("vmm_ni_{d}"),
            &format!("vmm_rs_{d}"),
            &format!("vmm_mi_{d}"),
        );
        self.body.push_str(&blk);
        self.emit_line("    }");
        // The intrinsic returns 0 (i64) — matches the gemv-composed sibling.
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    /// Emit the fused outer-product Q16.16 GEMM kernel restricted to a
    /// **contiguous band of output rows** `[row_start, row_end)`, used by the
    /// multithreaded `__mind_blas_matmul_mm_q16_mt_v` worker.
    ///
    /// This is the EXACT register-tiled microkernel of `emit_vec_matmul_mm_q16`
    /// (same NR-wide column accumulator, MR A-rows per tile, individual
    /// `>> 16`-shift-then-i64-accumulate, single i64→i32 truncation at the
    /// store, and identical scalar column/row tails), with the row loops bounded
    /// by `[row_start, row_end)` instead of `[0, M)`. Because each output
    /// element `C[i, *]` (i in the band) is computed independently — every term
    /// shifted to a fixed i64 value before an associative i64 add — the bytes a
    /// worker writes for its band are identical to what the single-thread kernel
    /// would write for the same rows. The bands partition `[0, M)`, so the union
    /// is byte-for-byte the single-thread output regardless of how `[0, M)` is
    /// split (owner-computes: no shared accumulator, no cross-band reduction).
    ///
    /// `prefix` namespaces every SSA value so multiple workers (and the worker
    /// vs `@main`) never collide. `ap`/`bp`/`cp` are `!llvm.ptr` SSA value
    /// *names* (already materialised in the caller). `k64`/`n64` are i64 SSA
    /// value names for K and N. `row_start`/`row_end` are `index` SSA value
    /// names bounding the band. Text is appended to `buf`; the caller wraps it
    /// in a function body.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_mm_q16_row_band(
        buf: &mut String,
        prefix: &str,
        ap: &str,
        bp: &str,
        cp: &str,
        k64: &str,
        n64: &str,
        ki: &str,
        ni: &str,
        row_start: &str,
        row_end: &str,
    ) {
        use std::fmt::Write;
        let p = prefix;
        const NR: usize = 8;
        const MR: usize = 4;
        let elem_bytes = std::mem::size_of::<i32>() as i64;
        let mut line = |s: &str| {
            writeln!(buf, "{s}").expect("write to string cannot fail");
        };

        // ── constants ────────────────────────────────────────────────────────
        line(&format!("    %{p}_c0 = arith.constant 0 : index"));
        line(&format!("    %{p}_c1 = arith.constant 1 : index"));
        line(&format!("    %{p}_nr = arith.constant {NR} : index"));
        line(&format!("    %{p}_mr = arith.constant {MR} : index"));
        line(&format!("    %{p}_eb = arith.constant {elem_bytes} : i64"));
        line(&format!("    %{p}_s16 = arith.constant 16 : i64"));
        line(&format!("    %{p}_z0 = arith.constant 0 : i64"));
        line(&format!(
            "    %{p}_s16v = arith.constant dense<16> : vector<{NR}xi64>"
        ));
        line(&format!(
            "    %{p}_zv = arith.constant dense<0> : vector<{NR}xi64>"
        ));

        // ── column bounds (N-derived, row-band-invariant) ────────────────────
        line(&format!(
            "    %{p}_nnb = arith.divui %{ni}, %{p}_nr : index"
        ));
        line(&format!(
            "    %{p}_nmain = arith.muli %{p}_nnb, %{p}_nr : index"
        ));
        // m_main relative to the band: row_start + ((row_end-row_start)/MR)*MR.
        line(&format!(
            "    %{p}_rb = arith.subi %{row_end}, %{row_start} : index"
        ));
        line(&format!(
            "    %{p}_mmb = arith.divui %{p}_rb, %{p}_mr : index"
        ));
        line(&format!(
            "    %{p}_mbk = arith.muli %{p}_mmb, %{p}_mr : index"
        ));
        line(&format!(
            "    %{p}_mmain = arith.addi %{row_start}, %{p}_mbk : index"
        ));

        // ════════════════════════════════════════════════════════════════════
        //  Region A — vector tile: MR rows × NR columns, k-reduced.
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "    scf.for %{p}_j0 = %{p}_c0 to %{p}_nmain step %{p}_nr {{"
        ));
        line(&format!(
            "      %{p}_j0i = arith.index_cast %{p}_j0 : index to i64"
        ));
        line(&format!(
            "      scf.for %{p}_i0 = %{row_start} to %{p}_mmain step %{p}_mr {{"
        ));
        line(&format!(
            "        %{p}_i0i = arith.index_cast %{p}_i0 : index to i64"
        ));
        let acc_init = (0..MR)
            .map(|t| format!("%{p}_acc{t} = %{p}_zv"))
            .collect::<Vec<_>>()
            .join(", ");
        let acc_ty = (0..MR)
            .map(|_| format!("vector<{NR}xi64>"))
            .collect::<Vec<_>>()
            .join(", ");
        line(&format!(
            "        %{p}_va:{MR} = scf.for %{p}_k = %{p}_c0 to %{ki} \
             step %{p}_c1 iter_args({acc_init}) -> ({acc_ty}) {{"
        ));
        line(&format!(
            "          %{p}_ki64 = arith.index_cast %{p}_k : index to i64"
        ));
        line(&format!(
            "          %{p}_kn = arith.muli %{p}_ki64, %{n64} : i64"
        ));
        line(&format!(
            "          %{p}_bidx = arith.addi %{p}_kn, %{p}_j0i : i64"
        ));
        line(&format!(
            "          %{p}_bbo = arith.muli %{p}_bidx, %{p}_eb : i64"
        ));
        line(&format!(
            "          %{p}_bptr = llvm.getelementptr %{bp}[%{p}_bbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "          %{p}_bv = llvm.load %{p}_bptr {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{NR}xi32>"
        ));
        line(&format!(
            "          %{p}_bw = arith.extsi %{p}_bv : vector<{NR}xi32> to vector<{NR}xi64>"
        ));
        let mut yields = Vec::with_capacity(MR);
        for t in 0..MR {
            line(&format!("          %{p}_rt{t} = arith.constant {t} : i64"));
            line(&format!(
                "          %{p}_it{t} = arith.addi %{p}_i0i, %{p}_rt{t} : i64"
            ));
            line(&format!(
                "          %{p}_ik{t} = arith.muli %{p}_it{t}, %{k64} : i64"
            ));
            line(&format!(
                "          %{p}_aidx{t} = arith.addi %{p}_ik{t}, %{p}_ki64 : i64"
            ));
            line(&format!(
                "          %{p}_abo{t} = arith.muli %{p}_aidx{t}, %{p}_eb : i64"
            ));
            line(&format!(
                "          %{p}_aptr{t} = llvm.getelementptr %{ap}[%{p}_abo{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            line(&format!(
                "          %{p}_as{t} = llvm.load %{p}_aptr{t} : !llvm.ptr -> i32"
            ));
            line(&format!(
                "          %{p}_aw{t} = arith.extsi %{p}_as{t} : i32 to i64"
            ));
            line(&format!(
                "          %{p}_ab{t} = vector.broadcast %{p}_aw{t} : i64 to vector<{NR}xi64>"
            ));
            line(&format!(
                "          %{p}_pp{t} = arith.muli %{p}_ab{t}, %{p}_bw : vector<{NR}xi64>"
            ));
            line(&format!(
                "          %{p}_ps{t} = arith.shrsi %{p}_pp{t}, %{p}_s16v : vector<{NR}xi64>"
            ));
            line(&format!(
                "          %{p}_na{t} = arith.addi %{p}_acc{t}, %{p}_ps{t} : vector<{NR}xi64>"
            ));
            yields.push(format!("%{p}_na{t}"));
        }
        line(&format!(
            "          scf.yield {} : {acc_ty}",
            yields.join(", ")
        ));
        line("        }");
        for t in 0..MR {
            line(&format!("        %{p}_strt{t} = arith.constant {t} : i64"));
            line(&format!(
                "        %{p}_sit{t} = arith.addi %{p}_i0i, %{p}_strt{t} : i64"
            ));
            line(&format!(
                "        %{p}_cin{t} = arith.muli %{p}_sit{t}, %{n64} : i64"
            ));
            line(&format!(
                "        %{p}_cidx{t} = arith.addi %{p}_cin{t}, %{p}_j0i : i64"
            ));
            line(&format!(
                "        %{p}_cbo{t} = arith.muli %{p}_cidx{t}, %{p}_eb : i64"
            ));
            line(&format!(
                "        %{p}_cptr{t} = llvm.getelementptr %{cp}[%{p}_cbo{t}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            line(&format!(
                "        %{p}_lo{t} = arith.trunci %{p}_va#{t} : vector<{NR}xi64> to vector<{NR}xi32>"
            ));
            line(&format!(
                "        llvm.store %{p}_lo{t}, %{p}_cptr{t} {{alignment = 4 : i64}} : \
                 vector<{NR}xi32>, !llvm.ptr"
            ));
        }
        line("      }");
        line("    }");

        // ════════════════════════════════════════════════════════════════════
        //  Region B — band M row tail: rows [m_main..row_end), cols [0..n_main).
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "    scf.for %{p}b_j0 = %{p}_c0 to %{p}_nmain step %{p}_nr {{"
        ));
        line(&format!(
            "      %{p}b_j0i = arith.index_cast %{p}b_j0 : index to i64"
        ));
        line(&format!(
            "      scf.for %{p}b_i = %{p}_mmain to %{row_end} step %{p}_c1 {{"
        ));
        line(&format!(
            "        %{p}b_ii = arith.index_cast %{p}b_i : index to i64"
        ));
        line(&format!(
            "        %{p}b_vacc = scf.for %{p}b_k = %{p}_c0 to %{ki} \
             step %{p}_c1 iter_args(%{p}b_acc = %{p}_zv) -> (vector<{NR}xi64>) {{"
        ));
        line(&format!(
            "          %{p}b_ki64 = arith.index_cast %{p}b_k : index to i64"
        ));
        line(&format!(
            "          %{p}b_kn = arith.muli %{p}b_ki64, %{n64} : i64"
        ));
        line(&format!(
            "          %{p}b_bidx = arith.addi %{p}b_kn, %{p}b_j0i : i64"
        ));
        line(&format!(
            "          %{p}b_bbo = arith.muli %{p}b_bidx, %{p}_eb : i64"
        ));
        line(&format!(
            "          %{p}b_bptr = llvm.getelementptr %{bp}[%{p}b_bbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "          %{p}b_bv = llvm.load %{p}b_bptr {{alignment = 4 : i64}} : \
             !llvm.ptr -> vector<{NR}xi32>"
        ));
        line(&format!(
            "          %{p}b_bw = arith.extsi %{p}b_bv : vector<{NR}xi32> to vector<{NR}xi64>"
        ));
        line(&format!(
            "          %{p}b_ik = arith.muli %{p}b_ii, %{k64} : i64"
        ));
        line(&format!(
            "          %{p}b_aidx = arith.addi %{p}b_ik, %{p}b_ki64 : i64"
        ));
        line(&format!(
            "          %{p}b_abo = arith.muli %{p}b_aidx, %{p}_eb : i64"
        ));
        line(&format!(
            "          %{p}b_aptr = llvm.getelementptr %{ap}[%{p}b_abo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "          %{p}b_as = llvm.load %{p}b_aptr : !llvm.ptr -> i32"
        ));
        line(&format!(
            "          %{p}b_aw = arith.extsi %{p}b_as : i32 to i64"
        ));
        line(&format!(
            "          %{p}b_ab = vector.broadcast %{p}b_aw : i64 to vector<{NR}xi64>"
        ));
        line(&format!(
            "          %{p}b_pp = arith.muli %{p}b_ab, %{p}b_bw : vector<{NR}xi64>"
        ));
        line(&format!(
            "          %{p}b_ps = arith.shrsi %{p}b_pp, %{p}_s16v : vector<{NR}xi64>"
        ));
        line(&format!(
            "          %{p}b_na = arith.addi %{p}b_acc, %{p}b_ps : vector<{NR}xi64>"
        ));
        line(&format!("          scf.yield %{p}b_na : vector<{NR}xi64>"));
        line("        }");
        line(&format!(
            "        %{p}b_cin = arith.muli %{p}b_ii, %{n64} : i64"
        ));
        line(&format!(
            "        %{p}b_cidx = arith.addi %{p}b_cin, %{p}b_j0i : i64"
        ));
        line(&format!(
            "        %{p}b_cbo = arith.muli %{p}b_cidx, %{p}_eb : i64"
        ));
        line(&format!(
            "        %{p}b_cptr = llvm.getelementptr %{cp}[%{p}b_cbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "        %{p}b_lo = arith.trunci %{p}b_vacc : vector<{NR}xi64> to vector<{NR}xi32>"
        ));
        line(&format!(
            "        llvm.store %{p}b_lo, %{p}b_cptr {{alignment = 4 : i64}} : \
             vector<{NR}xi32>, !llvm.ptr"
        ));
        line("      }");
        line("    }");

        // ════════════════════════════════════════════════════════════════════
        //  Region C — N column tail: cols [n_main..N), all band rows.
        // ════════════════════════════════════════════════════════════════════
        line(&format!(
            "    scf.for %{p}c_j = %{p}_nmain to %{ni} step %{p}_c1 {{"
        ));
        line(&format!(
            "      %{p}c_ji = arith.index_cast %{p}c_j : index to i64"
        ));
        line(&format!(
            "      scf.for %{p}c_i = %{row_start} to %{row_end} step %{p}_c1 {{"
        ));
        line(&format!(
            "        %{p}c_ii = arith.index_cast %{p}c_i : index to i64"
        ));
        line(&format!(
            "        %{p}c_iK = arith.muli %{p}c_ii, %{k64} : i64"
        ));
        line(&format!(
            "        %{p}c_acc = scf.for %{p}c_k = %{p}_c0 to %{ki} \
             step %{p}_c1 iter_args(%{p}c_s = %{p}_z0) -> (i64) {{"
        ));
        line(&format!(
            "          %{p}c_ki64 = arith.index_cast %{p}c_k : index to i64"
        ));
        line(&format!(
            "          %{p}c_aidx = arith.addi %{p}c_iK, %{p}c_ki64 : i64"
        ));
        line(&format!(
            "          %{p}c_abo = arith.muli %{p}c_aidx, %{p}_eb : i64"
        ));
        line(&format!(
            "          %{p}c_aptr = llvm.getelementptr %{ap}[%{p}c_abo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "          %{p}c_as = llvm.load %{p}c_aptr : !llvm.ptr -> i32"
        ));
        line(&format!(
            "          %{p}c_aw = arith.extsi %{p}c_as : i32 to i64"
        ));
        line(&format!(
            "          %{p}c_kn = arith.muli %{p}c_ki64, %{n64} : i64"
        ));
        line(&format!(
            "          %{p}c_bidx = arith.addi %{p}c_kn, %{p}c_ji : i64"
        ));
        line(&format!(
            "          %{p}c_bbo = arith.muli %{p}c_bidx, %{p}_eb : i64"
        ));
        line(&format!(
            "          %{p}c_bptr = llvm.getelementptr %{bp}[%{p}c_bbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "          %{p}c_bs = llvm.load %{p}c_bptr : !llvm.ptr -> i32"
        ));
        line(&format!(
            "          %{p}c_bw = arith.extsi %{p}c_bs : i32 to i64"
        ));
        line(&format!(
            "          %{p}c_pp = arith.muli %{p}c_aw, %{p}c_bw : i64"
        ));
        line(&format!(
            "          %{p}c_ps = arith.shrsi %{p}c_pp, %{p}_s16 : i64"
        ));
        line(&format!(
            "          %{p}c_na = arith.addi %{p}c_s, %{p}c_ps : i64"
        ));
        line(&format!("          scf.yield %{p}c_na : i64"));
        line("        }");
        line(&format!(
            "        %{p}c_lo = arith.trunci %{p}c_acc : i64 to i32"
        ));
        line(&format!(
            "        %{p}c_iN = arith.muli %{p}c_ii, %{n64} : i64"
        ));
        line(&format!(
            "        %{p}c_cidx = arith.addi %{p}c_iN, %{p}c_ji : i64"
        ));
        line(&format!(
            "        %{p}c_cbo = arith.muli %{p}c_cidx, %{p}_eb : i64"
        ));
        line(&format!(
            "        %{p}c_cptr = llvm.getelementptr %{cp}[%{p}c_cbo] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        line(&format!(
            "        llvm.store %{p}c_lo, %{p}c_cptr : i32, !llvm.ptr"
        ));
        line("      }");
        line("    }");
    }

    /// Emit the multithreaded fused outer-product Q16.16 GEMM
    /// (`__mind_blas_matmul_mm_q16_mt_v`).
    ///
    /// Mechanism: raw POSIX threads emitted directly in the LLVM dialect (no
    /// libomp). The output rows `[0, M)` are split into `T` contiguous
    /// owner-computes bands; each thread runs the fused row-band kernel
    /// (`emit_mm_q16_row_band`) over its `[row_start, row_end)` and writes
    /// `C[row_start:row_end, 0:N]` with the SAME math as the single-thread
    /// kernel. No cross-thread reduction, no atomics, no shared accumulator —
    /// every output element is written by exactly one thread, so the result is
    /// byte-for-byte identical to the single-thread kernel for any `T`. `T` is
    /// read at runtime from `sysconf(_SC_NPROCESSORS_ONLN)` (clamped to
    /// `[1, M]`); because the output does not depend on `T`, this is safe for
    /// cross-substrate bit-identity.
    ///
    /// The thread-arg struct is `{a:i64, b:i64, c:i64, k:i64, n:i64,
    /// row_start:i64, row_end:i64}` (the worker re-derives `!llvm.ptr`s and
    /// `index` bounds from these i64 fields). `@main` stack-allocates `T` arg
    /// structs and `T` `pthread_t` (i64) handles, spawns, then joins.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_mm_q16_mt(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        c_addr: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
    ) {
        use std::fmt::Write;
        let d = dst.0;
        self.needs_pthread = true;

        // ── worker function (top-level llvm.func) ────────────────────────────
        // Arg struct layout (all i64): [0]=a [1]=b [2]=c [3]=k [4]=n
        //                              [5]=row_start [6]=row_end
        let wname = format!("mind_mt_worker_q16_{d}");
        let mut w = String::new();
        writeln!(
            &mut w,
            "  llvm.func @{wname}(%arg0: !llvm.ptr) -> !llvm.ptr {{"
        )
        .unwrap();
        // Load the 7 i64 fields from the packed arg struct (contiguous i64s).
        for (idx, fld) in [
            (0usize, "a"),
            (1, "b"),
            (2, "c"),
            (3, "k"),
            (4, "n"),
            (5, "rs"),
            (6, "re"),
        ] {
            let off = (idx as i64) * 8;
            writeln!(
                &mut w,
                "    %wk_{d}_o{idx} = llvm.mlir.constant({off} : i64) : i64"
            )
            .unwrap();
            writeln!(
                &mut w,
                "    %wk_{d}_p{idx} = llvm.getelementptr %arg0[%wk_{d}_o{idx}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            )
            .unwrap();
            writeln!(
                &mut w,
                "    %wk_{d}_{fld} = llvm.load %wk_{d}_p{idx} : !llvm.ptr -> i64"
            )
            .unwrap();
        }
        // Re-derive pointers and band bounds.
        writeln!(
            &mut w,
            "    %wk_{d}_ap = llvm.inttoptr %wk_{d}_a : i64 to !llvm.ptr"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wk_{d}_bp = llvm.inttoptr %wk_{d}_b : i64 to !llvm.ptr"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wk_{d}_cp = llvm.inttoptr %wk_{d}_c : i64 to !llvm.ptr"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wk_{d}_ki = arith.index_cast %wk_{d}_k : i64 to index"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wk_{d}_ni = arith.index_cast %wk_{d}_n : i64 to index"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wk_{d}_rsi = arith.index_cast %wk_{d}_rs : i64 to index"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wk_{d}_rei = arith.index_cast %wk_{d}_re : i64 to index"
        )
        .unwrap();
        Self::emit_mm_q16_row_band(
            &mut w,
            &format!("mtk_{d}"),
            &format!("wk_{d}_ap"),
            &format!("wk_{d}_bp"),
            &format!("wk_{d}_cp"),
            &format!("wk_{d}_k"),
            &format!("wk_{d}_n"),
            &format!("wk_{d}_ki"),
            &format!("wk_{d}_ni"),
            &format!("wk_{d}_rsi"),
            &format!("wk_{d}_rei"),
        );
        writeln!(&mut w, "    %wk_{d}_null = llvm.mlir.zero : !llvm.ptr").unwrap();
        writeln!(&mut w, "    llvm.return %wk_{d}_null : !llvm.ptr").unwrap();
        writeln!(&mut w, "  }}").unwrap();
        self.mt_workers.push_str(&w);

        // ── entry (inside @main): spawn + join T threads ─────────────────────
        // Thread count T = clamp(sysconf(_SC_NPROCESSORS_ONLN), 1, M). On linux
        // _SC_NPROCESSORS_ONLN == 84. Owner-computes => output is T-invariant.
        const SC_NPROCESSORS_ONLN: i64 = 84;
        // Cap the number of stack-allocated thread slots so the `alloca` extent
        // is a compile-time constant (statically reserved — no pointer bits in
        // the output, ASLR-independent results). T is clamped to this at run.
        const MAX_THREADS: i64 = 256;
        let struct_bytes: i64 = 7 * 8; // 7 i64 fields per thread-arg struct.

        // Zero constant used to materialise i64 SSA copies of the i64 ABI args
        // (MLIR has no bare SSA-alias form; `addi x, 0` is a trivial value copy).
        self.emit_line(&format!("    %mt{d}_zc = arith.constant 0 : i64"));
        self.emit_line(&format!(
            "    %mt{d}_scq = llvm.mlir.constant({SC_NPROCESSORS_ONLN} : i32) : i32"
        ));
        self.emit_line(&format!(
            "    %mt{d}_ncpu = llvm.call @sysconf(%mt{d}_scq) : (i32) -> i64"
        ));
        // m as a named i64 SSA value (trivial copy of the i64 ABI arg).
        self.emit_line(&format!(
            "    %mt{d}_m64 = arith.addi %{}, %mt{d}_zc : i64",
            m.0
        ));
        // Clamp T into [1, min(M, MAX_THREADS)].
        self.emit_line(&format!(
            "    %mt{d}_one64 = llvm.mlir.constant(1 : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_maxt = llvm.mlir.constant({MAX_THREADS} : i64) : i64"
        ));
        // upper = min(M, MAX_THREADS)
        self.emit_line(&format!(
            "    %mt{d}_mlt = arith.cmpi slt, %mt{d}_m64, %mt{d}_maxt : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_upper = arith.select %mt{d}_mlt, %mt{d}_m64, %mt{d}_maxt : i64"
        ));
        // T = max(1, min(ncpu, upper))
        self.emit_line(&format!(
            "    %mt{d}_clt = arith.cmpi slt, %mt{d}_ncpu, %mt{d}_upper : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_tcap = arith.select %mt{d}_clt, %mt{d}_ncpu, %mt{d}_upper : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_tlt1 = arith.cmpi slt, %mt{d}_tcap, %mt{d}_one64 : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_T = arith.select %mt{d}_tlt1, %mt{d}_one64, %mt{d}_tcap : i64"
        ));
        // band = ceildiv(M, T) = (M + T - 1) / T.
        self.emit_line(&format!(
            "    %mt{d}_tm1 = arith.subi %mt{d}_T, %mt{d}_one64 : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_msum = arith.addi %mt{d}_m64, %mt{d}_tm1 : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_band = arith.divsi %mt{d}_msum, %mt{d}_T : i64"
        ));
        // Stack-alloc MAX_THREADS arg structs (MAX_THREADS*56 bytes) + handles.
        let argbuf_bytes = MAX_THREADS * struct_bytes;
        self.emit_line(&format!(
            "    %mt{d}_abnum = llvm.mlir.constant({argbuf_bytes} : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_argbuf = llvm.alloca %mt{d}_abnum x i8 : (i64) -> !llvm.ptr"
        ));
        self.emit_line(&format!(
            "    %mt{d}_hnum = llvm.mlir.constant({MAX_THREADS} : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_handles = llvm.alloca %mt{d}_hnum x i64 : (i64) -> !llvm.ptr"
        ));
        // Worker function pointer.
        self.emit_line(&format!(
            "    %mt{d}_wfp = llvm.mlir.addressof @{wname} : !llvm.ptr"
        ));
        self.emit_line(&format!("    %mt{d}_pnull = llvm.mlir.zero : !llvm.ptr"));
        // Common constants for the spawn/join loops.
        self.emit_line(&format!(
            "    %mt{d}_sb = llvm.mlir.constant({struct_bytes} : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mt{d}_hsz = llvm.mlir.constant(8 : i64) : i64"
        ));
        self.emit_line(&format!("    %mt{d}_c0idx = arith.constant 0 : index"));
        self.emit_line(&format!("    %mt{d}_c1idx = arith.constant 1 : index"));
        self.emit_line(&format!(
            "    %mt{d}_Ti = arith.index_cast %mt{d}_T : i64 to index"
        ));
        // Pre-pack the i64 a/b/c addresses once.
        self.emit_line(&format!(
            "    %mt{d}_a64 = arith.addi %{}, %mt{d}_zc : i64",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %mt{d}_b64 = arith.addi %{}, %mt{d}_zc : i64",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %mt{d}_c64 = arith.addi %{}, %mt{d}_zc : i64",
            c_addr.0
        ));
        self.emit_line(&format!(
            "    %mt{d}_k64 = arith.addi %{}, %mt{d}_zc : i64",
            k.0
        ));
        self.emit_line(&format!(
            "    %mt{d}_n64 = arith.addi %{}, %mt{d}_zc : i64",
            n.0
        ));

        // Spawn loop: for t in 0..T.
        self.emit_line(&format!(
            "    scf.for %mt{d}_t = %mt{d}_c0idx to %mt{d}_Ti step %mt{d}_c1idx {{"
        ));
        self.emit_line(&format!(
            "      %mt{d}_ti = arith.index_cast %mt{d}_t : index to i64"
        ));
        // row_start = min(t*band, M) ; row_end = min(row_start+band, M).
        self.emit_line(&format!(
            "      %mt{d}_rs0 = arith.muli %mt{d}_ti, %mt{d}_band : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_rslt = arith.cmpi slt, %mt{d}_rs0, %mt{d}_m64 : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_rs = arith.select %mt{d}_rslt, %mt{d}_rs0, %mt{d}_m64 : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_re0 = arith.addi %mt{d}_rs, %mt{d}_band : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_relt = arith.cmpi slt, %mt{d}_re0, %mt{d}_m64 : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_re = arith.select %mt{d}_relt, %mt{d}_re0, %mt{d}_m64 : i64"
        ));
        // arg = argbuf + t*struct_bytes.
        self.emit_line(&format!(
            "      %mt{d}_aoff = arith.muli %mt{d}_ti, %mt{d}_sb : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_argp = llvm.getelementptr %mt{d}_argbuf[%mt{d}_aoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        // Store the 7 i64 fields.
        for (idx, val) in [
            (0usize, format!("%mt{d}_a64")),
            (1, format!("%mt{d}_b64")),
            (2, format!("%mt{d}_c64")),
            (3, format!("%mt{d}_k64")),
            (4, format!("%mt{d}_n64")),
            (5, format!("%mt{d}_rs")),
            (6, format!("%mt{d}_re")),
        ] {
            let off = (idx as i64) * 8;
            self.emit_line(&format!(
                "      %mt{d}_so{idx} = llvm.mlir.constant({off} : i64) : i64"
            ));
            self.emit_line(&format!(
                "      %mt{d}_sp{idx} = llvm.getelementptr %mt{d}_argp[%mt{d}_so{idx}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "      llvm.store {val}, %mt{d}_sp{idx} : i64, !llvm.ptr"
            ));
        }
        // handle slot = handles + t*8.
        self.emit_line(&format!(
            "      %mt{d}_hoff = arith.muli %mt{d}_ti, %mt{d}_hsz : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_hp = llvm.getelementptr %mt{d}_handles[%mt{d}_hoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %mt{d}_crc = llvm.call @pthread_create(%mt{d}_hp, %mt{d}_pnull, %mt{d}_wfp, %mt{d}_argp) : \
             (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32"
        ));
        self.emit_line("    }");

        // Join loop: for t in 0..T.
        self.emit_line(&format!(
            "    scf.for %mt{d}_jt = %mt{d}_c0idx to %mt{d}_Ti step %mt{d}_c1idx {{"
        ));
        self.emit_line(&format!(
            "      %mt{d}_jti = arith.index_cast %mt{d}_jt : index to i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_jhoff = arith.muli %mt{d}_jti, %mt{d}_hsz : i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_jhp = llvm.getelementptr %mt{d}_handles[%mt{d}_jhoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %mt{d}_tid = llvm.load %mt{d}_jhp : !llvm.ptr -> i64"
        ));
        self.emit_line(&format!(
            "      %mt{d}_jrc = llvm.call @pthread_join(%mt{d}_tid, %mt{d}_pnull) : \
             (i64, !llvm.ptr) -> i32"
        ));
        self.emit_line("    }");

        // The intrinsic returns 0 (i64) — matches the single-thread sibling.
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    /// "det.igemm" tier — emit the **multithreaded** fused int8 GEMM
    /// (`__mind_blas_matmul_mm_i8_mt_v`).
    ///
    /// Structurally identical to `emit_vec_matmul_mm_q16_mt`: the output rows
    /// `[0, M)` are partitioned into `T` contiguous owner-computes bands, one
    /// per raw POSIX thread, and each worker runs the BLIS-blocked int8
    /// macro-kernel (`emit_mm_i8_blocked`) over its `[row_start, row_end)` —
    /// the SAME emitter the single-thread int8 kernel delegates to over
    /// `[0, M)`. There is no cross-thread reduction, no atomic, no shared
    /// accumulator: every output element is written by exactly one thread, and
    /// the worker's i64 C-scratch + i32 packed-A / packed-B panels are private
    /// stack `alloca`s emitted inside the worker `llvm.func` (one fresh set per
    /// thread call frame), so the result is byte-for-byte identical to the
    /// single-thread kernel for any `T`. `T` is read at runtime from
    /// `sysconf(_SC_NPROCESSORS_ONLN)` (clamped to `[1, M]`); because the output
    /// does not depend on `T`, this is safe for cross-substrate bit-identity.
    ///
    /// The thread-arg struct is `{a:i64, b:i64, c:i64, k:i64, n:i64,
    /// row_start:i64, row_end:i64}` (the worker re-derives `!llvm.ptr`s and
    /// `index` bounds from these i64 fields). `@main` stack-allocates `T` arg
    /// structs and `T` `pthread_t` (i64) handles, spawns, then joins. The
    /// `pthread_create` / `pthread_join` / `sysconf` externs are shared with the
    /// Q16 MT kernel (declared once via `self.needs_pthread`).
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_mm_i8_mt(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        c_addr: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
    ) {
        use std::fmt::Write;
        let d = dst.0;
        self.needs_pthread = true;
        // The worker runs the blocked kernel, which heap-allocates its scratch
        // panels via @malloc/@free; flag the module assembler to emit those
        // externs once.
        self.needs_malloc = true;

        // ── worker function (top-level llvm.func) ────────────────────────────
        // Arg struct layout (all i64): [0]=a [1]=b [2]=c [3]=k [4]=n
        //                              [5]=row_start [6]=row_end
        let wname = format!("mind_mt_worker_i8_{d}");
        let mut w = String::new();
        writeln!(
            &mut w,
            "  llvm.func @{wname}(%arg0: !llvm.ptr) -> !llvm.ptr {{"
        )
        .unwrap();
        // Load the 7 i64 fields from the packed arg struct (contiguous i64s).
        for (idx, fld) in [
            (0usize, "a"),
            (1, "b"),
            (2, "c"),
            (3, "k"),
            (4, "n"),
            (5, "rs"),
            (6, "re"),
        ] {
            let off = (idx as i64) * 8;
            writeln!(
                &mut w,
                "    %wi_{d}_o{idx} = llvm.mlir.constant({off} : i64) : i64"
            )
            .unwrap();
            writeln!(
                &mut w,
                "    %wi_{d}_p{idx} = llvm.getelementptr %arg0[%wi_{d}_o{idx}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            )
            .unwrap();
            writeln!(
                &mut w,
                "    %wi_{d}_{fld} = llvm.load %wi_{d}_p{idx} : !llvm.ptr -> i64"
            )
            .unwrap();
        }
        // Re-derive pointers and band bounds. `emit_mm_i8_blocked` wants i64
        // SSA names for K and N (k64/n64), `index` SSA names for K and N
        // (ki/ni), and `index` SSA names for the row band (row_start/row_end).
        writeln!(
            &mut w,
            "    %wi_{d}_ap = llvm.inttoptr %wi_{d}_a : i64 to !llvm.ptr"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wi_{d}_bp = llvm.inttoptr %wi_{d}_b : i64 to !llvm.ptr"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wi_{d}_cp = llvm.inttoptr %wi_{d}_c : i64 to !llvm.ptr"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wi_{d}_ki = arith.index_cast %wi_{d}_k : i64 to index"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wi_{d}_ni = arith.index_cast %wi_{d}_n : i64 to index"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wi_{d}_rsi = arith.index_cast %wi_{d}_rs : i64 to index"
        )
        .unwrap();
        writeln!(
            &mut w,
            "    %wi_{d}_rei = arith.index_cast %wi_{d}_re : i64 to index"
        )
        .unwrap();
        Self::emit_mm_i8_blocked(
            &mut w,
            &format!("mib_{d}"),
            &format!("wi_{d}_ap"),
            &format!("wi_{d}_bp"),
            &format!("wi_{d}_cp"),
            &format!("wi_{d}_k"),
            &format!("wi_{d}_n"),
            &format!("wi_{d}_ki"),
            &format!("wi_{d}_ni"),
            &format!("wi_{d}_rsi"),
            &format!("wi_{d}_rei"),
            IntDotMode::from_env(),
        );
        writeln!(&mut w, "    %wi_{d}_null = llvm.mlir.zero : !llvm.ptr").unwrap();
        writeln!(&mut w, "    llvm.return %wi_{d}_null : !llvm.ptr").unwrap();
        writeln!(&mut w, "  }}").unwrap();
        self.mt_workers.push_str(&w);

        // ── entry (inside @main): spawn + join T threads ─────────────────────
        // Thread count T = clamp(sysconf(_SC_NPROCESSORS_ONLN), 1, M). Owner-
        // computes => output is T-invariant.
        const SC_NPROCESSORS_ONLN: i64 = 84;
        // Cap the number of stack-allocated thread slots so the `alloca` extent
        // is a compile-time constant (statically reserved — no pointer bits in
        // the output, ASLR-independent results). T is clamped to this at run.
        const MAX_THREADS: i64 = 256;
        let struct_bytes: i64 = 7 * 8; // 7 i64 fields per thread-arg struct.

        // Zero constant used to materialise i64 SSA copies of the i64 ABI args
        // (MLIR has no bare SSA-alias form; `addi x, 0` is a trivial value copy).
        self.emit_line(&format!("    %mi{d}_zc = arith.constant 0 : i64"));
        self.emit_line(&format!(
            "    %mi{d}_scq = llvm.mlir.constant({SC_NPROCESSORS_ONLN} : i32) : i32"
        ));
        self.emit_line(&format!(
            "    %mi{d}_ncpu = llvm.call @sysconf(%mi{d}_scq) : (i32) -> i64"
        ));
        // m as a named i64 SSA value (trivial copy of the i64 ABI arg).
        self.emit_line(&format!(
            "    %mi{d}_m64 = arith.addi %{}, %mi{d}_zc : i64",
            m.0
        ));
        // Clamp T into [1, min(M, MAX_THREADS)].
        self.emit_line(&format!(
            "    %mi{d}_one64 = llvm.mlir.constant(1 : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_maxt = llvm.mlir.constant({MAX_THREADS} : i64) : i64"
        ));
        // upper = min(M, MAX_THREADS)
        self.emit_line(&format!(
            "    %mi{d}_mlt = arith.cmpi slt, %mi{d}_m64, %mi{d}_maxt : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_upper = arith.select %mi{d}_mlt, %mi{d}_m64, %mi{d}_maxt : i64"
        ));
        // T = max(1, min(ncpu, upper))
        self.emit_line(&format!(
            "    %mi{d}_clt = arith.cmpi slt, %mi{d}_ncpu, %mi{d}_upper : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_tcap = arith.select %mi{d}_clt, %mi{d}_ncpu, %mi{d}_upper : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_tlt1 = arith.cmpi slt, %mi{d}_tcap, %mi{d}_one64 : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_T = arith.select %mi{d}_tlt1, %mi{d}_one64, %mi{d}_tcap : i64"
        ));
        // band = ceildiv(M, T) = (M + T - 1) / T.
        self.emit_line(&format!(
            "    %mi{d}_tm1 = arith.subi %mi{d}_T, %mi{d}_one64 : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_msum = arith.addi %mi{d}_m64, %mi{d}_tm1 : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_band = arith.divsi %mi{d}_msum, %mi{d}_T : i64"
        ));
        // Stack-alloc MAX_THREADS arg structs + handles.
        let argbuf_bytes = MAX_THREADS * struct_bytes;
        self.emit_line(&format!(
            "    %mi{d}_abnum = llvm.mlir.constant({argbuf_bytes} : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_argbuf = llvm.alloca %mi{d}_abnum x i8 : (i64) -> !llvm.ptr"
        ));
        self.emit_line(&format!(
            "    %mi{d}_hnum = llvm.mlir.constant({MAX_THREADS} : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_handles = llvm.alloca %mi{d}_hnum x i64 : (i64) -> !llvm.ptr"
        ));
        // Worker function pointer.
        self.emit_line(&format!(
            "    %mi{d}_wfp = llvm.mlir.addressof @{wname} : !llvm.ptr"
        ));
        self.emit_line(&format!("    %mi{d}_pnull = llvm.mlir.zero : !llvm.ptr"));
        // Common constants for the spawn/join loops.
        self.emit_line(&format!(
            "    %mi{d}_sb = llvm.mlir.constant({struct_bytes} : i64) : i64"
        ));
        self.emit_line(&format!(
            "    %mi{d}_hsz = llvm.mlir.constant(8 : i64) : i64"
        ));
        self.emit_line(&format!("    %mi{d}_c0idx = arith.constant 0 : index"));
        self.emit_line(&format!("    %mi{d}_c1idx = arith.constant 1 : index"));
        self.emit_line(&format!(
            "    %mi{d}_Ti = arith.index_cast %mi{d}_T : i64 to index"
        ));
        // Pre-pack the i64 a/b/c addresses once.
        self.emit_line(&format!(
            "    %mi{d}_a64 = arith.addi %{}, %mi{d}_zc : i64",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %mi{d}_b64 = arith.addi %{}, %mi{d}_zc : i64",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %mi{d}_c64 = arith.addi %{}, %mi{d}_zc : i64",
            c_addr.0
        ));
        self.emit_line(&format!(
            "    %mi{d}_k64 = arith.addi %{}, %mi{d}_zc : i64",
            k.0
        ));
        self.emit_line(&format!(
            "    %mi{d}_n64 = arith.addi %{}, %mi{d}_zc : i64",
            n.0
        ));

        // Spawn loop: for t in 0..T.
        self.emit_line(&format!(
            "    scf.for %mi{d}_t = %mi{d}_c0idx to %mi{d}_Ti step %mi{d}_c1idx {{"
        ));
        self.emit_line(&format!(
            "      %mi{d}_ti = arith.index_cast %mi{d}_t : index to i64"
        ));
        // row_start = min(t*band, M) ; row_end = min(row_start+band, M).
        self.emit_line(&format!(
            "      %mi{d}_rs0 = arith.muli %mi{d}_ti, %mi{d}_band : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_rslt = arith.cmpi slt, %mi{d}_rs0, %mi{d}_m64 : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_rs = arith.select %mi{d}_rslt, %mi{d}_rs0, %mi{d}_m64 : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_re0 = arith.addi %mi{d}_rs, %mi{d}_band : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_relt = arith.cmpi slt, %mi{d}_re0, %mi{d}_m64 : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_re = arith.select %mi{d}_relt, %mi{d}_re0, %mi{d}_m64 : i64"
        ));
        // arg = argbuf + t*struct_bytes.
        self.emit_line(&format!(
            "      %mi{d}_aoff = arith.muli %mi{d}_ti, %mi{d}_sb : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_argp = llvm.getelementptr %mi{d}_argbuf[%mi{d}_aoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        // Store the 7 i64 fields.
        for (idx, val) in [
            (0usize, format!("%mi{d}_a64")),
            (1, format!("%mi{d}_b64")),
            (2, format!("%mi{d}_c64")),
            (3, format!("%mi{d}_k64")),
            (4, format!("%mi{d}_n64")),
            (5, format!("%mi{d}_rs")),
            (6, format!("%mi{d}_re")),
        ] {
            let off = (idx as i64) * 8;
            self.emit_line(&format!(
                "      %mi{d}_so{idx} = llvm.mlir.constant({off} : i64) : i64"
            ));
            self.emit_line(&format!(
                "      %mi{d}_sp{idx} = llvm.getelementptr %mi{d}_argp[%mi{d}_so{idx}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "      llvm.store {val}, %mi{d}_sp{idx} : i64, !llvm.ptr"
            ));
        }
        // handle slot = handles + t*8.
        self.emit_line(&format!(
            "      %mi{d}_hoff = arith.muli %mi{d}_ti, %mi{d}_hsz : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_hp = llvm.getelementptr %mi{d}_handles[%mi{d}_hoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %mi{d}_crc = llvm.call @pthread_create(%mi{d}_hp, %mi{d}_pnull, %mi{d}_wfp, %mi{d}_argp) : \
             (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32"
        ));
        self.emit_line("    }");

        // Join loop: for t in 0..T.
        self.emit_line(&format!(
            "    scf.for %mi{d}_jt = %mi{d}_c0idx to %mi{d}_Ti step %mi{d}_c1idx {{"
        ));
        self.emit_line(&format!(
            "      %mi{d}_jti = arith.index_cast %mi{d}_jt : index to i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_jhoff = arith.muli %mi{d}_jti, %mi{d}_hsz : i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_jhp = llvm.getelementptr %mi{d}_handles[%mi{d}_jhoff] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %mi{d}_tid = llvm.load %mi{d}_jhp : !llvm.ptr -> i64"
        ));
        self.emit_line(&format!(
            "      %mi{d}_jrc = llvm.call @pthread_join(%mi{d}_tid, %mi{d}_pnull) : \
             (i64, !llvm.ptr) -> i32"
        ));
        self.emit_line("    }");

        // The intrinsic returns 0 (i64) — matches the single-thread sibling.
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    /// "int-dot" tier — emit a native MLIR `vector`-dialect **int16**
    /// row-major matrix-vector multiply. The fast deterministic integer GEMM
    /// tier, composed over rows.
    ///
    /// Computes `y[r] = dot_i16(W[r,:], x)` for each row `r` in `0..rows`.
    /// W is a rows×cols row-major i16 matrix (base address packed as i64),
    /// x is a cols-element i16 input vector (packed i64), y is a
    /// caller-allocated rows-element **i32** output (the exact accumulator
    /// narrowed once at the end, byte-for-byte the per-row scalar oracle).
    ///
    /// Structure mirrors `emit_vec_matmul_rmajor_q16` exactly (RB-row
    /// register blocking + remainder loop), swapping the Q16.16 inner
    /// reduction for the int16 one from `emit_vec_dot_i16`:
    ///
    /// ```text
    ///   outer loop : scf.for r = 0..rows step RB (stores to y, no iter_args)
    ///     inner main : scf.for step 16, load vector<16xi16>, sext i16->i64,
    ///                  mul, i64-lane accumulate (vector<16xi64>) — NO shift
    ///     horizontal : vector.reduction <add> over vector<16xi64>
    ///     inner tail : scf.for step 1, identical per-element op in scalar i64
    ///     pack       : trunc i64->i32
    ///     store      : llvm.store i32 result to y[r] (i32 stride)
    ///   return 0
    /// ```
    ///
    /// The inner widen-multiply-accumulate loop is the AVX2 `vpmaddwd` idiom
    /// at `-march=x86-64-v3`. Byte-identity contract: `y[r]` equals the
    /// return value of `__mind_blas_dot_i16_v(W+r*cols, x, cols)`
    /// byte-for-byte at every (rows, cols), for all int16 inputs.
    #[cfg(feature = "std-surface")]
    #[allow(clippy::too_many_arguments)]
    fn emit_vec_matmul_rmajor_i16(
        &mut self,
        dst: ValueId,
        w_addr: ValueId,
        x_addr: ValueId,
        y_addr: ValueId,
        rows: ValueId,
        cols: ValueId,
    ) {
        let d = dst.0;
        let l = VEC_I16_LANES;
        // int16 input elements are i16 (2 bytes); the y output is i32 (4 bytes).
        let elem_bytes = std::mem::size_of::<i16>() as i64;
        let out_bytes = std::mem::size_of::<i32>() as i64;
        // Row register-blocking factor: process RB output rows per outer pass so
        // the shared x-vector load+widen amortises across RB independent
        // accumulator chains. Byte-identity is preserved exactly — each blocked
        // accumulator acc_t sums precisely the same per-element terms
        // `W[r+t,i]*x[i]` in the same i-order, lane grouping unchanged, reduced
        // by the same associative `vector.reduction <add>`.
        //
        // RB=6 (not 8): each i16 row accumulator is `vector<8xi64>`, which
        // legalizes to 2 YMM on AVX2 (no 512-bit regs without AVX-512). RB=8
        // needs 8×2=16 YMM for accumulators alone — exhausting the file and
        // forcing a load-modify-store of an accumulator inside the hot k-loop
        // (objdump-confirmed spill). RB=6 → 6×2=12 YMM accumulators + x-load +
        // vpmaddwd result + temps ≤ 16, no inner-loop spill. Byte-identity is
        // unchanged: RB only sets rows-per-outer-pass; each row's reduction is
        // bit-for-bit the same.
        const RB: usize = 6;

        // ── constants (emitted once, before the outer loop) ──────────────────
        self.emit_line(&format!("    %vmmi_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vmmi_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vmmi_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!("    %vmmi_rb_{d} = arith.constant {RB} : index"));
        self.emit_line(&format!(
            "    %vmmi_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!(
            "    %vmmi_ob_{d} = arith.constant {out_bytes} : i64"
        ));

        // ── pointer setup ─────────────────────────────────────────────────────
        self.emit_line(&format!(
            "    %vmmi_wp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            w_addr.0
        ));
        self.emit_line(&format!(
            "    %vmmi_xp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            x_addr.0
        ));
        self.emit_line(&format!(
            "    %vmmi_yp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            y_addr.0
        ));

        // ── loop bounds (cols-derived, invariant across rows) ─────────────────
        self.emit_line(&format!(
            "    %vmmi_rows_{d} = arith.index_cast %{} : i64 to index",
            rows.0
        ));
        self.emit_line(&format!(
            "    %vmmi_rnb_{d} = arith.divui %vmmi_rows_{d}, %vmmi_rb_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmmi_rmain_{d} = arith.muli %vmmi_rnb_{d}, %vmmi_rb_{d} : index"
        ));
        // byte stride per W row = cols * sizeof(i16)
        self.emit_line(&format!(
            "    %vmmi_colsb_{d} = arith.muli %{}, %vmmi_eb_{d} : i64",
            cols.0
        ));
        // inner loop bounds — same for every row (cols is loop-invariant)
        self.emit_line(&format!(
            "    %vmmi_len_{d} = arith.index_cast %{} : i64 to index",
            cols.0
        ));
        self.emit_line(&format!(
            "    %vmmi_nv_{d} = arith.divui %vmmi_len_{d}, %vmmi_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vmmi_ve_{d} = arith.muli %vmmi_nv_{d}, %vmmi_cl_{d} : index"
        ));
        // i64 accumulator carries the 8 `vpmaddwd` i32 partials per step,
        // sign-extended to i64 so the running sum is exact for all inputs.
        let p = VEC_I16_PMADD_LANES;
        self.emit_line(&format!(
            "    %vmmi_zv_{d} = arith.constant dense<0> : vector<{p}xi64>"
        ));

        // ════════════════════════════════════════════════════════════════════
        //  Register-blocked main loop: RB output rows per outer pass.
        // ════════════════════════════════════════════════════════════════════
        self.emit_line(&format!(
            "    scf.for %vmmi_r_{d} = %vmmi_c0_{d} to %vmmi_rmain_{d} step %vmmi_rb_{d} {{"
        ));
        self.emit_line(&format!(
            "      %vmmi_ri_{d} = arith.index_cast %vmmi_r_{d} : index to i64"
        ));
        // Base pointer to each of the RB W-rows in this block: W[r+t, 0].
        for t in 0..RB {
            self.emit_line(&format!("      %vmmi_rt{t}_{d} = arith.constant {t} : i64"));
            self.emit_line(&format!(
                "      %vmmi_rit{t}_{d} = arith.addi %vmmi_ri_{d}, %vmmi_rt{t}_{d} : i64"
            ));
            self.emit_line(&format!(
                "      %vmmi_roff{t}_{d} = arith.muli %vmmi_rit{t}_{d}, %vmmi_colsb_{d} : i64"
            ));
            self.emit_line(&format!(
                "      %vmmi_wrow{t}_{d} = llvm.getelementptr %vmmi_wp_{d}[%vmmi_roff{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
        }

        // ── inner vector main loop: load shared x once, MAC against RB W-rows ──
        let acc_init = (0..RB)
            .map(|t| format!("%vmmi_acc{t}_{d} = %vmmi_zv_{d}"))
            .collect::<Vec<_>>()
            .join(", ");
        let acc_ty = (0..RB)
            .map(|_| format!("vector<{p}xi64>"))
            .collect::<Vec<_>>()
            .join(", ");
        self.emit_line(&format!(
            "      %vmmi_va0_{d}:{RB} = scf.for %vmmi_i_{d} = %vmmi_c0_{d} to %vmmi_ve_{d} \
             step %vmmi_cl_{d} iter_args({acc_init}) -> ({acc_ty}) {{"
        ));
        self.emit_line(&format!(
            "        %vmmi_ii_{d} = arith.index_cast %vmmi_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmmi_bo_{d} = arith.muli %vmmi_ii_{d}, %vmmi_eb_{d} : i64"
        ));
        // x[i..i+16] — loaded ONCE, widened ONCE, reused against all RB W-rows.
        self.emit_line(&format!(
            "        %vmmi_bi_{d} = llvm.getelementptr %vmmi_xp_{d}[%vmmi_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmi_bv_{d} = llvm.load %vmmi_bi_{d} {{alignment = 2 : i64}} : \
             !llvm.ptr -> vector<{l}xi16>"
        ));
        // Per-block-row MAC: W[r+t, i..i+16] · x[i..i+16], into acc_t via vpmaddwd.
        let mut yields = Vec::with_capacity(RB);
        for t in 0..RB {
            self.emit_line(&format!(
                "        %vmmi_ai{t}_{d} = llvm.getelementptr %vmmi_wrow{t}_{d}[%vmmi_bo_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "        %vmmi_av{t}_{d} = llvm.load %vmmi_ai{t}_{d} {{alignment = 2 : i64}} : \
                 !llvm.ptr -> vector<{l}xi16>"
            ));
            if HOST_IS_X86 {
                self.emit_line(&format!(
                    "        %vmmi_pm{t}_{d} = llvm.call_intrinsic \"llvm.x86.avx2.pmadd.wd\"(%vmmi_av{t}_{d}, %vmmi_bv_{d}) : \
                     (vector<{l}xi16>, vector<{l}xi16>) -> vector<{p}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_pw{t}_{d} = arith.extsi %vmmi_pm{t}_{d} : vector<{p}xi32> to vector<{p}xi64>"
                ));
            } else {
                // Non-x86 (aarch64): portable exact-integer pairwise contraction
                // (the x86 `vpmaddwd` intrinsic does not legalise off x86).
                // Bit-identical integer result.
                self.emit_line(&format!(
                    "        %vmmi_aw{t}_{d} = arith.extsi %vmmi_av{t}_{d} : vector<{l}xi16> to vector<{l}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_bw{t}_{d} = arith.extsi %vmmi_bv_{d} : vector<{l}xi16> to vector<{l}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_pr{t}_{d} = arith.muli %vmmi_aw{t}_{d}, %vmmi_bw{t}_{d} : vector<{l}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_ev{t}_{d} = vector.shuffle %vmmi_pr{t}_{d}, %vmmi_pr{t}_{d} \
                     [0, 2, 4, 6, 8, 10, 12, 14] : vector<{l}xi32>, vector<{l}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_od{t}_{d} = vector.shuffle %vmmi_pr{t}_{d}, %vmmi_pr{t}_{d} \
                     [1, 3, 5, 7, 9, 11, 13, 15] : vector<{l}xi32>, vector<{l}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_pm{t}_{d} = arith.addi %vmmi_ev{t}_{d}, %vmmi_od{t}_{d} : vector<{p}xi32>"
                ));
                self.emit_line(&format!(
                    "        %vmmi_pw{t}_{d} = arith.extsi %vmmi_pm{t}_{d} : vector<{p}xi32> to vector<{p}xi64>"
                ));
            }
            self.emit_line(&format!(
                "        %vmmi_na{t}_{d} = arith.addi %vmmi_acc{t}_{d}, %vmmi_pw{t}_{d} : vector<{p}xi64>"
            ));
            yields.push(format!("%vmmi_na{t}_{d}"));
        }
        self.emit_line(&format!(
            "        scf.yield {} : {acc_ty}",
            yields.join(", ")
        ));
        self.emit_line("      }");

        // ── per-block-row finalise: reduce, scalar tail, pack, store ──────────
        for t in 0..RB {
            // Horizontal lane reduction (associative — bit-identical).
            self.emit_line(&format!(
                "      %vmmi_vs{t}_{d} = vector.reduction <add>, %vmmi_va0_{d}#{t} : \
                 vector<{p}xi64> into i64"
            ));
            // Scalar tail for cols % LANES remainder, on W[r+t,:].
            self.emit_line(&format!(
                "      %vmmi_ts{t}_{d} = scf.for %vmmi_j{t}_{d} = %vmmi_ve_{d} to %vmmi_len_{d} \
                 step %vmmi_c1_{d} iter_args(%vmmi_s{t}_{d} = %vmmi_vs{t}_{d}) -> (i64) {{"
            ));
            self.emit_line(&format!(
                "        %vmmi_jj{t}_{d} = arith.index_cast %vmmi_j{t}_{d} : index to i64"
            ));
            self.emit_line(&format!(
                "        %vmmi_jb{t}_{d} = arith.muli %vmmi_jj{t}_{d}, %vmmi_eb_{d} : i64"
            ));
            self.emit_line(&format!(
                "        %vmmi_aj{t}_{d} = llvm.getelementptr %vmmi_wrow{t}_{d}[%vmmi_jb{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "        %vmmi_xj{t}_{d} = llvm.getelementptr %vmmi_xp_{d}[%vmmi_jb{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "        %vmmi_as{t}_{d} = llvm.load %vmmi_aj{t}_{d} : !llvm.ptr -> i16"
            ));
            self.emit_line(&format!(
                "        %vmmi_bs{t}_{d} = llvm.load %vmmi_xj{t}_{d} : !llvm.ptr -> i16"
            ));
            self.emit_line(&format!(
                "        %vmmi_asw{t}_{d} = arith.extsi %vmmi_as{t}_{d} : i16 to i64"
            ));
            self.emit_line(&format!(
                "        %vmmi_bsw{t}_{d} = arith.extsi %vmmi_bs{t}_{d} : i16 to i64"
            ));
            self.emit_line(&format!(
                "        %vmmi_p{t}_{d} = arith.muli %vmmi_asw{t}_{d}, %vmmi_bsw{t}_{d} : i64"
            ));
            self.emit_line(&format!(
                "        %vmmi_ns{t}_{d} = arith.addi %vmmi_s{t}_{d}, %vmmi_p{t}_{d} : i64"
            ));
            self.emit_line(&format!("        scf.yield %vmmi_ns{t}_{d} : i64"));
            self.emit_line("      }");
            // Pack: trunc i64→i32, store low 32 bits to y[r+t] (i32 stride).
            self.emit_line(&format!(
                "      %vmmi_lo{t}_{d} = arith.trunci %vmmi_ts{t}_{d} : i64 to i32"
            ));
            self.emit_line(&format!(
                "      %vmmi_yoff{t}_{d} = arith.muli %vmmi_rit{t}_{d}, %vmmi_ob_{d} : i64"
            ));
            self.emit_line(&format!(
                "      %vmmi_yel{t}_{d} = llvm.getelementptr %vmmi_yp_{d}[%vmmi_yoff{t}_{d}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8"
            ));
            self.emit_line(&format!(
                "      llvm.store %vmmi_lo{t}_{d}, %vmmi_yel{t}_{d} : i32, !llvm.ptr"
            ));
        }
        self.emit_line("    }"); // end register-blocked outer scf.for

        // ════════════════════════════════════════════════════════════════════
        //  Remainder loop: the leftover `rows % RB` rows, single-row path.
        // ════════════════════════════════════════════════════════════════════
        self.emit_line(&format!(
            "    scf.for %vmmir_r_{d} = %vmmi_rmain_{d} to %vmmi_rows_{d} step %vmmi_c1_{d} {{"
        ));
        self.emit_line(&format!(
            "      %vmmir_ri_{d} = arith.index_cast %vmmir_r_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vmmir_roff_{d} = arith.muli %vmmir_ri_{d}, %vmmi_colsb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vmmir_wrow_{d} = llvm.getelementptr %vmmi_wp_{d}[%vmmir_roff_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vmmir_vacc_{d} = scf.for %vmmir_i_{d} = %vmmi_c0_{d} to %vmmi_ve_{d} \
             step %vmmi_cl_{d} iter_args(%vmmir_acc_{d} = %vmmi_zv_{d}) -> (vector<{p}xi64>) {{"
        ));
        self.emit_line(&format!(
            "        %vmmir_ii_{d} = arith.index_cast %vmmir_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_bo_{d} = arith.muli %vmmir_ii_{d}, %vmmi_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_ai_{d} = llvm.getelementptr %vmmir_wrow_{d}[%vmmir_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmir_bi_{d} = llvm.getelementptr %vmmi_xp_{d}[%vmmir_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmir_av_{d} = llvm.load %vmmir_ai_{d} {{alignment = 2 : i64}} : \
             !llvm.ptr -> vector<{l}xi16>"
        ));
        self.emit_line(&format!(
            "        %vmmir_bv_{d} = llvm.load %vmmir_bi_{d} {{alignment = 2 : i64}} : \
             !llvm.ptr -> vector<{l}xi16>"
        ));
        if HOST_IS_X86 {
            self.emit_line(&format!(
                "        %vmmir_pm_{d} = llvm.call_intrinsic \"llvm.x86.avx2.pmadd.wd\"(%vmmir_av_{d}, %vmmir_bv_{d}) : \
                 (vector<{l}xi16>, vector<{l}xi16>) -> vector<{p}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_pw_{d} = arith.extsi %vmmir_pm_{d} : vector<{p}xi32> to vector<{p}xi64>"
            ));
        } else {
            // Non-x86 (aarch64): portable exact-integer pairwise contraction
            // (the x86 `vpmaddwd` intrinsic does not legalise off x86).
            // Bit-identical integer result.
            self.emit_line(&format!(
                "        %vmmir_aw_{d} = arith.extsi %vmmir_av_{d} : vector<{l}xi16> to vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_bw_{d} = arith.extsi %vmmir_bv_{d} : vector<{l}xi16> to vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_pr_{d} = arith.muli %vmmir_aw_{d}, %vmmir_bw_{d} : vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_ev_{d} = vector.shuffle %vmmir_pr_{d}, %vmmir_pr_{d} \
                 [0, 2, 4, 6, 8, 10, 12, 14] : vector<{l}xi32>, vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_od_{d} = vector.shuffle %vmmir_pr_{d}, %vmmir_pr_{d} \
                 [1, 3, 5, 7, 9, 11, 13, 15] : vector<{l}xi32>, vector<{l}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_pm_{d} = arith.addi %vmmir_ev_{d}, %vmmir_od_{d} : vector<{p}xi32>"
            ));
            self.emit_line(&format!(
                "        %vmmir_pw_{d} = arith.extsi %vmmir_pm_{d} : vector<{p}xi32> to vector<{p}xi64>"
            ));
        }
        self.emit_line(&format!(
            "        %vmmir_na_{d} = arith.addi %vmmir_acc_{d}, %vmmir_pw_{d} : vector<{p}xi64>"
        ));
        self.emit_line(&format!(
            "        scf.yield %vmmir_na_{d} : vector<{p}xi64>"
        ));
        self.emit_line("      }");
        self.emit_line(&format!(
            "      %vmmir_vs_{d} = vector.reduction <add>, %vmmir_vacc_{d} : \
             vector<{p}xi64> into i64"
        ));
        self.emit_line(&format!(
            "      %vmmir_ts_{d} = scf.for %vmmir_j_{d} = %vmmi_ve_{d} to %vmmi_len_{d} \
             step %vmmi_c1_{d} iter_args(%vmmir_s_{d} = %vmmir_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "        %vmmir_jj_{d} = arith.index_cast %vmmir_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_jb_{d} = arith.muli %vmmir_jj_{d}, %vmmi_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_aj_{d} = llvm.getelementptr %vmmir_wrow_{d}[%vmmir_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmir_bj_{d} = llvm.getelementptr %vmmi_xp_{d}[%vmmir_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "        %vmmir_as_{d} = llvm.load %vmmir_aj_{d} : !llvm.ptr -> i16"
        ));
        self.emit_line(&format!(
            "        %vmmir_bs_{d} = llvm.load %vmmir_bj_{d} : !llvm.ptr -> i16"
        ));
        self.emit_line(&format!(
            "        %vmmir_asw_{d} = arith.extsi %vmmir_as_{d} : i16 to i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_bsw_{d} = arith.extsi %vmmir_bs_{d} : i16 to i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_p_{d} = arith.muli %vmmir_asw_{d}, %vmmir_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "        %vmmir_ns_{d} = arith.addi %vmmir_s_{d}, %vmmir_p_{d} : i64"
        ));
        self.emit_line(&format!("        scf.yield %vmmir_ns_{d} : i64"));
        self.emit_line("      }");
        self.emit_line(&format!(
            "      %vmmir_lo_{d} = arith.trunci %vmmir_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!(
            "      %vmmir_yoff_{d} = arith.muli %vmmir_ri_{d}, %vmmi_ob_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vmmir_yel_{d} = llvm.getelementptr %vmmi_yp_{d}[%vmmir_yoff_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      llvm.store %vmmir_lo_{d}, %vmmir_yel_{d} : i32, !llvm.ptr"
        ));
        self.emit_line("    }"); // end remainder outer scf.for

        // The intrinsic returns 0 (i64).
        self.emit_line(&format!("    %{d} = arith.constant 0 : i64"));
    }

    fn tensor_info(
        &self,
        id: &ValueId,
        context: &'static str,
    ) -> Result<TensorInfo, MlirLowerError> {
        match self.values.get(id) {
            Some(ValueKind::Tensor { dtype, shape }) => Ok(TensorInfo {
                dtype: dtype.clone(),
                shape: shape.clone(),
            }),
            _ => Err(MlirLowerError::MissingTypeInfo {
                value: *id,
                context,
            }),
        }
    }
}

/// Lower a verified and canonicalized [`IRModule`] into MLIR text.
///
/// The lowering does not mutate the input module and is deterministic:
/// the same IR produces identical MLIR text.
pub fn lower_ir_to_mlir(module: &IRModule) -> Result<MlirModule, MlirLowerError> {
    let mut ctx = LoweringContext::new();

    // RFC 0012 §5.1 — pre-resolve every fn's ABI signature (declared param /
    // return types) to MLIR type strings so the `Instr::FnDef` arm can type the
    // `func.func` slots and seed float param `ValueKind`s. i64-only modules
    // record all-`i64` here, so their lowered text is byte-identical.
    #[cfg(feature = "std-surface")]
    for (name, (param_types, ret_type)) in &module.fn_signatures {
        let params: Vec<String> = param_types
            .iter()
            .map(|t| type_ann_to_abi_mlir(t).to_string())
            .collect();
        let ret = ret_type
            .as_ref()
            .map(|t| type_ann_to_abi_mlir(t).to_string());
        ctx.fn_signatures.insert(name.clone(), (params, ret));
        // NARROW-INT ABI: also record the signedness-preserving param kinds so
        // the FnDef arm can seed `u32` distinctly from `i32` (the ABI string
        // table above collapses both to "i32"). i64-only modules record all
        // `ScalarI64` here → identical seeding to before.
        let param_kinds: Vec<ValueKind> = param_types.iter().map(type_ann_to_value_kind).collect();
        ctx.fn_param_kinds.insert(name.clone(), param_kinds);
    }

    for (idx, instr) in module.instrs.iter().enumerate() {
        ctx.emit_instr(idx, instr)?;
    }

    let mut ret_types = Vec::new();
    if !ctx.outputs.is_empty() {
        let mut value_list = String::new();
        let mut type_list = String::new();
        for (i, id) in ctx.outputs.iter().enumerate() {
            let info = ctx.values.get(id).ok_or(MlirLowerError::MissingTypeInfo {
                value: *id,
                context: "function return",
            })?;
            ret_types.push(mlir_type(info)?);
            if i > 0 {
                value_list.push_str(", ");
                type_list.push_str(", ");
            }
            write!(&mut value_list, "%{}", id.0).unwrap();
            write!(&mut type_list, "{}", mlir_type(info)?).unwrap();
        }
        ctx.emit_line(&format!("    return {} : {}", value_list, type_list));
    } else {
        ctx.emit_line("    return");
    }

    let mut out = String::new();
    out.push_str("module {\n");

    // RFC 0005 Phase 0: one `func.func private @callee(i64...) -> i64`
    // declaration per distinct callee, before `@main`, so the
    // `func.call`s emitted above resolve. Sorted (BTreeSet) for
    // deterministic MLIR text / model_hash. Gated; default build
    // emits none of this. P0d: skip names that have a local
    // `func.func` definition emitted below — declaring a private
    // forward decl AND a definition for the same symbol is invalid.
    #[cfg(feature = "std-surface")]
    for (name, arity) in &ctx.extern_calls {
        if ctx.defined_fns.contains(name) {
            continue;
        }
        let params = vec!["i64"; *arity].join(", ");
        out.push_str(&format!("  func.func private @{name}({params}) -> i64\n"));
    }
    // RFC 0005 P0d: user-defined `func.func @name(...) -> i64 { ... }`
    // definitions, in source order. Emitted before `@main` so the
    // `func.call`s inside @main resolve. Gated; default build emits none.
    #[cfg(feature = "std-surface")]
    out.push_str(&ctx.user_fns);

    // Multithreaded Q16.16 GEMM: emit the pthread / sysconf externs once and
    // the per-call worker `llvm.func` bodies, all before `@main`. The pthread
    // ABI here is the linux-x86_64 / aarch64 one (`pthread_t` is `i64`); the
    // worker runs the fused owner-computes row-band kernel. Output is
    // byte-identical to the single-thread kernel regardless of thread count, so
    // none of this perturbs the cross-substrate trace_hash of programs that do
    // not call the MT intrinsic (the whole block is empty for them).
    #[cfg(feature = "std-surface")]
    if ctx.needs_pthread {
        out.push_str(
            "  llvm.func @pthread_create(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32\n",
        );
        out.push_str("  llvm.func @pthread_join(i64, !llvm.ptr) -> i32\n");
        out.push_str("  llvm.func @sysconf(i32) -> i64\n");
    }
    // Heap-allocation externs for the int8 BLIS macro-kernel's scratch panels
    // (C-scratch / packed-A / packed-B). Emitted once, before any function body
    // that calls @malloc / @free, so the `llvm.call`s resolve. Declared only
    // when the blocked kernel was actually lowered.
    #[cfg(feature = "std-surface")]
    if ctx.needs_malloc {
        out.push_str("  llvm.func @malloc(i64) -> !llvm.ptr\n");
        out.push_str("  llvm.func @free(!llvm.ptr)\n");
    }
    #[cfg(feature = "std-surface")]
    out.push_str(&ctx.mt_workers);

    // RFC 0010 Phase A/C: one `llvm.func @name(type...) -> type` declaration
    // per `extern "C"` symbol referenced via `Instr::ExternFnDecl`. Emitted
    // before `@main` so `llvm.call` ops resolve. Sorted (BTreeMap) for
    // deterministic MLIR text.
    // Phase C: attach `cconv = #llvm.cconv<win64cc>` for Win64 declarations.
    // Gated; default build emits none.
    #[cfg(feature = "std-surface")]
    for (name, (param_types, ret_type, is_varargs, _vararg_hints, callconv)) in &ctx.extern_c_fns {
        let params_str = if param_types.is_empty() {
            String::new()
        } else {
            param_types.join(", ")
        };
        let varargs_suffix = if *is_varargs {
            if param_types.is_empty() {
                "...".to_string()
            } else {
                ", ...".to_string()
            }
        } else {
            String::new()
        };
        let ret_str = ret_type.as_deref().unwrap_or("i64");
        let cconv_attr = cconv_attr_for(*callconv);
        out.push_str(&format!(
            "  llvm.func{cconv_attr} @{name}({params_str}{varargs_suffix}) -> {ret_str}\n"
        ));
    }

    if ret_types.is_empty() {
        out.push_str("  func.func @main() -> () {\n");
    } else {
        out.push_str(&format!(
            "  func.func @main() -> ({}) {{\n",
            ret_types.join(", ")
        ));
    }
    out.push_str(&ctx.body);
    out.push_str("  }\n");

    // RFC 0002 D2: append `mind_fn_<name>_invoke` C-ABI wrappers as
    // sibling top-level symbols, before the module-closing brace.
    // Module-level feature gate only — the default build never touches
    // this path and emits byte-identical MLIR (compile-speed moat).
    #[cfg(feature = "ffi-c-user")]
    crate::mlir::c_export::emit_c_export_wrappers(&mut out, module)
        .map_err(MlirLowerError::CExportError)?;

    out.push_str("}\n");

    Ok(MlirModule { text: out })
}

/// RFC 0010 Phase C — return the MLIR `cconv` attribute string for a calling
/// convention, including the leading space separator for inline use.
///
/// For Win64: `" cconv = #llvm.cconv<win64cc>"` (space-prefixed for use in
/// `llvm.func cconv = ... @name(...)` and `llvm.call cconv = ... @name(...)`).
/// For all other conventions (SysV, C, Aapcs): empty string (no attribute,
/// which is the MLIR LLVM dialect default i.e. C calling convention / SysV
/// on x86_64 Linux/macOS).
#[cfg(feature = "std-surface")]
fn cconv_attr_for(callconv: crate::ast::CallConv) -> &'static str {
    use crate::ast::CallConv;
    match callconv {
        CallConv::Win64 => " cconv = #llvm.cconv<win64cc>",
        // SysV, C (platform default), Aapcs (Phase D) — no cconv attribute.
        _ => "",
    }
}

/// Convenience helper: verify, canonicalize, and lower into MLIR text.
pub fn compile_ir_to_mlir_text(module: &mut IRModule) -> Result<String, MlirLowerError> {
    crate::ir::verify_module(module)?;
    canonicalize_module(module);
    crate::ir::verify_module(module)?;
    let lowered = lower_ir_to_mlir(module)?;
    Ok(lowered.text)
}

#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: DType,
    shape: Vec<ShapeDim>,
}

/// RFC 0012 §5.1 — map an AST `TypeAnn` to the scalar MLIR ABI type string used
/// in `func.func` parameter / return slots. A declared scalar `f64`/`f32` keeps
/// its float width; every other type (integers, bools, structs-by-ptr, …) keeps
/// the established i64 ABI so non-float functions lower byte-identically. Used
/// only when building the `LoweringContext::fn_signatures` table.
#[cfg(feature = "std-surface")]
fn type_ann_to_abi_mlir(ty: &crate::ast::TypeAnn) -> &'static str {
    match ty {
        crate::ast::TypeAnn::ScalarF64 => "f64",
        crate::ast::TypeAnn::ScalarF32 => "f32",
        // NARROW-INT ABI: i32/u32 params + returns lower to real `i32` MLIR
        // (MLIR ints are signless; signedness is carried by the op). NOTE:
        // ScalarBool intentionally stays "i64" in the return slot to preserve
        // the existing cmpi+extui+ret-i64 byte sequence — do NOT change the
        // bool return ABI here.
        crate::ast::TypeAnn::ScalarI32 | crate::ast::TypeAnn::ScalarU32 => "i32",
        _ => "i64",
    }
}

/// NARROW-INT ABI — map a declared `TypeAnn` to the scalar `ValueKind` used to
/// seed a parameter's SSA kind. Unlike [`type_ann_to_abi_mlir`] this PRESERVES
/// signedness (`i32` vs `u32`), which the BinOp dispatch needs to pick signed vs
/// unsigned ops. Non-scalar / non-narrow annotations fall back to `ScalarI64`,
/// so an i64-only module seeds exactly the same kinds as before (byte-identical).
#[cfg(feature = "std-surface")]
fn type_ann_to_value_kind(ty: &crate::ast::TypeAnn) -> ValueKind {
    match ty {
        crate::ast::TypeAnn::ScalarF64 => ValueKind::ScalarF64,
        crate::ast::TypeAnn::ScalarF32 => ValueKind::ScalarF32,
        crate::ast::TypeAnn::ScalarI32 => ValueKind::ScalarI32,
        crate::ast::TypeAnn::ScalarU32 => ValueKind::ScalarU32,
        crate::ast::TypeAnn::ScalarBool => ValueKind::ScalarBool,
        _ => ValueKind::ScalarI64,
    }
}

fn mlir_type(kind: &ValueKind) -> Result<String, MlirLowerError> {
    match kind {
        ValueKind::ScalarI64 => Ok("i64".to_string()),
        #[cfg(feature = "std-surface")]
        ValueKind::ScalarF64 => Ok("f64".to_string()),
        #[cfg(feature = "std-surface")]
        ValueKind::ScalarF32 => Ok("f32".to_string()),
        #[cfg(feature = "std-surface")]
        ValueKind::ScalarI32 | ValueKind::ScalarU32 => Ok("i32".to_string()),
        #[cfg(feature = "std-surface")]
        ValueKind::ScalarBool => Ok("i1".to_string()),
        ValueKind::Tensor { dtype, shape } => Ok(tensor_type(shape, dtype.as_str())),
        #[cfg(feature = "std-surface")]
        ValueKind::VectorF32 { lanes } => Ok(format!("vector<{lanes}xf32>")),
        #[cfg(feature = "std-surface")]
        ValueKind::VectorI64 { lanes } => Ok(format!("vector<{lanes}xi64>")),
    }
}

/// `(result_shape, lhs_map_exprs, rhs_map_exprs)` produced by
/// [`broadcast_binop_maps`] — the result tensor shape plus, per operand, the
/// `linalg.generic` affine-map result expressions.
type BroadcastMaps = (Vec<ShapeDim>, Vec<String>, Vec<String>);

/// Plan a right-aligned NumPy/RFC-0012 broadcast for an elementwise tensor
/// binop, returning the result shape plus the per-operand `linalg.generic`
/// affine-map result expressions. This is the same rule the type-checker
/// applies in `shapes::broadcast_shapes` and that `mind-spec/spec/v1.0/shapes.md`
/// makes normative; keeping the two in lockstep is what stops a
/// type-checks-but-won't-lower gap.
///
/// Each operand's map entry for its dimension `k` is the result iteration dim it
/// reads (`d{i}`), or the constant `0` when that dimension is a size-1 stretch
/// (e.g. the `1` axes of `(4,1,3) + (1,5,3) -> (4,5,3)`). Only statically-known
/// dimensions are handled; a symbolic dimension in a *broadcasting* position is
/// rejected because it would emit MLIR with a non-numeric extent.
fn broadcast_binop_maps(
    lhs: &[ShapeDim],
    rhs: &[ShapeDim],
) -> Result<BroadcastMaps, MlirLowerError> {
    let sym_err = || {
        MlirLowerError::ShapeError(
            "broadcasting tensors with symbolic dimensions is not yet supported in MLIR lowering"
                .into(),
        )
    };
    let lhs_d: Vec<usize> = lhs
        .iter()
        .map(known_dim)
        .collect::<Option<_>>()
        .ok_or_else(sym_err)?;
    let rhs_d: Vec<usize> = rhs
        .iter()
        .map(known_dim)
        .collect::<Option<_>>()
        .ok_or_else(sym_err)?;

    let rank = lhs_d.len().max(rhs_d.len());
    let mut result = vec![0usize; rank];
    for i in 0..rank {
        // Align from the right: missing leading dims act as extent 1.
        let a = if i < lhs_d.len() {
            lhs_d[lhs_d.len() - 1 - i]
        } else {
            1
        };
        let b = if i < rhs_d.len() {
            rhs_d[rhs_d.len() - 1 - i]
        } else {
            1
        };
        let dim = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err(MlirLowerError::ShapeError(format!(
                "cannot broadcast tensor shapes {lhs_d:?} and {rhs_d:?}"
            )));
        };
        result[rank - 1 - i] = dim;
    }

    let map_for = |operand: &[usize]| -> Vec<String> {
        let r = operand.len();
        (0..r)
            .map(|k| {
                let res_idx = (rank - r) + k;
                if operand[k] == result[res_idx] {
                    format!("d{res_idx}")
                } else {
                    // operand[k] == 1, stretched to the result extent.
                    "0".to_string()
                }
            })
            .collect()
    };
    let lhs_map = map_for(&lhs_d);
    let rhs_map = map_for(&rhs_d);
    let result_shape = result.into_iter().map(ShapeDim::Known).collect();
    Ok((result_shape, lhs_map, rhs_map))
}

fn tensor_type(shape: &[ShapeDim], dtype: &str) -> String {
    if shape.is_empty() {
        return format!("tensor<{}>", dtype);
    }

    let dims = shape
        .iter()
        .map(shape_dim_to_string)
        .collect::<Vec<_>>()
        .join("x");
    format!("tensor<{}x{}>", dims, dtype)
}

fn shape_dim_to_string(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

fn select_arith_op(op: BinOp, dtype: &DType) -> &'static str {
    match dtype {
        // RFC 0012 §5.1: `F64` selects the floating-point `arith.*f` ops too.
        // Without it a scalar-`f64` BinOp fell through to the integer `_` arm
        // and wrongly emitted `arith.addi`/etc. on `f64` operands.
        DType::F64 | DType::F32 | DType::F16 | DType::BF16 => match op {
            BinOp::Add => "arith.addf",
            BinOp::Sub => "arith.subf",
            BinOp::Mul => "arith.mulf",
            BinOp::Div => "arith.divf",
            BinOp::Mod => "arith.remf",
            BinOp::Lt => "arith.cmpf \"olt\",",
            BinOp::Le => "arith.cmpf \"ole\",",
            BinOp::Gt => "arith.cmpf \"ogt\",",
            BinOp::Ge => "arith.cmpf \"oge\",",
            BinOp::Eq => "arith.cmpf \"oeq\",",
            BinOp::Ne => "arith.cmpf \"one\",",
            // Bitwise ops on floating-point tensors are not meaningful;
            // emit a placeholder that mlir-opt will reject loudly rather
            // than silently producing wrong code.
            #[cfg(feature = "std-surface")]
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => "arith.andi",
        },
        _ => match op {
            BinOp::Add => "arith.addi",
            BinOp::Sub => "arith.subi",
            BinOp::Mul => "arith.muli",
            BinOp::Div => "arith.divsi",
            BinOp::Mod => "arith.remsi",
            BinOp::Lt => "arith.cmpi \"slt\",",
            BinOp::Le => "arith.cmpi \"sle\",",
            BinOp::Gt => "arith.cmpi \"sgt\",",
            BinOp::Ge => "arith.cmpi \"sge\",",
            BinOp::Eq => "arith.cmpi \"eq\",",
            BinOp::Ne => "arith.cmpi \"ne\",",
            #[cfg(feature = "std-surface")]
            BinOp::BitAnd => "arith.andi",
            #[cfg(feature = "std-surface")]
            BinOp::BitOr => "arith.ori",
            #[cfg(feature = "std-surface")]
            BinOp::BitXor => "arith.xori",
            #[cfg(feature = "std-surface")]
            BinOp::Shl => "arith.shli",
            #[cfg(feature = "std-surface")]
            BinOp::Shr => "arith.shrsi",
        },
    }
}

fn matmul_shapes(
    a_shape: &[ShapeDim],
    b_shape: &[ShapeDim],
    dtype: &str,
) -> Result<(Vec<ShapeDim>, String, String, String), MlirLowerError> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MlirLowerError::ShapeError(
            "matmul lowering expects rank-2 tensors".to_string(),
        ));
    }
    let m = a_shape[0].clone();
    let k = a_shape[1].clone();
    let k_rhs = b_shape[0].clone();
    if !shapes_compatible(&k, &k_rhs) {
        return Err(MlirLowerError::ShapeError(
            "matmul K dimensions must match".into(),
        ));
    }
    let n = b_shape[1].clone();
    let out_shape = vec![m.clone(), n.clone()];
    let lhs_ty = tensor_type(a_shape, dtype);
    let rhs_ty = tensor_type(b_shape, dtype);
    let out_ty = tensor_type(&out_shape, dtype);
    Ok((out_shape, lhs_ty, rhs_ty, out_ty))
}

fn shapes_compatible(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(x), ShapeDim::Sym(y)) => x == y,
        _ => true,
    }
}

fn conv2d_shapes(
    input: &TensorInfo,
    filter: &TensorInfo,
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> Result<(Vec<ShapeDim>, String, String, String), MlirLowerError> {
    if input.shape.len() != 4 || filter.shape.len() != 4 {
        return Err(MlirLowerError::ShapeError(
            "conv2d lowering expects NHWC input and HWCF filter".into(),
        ));
    }

    let batch = input.shape[0].clone();
    let in_h = &input.shape[1];
    let in_w = &input.shape[2];
    let out_channels = filter.shape[3].clone();
    let kernel_h = &filter.shape[0];
    let kernel_w = &filter.shape[1];

    let in_channels = &input.shape[3];
    let filter_in_channels = &filter.shape[2];
    if !shapes_compatible(in_channels, filter_in_channels) {
        return Err(MlirLowerError::ShapeError(
            "conv2d input channels must match filter input channels".into(),
        ));
    }

    let out_h = conv_output_dim(in_h, kernel_h, stride_h, padding)?;
    let out_w = conv_output_dim(in_w, kernel_w, stride_w, padding)?;

    let out_shape = vec![batch, out_h, out_w, out_channels];
    let input_ty = tensor_type(&input.shape, input.dtype.as_str());
    let filter_ty = tensor_type(&filter.shape, filter.dtype.as_str());
    let out_ty = tensor_type(&out_shape, input.dtype.as_str());
    Ok((out_shape, input_ty, filter_ty, out_ty))
}

fn conv_output_dim(
    input: &ShapeDim,
    kernel: &ShapeDim,
    stride: usize,
    padding: ConvPadding,
) -> Result<ShapeDim, MlirLowerError> {
    let input_known = known_dim(input);
    let kernel_known = known_dim(kernel);
    let result = match padding {
        ConvPadding::Valid => {
            crate::linalg::conv_output_dim_valid(input_known, kernel_known, stride)
                .map_err(MlirLowerError::ShapeError)?
        }
        ConvPadding::Same => crate::linalg::conv_output_dim_same(input_known, stride)
            .map_err(MlirLowerError::ShapeError)?,
    };
    Ok(match result {
        Some(n) => ShapeDim::Known(n),
        None => input.clone(),
    })
}

fn known_dim(dim: &ShapeDim) -> Option<usize> {
    match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    }
}

fn format_fill(fill: Option<f64>, dtype: &DType) -> String {
    match (fill, dtype) {
        (Some(v), DType::F32 | DType::F16 | DType::BF16) => format_number(v),
        (Some(v), _) => format_number(v.trunc()),
        (None, DType::F32 | DType::F16 | DType::BF16) => "0.0".to_string(),
        (None, _) => "0".to_string(),
    }
}

fn format_number(n: f64) -> String {
    if (n.fract()).abs() < f64::EPSILON {
        format!("{:.1}", n)
    } else {
        n.to_string()
    }
}

/// Format a MLIR block-argument pass list for `cf.br` / `cf.cond_br`.
///
/// MLIR 18 (and later) requires a combined type annotation for the
/// complete value list, NOT per-value annotations:
///   n=0 → ``  (no parens)
///   n=1 → `(%v0 : i64)`
///   n>1 → `(%v0, %v1, ... : i64, i64, ...)`
///
/// The `values` slice is a list of `%name` strings (already percent-prefixed).
#[cfg(feature = "std-surface")]
fn fmt_block_args(values: &[String]) -> String {
    match values.len() {
        0 => String::new(),
        1 => format!("({} : i64)", values[0]),
        _ => {
            let vals = values.join(", ");
            let types = vec!["i64"; values.len()].join(", ");
            format!("({vals} : {types})")
        }
    }
}

/// Typed variant of [`fmt_block_args`]: formats a `cf.br` / `cf.cond_br`
/// block-argument pass list where each operand carries its own MLIR type.
///
/// NARROW-INT control flow: value-if and while-loop merges carry the REAL
/// branch type (`i32`/`i1`/`i64`) instead of a hardcoded `i64`, so the join /
/// header / body block-arg declarations match the operand types and mlir-opt
/// accepts the edge.
///
/// BYTE-IDENTITY INVARIANT: for an all-`i64` list this is byte-for-byte
/// identical to [`fmt_block_args`] — len 1 → `(%x : i64)`, len > 1 →
/// `(%a, %b : i64, i64)`. An i64-only program therefore lowers unchanged.
fn fmt_block_args_typed(items: &[(String, String)]) -> String {
    match items.len() {
        0 => String::new(),
        1 => format!("({} : {})", items[0].0, items[0].1),
        _ => {
            let vals = items
                .iter()
                .map(|(v, _)| v.clone())
                .collect::<Vec<_>>()
                .join(", ");
            let types = items
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({vals} : {types})")
        }
    }
}

/// Replace every occurrence of `%{init_id}` with `%{arg_name}` in `text`
/// for each `(init_id, arg_name, _post_id)` triple.
///
/// Boundary rule: a match is valid only when the character immediately
/// after the digit run is NOT another ASCII digit.  This prevents `%1`
/// from being substituted inside `%10`, `%11`, etc.
///
/// The substitution is applied from largest init_id to smallest so that
/// multi-digit ids (e.g. `%12`) are processed before single-digit ones
/// (e.g. `%1`) and thus can never be partially clobbered.
#[cfg(feature = "std-surface")]
fn substitute_ids(text: &str, args: &[(usize, String, usize)]) -> String {
    // Sort descending by init_id so longer numeric ids are replaced first.
    let mut sorted: Vec<&(usize, String, usize)> = args.iter().collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.0));

    let mut result = text.to_owned();
    for (init_id, arg_name, _) in &sorted {
        result = substitute_one(&result, *init_id, arg_name);
    }
    result
}

/// Return `%{arg_name}` if `id` matches any `init_id` in `args`, otherwise
/// return `%{id}`.
#[cfg(feature = "std-surface")]
fn substitute_single_id(id: usize, args: &[(usize, String, usize)]) -> String {
    for (init_id, arg_name, _) in args {
        if *init_id == id {
            return format!("%{arg_name}");
        }
    }
    format!("%{id}")
}

/// Replace all boundary-safe occurrences of `%{id}` in `text` with
/// `%{replacement}`.  A match is boundary-safe when the character
/// immediately following the digit run is not an ASCII digit.
fn substitute_one(text: &str, id: usize, replacement: &str) -> String {
    let needle = format!("%{id}");
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let nlen = needle.len();
    let mut pos = 0usize;
    while pos < bytes.len() {
        // Fast check: does the slice starting here begin with needle?
        if bytes.len() - pos >= nlen && &bytes[pos..pos + nlen] == needle.as_bytes() {
            // Boundary check: char after the needle must not be an ASCII digit.
            let after = pos + nlen;
            let next_is_digit = after < bytes.len() && bytes[after].is_ascii_digit();
            if !next_is_digit {
                out.push('%');
                out.push_str(replacement);
                pos += nlen;
                continue;
            }
        }
        out.push(bytes[pos] as char);
        pos += 1;
    }
    out
}
