//! FP-contract mode — the strict-vs-relaxed floating-point determinism state of
//! a module, derived *purely* from its ops.
//!
//! Design: `mind-internal/plans/FP-CONTRACT-VERIFIER-INVARIANT-DESIGN.md`.
//!
//! Strict-FP in MIND is an emergent *negative* property: the compiler never
//! emits a fastmath / reassociate / fma-contract attribute, so scalar `f64`/`f32`
//! (`+ − × ÷ √`) lowers to plain `arith.*f` and is bit-identical by construction
//! (`src/mlir/lowering.rs:930-945`). What breaks it is a small, enumerable set of
//! RFC 0006 Track B f32 vector operations that lower to a contracted
//! (`vector.fma`) or reassociating (`vector.reduction <add>`) float op.
//!
//! Crucially, these appear in the *serialized* IR in **two** representations,
//! and the scan must catch both:
//!
//! 1. **Intrinsic `Instr::Call`** (the representation a real program produces).
//!    The Track B intrinsics stay `Instr::Call { name: "__mind_blas_*_f32_v" }`
//!    in the mic@3 body and are expanded to `vector.fma` / `vector.reduction
//!    <add>` only later, at the MLIR-lowering stage. So a `dot_f32_v` artifact
//!    carries the *Call*, not a `VecFma` — matching only the latter would
//!    falsely report it strict (verified empirically). See
//!    [`RELAXED_F32_INTRINSICS`].
//! 2. **`Instr::VecFma` / f32 `Instr::VecReduceAdd`** variants held directly in
//!    the IR (mic@3 round-trip of a hand-built module; a future direct-emit
//!    frontend). Their integer siblings `VecMulAddQ16` / `VecReduceAddI64` are
//!    associative and byte-identical, so they are **not** taint.
//! 3. **A tensor-level float reduction** — `Instr::Sum` / `Instr::Mean` over an
//!    IEEE-float (`f32`/`f64`/`f16`/`bf16`) tensor. Such a reduction reassociates
//!    at the linalg level (a `linalg.reduce` / tree fold is order-free), so the
//!    result is bit-reproducible ONLY through a *specific* pinned lowering (the
//!    MLIR `emit_tensor_reduce_pinned` fold) that this pre-lowering IR scan cannot
//!    see — so it must NOT be labelled strict here, or the invariant would emit a
//!    false `Strict` (a soundness break). Integer / Q16 reductions are associative
//!    (MIND-CONSTITUTION §III) and stay strict; a reduction whose source dtype is
//!    not statically provable in the scanned body fails closed to taint (never a
//!    false strict). The source element type is recovered by a minimal forward
//!    `ValueId -> DType` pass over literals and dtype-preserving ops.
//! 4. **A tensor-level float CONTRACTION** — `Instr::MatMul` / `Instr::Dot` /
//!    `Instr::Conv2d` / `Instr::Conv2dGradInput` / `Instr::Conv2dGradFilter` over
//!    an IEEE-float tensor. A contraction is a *sum of products* — the exact same
//!    reassociating float reduction as `sum`/`mean`, just expressed as a
//!    contraction — and lowers to `linalg.matmul` / `linalg.conv_*`
//!    (`src/eval/mlir_export.rs`), which the backend tiles and reassociates, so an
//!    IEEE-float matmul/dot/conv is NOT byte-identical across substrates. By the
//!    identical logic that taints `Sum`/`Mean` it must taint here. Integer / Q16
//!    contractions are associative + commutative and stay strict; a contraction
//!    whose operand dtype is not statically provable fails closed to taint (never
//!    a false strict).
//!
//! ## Scoped `Instr::Call` defense (#98 — three layers)
//!
//! The Call check historically matched only the two curated *relaxed* lists
//! ([`RELAXED_F32_INTRINSICS`], [`RELAXED_RUNTIME_F32_EXTERNS`], both empty
//! today), so any *other* Call name was silently non-taint. This is latent, not
//! live — no float transcendental is emitted today (the full emitted-intrinsic
//! inventory is `__mind_conv_*` / `__mind_bits_to_f64` / `__mind_f64_to_bits` /
//! the strict `__mind_blas_*` set / alloc·load·store·region) — but two future
//! paths reopen it: a transcendental intrinsic (`__mind_math_sin` → libm differs
//! per platform → a false `Strict` attestation would be a soundness break of
//! `--require-strict-fp`) and a user `extern "C" fn sin(x: f64) -> f64`.
//!
//! Blanket-tainting every unknown Call is WRONG and is deliberately rejected:
//! user fns recurse through their `FnDef` bodies (so a taint inside is already
//! seen), and `alloc` / `store` / runtime-bridge Calls are integer plumbing —
//! tainting them would flip every real program to `Relaxed` and destroy the
//! strict attestation. Instead the defense is scoped to exactly the float paths:
//!
//! - **Layer 1 — extern-decl-driven taint (the real defense, works today).**
//!   `Instr::ExternFnDecl` carries `ret_type: Option<String>` (MLIR type
//!   strings). The scan collects [`FpScan::extern_float_rets`] — every decl whose
//!   `ret_type` is `"f32"` / `"f64"` — during the walk (decls precede calls in
//!   program order; SSA is defined-before-use), and taints any `Instr::Call`
//!   whose name is in that set. Float-*param*-only externs (e.g. `printf("%f",
//!   …)`) do **NOT** taint: the float leaves the dataflow through the call, it
//!   is not a reassociated result — considered and rejected as an over-taint.
//! - **Layer 2 — emitter-enforced strict registry (build-breaks-until-classified).**
//!   [`STRICT_FLOAT_INTRINSICS`] lists the `__mind_`-prefixed float-returning
//!   intrinsics that are strict. The MLIR lowerer asserts (fail-loud, debug +
//!   release) that any `__mind_` callee registering a `ScalarF32`/`ScalarF64`
//!   result is in `STRICT_FLOAT_INTRINSICS ∪ RELAXED_F32_INTRINSICS` — so a
//!   float-returning intrinsic CANNOT be added without classifying its
//!   FP-contract behaviour: the build fails until the author lists the name.
//! - **Layer 3 — dtype precision.** `record_dtype` records `DType::F32`/`F64`
//!   for the conv/bits intrinsic dsts, so a float produced by a cast that feeds a
//!   tensor reduction resolves precisely (today it fails closed to taint — sound,
//!   just over-tainting; this is precision, not soundness).
//!
//! Why this is trustworthy without any wire-format change: `trace_hash =
//! mini_sha256(emit_mic3(ir))` already covers the entire canonical body, taint
//! ops included. So [`fp_contract_mode`] re-derived from the re-parsed body is
//! exactly as trustworthy as `trace_hash` itself — a hidden `VecFma` cannot sit
//! in a "strict" body without changing the bytes and breaking the hash. The
//! verifier attests strict-FP for free.

use crate::ir::{IRModule, Instr, ValueId};
use crate::types::DType;
use std::collections::{BTreeSet, HashMap};

/// RFC 0006 Track B f32 vector intrinsics whose MLIR lowering emits a
/// contracted (`vector.fma`) and/or reassociating (`vector.reduction <add>`)
/// float op — i.e. they break scalar-strict bit-identity. These stay as
/// `Instr::Call` in the serialized IR (lowered only at the MLIR stage), so the
/// FP-mode scan matches them by name.
///
/// Empirically derived (NOT guessed) by emitting each intrinsic's MLIR and
/// grepping for `vector.fma` / `vector.reduction <add>`:
///   `mindc <call>.mind --emit-mlir | grep -E 'vector.fma|reduction <add>'`
/// EMPTY as of the strict-vector-tier completion: EVERY f32 `_v` vector
/// intrinsic is now strict-FP. `dot_f32_v`, `dot_l1_f32_v`, and
/// `matmul_rmajor_f32_v` had their FMA unfused to `mulf`+`addf` and their
/// horizontal `vector.reduction <add>` replaced by a pinned left-to-right scalar
/// fold (see emit_vec_dot_f32 / emit_vec_dot_metric_f32 /
/// emit_vec_matmul_rmajor_f32); `dot_linf_f32_v` only ever used the associative
/// `vector.reduction <maximumf>`; the Q16 / i16 / i8 intrinsics are associative
/// integer arithmetic. So no intrinsic Call is a taint any more — the only
/// remaining relaxed representation is a raw `Instr::VecFma` / f32
/// `Instr::VecReduceAdd` variant (round-trip / hand-built IR).
/// upgrade: if a NEW non-strict f32 `_v` intrinsic is ever added, classify it
/// with `mindc <call>.mind --emit-mlir | grep -E 'vector.fma|reduction <add>'`
/// and list it here.
pub(crate) const RELAXED_F32_INTRINSICS: &[&str] = &[];

/// The `__mind_`-prefixed float-returning intrinsics that ARE strict-FP —
/// scalar IEEE conversions / bit reinterprets and the strict Track-B f32 `_v`
/// vector intrinsics (FMA unfused + a pinned left-to-right fold, byte-identical
/// across substrates). This is the #98 layer-2 registry: the MLIR lowerer
/// (`src/mlir/lowering.rs`) asserts that ANY `__mind_` callee registering a
/// `ScalarF32`/`ScalarF64` result is in this set (or [`RELAXED_F32_INTRINSICS`]).
/// So a float-returning intrinsic cannot be added without a deliberate
/// classification here — the build breaks (debug AND release) until it is listed.
/// This is "the plan for when transcendentals arrive": adding `__mind_math_sin`
/// forces the author to classify it strict (proven bit-identical) or relaxed
/// (list it in `RELAXED_F32_INTRINSICS`), never silently `Strict`.
///
/// The four `_f32_v` names return their f32 result packed into an i64 SSA slot
/// (the "i64-packed-f32 ABI"), so the emitter assert never fires on them; they
/// are listed here for registry completeness and are the same names the
/// `all_f32_vector_intrinsics_are_strict` test pins.
///
/// Consumed by the MLIR emitter assert (src/mlir/lowering.rs), which is compiled
/// only behind the `mlir-lowering`/`mlir-build` features; kept always-compiled
/// (so the doc link and the classification registry stay stable) with
/// `allow(dead_code)` for the `--no-default-features` build that has no emitter.
#[allow(dead_code)]
pub(crate) const STRICT_FLOAT_INTRINSICS: &[&str] = &[
    "__mind_conv_f32",
    "__mind_conv_f64",
    "__mind_bits_to_f64",
    "__mind_blas_dot_f32_v",
    "__mind_blas_dot_l1_f32_v",
    "__mind_blas_dot_linf_f32_v",
    "__mind_blas_matmul_rmajor_f32_v",
];

/// Opaque Track-A runtime BLAS externs (the non-`_v` C helpers in
/// `runtime-support/mind_intrinsics.c`) that are KNOWN non-strict — objdump- and
/// source-confirmed, NOT assumed.
/// EMPTY as of task #66: `__mind_blas_dot_f32` and `__mind_blas_dot_l1_f32` were
/// made strict (Option C — strict-but-vectorized). Both the AVX2 and scalar
/// paths now share ONE pinned reduction schedule (8-lane accumulation +
/// left-to-right horizontal fold) with unfused (non-FMA) products — the explicit
/// `_mm256_fmadd_ps` was replaced by `_mm256_mul_ps` + `_mm256_add_ps`, so the
/// two dispatch paths sum in the identical order and return byte-identical f32
/// bits at every length. `__mind_blas_matmul_rmajor_f32` (which fans out to the
/// same dot dispatcher) and `__mind_blas_dot_linf_f32` (associative max) are
/// strict for the same reason.
/// upgrade: if a NEW non-strict f32 C helper is added, objdump-confirm the
/// divergence (`objdump -d <fn>.so | grep vfmadd`, or an AVX2-vs-scalar order
/// mismatch) and list it here. Prefer the strict Track-B `_v` surface.
const RELAXED_RUNTIME_F32_EXTERNS: &[&str] = &[];

/// Strict-FP contract mode of a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FpMode {
    /// Not derived — a MAP-only decode path that never re-parsed the body, so
    /// the strict-FP property could not be checked. Fail-closed: `is_strict()`
    /// is `false`, so `--require-strict-fp` rejects it. We never claim strict
    /// without having actually scanned the body (no-fake-wins).
    #[default]
    Unknown,
    /// No FMA-contraction / f32-reassociation op anywhere in the body — the
    /// proven bit-identical scalar / integer path.
    Strict,
    /// Contains at least one op that lowers to a contracted (`llvm.intr.fmuladd`)
    /// or reassociating (`vector.reduce.fadd`) float op.
    Relaxed,
}

impl FpMode {
    /// Stable lowercase tag for CLI / JSON output.
    pub fn as_str(self) -> &'static str {
        match self {
            FpMode::Unknown => "unknown",
            FpMode::Strict => "strict",
            FpMode::Relaxed => "relaxed",
        }
    }

    /// True only for the *proven* strict path. `--require-strict-fp` gates on
    /// this, so both `Relaxed` and `Unknown` fail closed.
    pub fn is_strict(self) -> bool {
        matches!(self, FpMode::Strict)
    }
}

/// Derive the FP-contract mode of `module` by scanning for the known non-strict
/// taint ops. Recurses into every nested instruction stream (function, loop
/// header + body, if condition + both branches, region body) — a taint op
/// hidden inside a branch or a loop condition still taints the module.
pub fn fp_contract_mode(module: &IRModule) -> FpMode {
    let mut scan = FpScan::default();
    if scan.stream_has_taint(&module.instrs) {
        FpMode::Relaxed
    } else {
        FpMode::Strict
    }
}

/// Single forward pass over the (possibly nested) instruction streams. Carries a
/// lightweight `ValueId -> DType` map so a tensor-level float reduction can be
/// distinguished from an associative integer one: the map is populated in
/// program order (SSA is defined-before-use), and each reduction is classified
/// against the element type of its source that was recorded earlier in the walk.
#[derive(Default)]
struct FpScan {
    /// Statically-recovered element type of each produced value. Only the subset
    /// needed to classify a reduction source is tracked (tensor/scalar literals
    /// plus dtype-preserving shape / elementwise ops); an entry is absent when
    /// the dtype cannot be proven from the scanned body (e.g. an opaque param).
    dtypes: HashMap<ValueId, DType>,
    /// #98 layer 1 — names of `extern "C"` functions declared with an `f32`/`f64`
    /// RETURN type. A `Call` to any of these is a taint: the result is a
    /// platform-`libm`-defined float that is not byte-identical across
    /// substrates. Populated from `Instr::ExternFnDecl` in program order (decls
    /// precede their calls; SSA is defined-before-use), so a later `Call` in the
    /// same or a nested stream resolves against it. Float-*param*-only externs
    /// are deliberately absent (the float leaves the dataflow — no reassociated
    /// result to attest).
    extern_float_rets: BTreeSet<String>,
}

impl FpScan {
    /// True iff `instrs` (or any nested sub-stream) contains an FP-contract taint
    /// op. Records value dtypes as it walks so later reductions resolve.
    fn stream_has_taint(&mut self, instrs: &[Instr]) -> bool {
        let mut tainted = false;
        for instr in instrs {
            if self.instr_taints(instr) {
                tainted = true;
            }
        }
        tainted
    }

    /// True iff `instr` is itself a taint op, or nests one. Also records the
    /// dtype `instr` produces (before recursing) so downstream reductions in the
    /// same stream can resolve their source element type.
    fn instr_taints(&mut self, instr: &Instr) -> bool {
        self.record_dtype(instr);

        // #98 layer 1: note an `extern "C"` declaration with a float RETURN type
        // so a later `Call` to it taints. Decls precede their calls in program
        // order (SSA is defined-before-use), so this set is complete by the time
        // any referring `Call` is scanned. `ExternFnDecl` is `std-surface`-gated;
        // a default build constructs none, so `extern_float_rets` stays empty and
        // this layer is inert there.
        #[cfg(feature = "std-surface")]
        if let Instr::ExternFnDecl {
            name,
            ret_type: Some(rt),
            ..
        } = instr
        {
            // Float-PARAM-only externs (rt is not a float) do NOT taint: the
            // float leaves the dataflow through the call rather than being a
            // reassociated result. Only a float-RETURN decl is a taint source.
            if rt == "f32" || rt == "f64" {
                self.extern_float_rets.insert(name.clone());
            }
        }

        // PRIMARY taint representation: an intrinsic CALL. The RFC 0006 Track B
        // f32 vector intrinsics stay `Instr::Call { name: "__mind_blas_*_f32_v" }`
        // in the serialized IR body and are intercepted only at the MLIR-lowering
        // stage (src/mlir/lowering.rs), where they expand into `vector.fma` /
        // `vector.reduction <add>`. So a real `dot_f32_v` artifact carries the
        // Call, NOT an `Instr::VecFma` — the scan MUST match the intrinsic name
        // or it would falsely report such an artifact strict. `Call` is a core
        // variant (not `std-surface`-gated), so this check compiles everywhere.
        if let Instr::Call { name, .. } = instr {
            let n = name.as_str();
            // Any relaxed intrinsic name taints the module. Both lists are EMPTY
            // today (all f32 `_v` intrinsics and the Track-A dot/L1 externs are
            // now strict), but the check stays so a future non-strict addition to
            // either list is honoured without touching this scan.
            if RELAXED_F32_INTRINSICS.contains(&n) || RELAXED_RUNTIME_F32_EXTERNS.contains(&n) {
                return true;
            }
            // #98 layer 1: a Call to an `extern "C"` fn declared with an f32/f64
            // return type taints — its result is a platform-libm float, not
            // byte-identical across substrates. Scoped to float-RETURN externs
            // ONLY (user fns, alloc/store, and float-param-only externs are NOT
            // tainted — see the module header's considered-and-rejected scope).
            if self.extern_float_rets.contains(n) {
                return true;
            }
        }

        // TAINT: a tensor-level float reduction. `sum` / `mean` over an IEEE-float
        // tensor reassociates at the linalg level (`linalg.reduce` / a tree fold
        // is order-free), so the total is not bit-reproducible by construction —
        // any bit-exact guarantee comes only from a *specific* pinned lowering
        // (currently the MLIR `emit_tensor_reduce_pinned` fold), which this
        // pre-lowering IR scan cannot see. Claiming `Strict` here would be a false
        // strict → a soundness break of the invariant itself. Integer / Q16
        // reductions are associative + commutative (MIND-CONSTITUTION §III) and
        // stay strict; a reduction whose source dtype is not statically provable
        // fails closed to taint (never a false strict). See `reduction_reassoc`.
        if let Instr::Sum { src, .. } | Instr::Mean { src, .. } = instr {
            if self.reduction_reassoc(*src) {
                return true;
            }
        }

        // TAINT: a tensor-level float CONTRACTION. `dot` / `matmul` / `conv*` are
        // each a *sum of products* — the SAME reassociating float reduction as
        // `sum`/`mean`, just expressed as a contraction — and lower to
        // `linalg.matmul` / `linalg.conv_*` (src/eval/mlir_export.rs), which the
        // backend tiles and reassociates. So an IEEE-float matmul/dot/conv is NOT
        // bit-reproducible across substrates and must taint, by the identical logic
        // that taints `Sum`/`Mean`. Integer / Q16 contractions are associative +
        // commutative (MIND-CONSTITUTION §III) and stay strict; a contraction whose
        // operand dtype is not statically provable fails closed to taint (never a
        // false strict). See `contraction_reassoc`. These are core IR variants (not
        // `std-surface`-gated), so this check compiles everywhere.
        let contraction_taint = match instr {
            Instr::Dot { a, b, .. } | Instr::MatMul { a, b, .. } => {
                self.contraction_reassoc(&[*a, *b])
            }
            Instr::Conv2d { input, filter, .. } => self.contraction_reassoc(&[*input, *filter]),
            Instr::Conv2dGradInput { dy, filter, .. } => self.contraction_reassoc(&[*dy, *filter]),
            Instr::Conv2dGradFilter { input, dy, .. } => self.contraction_reassoc(&[*input, *dy]),
            _ => false,
        };
        if contraction_taint {
            return true;
        }

        // SECONDARY taint representation: the `Instr::VecFma` / f32
        // `Instr::VecReduceAdd` variants themselves, for any path that carries
        // them directly in the IR (mic@3 round-trip of a hand-built module, a
        // future direct-emit frontend). Only exist under `std-surface`; absent in
        // a build without it → pure Strict.
        #[cfg(feature = "std-surface")]
        {
            if matches!(instr, Instr::VecFma { .. } | Instr::VecReduceAdd { .. }) {
                return true;
            }
        }

        // Recurse into ALL nested instruction streams (src/ir/mod.rs `Vec<Instr>`
        // fields). Missing one would be a soundness hole: a taint op hidden in a
        // loop/if condition would read as strict.
        match instr {
            Instr::FnDef { body, .. } => self.stream_has_taint(body),
            // The While / If / Region control-flow variants only exist under
            // `std-surface`; in a default build they are absent, so the recursion
            // into their nested streams must be gated to keep the no-feature lib
            // compiling. A default build never constructs them, so the
            // `_ => false` fallthrough is complete there.
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                let c = self.stream_has_taint(cond_instrs);
                let b = self.stream_has_taint(body);
                c || b
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                let c = self.stream_has_taint(cond_instrs);
                let t = self.stream_has_taint(then_instrs);
                let e = self.stream_has_taint(else_instrs);
                c || t || e
            }
            #[cfg(feature = "std-surface")]
            Instr::Region { body, .. } => self.stream_has_taint(body),
            _ => false,
        }
    }

    /// Classify a reduction (`sum` / `mean`) by the element type of its source:
    /// - integer / Q16 → associative + commutative → NOT reassociable → strict;
    /// - IEEE float (f32/f64/f16/bf16) → reassociates at the linalg level → taint;
    /// - dtype not statically resolvable in the scanned body → fail closed to
    ///   taint (cannot PROVE the source is a non-reassociable integer, so never
    ///   claim strict for it — soundness over precision).
    fn reduction_reassoc(&self, src: ValueId) -> bool {
        match self.dtypes.get(&src) {
            Some(DType::I32) | Some(DType::I64) | Some(DType::Q16) => false,
            Some(DType::F32) | Some(DType::F64) | Some(DType::F16) | Some(DType::BF16) => true,
            None => true,
        }
    }

    /// Classify a tensor CONTRACTION (`dot` / `matmul` / `conv*`) by the element
    /// type of its operands — the same fail-closed, dtype-conditional rule as
    /// [`Self::reduction_reassoc`], generalized over the (2+) operands a
    /// contraction reduces over (a matmul's operands share their element type):
    /// - any operand provably IEEE float (f32/f64/f16/bf16) → reassociates → taint;
    /// - otherwise, an operand provably integer / Q16 → associative + commutative
    ///   → NOT reassociable → strict;
    /// - no operand statically resolvable in the scanned body → fail closed to
    ///   taint (cannot PROVE the contraction is a non-reassociable integer, so
    ///   never claim strict for it — soundness over precision).
    fn contraction_reassoc(&self, operands: &[ValueId]) -> bool {
        let mut proven_integer = false;
        for v in operands {
            match self.dtypes.get(v) {
                Some(DType::F32) | Some(DType::F64) | Some(DType::F16) | Some(DType::BF16) => {
                    return true;
                }
                Some(DType::I32) | Some(DType::I64) | Some(DType::Q16) => {
                    proven_integer = true;
                }
                None => {}
            }
        }
        !proven_integer
    }

    /// Record the element type produced by `instr`, when it can be recovered from
    /// a literal or propagated from an already-tracked source. This is a minimal,
    /// forward, best-effort recovery — enough to keep a *provably* integer
    /// reduction strict without over-tainting it; any op not covered simply
    /// leaves its result untracked, which fails closed at a downstream reduction.
    fn record_dtype(&mut self, instr: &Instr) {
        match instr {
            // Literals carry their dtype directly.
            Instr::ConstI64(dst, _) => {
                self.dtypes.insert(*dst, DType::I64);
            }
            Instr::ConstF64(dst, _) => {
                self.dtypes.insert(*dst, DType::F64);
            }
            Instr::ConstTensor(dst, dtype, _, _) => {
                self.dtypes.insert(*dst, dtype.clone());
            }
            Instr::ConstDenseTensor { dst, dtype, .. } => {
                self.dtypes.insert(*dst, dtype.clone());
            }
            // Dtype-preserving unary / shape / reduction / activation ops:
            // the result element type equals the source element type.
            Instr::Sum { dst, src, .. }
            | Instr::Mean { dst, src, .. }
            | Instr::Relu { dst, src }
            | Instr::ReluGrad { dst, src, .. }
            | Instr::Reshape { dst, src, .. }
            | Instr::ExpandDims { dst, src, .. }
            | Instr::Squeeze { dst, src, .. }
            | Instr::Transpose { dst, src, .. }
            | Instr::Index { dst, src, .. }
            | Instr::Slice { dst, src, .. }
            | Instr::Gather { dst, src, .. } => {
                self.propagate(*dst, *src);
            }
            // Binary elementwise / contraction ops: both operands share the
            // element type, so the result takes whichever operand is known.
            Instr::BinOp { dst, lhs, rhs, .. } => {
                self.propagate_from(*dst, &[*lhs, *rhs]);
            }
            Instr::Dot { dst, a, b } | Instr::MatMul { dst, a, b } => {
                self.propagate_from(*dst, &[*a, *b]);
            }
            // #98 layer 3 (precision): the scalar `as f32`/`as f64` conversion and
            // enum-payload bit-reinterpret intrinsics produce a value of a known
            // float dtype. Recording it lets a downstream reduction over a
            // cast-produced float resolve precisely instead of failing closed to
            // an unknown-dtype taint. Sound either way (a float reduction still
            // taints); this only sharpens the dtype map. `Call` is a core variant.
            Instr::Call { dst, name, .. } => match name.as_str() {
                "__mind_conv_f32" => {
                    self.dtypes.insert(*dst, DType::F32);
                }
                "__mind_conv_f64" | "__mind_bits_to_f64" => {
                    self.dtypes.insert(*dst, DType::F64);
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// Copy the tracked dtype of `src` onto `dst`, if `src` is known.
    fn propagate(&mut self, dst: ValueId, src: ValueId) {
        if let Some(dt) = self.dtypes.get(&src).cloned() {
            self.dtypes.insert(dst, dt);
        }
    }

    /// Assign `dst` the dtype of the first known value in `srcs`.
    fn propagate_from(&mut self, dst: ValueId, srcs: &[ValueId]) {
        for s in srcs {
            if let Some(dt) = self.dtypes.get(s).cloned() {
                self.dtypes.insert(dst, dt);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRModule, ValueId};
    use crate::types::{DType, ShapeDim};

    fn module_with(instrs: Vec<Instr>) -> IRModule {
        let mut m = IRModule::new();
        m.instrs = instrs;
        m
    }

    fn call(name: &str) -> Instr {
        Instr::Call {
            dst: ValueId(1),
            name: name.into(),
            args: vec![ValueId(0)],
        }
    }

    #[test]
    fn empty_and_scalar_module_is_strict() {
        // No taint op anywhere → the proven strict path.
        assert_eq!(fp_contract_mode(&IRModule::new()), FpMode::Strict);
        let m = module_with(vec![Instr::ConstI64(ValueId(0), 7)]);
        assert_eq!(fp_contract_mode(&m), FpMode::Strict);
    }

    #[test]
    fn all_f32_vector_intrinsics_are_strict() {
        // Strict-vector-tier complete: every Track-B f32 `_v` intrinsic emits no
        // vector.fma / vector.reduction <add> (verified via --emit-mlir + objdump)
        // and is therefore strict. Regression guard — if any is re-classified
        // relaxed (or a new relaxed intrinsic sneaks into the taint list), this
        // fails.
        for name in [
            "__mind_blas_dot_f32_v",
            "__mind_blas_dot_l1_f32_v",
            "__mind_blas_dot_linf_f32_v",
            "__mind_blas_matmul_rmajor_f32_v",
        ] {
            let m = module_with(vec![call(name)]);
            assert_eq!(
                fp_contract_mode(&m),
                FpMode::Strict,
                "{name} must be strict"
            );
        }
    }

    #[test]
    fn strict_and_integer_intrinsic_calls_are_strict() {
        // dot_linf (maximumf-reduce, associative) and the integer/Q16 intrinsics
        // are bit-identical and must NOT be flagged.
        for name in [
            "__mind_blas_dot_f32_v",    // strict-vector-tier: unfused FMA + pinned fold
            "__mind_blas_dot_l1_f32_v", // strict-vector-tier: pinned fold (no FMA)
            "__mind_blas_dot_linf_f32_v",
            "__mind_blas_dot_q16_v",
            "__mind_blas_matmul_mm_i8_v",
            "__mind_blas_matmul_rmajor_f32", // Track-A: scalar sequential dot — strict
            "__mind_blas_dot_linf_f32",      // Track-A: associative max — strict
        ] {
            let m = module_with(vec![call(name)]);
            assert_eq!(
                fp_contract_mode(&m),
                FpMode::Strict,
                "{name} must be strict"
            );
        }
    }

    #[test]
    fn track_a_f32_dispatch_externs_are_strict() {
        // task #66: the Track-A f32 dot / L1 C helpers were made strict-but-
        // vectorized — AVX2 and scalar share one pinned 8-lane schedule +
        // left-to-right fold with unfused (non-FMA) products, so both dispatch
        // paths are byte-identical. They are no longer in RELAXED_RUNTIME_F32_
        // EXTERNS, so a program calling them is strict-FP. Regression guard: if
        // either is ever re-listed relaxed, this fails.
        for name in ["__mind_blas_dot_f32", "__mind_blas_dot_l1_f32"] {
            let m = module_with(vec![call(name)]);
            assert_eq!(
                fp_contract_mode(&m),
                FpMode::Strict,
                "{name} must be strict"
            );
        }
    }

    // --- #98 layer 1: extern-decl-driven Call taint (scoped, not blanket) ---

    #[cfg(feature = "std-surface")]
    fn extern_decl(name: &str, param_types: &[&str], ret: Option<&str>) -> Instr {
        Instr::ExternFnDecl {
            name: name.into(),
            param_types: param_types.iter().map(|s| s.to_string()).collect(),
            ret_type: ret.map(String::from),
            is_varargs: false,
            vararg_hints: vec![],
            callconv: crate::ast::CallConv::SysV,
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn extern_float_return_call_is_relaxed() {
        // #98 layer 1: a Call to an `extern "C"` fn declared with a float RETURN
        // type taints — the result is a platform-libm float, not byte-identical
        // across substrates. The decl precedes the call in program order, so the
        // scan resolves it (defined-before-use).
        let m64 = module_with(vec![extern_decl("sin", &["f64"], Some("f64")), call("sin")]);
        assert_eq!(
            fp_contract_mode(&m64),
            FpMode::Relaxed,
            "call to `-> f64` extern must be Relaxed"
        );

        let m32 = module_with(vec![
            extern_decl("sinf", &["f32"], Some("f32")),
            call("sinf"),
        ]);
        assert_eq!(
            fp_contract_mode(&m32),
            FpMode::Relaxed,
            "call to `-> f32` extern must be Relaxed"
        );

        // The taint resolves even when the call is nested inside a fn body, as
        // long as the decl was scanned first at the top level.
        let m_nested = module_with(vec![
            extern_decl("cos", &["f64"], Some("f64")),
            Instr::FnDef {
                name: "f".into(),
                params: vec![],
                ret_id: None,
                body: vec![call("cos")],
                reap_threshold: None,
            },
        ]);
        assert_eq!(
            fp_contract_mode(&m_nested),
            FpMode::Relaxed,
            "nested call to `-> f64` extern must be Relaxed"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn extern_int_return_call_stays_strict() {
        // DO NOT over-taint: an integer-returning extern (`write(...) -> i64`) is
        // integer plumbing and must stay Strict. This is the exact case a blanket
        // "taint every unknown Call" fix would wrongly flip to Relaxed.
        let m = module_with(vec![
            extern_decl("write", &["i64", "i64", "i64"], Some("i64")),
            call("write"),
        ]);
        assert_eq!(
            fp_contract_mode(&m),
            FpMode::Strict,
            "call to `-> i64` extern must stay Strict"
        );

        // A void-return extern likewise does not taint.
        let m_void = module_with(vec![extern_decl("puts", &["i64"], None), call("puts")]);
        assert_eq!(
            fp_contract_mode(&m_void),
            FpMode::Strict,
            "call to void extern must stay Strict"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn extern_float_param_only_call_stays_strict() {
        // Considered-and-rejected scope: `printf(fmt, f64) -> i64` takes a float
        // PARAM but returns an integer. The float leaves the dataflow through the
        // call; there is no reassociated float RESULT to attest, so a
        // float-param-only extern must NOT taint.
        let m = module_with(vec![
            extern_decl("printf", &["i64", "f64"], Some("i64")),
            call("printf"),
        ]);
        assert_eq!(
            fp_contract_mode(&m),
            FpMode::Strict,
            "float-param-only extern must stay Strict"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn top_level_vecfma_is_relaxed() {
        let m = module_with(vec![Instr::VecFma {
            dst: ValueId(3),
            a: ValueId(0),
            b: ValueId(1),
            acc: ValueId(2),
            lanes: 8,
        }]);
        assert_eq!(fp_contract_mode(&m), FpMode::Relaxed);
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn top_level_f32_vecreduceadd_is_relaxed() {
        let m = module_with(vec![Instr::VecReduceAdd {
            dst: ValueId(1),
            src: ValueId(0),
            lanes: 8,
        }]);
        assert_eq!(fp_contract_mode(&m), FpMode::Relaxed);
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn integer_simd_siblings_are_not_taint() {
        // VecReduceAddI64 is associative integer arithmetic → still strict.
        let m = module_with(vec![Instr::VecReduceAddI64 {
            dst: ValueId(1),
            src: ValueId(0),
            lanes: 8,
        }]);
        assert_eq!(fp_contract_mode(&m), FpMode::Strict);
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn vecfma_nested_in_fn_body_is_relaxed() {
        // Recursion into FnDef.body.
        let m = module_with(vec![Instr::FnDef {
            name: "f".into(),
            params: vec![],
            ret_id: None,
            body: vec![Instr::VecFma {
                dst: ValueId(3),
                a: ValueId(0),
                b: ValueId(1),
                acc: ValueId(2),
                lanes: 8,
            }],
            reap_threshold: None,
        }]);
        assert_eq!(fp_contract_mode(&m), FpMode::Relaxed);
    }

    // --- Tensor-level float reduction soundness (the linalg reassociation path) ---

    fn tensor(dst: ValueId, dt: DType) -> Instr {
        Instr::ConstTensor(dst, dt, vec![ShapeDim::Known(64)], Some(0.0))
    }

    fn sum_over(dst: ValueId, src: ValueId) -> Instr {
        Instr::Sum {
            dst,
            src,
            axes: vec![],
            keepdims: false,
        }
    }

    #[test]
    fn reassociable_f32_tensor_sum_is_relaxed() {
        // SOUNDNESS FIX: a tensor `sum` / `mean` over an f32 tensor reassociates
        // at the linalg level. `fp_contract_mode` scans the PRE-lowering IR and
        // cannot see whether a given backend happens to pin the fold, so it must
        // NOT claim strict — that would be a false `Strict` (a soundness break).
        // (Pre-fix this returned `Strict`; this test is RED before the fix.)
        let src = ValueId(0);
        let red = ValueId(1);
        let m_sum = module_with(vec![tensor(src, DType::F32), sum_over(red, src)]);
        assert_eq!(
            fp_contract_mode(&m_sum),
            FpMode::Relaxed,
            "f32 tensor sum reassociates → must be Relaxed"
        );

        // `mean` over f32 is the same reassociation class.
        let m_mean = module_with(vec![
            tensor(src, DType::F32),
            Instr::Mean {
                dst: red,
                src,
                axes: vec![],
                keepdims: false,
            },
        ]);
        assert_eq!(
            fp_contract_mode(&m_mean),
            FpMode::Relaxed,
            "f32 tensor mean reassociates → must be Relaxed"
        );

        // f64 reduction reassociates too.
        let m_f64 = module_with(vec![tensor(src, DType::F64), sum_over(red, src)]);
        assert_eq!(fp_contract_mode(&m_f64), FpMode::Relaxed);

        // Fail-closed: a reduction whose source dtype is NOT statically
        // resolvable in the scanned body (e.g. an opaque tensor param) cannot be
        // PROVEN integer/associative, so it must never be labelled strict.
        let opaque = ValueId(9);
        let m_opaque = module_with(vec![sum_over(ValueId(10), opaque)]);
        assert_eq!(
            fp_contract_mode(&m_opaque),
            FpMode::Relaxed,
            "unresolved-dtype reduction must fail closed to Relaxed"
        );
    }

    #[test]
    fn integer_tensor_reduction_stays_strict() {
        // DO NOT over-taint: integer add is associative + commutative, so an
        // integer / Q16 tensor reduction is byte-identical across substrates and
        // MUST stay strict (MIND-CONSTITUTION §III).
        let src = ValueId(0);
        let red = ValueId(1);
        for dt in [DType::I32, DType::I64, DType::Q16] {
            let m = module_with(vec![tensor(src, dt.clone()), sum_over(red, src)]);
            assert_eq!(
                fp_contract_mode(&m),
                FpMode::Strict,
                "{} tensor reduction is associative → must stay Strict",
                dt.as_str()
            );
        }

        // Propagation through a shape-preserving op keeps the integer source
        // integer (still strict), rather than degrading to unknown → Relaxed.
        let reshaped = ValueId(2);
        let m = module_with(vec![
            tensor(src, DType::I32),
            Instr::Reshape {
                dst: reshaped,
                src,
                new_shape: vec![ShapeDim::Known(8), ShapeDim::Known(8)],
            },
            sum_over(red, reshaped),
        ]);
        assert_eq!(fp_contract_mode(&m), FpMode::Strict);
    }

    // --- Tensor-level float CONTRACTION soundness (matmul / dot / conv) ---

    #[test]
    fn reassociable_f32_tensor_contraction_is_relaxed() {
        // SOUNDNESS FIX: a tensor CONTRACTION (`matmul` / `dot` / `conv*`) is a
        // sum of products — the same reassociating float reduction as `sum`/`mean`
        // — and lowers to `linalg.matmul` / `linalg.conv_*`, which the backend
        // tiles and reassociates. Over an f32 tensor the result is NOT byte-
        // identical across substrates, so it must be Relaxed, NOT Strict.
        // (Pre-fix a float matmul/dot/conv returned `Strict` — a false attestation
        // of strict-FP determinism; these are RED before the fix.)
        let a = ValueId(0);
        let b = ValueId(1);
        let out = ValueId(2);

        // f32 matmul.
        let m_matmul = module_with(vec![
            tensor(a, DType::F32),
            tensor(b, DType::F32),
            Instr::MatMul { dst: out, a, b },
        ]);
        assert_eq!(
            fp_contract_mode(&m_matmul),
            FpMode::Relaxed,
            "f32 matmul reassociates → must be Relaxed"
        );

        // f32 dot.
        let m_dot = module_with(vec![
            tensor(a, DType::F32),
            tensor(b, DType::F32),
            Instr::Dot { dst: out, a, b },
        ]);
        assert_eq!(
            fp_contract_mode(&m_dot),
            FpMode::Relaxed,
            "f32 dot reassociates → must be Relaxed"
        );

        // f32 conv2d (operands: input, filter).
        let m_conv = module_with(vec![
            tensor(a, DType::F32),
            tensor(b, DType::F32),
            Instr::Conv2d {
                dst: out,
                input: a,
                filter: b,
                stride_h: 1,
                stride_w: 1,
                padding: crate::types::ConvPadding::Valid,
            },
        ]);
        assert_eq!(
            fp_contract_mode(&m_conv),
            FpMode::Relaxed,
            "f32 conv2d reassociates → must be Relaxed"
        );

        // f64 matmul reassociates too.
        let m_f64 = module_with(vec![
            tensor(a, DType::F64),
            tensor(b, DType::F64),
            Instr::MatMul { dst: out, a, b },
        ]);
        assert_eq!(fp_contract_mode(&m_f64), FpMode::Relaxed);

        // Fail-closed: a contraction whose operand dtypes are NOT statically
        // resolvable (opaque params) cannot be PROVEN integer/associative, so it
        // must never be labelled strict.
        let m_opaque = module_with(vec![Instr::MatMul {
            dst: ValueId(10),
            a: ValueId(8),
            b: ValueId(9),
        }]);
        assert_eq!(
            fp_contract_mode(&m_opaque),
            FpMode::Relaxed,
            "unresolved-dtype contraction must fail closed to Relaxed"
        );
    }

    #[test]
    fn integer_tensor_contraction_stays_strict() {
        // DO NOT over-taint: integer add is associative + commutative, so an
        // integer / Q16 matmul / dot is byte-identical across substrates and MUST
        // stay strict (MIND-CONSTITUTION §III — the int8/int16/Q16 GEMM wedge).
        let a = ValueId(0);
        let b = ValueId(1);
        let out = ValueId(2);
        for dt in [DType::I32, DType::I64, DType::Q16] {
            let m_matmul = module_with(vec![
                tensor(a, dt.clone()),
                tensor(b, dt.clone()),
                Instr::MatMul { dst: out, a, b },
            ]);
            assert_eq!(
                fp_contract_mode(&m_matmul),
                FpMode::Strict,
                "{} matmul is associative → must stay Strict",
                dt.as_str()
            );
            let m_dot = module_with(vec![
                tensor(a, dt.clone()),
                tensor(b, dt.clone()),
                Instr::Dot { dst: out, a, b },
            ]);
            assert_eq!(
                fp_contract_mode(&m_dot),
                FpMode::Strict,
                "{} dot is associative → must stay Strict",
                dt.as_str()
            );
        }
    }

    #[test]
    fn scalar_float_chain_stays_strict() {
        // Regression guard: the tensor-reduction taint must NOT leak onto plain
        // scalar float arithmetic, which lowers to `arith.*f` with no reassoc /
        // no fma-contract and is bit-identical by construction.
        let a = ValueId(0);
        let b = ValueId(1);
        let c = ValueId(2);
        let m = module_with(vec![
            Instr::ConstF64(a, 1.0),
            Instr::ConstF64(b, 2.0),
            Instr::BinOp {
                dst: c,
                op: crate::ir::BinOp::Add,
                lhs: a,
                rhs: b,
            },
        ]);
        assert_eq!(
            fp_contract_mode(&m),
            FpMode::Strict,
            "scalar f64 chain must stay Strict"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn vecfma_in_while_condition_is_relaxed() {
        // Soundness canary: a taint op hidden in a loop CONDITION stream must
        // still taint the module (the `cond_instrs` recursion arm).
        let m = module_with(vec![Instr::While {
            cond_id: ValueId(9),
            cond_instrs: vec![Instr::VecFma {
                dst: ValueId(3),
                a: ValueId(0),
                b: ValueId(1),
                acc: ValueId(2),
                lanes: 8,
            }],
            body: vec![],
            live_vars: vec![],
            init_ids: vec![],
            exit_ids: vec![],
        }]);
        assert_eq!(fp_contract_mode(&m), FpMode::Relaxed);
    }
}
