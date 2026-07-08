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
//!
//! Why this is trustworthy without any wire-format change: `trace_hash =
//! mini_sha256(emit_mic3(ir))` already covers the entire canonical body, taint
//! ops included. So [`fp_contract_mode`] re-derived from the re-parsed body is
//! exactly as trustworthy as `trace_hash` itself — a hidden `VecFma` cannot sit
//! in a "strict" body without changing the bytes and breaking the hash. The
//! verifier attests strict-FP for free.

use crate::ir::{IRModule, Instr, ValueId};
use crate::types::DType;
use std::collections::HashMap;

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
const RELAXED_F32_INTRINSICS: &[&str] = &[];

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
