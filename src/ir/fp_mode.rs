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
//!
//! Why this is trustworthy without any wire-format change: `trace_hash =
//! mini_sha256(emit_mic3(ir))` already covers the entire canonical body, taint
//! ops included. So [`fp_contract_mode`] re-derived from the re-parsed body is
//! exactly as trustworthy as `trace_hash` itself — a hidden `VecFma` cannot sit
//! in a "strict" body without changing the bytes and breaking the hash. The
//! verifier attests strict-FP for free.

use crate::ir::{IRModule, Instr};

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
/// source-confirmed, NOT assumed. `__mind_blas_dot_f32` uses an explicit
/// `_mm256_fmadd_ps` (FMA contraction that `-ffp-contract=off` does NOT
/// neutralise) and dispatches an AVX2 lane-blocked reduction vs a scalar
/// sequential one at load time, so the two paths sum in different orders →
/// substrate-divergent. `__mind_blas_dot_l1_f32` has the same AVX2-vs-scalar
/// reduction-order dispatch. A program calling either is therefore not strict-FP,
/// even though the extern body is invisible to this IR-level scan. NOT listed:
/// `__mind_blas_matmul_rmajor_f32` (uses the scalar sequential dot — strict) and
/// `__mind_blas_dot_linf_f32` (associative max — strict).
/// upgrade: the deeper fix is to make the Track-A f32 BLAS itself strict (unfuse
/// the intrinsic FMA + pin ONE reduction order across both dispatch paths); once
/// done, remove the affected name here. Prefer the strict Track-B `_v` surface.
const RELAXED_RUNTIME_F32_EXTERNS: &[&str] = &["__mind_blas_dot_f32", "__mind_blas_dot_l1_f32"];

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
    if stream_has_taint(&module.instrs) {
        FpMode::Relaxed
    } else {
        FpMode::Strict
    }
}

/// True iff `instrs` (or any nested sub-stream) contains an FP-contract taint op.
fn stream_has_taint(instrs: &[Instr]) -> bool {
    instrs.iter().any(instr_taints)
}

/// True iff `instr` is itself a taint op, or nests one.
fn instr_taints(instr: &Instr) -> bool {
    // PRIMARY taint representation: an intrinsic CALL. The RFC 0006 Track B f32
    // vector intrinsics stay `Instr::Call { name: "__mind_blas_*_f32_v" }` in
    // the serialized IR body and are intercepted only at the MLIR-lowering stage
    // (src/mlir/lowering.rs), where they expand into `vector.fma` /
    // `vector.reduction <add>`. So a real `dot_f32_v` artifact carries the Call,
    // NOT an `Instr::VecFma` — the scan MUST match the intrinsic name or it
    // would falsely report such an artifact strict. `Call` is a core variant
    // (not `std-surface`-gated), so this check compiles in every build.
    if let Instr::Call { name, .. } = instr {
        let n = name.as_str();
        // Track-B `_v` intrinsics lowered to vector.fma / vector.reduction<add>,
        // plus known-non-strict Track-A runtime BLAS externs (explicit FMA /
        // AVX2-vs-scalar reduction-order dispatch). Either taints the module.
        if RELAXED_F32_INTRINSICS.contains(&n) || RELAXED_RUNTIME_F32_EXTERNS.contains(&n) {
            return true;
        }
    }
    // SECONDARY taint representation: the `Instr::VecFma` / f32 `Instr::VecReduceAdd`
    // variants themselves, for any path that carries them directly in the IR
    // (mic@3 round-trip of a hand-built module, a future direct-emit frontend).
    // Only exist under `std-surface`; absent in a build without it → pure Strict.
    #[cfg(feature = "std-surface")]
    {
        if matches!(instr, Instr::VecFma { .. } | Instr::VecReduceAdd { .. }) {
            return true;
        }
    }
    // deferred: tensor-level float reductions (`OP_SUM` over an f32 tensor) may
    // reassociate at the linalg level and are NOT yet in the taint set — needs
    // mind-det-gemm numerical review; upgrade: add once measured.
    // (The Track-A runtime BLAS externs are NO LONGER a blind spot: the
    // non-strict ones are enumerated in RELAXED_RUNTIME_F32_EXTERNS above from
    // objdump/source evidence, not assumed strict.)

    // Recurse into ALL nested instruction streams (src/ir/mod.rs `Vec<Instr>`
    // fields). Missing one would be a soundness hole: a taint op hidden in a
    // loop/if condition would read as strict.
    match instr {
        Instr::FnDef { body, .. } => stream_has_taint(body),
        Instr::While {
            cond_instrs, body, ..
        } => stream_has_taint(cond_instrs) || stream_has_taint(body),
        Instr::If {
            cond_instrs,
            then_instrs,
            else_instrs,
            ..
        } => {
            stream_has_taint(cond_instrs)
                || stream_has_taint(then_instrs)
                || stream_has_taint(else_instrs)
        }
        Instr::Region { body, .. } => stream_has_taint(body),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRModule, ValueId};

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
    fn track_a_fma_dispatch_externs_are_relaxed() {
        // Objdump/source-confirmed non-strict Track-A runtime BLAS f32 helpers:
        // explicit _mm256_fmadd_ps (dot_f32) and AVX2-vs-scalar reduction-order
        // dispatch (both dot_f32 and dot_l1_f32). fp_mode must NOT report a
        // program using them as strict (they'd falsely pass --require-strict-fp).
        for name in ["__mind_blas_dot_f32", "__mind_blas_dot_l1_f32"] {
            let m = module_with(vec![call(name)]);
            assert_eq!(
                fp_contract_mode(&m),
                FpMode::Relaxed,
                "{name} must be relaxed"
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
