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
/// Confirmed RELAXED: `dot_f32_v` (fma+add-reduce), `dot_l1_f32_v` (add-reduce),
/// `matmul_rmajor_f32_v` (fma+add-reduce). Confirmed STRICT and deliberately
/// EXCLUDED: `dot_linf_f32_v` (only `vector.reduction <maximumf>`, associative)
/// and every Q16 / i16 / i8 integer intrinsic (associative integer arithmetic).
/// deferred: re-run the emit-mlir check when a new f32 `_v` intrinsic is added.
const RELAXED_F32_INTRINSICS: &[&str] = &[
    "__mind_blas_dot_f32_v",
    "__mind_blas_dot_l1_f32_v",
    "__mind_blas_matmul_rmajor_f32_v",
];

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
        if RELAXED_F32_INTRINSICS.contains(&name.as_str()) {
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
    // deferred: two known gaps NOT yet in the taint set — never silently claimed
    // strict. (1) Tensor-level float reductions (`OP_SUM` over an f32 tensor) may
    // reassociate at the linalg level — needs mind-det-gemm numerical review.
    // (2) Opaque Track-A runtime BLAS externs (`__mind_blas_dot_f32`, non-`_v`)
    // are C functions whose FP behaviour is governed by the pinned
    // `-ffp-contract=off` clang flags, not visible to this IR-level scan.
    // upgrade: add each here once its determinism is measured.

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
    fn relaxed_f32_intrinsic_calls_are_relaxed() {
        // The representation a real program actually produces: an Instr::Call to
        // a Track-B f32 vector intrinsic. Each was confirmed via --emit-mlir to
        // emit vector.fma / vector.reduction <add>.
        for name in [
            "__mind_blas_dot_f32_v",
            "__mind_blas_dot_l1_f32_v",
            "__mind_blas_matmul_rmajor_f32_v",
        ] {
            let m = module_with(vec![call(name)]);
            assert_eq!(
                fp_contract_mode(&m),
                FpMode::Relaxed,
                "{name} must be relaxed"
            );
        }
    }

    #[test]
    fn strict_and_integer_intrinsic_calls_are_strict() {
        // dot_linf (maximumf-reduce, associative) and the integer/Q16 intrinsics
        // are bit-identical and must NOT be flagged.
        for name in [
            "__mind_blas_dot_linf_f32_v",
            "__mind_blas_dot_q16_v",
            "__mind_blas_matmul_mm_i8_v",
            "__mind_blas_dot_f32", // Track-A runtime extern — opaque, not IR taint
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
    fn relaxed_intrinsic_call_nested_in_fn_body_is_relaxed() {
        // Recursion + the Call representation together (a real program shape).
        let m = module_with(vec![Instr::FnDef {
            name: "f".into(),
            params: vec![],
            ret_id: None,
            body: vec![call("__mind_blas_dot_f32_v")],
            reap_threshold: None,
        }]);
        assert_eq!(fp_contract_mode(&m), FpMode::Relaxed);
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
