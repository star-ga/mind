// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! RI-D1 frozen-profile admission predicate.
//!
//! The load-bearing safety mechanism for making the pure-MIND native-ELF backend
//! the DEFAULT for a frozen production profile without a silent-miscompile risk.
//! The council-decided spine: the frozen profile is a **construct ALLOWLIST
//! checked on the canonical IR** — a *positive* statement of what the native path
//! has been proven on — NOT a "try native then refuse", because the dangerous
//! failure is not "native refuses" (loud, recoverable) but "native accepts and
//! emits WRONG bytes" (silent). This predicate is the 1st fence; the frozen
//! stage1.elf's own fail-closed refusal is the 2nd (defense in depth).
//!
//! `profile_frozen_admits` walks an [`IRModule`] and returns the FIRST out-of-
//! profile construct (so a build gate can NAME it: "rerun with --profile full").
//! The match is EXHAUSTIVE with no blanket `_`: a future `Instr` variant is a
//! COMPILE ERROR here, forcing a deliberate in/out-of-profile decision rather
//! than a silent admission — the same no-silent-escape discipline as
//! `find_nondeterministic_call_ext`.
//!
//! Byte-neutral: this is a read-only predicate, called by NOTHING in the emit
//! path today, so it changes zero mic@3 bytes and cannot perturb the keystone.
//! Wiring it into the default-flip decision is a separate, gated slice.

use crate::ir::frozen_intrinsics::{
    admit_discard_only_results, admit_frozen_program, admit_intrinsic_call, fold_const,
    frozen_native_intrinsic,
};
use crate::ir::{BinOp, IRModule, Instr, ValueId};

#[path = "native_scalar.rs"]
pub(crate) mod scalar;
#[path = "native_source.rs"]
mod source;

/// The first construct that is NOT in the frozen native profile, named for a
/// fail-loud diagnostic. `None`-free by construction: `Ok(())` means every
/// construct in the module is on the allowlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenProfileRejection {
    /// A stable, human-facing construct label (e.g. "tensor.matmul", "region").
    pub construct: &'static str,
}

/// Admit `module` to the frozen native profile iff every construct it contains
/// has been proven byte-identical on the native path (the scalar / control-flow
/// / fixed-array subset the readiness gate covers). Returns the first offending
/// construct otherwise. Recurses into function bodies, loop bodies, and if-
/// branches so a rejected construct cannot hide inside a nested region.
pub fn profile_frozen_admits(module: &IRModule) -> Result<(), FrozenProfileRejection> {
    profile_frozen_admits_linked(module, &std::collections::BTreeSet::new())?;
    profile_frozen_admits_program(module, &[])
}

/// The whole-program half of the fence, over `user` and the IR of every std seed
/// function it links: facts that hold only across a call (an intrinsic operand
/// that arrives through a parameter, a callee's returned value, the bytes a call
/// tree allocates) are proven here, after every body has passed
/// [`profile_frozen_admits_linked`]. A native build is admitted only when both pass.
pub fn profile_frozen_admits_program(
    user: &IRModule,
    linked: &[IRModule],
) -> Result<(), FrozenProfileRejection> {
    let parts: Vec<&IRModule> = std::iter::once(user).chain(linked).collect();
    admit_frozen_program(&parts, false)
}

/// Explicit bounded allocation policy; all non-allocation fences still apply.
pub fn profile_frozen_admits_program_bounded(
    user: &IRModule,
    linked: &[IRModule],
    bounded: bool,
) -> Result<(), FrozenProfileRejection> {
    let parts: Vec<&IRModule> = std::iter::once(user).chain(linked).collect();
    admit_frozen_program(&parts, bounded)
}

/// [`profile_frozen_admits`] for a module whose calls may also bind to `linked`
/// functions defined OUTSIDE it — the std seed functions the native closure
/// resolved (`native_closure::admit_native_closure_linked`). A call to a linked
/// name is admitted here only because the caller fences each linked body with
/// this same predicate; passing a name whose body is not fenced is the caller's
/// soundness bug, not something this predicate can detect.
pub fn profile_frozen_admits_linked(
    module: &IRModule,
    linked: &std::collections::BTreeSet<String>,
) -> Result<(), FrozenProfileRejection> {
    profile_frozen_admits_linked_bounded(module, linked, false)
}

/// Body admission with an explicit allocation policy, never ambient state.
pub fn profile_frozen_admits_linked_bounded(
    module: &IRModule,
    linked: &std::collections::BTreeSet<String>,
    bounded: bool,
) -> Result<(), FrozenProfileRejection> {
    // Collect the functions this module DEFINES. A call to one of them is in-profile
    // because its body is walked by this same predicate; a call to anything else is
    // not, and cannot be assumed proven. See `admit_call`.
    let mut defined = linked.clone();
    collect_fn_names(&module.instrs, &mut defined);
    admit_instrs_in(
        &module.instrs,
        &defined,
        module,
        &crate::ir::frozen_profile::scalar::types(module, None, &module.instrs),
        bounded,
    )?;
    admit_discard_only_results(&module.instrs)
}

/// Admit the SOURCE-level facts the IR cannot carry: the declared width of a scalar.
///
/// * Float width: any `f32` / `f16` / `bf16` type anywhere in the program is out of
///   profile, because the IR records a scalar float only as `ConstF64` and every native
///   float lowering is proven for f64 alone (see `admit_binop`). The frozen stage1.elf
///   refuses the same programs on its own (`nb_user_float_tok` in `main.mind`); this
///   makes the first fence refuse them too, and name the construct.
/// * Integer width: narrow locals, parameters, aggregate elements, and aliases
///   remain out of profile (`type.non_i64_integer`). Full-width u64 arithmetic
///   is admitted with unsigned facts; ordered unsigned comparisons still refuse.
///   An i32 cast and byte-pointer cast are explicit wave-three exceptions, not
///   permission to use a narrow local/parameter ABI.
pub fn profile_frozen_admits_source(
    module: &crate::ast::Module,
) -> Result<(), FrozenProfileRejection> {
    if crate::eval::narrow_scan::module_mentions_non_f64_float(module) {
        reject("type.non_f64_float")
    } else {
        source::admit(module)
    }
}

fn collect_fn_names(instrs: &[Instr], out: &mut std::collections::BTreeSet<String>) {
    for instr in instrs {
        if let Instr::FnDef { name, body, .. } = instr {
            out.insert(name.clone());
            collect_fn_names(body, out);
        }
    }
}

/// Admit a `Call` by its CALLEE, not by the constructor.
///
/// This is the same correction the 2026-08-21 audit forced on `admit_binop`: a
/// type-blind `BinOp {..} => {}` admitted operators the corpus had never proven, and
/// the fix was to key on the operator. `Instr::Call {..} => {}` had the identical
/// flaw one level up — it admitted EVERY callee, so any construct expressible as a
/// call bypassed the allowlist entirely. Measured: `zeros([4])` lowers to a call, so
/// a tensor program was admitted by this predicate and only stopped by the frozen
/// ELF's own refusal (the SECOND fence). A first fence that defers to the second
/// fence is not a fence.
///
/// In-profile: a call to a function DEFINED in this module (the readiness-gate corpus
/// proves user-function calls, and the callee's body is walked by this predicate),
/// and a call to one of the `frozen_intrinsics::FROZEN_NATIVE_INTRINSICS` with exactly that row's
/// arity, for `read`/`write` an offset operand proven to be `-1`, for `alloc`/`realloc` a
/// size proven `> 0` unless bounded, and (checked module-wide afterwards) every
/// operand proven an i64 slot. Numeric conversions have their own typed rule.
/// Out-of-profile: everything else — builtins, any other intrinsic, and any extern —
/// because "proven for user calls" is not "proven for every name that can appear in
/// a call".
fn admit_call(
    name: &str,
    args: &[ValueId],
    defined: &std::collections::BTreeSet<String>,
    consts: &std::collections::BTreeMap<ValueId, i64>,
    bounded: bool,
) -> Result<(), FrozenProfileRejection> {
    if crate::ir::frozen_profile::scalar::conversion(name).is_some() {
        return if args.len() == 1 {
            Ok(())
        } else {
            reject("call.intrinsic_arity")
        };
    }
    if defined.contains(name) {
        return Ok(());
    }
    let Some(spec) = frozen_native_intrinsic(name) else {
        // Leaked as a stable label rather than the raw name so the diagnostic is
        // deterministic; the CLI prints the construct class, and the callee appears
        // in the compiler's own error when the second fence also refuses.
        return reject("call.undefined_or_builtin");
    };
    admit_intrinsic_call(spec, args, consts, bounded)
}

fn reject(construct: &'static str) -> Result<(), FrozenProfileRejection> {
    Err(FrozenProfileRejection { construct })
}

/// Div/rem key on both operand types. Shifts mask the count with 63;
/// signed right shift is arithmetic and unsigned right shift is logical.
fn admit_binop(
    op: &BinOp,
    lhs: Option<crate::ir::frozen_profile::scalar::Scalar>,
    rhs: Option<crate::ir::frozen_profile::scalar::Scalar>,
) -> Result<(), FrozenProfileRejection> {
    use crate::ir::frozen_profile::scalar::Scalar::{Signed, Unsigned};
    let ints = matches!(lhs, Some(Signed | Unsigned)) && lhs == rhs;
    match op {
        BinOp::Div if !ints => reject("binop.div_operand_type"),
        BinOp::Mod if !ints => reject("binop.mod_operand_type"),
        #[cfg(feature = "std-surface")]
        BinOp::Shl
            if !matches!(
                (lhs, rhs),
                (Some(Signed | Unsigned), Some(Signed | Unsigned))
            ) =>
        {
            reject("binop.shl_operand_type")
        }
        #[cfg(feature = "std-surface")]
        BinOp::Shr
            if !matches!(
                (lhs, rhs),
                (Some(Signed | Unsigned), Some(Signed | Unsigned))
            ) =>
        {
            reject("binop.shr_operand_type")
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
            if lhs == Some(Unsigned) || rhs == Some(Unsigned) =>
        {
            reject("type.non_i64_integer")
        }
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Lt
        | BinOp::Le
        | BinOp::Gt
        | BinOp::Ge
        | BinOp::Eq
        | BinOp::Ne => Ok(()),
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => Ok(()),
    }
}

fn admit_instrs_in(
    instrs: &[Instr],
    defined: &std::collections::BTreeSet<String>,
    module: &IRModule,
    types: &std::collections::BTreeMap<ValueId, crate::ir::frozen_profile::scalar::Scalar>,
    bounded: bool,
) -> Result<(), FrozenProfileRejection> {
    let mut consts = std::collections::BTreeMap::new();
    for instr in instrs {
        fold_const(instr, &mut consts);
        match instr {
            // ---- IN PROFILE: scalar consts, arithmetic, calls, control flow ----
            // Proven native by the RI-D1 readiness gate: int arith, scalar match,
            // if/while/break/continue, user-function calls, and fixed-array
            // subscript READS. Descend into nested bodies.
            //
            // Intrinsic calls are admitted by CALLEE and ARITY against
            // `FROZEN_NATIVE_INTRINSICS` — the names the frozen `nb_intrinsic_id`
            // recognises, each proven by an `IN_PROFILE` behaviour row. That covers
            // the aggregate ABI (`__mind_alloc` / `__mind_store_i64` /
            // `__mind_load_i64`) behind struct return. Wave-three conversions
            // have a separate numeric operand/result contract (native_scalar).
            Instr::Call { name, args, .. } => {
                admit_call(name, args, defined, &consts, bounded)?;
                if crate::ir::frozen_profile::scalar::conversion(name).is_some()
                    && !types.contains_key(&args[0])
                {
                    reject("call.conversion_operand_type")?;
                }
            }
            Instr::ConstI64(..)
            | Instr::ConstF64(..)
            | Instr::Return { .. }
            | Instr::Param { .. }
            | Instr::Output(..) => {}
            // New arithmetic pairs require an operand-type proof, including
            // caller/return and region edges in the whole-program fence.
            Instr::BinOp { op, lhs, rhs, .. } => {
                admit_binop(op, types.get(lhs).copied(), types.get(rhs).copied())?
            }
            Instr::FnDef { name, body, .. } => admit_instrs_in(
                body,
                defined,
                module,
                &crate::ir::frozen_profile::scalar::types(module, Some(name), body),
                bounded,
            )?,
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                admit_instrs_in(cond_instrs, defined, module, types, bounded)?;
                admit_instrs_in(body, defined, module, types, bounded)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                admit_instrs_in(cond_instrs, defined, module, types, bounded)?;
                admit_instrs_in(then_instrs, defined, module, types, bounded)?;
                admit_instrs_in(else_instrs, defined, module, types, bounded)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { .. }
            | Instr::ArrayLoad { .. }
            | Instr::Break { .. }
            | Instr::Continue { .. }
            | Instr::ExternFnDecl { .. } => {}

            // ---- OUT OF PROFILE: aggregate MUTATION (row 12, RI-F) ----
            // `ArrayStore` (`a[i] = v`) was admitted with the read-only array ops
            // above purely because it shares their CONSTRUCTOR family — the same
            // constructor-vs-denotation defect the 2026-08-21 audit found in
            // `admit_binop` and the callee fix found in `admit_call`. Reading a
            // fixed array and WRITING one are different native emitter paths:
            // `ConstArray`/`ArrayLoad` are corpus-proven (readiness-gate program
            // `array_idx_loop`), `ArrayStore` is not ported to the frozen
            // stage1.elf at all.
            //
            // Measured on the shipped stage1.elf (2026-08-29), both directions:
            //   `let mut a=[1,2,3,4]; return a[0];`        -> admitted, runs, 1  (correct)
            //   `let mut a=[1,2,3,4]; a[0]=9; return a[0];` -> ADMITTED here, then
            //       refused by the SECOND fence: "error[backend-native]:
            //       unsupported construct". `mut` alone is fine, so the store is
            //       the differential.
            // Independently corroborated by `rh_f64_aggregate_canary_smoke.py`,
            // which records the native leg as fail-closed because "store/array-param
            // not yet ported to native".
            //
            // A first fence that defers to the second fence is not a fence: this
            // rejection is byte-neutral and capability-neutral (the build already
            // failed with rc=3 and no artifact) and only moves the refusal earlier,
            // where it can NAME the construct.
            //
            // deferred: re-admit `ArrayStore` only together with (a) a reseeded
            // stage1.elf that actually emits it and (b) a new `IN_PROFILE` entry in
            // `ri_d1_frozen_profile_gate.py` proving it byte-identically — the
            // allowlist tracks the FROZEN ELF, never main.mind source, and must be
            // re-admitted element-type-keyed (an f64 element is a different store
            // path from an i64 element) rather than by constructor.
            #[cfg(feature = "std-surface")]
            Instr::ArrayStore { .. } => reject("aggregate.array_store")?,

            // ---- OUT OF PROFILE: tensor surface (RI-E) ----
            // The frozen stage1.elf fail-closes on these (readiness gate: tensor
            // programs → error[backend-native]); admitting them would risk a
            // silent miscompile, so reject LOUDLY here first.
            Instr::ConstTensor(..) => reject("const-tensor")?,
            Instr::ConstDenseTensor { .. } => reject("const-dense-tensor")?,
            Instr::Sum { .. } => reject("tensor.sum")?,
            Instr::Mean { .. } => reject("tensor.mean")?,
            Instr::Relu { .. } => reject("tensor.relu")?,
            Instr::ReluGrad { .. } => reject("tensor.relu_grad")?,
            Instr::Reshape { .. } => reject("tensor.reshape")?,
            Instr::ExpandDims { .. } => reject("tensor.expand_dims")?,
            Instr::Squeeze { .. } => reject("tensor.squeeze")?,
            Instr::Transpose { .. } => reject("tensor.transpose")?,
            Instr::Dot { .. } => reject("tensor.dot")?,
            Instr::MatMul { .. } => reject("tensor.matmul")?,
            Instr::Conv2d { .. } => reject("tensor.conv2d")?,
            Instr::Conv2dGradInput { .. } => reject("tensor.conv2d_grad_input")?,
            Instr::Conv2dGradFilter { .. } => reject("tensor.conv2d_grad_filter")?,
            Instr::Index { .. } => reject("tensor.index")?,
            Instr::Slice { .. } => reject("tensor.slice")?,
            Instr::Gather { .. } => reject("tensor.gather")?,
            Instr::SparseAttr { .. } => reject("sparse_attr")?,

            // ---- OUT OF PROFILE: region + SIMD/BLAS vector ops (RI-E) ----
            #[cfg(feature = "std-surface")]
            Instr::Region { .. } => reject("region")?,
            #[cfg(feature = "std-surface")]
            Instr::VecLoad { .. }
            | Instr::VecFma { .. }
            | Instr::VecReduceAdd { .. }
            | Instr::VecStore { .. }
            | Instr::VecLoadI32 { .. }
            | Instr::VecMulAddQ16 { .. }
            | Instr::VecReduceAddI64 { .. } => reject("simd-vector-op")?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Instr, ValueId};

    /// Test-side mirror of [`profile_frozen_admits`] for a bare instruction slice.
    /// It builds the defined-function set with the SAME `collect_fn_names` pass the
    /// public entry uses, so a test can never accidentally prove a property under a
    /// different admission context than production.
    fn admits(instrs: &[Instr]) -> Result<(), FrozenProfileRejection> {
        let mut defined = std::collections::BTreeSet::new();
        collect_fn_names(instrs, &mut defined);
        admit_instrs_in(
            instrs,
            &defined,
            &IRModule::default(),
            &Default::default(),
            false,
        )
    }

    fn fndef(body: Vec<Instr>) -> Instr {
        Instr::FnDef {
            name: "f".into(),
            params: vec![],
            ret_id: None,
            body,
            reap_threshold: None,
            semantic_types: None,
            #[cfg(feature = "std-surface")]
            value_types: Default::default(),
        }
    }

    fn matmul() -> Instr {
        Instr::MatMul {
            dst: ValueId(2),
            a: ValueId(0),
            b: ValueId(1),
        }
    }

    #[test]
    fn scalar_body_is_admitted() {
        // fn f() -> i64 { return 42 } — pure scalar, in profile.
        let instrs = vec![fndef(vec![
            Instr::ConstI64(ValueId(0), 42),
            Instr::Return {
                value: Some(ValueId(0)),
            },
        ])];
        assert_eq!(admits(&instrs), Ok(()));
    }

    #[test]
    fn tensor_matmul_is_rejected_by_name() {
        let err = admits(&[matmul()]).unwrap_err();
        assert_eq!(err.construct, "tensor.matmul");
    }

    #[test]
    fn tensor_hidden_in_fn_body_is_still_rejected() {
        // The offender must not hide inside a nested function body.
        let err = admits(&[fndef(vec![matmul()])]).unwrap_err();
        assert_eq!(err.construct, "tensor.matmul");
    }

    #[test]
    fn public_wrapper_delegates() {
        // The IRModule wrapper walks module.instrs — smoke it via the helper it
        // delegates to, so this file needs no full IRModule literal.
        assert!(admits(&[Instr::Output(ValueId(0))]).is_ok());
    }

    fn binop(op: BinOp) -> Instr {
        Instr::BinOp {
            dst: ValueId(2),
            op,
            lhs: ValueId(0),
            rhs: ValueId(1),
        }
    }

    #[test]
    fn corpus_proven_binops_are_admitted() {
        // The 5-program readiness corpus exercises `+` (Add) and `<` (Lt, the for-loop
        // desugar); Sub/Mul/other-compares are substrate-invariant on the same axis.
        for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Lt, BinOp::Eq] {
            assert_eq!(admits(&[binop(op)]), Ok(()), "{op:?} should be admitted");
        }
    }

    #[test]
    fn divergent_binops_are_rejected_by_name() {
        // The H5 exploit ops: Div/Mod (i64::MIN/-1, sign-of-remainder) are not in the
        // byte-identity corpus and must be named, not silently admitted.
        assert_eq!(
            admits(&[binop(BinOp::Div)]).unwrap_err().construct,
            "binop.div_operand_type"
        );
        assert_eq!(
            admits(&[binop(BinOp::Mod)]).unwrap_err().construct,
            "binop.mod_operand_type"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn shifts_are_rejected_by_name() {
        // Shl (count >= width UB) / Shr (arith-vs-logical signedness split) — the ops
        // that keep the Q16.16 `(a*b)>>16` ecosystem tier out of the frozen profile.
        assert_eq!(
            admits(&[binop(BinOp::Shl)]).unwrap_err().construct,
            "binop.shl_operand_type"
        );
        assert_eq!(
            admits(&[binop(BinOp::Shr)]).unwrap_err().construct,
            "binop.shr_operand_type"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn array_read_is_admitted_but_array_store_is_rejected() {
        // Row 12: reading a fixed array is corpus-proven (readiness-gate program
        // `array_idx_loop`); WRITING one is not ported to the frozen stage1.elf.
        // They share a constructor family, so admitting them together was the
        // constructor-vs-denotation defect — this pins the split.
        assert_eq!(
            admits(&[Instr::ArrayLoad {
                dst: ValueId(2),
                base: ValueId(0),
                index: ValueId(1),
            }]),
            Ok(())
        );
        assert_eq!(
            admits(&[Instr::ArrayStore {
                dst: ValueId(3),
                base: ValueId(0),
                index: ValueId(1),
                value: ValueId(2),
            }])
            .unwrap_err()
            .construct,
            "aggregate.array_store"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn array_store_hidden_in_fn_body_is_rejected() {
        // Must not slip through the FnDef recursion, same as the Div case below.
        let err = admits(&[fndef(vec![Instr::ArrayStore {
            dst: ValueId(3),
            base: ValueId(0),
            index: ValueId(1),
            value: ValueId(2),
        }])])
        .unwrap_err();
        assert_eq!(err.construct, "aggregate.array_store");
    }

    #[test]
    fn intrinsic_callee_is_rejected_but_user_callee_is_admitted() {
        // Intrinsics outside the frozen emitter's table stay refused; pin BOTH
        // directions so widening the allowlist has to update this test deliberately.
        let call = |name: &str| Instr::legacy_call(ValueId(1), name, vec![]);
        assert_eq!(
            admits(&[call("__mind_conv_unknown")])
                .unwrap_err()
                .construct,
            "call.undefined_or_builtin"
        );
        // `f` IS defined by `fndef`, so a call to it is in profile.
        assert_eq!(admits(&[fndef(vec![]), call("f")]), Ok(()));
    }

    #[test]
    fn divergent_binop_hidden_in_fn_body_is_rejected() {
        // The op-blind admission bug: a Div buried in a function body must not slip
        // through the FnDef recursion.
        let err = admits(&[fndef(vec![binop(BinOp::Div)])]).unwrap_err();
        assert_eq!(err.construct, "binop.div_operand_type");
    }

    // ---- row 10 FLOAT_LANGUAGE_COVERAGE: the NaN-compare divergence ----

    /// A float literal alone stays in profile: `float_as_i64` (`(2.5+4.0) as i64`) is a
    /// readiness-corpus IN-profile program and carries no comparison.
    #[test]
    fn float_arithmetic_without_comparison_is_admitted() {
        let instrs = vec![fndef(vec![
            Instr::ConstF64(ValueId(0), 2.5),
            Instr::ConstF64(ValueId(1), 4.0),
            binop(BinOp::Add),
            Instr::Return {
                value: Some(ValueId(2)),
            },
        ])];
        assert_eq!(admits(&instrs), Ok(()));
    }

    /// A float comparison is now IN profile: the native lowering is IEEE-ordered
    /// (swapped-operand `seta`/`setae` for `<`/`<=`, `sete AND setnp` for `==`,
    /// `setne OR setp` for `!=`) and `ri_d1_frozen_profile_gate.py` proves native == MLIR
    /// == an IEEE oracle over NaN / +-0.0 / +-inf. Before that proof this exact shape was
    /// refused as `binop.compare_in_float_module` (old native exit 3 vs MLIR exit 0).
    #[test]
    fn comparison_in_a_float_module_is_admitted() {
        for op in [
            BinOp::Lt,
            BinOp::Le,
            BinOp::Gt,
            BinOp::Ge,
            BinOp::Eq,
            BinOp::Ne,
        ] {
            let instrs = vec![fndef(vec![
                Instr::ConstF64(ValueId(0), 0.0),
                Instr::ConstF64(ValueId(1), f64::NAN),
                binop(op),
            ])];
            assert_eq!(
                admits(&instrs),
                Ok(()),
                "{op:?} over f64 operands is native == MLIR proven and must be admitted"
            );
        }
    }

    /// Relaxing the float-COMPARE arm must not relax the operators it never proved:
    /// `Div` (and `Mod`) in a float module stay refused by name, exactly as in an
    /// integer module.
    #[test]
    fn div_in_a_float_module_is_still_rejected() {
        for (op, label) in [
            (BinOp::Div, "binop.div_operand_type"),
            (BinOp::Mod, "binop.mod_operand_type"),
        ] {
            let instrs = vec![fndef(vec![
                Instr::ConstF64(ValueId(0), 1.0),
                Instr::ConstF64(ValueId(1), 3.0),
                binop(BinOp::Lt),
                binop(op),
            ])];
            assert_eq!(admits(&instrs).unwrap_err().construct, label);
        }
    }

    /// Integer comparisons keep the profile's proven for-loop (`array_idx_loop`).
    #[test]
    fn integer_comparison_without_float_is_still_admitted() {
        assert_eq!(admits(&[binop(BinOp::Lt)]), Ok(()));
    }

    /// A comparison over float params whose literal sits in ANOTHER fn (the shape the old
    /// module-scoped taint existed for) is admitted too — the admission no longer depends
    /// on where a float literal appears.
    #[test]
    fn float_in_caller_with_comparison_in_another_fn_is_admitted() {
        let caller = Instr::FnDef {
            name: "main".into(),
            params: vec![],
            ret_id: None,
            body: vec![Instr::ConstF64(ValueId(0), 1.5)],
            reap_threshold: None,
            semantic_types: None,
            #[cfg(feature = "std-surface")]
            value_types: Default::default(),
        };
        let callee = fndef(vec![binop(BinOp::Lt)]);
        assert_eq!(admits(&[caller, callee]), Ok(()));
    }

    fn source_admits(src: &str) -> Result<(), FrozenProfileRejection> {
        let module = crate::parser::parse(src).expect("test source parses");
        profile_frozen_admits_source(&module)
    }

    /// The IR half cannot see a declared f32: both literals lower to `ConstF64` and the
    /// compare to a bare `Lt`, so the IR predicate admits it. The source half refuses it.
    #[test]
    fn declared_f32_comparison_is_rejected_on_the_source() {
        let src = "fn main() -> i64 { let x: f32 = 1.0; let y: f32 = 2.0; \
                   if x < y { return 1; } return 0; }";
        let module = crate::parser::parse(src).expect("parses");
        let ir = crate::eval::lower_to_ir(&module).expect("lowers");
        assert_eq!(profile_frozen_admits(&ir), Ok(()), "IR half is type-blind");
        assert_eq!(
            source_admits(src).unwrap_err().construct,
            "type.non_f64_float"
        );
    }

    /// Every spelling that puts a non-f64 float into the program is refused: params,
    /// returns, casts, the call-form cast, struct fields and nested element types.
    #[test]
    fn every_non_f64_float_spelling_is_rejected() {
        for src in [
            "fn f(x: f32) -> i64 { return 0; }",
            "fn f() -> f32 { return 1.0; }",
            "fn main() -> i64 { let a: f64 = 1.0; let b = a as f32; return 0; }",
            "fn main() -> i64 { let a: f64 = 1.0; let b = f32(a); return 0; }",
            "struct P { x: f32 }",
            "fn f(xs: [f32; 4]) -> i64 { return 0; }",
            "fn f(x: f16) -> i64 { return 0; }",
            "fn f(x: bf16) -> i64 { return 0; }",
        ] {
            assert_eq!(
                source_admits(src).map_err(|r| r.construct),
                Err("type.non_f64_float"),
                "{src}"
            );
        }
    }

    /// Positive control: f64 and integer programs, including f64 comparisons, pass.
    #[test]
    fn f64_and_integer_sources_are_admitted() {
        for src in [
            "fn main() -> i64 { let x: f64 = 1.0; let y: f64 = 2.0; \
             if x < y { return 1; } return 0; }",
            "fn d(x: f64, y: f64) -> f64 { return x * y; }",
            "fn main() -> i64 { let n: i64 = 3; return n; }",
            "fn main() -> i64 { let a: f64 = 1.5; return a as i64; }",
            // A tensor element dtype is not a scalar width; the tensor itself is refused
            // elsewhere, under its own name.
            "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; }",
        ] {
            assert_eq!(source_admits(src), Ok(()), "{src}");
        }
    }

    /// An integer declared anything but `i64` is out of profile wherever it is
    /// written. The frozen compiler carries every integer in an i64 slot and does
    /// not re-materialise a `u8`/`u16`/`u32` let or parameter at its width, nor
    /// compare a `u64` unsigned (measured: `let v: u8 = 300; if v > 100` is 1
    /// natively and 2 under MLIR). The IR cannot see a let annotation (an unsigned
    /// one lowers to a bare `BitAnd`), so the refusal is made on the source.
    #[test]
    fn narrow_integer_spellings_are_rejected() {
        for src in [
            "fn main() -> i64 { let v: u8 = 300; if v > 100 { return 1; } return 2; }",
            "fn f(x: u16) -> i64 { return 0; }",
            "fn f(x: u32) -> i64 { return 0; }",
            "fn g() -> u8 { return 3; }",
            "fn f(x: i8) -> i64 { return 0; }",
            "fn f(x: i16) -> i64 { return 0; }",
            "fn f(x: i32) -> i64 { return 0; }",
            "struct P { a: u8 }",
            "type Word = u16;",
            "fn f(xs: [u32; 4]) -> i64 { return 0; }",
            "fn main() -> i64 { let a: i64 = 300; let b = a as u8; return 0; }",
        ] {
            assert_eq!(
                source_admits(src).map_err(|r| r.construct),
                Err("type.non_i64_integer"),
                "{src}"
            );
        }
        // `i64`, and `isize`/`usize` (measured identical on both backends), and `bool`.
        for src in [
            "fn f(x: i64) -> i64 { return x; }",
            "fn f(x: isize) -> usize { return x; }",
            "fn f(b: bool) -> i64 { if b { return 1; } return 2; }",
        ] {
            assert_eq!(source_admits(src), Ok(()), "{src}");
        }
    }
}
