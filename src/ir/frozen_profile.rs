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
//! `find_nondeterministic_call`.
//!
//! Byte-neutral: this is a read-only predicate, called by NOTHING in the emit
//! path today, so it changes zero mic@3 bytes and cannot perturb the keystone.
//! Wiring it into the default-flip decision is a separate, gated slice.

use crate::ir::{BinOp, IRModule, Instr};

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
    admit_instrs(&module.instrs)
}

fn reject(construct: &'static str) -> Result<(), FrozenProfileRejection> {
    Err(FrozenProfileRejection { construct })
}

/// Admit a `BinOp` by OPERATOR. The byte-identity corpus (the RI-D1 readiness gate's
/// frozen-profile programs) proves only substrate-invariant integer ops, so the operators
/// whose RESULT can diverge across substrate — or by operand signedness in a way this
/// type-blind predicate cannot see — are rejected by name. `Div` and `Mod` are rejected
/// (`i64::MIN / -1` overflow, signed-vs-unsigned quotient, sign-of-remainder — none
/// corpus-proven; this compiler's #99 u64 div/rem/shr family). `Shl` and `Shr` are
/// rejected (shift-count >= bit-width is platform-divergent, and `Shr` is arithmetic on
/// signed vs logical on unsigned — a signedness split, again #99). `Add`/`Sub`/`Mul` and
/// the bitwise ops are 2's-complement bit-exact and substrate-identical regardless of
/// signedness, and equality (`Eq`/`Ne`) is a bit-compare, so all are admitted. The
/// ordered comparisons (`Lt`/`Le`/`Gt`/`Ge`) ARE signedness-dependent (i64 `<` is
/// corpus-proven; u64 `<` was the #99 miscompile) — admitted here only because the corpus
/// proves the i64 form and rejecting them would exclude the proven for-loop; that residual
/// u64-comparison gap is why the deferred (op, operand_type) keying below is required
/// before the flip.
///
/// Ecosystem fitment (audit directive 2026-08-21): rejecting Shr KEEPS the Q16.16
/// fixed-point tier — arch-mind metrics, mind-nerve routing, mind-runtime, 512-mind
/// money-path — OUT of the frozen profile, because Q16.16 multiply is `(a*b) >> 16`
/// (needs Shr). That exclusion is CORRECT: those consumers are not yet native-byte-
/// identity-proven, so they must stay on the MLIR backend until Shr is corpus-proven
/// AND operand-type-keyed. Tensor/GPU consumers (mind-runtime kernels, mind-inference)
/// are already excluded via the tensor rejections above.
///
/// deferred: the SOUND fix is (op, operand_type) keying against a corpus-DERIVED pair
/// set — the type axis (i64 Lt proven vs u64 Lt = #99) cannot be decided on `op` alone.
/// Thread `FnDef.value_types` into this predicate; upgrade path tracked in task #313 /
/// the RI-D1 bijection gate.
fn admit_binop(op: &BinOp) -> Result<(), FrozenProfileRejection> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => Ok(()),
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => Ok(()),
        BinOp::Div => reject("binop.div"),
        BinOp::Mod => reject("binop.mod"),
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => Ok(()),
        #[cfg(feature = "std-surface")]
        BinOp::Shl => reject("binop.shl"),
        #[cfg(feature = "std-surface")]
        BinOp::Shr => reject("binop.shr"),
    }
}

fn admit_instrs(instrs: &[Instr]) -> Result<(), FrozenProfileRejection> {
    for instr in instrs {
        match instr {
            // ---- IN PROFILE: scalar consts, arithmetic, calls, control flow ----
            // Proven native by the RI-D1 readiness gate (int arith, f64 scalar,
            // struct-return, scalar match/enum, narrow, calls, if/while/break/
            // continue, fixed-array subscript). Descend into nested bodies.
            Instr::ConstI64(..)
            | Instr::ConstF64(..)
            | Instr::Call { .. }
            | Instr::Return { .. }
            | Instr::Param { .. }
            | Instr::Output(..) => {}
            // BinOp admission is keyed on the OPERATOR, not the constructor (H5, cross-model
            // + corpus audit 2026-08-21). A type-blind `BinOp {..} => {}` admitted the
            // substrate-/signedness-divergent ops (Div/Mod/Shl/Shr) that the byte-identity
            // corpus never proves — see admit_binop for the rejection set + fitment note.
            Instr::BinOp { op, .. } => admit_binop(op)?,
            Instr::FnDef { body, .. } => admit_instrs(body)?,
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                admit_instrs(cond_instrs)?;
                admit_instrs(body)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                admit_instrs(cond_instrs)?;
                admit_instrs(then_instrs)?;
                admit_instrs(else_instrs)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { .. }
            | Instr::ArrayLoad { .. }
            | Instr::ArrayStore { .. }
            | Instr::Break { .. }
            | Instr::Continue { .. }
            | Instr::ExternFnDecl { .. } => {}

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

    fn fndef(body: Vec<Instr>) -> Instr {
        Instr::FnDef {
            name: "f".into(),
            params: vec![],
            ret_id: None,
            body,
            reap_threshold: None,
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
        assert_eq!(admit_instrs(&instrs), Ok(()));
    }

    #[test]
    fn tensor_matmul_is_rejected_by_name() {
        let err = admit_instrs(&[matmul()]).unwrap_err();
        assert_eq!(err.construct, "tensor.matmul");
    }

    #[test]
    fn tensor_hidden_in_fn_body_is_still_rejected() {
        // The offender must not hide inside a nested function body.
        let err = admit_instrs(&[fndef(vec![matmul()])]).unwrap_err();
        assert_eq!(err.construct, "tensor.matmul");
    }

    #[test]
    fn public_wrapper_delegates() {
        // The IRModule wrapper walks module.instrs — smoke it via the helper it
        // delegates to, so this file needs no full IRModule literal.
        assert!(admit_instrs(&[Instr::Output(ValueId(0))]).is_ok());
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
            assert_eq!(
                admit_instrs(&[binop(op)]),
                Ok(()),
                "{op:?} should be admitted"
            );
        }
    }

    #[test]
    fn divergent_binops_are_rejected_by_name() {
        // The H5 exploit ops: Div/Mod (i64::MIN/-1, sign-of-remainder) are not in the
        // byte-identity corpus and must be named, not silently admitted.
        assert_eq!(
            admit_instrs(&[binop(BinOp::Div)]).unwrap_err().construct,
            "binop.div"
        );
        assert_eq!(
            admit_instrs(&[binop(BinOp::Mod)]).unwrap_err().construct,
            "binop.mod"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn shifts_are_rejected_by_name() {
        // Shl (count >= width UB) / Shr (arith-vs-logical signedness split) — the ops
        // that keep the Q16.16 `(a*b)>>16` ecosystem tier out of the frozen profile.
        assert_eq!(
            admit_instrs(&[binop(BinOp::Shl)]).unwrap_err().construct,
            "binop.shl"
        );
        assert_eq!(
            admit_instrs(&[binop(BinOp::Shr)]).unwrap_err().construct,
            "binop.shr"
        );
    }

    #[test]
    fn divergent_binop_hidden_in_fn_body_is_rejected() {
        // The op-blind admission bug: a Div buried in a function body must not slip
        // through the FnDef recursion.
        let err = admit_instrs(&[fndef(vec![binop(BinOp::Div)])]).unwrap_err();
        assert_eq!(err.construct, "binop.div");
    }
}
