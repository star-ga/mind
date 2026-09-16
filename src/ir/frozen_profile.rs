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

#[cfg(feature = "std-surface")]
use crate::ast::TypeAnn;
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
    // Collect the functions this module DEFINES. A call to one of them is in-profile
    // because its body is walked by this same predicate; a call to anything else is
    // not, and cannot be assumed proven. See `admit_call`.
    let mut defined = std::collections::BTreeSet::new();
    collect_fn_names(&module.instrs, &mut defined);
    // Float taint is computed over the WHOLE module before the walk, for the same
    // order-independence reason `fp_mode::collect_extern_float_rets` is a pre-pass: a
    // comparison can appear textually before the float literal that types its operands.
    let float_taint = module_has_float_literal(&module.instrs);
    // Unsigned taint (RH-native Div/Mod + ordered-compare admission, arch review
    // ruling 2026-09-16). The native emitter has ONLY signed `idiv` and signed
    // `setcc`; admitting Div/Mod or an ORDERED compare on a full-width UNSIGNED
    // operand silently miscompiles vs MLIR (`divui` / `setb`/`seta` — the #99
    // divergence, measured live for `u64 <`). The one door a full-width unsigned
    // value enters through that this walk cannot otherwise see is a fn PARAMETER
    // or RETURN type (`Instr::Param` is type-blind); `let x: u64` / `x as u64` /
    // field / element loads all emit a `__mind_conv_u64` `Call` already rejected
    // by `admit_call`. So a module-wide taint over `fn_signatures` — mirroring
    // `float_taint`, with the same order-independence — closes the param/return
    // door and makes SIGNED-only Div/Mod + ordered compares sound. LOAD-BEARING:
    // `admit_call`'s refusal of `__mind_conv_u64` must stay (see its note) or this
    // taint must additionally cover that intrinsic's `dst`.
    let unsigned_taint = module_has_unsigned_signature(module);
    admit_instrs_in(&module.instrs, &defined, float_taint, unsigned_taint)
}

/// True when any function signature declares a full-width UNSIGNED parameter or
/// return (`u64` / `usize`). See the taint rationale in `profile_frozen_admits`.
/// NARROW unsigned (`u8`/`u16`/`u32`) is deliberately NOT tainted: it is masked
/// to its width and zero-extends into a NON-NEGATIVE i64, so the signed `idiv` /
/// `setcc` gives the same result as the unsigned op — only a full-width carrier
/// whose high bit is set (u64/usize) can diverge. `usize` is included as
/// future-proofing: MLIR seeds it `ScalarI64` (signed) today so it does not
/// diverge yet, but a later usize->unsigned fix would silently split the backends.
#[cfg(feature = "std-surface")]
fn module_has_unsigned_signature(module: &IRModule) -> bool {
    module
        .fn_signatures
        .values()
        .any(|(params, ret)| params.iter().chain(ret.iter()).any(is_unsigned_wide_type))
}

/// Without `std-surface` the IR carries NO `fn_signatures` side-table (the field
/// is `#[cfg(feature = "std-surface")]` on `IRModule`), so the one door a
/// full-width unsigned value enters through cannot be inspected at all. An
/// uninspectable door is treated as OPEN: report the module unsigned-tainted, which
/// REFUSES Div/Mod and ordered compares instead of admitting them unchecked —
/// the same "undecidable => refuse" rule the native bridge applies to a module it
/// cannot lower. This costs nothing real: `While`/`If` are themselves
/// std-surface-only, so a non-std-surface module has no loop for an ordered
/// compare to guard. (Found by `cargo check --no-default-features`, which the
/// std-surface-only unit gate could not see.)
#[cfg(not(feature = "std-surface"))]
fn module_has_unsigned_signature(_module: &IRModule) -> bool {
    true
}

#[cfg(feature = "std-surface")]
fn is_unsigned_wide_type(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Named(n) if n == "u64" || n == "usize")
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
/// proves user-function calls, and the callee's body is walked by this predicate).
/// Out-of-profile: everything else — builtins, intrinsics, and any extern — because
/// "proven for user calls" is not "proven for every name that can appear in a call".
fn admit_call(
    name: &str,
    defined: &std::collections::BTreeSet<String>,
) -> Result<(), FrozenProfileRejection> {
    if defined.contains(name) {
        Ok(())
    } else {
        // Leaked as a stable label rather than the raw name so the diagnostic is
        // deterministic; the CLI prints the construct class, and the callee appears
        // in the compiler's own error when the second fence also refuses.
        reject("call.undefined_or_builtin")
    }
}

/// Does this module contain a float literal anywhere?
///
/// This is the float-COMPARE taint seed (see [`admit_binop`]). It is deliberately
/// MODULE-scoped rather than per-`FnDef`: `Instr::Param` carries `{dst, name, index}` and
/// NO type, so a callee like `fn lt(x: f64, y: f64) -> i64 { if x < y {…} }` holds a float
/// comparison with no float literal in its own body — the float literal sits in the
/// CALLER. A per-body scan would admit exactly that function. Module scope is the
/// fail-CLOSED reading of the same fact.
///
/// Complete because the profile admits no other float producer: the tensor / SIMD arms
/// reject every vector and tensor float, `binop.div` is rejected, and `admit_call` admits
/// only module-DEFINED callees — so the conv-to-float intrinsics `__mind_conv_f32` /
/// `__mind_conv_f64` (the `as f32` / `as f64` lowering, `src/eval/lower.rs`) are refused
/// and `Instr::ConstF64` is the only door a float can come through.
///
/// LOAD-BEARING (unsigned, 2026-09-16): `__mind_conv_u64` MUST likewise stay off
/// any future `admit_call` allowlist, or `module_has_unsigned_signature`'s taint
/// must additionally cover its `dst` — otherwise `let x: u64 = y as u64; x / z`
/// re-opens the unsigned-div hole with no u64 in any `fn_signatures`.
///
/// LOAD-BEARING for whoever relaxes `admit_call`: the moment a compiler-synthesised
/// callee allowlist is introduced there (the deferred bijection fix noted in
/// `admit_instrs_in`), `__mind_conv_f32` / `__mind_conv_f64` MUST stay off it, or this
/// seed must additionally taint from their `dst` — otherwise `let f: f64 = x as f64;`
/// re-opens the float-compare hole with no `ConstF64` anywhere in the module.
fn module_has_float_literal(instrs: &[Instr]) -> bool {
    instrs.iter().any(|instr| match instr {
        Instr::ConstF64(..) => true,
        // The body-carrying admitted variants: a float literal must not hide in a nest.
        Instr::FnDef { body, .. } => module_has_float_literal(body),
        #[cfg(feature = "std-surface")]
        Instr::While {
            cond_instrs, body, ..
        } => module_has_float_literal(cond_instrs) || module_has_float_literal(body),
        #[cfg(feature = "std-surface")]
        Instr::If {
            cond_instrs,
            then_instrs,
            else_instrs,
            ..
        } => {
            module_has_float_literal(cond_instrs)
                || module_has_float_literal(then_instrs)
                || module_has_float_literal(else_instrs)
        }
        // Everything else. This blanket arm is the one place in this file that is NOT an
        // exhaustive match, so it owes an argument: a variant reached here is either
        //   (a) one of the remaining IN-profile arms of `admit_instrs_in` — `Call`,
        //       `ConstI64`, `Return`, `Param`, `Output`, `BinOp`, `ConstArray`,
        //       `ArrayLoad`, `Break`, `Continue`, `ExternFnDecl` — none of which carries
        //       an f64 payload or a nested body; or
        //   (b) OUT of profile, in which case `admit_instrs_in` rejects the module before
        //       any comparison verdict is reachable, so a missed seed cannot matter.
        // A future variant is therefore forced through the exhaustive match in
        // `admit_instrs_in` first: admitting it is the deliberate decision point, and
        // whoever makes it must ask here whether it can carry or produce a float.
        _ => false,
    })
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
/// FLOAT COMPARISON — WEDGE-BREAKING, rejected (row 10 FLOAT_LANGUAGE_COVERAGE).
/// `float_taint` is true when the module contains a float literal
/// ([`module_has_float_literal`]); every ordered/equality comparison in such a module is
/// refused, because the native path's float compare is NOT IEEE and DISAGREES WITH THE
/// MLIR PATH on the same source:
///
///   native (`examples/mindc_mind/main.mind::nb_fp_setcc_opcode`): `ucomisd` + an UNSIGNED
///     `setcc`. An unordered compare sets CF=ZF=PF=1, so `setb`(Lt) / `setbe`(Le) /
///     `sete`(Eq) read TRUE for a NaN operand and `setne`(Ne) reads FALSE.
///   MLIR (`src/mlir/lowering.rs`): `arith.cmpf "olt"/"ole"/"oeq"` and `"une"` — the
///     ORDERED/IEEE predicates, all false for NaN (Ne true).
///
/// MEASURED, both backends built from this tree, one source file:
///   `let a: f64 = 65536.0; …6 squarings→ +inf; let n = h - h;  // NaN`
///   `if n == 0.0 { if n < 0.0 { return 3; } return 1; } if n < 0.0 { return 2; } return 0;`
///   native exit 3 (`n == 0.0` AND `n < 0.0` BOTH true — impossible for any real number,
///   the ucomisd-unordered signature) vs MLIR exit 0. Per-operator: Lt/Le/Eq/Ne diverge,
///   Gt/Ge agree. Every construct in that program (`ConstF64`, `Mul`, `Sub`, `Lt`, `Eq`)
///   was ADMITTED by this predicate and the frozen stage1.elf compiled it, so BOTH fences
///   passed and the native artifact carried a silently wrong value — precisely the failure
///   mode the allowlist exists to prevent.
///
/// The rejection is module-wide (integer comparisons included) rather than float-only
/// because `Instr::Param` carries no type — see [`module_has_float_literal`]. That is
/// over-rejection: loud and recoverable, the safe direction.
///
/// deferred: the SOUND fix is (op, operand_type) keying against a corpus-DERIVED pair
/// set — the type axis (i64 Lt proven vs u64 Lt = #99, f64 Lt = the NaN divergence above)
/// cannot be decided on `op` alone. Thread `FnDef.value_types` into this predicate;
/// upgrade path tracked in task #313 / the RI-D1 bijection gate. The float half of it
/// additionally needs a NaN-aware native predicate (parity-flag masking for `==`/`!=`,
/// swapped-operand `seta`/`setae` for `<`/`<=`) landed in `nb_fp_setcc_opcode` FIRST —
/// only then may this arm relax, and only with a NaN program added to
/// `ri_d1_frozen_profile_gate.py::IN_PROFILE` proving native == MLIR.
fn admit_binop(
    op: &BinOp,
    float_taint: bool,
    unsigned_taint: bool,
) -> Result<(), FrozenProfileRejection> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => Ok(()),
        // `==` / `!=` are bit compares — sign-agnostic; only a float taints them.
        BinOp::Eq | BinOp::Ne => {
            if float_taint {
                reject("binop.compare_in_float_module")
            } else {
                Ok(())
            }
        }
        // ORDERED compares emit a signed `setcc`; an unsigned operand needs
        // `setb`/`seta` (the #99 divergence, measured live for `u64 <`), so an
        // unsigned-tainted module is rejected here too.
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
            if float_taint {
                reject("binop.compare_in_float_module")
            } else if unsigned_taint {
                reject("binop.ordered_compare_unsigned")
            } else {
                Ok(())
            }
        }
        // RH-native (arch review 2026-09-16): the native emitter's guarded
        // signed `idiv`/mod is emitter-proven byte/edge-correct and native==MLIR
        // for SIGNED i64 (div_shift_cmp_edge_smoke). Admit it, but reject in a
        // float- OR unsigned-tainted module — there is no unsigned `div` arm, so
        // `idiv` on an unsigned operand would silently miscompile.
        BinOp::Div | BinOp::Mod => {
            if float_taint {
                reject("binop.div_mod_in_float_module")
            } else if unsigned_taint {
                reject("binop.div_mod_unsigned")
            } else {
                Ok(())
            }
        }
        #[cfg(feature = "std-surface")]
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => Ok(()),
        #[cfg(feature = "std-surface")]
        BinOp::Shl => reject("binop.shl"),
        #[cfg(feature = "std-surface")]
        BinOp::Shr => reject("binop.shr"),
    }
}

fn admit_instrs_in(
    instrs: &[Instr],
    defined: &std::collections::BTreeSet<String>,
    float_taint: bool,
    unsigned_taint: bool,
) -> Result<(), FrozenProfileRejection> {
    for instr in instrs {
        match instr {
            // ---- IN PROFILE: scalar consts, arithmetic, calls, control flow ----
            // Proven native by the RI-D1 readiness gate: int arith, scalar match,
            // if/while/break/continue, user-function calls, and fixed-array
            // subscript READS. Descend into nested bodies.
            //
            // NOT listed here any more, and deliberately: struct-return, f64
            // scalar, and narrow (`as`) casts. The earlier version of this comment
            // claimed them, but they all lower to compiler intrinsic `Instr::Call`s
            // (`__mind_alloc` / `__mind_store_i64` / `__mind_load_i64` for the
            // aggregate ABI, `__mind_conv_i64` for a full-width cast), and
            // `admit_call` admits only callees DEFINED in the module. Measured
            // 2026-08-29: 3 of the 5 `IN_PROFILE` readiness-gate programs
            // (`float_as_i64`, `struct_return`, `narrow_u8_wrap`) are refused
            // `call.undefined_or_builtin`. That is an OVER-rejection — fail-closed
            // and safe, but it means the allowlist is not in bijection with the
            // corpus and `ri_d1_frozen_profile_gate.py` is RED.
            //
            // deferred: restore them by extending `admit_call` with the
            // corpus-DERIVED intrinsic set {__mind_alloc, __mind_store_i64,
            // __mind_load_i64, __mind_conv_i64}. NOT done here because a name-keyed
            // intrinsic admission repeats the very defect this file keeps fixing:
            // the corpus proves `__mind_store_i64` only for an i64 struct field,
            // and an f64 field is a different native store path. The sound form
            // needs the operand/field type, i.e. the same (op, operand_type) keying
            // `admit_binop` already defers — thread `FnDef.value_types` through
            // first. Tracked on matrix row 12 / the RI-D1 bijection gate.
            Instr::Call { name, .. } => admit_call(name, defined)?,
            Instr::ConstI64(..)
            | Instr::ConstF64(..)
            | Instr::Return { .. }
            | Instr::Param { .. }
            | Instr::Output(..) => {}
            // BinOp admission is keyed on the OPERATOR, not the constructor (H5, cross-model
            // + corpus audit 2026-08-21). A type-blind `BinOp {..} => {}` admitted the
            // substrate-/signedness-divergent ops (Div/Mod/Shl/Shr) that the byte-identity
            // corpus never proves — see admit_binop for the rejection set + fitment note.
            Instr::BinOp { op, .. } => admit_binop(op, float_taint, unsigned_taint)?,
            Instr::FnDef { body, .. } => admit_instrs_in(body, defined, float_taint, unsigned_taint)?,
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                admit_instrs_in(cond_instrs, defined, float_taint, unsigned_taint)?;
                admit_instrs_in(body, defined, float_taint, unsigned_taint)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                admit_instrs_in(cond_instrs, defined, float_taint, unsigned_taint)?;
                admit_instrs_in(then_instrs, defined, float_taint, unsigned_taint)?;
                admit_instrs_in(else_instrs, defined, float_taint, unsigned_taint)?;
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
        let float_taint = module_has_float_literal(instrs);
        // Bare-slice tests carry no `fn_signatures`; unsigned-taint cases build a
        // full module and call `profile_frozen_admits` directly.
        admit_instrs_in(instrs, &defined, float_taint, false)
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

    /// A module whose `main` signature declares a full-width `u64` parameter,
    /// carrying the given instrs — exercises `module_has_unsigned_signature`
    /// (the bare-slice `admits` helper cannot express `fn_signatures`).
    #[cfg(feature = "std-surface")]
    fn module_with_u64_sig(instrs: &[Instr]) -> IRModule {
        let mut m = IRModule::new();
        m.instrs = instrs.to_vec();
        m.fn_signatures.insert(
            "main".to_string(),
            (vec![TypeAnn::Named("u64".to_string())], None),
        );
        m
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn signed_div_mod_admitted_but_float_and_unsigned_rejected() {
        // RH-native (arch review 2026-09-16): SIGNED i64 Div/Mod are
        // emitter-proven byte/edge-correct and native==MLIR, so a clean
        // (non-float, non-unsigned) module ADMITS them.
        assert_eq!(admits(&[binop(BinOp::Div)]), Ok(()));
        assert_eq!(admits(&[binop(BinOp::Mod)]), Ok(()));
        // FLOAT taint rejects them (no signed idiv on floats).
        assert_eq!(
            admits(&[Instr::ConstF64(ValueId(0), 1.0), binop(BinOp::Div)])
                .unwrap_err()
                .construct,
            "binop.div_mod_in_float_module"
        );
        // UNSIGNED-signature taint rejects them (idiv is signed-only; u64 diverges).
        assert_eq!(
            profile_frozen_admits(&module_with_u64_sig(&[binop(BinOp::Div)]))
                .unwrap_err()
                .construct,
            "binop.div_mod_unsigned"
        );
        assert_eq!(
            profile_frozen_admits(&module_with_u64_sig(&[binop(BinOp::Mod)]))
                .unwrap_err()
                .construct,
            "binop.div_mod_unsigned"
        );
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn ordered_compare_rejected_under_unsigned_sig_but_eq_ne_ok() {
        // ORDERED compares emit signed setcc — an unsigned operand needs setb/seta
        // (the #99 divergence, live for `u64 <`), so an unsigned-sig module rejects.
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            assert_eq!(
                profile_frozen_admits(&module_with_u64_sig(&[binop(op)]))
                    .unwrap_err()
                    .construct,
                "binop.ordered_compare_unsigned",
                "{op:?} must be rejected under an unsigned signature"
            );
        }
        // Eq/Ne are bit compares — sign-agnostic — so they stay admitted.
        for op in [BinOp::Eq, BinOp::Ne] {
            assert_eq!(
                profile_frozen_admits(&module_with_u64_sig(&[binop(op)])),
                Ok(()),
                "{op:?} must stay admitted under an unsigned signature"
            );
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn shifts_are_rejected_by_name() {
        // Shl (count >= width UB) / Shr (arith-vs-logical signedness split) — the ops
        // that keep the Q16.16 `(a*b)>>16` ecosystem tier out of the frozen profile.
        assert_eq!(
            admits(&[binop(BinOp::Shl)]).unwrap_err().construct,
            "binop.shl"
        );
        assert_eq!(
            admits(&[binop(BinOp::Shr)]).unwrap_err().construct,
            "binop.shr"
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
        // The aggregate ABI reaches the IR as intrinsic calls; pin BOTH directions
        // so a future intrinsic allowlist has to update this test deliberately.
        let call = |name: &str| Instr::legacy_call(ValueId(1), name, vec![]);
        assert_eq!(
            admits(&[call("__mind_alloc")]).unwrap_err().construct,
            "call.undefined_or_builtin"
        );
        // `f` IS defined by `fndef`, so a call to it is in profile.
        assert_eq!(admits(&[fndef(vec![]), call("f")]), Ok(()));
    }

    /// Without `std-surface` there is no signature table to inspect, so the taint
    /// must fail CLOSED: Div/Mod and ordered compares refused, bit compares kept.
    /// Mutation-sensitive: returning `false` from the non-std-surface
    /// `module_has_unsigned_signature` admits all six and turns this red.
    #[cfg(not(feature = "std-surface"))]
    #[test]
    fn without_std_surface_the_unsigned_taint_fails_closed() {
        let m = |instrs: &[Instr]| {
            let mut m = IRModule::new();
            m.instrs = instrs.to_vec();
            m
        };
        for op in [BinOp::Div, BinOp::Mod] {
            assert_eq!(
                profile_frozen_admits(&m(&[binop(op)]))
                    .unwrap_err()
                    .construct,
                "binop.div_mod_unsigned",
                "{op:?} must be refused when the signature door cannot be inspected"
            );
        }
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            assert_eq!(
                profile_frozen_admits(&m(&[binop(op)]))
                    .unwrap_err()
                    .construct,
                "binop.ordered_compare_unsigned"
            );
        }
        for op in [BinOp::Eq, BinOp::Ne, BinOp::Add] {
            assert_eq!(profile_frozen_admits(&m(&[binop(op)])), Ok(()));
        }
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn divergent_binop_hidden_in_fn_body_is_rejected() {
        // The op-blind admission bug: a divergent op buried in a function body must
        // not slip through the FnDef recursion. Signed Div is now admitted, so use a
        // Div under an UNSIGNED signature — still divergent (idiv vs divui) — and it
        // must be caught inside the nested fn body by the recursion + taint together.
        let m = module_with_u64_sig(&[fndef(vec![binop(BinOp::Div)])]);
        assert_eq!(
            profile_frozen_admits(&m).unwrap_err().construct,
            "binop.div_mod_unsigned"
        );
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

    /// The wedge-breaking admission. Native lowers a float compare to `ucomisd` + an
    /// UNSIGNED `setcc`, which reads TRUE for a NaN operand on `<`/`<=`/`==`; MLIR lowers
    /// it to `arith.cmpf "olt"/"ole"/"oeq"`, all FALSE for NaN. Measured native exit 3 vs
    /// MLIR exit 0 on one source file. Both fences passed it before this rejection.
    #[test]
    fn comparison_in_a_float_module_is_rejected() {
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
                Instr::ConstF64(ValueId(1), 1.0),
                binop(op),
            ])];
            assert_eq!(
                admits(&instrs).unwrap_err().construct,
                "binop.compare_in_float_module",
                "{op:?} must not be admitted while the module carries a float"
            );
        }
    }

    /// Integer comparisons keep the profile's proven for-loop (`array_idx_loop`) — the
    /// rejection must be conditional on the float, never unconditional.
    #[test]
    fn integer_comparison_without_float_is_still_admitted() {
        assert_eq!(admits(&[binop(BinOp::Lt)]), Ok(()));
    }

    /// The taint is MODULE-scoped on purpose: `Instr::Param` carries no type, so a callee
    /// holding `x < y` over float params has no float literal in its OWN body — the
    /// literal is in the caller. A per-body scan would admit exactly that function.
    #[test]
    fn float_in_caller_taints_a_comparison_in_another_fn() {
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
        // `f`'s body is float-literal-free; only the caller names a float.
        let callee = fndef(vec![binop(BinOp::Lt)]);
        assert_eq!(
            admits(&[caller, callee]).unwrap_err().construct,
            "binop.compare_in_float_module"
        );
    }

    /// The float seed must not hide inside a nested body.
    #[test]
    fn float_nested_in_a_fn_body_still_taints() {
        let instrs = vec![
            fndef(vec![Instr::ConstF64(ValueId(0), 1.5)]),
            binop(BinOp::Eq),
        ];
        assert_eq!(
            admits(&instrs).unwrap_err().construct,
            "binop.compare_in_float_module"
        );
    }
}
