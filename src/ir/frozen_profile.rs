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

use crate::ir::{BinOp, IRModule, Instr, ValueId};

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
    // Aggregate classification is per function: each `FnDef` body mints a fresh
    // ValueId space. The top-level set covers instructions that are not inside a
    // function. `admit_instrs_in` opens a new set at every function boundary.
    let mut aggs = std::collections::BTreeSet::new();
    admit_instrs_in(&module.instrs, &defined, module, &mut aggs, false)
}

/// Refuse the first binary operator whose operand is an array or tensor value.
///
/// The lane profile admits operators [`profile_frozen_admits`] refuses. A
/// whole-array operand is still `binop.array_operand`, classified by the same
/// walk and decided by [`admit_binop_use`]. Scalar-operator refusals are not
/// returned.
pub(crate) fn admit_binop_array_operands(module: &IRModule) -> Result<(), FrozenProfileRejection> {
    let mut defined = std::collections::BTreeSet::new();
    collect_fn_names(&module.instrs, &mut defined);
    let mut aggs = std::collections::BTreeSet::new();
    admit_instrs_in(&module.instrs, &defined, module, &mut aggs, true)
}

/// Admit the SOURCE-level facts the IR cannot carry. Today that is the float width:
/// any `f32` / `f16` / `bf16` type anywhere in the program is out of profile, because
/// the IR records a scalar float only as `ConstF64` and every native float lowering is
/// proven for f64 alone (see `admit_binop`). The frozen stage1.elf refuses the same
/// programs on its own (`nb_user_float_tok` in `main.mind`); this makes the first fence
/// refuse them too, and name the construct.
pub fn profile_frozen_admits_source(
    module: &crate::ast::Module,
) -> Result<(), FrozenProfileRejection> {
    if crate::eval::narrow_scan::module_mentions_non_f64_float(module) {
        reject("type.non_f64_float")
    } else {
        Ok(())
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

fn reject(construct: &'static str) -> Result<(), FrozenProfileRejection> {
    Err(FrozenProfileRejection { construct })
}

/// Keep a frozen refusal only when this walk is the full profile, or when the
/// refusal is the array-operand kind the lane shares.
fn keep_refusal(
    array_operands_only: bool,
    decision: Result<(), FrozenProfileRejection>,
) -> Result<(), FrozenProfileRejection> {
    if !array_operands_only {
        return decision;
    }
    match decision {
        Err(rejection) if rejection.construct == "binop.array_operand" => Err(rejection),
        _ => Ok(()),
    }
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
/// fixed-point tier — arch-mind metrics, mind-nerve routing, mind-runtime, MIND-Law
/// money-path — OUT of the frozen profile, because Q16.16 multiply is `(a*b) >> 16`
/// (needs Shr). That exclusion is CORRECT: those consumers are not yet native-byte-
/// identity-proven, so they must stay on the MLIR backend until Shr is corpus-proven
/// AND operand-type-keyed. Tensor/GPU consumers (mind-runtime kernels, mind-inference)
/// are already excluded via the tensor rejections above.
///
/// FLOAT COMPARISON — admitted, native == MLIR proven for IEEE NaN / +-0.0 / +-inf
/// (row 10 FLOAT_LANGUAGE_COVERAGE). This arm used to refuse every comparison in a module
/// carrying a float literal, because the native lowering was NOT IEEE and DISAGREED WITH
/// THE MLIR PATH on the same source:
///
///   old native (`examples/mindc_mind/main.mind::nb_fp_setcc_opcode`): `ucomisd` + a bare
///     UNSIGNED `setcc`. An unordered compare sets CF=ZF=PF=1, so `setb`(Lt) / `setbe`(Le)
///     / `sete`(Eq) read TRUE for a NaN operand and `setne`(Ne) read FALSE.
///   MLIR (`src/mlir/lowering.rs`): `arith.cmpf "olt"/"ole"/"ogt"/"oge"/"oeq"` and `"une"`
///     — the IEEE predicates, all false for NaN except `!=`.
///
/// MEASURED on `let n = h - h; /* NaN */ if n == 0.0 { if n < 0.0 { return 3; } return 1; }
/// if n < 0.0 { return 2; } return 0;` — old native exit 3 (`n == 0.0` AND `n < 0.0` both
/// true, the ucomisd-unordered signature) vs MLIR exit 0.
///
/// The native lowering now emits the IEEE-ordered forms: `a < b` / `a <= b` as
/// `ucomisd b, a` + `seta` / `setae` (operands SWAPPED; both false when unordered),
/// `a == b` as `sete AND setnp`, `a != b` as `setne OR setp`; `>` / `>=` were already
/// `seta` / `setae`. PROVEN by `ri_d1_frozen_profile_gate.py::IN_PROFILE`: the historical
/// program above plus a truth-table corpus (all six predicates over a NaN on the left, on
/// the right, on both sides, NaN vs +-inf, ordinary finite pairs including two negatives,
/// -0.0 vs +0.0 both ways, -inf vs +inf, inf vs inf, inf vs finite) builds through the
/// frozen stage1.elf and returns the SAME exit as the MLIR backend AND as an IEEE-754
/// oracle computed independently of both backends. The proof covers f64 only, and this
/// arm is type-blind, so it cannot keep an f32 operand out by itself: a float literal
/// lowers to `Instr::ConstF64` whatever its declared width, so `let x: f32 = 1.0;
/// let y: f32 = 2.0; if x < y {..}` reaches here as two `ConstF64` and an `Lt` with no
/// conversion call for `admit_call` to refuse. The f32 exclusion is therefore made on the
/// SOURCE by [`profile_frozen_admits_source`], which the native bridge runs next to this
/// predicate. Add f32 rows to the gate corpus before relaxing that.
///
/// FENCE OWNERSHIP: with the module-scoped taint gone, this profile no longer backstops the
/// operand type of an f64-vs-integer comparison. Keeping a float compare on the f64 path is
/// owned by the native side: the per-slot float classifier in `main.mind` (params from
/// the signature, calls from the return registry, lets, fields, index, cast, neg, if),
/// its mixed-dtype poison, and type-checker E2013. Whoever edits that classifier edits the
/// fence. Encoding hazard in the same lowering: the parity fix-up (`setnp ah` / `setp ah`,
/// then `and`/`or al, ah`) relies on NO REX prefix on those bytes; a REX byte would turn
/// `ah` into `spl` and silently drop the NaN mask.
///
/// deferred: the SOUND fix for the residual u64-comparison gap noted above is
/// (op, operand_type) keying against a corpus-DERIVED pair set — the type axis (i64 Lt
/// proven vs u64 Lt = #99) cannot be decided on `op` alone. Thread `FnDef.value_types`
/// into this predicate; upgrade path tracked in task #313 / the RI-D1 bijection gate.
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

/// Refuse a binary operator when either operand is an array or tensor value.
///
/// `ConstArray` and `BinOp::Add` (and `Sub` / `Mul`, the comparisons, and the
/// bitwise ops) are each in profile on their own, so `[i64; N] + [i64; N]` was
/// admitted. The frozen native compiler then adds the two base addresses and
/// the artifact runs and returns the wrong value. An element read is
/// `ArrayLoad`, whose destination is a scalar, so `a[i] + b[i]` stays admitted.
///
/// The operand class is decided before [`admit_binop`], so an array `/` is
/// named `binop.array_operand` rather than `binop.div`. A scalar `/` is still
/// `binop.div`. `bytes[N]` and slices are not aggregate values: indexing them
/// lowers to a pointer add, and marking them would refuse that add.
///
/// Shared with the lane path ([`admit_binop_array_operands`]). The lane calls
/// this helper; it does not reimplement the operand test.
pub(crate) fn admit_binop_use(
    op: &BinOp,
    lhs: ValueId,
    rhs: ValueId,
    aggs: &std::collections::BTreeSet<ValueId>,
) -> Result<(), FrozenProfileRejection> {
    if aggs.contains(&lhs) || aggs.contains(&rhs) {
        return reject("binop.array_operand");
    }
    admit_binop(op)
}

#[cfg(feature = "std-surface")]
fn type_is_aggregate(ty: &crate::ast::TypeAnn) -> bool {
    match ty {
        crate::ast::TypeAnn::Array { .. }
        | crate::ast::TypeAnn::Tensor { .. }
        | crate::ast::TypeAnn::DiffTensor { .. }
        | crate::ast::TypeAnn::SparseTensor { .. } => true,
        crate::ast::TypeAnn::Generic { name, .. } if name == "array" => true,
        _ => false,
    }
}

/// Seed the aggregate set for one function. ValueIds restart at each `FnDef`,
/// so this must not inherit the caller's set. A parameter is an aggregate only
/// when its declared type is an array or tensor. `bytes[N]` is `Named`, a
/// slice is `Slice`, and neither is included: their index lowering is a
/// pointer add of a scalar address.
#[cfg(feature = "std-surface")]
fn seed_fn_aggregates(
    name: &str,
    params: &[(String, ValueId)],
    value_types: &std::collections::BTreeMap<ValueId, crate::ir::ArrayType>,
    module: &IRModule,
    aggs: &mut std::collections::BTreeSet<ValueId>,
) {
    if let Some((tys, _)) = module.fn_signatures.get(name) {
        for ((_, vid), ty) in params.iter().zip(tys.iter()) {
            if type_is_aggregate(ty) {
                aggs.insert(*vid);
            }
        }
    }
    for vid in value_types.keys() {
        aggs.insert(*vid);
    }
}

#[cfg(feature = "std-surface")]
fn note_aggregate_call(
    name: &str,
    dst: ValueId,
    module: &IRModule,
    aggs: &mut std::collections::BTreeSet<ValueId>,
) {
    if let Some((_, Some(ret))) = module.fn_signatures.get(name) {
        if type_is_aggregate(ret) {
            aggs.insert(dst);
        }
    }
}

/// A loop-carried aggregate is the same value under three ids (the pre-loop
/// value, the back-edge value, and the value after the loop). If any one of
/// them is an aggregate, the other two are too, including a use inside the body.
#[cfg(feature = "std-surface")]
fn propagate_while(
    init_ids: &[ValueId],
    live_vars: &[(String, ValueId)],
    exit_ids: &[ValueId],
    aggs: &mut std::collections::BTreeSet<ValueId>,
) {
    let n = init_ids.len().min(live_vars.len()).min(exit_ids.len());
    for i in 0..n {
        let carried = [init_ids[i], live_vars[i].1, exit_ids[i]];
        if carried.iter().any(|id| aggs.contains(id)) {
            for id in carried {
                aggs.insert(id);
            }
        }
    }
}

// `module` supplies the signature table used to classify array and tensor
// values. That classification is compiled only under `std-surface`. Without
// the feature the parameter is only passed into the nested-function walk.
#[cfg_attr(not(feature = "std-surface"), allow(clippy::only_used_in_recursion))]
fn admit_instrs_in(
    instrs: &[Instr],
    defined: &std::collections::BTreeSet<String>,
    module: &IRModule,
    aggs: &mut std::collections::BTreeSet<ValueId>,
    array_operands_only: bool,
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
            Instr::Call { name, dst, .. } => {
                // A call that returns an array or tensor produces an aggregate
                // value. Record it before the callee check so a later binop on
                // that result is named `binop.array_operand` when the call
                // itself is in profile.
                #[cfg(feature = "std-surface")]
                note_aggregate_call(name, *dst, module, aggs);
                #[cfg(not(feature = "std-surface"))]
                let _ = dst;
                keep_refusal(array_operands_only, admit_call(name, defined))?
            }
            Instr::ConstI64(..)
            | Instr::ConstF64(..)
            | Instr::Return { .. }
            | Instr::Param { .. }
            | Instr::Output(..) => {}
            // BinOp admission is keyed on the OPERATOR, not the constructor (H5, cross-model
            // + corpus audit 2026-08-21). A type-blind `BinOp {..} => {}` admitted the
            // substrate-/signedness-divergent ops (Div/Mod/Shl/Shr) that the byte-identity
            // corpus never proves — see admit_binop for the rejection set + fitment note.
            // The operand class is checked first (`binop.array_operand`): an array or
            // tensor value used as an operand is a different native path from the
            // scalar operator, and the frozen compiler miscompiles it.
            Instr::BinOp { op, lhs, rhs, .. } => {
                keep_refusal(array_operands_only, admit_binop_use(op, *lhs, *rhs, aggs))?
            }
            // Each function has its own ValueId space. Classify aggregates from
            // this function's signature and body, not from the caller.
            Instr::FnDef { body, .. } => {
                let mut child = std::collections::BTreeSet::new();
                #[cfg(feature = "std-surface")]
                if let Instr::FnDef {
                    name,
                    params,
                    value_types,
                    ..
                } = instr
                {
                    seed_fn_aggregates(name, params, value_types, module, &mut child);
                }
                admit_instrs_in(body, defined, module, &mut child, array_operands_only)?;
            }
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs,
                body,
                init_ids,
                live_vars,
                exit_ids,
                ..
            } => {
                propagate_while(init_ids, live_vars, exit_ids, aggs);
                admit_instrs_in(cond_instrs, defined, module, aggs, array_operands_only)?;
                admit_instrs_in(body, defined, module, aggs, array_operands_only)?;
                propagate_while(init_ids, live_vars, exit_ids, aggs);
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                then_result,
                else_result,
                dst,
                merges,
                ..
            } => {
                admit_instrs_in(cond_instrs, defined, module, aggs, array_operands_only)?;
                admit_instrs_in(then_instrs, defined, module, aggs, array_operands_only)?;
                admit_instrs_in(else_instrs, defined, module, aggs, array_operands_only)?;
                if aggs.contains(then_result) || aggs.contains(else_result) {
                    aggs.insert(*dst);
                }
                for (merge_id, then_val, else_val) in merges {
                    if aggs.contains(then_val) || aggs.contains(else_val) {
                        aggs.insert(*merge_id);
                    }
                }
            }
            // `ConstArray` is in profile (a read-only blob) AND it is an aggregate
            // value. Recording its destination is what lets the binop arm refuse
            // `a + b` without refusing `a[i]`, which is `ArrayLoad`.
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { dst, .. } => {
                aggs.insert(*dst);
            }
            #[cfg(feature = "std-surface")]
            Instr::ArrayLoad { .. }
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
            Instr::ArrayStore { .. } => {
                keep_refusal(array_operands_only, reject("aggregate.array_store"))?
            }

            // ---- OUT OF PROFILE: tensor surface (RI-E) ----
            // The frozen stage1.elf fail-closes on these (readiness gate: tensor
            // programs → error[backend-native]); admitting them would risk a
            // silent miscompile, so reject LOUDLY here first.
            // The array-operand walk keeps only `binop.array_operand` and
            // continues, so a later whole-array binop is still named. The lane
            // fence still refuses these constructs.
            Instr::ConstTensor(..) => keep_refusal(array_operands_only, reject("const-tensor"))?,
            Instr::ConstDenseTensor { .. } => {
                keep_refusal(array_operands_only, reject("const-dense-tensor"))?
            }
            Instr::Sum { .. } => keep_refusal(array_operands_only, reject("tensor.sum"))?,
            Instr::Mean { .. } => keep_refusal(array_operands_only, reject("tensor.mean"))?,
            Instr::Relu { .. } => keep_refusal(array_operands_only, reject("tensor.relu"))?,
            Instr::ReluGrad { .. } => {
                keep_refusal(array_operands_only, reject("tensor.relu_grad"))?
            }
            Instr::Reshape { .. } => keep_refusal(array_operands_only, reject("tensor.reshape"))?,
            Instr::ExpandDims { .. } => {
                keep_refusal(array_operands_only, reject("tensor.expand_dims"))?
            }
            Instr::Squeeze { .. } => keep_refusal(array_operands_only, reject("tensor.squeeze"))?,
            Instr::Transpose { .. } => {
                keep_refusal(array_operands_only, reject("tensor.transpose"))?
            }
            Instr::Dot { .. } => keep_refusal(array_operands_only, reject("tensor.dot"))?,
            Instr::MatMul { .. } => keep_refusal(array_operands_only, reject("tensor.matmul"))?,
            Instr::Conv2d { .. } => keep_refusal(array_operands_only, reject("tensor.conv2d"))?,
            Instr::Conv2dGradInput { .. } => {
                keep_refusal(array_operands_only, reject("tensor.conv2d_grad_input"))?
            }
            Instr::Conv2dGradFilter { .. } => {
                keep_refusal(array_operands_only, reject("tensor.conv2d_grad_filter"))?
            }
            Instr::Index { .. } => keep_refusal(array_operands_only, reject("tensor.index"))?,
            Instr::Slice { .. } => keep_refusal(array_operands_only, reject("tensor.slice"))?,
            Instr::Gather { .. } => keep_refusal(array_operands_only, reject("tensor.gather"))?,
            Instr::SparseAttr { .. } => keep_refusal(array_operands_only, reject("sparse_attr"))?,

            // ---- OUT OF PROFILE: region + SIMD/BLAS vector ops (RI-E) ----
            #[cfg(feature = "std-surface")]
            Instr::Region { .. } => keep_refusal(array_operands_only, reject("region"))?,
            #[cfg(feature = "std-surface")]
            Instr::VecLoad { .. }
            | Instr::VecFma { .. }
            | Instr::VecReduceAdd { .. }
            | Instr::VecStore { .. }
            | Instr::VecLoadI32 { .. }
            | Instr::VecMulAddQ16 { .. }
            | Instr::VecReduceAddI64 { .. } => {
                keep_refusal(array_operands_only, reject("simd-vector-op"))?
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "frozen_profile_tests.rs"]
mod tests;
