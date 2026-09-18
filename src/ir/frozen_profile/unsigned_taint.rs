// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Unsigned taint for the frozen native profile's signedness-sensitive operators.
//!
//! A submodule of `frozen_profile` split out (module-size budget) because it is one self-contained
//! question: can a full-width UNSIGNED (`u64`) value reach a given OPERAND? The native emitter has only signed `idiv` and signed `setcc`, so `/`, `%`, and
//! the ordered compares are admitted only where the answer is no. See the
//! rationale at `frozen_profile::profile_frozen_admits`.

#[cfg(feature = "std-surface")]
use crate::ast::TypeAnn;
use crate::ir::{IRModule, Instr, ValueId};

mod taint;
pub(crate) use taint::Taint;

#[cfg(feature = "std-surface")]
mod mask;
#[cfg(feature = "std-surface")]
use mask::{collect_consts, collect_loop_inits, mark_masked};

/// Which call results and parameters carry a full-width UNSIGNED (`u64`) value, at two
/// precisions (see [`Taint`]). NARROW unsigned (`u8`/`u16`/`u32`) is deliberately NOT
/// tainted: it is masked to its width and zero-extends into a NON-NEGATIVE i64, so signed
/// `idiv`/`setcc` give the same result as the unsigned op — only a full-width carrier
/// whose high bit is set can diverge. `usize` is NOT tainted either: MLIR lowers it as a
/// SIGNED i64 today (`arith.cmpi "slt"`, pinned by `usize_is_signed_on_the_mlir_backend_too`),
/// so native and MLIR agree, and main 5724a6ee compiles a `usize` compare natively.
#[cfg(feature = "std-surface")]
pub(crate) struct UnsignedDoors {
    /// Callee name -> is each parameter `u64` exactly (MLIR seeds `ScalarU64`)?
    params_exact: std::collections::BTreeMap<String, Vec<bool>>,
    /// Callee name -> does each parameter type CONTAIN `u64` (array/ref/tuple/generic)?
    params_wide: std::collections::BTreeMap<String, Vec<bool>>,
    /// Callees whose RESULT is `u64` exactly.
    ret_exact: std::collections::BTreeSet<String>,
    /// Callees whose result type contains `u64`.
    ret_wide: std::collections::BTreeSet<String>,
    /// Callee name -> is each parameter `u32` (MLIR seeds `ScalarU32`)?
    params_u32: std::collections::BTreeMap<String, Vec<bool>>,
    /// Callees whose RESULT is `u32`.
    ret_u32: std::collections::BTreeSet<String>,
    /// Every function DEFINED in the module (at any nesting depth).
    defined: std::collections::BTreeSet<String>,
}

/// Which taint set a propagation pass computes.
#[cfg(feature = "std-surface")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
    /// MLIR's `ScalarU64` kind, exactly.
    Exact,
    /// Any value whose declared type is or contains `u64`.
    Wide,
    /// An over-approximation of MLIR's `ScalarU32` kind.
    Narrow,
}

#[cfg(feature = "std-surface")]
impl UnsignedDoors {
    pub(crate) fn from_module(module: &IRModule) -> Self {
        let mut doors = Self {
            params_exact: Default::default(),
            params_wide: Default::default(),
            ret_exact: Default::default(),
            ret_wide: Default::default(),
            params_u32: Default::default(),
            ret_u32: Default::default(),
            defined: Default::default(),
        };
        collect_defined(&module.instrs, &mut doors.defined);
        for (name, (ps, r)) in &module.fn_signatures {
            doors
                .params_exact
                .insert(name.clone(), ps.iter().map(is_u64).collect());
            doors
                .params_wide
                .insert(name.clone(), ps.iter().map(contains_u64).collect());
            if r.as_ref().is_some_and(is_u64) {
                doors.ret_exact.insert(name.clone());
            }
            if r.as_ref().is_some_and(contains_u64) {
                doors.ret_wide.insert(name.clone());
            }
            let is_u32 = |ty: &TypeAnn| matches!(ty, TypeAnn::ScalarU32);
            doors
                .params_u32
                .insert(name.clone(), ps.iter().map(is_u32).collect());
            if r.as_ref().is_some_and(is_u32) {
                doors.ret_u32.insert(name.clone());
            }
        }
        doors
    }

    /// The tainted values of the body of `owner` (the module's top level for `None`),
    /// `inherited` only carries the no-signature-table answer (see below).
    ///
    /// UNDECIDABLE => REFUSE (audit 2026-09-16), at both precisions. A parameter of a
    /// function whose own signature is unknown, and the result of a call to a
    /// module-DEFINED function whose signature is unknown, are tainted. Measured before
    /// this rule: a NESTED `fn inner(x: u64) -> i64 { return x / 2 }` has no
    /// `fn_signatures` entry (signature collection does not recurse into bodies), so it
    /// read as untainted and fence 1 admitted the unsigned division — stopped only by
    /// the frozen ELF's refusal of nested functions, i.e. a first fence deferring to the
    /// second.
    pub(crate) fn body_taint(
        &self,
        owner: Option<&str>,
        fn_params: &[(String, ValueId)],
        body: &[Instr],
        inherited: &Taint,
    ) -> Taint {
        // ValueIds RESTART in every function (`lower.rs` builds each body in a fresh
        // `IRModule`), so a body never refers to an enclosing scope's ids: nothing but the
        // "no signature table" answer is inherited (audit 2026-09-16 — module-wide id sets
        // matched the wrong values).
        if let Taint::All = inherited {
            return Taint::All;
        }
        let mut cmp = std::collections::BTreeSet::new();
        let mut div = std::collections::BTreeSet::new();
        let mut narrow = std::collections::BTreeSet::new();
        for (precision, set) in [
            (Precision::Exact, &mut cmp),
            (Precision::Wide, &mut div),
            (Precision::Narrow, &mut narrow),
        ] {
            let table = match precision {
                Precision::Exact => &self.params_exact,
                Precision::Wide => &self.params_wide,
                Precision::Narrow => &self.params_u32,
            };
            let param_tainted = |index: usize| match owner {
                None => false,
                Some(name) => table
                    .get(name)
                    .is_none_or(|ps| ps.get(index).copied().unwrap_or(true)),
            };
            for (index, (_, id)) in fn_params.iter().enumerate() {
                if param_tainted(index) {
                    set.insert(*id);
                }
            }
            // Loop back-edges carry a value defined LATER in the body into an earlier
            // use, so propagate to a fixpoint (the set only grows; bounded by the ids).
            while self.propagate(body, &param_tainted, set, precision) {}
        }
        let mut poisoned = std::collections::BTreeSet::new();
        collect_loop_inits(body, &mut poisoned);
        // A loop init id is a header block argument, not a literal (see mask.rs).
        let mut consts = std::collections::BTreeMap::new();
        collect_consts(body, &mut consts);
        consts.retain(|id, _| !poisoned.contains(id));
        let nonneg_consts: std::collections::BTreeSet<ValueId> = consts
            .iter()
            .filter(|(_, v)| **v >= 0)
            .map(|(id, _)| *id)
            .collect();
        let mut masked = std::collections::BTreeSet::new();
        while mark_masked(body, &nonneg_consts, &poisoned, &mut masked) {}
        let nonneg = masked.union(&nonneg_consts).copied().collect();
        Taint::Values {
            cmp,
            div,
            narrow,
            nonneg,
            consts,
        }
    }

    /// One forward pass over `body`; returns whether the set grew. Covers exactly the
    /// value-producing instructions the frozen profile ADMITS (everything else is
    /// refused by the walk, so it cannot reach a native artifact). Does not enter a
    /// nested `FnDef`: that body gets its own `body_taint`.
    fn propagate(
        &self,
        body: &[Instr],
        param_tainted: &dyn Fn(usize) -> bool,
        set: &mut std::collections::BTreeSet<ValueId>,
        precision: Precision,
    ) -> bool {
        let ret = match precision {
            Precision::Exact => &self.ret_exact,
            Precision::Wide => &self.ret_wide,
            Precision::Narrow => &self.ret_u32,
        };
        let unknown_sig =
            |name: &str| self.defined.contains(name) && !self.params_exact.contains_key(name);
        let mut nested_grew = false;
        let mut grew = false;
        let mut mark = |set: &mut std::collections::BTreeSet<ValueId>, id: ValueId| {
            grew |= set.insert(id);
        };
        for instr in body {
            match instr {
                Instr::Param { dst, index, .. } if param_tainted(*index) => mark(set, *dst),
                Instr::Call { dst, name, .. } if ret.contains(name) || unknown_sig(name) => {
                    mark(set, *dst)
                }
                // A compare yields 0/1 — identical signed and unsigned — so, as in MLIR,
                // it does not propagate the kind. A MASKED result still propagates (see
                // `Taint::Values::masked`).
                Instr::BinOp { dst, op, lhs, rhs }
                    if !is_compare(op) && (set.contains(lhs) || set.contains(rhs)) =>
                {
                    mark(set, *dst)
                }
                // MLIR loads a `[u64; N]` element as `ScalarI64` (the element maps to the
                // i64 ABI type), so only the wide set follows an element.
                Instr::ArrayLoad { dst, base, .. }
                    if precision == Precision::Wide && set.contains(base) =>
                {
                    mark(set, *dst)
                }
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
                    for nested in [cond_instrs, then_instrs, else_instrs] {
                        nested_grew |= self.propagate(nested, param_tainted, set, precision);
                    }
                    // Signedness ABSORPTION at the join (MLIR: I64 ⊔ U64 = U64).
                    if set.contains(then_result) || set.contains(else_result) {
                        mark(set, *dst);
                    }
                    for (merge, then_val, else_val) in merges {
                        if set.contains(then_val) || set.contains(else_val) {
                            mark(set, *merge);
                        }
                    }
                }
                Instr::While {
                    cond_instrs,
                    body: loop_body,
                    live_vars,
                    init_ids,
                    exit_ids,
                    ..
                } => {
                    for nested in [cond_instrs, loop_body] {
                        nested_grew |= self.propagate(nested, param_tainted, set, precision);
                    }
                    let mut exits = Vec::new();
                    if precision != Precision::Exact {
                        collect_loop_exits(loop_body, &mut exits);
                    }
                    for (k, (var, post)) in live_vars.iter().enumerate() {
                        let init = init_ids.get(k).copied();
                        let exit = exit_ids.get(k).copied();
                        let init_tainted = init.is_some_and(|i| set.contains(&i));
                        match precision {
                            // MLIR types the header and exit block args by the INIT
                            // kind; the post-body value does not absorb into them.
                            Precision::Exact => {
                                if let (true, Some(exit)) = (init_tainted, exit) {
                                    mark(set, exit);
                                }
                            }
                            // Wide: the variable is one value across the header — its
                            // init, post-body id, any break/continue snapshot and exit.
                            Precision::Wide | Precision::Narrow => {
                                let carried = init_tainted
                                    || set.contains(post)
                                    || exits.iter().any(|(n, id)| n == var && set.contains(id));
                                if carried {
                                    for id in [init, Some(*post), exit].into_iter().flatten() {
                                        mark(set, id);
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        grew || nested_grew
    }
}

#[cfg(feature = "std-surface")]
fn is_compare(op: &crate::ir::BinOp) -> bool {
    use crate::ir::BinOp;
    matches!(
        op,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
    )
}

/// Every `break` / `continue` live snapshot in `body`, including those of nested loops
/// (an over-approximation: a nested loop's snapshot naming the same variable can only
/// add taint), but not inside a nested `FnDef`.
#[cfg(feature = "std-surface")]
fn collect_loop_exits(body: &[Instr], out: &mut Vec<(String, ValueId)>) {
    for instr in body {
        match instr {
            Instr::Break { live } | Instr::Continue { live } => out.extend(live.iter().cloned()),
            Instr::FnDef { .. } => {}
            other => {
                for nested in crate::ir::instr_bodies(other) {
                    collect_loop_exits(nested, out);
                }
            }
        }
    }
}

/// Every `FnDef` name in `instrs`, at any nesting depth (through every body the
/// shared `instr_bodies` enumeration knows about).
#[cfg(feature = "std-surface")]
fn collect_defined(instrs: &[Instr], out: &mut std::collections::BTreeSet<String>) {
    for instr in instrs {
        if let Instr::FnDef { name, .. } = instr {
            out.insert(name.clone());
        }
        for nested in crate::ir::instr_bodies(instr) {
            collect_defined(nested, out);
        }
    }
}

/// Without `std-surface` the IR carries NO `fn_signatures` side-table (the field is
/// `#[cfg(feature = "std-surface")]` on `IRModule`), so no door can be inspected.
/// An uninspectable door is treated as OPEN: every value is tainted, which REFUSES
/// Div/Mod (as main 5724a6ee refused them in every configuration) instead of admitting
/// them unchecked — "undecidable => refuse". Ordered compares keep main's no-std
/// verdict (see `admit_binop`). (Found by `cargo check --no-default-features`, which
/// the std-surface unit gate could not see.)
#[cfg(not(feature = "std-surface"))]
pub(crate) struct UnsignedDoors;

#[cfg(not(feature = "std-surface"))]
impl UnsignedDoors {
    pub(crate) fn from_module(_module: &IRModule) -> Self {
        Self
    }

    pub(crate) fn body_taint(
        &self,
        _owner: Option<&str>,
        _fn_params: &[(String, ValueId)],
        _body: &[Instr],
        _inherited: &Taint,
    ) -> Taint {
        Taint::All
    }
}

/// `u64` exactly: the one annotation MLIR seeds as `ScalarU64` (`type_ann_to_value_kind`).
#[cfg(feature = "std-surface")]
fn is_u64(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Named(n) if n == "u64")
}

#[cfg(feature = "std-surface")]
fn contains_u64(ty: &TypeAnn) -> bool {
    // Recurse through every type constructor that can CARRY a full-width unsigned
    // value into a body: an array/slice element, a reference target, a tuple element,
    // a generic argument. A bare `Named("u64")` match missed `fn f(a: [u64; 2]) ->
    // i64 { return a[0] / a[1] }` — no `__mind_conv_u64`, no tainted signature
    // (audit 2026-09-16).
    match ty {
        TypeAnn::Named(n) => n == "u64",
        TypeAnn::Slice { element, .. } | TypeAnn::Array { element, .. } => contains_u64(element),
        TypeAnn::Ref { target, .. } => contains_u64(target),
        TypeAnn::Tuple { elements, .. } => elements.iter().any(contains_u64),
        TypeAnn::Generic { args, .. } => args.iter().any(contains_u64),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std-surface")]
    use crate::ast::TypeAnn;
    use crate::ir::frozen_profile::profile_frozen_admits;
    use crate::ir::{BinOp, IRModule, Instr, ValueId};

    fn binop(op: BinOp) -> Instr {
        Instr::BinOp {
            dst: ValueId(2),
            op,
            lhs: ValueId(0),
            rhs: ValueId(1),
        }
    }

    #[cfg(feature = "std-surface")]
    fn fndef(body: Vec<Instr>) -> Instr {
        fndef_named("f", body)
    }

    #[cfg(feature = "std-surface")]
    fn fndef_named(name: &str, body: Vec<Instr>) -> Instr {
        fndef_params(name, 0, body)
    }

    /// A function whose `n` parameters are `ValueId(0)..ValueId(n)` — the ids `binop`
    /// reads, so a parameter's taint reaches the operator.
    #[cfg(feature = "std-surface")]
    fn fndef_params(name: &str, n: usize, body: Vec<Instr>) -> Instr {
        Instr::FnDef {
            name: name.into(),
            params: (0..n).map(|i| (format!("p{i}"), ValueId(i))).collect(),
            ret_id: None,
            body,
            reap_threshold: None,
            semantic_types: None,
            #[cfg(feature = "std-surface")]
            value_types: Default::default(),
        }
    }

    /// Admission of a bare top-level instruction list, through the production entry.
    #[cfg(feature = "std-surface")]
    fn admits(instrs: &[Instr]) -> Result<(), crate::ir::frozen_profile::FrozenProfileRejection> {
        let mut m = IRModule::new();
        m.instrs = instrs.to_vec();
        profile_frozen_admits(&m)
    }

    /// A module carrying `instrs` whose `fn_signatures` declare each
    /// `(name, param types, return type)` — exercises `UnsignedDoors` (the bare-slice
    /// `admits` helper cannot express signatures).
    #[cfg(feature = "std-surface")]
    fn module_with_sigs(instrs: &[Instr], sigs: &[(&str, &[&str], Option<&str>)]) -> IRModule {
        let ty = |s: &str| TypeAnn::Named(s.to_string());
        let mut m = IRModule::new();
        m.instrs = instrs.to_vec();
        for (name, params, ret) in sigs {
            m.fn_signatures.insert(
                name.to_string(),
                (params.iter().map(|p| ty(p)).collect(), ret.map(ty)),
            );
        }
        m
    }

    #[cfg(feature = "std-surface")]
    fn call(name: &str) -> Instr {
        Instr::legacy_call(ValueId(3), name, vec![])
    }

    #[cfg(feature = "std-surface")]
    fn construct(m: &IRModule) -> Option<&'static str> {
        profile_frozen_admits(m).err().map(|r| r.construct)
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
        // A u64 PARAMETER of the function that divides taints that body.
        for op in [BinOp::Div, BinOp::Mod] {
            let m = module_with_sigs(
                &[fndef_params("d", 2, vec![binop(op)])],
                &[("d", &["u64", "u64"], Some("i64"))],
            );
            assert_eq!(construct(&m), Some("binop.div_mod_unsigned"), "{op:?}");
        }
        // A u64 RETURN type does not make the body's i64 operands unsigned: MLIR picks
        // `remsi` from the operand kinds here too, so native `idiv` agrees.
        let m = module_with_sigs(
            &[fndef_params("m", 2, vec![binop(BinOp::Mod)])],
            &[("m", &["i64", "i64"], Some("u64"))],
        );
        assert_eq!(construct(&m), None);
        // Only the unsigned OPERAND is refused: `d(a: u64, b: i64)` dividing `b / b`
        // stays admitted, `a / b` does not.
        let b_by_b = Instr::BinOp {
            dst: ValueId(2),
            op: BinOp::Div,
            lhs: ValueId(1),
            rhs: ValueId(1),
        };
        let m = module_with_sigs(
            &[fndef_params("d", 2, vec![b_by_b])],
            &[("d", &["u64", "i64"], Some("i64"))],
        );
        assert_eq!(
            construct(&m),
            None,
            "an i64 operand beside a u64 param is signed"
        );
    }

    /// The taint is per VALUE in each function body. A u64 signature elsewhere in the image must
    /// not refuse an unrelated signed division — measured on the merged std + user
    /// image, `std/json.mind`'s `num_u64(v: u64)` / `get_u64(..) -> u64` refused
    /// EVERY native `/` while the taint was module-wide. Mutation-sensitive: making
    /// `body_taint` ignore its owner and return "any signature in the module"
    /// turns the first assertion red.
    #[cfg(feature = "std-surface")]
    #[test]
    fn a_u64_signature_elsewhere_does_not_taint_an_unrelated_body() {
        let json_like = fndef_named("get_u64", vec![Instr::ConstI64(ValueId(9), 0)]);
        let m = module_with_sigs(
            &[json_like, fndef_named("main", vec![binop(BinOp::Div)])],
            &[
                ("get_u64", &["i64"], Some("u64")),
                ("main", &[], Some("i64")),
            ],
        );
        assert_eq!(
            construct(&m),
            None,
            "an uncalled u64 fn must not taint `main`"
        );

        // ...but CALLING it and dividing its RESULT is refused; dividing unrelated
        // values beside that call is not.
        let call_u64 = call("get_u64");
        let div_result = Instr::BinOp {
            dst: ValueId(4),
            op: BinOp::Div,
            lhs: ValueId(3),
            rhs: ValueId(1),
        };
        let m = module_with_sigs(
            &[
                fndef_named("get_u64", vec![]),
                fndef_named("main", vec![call_u64.clone(), div_result]),
            ],
            &[("get_u64", &[], Some("u64")), ("main", &[], Some("i64"))],
        );
        assert_eq!(construct(&m), Some("binop.div_mod_unsigned"));
        let m = module_with_sigs(
            &[
                fndef_named("get_u64", vec![]),
                fndef_named("main", vec![call_u64.clone(), binop(BinOp::Div)]),
            ],
            &[("get_u64", &[], Some("u64")), ("main", &[], Some("i64"))],
        );
        assert_eq!(construct(&m), None, "the u64 result is not an operand here");

        // A u64 PARAMETER of a callee does not flow back into the caller.
        let m = module_with_sigs(
            &[
                fndef_named("take", vec![]),
                fndef_named("main", vec![call("take"), binop(BinOp::Div)]),
            ],
            &[("take", &["u64"], Some("i64")), ("main", &[], Some("i64"))],
        );
        assert_eq!(
            construct(&m),
            None,
            "a callee's u64 PARAM is not a door into the caller"
        );
    }

    /// Ordered compares are refused in an unsigned-tainted body (signed `setcc` vs MLIR
    /// `ult`/`ugt`, #99) — per operand, like Div/Mod — while Eq/Ne (bit compares) and the
    /// same compares in an UNtainted body stay admitted.
    #[cfg(feature = "std-surface")]
    #[test]
    fn ordered_compare_rejected_in_unsigned_body_but_eq_ne_ok() {
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            let m = module_with_sigs(
                &[fndef_params("lt", 2, vec![binop(op)])],
                &[("lt", &["u64", "u64"], Some("i64"))],
            );
            assert_eq!(
                construct(&m),
                Some("binop.ordered_compare_unsigned"),
                "{op:?}"
            );
            // Control: the same body under an i64 signature is admitted.
            let m = module_with_sigs(
                &[fndef_params("lt", 2, vec![binop(op)])],
                &[("lt", &["i64", "i64"], Some("i64"))],
            );
            assert_eq!(construct(&m), None, "{op:?} under i64 must be admitted");
        }
        for op in [BinOp::Eq, BinOp::Ne] {
            let m = module_with_sigs(
                &[fndef_params("eq", 2, vec![binop(op)])],
                &[("eq", &["u64", "u64"], Some("i64"))],
            );
            assert_eq!(construct(&m), None, "{op:?} is sign-agnostic");
        }
        // A u64 SIGNATURE elsewhere (std/json's `num_u64` shape) does not refuse a loop
        // compare in an unrelated body.
        let m = module_with_sigs(
            &[
                fndef_params("num_u64", 1, vec![binop(BinOp::Gt)]),
                fndef_named("main", vec![binop(BinOp::Lt)]),
            ],
            &[
                ("num_u64", &["u64"], Some("i64")),
                ("main", &[], Some("i64")),
            ],
        );
        assert_eq!(
            construct(&m),
            Some("binop.ordered_compare_unsigned"),
            "num_u64's own body IS tainted when it is walked"
        );
        let m = module_with_sigs(
            &[fndef_named("main", vec![binop(BinOp::Lt)])],
            &[
                ("num_u64", &["u64"], Some("i64")),
                ("main", &[], Some("i64")),
            ],
        );
        assert_eq!(
            construct(&m),
            None,
            "a signature-only std fn must not taint `main`"
        );
    }

    /// Without `std-surface` there is no signature table to inspect, so the Div/Mod
    /// taint fails CLOSED (refused — as main 5724a6ee refused Div/Mod unconditionally),
    /// while ordered compares keep main's admitted verdict. Mutation-sensitive:
    /// returning `false` from the non-std-surface `UnsignedDoors::body_taint` admits
    /// Div/Mod and turns this red.
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
        // Ordered compares keep main 5724a6ee's no-std verdict (admitted): refusing them
        // would remove a capability main had.
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            assert_eq!(profile_frozen_admits(&m(&[binop(op)])), Ok(()), "{op:?}");
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
        let m = module_with_sigs(
            &[fndef(vec![fndef_params(
                "inner",
                1,
                vec![binop(BinOp::Div)],
            )])],
            &[("inner", &["u64"], Some("i64"))],
        );
        assert_eq!(
            profile_frozen_admits(&m).unwrap_err().construct,
            "binop.div_mod_unsigned"
        );
    }

    /// ValueIds restart per function, and the top level mints a placeholder
    /// `ConstI64(id, 0)` per `FnDef`, so ValueId(0) is a "non-negative constant" at the top
    /// level while being a PARAMETER inside `f`. A module-wide constant set exempted
    /// `a & b` on `f`'s first two params from the taint (audit 2026-09-16). Positive
    /// control: the same body with a constant defined in `f` itself IS exempt.
    #[cfg(feature = "std-surface")]
    #[test]
    fn mask_constants_are_per_body_not_module_wide() {
        let and_params = Instr::BinOp {
            dst: ValueId(2),
            op: BinOp::BitAnd,
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        let lt = Instr::BinOp {
            dst: ValueId(3),
            op: BinOp::Lt,
            lhs: ValueId(2),
            rhs: ValueId(1),
        };
        let m = module_with_sigs(
            &[
                Instr::ConstI64(ValueId(0), 0),
                fndef_params("f", 2, vec![and_params.clone(), lt.clone()]),
            ],
            &[("f", &["u64", "u64"], Some("i64"))],
        );
        assert_eq!(construct(&m), Some("binop.ordered_compare_unsigned"));
        let mask = Instr::BinOp {
            dst: ValueId(2),
            op: BinOp::BitAnd,
            lhs: ValueId(0),
            rhs: ValueId(9),
        };
        let lt_masked = Instr::BinOp {
            dst: ValueId(3),
            op: BinOp::Lt,
            lhs: ValueId(2),
            rhs: ValueId(9),
        };
        let m = module_with_sigs(
            &[fndef_params(
                "f",
                2,
                vec![Instr::ConstI64(ValueId(9), 255), mask, lt_masked],
            )],
            &[("f", &["u64", "u64"], Some("i64"))],
        );
        assert_eq!(
            construct(&m),
            None,
            "`a & 255` defined in the body is a mask"
        );
    }
}
