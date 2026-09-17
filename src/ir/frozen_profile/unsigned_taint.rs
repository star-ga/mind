// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Unsigned taint for the frozen native profile's signedness-sensitive operators.
//!
//! A submodule of `frozen_profile` split out (module-size budget) because it is one self-contained
//! question: can a full-width UNSIGNED (`u64` / `usize`) value reach a given function
//! body? The native emitter has only signed `idiv` and signed `setcc`, so `/`, `%`, and
//! the ordered compares are admitted only in bodies where the answer is no. See the
//! rationale at `frozen_profile::profile_frozen_admits`.

#[cfg(feature = "std-surface")]
use crate::ast::TypeAnn;
use crate::ir::{IRModule, Instr, ValueId};

/// Which call results and parameters carry a full-width UNSIGNED (`u64`) value.
/// NARROW unsigned (`u8`/`u16`/`u32`) is deliberately NOT tainted: it is masked to its
/// width and zero-extends into a NON-NEGATIVE i64, so signed `idiv`/`setcc` give the
/// same result as the unsigned op — only a full-width carrier whose high bit is set can
/// diverge. `usize` is NOT tainted either: MLIR lowers it as a SIGNED i64 today
/// (`arith.cmpi "slt"`, pinned by `usize_is_signed_on_the_mlir_backend_too`), so native
/// and MLIR agree, and main 5724a6ee compiles a `usize` compare natively.
#[cfg(feature = "std-surface")]
pub(crate) struct UnsignedDoors {
    /// Callee name -> is each parameter full-width unsigned?
    params: std::collections::BTreeMap<String, Vec<bool>>,
    /// Callees whose RESULT is full-width unsigned.
    ret: std::collections::BTreeSet<String>,
    /// Every function DEFINED in the module (at any nesting depth).
    defined: std::collections::BTreeSet<String>,
}

/// The values of ONE body that may hold a full-width unsigned value at run time.
///
/// VALUE-level, mirroring how the MLIR backend chooses `divui`/`ult` (operand kind
/// `ScalarU64`, seeded by `u64` parameters and `u64` call results, propagated through
/// non-compare arithmetic and joins). It used to be BODY-level — any `u64` in the
/// signature refused every `/`, `%` and ordered compare in the body — which refused
/// programs main 5724a6ee compiles and runs correctly natively (measured 2026-09-16:
/// `fn f(a: u64) -> i64 { let i = 0 while i < 3 { i = i + 1 } return i }` returns 3
/// on main, native and MLIR alike).
pub(crate) enum Taint {
    /// No signature table: every value is treated as possibly unsigned.
    #[cfg_attr(feature = "std-surface", allow(dead_code))]
    All,
    #[cfg_attr(not(feature = "std-surface"), allow(dead_code))]
    Values(std::collections::BTreeSet<ValueId>),
}

impl Taint {
    /// May any of `ids` hold a full-width unsigned value?
    pub(crate) fn any(&self, ids: &[ValueId]) -> bool {
        match self {
            Taint::All => true,
            Taint::Values(set) => ids.iter().any(|id| set.contains(id)),
        }
    }
}

#[cfg(feature = "std-surface")]
impl UnsignedDoors {
    pub(crate) fn from_module(module: &IRModule) -> Self {
        let mut params = std::collections::BTreeMap::new();
        let mut ret = std::collections::BTreeSet::new();
        let mut defined = std::collections::BTreeSet::new();
        collect_defined(&module.instrs, &mut defined);
        for (name, (ps, r)) in &module.fn_signatures {
            params.insert(name.clone(), ps.iter().map(is_unsigned_wide_type).collect());
            if r.iter().any(is_unsigned_wide_type) {
                ret.insert(name.clone());
            }
        }
        Self {
            params,
            ret,
            defined,
        }
    }

    /// The tainted values of the body of `owner` (the module's top level for `None`),
    /// starting from `inherited` (the enclosing body's taint).
    ///
    /// UNDECIDABLE => REFUSE (audit 2026-09-16). A parameter of a function whose own
    /// signature is unknown, and the result of a call to a module-DEFINED function
    /// whose signature is unknown, are tainted. Measured before this rule: a NESTED
    /// `fn inner(x: u64) -> i64 { return x / 2 }` has no `fn_signatures` entry
    /// (signature collection does not recurse into bodies), so it read as untainted
    /// and fence 1 admitted the unsigned division — stopped only by the frozen ELF's
    /// refusal of nested functions, i.e. a first fence deferring to the second.
    pub(crate) fn body_taint(
        &self,
        owner: Option<&str>,
        fn_params: &[(String, ValueId)],
        body: &[Instr],
        inherited: &Taint,
    ) -> Taint {
        let mut set = match inherited {
            Taint::All => return Taint::All,
            Taint::Values(s) => s.clone(),
        };
        let param_tainted = |index: usize| match owner {
            None => false,
            Some(name) => self
                .params
                .get(name)
                .is_none_or(|ps| ps.get(index).copied().unwrap_or(true)),
        };
        for (index, (_, id)) in fn_params.iter().enumerate() {
            if param_tainted(index) {
                set.insert(*id);
            }
        }
        // Loop back-edges carry a value defined LATER in the body into an earlier use,
        // so propagate to a fixpoint (the set only grows; it is bounded by the ids).
        while self.propagate(body, &param_tainted, &mut set) {}
        Taint::Values(set)
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
    ) -> bool {
        let mut nested_grew = false;
        let mut grew = false;
        let mut mark = |set: &mut std::collections::BTreeSet<ValueId>, id: ValueId| {
            grew |= set.insert(id);
        };
        for instr in body {
            match instr {
                Instr::Param { dst, index, .. } if param_tainted(*index) => mark(set, *dst),
                Instr::Call { dst, name, .. }
                    if self.ret.contains(name)
                        || (self.defined.contains(name) && !self.params.contains_key(name)) =>
                {
                    mark(set, *dst)
                }
                // A compare yields 0/1 — identical signed and unsigned — so, as in MLIR,
                // it does not propagate the kind.
                Instr::BinOp { dst, op, lhs, rhs }
                    if !is_compare(op) && (set.contains(lhs) || set.contains(rhs)) =>
                {
                    mark(set, *dst)
                }
                Instr::ArrayLoad { dst, base, .. } if set.contains(base) => mark(set, *dst),
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
                        nested_grew |= self.propagate(nested, param_tainted, set);
                    }
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
                        nested_grew |= self.propagate(nested, param_tainted, set);
                    }
                    // A carried variable is one value across the header block: its
                    // pre-loop id (which the condition and body reference), its
                    // post-body id, any `break`/`continue` snapshot of it, and its exit
                    // id. If any of them is tainted, all of them are.
                    let mut exits = Vec::new();
                    collect_loop_exits(loop_body, &mut exits);
                    for (k, (var, post)) in live_vars.iter().enumerate() {
                        let init = init_ids.get(k).copied();
                        let exit = exit_ids.get(k).copied();
                        let carried = init.is_some_and(|i| set.contains(&i))
                            || set.contains(post)
                            || exits.iter().any(|(n, id)| n == var && set.contains(id));
                        if carried {
                            for id in [init, Some(*post), exit].into_iter().flatten() {
                                mark(set, id);
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

#[cfg(feature = "std-surface")]
fn is_unsigned_wide_type(ty: &TypeAnn) -> bool {
    // Recurse through every type constructor that can CARRY a full-width unsigned
    // value into a body: an array/slice element, a reference target, a tuple element,
    // a generic argument. A bare `Named("u64")` match missed `fn f(a: [u64; 2]) ->
    // i64 { return a[0] / a[1] }` — no `__mind_conv_u64`, no tainted signature
    // (audit 2026-09-16).
    match ty {
        TypeAnn::Named(n) => n == "u64",
        TypeAnn::Slice { element, .. } | TypeAnn::Array { element, .. } => {
            is_unsigned_wide_type(element)
        }
        TypeAnn::Ref { target, .. } => is_unsigned_wide_type(target),
        TypeAnn::Tuple { elements, .. } => elements.iter().any(is_unsigned_wide_type),
        TypeAnn::Generic { args, .. } => args.iter().any(is_unsigned_wide_type),
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

    /// Lower real source through the production pipeline and ask the fence. These
    /// tests pin the module SHAPE lowering actually produces — the hand-built-IR tests
    /// above can insert a signature that lowering never provides (which is exactly how
    /// the nested-function door stayed open).
    #[cfg(feature = "std-surface")]
    fn fence(src: &str) -> Option<&'static str> {
        let module = crate::parser::parse(src).expect("fixture must parse");
        let ir = crate::eval::lower_to_ir(&module).expect("fixture must lower");
        profile_frozen_admits(&ir).err().map(|r| r.construct)
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn pipeline_nested_fn_with_u64_param_is_refused() {
        // Before: admitted by fence 1 (no `fn_signatures` entry for `inner`).
        let nested = "fn main() -> i64 {\n    fn inner(x: u64) -> i64 {\n        return x / 2\n    }\n    return 0\n}\n";
        assert!(fence(nested).is_some(), "nested u64 div must be refused");
        // Undecidable => refuse also covers a nested i64 body (its signature is equally
        // unknown). No capability is lost: the frozen ELF refuses nested fns anyway.
        let nested_i64 = "fn main() -> i64 {\n    fn inner(x: i64) -> i64 {\n        return x / 2\n    }\n    return 0\n}\n";
        assert!(fence(nested_i64).is_some());
        // Control: the same helper at TOP level has a known i64 signature and stays
        // admitted, so the refusal above is about nesting, not about `/`.
        let top = "fn inner(x: i64) -> i64 {\n    return x / 2\n}\nfn main() -> i64 {\n    return inner(8)\n}\n";
        assert_eq!(
            fence(top),
            None,
            "top-level signed division must stay admitted"
        );
        let top_u64 =
            "fn inner(x: u64) -> i64 {\n    return x / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(top_u64), Some("binop.div_mod_unsigned"));
    }

    #[cfg(feature = "std-surface")]
    #[test]
    fn pipeline_u64_inside_an_array_param_taints_the_body() {
        let arr = "fn f(a: [u64; 2]) -> i64 {\n    return a[0] / a[1]\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(arr), Some("binop.div_mod_unsigned"));
        let arr_i64 = "fn f(a: [i64; 2]) -> i64 {\n    return a[0] / a[1]\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(arr_i64), None, "the i64 twin must stay admitted");
    }

    /// The `let x: u64` / `as u64` door is shut by `admit_call` refusing the synthesised
    /// `__mind_conv_u64`, INDEPENDENTLY of the native bridge's closure check. Pinned by
    /// name because nothing else asserts it: an intrinsic allowlist added to
    /// `admit_call` by analogy with `__mind_conv_i64` would silently reopen the
    /// unsigned-division hole (audit 2026-09-16).
    #[cfg(feature = "std-surface")]
    #[test]
    fn conv_u64_call_is_refused_by_the_fence_itself() {
        let cast =
            "fn main() -> i64 {\n    let y: i64 = 84\n    let x: u64 = y as u64\n    return 0\n}\n";
        let module = crate::parser::parse(cast).expect("parse");
        let ir = crate::eval::lower_to_ir(&module).expect("fixture must lower");
        fn has_conv_u64(instrs: &[Instr]) -> bool {
            instrs.iter().any(|i| {
                matches!(i, Instr::Call { name, .. } if name == "__mind_conv_u64")
                    || crate::ir::instr_bodies(i).iter().any(|b| has_conv_u64(b))
            })
        }
        assert!(
            has_conv_u64(&ir.instrs),
            "positive control: lowering emits __mind_conv_u64"
        );
        assert_eq!(fence(cast), Some("call.undefined_or_builtin"));
    }

    /// PARITY with main 5724a6ee (measured 2026-09-16: native and MLIR both return 3):
    /// an i64 loop guard in a function with a `u64` parameter is not an unsigned
    /// compare. The body-level taint refused it.
    #[cfg(feature = "std-surface")]
    #[test]
    fn pipeline_i64_compare_beside_a_u64_param_is_admitted() {
        let src = "fn f(a: u64) -> i64 {\n    let i = 0\n    while i < 3 {\n        i = i + 1\n    }\n    return i\n}\nfn main() -> i64 {\n    return f(7)\n}\n";
        assert_eq!(fence(src), None);
    }

    /// `usize` is signed on BOTH backends today, so it is not tainted (main compiles
    /// `a < b` on `usize` natively, returning 1 for `f(1, 2)`). If MLIR ever selects an
    /// unsigned predicate for `usize`, this fails and `is_unsigned_wide_type` must add it.
    #[cfg(feature = "std-surface")]
    #[test]
    fn usize_is_signed_on_the_mlir_backend_too() {
        let src = "fn f(a: usize, b: usize) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(1, 2)\n}\n";
        assert_eq!(fence(src), None);
        // The MLIR half needs the lowering module; the `mlir-build` tiers run it.
        #[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
        {
            let module = crate::parser::parse(src).expect("parse");
            let mut ir = crate::eval::lower_to_ir(&module).expect("lower");
            let text = crate::mlir::compile_ir_to_mlir_text(&mut ir).expect("mlir text");
            assert!(
                text.contains("cmpi \"slt\""),
                "usize compare must be signed:\n{text}"
            );
            assert!(
                !text.contains("cmpi \"ult\""),
                "usize compare turned unsigned:\n{text}"
            );
        }
    }

    /// The miscompile the compare refusal exists for, measured on main 5724a6ee: with
    /// `b = 18446744073709551615`, `a < b` gives 0 natively and 1 on MLIR, and
    /// `big() > 5` gives 0 natively and 1 on MLIR.
    #[cfg(feature = "std-surface")]
    #[test]
    fn pipeline_unsigned_operand_compares_are_refused() {
        let param = "fn f(a: u64, b: u64) -> i64 {\n    if a < b {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return f(1, 18446744073709551615)\n}\n";
        assert_eq!(fence(param), Some("binop.ordered_compare_unsigned"));
        let ret = "fn big() -> u64 {\n    return 18446744073709551615\n}\nfn main() -> i64 {\n    if big() > 5 {\n        return 1\n    }\n    return 0\n}\n";
        assert_eq!(fence(ret), Some("binop.ordered_compare_unsigned"));
    }

    /// The taint follows the value through arithmetic, an `if` merge, and a loop-carried
    /// variable — including a use in the loop CONDITION of a value that only becomes
    /// unsigned on the back-edge (the fixpoint).
    #[cfg(feature = "std-surface")]
    #[test]
    fn pipeline_taint_follows_arithmetic_merges_and_loops() {
        let arith = "fn f(a: u64) -> i64 {\n    let s = a + 1\n    if s > 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(arith), Some("binop.ordered_compare_unsigned"));
        let merge = "fn f(a: u64, c: i64) -> i64 {\n    let m = 0\n    if c == 1 {\n        m = a\n    }\n    if m > 5 {\n        return 1\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(merge), Some("binop.ordered_compare_unsigned"));
        let back_edge = "fn f(a: u64) -> i64 {\n    let s = 0\n    while s < 100 {\n        s = s + a\n    }\n    return 0\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(back_edge), Some("binop.ordered_compare_unsigned"));
        let after_loop = "fn f(a: u64) -> i64 {\n    let s = 0\n    let i = 0\n    while i < 3 {\n        s = s + a\n        i = i + 1\n    }\n    return s / 2\n}\nfn main() -> i64 {\n    return 0\n}\n";
        assert_eq!(fence(after_loop), Some("binop.div_mod_unsigned"));
    }
}
