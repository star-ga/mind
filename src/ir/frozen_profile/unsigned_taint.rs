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
use crate::ir::{IRModule, Instr};

/// Which functions let a full-width UNSIGNED (`u64` / `usize`) value into a body:
/// `sig` = any parameter or the return is unsigned; `ret` = the return is unsigned.
/// NARROW unsigned (`u8`/`u16`/`u32`) is deliberately NOT tainted: it is masked to its
/// width and zero-extends into a NON-NEGATIVE i64, so signed `idiv` gives the same
/// result as the unsigned op — only a full-width carrier whose high bit is set can
/// diverge. `usize` is included as future-proofing: MLIR seeds it `ScalarI64` today,
/// but a later usize->unsigned fix would otherwise silently split the backends.
#[cfg(feature = "std-surface")]
pub(crate) struct UnsignedDoors {
    sig: std::collections::BTreeSet<String>,
    ret: std::collections::BTreeSet<String>,
}

#[cfg(feature = "std-surface")]
impl UnsignedDoors {
    pub(crate) fn from_module(module: &IRModule) -> Self {
        let mut sig = std::collections::BTreeSet::new();
        let mut ret = std::collections::BTreeSet::new();
        for (name, (params, r)) in &module.fn_signatures {
            let ret_unsigned = r.iter().any(is_unsigned_wide_type);
            if ret_unsigned || params.iter().any(is_unsigned_wide_type) {
                sig.insert(name.clone());
            }
            if ret_unsigned {
                ret.insert(name.clone());
            }
        }
        Self { sig, ret }
    }

    /// Is the body of `owner` (or the module's top level, for `None`) tainted?
    pub(crate) fn body_taint(&self, owner: Option<&str>, body: &[Instr]) -> bool {
        owner.is_some_and(|n| self.sig.contains(n)) || calls_any(body, &self.ret)
    }
}

/// Does `body` call any function in `callees`? Descends into every nested body via the
/// single enumeration `crate::ir::instr_bodies` (While / If / Region), but NOT into a
/// nested `FnDef`: a nested function is its own body with its own taint, and its calls
/// do not put a value into THIS body. Using the shared enumeration rather than a local
/// match keeps a future body-carrying variant from being invisible here.
#[cfg(feature = "std-surface")]
fn calls_any(body: &[Instr], callees: &std::collections::BTreeSet<String>) -> bool {
    body.iter().any(|instr| match instr {
        Instr::Call { name, .. } => callees.contains(name),
        Instr::FnDef { .. } => false,
        other => crate::ir::instr_bodies(other)
            .iter()
            .any(|nested| calls_any(nested, callees)),
    })
}

/// Without `std-surface` the IR carries NO `fn_signatures` side-table (the field is
/// `#[cfg(feature = "std-surface")]` on `IRModule`), so no door can be inspected.
/// An uninspectable door is treated as OPEN: every body is tainted, which REFUSES
/// Div/Mod and ordered compares instead of admitting them unchecked — "undecidable => refuse". (Found by
/// `cargo check --no-default-features`, which the std-surface unit gate could not see.)
#[cfg(not(feature = "std-surface"))]
pub(crate) struct UnsignedDoors;

#[cfg(not(feature = "std-surface"))]
impl UnsignedDoors {
    pub(crate) fn from_module(_module: &IRModule) -> Self {
        Self
    }

    pub(crate) fn body_taint(&self, _owner: Option<&str>, _body: &[Instr]) -> bool {
        true
    }
}

#[cfg(feature = "std-surface")]
fn is_unsigned_wide_type(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Named(n) if n == "u64" || n == "usize")
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
        Instr::FnDef {
            name: name.into(),
            params: vec![],
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
                &[fndef_named("d", vec![binop(op)])],
                &[("d", &["u64", "u64"], Some("i64"))],
            );
            assert_eq!(construct(&m), Some("binop.div_mod_unsigned"), "{op:?}");
        }
        // A u64 RETURN of that function taints it as well.
        let m = module_with_sigs(
            &[fndef_named("m", vec![binop(BinOp::Mod)])],
            &[("m", &["i64", "i64"], Some("u64"))],
        );
        assert_eq!(construct(&m), Some("binop.div_mod_unsigned"));
    }

    /// The taint is PER FUNCTION BODY. A u64 signature elsewhere in the image must
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

        // ...but CALLING it puts a u64 value into the body, so the body is tainted —
        // including when the call sits inside nested control flow.
        let call_u64 = call("get_u64");
        let m = module_with_sigs(
            &[
                fndef_named("get_u64", vec![]),
                fndef_named("main", vec![call_u64.clone(), binop(BinOp::Div)]),
            ],
            &[("get_u64", &[], Some("u64")), ("main", &[], Some("i64"))],
        );
        assert_eq!(construct(&m), Some("binop.div_mod_unsigned"));

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
    /// `ult`/`ugt`, #99) — per body, like Div/Mod — while Eq/Ne (bit compares) and the
    /// same compares in an UNtainted body stay admitted.
    #[cfg(feature = "std-surface")]
    #[test]
    fn ordered_compare_rejected_in_unsigned_body_but_eq_ne_ok() {
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            let m = module_with_sigs(
                &[fndef_named("lt", vec![binop(op)])],
                &[("lt", &["u64", "u64"], Some("i64"))],
            );
            assert_eq!(
                construct(&m),
                Some("binop.ordered_compare_unsigned"),
                "{op:?}"
            );
            // Control: the same body under an i64 signature is admitted.
            let m = module_with_sigs(
                &[fndef_named("lt", vec![binop(op)])],
                &[("lt", &["i64", "i64"], Some("i64"))],
            );
            assert_eq!(construct(&m), None, "{op:?} under i64 must be admitted");
        }
        for op in [BinOp::Eq, BinOp::Ne] {
            let m = module_with_sigs(
                &[fndef_named("eq", vec![binop(op)])],
                &[("eq", &["u64", "u64"], Some("i64"))],
            );
            assert_eq!(construct(&m), None, "{op:?} is sign-agnostic");
        }
        // A u64 SIGNATURE elsewhere (std/json's `num_u64` shape) does not refuse a loop
        // compare in an unrelated body.
        let m = module_with_sigs(
            &[
                fndef_named("num_u64", vec![binop(BinOp::Gt)]),
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

    /// Without `std-surface` there is no signature table to inspect, so the unsigned
    /// taint must fail CLOSED: Div/Mod and ordered compares refused, bit compares kept.
    /// Mutation-sensitive: returning `false` from the non-std-surface
    /// `UnsignedDoors::body_taint` admits all six and turns this red.
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
        let m = module_with_sigs(
            &[fndef(vec![fndef_named("inner", vec![binop(BinOp::Div)])])],
            &[("inner", &["u64"], Some("i64"))],
        );
        assert_eq!(
            profile_frozen_admits(&m).unwrap_err().construct,
            "binop.div_mod_unsigned"
        );
    }
}
