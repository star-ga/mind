// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Determinism of code a module calls but does not contain.
//!
//! The evidence classifier above judges ONE lowered module: every function body
//! it holds, plus each callee it can name (an intrinsic row, a world builtin, an
//! `extern "C"` symbol). A call into another project module holds only the
//! callee's name, and that name is in no registry, so the classifier admits it.
//! Measured before this module: a project whose sibling module called
//! `entropy.entropy_fill` (or `time.now_ns`) passed the determinism veto and was
//! attested `deterministic`, while the same call in the entry was refused.
//!
//! Two consumers close that gap:
//!
//! * the BUILD side (`project::single_file_scope::world_reach`) lowers each
//!   project module the entry calls into and follows the calls from function to
//!   function with [`fn_reach`], the same classifier per body;
//! * the VERIFY side cannot see those bodies. A body that calls a function it
//!   neither defines nor declares ([`calls_outside_body`]) cannot prove that
//!   function deterministic, so there a stored `nondeterministic` label is the
//!   producer's classification of code outside the artifact, and
//!   [`reported_determinism`] keeps it. A stored `deterministic` that the body
//!   contradicts is a tamper in every case, as before.

use std::collections::{BTreeMap, BTreeSet};

use super::{ExternPolicy, find_nondeterministic_call_ext};
use crate::ir::{IRModule, Instr, instr_bodies};

/// The determinism of one top-level function of a lowered module.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FnReach {
    /// The first HARD nondeterministic callee its body calls (the veto's view:
    /// an unclassified `extern "C"` callee is unknown, not nondeterministic).
    pub hard: Option<String>,
    /// The first callee the attestation counts as nondeterministic (an
    /// unclassified `extern "C"` callee included).
    pub taint: Option<String>,
    /// Every callee name its body calls, at any depth.
    pub calls: BTreeSet<String>,
}

/// [`FnReach`] for every top-level `FnDef` of `module`, by name.
///
/// Each body is classified with the module's own extern set, so a call into an
/// `extern "C"` block the module declares is judged exactly as the whole-module
/// scan judges it. Two definitions of one name merge, keeping the first hit of
/// each kind and the union of their calls.
pub fn fn_reach(module: &IRModule) -> BTreeMap<String, FnReach> {
    let externs = collect_extern_symbols(&module.instrs);
    let mut out: BTreeMap<String, FnReach> = BTreeMap::new();
    for instr in &module.instrs {
        let Instr::FnDef { name, .. } = instr else {
            continue;
        };
        let one = std::slice::from_ref(instr);
        let slot = out.entry(name.clone()).or_default();
        if slot.hard.is_none() {
            slot.hard = find_nondeterministic_call_ext(one, &externs, ExternPolicy::Ignore);
        }
        if slot.taint.is_none() {
            slot.taint = find_nondeterministic_call_ext(one, &externs, ExternPolicy::Taint);
        }
        slot.calls.extend(called_names(one));
    }
    out
}

/// Every callee name `instrs` calls, through every nested body.
pub fn called_names(instrs: &[Instr]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut worklist: Vec<&[Instr]> = vec![instrs];
    while let Some(stream) = worklist.pop() {
        for instr in stream {
            if let Instr::Call { name, .. } = instr {
                out.insert(name.clone());
            }
            worklist.extend(instr_bodies(instr));
        }
    }
    out
}

/// Whether `module` calls a function whose body it does not hold and whose
/// determinism no registry states: not a `FnDef` of the module, not an
/// `extern "C"` symbol it declares (those are classified by the extern policy),
/// and not an intrinsic row. A std or project function called by name is such a
/// call.
pub fn calls_outside_body(module: &IRModule) -> bool {
    let mut defined = BTreeSet::new();
    let mut worklist: Vec<&[Instr]> = vec![&module.instrs];
    while let Some(stream) = worklist.pop() {
        for instr in stream {
            defined.extend(defined_name(instr));
            worklist.extend(instr_bodies(instr));
        }
    }
    called_names(&module.instrs).iter().any(|name| {
        !defined.contains(name.as_str()) && crate::intrinsics::intrinsic_determinism(name).is_none()
    })
}

/// The name a `FnDef` or an `extern "C"` declaration gives a body or a symbol.
fn defined_name(instr: &Instr) -> Option<&str> {
    match instr {
        Instr::FnDef { name, .. } => Some(name),
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl { name, .. } => Some(name),
        _ => None,
    }
}

/// The determinism `mindc verify` reports: `stored` is the label, `rederived`
/// the value re-derived from the hashed body (`None` when the body did not
/// decode), `calls_outside` is [`calls_outside_body`] of that body.
///
/// The body is authoritative for `nondeterministic`. It is authoritative for
/// `deterministic` only when it holds every function it calls; otherwise a
/// stored `nondeterministic` stands, because the producer saw callee bodies the
/// verifier cannot. A result that differs from `stored` is a tampered label.
pub fn reported_determinism(stored: bool, rederived: Option<bool>, calls_outside: bool) -> bool {
    let declined = !stored && calls_outside;
    rederived.unwrap_or(stored) && !declined
}

/// Every `extern "C"` symbol declared anywhere in the module.
///
/// Needed because the determinism classifier is a REGISTRY lookup: a name it does
/// not recognise returns "deterministic". For a user function that is correct —
/// its body is in the module and is classified on its own merits. For an
/// `extern "C"` symbol it is default-ADMIT on an unknown, which is backwards for
/// an ATTESTATION: the callee's body is outside the artifact entirely, so nothing
/// in the module can witness what it does. Measured before this: a program
/// declaring libc `time()` or `getenv()` and calling it emitted evidence attesting
/// `determinism: deterministic`.
pub(super) fn collect_extern_symbols(
    instrs: &[crate::ir::Instr],
) -> std::collections::BTreeSet<String> {
    use crate::ir::Instr;
    // `mut` is dead in the `--no-default-features` build for the same reason the
    // worklist note below gives: `Instr::ExternFnDecl` does not exist there, so
    // nothing inserts and the set is provably empty. Empty is the CORRECT answer
    // in that configuration (an IR with no extern declarations has no
    // unclassified externs to taint), not a silenced gap.
    #[cfg_attr(not(feature = "std-surface"), allow(unused_mut))]
    let mut out = std::collections::BTreeSet::new();
    // An explicit worklist, not recursion. Two reasons, both load-bearing:
    // the traversal runs over UNTRUSTED decoded IR (`parse_mic3` on a consumer
    // artifact), so an arbitrarily deep nest must not be able to exhaust the
    // stack before the classifier reaches a verdict; and under
    // `--no-default-features` — the configuration the CI clippy gate builds —
    // every `out`-writing arm below is compiled out, leaving a recursive helper
    // whose accumulator is threaded purely through its own calls
    // (`clippy::only_used_in_recursion`, which an item-level `allow` does not
    // suppress). Order is irrelevant: the result is a `BTreeSet`, so the set
    // and every downstream lookup are identical whatever order the streams are
    // visited in.
    let mut worklist: Vec<&[Instr]> = vec![instrs];
    while let Some(stream) = worklist.pop() {
        for instr in stream {
            match instr {
                #[cfg(feature = "std-surface")]
                Instr::ExternFnDecl { name, binding, .. } => {
                    out.insert(name.clone());
                    if *binding == crate::ast::ExternBinding::Address {
                        out.insert(format!("{name} bind(.address)"));
                    }
                }
                Instr::FnDef { body, .. } => worklist.push(body),
                #[cfg(feature = "std-surface")]
                Instr::While {
                    cond_instrs, body, ..
                } => {
                    worklist.push(cond_instrs);
                    worklist.push(body);
                }
                #[cfg(feature = "std-surface")]
                Instr::If {
                    cond_instrs,
                    then_instrs,
                    else_instrs,
                    ..
                } => {
                    worklist.push(cond_instrs);
                    worklist.push(then_instrs);
                    worklist.push(else_instrs);
                }
                // RFC 0010 Phase J-A `region { }` carries a full nested stream.
                // `scan_scope` already descends into it, so an `extern "C"`
                // DECLARED inside a region and CALLED there would otherwise be
                // absent from this set and hit `extern_call_is_unclassified`'s
                // default-ADMIT — the exact forge the extern check exists to
                // close, one nesting level down. Descended here for the same
                // reason `scan_scope` descends.
                #[cfg(feature = "std-surface")]
                Instr::Region { body, .. } => worklist.push(body),
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(dst: usize, name: &str) -> Instr {
        Instr::legacy_call(crate::ir::ValueId(dst), name, Vec::new())
    }

    fn fndef(name: &str, body: Vec<Instr>) -> Instr {
        Instr::FnDef {
            name: name.to_string(),
            params: Vec::new(),
            ret_id: None,
            body,
            reap_threshold: None,
            semantic_types: None,
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
        }
    }

    fn module(instrs: Vec<Instr>) -> IRModule {
        let mut m = IRModule::new();
        m.instrs = instrs;
        m
    }

    #[test]
    fn reach_per_function_names_the_first_world_callee_and_every_call() {
        let m = module(vec![
            fndef("pure", vec![call(0, "helper")]),
            fndef("world", vec![call(0, "helper"), call(1, "now_ns")]),
            fndef("helper", Vec::new()),
        ]);
        let reach = fn_reach(&m);
        assert_eq!(reach.len(), 3);
        assert_eq!(reach["pure"].hard, None);
        assert_eq!(reach["pure"].taint, None);
        assert_eq!(reach["world"].hard.as_deref(), Some("now_ns"));
        assert_eq!(reach["world"].taint.as_deref(), Some("now_ns"));
        assert!(reach["world"].calls.contains("helper"));
        assert!(reach["world"].calls.contains("now_ns"));
    }

    #[test]
    fn a_call_outside_the_body_is_one_no_definition_or_registry_row_covers() {
        let local = module(vec![fndef("a", vec![call(0, "b")]), fndef("b", Vec::new())]);
        assert!(!calls_outside_body(&local), "both callees are defined");
        let intrinsic = module(vec![fndef("a", vec![call(0, "__mind_alloc")])]);
        assert!(!calls_outside_body(&intrinsic), "an intrinsic row");
        let sibling = module(vec![fndef("a", vec![call(0, "g")])]);
        assert!(calls_outside_body(&sibling), "`g` has no body here");
    }

    #[test]
    fn only_a_body_that_calls_outside_itself_keeps_a_stricter_label() {
        // (stored, rederived, calls_outside) -> reported
        let cases = [
            (true, Some(true), false, true),
            (true, Some(true), true, true),
            (true, Some(false), false, false),
            (true, Some(false), true, false),
            (false, Some(false), false, false),
            (false, Some(false), true, false),
            (false, Some(true), true, false),
            (false, Some(true), false, true),
            (true, None, false, true),
            (false, None, true, false),
        ];
        for (stored, rederived, outside, want) in cases {
            assert_eq!(
                reported_determinism(stored, rederived, outside),
                want,
                "stored={stored} rederived={rederived:?} outside={outside}"
            );
        }
    }

    /// Spec "Extern bindings" E2/E3: the binding is tested before the registry
    /// name. `__mind_calloc` is a Pure row: base keeps `deterministic`, an
    /// address binding of the same name taints. The loader rows veto shipping.
    #[cfg(feature = "std-surface")]
    #[test]
    fn an_address_binding_taints_whatever_its_name() {
        use crate::ast::ExternBinding;
        let module = |name: &str, binding| {
            let mut m = IRModule::new();
            m.instrs.push(Instr::ExternFnDecl {
                name: name.into(),
                param_types: vec!["!llvm.ptr".into()],
                ret_type: Some("i64".into()),
                is_varargs: false,
                vararg_hints: vec![],
                callconv: crate::ast::CallConv::C,
                binding,
            });
            m.instrs.push(call(1, name));
            m
        };
        let base = module("__mind_calloc", ExternBinding::Base);
        assert_eq!(super::super::ir_first_nondeterministic_call(&base), None);
        let addr = module("__mind_calloc", ExternBinding::Address);
        let got = super::super::ir_first_nondeterministic_call(&addr);
        assert_eq!(got.as_deref(), Some("__mind_calloc"));
        for loader in ["dlopen", "dlsym", "dlclose", "dlerror"] {
            let m = module(loader, ExternBinding::Base);
            let veto = super::super::ir_first_hard_nondeterministic_call(&m);
            assert_eq!(veto.as_deref(), Some(loader), "the loader reads host state");
        }
    }
}
