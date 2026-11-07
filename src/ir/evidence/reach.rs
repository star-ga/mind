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

use super::{ExternPolicy, collect_extern_symbols, find_nondeterministic_call_ext};
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
}
