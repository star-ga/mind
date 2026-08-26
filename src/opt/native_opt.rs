// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Opt-in native `mic@3 → mic@3` optimizer (roadmap C6).
//!
//! The load-bearing architectural decision (see `docs/INDEPENDENCE_ROADMAP.md` §C6):
//! this optimizer is a **canonical IR transform sitting UPSTREAM of emitter
//! divergence**, so both the Rust `lower.rs`/mic@3 oracle and the self-host `nb_*`
//! emitter consume identical *optimized* bytes — each pass is written once and stays
//! oracle-parity-safe by construction, never twice-to-byte-parity.
//!
//! **DEFAULT OFF.** With [`OptLevel::Off`] (the default, when `MIND_NATIVE_OPT` is
//! unset) [`optimize_mic3`] is a no-op and every emitted byte is byte-identical to the
//! un-optimized canonical IR — so keystone / cross-substrate canaries / frozen self-host
//! seeds are untouched until a deliberate opt-in. Enabling the optimizer changes the
//! emitted bytes and is therefore a whole-corpus reseed event (one per landing).
//!
//! **Determinism contract.** Every pass is a *pure function with a pinned schedule*
//! (fixed pass order, fixed iteration count or a provably order-independent fixpoint —
//! never "loop until quiet"), so the optimized `mic@3` — which `trace_hash` anchors,
//! oracle-parity compares, and the canaries pin — is a deterministic function of the
//! input IR, stable run-to-run and across hosts.
//!
//! Planned pass schedule (roadmap order; MIND's `If`/`While` region IR makes these
//! structural rewrites — dominance == region nesting, LICM preheader == the slot before
//! the `While`, dead-arm pruning rewrites the region to its surviving child, no φ):
//! SCCP-lite → strength-reduction (`(op, operand_type)`-keyed) → region-scoped CSE/GVN
//! (state edge in the key) → LICM-on-`While` → linear-scan register allocation.

use crate::ir::{IRModule, Instr, ValueId};

/// Native-optimizer level. `Off` (default) is a strict no-op — byte-identical output.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OptLevel {
    /// No native passes; the emitted `mic@3` is byte-identical to the un-optimized
    /// canonical IR. This is the shipped default until the opt-in passes land + reseed.
    #[default]
    Off,
    /// SCCP-lite slice 1: const-condition value-`If` pruning (see [`const_if_prune`]).
    /// Opt-in only (`MIND_NATIVE_OPT=basic`); changes emitted bytes, so enabling it is a
    /// whole-corpus reseed event. More SCCP-lite / strength-reduction / CSE land here.
    Basic,
    // Future (each landing = one whole-corpus reseed):
    //   Full — Basic + LICM-on-While + linear-scan RA
}

/// Resolve the opt level from the environment. Deterministic (a single env read, no
/// host-dependent heuristic). Any unrecognized value is `Off` (fail-safe): an unknown
/// flag never silently enables a byte-changing transform.
pub fn opt_level_from_env() -> OptLevel {
    match std::env::var("MIND_NATIVE_OPT").ok().as_deref() {
        Some("basic") => OptLevel::Basic,
        // Any other value maps to Off — an unknown flag never enables a byte-changing pass.
        _ => OptLevel::Off,
    }
}

/// Run the opt-in native `mic@3 → mic@3` optimizer on the canonical IR, in place.
///
/// With [`OptLevel::Off`] this returns immediately, leaving the module — and therefore
/// every emitted byte — unchanged. Higher levels dispatch a *pinned* pass schedule.
pub fn optimize_mic3(module: &mut IRModule, level: OptLevel) {
    match level {
        OptLevel::Off => {
            // Strict no-op: do not touch `module`. Byte-identity is the contract.
        }
        OptLevel::Basic => {
            let mut next_id = module.next_id;
            const_if_prune(&mut module.instrs, &mut next_id);
            module.next_id = next_id;
        }
    }
}

/// The constant value of an `If` condition, iff `cond_instrs` is exactly one `ConstI64`
/// that defines `cond_id`. `None` for any non-constant / multi-instruction condition —
/// fail-safe: an ambiguous condition is never pruned.
fn const_cond_value(cond_instrs: &[Instr], cond_id: ValueId) -> Option<i64> {
    match cond_instrs {
        [Instr::ConstI64(id, v)] if *id == cond_id => Some(*v),
        _ => None,
    }
}

/// The `i64` constant that a value is defined to be, iff some `ConstI64` in `instrs`
/// defines `result`. Used to keep the pass PROVABLY type-safe: we only bind the outer
/// `dst` when we know the branch result is a concrete `i64` constant (so `dst` gets a
/// `ConstI64`, never a wrongly-typed copy). `None` (skip) otherwise.
fn i64_const_of(instrs: &[Instr], result: ValueId) -> Option<i64> {
    instrs.iter().find_map(|ins| match ins {
        Instr::ConstI64(id, v) if *id == result => Some(*v),
        _ => None,
    })
}

/// SCCP-lite slice 1 — **const-condition value-`If` pruning**, a structural region
/// rewrite (roadmap C6). When an `If`'s condition is a compile-time constant AND the `If`
/// rebinds no outer variable (`merges` and `branch_bindings` both empty — a pure value-if)
/// AND the taken branch's result is a concrete `i64` constant, replace the whole `If`
/// with the taken branch's instructions followed by `ConstI64(dst, that_value)` — binding
/// the outer `dst` to the constant. This is PROVABLY correct with no operand-type
/// reasoning and no cross-instruction value remap: the taken branch's side effects are
/// preserved by splicing, and `dst` is bound to a known `i64` constant. Every other shape
/// (non-constant condition, any merges/bindings, non-`i64`-const result) is left untouched
/// (fail-safe). Recurses into nested branches / loop bodies / fn bodies.
///
/// Deterministic: one forward pass in instruction order, no hashmap iteration; the only
/// new state is monotonic (`next_id`). The pinned schedule is "this pass, once".
fn const_if_prune(instrs: &mut Vec<Instr>, next_id: &mut usize) {
    let mut i = 0;
    while i < instrs.len() {
        // Recurse into nested regions first (a nested const-if becomes prunable too).
        match &mut instrs[i] {
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                const_if_prune(cond_instrs, next_id);
                const_if_prune(then_instrs, next_id);
                const_if_prune(else_instrs, next_id);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                const_if_prune(cond_instrs, next_id);
                const_if_prune(body, next_id);
            }
            Instr::FnDef { body, .. } => const_if_prune(body, next_id),
            _ => {}
        }

        // Decide whether THIS instruction is a prunable const-condition value-if, and if
        // so which branch is taken and what i64 constant its result is.
        let plan: Option<(bool, i64)> = match &instrs[i] {
            Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                merges,
                branch_bindings,
                ..
            } if merges.is_empty() && branch_bindings.is_empty() => {
                match const_cond_value(cond_instrs, *cond_id) {
                    Some(cv) => {
                        let take_then = cv != 0;
                        let (branch, result) = if take_then {
                            (then_instrs, *then_result)
                        } else {
                            (else_instrs, *else_result)
                        };
                        i64_const_of(branch, result).map(|v| (take_then, v))
                    }
                    None => None,
                }
            }
            _ => None,
        };

        if let Some((take_then, const_val)) = plan {
            let stolen = instrs.remove(i);
            if let Instr::If {
                then_instrs,
                else_instrs,
                dst,
                ..
            } = stolen
            {
                let mut branch = if take_then { then_instrs } else { else_instrs };
                let mut repl: Vec<Instr> = Vec::with_capacity(branch.len() + 1);
                repl.append(&mut branch);
                repl.push(Instr::ConstI64(dst, const_val));
                let n = repl.len();
                instrs.splice(i..i, repl);
                i += n;
                continue;
            } else {
                // Unreachable (we matched `If` above); restore defensively and move on.
                instrs.insert(i, stolen);
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_the_default() {
        assert_eq!(OptLevel::default(), OptLevel::Off);
    }

    #[test]
    fn basic_prunes_const_cond_value_if() {
        use crate::ir::Instr;
        // if (const 1) { const 7 } else { const 9 } -> %3 ; output %3
        let build = || {
            let mut m = IRModule::new();
            m.instrs = vec![
                Instr::If {
                    cond_id: ValueId(0),
                    cond_instrs: vec![Instr::ConstI64(ValueId(0), 1)],
                    then_instrs: vec![Instr::ConstI64(ValueId(1), 7)],
                    then_result: ValueId(1),
                    else_instrs: vec![Instr::ConstI64(ValueId(2), 9)],
                    else_result: ValueId(2),
                    dst: ValueId(3),
                    branch_bindings: vec![],
                    merges: vec![],
                },
                Instr::Output(ValueId(3)),
            ];
            m.next_id = 4;
            m
        };

        // OFF: strict no-op — the If is preserved (byte-identical path).
        let mut off = build();
        optimize_mic3(&mut off, OptLevel::Off);
        assert!(
            off.instrs.iter().any(|i| matches!(i, Instr::If { .. })),
            "OFF must not prune"
        );

        // Basic: prune to `dst = const 7` (then taken, cond=1); Output(%3) preserved.
        let mut basic = build();
        optimize_mic3(&mut basic, OptLevel::Basic);
        assert!(
            !basic.instrs.iter().any(|i| matches!(i, Instr::If { .. })),
            "Basic must prune the const-condition If"
        );
        assert!(
            basic
                .instrs
                .iter()
                .any(|i| matches!(i, Instr::ConstI64(id, 7) if *id == ValueId(3))),
            "dst must be bound to the taken-branch constant 7"
        );
        assert!(
            matches!(basic.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(3)),
            "Output(%3) must be preserved after the splice"
        );
    }

    #[test]
    fn basic_is_fail_safe_on_nonconstant_condition() {
        use crate::ir::{BinOp, Instr};
        // if (%0 > %1) { const 7 } else { const 9 } -> non-constant condition: never pruned.
        let mut m = IRModule::new();
        m.instrs = vec![Instr::If {
            cond_id: ValueId(2),
            cond_instrs: vec![Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Gt,
                lhs: ValueId(0),
                rhs: ValueId(1),
            }],
            then_instrs: vec![Instr::ConstI64(ValueId(3), 7)],
            then_result: ValueId(3),
            else_instrs: vec![Instr::ConstI64(ValueId(4), 9)],
            else_result: ValueId(4),
            dst: ValueId(5),
            branch_bindings: vec![],
            merges: vec![],
        }];
        m.next_id = 6;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            m.instrs.iter().any(|i| matches!(i, Instr::If { .. })),
            "a non-constant condition must never be pruned (fail-safe)"
        );
    }

    #[test]
    fn env_defaults_to_off() {
        // With no level wired, any environment resolves to Off (fail-safe).
        assert_eq!(opt_level_from_env(), OptLevel::Off);
    }
}
