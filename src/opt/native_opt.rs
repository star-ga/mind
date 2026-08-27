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
            const_if_prune(&mut module.instrs);
            fold_mul_by_zero(&mut module.instrs);
            fold_algebraic_identities(&mut module.instrs);
            cse_binops(&mut module.instrs);
            // Cleanup: the passes above orphan constants (an eliminated `*1`'s `1`) and
            // expose new const-const folds. Re-run the audited canonical passes ONCE — a
            // bounded, deterministic finalize (not "loop until quiet"), reusing
            // `prune_dead`/`constant_fold`/`reorder` rather than re-implementing DCE.
            crate::opt::ir_canonical::canonicalize_module(module);
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

/// SCCP-lite slice 1 — **const-condition value-`If` pruning**, a structural region
/// rewrite (roadmap C6). When an `If`'s condition is a compile-time constant AND the `If`
/// rebinds no outer variable (`merges` and `branch_bindings` both empty — a pure value-if),
/// replace the whole `If` with the taken branch's instructions and alias the `If`'s outer
/// `dst` to that branch's result value at every downstream use (via [`remap_operands`]).
/// Type-agnostic and correct with no copy and no operand-type reasoning: the taken
/// branch's side effects and result definition are preserved by splicing, and `dst` is a
/// pure rename to the surviving value (its definition, inside the spliced branch,
/// dominates the tail). Every other shape (non-constant condition, any merges/bindings) is
/// left untouched (fail-safe). Recurses into nested branches / loop bodies / fn bodies.
///
/// Deterministic: one forward pass in instruction order, no hashmap iteration. The pinned schedule is "this pass, once".
fn const_if_prune(instrs: &mut Vec<Instr>) {
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
                const_if_prune(cond_instrs);
                const_if_prune(then_instrs);
                const_if_prune(else_instrs);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                const_if_prune(cond_instrs);
                const_if_prune(body);
            }
            Instr::FnDef { body, .. } => const_if_prune(body),
            _ => {}
        }

        // Decide whether THIS instruction is a prunable const-condition value-if, and if
        // so which branch is taken. Guarded to a pure value-if (no `merges` /
        // `branch_bindings`) so the only outer reference is `dst`, aliased below via a
        // remap — no branch-merge / fn-env-binding reasoning.
        let take_then: Option<bool> = match &instrs[i] {
            Instr::If {
                cond_id,
                cond_instrs,
                merges,
                branch_bindings,
                ..
            } if merges.is_empty() && branch_bindings.is_empty() => {
                const_cond_value(cond_instrs, *cond_id).map(|cv| cv != 0)
            }
            _ => None,
        };

        if let Some(take_then) = take_then {
            let stolen = instrs.remove(i);
            if let Instr::If {
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                dst,
                ..
            } = stolen
            {
                let (branch, result) = if take_then {
                    (then_instrs, then_result)
                } else {
                    (else_instrs, else_result)
                };
                let n = branch.len();
                // Splice the surviving branch in place of the `If` …
                instrs.splice(i..i, branch);
                // … then alias the `If`'s outer `dst` to that branch's result value at
                // every downstream use (no copy, no type assumption; `result`'s
                // definition is inside the just-spliced branch, so it dominates the tail).
                remap_operands(&mut instrs[i + n..], dst, result);
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

/// SCCP-lite slice 2 — **integer `mul`-by-zero folding**, an in-place algebraic rewrite.
/// A `BinOp { op: Mul, .. }` one of whose operands is an `i64` constant `0` (defined by a
/// `ConstI64(id, 0)` in the *same* instruction vector — hence the same SSA value space and
/// a valid earlier definition) is replaced in place by `ConstI64(dst, 0)`. Provably exact
/// and type-safe: `x * 0 == 0` for every integer / Q16.16 representation, and matching only
/// `ConstI64`-0 operands means float multiplies (whose zero is a `ConstF64`, where
/// `inf*0`/`NaN*0`/`-0.0` differ) are never touched. The result vid is preserved, so no
/// cross-instruction value remap is needed. Recurses into nested branches / bodies;
/// conservative across scopes (only same-vector const-0s fold — a safe miss, never a
/// wrong fold). Deterministic: one forward pass, no hashmap iteration.
fn fold_mul_by_zero(instrs: &mut Vec<Instr>) {
    use std::collections::BTreeSet;
    let zeros: BTreeSet<ValueId> = instrs
        .iter()
        .filter_map(|ins| match ins {
            Instr::ConstI64(id, 0) => Some(*id),
            _ => None,
        })
        .collect();
    let mut i = 0;
    while i < instrs.len() {
        match &mut instrs[i] {
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                fold_mul_by_zero(cond_instrs);
                fold_mul_by_zero(then_instrs);
                fold_mul_by_zero(else_instrs);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                fold_mul_by_zero(cond_instrs);
                fold_mul_by_zero(body);
            }
            Instr::FnDef { body, .. } => fold_mul_by_zero(body),
            _ => {}
        }
        if let Instr::BinOp {
            dst,
            op: crate::ir::BinOp::Mul,
            lhs,
            rhs,
        } = &instrs[i]
        {
            if zeros.contains(lhs) || zeros.contains(rhs) {
                let d = *dst;
                instrs[i] = Instr::ConstI64(d, 0);
            }
        }
        i += 1;
    }
}

/// The surviving operand of a `BinOp` that is an exact algebraic identity, or `None`.
/// `zeros`/`ones` are the value ids of same-scope `ConstI64` `0`/`1` definitions — so a
/// match is always an *integer* identity: a float zero/one is a `ConstF64` (never in these
/// sets), and every Q16.16 fixed-point identity uses the scaled constant `65536` (its
/// "one"), never a literal `0`/`1`, so the fixed-point tier is structurally never matched.
/// Commutativity is respected exactly — `x-0`/`x/1`/`x>>0` fold (but not `0-x`, `1/x`,
/// `0>>x`, which are negate / reciprocal / different values).
fn identity_survivor(
    op: crate::ir::BinOp,
    lhs: ValueId,
    rhs: ValueId,
    zeros: &std::collections::BTreeSet<ValueId>,
    ones: &std::collections::BTreeSet<ValueId>,
) -> Option<ValueId> {
    use crate::ir::BinOp as B;
    match op {
        // Additive identity, commutative: x+0 == 0+x == x ; x|0 == x ; x^0 == x.
        B::Add | B::BitOr | B::BitXor => {
            if zeros.contains(&rhs) {
                Some(lhs)
            } else if zeros.contains(&lhs) {
                Some(rhs)
            } else {
                None
            }
        }
        // Right-identity only (non-commutative): x-0 == x ; x<<0 == x ; x>>0 == x.
        B::Sub | B::Shl | B::Shr => zeros.contains(&rhs).then_some(lhs),
        // Multiplicative identity, commutative: x*1 == 1*x == x.
        B::Mul => {
            if ones.contains(&rhs) {
                Some(lhs)
            } else if ones.contains(&lhs) {
                Some(rhs)
            } else {
                None
            }
        }
        // Right-identity only (non-commutative): x/1 == x (exact for every integer,
        // including INT_MIN unlike /-1). `1/x` is a reciprocal — never folded.
        B::Div => ones.contains(&rhs).then_some(lhs),
        _ => None,
    }
}

/// SCCP-lite slice 3 — **exact integer algebraic-identity elimination**. A `BinOp` that is
/// an identity on one operand (see [`identity_survivor`]) is removed and its `dst` aliased
/// to the surviving operand at every downstream use via [`remap_operands`] — the survivor's
/// definition dominates (it was an operand of the removed op), so this is a pure rename with
/// no copy and no type reasoning. Only same-vector `ConstI64` `0`/`1` operands match, so the
/// fold is always integer-exact and never touches the Q16.16 or float tiers (a safe miss
/// across scopes, never a wrong fold). Runs after [`fold_mul_by_zero`], so a folded `x*0`'s
/// fresh `ConstI64 0` feeds this pass too (e.g. `y + (x*0)` collapses to `y`). Recurses into
/// nested branches / bodies. Deterministic: one forward pass, no hashmap iteration.
fn fold_algebraic_identities(instrs: &mut Vec<Instr>) {
    use std::collections::BTreeSet;
    let zeros: BTreeSet<ValueId> = instrs
        .iter()
        .filter_map(|ins| match ins {
            Instr::ConstI64(id, 0) => Some(*id),
            _ => None,
        })
        .collect();
    let ones: BTreeSet<ValueId> = instrs
        .iter()
        .filter_map(|ins| match ins {
            Instr::ConstI64(id, 1) => Some(*id),
            _ => None,
        })
        .collect();
    let mut i = 0;
    while i < instrs.len() {
        // Recurse into nested regions first (their operand interface is remapped below).
        match &mut instrs[i] {
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                fold_algebraic_identities(cond_instrs);
                fold_algebraic_identities(then_instrs);
                fold_algebraic_identities(else_instrs);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                fold_algebraic_identities(cond_instrs);
                fold_algebraic_identities(body);
            }
            Instr::FnDef { body, .. } => fold_algebraic_identities(body),
            _ => {}
        }

        let alias: Option<(ValueId, ValueId)> = match &instrs[i] {
            Instr::BinOp { dst, op, lhs, rhs } => {
                identity_survivor(*op, *lhs, *rhs, &zeros, &ones).map(|s| (*dst, s))
            }
            _ => None,
        };

        if let Some((dst, survivor)) = alias {
            // Remove the identity op and rename its result to the surviving operand at every
            // enclosing-scope use in the tail (all uses are define-after this point in SSA).
            instrs.remove(i);
            remap_operands(&mut instrs[i..], dst, survivor);
            continue; // a new instruction now occupies index `i`.
        }
        i += 1;
    }
}

/// The value-numbering key of a pure `BinOp`: its opcode discriminant plus operand value
/// ids, with the operands sorted for the commutative opcodes so `a+b` and `b+a` collide.
/// `BinOp` is a field-less enum, so `op as u8` is a stable per-opcode discriminant.
fn binop_key(op: crate::ir::BinOp, lhs: ValueId, rhs: ValueId) -> (u8, ValueId, ValueId) {
    use crate::ir::BinOp as B;
    let commutative = matches!(
        op,
        B::Add | B::Mul | B::BitAnd | B::BitOr | B::BitXor | B::Eq | B::Ne
    );
    let (a, b) = if commutative && rhs < lhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    };
    (op as u8, a, b)
}

/// CSE/GVN slice — **region-scoped common-subexpression elimination of pure `BinOp`s**.
/// Within a single instruction vector (one straight-line region body — one basic block for
/// dominance purposes), a `BinOp` whose value-numbering key ([`binop_key`]) already appeared
/// is redundant: because value ids are immutable in SSA, identical (opcode, operands) means a
/// bit-identical result for EVERY operand type (no type reasoning needed). The later op is
/// removed and its `dst` aliased to the first op's `dst` via [`remap_operands`]; the kept op
/// dominates (it is earlier in the same block) so this is a pure rename, and nothing is ever
/// reordered (so trap ordering for `Div`/`Mod` is preserved). Merging compounds: a remap can
/// make a downstream op newly match an earlier key. Scoped strictly to one vector — nested
/// regions are each an INDEPENDENT CSE scope (a value defined in one branch is not available
/// in a sibling), so cross-region merges are conservatively never made. Deterministic: one
/// forward pass keyed on a `BTreeMap`, no iteration-order dependence.
fn cse_binops(instrs: &mut Vec<Instr>) {
    use std::collections::BTreeMap;
    let mut seen: BTreeMap<(u8, ValueId, ValueId), ValueId> = BTreeMap::new();
    let mut i = 0;
    while i < instrs.len() {
        // Recurse into nested regions first — each is its own independent CSE scope.
        match &mut instrs[i] {
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                cse_binops(cond_instrs);
                cse_binops(then_instrs);
                cse_binops(else_instrs);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                cse_binops(cond_instrs);
                cse_binops(body);
            }
            Instr::FnDef { body, .. } => cse_binops(body),
            _ => {}
        }

        let redundant: Option<(ValueId, ValueId)> = match &instrs[i] {
            Instr::BinOp { dst, op, lhs, rhs } => {
                let key = binop_key(*op, *lhs, *rhs);
                match seen.get(&key) {
                    Some(&first) => Some((*dst, first)),
                    None => {
                        seen.insert(key, *dst);
                        None
                    }
                }
            }
            _ => None,
        };

        if let Some((dst, first)) = redundant {
            // The kept (first) op dominates in this block; alias the redundant result to it.
            instrs.remove(i);
            remap_operands(&mut instrs[i..], dst, first);
            continue; // a new instruction now occupies index `i`.
        }
        i += 1;
    }
}

/// Mutable mirror of `crate::opt::ir_canonical::for_each_operand`: visit every
/// ENCLOSING-SCOPE operand value id of `instr` by mutable reference. It MUST cover the
/// exact same operand set as `for_each_operand` (they are two views of one audited list)
/// — deliberately NOT descending into `If`/`While`/`Region`/`FnDef` bodies, because those
/// hold their own SSA namespace; an enclosing value they read is threaded through
/// `If.merges` (`then_val`/`else_val`) or `While.init_ids`, which ARE visited here. Used
/// by [`remap_operands`] to rename a value id everywhere it is *used* (never at a def).
fn for_each_operand_mut(instr: &mut Instr, mut f: impl FnMut(&mut ValueId)) {
    use crate::ir::Instr as I;
    match instr {
        I::ConstI64(_, _)
        | I::ConstF64(_, _)
        | I::ConstTensor(_, _, _, _)
        | I::ConstDenseTensor { .. } => {}
        I::BinOp { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        I::Sum { src, .. }
        | I::Mean { src, .. }
        | I::Relu { src, .. }
        | I::Reshape { src, .. }
        | I::ExpandDims { src, .. }
        | I::Squeeze { src, .. }
        | I::Transpose { src, .. }
        | I::Index { src, .. }
        | I::Slice { src, .. }
        | I::SparseAttr { src, .. } => f(src),
        I::Dot { a, b, .. } | I::MatMul { a, b, .. } => {
            f(a);
            f(b);
        }
        I::Conv2d { input, filter, .. } => {
            f(input);
            f(filter);
        }
        I::Conv2dGradInput { dy, filter, .. } => {
            f(dy);
            f(filter);
        }
        I::Conv2dGradFilter { input, dy, .. } => {
            f(input);
            f(dy);
        }
        I::ReluGrad { grad, src, .. } => {
            f(grad);
            f(src);
        }
        I::Gather { src, indices, .. } => {
            f(src);
            f(indices);
        }
        I::Output(id) => f(id),
        I::Call { args, .. } => {
            for a in args.iter_mut() {
                f(a);
            }
        }
        I::Return { value } => {
            if let Some(v) = value {
                f(v);
            }
        }
        I::Param { .. } | I::FnDef { .. } => {}
        #[cfg(feature = "std-surface")]
        I::ConstArray { .. } => {}
        #[cfg(feature = "std-surface")]
        I::ArrayLoad { base, index, .. } => {
            f(base);
            f(index);
        }
        #[cfg(feature = "std-surface")]
        I::While { init_ids, .. } => {
            for v in init_ids.iter_mut() {
                f(v);
            }
        }
        #[cfg(feature = "std-surface")]
        I::If { merges, .. } => {
            for (_merge, then_val, else_val) in merges.iter_mut() {
                f(then_val);
                f(else_val);
            }
        }
        #[cfg(feature = "std-surface")]
        I::VecLoad { base, offset, .. } | I::VecLoadI32 { base, offset, .. } => {
            f(base);
            f(offset);
        }
        #[cfg(feature = "std-surface")]
        I::VecFma { a, b, acc, .. } | I::VecMulAddQ16 { a, b, acc, .. } => {
            f(a);
            f(b);
            f(acc);
        }
        #[cfg(feature = "std-surface")]
        I::VecReduceAdd { src, .. } | I::VecReduceAddI64 { src, .. } => f(src),
        #[cfg(feature = "std-surface")]
        I::VecStore {
            src, base, offset, ..
        } => {
            f(src);
            f(base);
            f(offset);
        }
        #[cfg(feature = "std-surface")]
        I::ExternFnDecl { .. } => {}
        #[cfg(feature = "std-surface")]
        I::Region { .. } => {}
        #[cfg(feature = "std-surface")]
        I::Break { live } | I::Continue { live } => {
            for (_, v) in live.iter_mut() {
                f(v);
            }
        }
    }
}

/// Rename value id `from` -> `to` at every ENCLOSING-SCOPE USE in `instrs` (never at a
/// definition — `dst` fields are not visited by [`for_each_operand_mut`]). Used to alias
/// a pruned `If`'s `dst`/merge ids to the surviving branch's value without a copy or a
/// type assumption. Deterministic: one forward pass, no hashmap iteration.
fn remap_operands(instrs: &mut [Instr], from: ValueId, to: ValueId) {
    if from == to {
        return;
    }
    for ins in instrs.iter_mut() {
        for_each_operand_mut(ins, |op| {
            if *op == from {
                *op = to;
            }
        });
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

        // Basic: prune to the then-branch (cond=1). Its ConstI64(%1, 7) survives; the
        // outer dst (%3) is aliased to the branch result (%1), so Output(%3) -> Output(%1).
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
                .any(|i| matches!(i, Instr::ConstI64(id, 7) if *id == ValueId(1))),
            "the taken (then) branch's const 7 (%1) must survive the splice"
        );
        assert!(
            matches!(basic.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(1)),
            "Output must be remapped from dst(%3) to the branch result(%1)"
        );
    }

    #[test]
    fn basic_prunes_const_if_with_computed_result_via_remap() {
        use crate::ir::{BinOp, Instr};
        // params %0,%1 ; if (const 1) { %3 = %0 + %1 } else { const 9 } -> %4 ; output %4
        // then taken: splice `%3 = %0 + %1`; alias dst(%4) -> result(%3) -> Output(%3).
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::If {
                cond_id: ValueId(2),
                cond_instrs: vec![Instr::ConstI64(ValueId(2), 1)],
                then_instrs: vec![Instr::BinOp {
                    dst: ValueId(3),
                    op: BinOp::Add,
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                }],
                then_result: ValueId(3),
                else_instrs: vec![Instr::ConstI64(ValueId(5), 9)],
                else_result: ValueId(5),
                dst: ValueId(4),
                branch_bindings: vec![],
                merges: vec![],
            },
            Instr::Output(ValueId(4)),
        ];
        m.next_id = 6;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            !m.instrs.iter().any(|i| matches!(i, Instr::If { .. })),
            "the const-condition If must be pruned"
        );
        // the then-branch's Add(%3) survives; Output(%4) is remapped to Output(%3).
        assert!(
            m.instrs.iter().any(
                |i| matches!(i, Instr::BinOp { dst, op: BinOp::Add, .. } if *dst == ValueId(3))
            ),
            "the taken branch's computation (%3 = %0+%1) must survive"
        );
        assert!(
            matches!(m.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(3)),
            "Output must be remapped dst(%4) -> branch result(%3)"
        );
        // the dead else-branch's const 9 must be gone.
        assert!(
            !m.instrs.iter().any(|i| matches!(i, Instr::ConstI64(_, 9))),
            "the dead else-branch must be dropped"
        );
    }

    #[test]
    fn basic_is_fail_safe_on_nonconstant_condition() {
        use crate::ir::{BinOp, Instr};
        // if (%0 > %1) { const 7 } else { const 9 } -> non-constant condition: never pruned.
        // %0/%1 are params (non-const) and Output(%5) keeps the If's result live so the
        // canonical cleanup's dead-code pass does not remove the whole If.
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".into(),
                index: 0,
            },
            Instr::Param {
                dst: ValueId(1),
                name: "b".into(),
                index: 1,
            },
            Instr::If {
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
            },
            Instr::Output(ValueId(5)),
        ];
        m.next_id = 6;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            m.instrs.iter().any(|i| matches!(i, Instr::If { .. })),
            "a non-constant condition must never be pruned (fail-safe)"
        );
    }

    #[test]
    fn basic_folds_integer_mul_by_zero() {
        use crate::ir::{BinOp, Instr};
        // %0 = const 0 ; %1 = const 5 ; %2 = %1 * %0 ; output %2  ->  %2 = const 0
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::ConstI64(ValueId(0), 0),
            Instr::ConstI64(ValueId(1), 5),
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Mul,
                lhs: ValueId(1),
                rhs: ValueId(0),
            },
            Instr::Output(ValueId(2)),
        ];
        m.next_id = 3;

        // OFF: unchanged (the Mul stays).
        let mut off = m.clone();
        optimize_mic3(&mut off, OptLevel::Off);
        assert!(
            off.instrs
                .iter()
                .any(|i| matches!(i, Instr::BinOp { op: BinOp::Mul, .. })),
            "OFF must not fold"
        );

        // Basic: the mul-by-zero folds to `%2 = const 0` in place.
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            !m.instrs
                .iter()
                .any(|i| matches!(i, Instr::BinOp { op: BinOp::Mul, .. })),
            "mul-by-zero must be folded away"
        );
        assert!(
            m.instrs
                .iter()
                .any(|i| matches!(i, Instr::ConstI64(id, 0) if *id == ValueId(2))),
            "the mul's dst (%2) must be bound to const 0 (result vid preserved)"
        );
    }

    #[test]
    fn basic_folds_add_zero_via_remap() {
        use crate::ir::{BinOp, Instr};
        // %0 = const 5 (x) ; %1 = const 0 ; %2 = %0 + %1 ; output %2  ->  output %0
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::ConstI64(ValueId(0), 5),
            Instr::ConstI64(ValueId(1), 0),
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::Output(ValueId(2)),
        ];
        m.next_id = 3;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            !m.instrs
                .iter()
                .any(|i| matches!(i, Instr::BinOp { op: BinOp::Add, .. })),
            "x+0 must be eliminated"
        );
        assert!(
            matches!(m.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(0)),
            "Output must be remapped from the add's dst(%2) to the surviving operand(%0)"
        );
    }

    #[test]
    fn basic_folds_mul_one_commutative() {
        use crate::ir::{BinOp, Instr};
        // %0 = const 1 ; %1 = const 7 (x) ; %2 = %0 * %1 (1*x) ; output %2  ->  output %1
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::ConstI64(ValueId(0), 1),
            Instr::ConstI64(ValueId(1), 7),
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Mul,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::Output(ValueId(2)),
        ];
        m.next_id = 3;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            matches!(m.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(1)),
            "1*x must alias to x (the non-one operand)"
        );
    }

    #[test]
    fn basic_sub_folds_only_right_zero() {
        use crate::ir::{BinOp, Instr};
        // 0 - x is NEGATION, never an identity: %0=const 0 ; %1=param x ; %2 = %0 - %1.
        // x is a param (not const) so the canonical cleanup cannot const-fold the Sub away —
        // isolating the identity pass's decision (it correctly leaves 0-x alone).
        let mut neg = IRModule::new();
        neg.instrs = vec![
            Instr::ConstI64(ValueId(0), 0),
            Instr::Param {
                dst: ValueId(1),
                name: "x".into(),
                index: 0,
            },
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Sub,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::Output(ValueId(2)),
        ];
        neg.next_id = 3;
        optimize_mic3(&mut neg, OptLevel::Basic);
        assert!(
            neg.instrs
                .iter()
                .any(|i| matches!(i, Instr::BinOp { op: BinOp::Sub, .. })),
            "0 - x is negation and must NOT be folded"
        );

        // x - 0 IS x: %0=param x ; %1=const 0 ; %2 = %0 - %1 ; output %2 -> output %0.
        let mut ident = IRModule::new();
        ident.instrs = vec![
            Instr::Param {
                dst: ValueId(0),
                name: "x".into(),
                index: 0,
            },
            Instr::ConstI64(ValueId(1), 0),
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Sub,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::Output(ValueId(2)),
        ];
        ident.next_id = 3;
        optimize_mic3(&mut ident, OptLevel::Basic);
        assert!(
            matches!(ident.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(0)),
            "x - 0 must alias to x"
        );
    }

    #[test]
    fn basic_mul_by_zero_then_add_identity_compounds() {
        use crate::ir::{BinOp, Instr};
        // %0=const 3 (x) ; %1=const 0 ; %2 = %0*%1 (=0) ; %3=const 8 (y) ; %4 = %3 + %2 ;
        // output %4.  mul-by-zero -> %2=const 0 ; then y+0 -> output %3.
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::ConstI64(ValueId(0), 3),
            Instr::ConstI64(ValueId(1), 0),
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Mul,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::ConstI64(ValueId(3), 8),
            Instr::BinOp {
                dst: ValueId(4),
                op: BinOp::Add,
                lhs: ValueId(3),
                rhs: ValueId(2),
            },
            Instr::Output(ValueId(4)),
        ];
        m.next_id = 5;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert!(
            !m.instrs.iter().any(|i| matches!(i, Instr::BinOp { .. })),
            "both the mul-by-zero and the resulting add-zero must be eliminated"
        );
        assert!(
            matches!(m.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(3)),
            "the chain must collapse to y (%3)"
        );
    }

    #[test]
    fn basic_cse_merges_redundant_binop() {
        use crate::ir::{BinOp, Instr};
        // %2 = %0 + %1 ; %3 = %0 + %1 (redundant) ; output %3  ->  output %2, one Add.
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".into(),
                index: 0,
            },
            Instr::Param {
                dst: ValueId(1),
                name: "b".into(),
                index: 1,
            },
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::BinOp {
                dst: ValueId(3),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::Output(ValueId(3)),
        ];
        m.next_id = 4;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert_eq!(
            m.instrs
                .iter()
                .filter(|i| matches!(i, Instr::BinOp { op: BinOp::Add, .. }))
                .count(),
            1,
            "the redundant Add must be eliminated (one survives)"
        );
        assert!(
            matches!(m.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(2)),
            "the redundant result(%3) must be aliased to the first(%2)"
        );
    }

    #[test]
    fn basic_cse_normalizes_commutative_operands() {
        use crate::ir::{BinOp, Instr};
        // %2 = %0 + %1 ; %3 = %1 + %0 (swapped operands) ; output %3  ->  merged, output %2.
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".into(),
                index: 0,
            },
            Instr::Param {
                dst: ValueId(1),
                name: "b".into(),
                index: 1,
            },
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::BinOp {
                dst: ValueId(3),
                op: BinOp::Add,
                lhs: ValueId(1),
                rhs: ValueId(0),
            },
            Instr::Output(ValueId(3)),
        ];
        m.next_id = 4;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert_eq!(
            m.instrs
                .iter()
                .filter(|i| matches!(i, Instr::BinOp { .. }))
                .count(),
            1,
            "a+b and b+a must value-number to the same expression"
        );
        assert!(
            matches!(m.instrs.last(), Some(Instr::Output(id)) if *id == ValueId(2)),
            "swapped-operand duplicate must alias to the first"
        );
    }

    #[test]
    fn basic_cse_keeps_noncommutative_distinct() {
        use crate::ir::{BinOp, Instr};
        // %2 = %0 - %1 ; %3 = %1 - %0 : different values — must NOT be merged.
        let mut m = IRModule::new();
        m.instrs = vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".into(),
                index: 0,
            },
            Instr::Param {
                dst: ValueId(1),
                name: "b".into(),
                index: 1,
            },
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Sub,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            Instr::BinOp {
                dst: ValueId(3),
                op: BinOp::Sub,
                lhs: ValueId(1),
                rhs: ValueId(0),
            },
            // consume BOTH so the cleanup's dead-code pass keeps them live.
            Instr::BinOp {
                dst: ValueId(4),
                op: BinOp::Add,
                lhs: ValueId(2),
                rhs: ValueId(3),
            },
            Instr::Output(ValueId(4)),
        ];
        m.next_id = 5;
        optimize_mic3(&mut m, OptLevel::Basic);
        assert_eq!(
            m.instrs
                .iter()
                .filter(|i| matches!(i, Instr::BinOp { op: BinOp::Sub, .. }))
                .count(),
            2,
            "a-b and b-a are distinct and must both survive"
        );
    }

    #[test]
    fn env_defaults_to_off() {
        // With no level wired, any environment resolves to Off (fail-safe).
        assert_eq!(opt_level_from_env(), OptLevel::Off);
    }
}
