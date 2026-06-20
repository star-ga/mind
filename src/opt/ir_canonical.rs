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

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{BinOp, IRModule, Instr, ValueId, instruction_dst};

/// Canonicalize the public MIND IR in-place.
///
/// The pass is intentionally conservative: it keeps the existing SSA IDs,
/// performs deterministic cleanups (operand ordering, trivial constant folding),
/// and prunes provably dead instructions. Running the pass repeatedly is
/// idempotent.
///
/// Feature gate for REAP MoE pruning: the dead-expert DCE sub-pass only runs
/// when at least one function in the module carries a `reap_threshold` attribute.
/// This keeps the headline compile-speed benches unaffected for non-MoE code.
///
/// Reference (REAP pruning): arXiv:2510.13999.
pub fn canonicalize_module(module: &mut IRModule) {
    // REAP dead-expert DCE: feature-gated on presence of `reap_threshold`.
    if module_has_reap_threshold(module) {
        prune_dead_experts(module);
    }

    // Own the stream (no clone) and run every pass in place — prune_dead used to
    // clone the whole instruction stream into a fresh Vec twice; that was ~25% of
    // a small compile (perf: __memmove + prune_dead malloc).
    let mut instrs = std::mem::take(&mut module.instrs);
    let n = module.next_id;
    prune_dead(&mut instrs, n);
    reorder_commutative_ops(&mut instrs);
    constant_fold(&mut instrs);
    prune_dead(&mut instrs, n);

    module.instrs = instrs;
    module.next_id = next_sequential_id(module);
}

// ---------------------------------------------------------------------------
// REAP dead-expert DCE
// ---------------------------------------------------------------------------

/// Returns true iff any `FnDef` in the module's top-level instruction stream
/// carries a `reap_threshold` attribute.  O(n) scan; result cached by caller.
fn module_has_reap_threshold(module: &IRModule) -> bool {
    module.instrs.iter().any(|instr| {
        matches!(
            instr,
            Instr::FnDef {
                reap_threshold: Some(_),
                ..
            }
        )
    })
}

/// Dead-expert DCE pass (REAP-style).
///
/// For each `FnDef` with `reap_threshold = Some(t)`:
///  1. Collect all `Instr::Call` sites that reference the function anywhere in
///     the module (top-level and nested inside other `FnDef` bodies).
///  2. If the function has zero `Instr::Call` references AND there are no other
///     `FnDef` nodes in the module that could act as implicit callers (e.g.
///     through paths the lowerer has not yet emitted `Instr::Call` for), the
///     expert is considered dead and its body is replaced with a
///     `ConstI64(ret_id, 0)` tombstone to preserve SSA integrity.
///
/// The conservative "no other FnDefs" guard prevents false positives when the
/// IR lowering pass does not yet emit `Instr::Call` for all user-defined
/// function invocations.  In that regime a router function is represented as a
/// `FnDef` in the module but does not produce explicit `Instr::Call` entries
/// for experts it dispatches to.
///
/// In production MoE routing, routing decisions gate expert reachability at
/// compile time when the threshold is known statically.  This pass implements
/// the conservative baseline; threshold-guided pruning (retaining top-k experts
/// based on activation statistics) is a follow-up requiring routing-function
/// analysis.
///
/// Reference (REAP pruning): arXiv:2510.13999.
///
/// The pass is idempotent and feature-gated (only runs when
/// `module_has_reap_threshold` returns true).
fn prune_dead_experts(module: &mut IRModule) {
    // Collect names of all functions that are explicitly called via Instr::Call
    // — both at top-level and inside any FnDef body.
    let mut all_called: BTreeSet<String> = BTreeSet::new();
    for instr in &module.instrs {
        match instr {
            Instr::Call { name, .. } => {
                all_called.insert(name.clone());
            }
            Instr::FnDef { body, .. } => {
                for bi in body {
                    if let Instr::Call { name, .. } = bi {
                        all_called.insert(name.clone());
                    }
                }
            }
            _ => {}
        }
    }

    // Conservative guard: if there are any FnDef nodes WITHOUT reap_threshold
    // in the module, they could be router/caller functions that invoke experts
    // through code paths the lowerer has not yet represented as Instr::Call.
    // In that case we cannot safely prune — skip the pass entirely.
    let has_potential_callers = module.instrs.iter().any(|instr| {
        matches!(
            instr,
            Instr::FnDef {
                reap_threshold: None,
                ..
            }
        )
    });
    if has_potential_callers {
        return;
    }

    // Tombstone unreachable expert bodies.
    for instr in &mut module.instrs {
        if let Instr::FnDef {
            name,
            reap_threshold: Some(_),
            body,
            ret_id,
            ..
        } = instr
        {
            if !all_called.contains(name) {
                // Replace body with a single tombstone instruction.
                let tombstone_id = ret_id.unwrap_or(ValueId(usize::MAX));
                *body = vec![Instr::ConstI64(tombstone_id, 0)];
            }
        }
    }
}

fn prune_dead(instrs: &mut Vec<Instr>, n: usize) {
    // Flat bitset indexed by ValueId.0 — eliminates the BTreeSet red-black malloc
    // + pointer-chasing for what is a dense integer key space. The lowerer mints
    // every normal id in 0..n, so for in-range ids the marked set and visit order
    // are byte-identical to the old BTreeSet. The lone out-of-range id is the REAP
    // dead-expert tombstone `ValueId(usize::MAX)` (only when a fn carries
    // `reap_threshold`); it is never read, so the bounds-checked `get`/`get_mut`
    // treat it as unused — exactly as `BTreeSet::contains` would — instead of
    // panicking. The checks are perfectly predicted for the common in-range path.
    let mut used: Vec<bool> = vec![false; n];
    let is_used = |used: &[bool], id: ValueId| used.get(id.0).copied().unwrap_or(false);
    let mark = |used: &mut [bool], id: ValueId| {
        if let Some(slot) = used.get_mut(id.0) {
            *slot = true;
        }
    };
    for instr in instrs.iter().rev() {
        match instr {
            Instr::Output(id) => {
                mark(&mut used, *id);
            }
            other => {
                let dst = instruction_dst(other);
                // Keep instructions whose destination is either unused (None) or present
                // in the `used` set. Clippy prefers `is_none_or` over `map_or(true, ...)`.
                if dst.is_none_or(|id| is_used(&used, id)) {
                    for_each_operand(other, |operand| {
                        mark(&mut used, operand);
                    });
                }
            }
        }
    }

    // Retain in place — the predicate keeps exactly the same instructions, in the
    // same order, as the old clone-into-a-new-Vec loop, so canonical output stays
    // byte-identical while avoiding the per-instruction clone + the second Vec.
    instrs.retain(|instr| instruction_dst(instr).is_none_or(|dst| is_used(&used, dst)));
}

fn reorder_commutative_ops(instrs: &mut [Instr]) {
    for instr in instrs.iter_mut() {
        if let Instr::BinOp { op, lhs, rhs, .. } = instr {
            if matches!(op, BinOp::Add | BinOp::Mul) && rhs < lhs {
                std::mem::swap(lhs, rhs);
            }
        }
    }
}

fn constant_fold(instrs: &mut [Instr]) {
    let mut constants: BTreeMap<ValueId, i64> = BTreeMap::new();
    for instr in instrs.iter_mut() {
        match instr {
            Instr::ConstI64(id, value) => {
                constants.insert(*id, *value);
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                let dst_id = *dst;
                let lhs_id = *lhs;
                let rhs_id = *rhs;
                let op_kind = *op;

                if let (Some(l), Some(r)) = (
                    constants.get(&lhs_id).copied(),
                    constants.get(&rhs_id).copied(),
                ) {
                    let folded = match op_kind {
                        BinOp::Add => l.saturating_add(r),
                        BinOp::Sub => l.saturating_sub(r),
                        BinOp::Mul => l.saturating_mul(r),
                        BinOp::Div => {
                            if r == 0 || (l == i64::MIN && r == -1) {
                                continue;
                            }
                            l / r
                        }
                        BinOp::Mod => {
                            if r == 0 || (l == i64::MIN && r == -1) {
                                continue;
                            }
                            l % r
                        }
                        BinOp::Lt => (l < r) as i64,
                        BinOp::Le => (l <= r) as i64,
                        BinOp::Gt => (l > r) as i64,
                        BinOp::Ge => (l >= r) as i64,
                        BinOp::Eq => (l == r) as i64,
                        BinOp::Ne => (l != r) as i64,
                        // Phase 6.5 Stage 1a — bitwise constant folding.
                        #[cfg(feature = "std-surface")]
                        BinOp::BitAnd => l & r,
                        #[cfg(feature = "std-surface")]
                        BinOp::BitOr => l | r,
                        #[cfg(feature = "std-surface")]
                        BinOp::BitXor => l ^ r,
                        #[cfg(feature = "std-surface")]
                        BinOp::Shl => l.wrapping_shl(r as u32),
                        #[cfg(feature = "std-surface")]
                        BinOp::Shr => l >> r,
                    };
                    *instr = Instr::ConstI64(dst_id, folded);
                    constants.insert(dst_id, folded);
                    continue;
                }

                constants.remove(&dst_id);
            }
            _ => {
                if let Some(dst) = instruction_dst(instr) {
                    constants.remove(&dst);
                }
            }
        }
    }
}

fn next_sequential_id(module: &IRModule) -> usize {
    module
        .instrs
        .iter()
        .filter_map(instruction_dst)
        .map(|id| id.0 + 1)
        .max()
        .unwrap_or(0)
}

/// Enumerate every SSA value an instruction *reads* from the scope that
/// `prune_dead` iterates (the flat top-level / fn-body instruction stream).
///
/// This match is intentionally **exhaustive with no `_` catch-all**: a wildcard
/// previously returned an empty operand set for `Call`, `Return`, `Param` and
/// every std-surface instruction, so a value consumed *only* as a call argument
/// or return value was never marked live and got pruned — a silent miscompile
/// (the std-surface UFCS bricks lower `v.push(x)` to a `vec_push` `Instr::Call`).
/// Removing the catch-all makes any future `Instr` variant a compile error here,
/// forcing the author to declare its operands.
///
/// Control-flow instructions (`While`/`If`/`Region`) carry sub-instruction
/// streams in their own SSA namespaces; only the ids they read from the
/// *enclosing* scope are returned here (`While::init_ids` pre-loop values, and
/// the `then_val`/`else_val` of `If::merges`, which may forward an enclosing
/// value for a var unassigned on one branch). The sub-body result/exit/merge
/// ids are produced by the region, not consumed from the enclosing scope.
/// Invoke `f` once per SSA operand `instr` READS from the enclosing scope —
/// allocation-free. This is the hot primitive (`prune_dead`'s used-set pass);
/// `instruction_operands` is the collect-into-Vec form for callers that need one.
/// The two MUST visit the exact same operand set (DCE correctness + determinism).
pub(crate) fn for_each_operand(instr: &Instr, mut f: impl FnMut(ValueId)) {
    match instr {
        Instr::ConstI64(_, _) | Instr::ConstF64(_, _) | Instr::ConstTensor(_, _, _, _) => {}
        Instr::BinOp { lhs, rhs, .. } => {
            f(*lhs);
            f(*rhs);
        }
        Instr::Sum { src, .. }
        | Instr::Mean { src, .. }
        | Instr::Relu { src, .. }
        | Instr::Reshape { src, .. }
        | Instr::ExpandDims { src, .. }
        | Instr::Squeeze { src, .. }
        | Instr::Transpose { src, .. }
        | Instr::Index { src, .. }
        | Instr::Slice { src, .. }
        | Instr::SparseAttr { src, .. } => f(*src),
        Instr::Dot { a, b, .. } | Instr::MatMul { a, b, .. } => {
            f(*a);
            f(*b);
        }
        Instr::Conv2d { input, filter, .. } => {
            f(*input);
            f(*filter);
        }
        Instr::Conv2dGradInput { dy, filter, .. } => {
            f(*dy);
            f(*filter);
        }
        Instr::Conv2dGradFilter { input, dy, .. } => {
            f(*input);
            f(*dy);
        }
        Instr::ReluGrad { grad, src, .. } => {
            f(*grad);
            f(*src);
        }
        Instr::Gather { src, indices, .. } => {
            f(*src);
            f(*indices);
        }
        Instr::Output(id) => f(*id),
        Instr::Call { args, .. } => {
            for &a in args {
                f(a);
            }
        }
        Instr::Return { value } => {
            if let Some(v) = value {
                f(*v);
            }
        }
        // A parameter / function definition / pure declaration reads no operands.
        Instr::Param { .. } | Instr::FnDef { .. } => {}
        #[cfg(feature = "std-surface")]
        Instr::ConstArray { .. } => {}
        #[cfg(feature = "std-surface")]
        Instr::ArrayLoad { base, index, .. } => {
            f(*base);
            f(*index);
        }
        // `init_ids` are the pre-loop (enclosing-scope) values threaded into the
        // header; cond/live_vars/exit_ids/body live in the loop's own namespace.
        #[cfg(feature = "std-surface")]
        Instr::While { init_ids, .. } => {
            for &v in init_ids {
                f(v);
            }
        }
        // `then_val`/`else_val` of each merge may forward an enclosing-scope value.
        #[cfg(feature = "std-surface")]
        Instr::If { merges, .. } => {
            for (_merge, then_val, else_val) in merges {
                f(*then_val);
                f(*else_val);
            }
        }
        #[cfg(feature = "std-surface")]
        Instr::VecLoad { base, offset, .. } | Instr::VecLoadI32 { base, offset, .. } => {
            f(*base);
            f(*offset);
        }
        #[cfg(feature = "std-surface")]
        Instr::VecFma { a, b, acc, .. } | Instr::VecMulAddQ16 { a, b, acc, .. } => {
            f(*a);
            f(*b);
            f(*acc);
        }
        #[cfg(feature = "std-surface")]
        Instr::VecReduceAdd { src, .. } | Instr::VecReduceAddI64 { src, .. } => f(*src),
        #[cfg(feature = "std-surface")]
        Instr::VecStore {
            src, base, offset, ..
        } => {
            f(*src);
            f(*base);
            f(*offset);
        }
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl { .. } => {}
        // result/enter/exit/alloc ids are produced inside the region body's own
        // SSA namespace; nothing is read from the enclosing scope.
        #[cfg(feature = "std-surface")]
        Instr::Region { .. } => {}
        // break/continue READ their `live` snapshot values (forwarded as loop
        // block-args), so they are genuine operands — DCE must keep them alive.
        #[cfg(feature = "std-surface")]
        Instr::Break { live } | Instr::Continue { live } => {
            for (_, v) in live {
                f(*v);
            }
        }
    }
}

pub(crate) fn instruction_operands(instr: &Instr) -> Vec<ValueId> {
    let mut ops = Vec::new();
    for_each_operand(instr, |op| ops.push(op));
    ops
}
