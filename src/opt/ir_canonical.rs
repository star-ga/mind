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

    let mut instrs = prune_dead(&module.instrs);
    reorder_commutative_ops(&mut instrs);
    constant_fold(&mut instrs);
    instrs = prune_dead(&instrs);

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

fn prune_dead(instrs: &[Instr]) -> Vec<Instr> {
    let mut used: BTreeSet<ValueId> = BTreeSet::new();
    for instr in instrs.iter().rev() {
        match instr {
            Instr::Output(id) => {
                used.insert(*id);
            }
            other => {
                let dst = instruction_dst(other);
                // Keep instructions whose destination is either unused (None) or present
                // in the `used` set. Clippy prefers `is_none_or` over `map_or(true, ...)`.
                if dst.is_none_or(|id| used.contains(&id)) {
                    for operand in instruction_operands(other) {
                        used.insert(operand);
                    }
                }
            }
        }
    }

    let mut pruned = Vec::with_capacity(instrs.len());
    for instr in instrs {
        if let Some(dst) = instruction_dst(instr) {
            if !used.contains(&dst) {
                continue;
            }
        }
        pruned.push(instr.clone());
    }
    pruned
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

fn instruction_operands(instr: &Instr) -> Vec<ValueId> {
    match instr {
        Instr::ConstI64(_, _) | Instr::ConstTensor(_, _, _, _) => Vec::new(),
        Instr::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        Instr::Sum { src, .. }
        | Instr::Mean { src, .. }
        | Instr::Reshape { src, .. }
        | Instr::ExpandDims { src, .. }
        | Instr::Squeeze { src, .. }
        | Instr::Transpose { src, .. }
        | Instr::Index { src, .. }
        | Instr::Slice { src, .. }
        | Instr::SparseAttr { src, .. } => vec![*src],
        Instr::Dot { a, b, .. } | Instr::MatMul { a, b, .. } => vec![*a, *b],
        Instr::Conv2d { input, filter, .. } => vec![*input, *filter],
        Instr::Conv2dGradInput { dy, filter, .. } => vec![*dy, *filter],
        Instr::Conv2dGradFilter { input, dy, .. } => vec![*input, *dy],
        Instr::Gather { src, indices, .. } => vec![*src, *indices],
        Instr::Output(id) => vec![*id],
        _ => vec![],
    }
}
