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

use crate::ir::{instruction_dst, BinOp, IRModule, Instr, ValueId};

/// Canonicalize the public MIND IR in-place.
///
/// The pass is intentionally conservative: it keeps the existing SSA IDs,
/// performs deterministic cleanups (operand ordering, trivial constant folding),
/// and prunes provably dead instructions. Running the pass repeatedly is
/// idempotent.
pub fn canonicalize_module(module: &mut IRModule) {
    let mut instrs = prune_dead(&module.instrs);
    reorder_commutative_ops(&mut instrs);
    constant_fold(&mut instrs);
    instrs = prune_dead(&instrs);

    module.instrs = instrs;
    module.next_id = next_sequential_id(module);
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
                if dst.map_or(true, |id| used.contains(&id)) {
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

fn constant_fold(instrs: &mut Vec<Instr>) {
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
        | Instr::Slice { src, .. } => vec![*src],
        Instr::Dot { a, b, .. } | Instr::MatMul { a, b, .. } => vec![*a, *b],
        Instr::Conv2d { input, filter, .. } => vec![*input, *filter],
        Instr::Gather { src, indices, .. } => vec![*src, *indices],
        Instr::Output(id) => vec![*id],
    }
}
