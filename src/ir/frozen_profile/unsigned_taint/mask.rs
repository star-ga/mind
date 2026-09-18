// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Values provably in `[0, 2^63)` for the unsigned taint (`Taint::Values::masked`): bit 63
//! clear, so signed and unsigned operators agree on them. Per function BODY — ValueIds
//! restart in every function. Split from `unsigned_taint.rs` for the module-size budget.

use crate::ir::{Instr, ValueId};

/// Every `ConstI64` DEFINED in `instrs`, id -> value (through `if`/`while` bodies, never
/// into a nested `FnDef`, whose ids are a different namespace).
pub(super) fn collect_consts(instrs: &[Instr], out: &mut std::collections::BTreeMap<ValueId, i64>) {
    for instr in instrs {
        match instr {
            Instr::ConstI64(id, v) => {
                out.insert(*id, *v);
            }
            Instr::FnDef { .. } => {}
            other => {
                for nested in crate::ir::instr_bodies(other) {
                    collect_consts(nested, out);
                }
            }
        }
    }
}

/// Every loop-carried variable's `init_id` in `instrs` (any depth, not across a `FnDef`).
///
/// Inside a `while` the init id is the NAME of the header block argument
/// (`Instr::While::init_ids`): from the second iteration on it holds the post-body value, so
/// it is not the fixed value its defining instruction suggests. Such an id is never
/// non-negative — neither as a mask nor as a constant (audit 2026-09-16: `let m = a & 255`
/// then `m = m - 256` in the body made `m / 2` native 0 vs MLIR `divui` 255; `let c = 255`
/// then `c = -1` did the same through `a & c`). Pre-loop uses of the same id are poisoned
/// too; that only refuses compares MLIR emits unsigned and divisions main refused.
pub(super) fn collect_loop_inits(instrs: &[Instr], out: &mut std::collections::BTreeSet<ValueId>) {
    for instr in instrs {
        match instr {
            Instr::While {
                cond_instrs,
                body,
                init_ids,
                ..
            } => {
                out.extend(init_ids.iter().copied());
                collect_loop_inits(cond_instrs, out);
                collect_loop_inits(body, out);
            }
            Instr::FnDef { .. } => {}
            other => {
                for nested in crate::ir::instr_bodies(other) {
                    collect_loop_inits(nested, out);
                }
            }
        }
    }
}

/// One pass marking values provably in `[0, 2^63)`: `x & y` where either side is a
/// non-negative constant of this body or already masked (AND with a value whose bit 63 is
/// clear clears it), `x | y` / `x ^ y` where BOTH sides are, and an `if` join whose
/// incoming values are all non-negative. Returns
/// whether the set grew. Everything else — including arithmetic on a masked value — is
/// NOT masked (audit 2026-09-16: `(a & 255) - 256` is negative), and no loop init id
/// (`poisoned`, see [`collect_loop_inits`]) is ever non-negative.
pub(super) fn mark_masked(
    instrs: &[Instr],
    nonneg: &std::collections::BTreeSet<ValueId>,
    poisoned: &std::collections::BTreeSet<ValueId>,
    masked: &mut std::collections::BTreeSet<ValueId>,
) -> bool {
    let mut grew = false;
    for instr in instrs {
        let is_nonneg = |masked: &std::collections::BTreeSet<ValueId>, id: &ValueId| {
            !poisoned.contains(id) && (nonneg.contains(id) || masked.contains(id))
        };
        match instr {
            Instr::BinOp {
                dst,
                op: crate::ir::BinOp::BitAnd,
                lhs,
                rhs,
            } if is_nonneg(masked, lhs) || is_nonneg(masked, rhs) => {
                grew |= mark(poisoned, masked, dst);
            }
            // `|` / `^` keep bit 63 clear only when BOTH sides have it clear.
            Instr::BinOp {
                dst,
                op: crate::ir::BinOp::BitOr | crate::ir::BinOp::BitXor,
                lhs,
                rhs,
            } if is_nonneg(masked, lhs) && is_nonneg(masked, rhs) => {
                grew |= mark(poisoned, masked, dst);
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
                    grew |= mark_masked(nested, nonneg, poisoned, masked);
                }
                if is_nonneg(masked, then_result) && is_nonneg(masked, else_result) {
                    grew |= mark(poisoned, masked, dst);
                }
                for (merge, then_val, else_val) in merges {
                    if is_nonneg(masked, then_val) && is_nonneg(masked, else_val) {
                        grew |= mark(poisoned, masked, merge);
                    }
                }
            }
            Instr::FnDef { .. } => {}
            other => {
                for nested in crate::ir::instr_bodies(other) {
                    grew |= mark_masked(nested, nonneg, poisoned, masked);
                }
            }
        }
    }
    grew
}

/// Insert `id` into `masked` unless it is a loop init id; returns whether the set grew.
fn mark(
    poisoned: &std::collections::BTreeSet<ValueId>,
    masked: &mut std::collections::BTreeSet<ValueId>,
    id: &ValueId,
) -> bool {
    !poisoned.contains(id) && masked.insert(*id)
}
