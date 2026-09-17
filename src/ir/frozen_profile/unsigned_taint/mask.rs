// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! Values provably in `[0, 2^63)` for the unsigned taint (`Taint::Values::masked`): bit 63
//! clear, so signed and unsigned operators agree on them. Per function BODY — ValueIds
//! restart in every function. Split from `unsigned_taint.rs` for the module-size budget.

use crate::ir::{Instr, ValueId};

/// Non-negative `ConstI64` ids DEFINED in `instrs` (through `if`/`while` bodies, never
/// into a nested `FnDef`, whose ids are a different namespace).
pub(super) fn collect_nonneg_consts(
    instrs: &[Instr],
    out: &mut std::collections::BTreeSet<ValueId>,
) {
    for instr in instrs {
        match instr {
            Instr::ConstI64(id, v) if *v >= 0 => {
                out.insert(*id);
            }
            Instr::FnDef { .. } => {}
            other => {
                for nested in crate::ir::instr_bodies(other) {
                    collect_nonneg_consts(nested, out);
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
/// NOT masked (audit 2026-09-16: `(a & 255) - 256` is negative).
pub(super) fn mark_masked(
    instrs: &[Instr],
    nonneg: &std::collections::BTreeSet<ValueId>,
    masked: &mut std::collections::BTreeSet<ValueId>,
) -> bool {
    let mut grew = false;
    for instr in instrs {
        let is_nonneg = |masked: &std::collections::BTreeSet<ValueId>, id: &ValueId| {
            nonneg.contains(id) || masked.contains(id)
        };
        match instr {
            Instr::BinOp {
                dst,
                op: crate::ir::BinOp::BitAnd,
                lhs,
                rhs,
            } if is_nonneg(masked, lhs) || is_nonneg(masked, rhs) => {
                grew |= masked.insert(*dst);
            }
            // `|` / `^` keep bit 63 clear only when BOTH sides have it clear.
            Instr::BinOp {
                dst,
                op: crate::ir::BinOp::BitOr | crate::ir::BinOp::BitXor,
                lhs,
                rhs,
            } if is_nonneg(masked, lhs) && is_nonneg(masked, rhs) => {
                grew |= masked.insert(*dst);
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
                    grew |= mark_masked(nested, nonneg, masked);
                }
                if is_nonneg(masked, then_result) && is_nonneg(masked, else_result) {
                    grew |= masked.insert(*dst);
                }
                for (merge, then_val, else_val) in merges {
                    if is_nonneg(masked, then_val) && is_nonneg(masked, else_val) {
                        grew |= masked.insert(*merge);
                    }
                }
            }
            Instr::FnDef { .. } => {}
            other => {
                for nested in crate::ir::instr_bodies(other) {
                    grew |= mark_masked(nested, nonneg, masked);
                }
            }
        }
    }
    grew
}
