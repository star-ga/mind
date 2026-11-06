// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Preserve integer address additions as byte offsets on an LLVM pointer.
//!
//! A memory address is an `i64` sum. Rewriting it as `inttoptr` plus byte
//! `getelementptr`s keeps address strides visible to the optimizer, but only
//! when the byte offset is provably free of pointer provenance: promoting a
//! value that may itself be an allocation base to GEP *index* place, or
//! promoting a bare integer offset to GEP *root*, both invent an aliasing
//! relationship the source never had. When neither addend is provably
//! provenance-free the sum stays an opaque integer address, exactly as a
//! plain `inttoptr` of the whole sum.

use std::collections::{BTreeMap, BTreeSet};

use super::{BinOp, Instr, LoweringContext, ValueId, ValueKind};

#[derive(Clone, Default)]
pub(super) struct MemoryAddresses {
    adds: BTreeMap<ValueId, (ValueId, ValueId)>,
    /// Values that are provably provenance-free: a constant, a comparison, or
    /// an integer arithmetic result of provenance-free operands. Such a value
    /// is safe to add to a pointer as a byte offset.
    offsets: BTreeSet<ValueId>,
}

impl MemoryAddresses {
    /// A loop block argument replaces its initializer. Expressions computed
    /// from that initializer outside the loop must also remain opaque: expanding
    /// them would incorrectly recompute them from the current iteration's value.
    pub(super) fn forget(&mut self, carried: &[ValueId]) {
        let mut invalid: BTreeSet<_> = carried.iter().copied().collect();
        loop {
            let before = invalid.len();
            for (&dst, &(lhs, rhs)) in &self.adds {
                if invalid.contains(&lhs) || invalid.contains(&rhs) {
                    invalid.insert(dst);
                }
            }
            if invalid.len() == before {
                break;
            }
        }
        self.adds.retain(|id, _| !invalid.contains(id));
        self.offsets.retain(|id| !invalid.contains(id));
    }

    /// Extend the provenance-free set with the values defined by a straight-line
    /// instruction sequence. Used to prove a loop-carried variable immutable in
    /// provenance: its initializer and every body update are provenance-free.
    /// Nested control flow is deliberately not traversed, so a value updated
    /// under a branch stays unknown (conservative, hence sound).
    pub(super) fn offsets_after(&self, instrs: &[Instr]) -> BTreeSet<ValueId> {
        let mut offsets = self.offsets.clone();
        for instr in instrs {
            if let Instr::ConstI64(id, _) = instr {
                offsets.insert(*id);
            } else if let Instr::BinOp { dst, lhs, rhs, .. } = instr {
                if offsets.contains(lhs) && offsets.contains(rhs) {
                    offsets.insert(*dst);
                }
            }
        }
        offsets
    }

    pub(super) fn is_offset(&self, id: &ValueId) -> bool {
        self.offsets.contains(id)
    }

    /// Re-assert offset-ness for a loop-carried value after `forget`: the value
    /// is proven provenance-free by `offsets_after`, so `forget` dropping the
    /// stale initializer facts must not also drop that proof.
    pub(super) fn restore_offset(&mut self, id: ValueId) {
        self.offsets.insert(id);
    }
}

impl LoweringContext {
    /// A loop-carried variable is a proven byte offset when its pre-loop value and every body update
    /// are provenance-free (a constant induction counter, say). The cond/body sub-contexts re-assert
    /// that fact after `forget` drops the stale initializer facts, so an address `base + counter`
    /// still lowers to a byte GEP on `base` rather than an opaque sum. `carried` yields
    /// `(init_id, post_id)` per loop-carried value.
    pub(super) fn carried_offsets(
        &self,
        body: &[Instr],
        carried: impl Iterator<Item = (usize, usize)>,
    ) -> Vec<ValueId> {
        let body_offsets = self.memory_addresses.offsets_after(body);
        carried
            .filter(|&(init, post)| {
                self.memory_addresses.is_offset(&ValueId(init))
                    && body_offsets.contains(&ValueId(post))
            })
            .map(|(init, _)| ValueId(init))
            .collect()
    }

    pub(super) fn record_memory_address(&mut self, instr: &Instr) {
        match instr {
            Instr::ConstI64(id, _) => {
                self.memory_addresses.offsets.insert(*id);
            }
            Instr::BinOp {
                dst,
                op: BinOp::Add,
                lhs,
                rhs,
            } if [*lhs, *rhs].iter().all(|id| {
                matches!(
                    self.values.get(id),
                    Some(ValueKind::ScalarI64 | ValueKind::ScalarU64)
                ) && !self.i1_values.contains(id)
            }) =>
            {
                self.memory_addresses.adds.insert(*dst, (*lhs, *rhs));
                if self.memory_addresses.is_offset(lhs) && self.memory_addresses.is_offset(rhs) {
                    self.memory_addresses.offsets.insert(*dst);
                }
            }
            Instr::BinOp { dst, lhs, rhs, .. }
                if self.memory_addresses.is_offset(lhs) && self.memory_addresses.is_offset(rhs) =>
            {
                self.memory_addresses.offsets.insert(*dst);
            }
            _ => {}
        }
    }

    pub(super) fn emit_memory_address(&mut self, mut address: ValueId, name: &str) {
        let mut offsets = Vec::new();
        // Follow the root side of each add. The GEP root may be any value (it
        // is `inttoptr`'d anyway); the GEP *index* must be a proven byte
        // offset. An add with zero or two provenance-free addends is left
        // intact so both contributors stay in the integer address.
        // Iteration keeps arbitrarily long source add chains off the call stack.
        while let Some(&(lhs, rhs)) = self.memory_addresses.adds.get(&address) {
            let (base, offset) = match (
                self.memory_addresses.is_offset(&lhs),
                self.memory_addresses.is_offset(&rhs),
            ) {
                (true, false) => (rhs, lhs),
                (false, true) => (lhs, rhs),
                _ => break,
            };
            address = base;
            offsets.push(offset);
        }
        let mut ptr = if offsets.is_empty() {
            name.to_string()
        } else {
            format!("{name}_base")
        };
        self.emit_line(&format!(
            "    %{ptr} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            address.0
        ));
        for (i, offset) in offsets.iter().rev().enumerate() {
            let next = if i + 1 == offsets.len() {
                name.to_string()
            } else {
                format!("{name}_o{i}")
            };
            // No inbounds or no-wrap promise: integer addresses and signed byte
            // offsets retain the original modulo-i64 arithmetic and alignment.
            self.emit_line(&format!(
                "    %{next} = llvm.getelementptr %{ptr}[%{}] : \
                 (!llvm.ptr, i64) -> !llvm.ptr, i8",
                offset.0
            ));
            ptr = next;
        }
    }
}
