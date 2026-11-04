// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Preserve integer address additions as byte offsets on an LLVM pointer.

use std::collections::{BTreeMap, BTreeSet};

use super::{BinOp, Instr, LoweringContext, ValueId, ValueKind};

#[derive(Clone, Default)]
pub(super) struct MemoryAddresses {
    adds: BTreeMap<ValueId, (ValueId, ValueId)>,
    constants: BTreeSet<ValueId>,
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
        self.constants.retain(|id| !invalid.contains(id));
    }
}

impl LoweringContext {
    pub(super) fn record_memory_address(&mut self, instr: &Instr) {
        match instr {
            Instr::ConstI64(id, _) => {
                self.memory_addresses.constants.insert(*id);
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
            }
            _ => {}
        }
    }

    pub(super) fn emit_memory_address(&mut self, mut address: ValueId, name: &str) {
        let mut offsets = Vec::new();
        // Follow the base side of each add, preferring a nonconstant operand.
        // Iteration keeps arbitrarily long source add chains off the call stack.
        while let Some(&(lhs, rhs)) = self.memory_addresses.adds.get(&address) {
            let (base, offset) = if self.memory_addresses.constants.contains(&lhs)
                && !self.memory_addresses.constants.contains(&rhs)
            {
                (rhs, lhs)
            } else {
                (lhs, rhs)
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
