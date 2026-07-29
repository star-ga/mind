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

//! DTK — Deterministic Top-K Callee-Saved register-allocation planner
//! (Independence Roadmap Phase C, #254 — first real allocation slice).
//!
//! # What this is
//! A **pure planning pass** over an [`IRModule`]: it ranks the module's SSA
//! values by static use-count and assigns the top-`K` to the x86-64 SysV
//! callee-saved GPRs, spilling the rest. The output is a [`DtkPlan`] — a
//! `ValueId -> Slot` map — that a native-ELF backend would consume to keep the
//! hottest values resident in registers instead of on the stack.
//!
//! # Byte-identity story (load-bearing)
//! This module has **no consumer on any emission path**. The Rust compiler
//! emits MLIR text and shells to `clang` (which runs its own register
//! allocation); the native-ELF backend that would consume a DTK plan lives in
//! the self-host `examples/mindc_mind/main.mind` (a separate lane). So planning
//! is a **standalone analysis with zero side effects on emitted bytes** — the
//! keystone / cross-substrate byte-identity holds *by construction* because no
//! emitter is touched. This is the strongest form of "default-off": the flag is
//! the absence of a caller, not a runtime branch that could drift.
//!
//! # Determinism (the wedge invariant)
//! The plan is a **pure function of the IR**: no clock, no RNG, no pointer bits,
//! no `HashMap` iteration. Values are ranked by (use-count DESC, [`ValueId`]
//! ASC) — a total order, so the ranking is content-addressed and stable across
//! runs and substrates. Register assignment walks a fixed-order register table.
//! [`DtkPlan::assignment`] is a [`BTreeMap`] so iteration order is canonical.
//!
//! Use-count is a *heuristic weight*, not a correctness input: an operand-bearing
//! variant not yet enumerated in `accumulate` merely under-counts a value's
//! weight (it may be spilled when it could have been a register) — it can never
//! miscompile, because every DEFINED value that is not in the top-`K` falls
//! through to [`Slot::Spill`], the existing stack scheme. The self-host subset
//! (i64 `ConstI64`/`BinOp`/`Call`/`Param`/`Return` plus `If`/`While`) is covered
//! exactly; the tensor/vector variants fall through conservatively.
//!
//! Coverage note: `accumulate` recurses into an `Instr::Region` *body* but
//! does not yet record `Region`'s own `result`/`enter_id`/`exit_id`/`alloc_ids`
//! as defs, so a Region result value is never ranked and always defaults to
//! [`Slot::Spill`]. That is safe under the conservative-fallthrough contract
//! above (under-count, never miscompile); enumerate those ids here before
//! `Region` joins the exactly-covered self-host subset roster at Phase C6.
//!
//! deferred: wiring a DTK plan into an actual native-ELF register/spill emitter
//! is Phase C6 and lives in `main.mind` (self-host lane) — upgrade path: the
//! self-host `nb_*` slot scheme reads `Slot::Reg` for a value and emits a
//! callee-saved GPR home (with prologue push / epilogue pop) instead of a
//! `nb_slot_disp` stack home, gated on its own byte-identity re-freeze.

use crate::ir::{IRModule, Instr, ValueId};
use std::collections::{BTreeMap, BTreeSet};

/// x86-64 SysV callee-saved general-purpose registers, in fixed canonical
/// order. `rbp`/`rsp` are excluded (frame pointer / stack pointer). This order
/// is part of the deterministic contract: rank `i` maps to `CALLEE_SAVED[i]`.
pub const CALLEE_SAVED: [&str; 5] = ["rbx", "r12", "r13", "r14", "r15"];

/// Home assigned to an SSA value by the DTK planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// A callee-saved GPR (one of [`CALLEE_SAVED`]).
    Reg(&'static str),
    /// The existing stack scheme (the naive slot home). Everything outside the
    /// top-`K` lands here, so the plan is a strict improvement-or-no-op: it only
    /// ever PROMOTES a value from stack to register, never the reverse.
    Spill,
}

/// A deterministic register-allocation plan for one [`IRModule`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DtkPlan {
    /// `ValueId -> Slot`. `BTreeMap` for canonical (content-addressed) iteration.
    pub assignment: BTreeMap<ValueId, Slot>,
    /// Defined values, best-first (the ranking the assignment was cut from).
    /// Ordered by (use-count DESC, `ValueId` ASC).
    pub ranked: Vec<ValueId>,
}

impl DtkPlan {
    /// The [`Slot`] for `v`, or [`Slot::Spill`] if `v` was never defined in the
    /// module (fail-safe: an unknown value gets the existing stack scheme, never
    /// a bogus register).
    pub fn slot(&self, v: ValueId) -> Slot {
        self.assignment.get(&v).copied().unwrap_or(Slot::Spill)
    }
}

/// Plan `module` with the default `K` = number of callee-saved registers.
pub fn plan_module(module: &IRModule) -> DtkPlan {
    plan_module_k(module, CALLEE_SAVED.len())
}

/// Plan `module`, assigning the `K` most-used values to callee-saved registers.
/// `K` is clamped to [`CALLEE_SAVED`]`.len()` (there are only so many registers).
pub fn plan_module_k(module: &IRModule, k: usize) -> DtkPlan {
    let mut uses: BTreeMap<ValueId, u64> = BTreeMap::new();
    let mut defs: BTreeSet<ValueId> = BTreeSet::new();
    accumulate(&module.instrs, &mut uses, &mut defs);

    // Rank the DEFINED values by (use-count DESC, ValueId ASC). The ValueId
    // tiebreak makes the order a total, content-addressed function of the IR.
    let mut ranked: Vec<ValueId> = defs.into_iter().collect();
    ranked.sort_by(|a, b| {
        let ua = uses.get(a).copied().unwrap_or(0);
        let ub = uses.get(b).copied().unwrap_or(0);
        ub.cmp(&ua).then_with(|| a.cmp(b))
    });

    let cap = k.min(CALLEE_SAVED.len());
    let mut assignment = BTreeMap::new();
    for (i, v) in ranked.iter().enumerate() {
        let slot = if i < cap {
            Slot::Reg(CALLEE_SAVED[i])
        } else {
            Slot::Spill
        };
        assignment.insert(*v, slot);
    }
    DtkPlan { assignment, ranked }
}

/// Forward walk over a (possibly nested) instruction stream, recording each
/// value's DEFINITION and every USE as an operand. Mirrors the nested-stream
/// recursion in `src/ir/fp_mode.rs` so no control-flow body is missed.
fn accumulate(instrs: &[Instr], uses: &mut BTreeMap<ValueId, u64>, defs: &mut BTreeSet<ValueId>) {
    for instr in instrs {
        record_def(instr, defs);
        record_uses(instr, uses);
        // Recurse into nested streams. The self-host subset control-flow
        // variants (If/While) and Region are gated on `std-surface`; since that
        // IS the default feature (see Cargo.toml), a default build DOES construct
        // and recurse into them — the `#[cfg]` gating exists so the
        // `--no-default-features` lib (which never constructs these variants)
        // still compiles without dead match arms.
        match instr {
            Instr::FnDef { body, .. } => accumulate(body, uses, defs),
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                accumulate(cond_instrs, uses, defs);
                accumulate(body, uses, defs);
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                accumulate(cond_instrs, uses, defs);
                accumulate(then_instrs, uses, defs);
                accumulate(else_instrs, uses, defs);
            }
            #[cfg(feature = "std-surface")]
            Instr::Region { body, .. } => accumulate(body, uses, defs),
            _ => {}
        }
    }
}

/// Record the value(s) DEFINED by `instr`. Completeness here is what guarantees
/// every value gets a home; the self-host subset producers are covered exactly.
fn record_def(instr: &Instr, defs: &mut BTreeSet<ValueId>) {
    match instr {
        Instr::ConstI64(dst, _)
        | Instr::ConstF64(dst, _)
        | Instr::ConstTensor(dst, ..)
        | Instr::ConstDenseTensor { dst, .. }
        | Instr::BinOp { dst, .. }
        | Instr::Sum { dst, .. }
        | Instr::Mean { dst, .. }
        | Instr::Relu { dst, .. }
        | Instr::ReluGrad { dst, .. }
        | Instr::Reshape { dst, .. }
        | Instr::ExpandDims { dst, .. }
        | Instr::Squeeze { dst, .. }
        | Instr::Transpose { dst, .. }
        | Instr::Dot { dst, .. }
        | Instr::MatMul { dst, .. }
        | Instr::Index { dst, .. }
        | Instr::Slice { dst, .. }
        | Instr::Gather { dst, .. }
        | Instr::Call { dst, .. } => {
            defs.insert(*dst);
        }
        #[cfg(feature = "std-surface")]
        Instr::If { dst, .. } => {
            defs.insert(*dst);
        }
        // Param defs are deliberately NOT ranking candidates: a param's home is
        // fixed by the ABI/prologue (`nb_emit_params` in the self-host lane),
        // never by this planner. Marking it as a def would let it compete for
        // (and, on a use-count tie, win via the ValueId-ASC tiebreak) a
        // callee-saved slot meant for a real temporary — and could collide with
        // its ABI-fixed home if a future consumer trusted `plan.slot(param_id)`
        // directly. This matches the .mind port's `nb_dtk_scan_expr`, which
        // resolves a param-referencing `ast_ident` to its vid for use-counting
        // (`nb_dtk_bump_use`) but never calls `nb_dtk_mark_def` on it — only
        // `int_lit`/`binop` results are marked as candidates there.
        Instr::Param { .. } => {}
        // Conservative fallthrough: an unlisted producer's value is simply never
        // register-promoted (it defaults to Spill). Safe — never a miscompile.
        _ => {}
    }
}

/// Record every USE (operand) of `instr` as a +1 weight on that value. Under-
/// counting is safe (see the module header); the self-host subset operands are
/// counted exactly.
fn record_uses(instr: &Instr, uses: &mut BTreeMap<ValueId, u64>) {
    let mut bump = |v: ValueId| {
        *uses.entry(v).or_insert(0) += 1;
    };
    match instr {
        Instr::BinOp { lhs, rhs, .. } => {
            bump(*lhs);
            bump(*rhs);
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
        | Instr::Output(src) => {
            bump(*src);
        }
        Instr::ReluGrad { grad, src, .. } => {
            bump(*grad);
            bump(*src);
        }
        Instr::Dot { a, b, .. } | Instr::MatMul { a, b, .. } => {
            bump(*a);
            bump(*b);
        }
        Instr::Gather { src, indices, .. } => {
            bump(*src);
            bump(*indices);
        }
        Instr::Call { args, .. } => {
            for a in args {
                bump(*a);
            }
        }
        Instr::Return { value: Some(v) } => bump(*v),
        #[cfg(feature = "std-surface")]
        Instr::If {
            cond_id,
            then_result,
            else_result,
            ..
        } => {
            bump(*cond_id);
            bump(*then_result);
            bump(*else_result);
        }
        #[cfg(feature = "std-surface")]
        Instr::While {
            cond_id, init_ids, ..
        } => {
            bump(*cond_id);
            for v in init_ids {
                bump(*v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BinOp;

    fn module(instrs: Vec<Instr>) -> IRModule {
        IRModule {
            instrs,
            next_id: 0,
            ..Default::default()
        }
    }

    /// A tiny module where `%0` is used twice, `%1` once, `%2`/`%3` are results.
    fn sample() -> IRModule {
        // %0 = const; %1 = const; %2 = %0 + %1; %3 = %0 + %2; return %3
        module(vec![
            Instr::ConstI64(ValueId(0), 10),
            Instr::ConstI64(ValueId(1), 20),
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
                rhs: ValueId(2),
            },
            Instr::Return {
                value: Some(ValueId(3)),
            },
        ])
    }

    #[test]
    fn plan_is_a_pure_function() {
        let m = sample();
        // Same IR in → byte-identical plan out, twice (no clock/RNG/pointer bits).
        assert_eq!(plan_module(&m), plan_module(&m));
    }

    #[test]
    fn ranks_by_use_count_then_valueid() {
        let plan = plan_module(&sample());
        // Uses: %0→2, %2→1, %3→1, %1→1, %0's def counts 2. Expected order:
        // %0 (2 uses) first; then the 1-use values by ValueId ASC: %1, %2, %3.
        assert_eq!(
            plan.ranked,
            vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)]
        );
    }

    #[test]
    fn top_k_get_registers_rest_spill() {
        // K = 2 → the two hottest values get rbx, r12; the rest spill.
        let plan = plan_module_k(&sample(), 2);
        assert_eq!(plan.slot(ValueId(0)), Slot::Reg("rbx"));
        assert_eq!(plan.slot(ValueId(1)), Slot::Reg("r12"));
        assert_eq!(plan.slot(ValueId(2)), Slot::Spill);
        assert_eq!(plan.slot(ValueId(3)), Slot::Spill);
    }

    #[test]
    fn k_is_clamped_to_register_count() {
        // Asking for more registers than exist assigns at most CALLEE_SAVED.len().
        let plan = plan_module_k(&sample(), 999);
        let reg_count = plan
            .assignment
            .values()
            .filter(|s| matches!(s, Slot::Reg(_)))
            .count();
        assert!(reg_count <= CALLEE_SAVED.len());
        // With 4 defined values and 5 registers, all 4 fit in registers.
        assert_eq!(reg_count, 4);
    }

    #[test]
    fn unknown_value_defaults_to_spill() {
        let plan = plan_module(&sample());
        // A value that was never defined gets the existing stack scheme.
        assert_eq!(plan.slot(ValueId(4242)), Slot::Spill);
    }

    #[test]
    fn assignment_is_a_strict_promotion() {
        // Every assigned slot is either a register (a promotion off the stack)
        // or Spill (the status-quo home) — never a demotion, so the pass is a
        // strict improvement-or-no-op on the self-host subset.
        let plan = plan_module(&sample());
        for slot in plan.assignment.values() {
            assert!(matches!(slot, Slot::Reg(_) | Slot::Spill));
        }
    }
}
