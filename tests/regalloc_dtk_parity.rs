// DTK slice 1 (#254) — parity consumer for src/opt/regalloc_dtk.rs.
//
// This test CURES the reference planner's dormancy: `plan_module_k` had no
// consumer on any path (its own module header notes this), so nothing exercised
// its (use-count DESC, ValueId ASC) ranking outside its unit tests. Here we run
// it over the fixture corpus the pure-MIND native-ELF port must match and pin the
// slice-1 contract: the top-K=2 NON-PARAM defs get callee-saved homes rbx (rank
// 0) then r12 (rank 1); everything else spills.
//
// Cross-implementation parity (Rust plan_module_k  ==  the pure-MIND
// `selftest_dtk_plan` export in examples/mindc_mind/main.mind) is asserted by the
// companion `dtk_plan_parity_smoke.py`, which loads the built .so and diffs its
// serialized [count | (id, reg_code)*count] table against `expected_slice1`
// below. That smoke needs the cdylib built (U1's gate), so it is out of this
// front-end-only test; the expectations it consumes live here as the single
// source of truth.
//
// SLICE 1 candidate rule (matches the .mind scan): params are EXCLUDED from
// candidates (they are homed on the stack by nb_emit_params this slice), so we
// filter Param defs out of `DtkPlan.ranked` before cutting the top-K.

use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use libmind::opt::regalloc_dtk::{plan_module_k, DtkPlan, Slot, CALLEE_SAVED};

/// The reg_code the pure-MIND side serializes: 0 = rbx, 1 = r12.
fn reg_code(reg: &str) -> i64 {
    CALLEE_SAVED.iter().position(|r| *r == reg).unwrap() as i64
}

/// Param SSA ids of a single-fn module (ids the slice-1 candidate set excludes).
fn param_ids(m: &IRModule) -> Vec<ValueId> {
    m.instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Param { dst, .. } => Some(*dst),
            _ => None,
        })
        .collect()
}

/// Slice-1 plan as the pure-MIND export emits it: [(id, reg_code)], sorted by
/// rank (rbx first, r12 second), params filtered out, K = 2.
fn expected_slice1(m: &IRModule) -> Vec<(i64, i64)> {
    let plan: DtkPlan = plan_module_k(m, CALLEE_SAVED.len());
    let params = param_ids(m);
    let ranked_nonparam: Vec<ValueId> = plan
        .ranked
        .into_iter()
        .filter(|v| !params.contains(v))
        .collect();
    ranked_nonparam
        .iter()
        .take(2)
        .enumerate()
        .map(|(rank, v)| (v.0 as i64, rank as i64))
        .collect()
}

/// `fn add(a, b) -> i64 { a + b }` — the frozen 498-byte native oracle fixture.
/// IR: Param0, Param1, BinOp(2)=a+b, Return(2). Non-param defs = {2}, used once
/// (by Return). So the slice-1 plan is a single home: id 2 -> rbx.
fn add_module() -> IRModule {
    IRModule {
        instrs: vec![
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
            Instr::Return {
                value: Some(ValueId(2)),
            },
        ],
        next_id: 3,
        ..Default::default()
    }
}

/// Two non-param temps, one hotter than the other, so both regs get used and the
/// (use-count DESC, id ASC) order is exercised:
/// %2 = a+b ; %3 = %2 + %2 ; %4 = %2 + %3 ; return %4
/// uses: %2 -> 3, %3 -> 1, %4 -> 1. Non-param ranking = [2, 3, 4]; top-2 -> rbx,r12.
fn hot_module() -> IRModule {
    IRModule {
        instrs: vec![
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
                lhs: ValueId(2),
                rhs: ValueId(2),
            },
            Instr::BinOp {
                dst: ValueId(4),
                op: BinOp::Add,
                lhs: ValueId(2),
                rhs: ValueId(3),
            },
            Instr::Return {
                value: Some(ValueId(4)),
            },
        ],
        next_id: 5,
        ..Default::default()
    }
}

#[test]
fn dtk_reference_planner_is_not_dormant() {
    // Merely constructing a plan exercises plan_module_k on a real module.
    let plan = plan_module_k(&add_module(), CALLEE_SAVED.len());
    assert!(!plan.ranked.is_empty());
}

#[test]
fn add_fixture_slice1_plan() {
    // id 2 (a+b) is the only non-param def -> rank 0 -> rbx (reg_code 0).
    assert_eq!(expected_slice1(&add_module()), vec![(2, 0)]);
    // Sanity against the raw plan: id 2 is a register, params/results by rule.
    let plan = plan_module_k(&add_module(), CALLEE_SAVED.len());
    assert_eq!(plan.slot(ValueId(2)), Slot::Reg("rbx"));
}

#[test]
fn tiebreak_use_count_then_valueid() {
    // %2 (3 uses) is rank 0 -> rbx; then %3 and %4 tie at 1 use, id ASC -> %3
    // rank 1 -> r12. %4 spills. This pins the exact regalloc_dtk tie-break the
    // .mind scan must reproduce.
    assert_eq!(expected_slice1(&hot_module()), vec![(2, 0), (3, 1)]);
    assert_eq!(reg_code("rbx"), 0);
    assert_eq!(reg_code("r12"), 1);
}
