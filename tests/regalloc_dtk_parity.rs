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
// `selftest_dtk_plan` export in examples/mindc_mind/main.mind) IS gated
// end-to-end by the companion `examples/mindc_mind/testdata/dtk_plan_parity_smoke.py`:
// it builds the cdylib, runs `selftest_dtk_plan`, and diffs its serialized
// [count | (id, reg_code)*count] table against `plan_module_k` over the eligible
// corpus (field-exact), plus asserting count==0 for every ineligible class
// (call/if/while/float-param/narrow-param/div/shift/deref-assign; the entry-fn
// class is deferred — the export hardcodes is_entry=0, see the smoke's marker).
// The Rust oracle tables the smoke diffs against are produced live by the
// `dump_dtk_slice1_parity` test below (fallback: the pinned `expected_slice1`
// tables, Rust-gated by `parity_corpus_tables` so they cannot drift). This file
// also independently checks `plan_module_k` against those expectations.
//
// SLICE 1 candidate rule (matches the .mind scan): params are EXCLUDED from
// candidates (they are homed on the stack by nb_emit_params this slice), so we
// filter Param defs out of `DtkPlan.ranked` before cutting the top-K.

use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use libmind::opt::regalloc_dtk::{CALLEE_SAVED, DtkPlan, Slot, plan_module_k};

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

// ---------------------------------------------------------------------------
// PARITY-SMOKE ORACLE (dtk_plan_parity_smoke.py, audit finding #1).
//
// The pure-MIND `selftest_dtk_plan` export in examples/mindc_mind/main.mind
// scans SOURCE TEXT and serializes the slice-1 plan as [count | (id,reg)*count].
// The parity smoke feeds each SOURCE below to that export and diffs its table
// against this Rust `plan_module_k` result. To make that a real cross-impl gate,
// each source's corresponding IRModule is hand-built here with the SAME SSA
// numbering the .mind scan mirrors (params 0..n_params-1, then int-lits/binops in
// eval order) — but the RANKING/use-counting is computed INDEPENDENTLY by the
// Rust `plan_module_k` (record_uses + sort), never copied from the port. The
// `dump_dtk_slice1_parity` test emits these tables so the smoke can regenerate
// the oracle live (fallback: the pinned tables below, which the non-ignored
// tests in this file gate). SOURCE<->module correspondence is asserted faithful
// by construction and pinned by `parity_corpus_tables`.

/// `fn f(a: i64) -> i64 { let x: i64 = a + a; return x + x; }`
/// IR: Param0, BinOp1(a+a), BinOp2(%1+%1), Return2. Non-param defs {1,2};
/// uses %1 -> 2, %2 -> 1. Ranking [1,2]; top-2 -> (1,rbx),(2,r12).
fn twice_module() -> IRModule {
    IRModule {
        instrs: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".into(),
                index: 0,
            },
            Instr::BinOp {
                dst: ValueId(1),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(0),
            },
            Instr::BinOp {
                dst: ValueId(2),
                op: BinOp::Add,
                lhs: ValueId(1),
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

/// `fn f(a,b) -> i64 { let x = a+b; let y = a+b; return x+y; }`
/// IR: P0,P1,BinOp2(a+b),BinOp3(a+b),BinOp4(%2+%3),Return4. Non-param defs
/// {2,3,4}, all used once -> a pure (id-ASC) tie: top-2 (2,rbx),(3,r12).
fn tie_module() -> IRModule {
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
                lhs: ValueId(0),
                rhs: ValueId(1),
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

/// `fn f(a,b) -> i64 { let x = a + b; return a; }` — %2 (a+b) is a ZERO-USE def
/// (x is never read). It is still the only non-param candidate, so it ranks
/// rbx: a zero-use def is homed, not dropped. IR: P0,P1,BinOp2(a+b),Return0.
fn zero_use_module() -> IRModule {
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
                value: Some(ValueId(0)),
            },
        ],
        next_id: 3,
        ..Default::default()
    }
}

/// `fn f(a: i64) -> i64 { return a + a; }` — a single non-param def. IR:
/// P0, BinOp1(a+a), Return1. Non-param defs {1} -> (1,rbx).
fn single_def_module() -> IRModule {
    IRModule {
        instrs: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".into(),
                index: 0,
            },
            Instr::BinOp {
                dst: ValueId(1),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(0),
            },
            Instr::Return {
                value: Some(ValueId(1)),
            },
        ],
        next_id: 2,
        ..Default::default()
    }
}

/// The named ELIGIBLE parity corpus (name, module). The smoke feeds a SOURCE of
/// the same name to `selftest_dtk_plan` and diffs the serialized table against
/// `expected_slice1(module)`.
fn parity_corpus() -> Vec<(&'static str, IRModule)> {
    vec![
        ("add", add_module()),
        ("twice", twice_module()),
        ("tie", tie_module()),
        ("zero_use", zero_use_module()),
        ("single_def", single_def_module()),
    ]
}

#[test]
fn parity_corpus_tables() {
    // Pin every eligible corpus table (Rust plan_module_k). These are the exact
    // values dtk_plan_parity_smoke.py falls back to when cargo cannot regenerate
    // them live, so pinning them here keeps the fallback honest (Rust-gated).
    assert_eq!(expected_slice1(&add_module()), vec![(2, 0)]);
    assert_eq!(expected_slice1(&twice_module()), vec![(1, 0), (2, 1)]);
    assert_eq!(expected_slice1(&tie_module()), vec![(2, 0), (3, 1)]);
    assert_eq!(expected_slice1(&zero_use_module()), vec![(2, 0)]);
    assert_eq!(expected_slice1(&single_def_module()), vec![(1, 0)]);
}

/// Emit the eligible corpus tables for dtk_plan_parity_smoke.py to consume as a
/// LIVE Rust oracle. Run: `cargo test --test regalloc_dtk_parity
/// dump_dtk_slice1_parity -- --ignored --nocapture`. One line per fixture:
///   `DTK <name>: [(id,reg),(id,reg)]`
#[test]
#[ignore = "oracle dump for dtk_plan_parity_smoke.py; run with --ignored --nocapture"]
fn dump_dtk_slice1_parity() {
    for (name, m) in parity_corpus() {
        let table = expected_slice1(&m);
        let body: Vec<String> = table.iter().map(|(id, r)| format!("({id},{r})")).collect();
        println!("DTK {name}: [{}]", body.join(","));
    }
}
