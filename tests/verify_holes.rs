// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Part of the MIND project (Machine Intelligence Native Design).

//! Regression tests for two SSA-verification holes.
//!
//! FINDING #123 — `verify_module`'s `FnDef` body arm exposed a straight-line op's
//! own `dst` BEFORE checking its operands, so a self-referential / use-before-def
//! op (`%5 = add %5, %0`) and a duplicate-def inside a fn body both slipped past
//! the in-pipeline gate. The fix reorders the operand check before exposure and
//! adds a single-assignment guard. These tests assert those forms are now
//! REJECTED, plus a positive control.
//!
//! FINDING #9 — the mindc-verify consumer path (`check_ssa_well_formed` ->
//! `check_ssa_stream`) never validated a `While`/`If` region's `cond_id` nor an
//! `If`'s `then_result`/`else_result`, so a crafted mic@3 with a dangling
//! condition or branch-result passed `mindc verify`. The fix adds dedicated
//! definedness checks within the correct sub-scope (loop namespace / per-branch
//! scope). These tests assert dangling forms are REJECTED with zero false
//! positives on valid control flow, and (std-surface) a mic@3 round-trip of a
//! dangling-cond While fails `check_ssa_well_formed`.

use libmind::ir::{BinOp, IRModule, Instr, IrVerifyError, ValueId, verify_module};
// `check_ssa_well_formed` / `SsaRule` back the FINDING #9 (While/If) tests, which
// are std-surface-gated — import them only there so a `--no-default-features`
// build (where those tests compile out) stays unused-import-clean.
#[cfg(feature = "std-surface")]
use libmind::ir::{SsaRule, check_ssa_well_formed};

// ---------------------------------------------------------------------------
// FINDING #123 — FnDef body: self-ref, duplicate-def (verify_module).
// ---------------------------------------------------------------------------

/// `%51 = %51 + %50` inside a fn body references its OWN dst — a self-reference /
/// use-before-def. Must now be rejected by `verify_module`.
#[test]
fn fndef_body_self_reference_rejected() {
    let mut m = IRModule::new();
    let a = ValueId(50);
    let selfref = ValueId(51);
    m.instrs.push(Instr::FnDef {
        name: "self_ref".to_string(),
        params: vec![],
        ret_id: None,
        reap_threshold: None,
        body: vec![
            Instr::ConstI64(a, 1),
            Instr::BinOp {
                dst: selfref,
                op: BinOp::Add,
                lhs: selfref, // reads its own (not-yet-defined) result
                rhs: a,
            },
        ],
    });
    // Module needs an Output to be well-formed.
    let out = m.fresh();
    m.instrs.push(Instr::ConstI64(out, 0));
    m.instrs.push(Instr::Output(out));

    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::UseBeforeDefinition { value, .. } if value == selfref),
        "self-referential body op must be rejected as use-before-def, got {err:?}"
    );
}

/// Two body instructions define `%60` — a single-assignment violation that the
/// old exposure path silently swallowed. Must now be a `DuplicateDefinition`.
#[test]
fn fndef_body_duplicate_def_rejected() {
    let mut m = IRModule::new();
    let dup = ValueId(60);
    m.instrs.push(Instr::FnDef {
        name: "dup_def".to_string(),
        params: vec![],
        ret_id: None,
        reap_threshold: None,
        body: vec![Instr::ConstI64(dup, 1), Instr::ConstI64(dup, 2)],
    });
    let out = m.fresh();
    m.instrs.push(Instr::ConstI64(out, 0));
    m.instrs.push(Instr::Output(out));

    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::DuplicateDefinition(v) if v == dup),
        "duplicate body definition must be rejected, got {err:?}"
    );
}

/// Positive control: a straight-line fn body with a legitimate forward chain
/// (`%70 = 1`, `%71 = 2`, `%72 = %70 + %71`) must still pass after the reorder.
#[test]
fn fndef_body_forward_chain_still_passes() {
    let mut m = IRModule::new();
    let (b0, b1, b2) = (ValueId(70), ValueId(71), ValueId(72));
    m.instrs.push(Instr::FnDef {
        name: "ok_fn".to_string(),
        params: vec![],
        ret_id: None,
        reap_threshold: None,
        body: vec![
            Instr::ConstI64(b0, 1),
            Instr::ConstI64(b1, 2),
            Instr::BinOp {
                dst: b2,
                op: BinOp::Add,
                lhs: b0,
                rhs: b1,
            },
        ],
    });
    let out = m.fresh();
    m.instrs.push(Instr::ConstI64(out, 0));
    m.instrs.push(Instr::Output(out));

    assert!(
        verify_module(&m).is_ok(),
        "a valid forward-chained fn body must still pass"
    );
}

// ---------------------------------------------------------------------------
// FINDING #9 — While/If region cond & branch-result definedness
// (check_ssa_well_formed / check_ssa_stream). std-surface only.
// ---------------------------------------------------------------------------

#[cfg(feature = "std-surface")]
fn valid_while_module() -> IRModule {
    let mut m = IRModule::new();
    let init = ValueId(0);
    m.instrs.push(Instr::ConstI64(init, 0));
    m.instrs.push(Instr::While {
        cond_id: ValueId(1), // produced by cond_instrs
        cond_instrs: vec![Instr::ConstI64(ValueId(1), 1)],
        body: vec![Instr::ConstI64(ValueId(2), 2)], // back-edge value
        live_vars: vec![("i".to_string(), ValueId(2))],
        init_ids: vec![init],
        exit_ids: vec![ValueId(3)],
    });
    m.instrs.push(Instr::Output(init));
    m.next_id = 4;
    m
}

#[cfg(feature = "std-surface")]
#[test]
fn while_valid_cond_passes() {
    assert!(
        check_ssa_well_formed(&valid_while_module()).is_ok(),
        "a While whose cond_id is produced by cond_instrs must pass"
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn while_dangling_cond_rejected() {
    let mut m = valid_while_module();
    // Corrupt the While cond_id to a value defined nowhere in the loop namespace.
    if let Some(Instr::While { cond_id, .. }) = m.instrs.get_mut(1) {
        *cond_id = ValueId(999);
    } else {
        panic!("expected While at index 1");
    }
    let err = check_ssa_well_formed(&m).unwrap_err();
    assert_eq!(err.value, ValueId(999), "must name the dangling cond id");
    assert!(matches!(err.rule, SsaRule::DefineBeforeUse));
}

#[cfg(feature = "std-surface")]
fn valid_if_module() -> IRModule {
    let mut m = IRModule::new();
    m.instrs.push(Instr::If {
        cond_id: ValueId(0), // produced by cond_instrs
        cond_instrs: vec![Instr::ConstI64(ValueId(0), 1)],
        then_instrs: vec![Instr::ConstI64(ValueId(1), 2)],
        then_result: ValueId(1), // in then scope
        else_instrs: vec![Instr::ConstI64(ValueId(2), 3)],
        else_result: ValueId(2), // in else scope
        dst: ValueId(3),
        branch_bindings: vec![],
        merges: vec![],
    });
    m.instrs.push(Instr::Output(ValueId(3)));
    m.next_id = 4;
    m
}

#[cfg(feature = "std-surface")]
#[test]
fn if_valid_passes() {
    assert!(
        check_ssa_well_formed(&valid_if_module()).is_ok(),
        "an If with defined cond/then/else results must pass"
    );
}

#[cfg(feature = "std-surface")]
#[test]
fn if_dangling_cond_rejected() {
    let mut m = valid_if_module();
    if let Some(Instr::If { cond_id, .. }) = m.instrs.get_mut(0) {
        *cond_id = ValueId(888);
    } else {
        panic!("expected If at index 0");
    }
    let err = check_ssa_well_formed(&m).unwrap_err();
    assert_eq!(err.value, ValueId(888));
    assert!(matches!(err.rule, SsaRule::DefineBeforeUse));
}

#[cfg(feature = "std-surface")]
#[test]
fn if_dangling_then_result_rejected() {
    let mut m = valid_if_module();
    if let Some(Instr::If { then_result, .. }) = m.instrs.get_mut(0) {
        *then_result = ValueId(777); // defined nowhere in then scope
    } else {
        panic!("expected If at index 0");
    }
    let err = check_ssa_well_formed(&m).unwrap_err();
    assert_eq!(err.value, ValueId(777));
    assert!(matches!(err.rule, SsaRule::DefineBeforeUse));
}

#[cfg(feature = "std-surface")]
#[test]
fn if_dangling_else_result_rejected() {
    let mut m = valid_if_module();
    if let Some(Instr::If { else_result, .. }) = m.instrs.get_mut(0) {
        *else_result = ValueId(666); // defined nowhere in else scope
    } else {
        panic!("expected If at index 0");
    }
    let err = check_ssa_well_formed(&m).unwrap_err();
    assert_eq!(err.value, ValueId(666));
    assert!(matches!(err.rule, SsaRule::DefineBeforeUse));
}

/// End-to-end: a crafted mic@3 artifact with a dangling While cond must FAIL
/// `check_ssa_well_formed` after a real decode round-trip (parse_mic3), proving
/// the gate closes on the untrusted-artifact path — not just on hand-built IR.
#[cfg(feature = "std-surface")]
#[test]
fn mic3_roundtrip_dangling_while_cond_rejected() {
    use libmind::ir::compact::{emit_mic3, parse_mic3};

    // Sanity: the valid module round-trips and still verifies clean.
    let good = valid_while_module();
    let good_bytes = emit_mic3(&good);
    let good_rt = parse_mic3(&good_bytes).expect("valid mic@3 must decode");
    assert!(
        check_ssa_well_formed(&good_rt).is_ok(),
        "valid mic@3 round-trip must verify clean"
    );

    // Now corrupt the cond and round-trip: decode must succeed, verify must fail.
    let mut bad = valid_while_module();
    if let Some(Instr::While { cond_id, .. }) = bad.instrs.get_mut(1) {
        *cond_id = ValueId(999);
    } else {
        panic!("expected While at index 1");
    }
    let bad_bytes = emit_mic3(&bad);
    let bad_rt = parse_mic3(&bad_bytes).expect("crafted mic@3 must still decode");
    let err = check_ssa_well_formed(&bad_rt)
        .expect_err("a decoded mic@3 with a dangling While cond must fail verify");
    assert_eq!(err.value, ValueId(999));
    assert!(matches!(err.rule, SsaRule::DefineBeforeUse));
}
