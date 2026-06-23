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

//! Tests for the SSA well-formedness slice of `mindc verify` (RFC 0017,
//! second static-verification property after the trace_hash check).
//!
//! Two layers:
//!   * library-level — [`check_ssa_well_formed`] over hand-built / corrupted
//!     `IRModule`s, asserting the precise violated rule and value id;
//!   * CLI-level — a well-formed mic@3 artifact verifies `ssa_valid: yes` and a
//!     hand-corrupted one (duplicate result id) fails with `ssa_valid: NO` and
//!     a nonzero exit, mirroring the trace_hash gate in `verify_cli.rs`.

mod common;
use common::mindc_bin;

use std::process::Command;

use libmind::ir::compact::emit_mic3;
use libmind::ir::{BinOp, IRModule, Instr, SsaRule, ValueId, check_ssa_well_formed};

// ---------------------------------------------------------------------------
// Library-level: check_ssa_well_formed directly.
// ---------------------------------------------------------------------------

/// `(%0 = 2) (%1 = 3) (%2 = %0 + %1) output %2` — single-assignment and every
/// operand defined before use. Must verify clean.
fn well_formed_module() -> IRModule {
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 2));
    m.instrs.push(Instr::ConstI64(ValueId(1), 3));
    m.instrs.push(Instr::BinOp {
        dst: ValueId(2),
        op: BinOp::Add,
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    m.instrs.push(Instr::Output(ValueId(2)));
    m.next_id = 3;
    m
}

#[test]
fn ssa_well_formed_module_passes() {
    let m = well_formed_module();
    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "a single-assignment, define-before-use module must pass"
    );
}

#[test]
fn ssa_duplicate_result_id_fails() {
    // Two instructions produce %1 — single-assignment violation on %1.
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 2));
    m.instrs.push(Instr::ConstI64(ValueId(1), 3));
    // Re-define %1 with a second instruction.
    m.instrs.push(Instr::BinOp {
        dst: ValueId(1),
        op: BinOp::Add,
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    m.instrs.push(Instr::Output(ValueId(1)));
    m.next_id = 2;

    let err = check_ssa_well_formed(&m).expect_err("duplicate %1 must fail");
    assert_eq!(err.value, ValueId(1), "violation must name %1");
    assert_eq!(
        err.rule,
        SsaRule::SingleAssignment,
        "violation must be a single-assignment fault"
    );
}

#[test]
fn ssa_undefined_operand_fails() {
    // %2 references %5, which is never defined — define-before-use violation.
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 2));
    m.instrs.push(Instr::ConstI64(ValueId(1), 3));
    m.instrs.push(Instr::BinOp {
        dst: ValueId(2),
        op: BinOp::Add,
        lhs: ValueId(0),
        rhs: ValueId(5), // undefined
    });
    m.instrs.push(Instr::Output(ValueId(2)));
    m.next_id = 6;

    let err = check_ssa_well_formed(&m).expect_err("undefined %5 must fail");
    assert_eq!(
        err.value,
        ValueId(5),
        "violation must name the undefined %5"
    );
    assert_eq!(
        err.rule,
        SsaRule::DefineBeforeUse,
        "violation must be a define-before-use fault"
    );
}

/// A function with a parameter AND body values that reuse ids from a *separate*
/// per-function namespace must verify clean — the `scalar_arith.mind` shape that
/// MIND-Fuzz flagged as a verifier false-positive.
///
/// Models `pub fn f(a: i64) -> i64 { let x = 5; return a + x; }`: param `a` is
/// `%0`, the body materializes `Param %0`, `ConstI64 %1`, `BinOp %2 = %0 + %1`,
/// then a top-level `%0`/`Output %0` follows the `FnDef` in the module's own
/// (separate) namespace. The pre-fix verifier threaded a single shared
/// `defined` set through the whole module and wrongly reported `%0` defined more
/// than once.
#[test]
fn ssa_fn_param_and_body_values_pass() {
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: "f".to_string(),
        params: vec![("a".to_string(), ValueId(0))],
        ret_id: Some(ValueId(2)),
        body: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".to_string(),
                index: 0,
            },
            Instr::ConstI64(ValueId(1), 5),
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
        reap_threshold: None,
    });
    // Top-level namespace reuses %0 — must NOT collide with the function's %0.
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::Output(ValueId(0)));
    m.next_id = 3;

    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "a function with a param and body values in a separate namespace, \
         followed by a top-level value reusing an id, must verify clean: {:?}",
        check_ssa_well_formed(&m)
    );
}

/// A genuine duplicate result id *inside one function body* must still be
/// rejected — the per-function fresh scope must not be so loose that it stops
/// catching real single-assignment faults.
#[test]
fn ssa_duplicate_result_id_in_fn_body_fails() {
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: "f".to_string(),
        params: vec![("a".to_string(), ValueId(0))],
        ret_id: Some(ValueId(1)),
        body: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "a".to_string(),
                index: 0,
            },
            Instr::ConstI64(ValueId(1), 5),
            // Re-define %1 with a second instruction — single-assignment fault.
            Instr::ConstI64(ValueId(1), 6),
            Instr::Return {
                value: Some(ValueId(1)),
            },
        ],
        reap_threshold: None,
    });
    m.next_id = 2;

    let err = check_ssa_well_formed(&m).expect_err("duplicate %1 in fn body must fail");
    assert_eq!(err.value, ValueId(1), "violation must name %1");
    assert_eq!(
        err.rule,
        SsaRule::SingleAssignment,
        "violation must be a single-assignment fault"
    );
}

// ---------------------------------------------------------------------------
// CLI-level: mindc verify reports ssa_valid over a mic@3 artifact.
// ---------------------------------------------------------------------------

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

fn tempfile_path(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("mindc_verify_ssa_test_{name}"));
    p.to_string_lossy().into_owned()
}

#[test]
fn cli_verify_reports_ssa_valid_on_well_formed_artifact() {
    let bin = mindc_bin();
    if !bin.exists() {
        eprintln!("Skipping: mindc binary not found at {bin:?}");
        return;
    }

    // Emit a plain mic@3 from the well-formed module (no evidence chain needed
    // to observe the ssa_* fields — they are emitted unconditionally).
    let tmp = tempfile_path("wellformed.mic3");
    std::fs::write(&tmp, emit_mic3(&well_formed_module())).expect("write mic@3");

    let out = Command::new(&bin)
        .args(["verify", &tmp, "--json"])
        .output()
        .expect("run mindc verify --json");
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("\"ssa_valid\":true"),
        "well-formed artifact must report ssa_valid:true, got:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn cli_verify_fails_on_ssa_corrupted_artifact() {
    let bin = mindc_bin();
    if !bin.exists() {
        eprintln!("Skipping: mindc binary not found at {bin:?}");
        return;
    }

    // Hand-corrupt the IR: two instructions define %1 (single-assignment
    // violation). Emit it to a real mic@3 artifact and verify through the CLI.
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 2));
    m.instrs.push(Instr::ConstI64(ValueId(1), 3));
    m.instrs.push(Instr::BinOp {
        dst: ValueId(1), // duplicate definition of %1
        op: BinOp::Add,
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    m.instrs.push(Instr::Output(ValueId(1)));
    m.next_id = 2;

    let tmp = tempfile_path("corrupt.mic3");
    std::fs::write(&tmp, emit_mic3(&m)).expect("write corrupt mic@3");

    let out = Command::new(&bin)
        .args(["verify", &tmp])
        .output()
        .expect("run mindc verify on SSA-corrupt artifact");

    assert_eq!(
        out.status.code(),
        Some(1),
        "SSA-corrupt artifact must exit 1, stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("ssa_valid:        NO"),
        "human report must show ssa_valid: NO, got:\n{stdout}"
    );
    assert!(
        stderr.contains("SSA well-formedness check FAILED")
            && stderr.contains("single-assignment")
            && stderr.contains("%1"),
        "error must name the single-assignment fault on %1, got:\n{stderr}"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// Issue #24 SOUNDNESS — If-merge `then_val`/`else_val` must be validated against
// PER-BRANCH scopes, not the merged (enclosing ∪ then ∪ else) scope. A cross-
// branch forgery (then_val pointing at an else-only value, or vice versa) must
// be REJECTED by BOTH the consumer verifier (check_ssa_well_formed) and the
// in-pipeline verifier (verify_module). These are the permanent regression
// tests; before the per-branch fix the malformed cases were wrongly accepted.
// ---------------------------------------------------------------------------

/// Build a `FnDef` containing a single `If` whose then/else branches each define
/// one value, plus a `merges` tuple wiring `(merge_id, then_val, else_val)`.
/// Both branches fall through (no `Return` terminator). Ids are laid out so the
/// branch-internal values are distinct across the two arms:
///   then-branch defines %then_def, else-branch defines %else_def.
#[cfg(feature = "std-surface")]
fn if_merge_module(
    then_val: ValueId,
    else_val: ValueId,
    then_def: ValueId,
    else_def: ValueId,
    merge_id: ValueId,
    dst: ValueId,
    next_id: usize,
) -> IRModule {
    let mut m = IRModule::new();
    m.instrs.push(Instr::FnDef {
        name: "f".to_string(),
        params: vec![("c".to_string(), ValueId(0))],
        ret_id: Some(merge_id),
        body: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "c".to_string(),
                index: 0,
            },
            Instr::If {
                cond_id: ValueId(0),
                cond_instrs: vec![],
                // then-branch: defines %then_def (e.g. 10), falls through.
                then_instrs: vec![Instr::ConstI64(then_def, 10)],
                then_result: then_def,
                // else-branch: defines %else_def (e.g. 20), falls through.
                else_instrs: vec![Instr::ConstI64(else_def, 20)],
                else_result: else_def,
                dst,
                branch_bindings: vec![("x".to_string(), merge_id)],
                merges: vec![(merge_id, then_val, else_val)],
            },
            Instr::Return {
                value: Some(merge_id),
            },
        ],
        reap_threshold: None,
    });
    // Top-level value + Output, in the module's own SSA namespace (distinct from
    // the fn-body namespace), so verify_module's MissingOutput check is satisfied
    // and the Output operand is genuinely defined at module scope.
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::Output(ValueId(0)));
    m.next_id = next_id;
    m
}

/// GATE 3 — a well-formed `If` where `then_val` is the then-branch's own value
/// and `else_val` is the else-branch's own value must verify clean under BOTH
/// verifiers (no false positive).
#[cfg(feature = "std-surface")]
#[test]
fn ssa_if_merge_per_branch_legit_passes() {
    // then_def=%1, else_def=%2, then_val=%1 (then-defined), else_val=%2 (else).
    let m = if_merge_module(
        ValueId(1),
        ValueId(2),
        ValueId(1),
        ValueId(2),
        ValueId(3),
        ValueId(4),
        5,
    );
    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "legit per-branch merge must pass check_ssa_well_formed: {:?}",
        check_ssa_well_formed(&m)
    );
    assert!(
        libmind::ir::verify_module(&m).is_ok(),
        "legit per-branch merge must pass verify_module: {:?}",
        libmind::ir::verify_module(&m)
    );
}

/// GATE 2 (★ SOUNDNESS) — `then_val` points at an ELSE-branch-only value
/// (%else_def). This is a genuine cross-branch dominance violation: the
/// then-edge of the merge cannot reference a value defined only in the else
/// branch. BOTH verifiers must REJECT it with define-before-use. Before the
/// per-branch fix this was wrongly accepted.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_if_merge_then_val_from_else_branch_rejected() {
    // then_def=%1, else_def=%2; then_val=%2 (ELSE-only) ← FORGERY.
    let m = if_merge_module(
        ValueId(2), // then_val = else-branch-only value
        ValueId(2), // else_val = legit else value
        ValueId(1),
        ValueId(2),
        ValueId(3),
        ValueId(4),
        5,
    );
    let err = check_ssa_well_formed(&m)
        .expect_err("then_val from else-branch must be rejected by check_ssa_well_formed");
    assert_eq!(
        err.value,
        ValueId(2),
        "violation must name the else-only %2"
    );
    assert_eq!(
        err.rule,
        SsaRule::DefineBeforeUse,
        "cross-branch merge forgery is a define-before-use fault"
    );
    assert!(
        libmind::ir::verify_module(&m).is_err(),
        "verify_module must also reject then_val from else-branch"
    );
}

/// GATE 2 symmetric — `else_val` points at a THEN-branch-only value
/// (%then_def). BOTH verifiers must REJECT.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_if_merge_else_val_from_then_branch_rejected() {
    // then_def=%1, else_def=%2; else_val=%1 (THEN-only) ← FORGERY.
    let m = if_merge_module(
        ValueId(1), // then_val = legit then value
        ValueId(1), // else_val = then-branch-only value
        ValueId(1),
        ValueId(2),
        ValueId(3),
        ValueId(4),
        5,
    );
    let err = check_ssa_well_formed(&m)
        .expect_err("else_val from then-branch must be rejected by check_ssa_well_formed");
    assert_eq!(
        err.value,
        ValueId(1),
        "violation must name the then-only %1"
    );
    assert_eq!(err.rule, SsaRule::DefineBeforeUse);
    assert!(
        libmind::ir::verify_module(&m).is_err(),
        "verify_module must also reject else_val from then-branch"
    );
}

/// GATE 4 (DIFFERENTIAL) — `check_ssa_well_formed(parse(emit(ir)))` agrees with
/// `verify_module(ir)` for the legit case AND both forgery cases. The codec
/// round-trip preserves `merges` (mic@3 version 0x02), so the consumer verifier
/// sees the same cross-branch forgery the in-pipeline verifier rejects.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_if_merge_differential_emit_parse_agrees() {
    use libmind::ir::compact::v3::parse_mic3;

    let cases = [
        // (module, expect_ok)
        (
            if_merge_module(
                ValueId(1),
                ValueId(2),
                ValueId(1),
                ValueId(2),
                ValueId(3),
                ValueId(4),
                5,
            ),
            true,
        ),
        (
            if_merge_module(
                ValueId(2),
                ValueId(2),
                ValueId(1),
                ValueId(2),
                ValueId(3),
                ValueId(4),
                5,
            ),
            false,
        ),
        (
            if_merge_module(
                ValueId(1),
                ValueId(1),
                ValueId(1),
                ValueId(2),
                ValueId(3),
                ValueId(4),
                5,
            ),
            false,
        ),
    ];

    for (m, expect_ok) in cases {
        let vm_ok = libmind::ir::verify_module(&m).is_ok();
        let bytes = emit_mic3(&m);
        let parsed = parse_mic3(&bytes).expect("parse round-trip");
        let consumer_ok = check_ssa_well_formed(&parsed).is_ok();
        assert_eq!(
            vm_ok, expect_ok,
            "verify_module verdict mismatch for expect_ok={expect_ok}"
        );
        assert_eq!(
            consumer_ok, expect_ok,
            "check_ssa_well_formed(parse(emit)) verdict mismatch for expect_ok={expect_ok}"
        );
        assert_eq!(
            vm_ok, consumer_ok,
            "DIFFERENTIAL: verify_module and check_ssa_well_formed must agree"
        );
    }
}

// ---------------------------------------------------------------------------
// SOUNDNESS (While loop-carry) — a `While`'s `live_vars` post-body id is the
// BACK-EDGE value of a loop-carried variable: the value the var holds at the
// END of the body, fed back to the loop header on the next iteration. It must be
// an SSA value that is DEFINED and dominates the back-edge — produced inside the
// loop (`cond` ∪ `body`) OR live from the enclosing (pre-loop) scope (a plain
// rebinding such as `a = b` legitimately re-points the carried var at another
// already-live id). Post-loop code, by contrast, reads the loop's `exit_ids`
// (`^while_after` block args), NOT the post-body ids.
//
// A `live_vars` back-edge id that is defined NOWHERE in `enclosing ∪ cond ∪ body`
// is a forgery — the loop header would read an SSA id out of thin air (a
// use-before-def on the back-edge, a wrong/dangling value). This is the loop
// analog of the `If` cross-branch merge forgery (issue #24): both verifiers MUST
// reject it.
//
// Before the fix `expose_region_definitions` wrongly EXPOSED the post-body ids
// into the enclosing scope (so a forged id was silently treated as "defined" for
// the rest of the fn body), and neither verifier validated them at all, so a
// `While` whose back-edge value points at an undefined id passed clean — a
// wrong-value module certified valid.
// ---------------------------------------------------------------------------

/// Build a `FnDef` with a single counted `while` loop carrying one var `s`.
/// The body computes `s_body = s + 1` (`%4`), the genuine back-edge value.
/// `live_post` is what `While.live_vars` claims that back-edge id is — set it to
/// `%4` for a legit loop, or to a forged id to model a tampered artifact.
#[cfg(feature = "std-surface")]
fn while_carry_module(live_post: ValueId) -> IRModule {
    let mut m = IRModule::new();
    // %0 = param n ; %1 = const 0 (s init) ; %2 = const 1 (cond literal) ;
    // body: %3 = const 1 ; %4 = s + 1 (back-edge value) ;
    // exit ids: %5 (^while_after block arg for s) ; return %5.
    m.instrs.push(Instr::FnDef {
        name: "f".to_string(),
        params: vec![("n".to_string(), ValueId(0))],
        ret_id: Some(ValueId(5)),
        body: vec![
            Instr::Param {
                dst: ValueId(0),
                name: "n".to_string(),
                index: 0,
            },
            Instr::ConstI64(ValueId(1), 0), // s = 0 (pre-loop / init)
            Instr::While {
                cond_id: ValueId(2),
                cond_instrs: vec![Instr::ConstI64(ValueId(2), 1)],
                body: vec![
                    Instr::ConstI64(ValueId(3), 1),
                    Instr::BinOp {
                        dst: ValueId(4),
                        op: BinOp::Add,
                        lhs: ValueId(1),
                        rhs: ValueId(3),
                    },
                ],
                // back-edge value for `s`: legit = %4 (body-defined); forged otherwise.
                live_vars: vec![("s".to_string(), live_post)],
                init_ids: vec![ValueId(1)],
                exit_ids: vec![ValueId(5)],
            },
            Instr::Return {
                value: Some(ValueId(5)),
            },
        ],
        reap_threshold: None,
    });
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::Output(ValueId(0)));
    m.next_id = 6;
    m
}

/// A well-formed loop whose back-edge value `%4` is defined inside the body must
/// pass BOTH verifiers (no false positive).
#[cfg(feature = "std-surface")]
#[test]
fn ssa_while_carry_legit_passes() {
    let m = while_carry_module(ValueId(4));
    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "legit loop-carry must pass check_ssa_well_formed: {:?}",
        check_ssa_well_formed(&m)
    );
    assert!(
        libmind::ir::verify_module(&m).is_ok(),
        "legit loop-carry must pass verify_module: {:?}",
        libmind::ir::verify_module(&m)
    );
}

/// A legit loop whose carried back-edge value is a PRE-LOOP value (`%1`, the
/// init) must still pass: a plain rebinding (`a = b`) or a no-op (`s = s`) points
/// the carried var at an already-live id without emitting a new instruction, so
/// an enclosing-scope back-edge id is valid. (Guards against the over-strict
/// "must be a fresh body def" check that would false-positive real `fib`-style
/// loops, where `a = b` makes `a`'s back-edge `b`'s pre-loop id.)
#[cfg(feature = "std-surface")]
#[test]
fn ssa_while_carry_preloop_backedge_passes() {
    let m = while_carry_module(ValueId(1));
    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "enclosing-scope (rebind) back-edge must pass check_ssa_well_formed: {:?}",
        check_ssa_well_formed(&m)
    );
    assert!(
        libmind::ir::verify_module(&m).is_ok(),
        "enclosing-scope (rebind) back-edge must pass verify_module: {:?}",
        libmind::ir::verify_module(&m)
    );
}

/// ★ SOUNDNESS — the `live_vars` back-edge id `%99` is defined NOWHERE in
/// `enclosing ∪ cond ∪ body`. This is a loop-carry forgery: the loop header
/// would read an undefined SSA value on the back-edge (a wrong/dangling value).
/// BOTH verifiers must REJECT it with define-before-use. Before the fix this was
/// wrongly accepted (the post-body id was exposed, never validated).
#[cfg(feature = "std-surface")]
#[test]
fn ssa_while_carry_undefined_post_id_rejected() {
    let m = while_carry_module(ValueId(99));
    let err = check_ssa_well_formed(&m)
        .expect_err("undefined loop-carry back-edge id must be rejected by check_ssa_well_formed");
    assert_eq!(err.value, ValueId(99), "violation must name the undefined %99");
    assert_eq!(
        err.rule,
        SsaRule::DefineBeforeUse,
        "loop-carry back-edge forgery is a define-before-use fault"
    );
    assert!(
        libmind::ir::verify_module(&m).is_err(),
        "verify_module must also reject the undefined loop-carry back-edge id"
    );
}

/// ★ SOUNDNESS (DIFFERENTIAL) — the consumer verifier on the codec round-trip
/// (`check_ssa_well_formed(parse(emit(ir)))`) and the in-pipeline `verify_module`
/// agree on the legit, rebind, and forged loop-carry cases. `While.live_vars` is
/// serialized in mic@3 (version 0x02), so a tampered artifact's forged back-edge
/// id survives the round-trip and must be rejected on the consumer side too.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_while_carry_differential_emit_parse_agrees() {
    use libmind::ir::compact::v3::parse_mic3;
    let cases = [
        (while_carry_module(ValueId(4)), true),
        (while_carry_module(ValueId(1)), true),
        (while_carry_module(ValueId(99)), false),
    ];
    for (m, expect_ok) in cases {
        let vm_ok = libmind::ir::verify_module(&m).is_ok();
        let bytes = emit_mic3(&m);
        let parsed = parse_mic3(&bytes).expect("parse round-trip");
        let consumer_ok = check_ssa_well_formed(&parsed).is_ok();
        assert_eq!(
            vm_ok, expect_ok,
            "verify_module verdict mismatch for expect_ok={expect_ok}"
        );
        assert_eq!(
            consumer_ok, expect_ok,
            "check_ssa_well_formed(parse(emit)) verdict mismatch for expect_ok={expect_ok}"
        );
        assert_eq!(
            vm_ok, consumer_ok,
            "DIFFERENTIAL: verify_module and check_ssa_well_formed must agree on loop-carry"
        );
    }
}
