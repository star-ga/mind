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

use std::path::PathBuf;
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
// Dominance-awareness: control-flow regions (MIND-Fuzz bug #5).
//
// The earlier linear-program-order check over-rejected SSA WITH control flow:
// it checked a region node's forwarded operands before recursing into the
// branch/body that defines them, and never exposed a loop's post-body /
// exit ids to code after the loop. These programs compile, run, and are
// deterministic; they must verify clean. The negative test below proves the
// dominance relaxation did NOT start accepting genuinely ill-formed SSA.
// ---------------------------------------------------------------------------

/// `if cond { x = 10 } else { x = 20 }; return x` — the merge forwards the
/// then-branch incarnation (`%4`) and the else-branch incarnation (`%5`), and
/// the post-merge value (`%6`) is read afterward. Both branch incarnations are
/// defined INSIDE their respective branches, and the merge id dominates the
/// post-branch use. This is the control-flow shape MIND-Fuzz false-flagged with
/// "define-before-use %4". It must now verify clean.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_if_merge_branch_defined_values_pass() {
    let mut m = IRModule::new();
    // %0 = condition.
    m.instrs.push(Instr::ConstI64(ValueId(0), 1));
    m.instrs.push(Instr::If {
        cond_id: ValueId(0),
        cond_instrs: vec![],
        // then branch defines %4 (a branch-local incarnation of `x`).
        then_instrs: vec![Instr::ConstI64(ValueId(4), 10)],
        then_result: ValueId(4),
        // else branch defines %5.
        else_instrs: vec![Instr::ConstI64(ValueId(5), 20)],
        else_result: ValueId(5),
        dst: ValueId(3),
        branch_bindings: vec![("x".to_string(), ValueId(6))],
        // merge id %6 = phi(then=%4, else=%5).
        merges: vec![(ValueId(6), ValueId(4), ValueId(5))],
    });
    // Post-merge code reads the merge id %6 (a `^if_after` block arg that
    // dominates every path), then outputs it.
    m.instrs.push(Instr::BinOp {
        dst: ValueId(7),
        op: BinOp::Add,
        lhs: ValueId(6),
        rhs: ValueId(0),
    });
    m.instrs.push(Instr::Output(ValueId(7)));
    m.next_id = 8;

    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "an if/else whose merge forwards branch-defined incarnations, with the \
         merge id read after the branch, must verify clean: {:?}",
        check_ssa_well_formed(&m)
    );
}

/// `let mut s = 0; while cond { s = s + 1 }; return s` — the loop's post-body
/// `live_vars` id (`%5`) and `exit_ids` (`%6`) are `^while_after` block args
/// that dominate code after the loop. The earlier check never exposed them, so
/// a post-loop read false-flagged "define-before-use". Must verify clean.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_while_post_loop_carried_value_passes() {
    let mut m = IRModule::new();
    // %0 = initial accumulator (pre-loop, enclosing scope).
    m.instrs.push(Instr::ConstI64(ValueId(0), 0));
    m.instrs.push(Instr::While {
        cond_id: ValueId(1),
        // The header re-evaluates the condition into its own region id %1.
        cond_instrs: vec![Instr::ConstI64(ValueId(1), 1)],
        // body increments the carried var into a post-body id %5.
        body: vec![
            Instr::ConstI64(ValueId(4), 1),
            Instr::BinOp {
                dst: ValueId(5),
                op: BinOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(4),
            },
        ],
        live_vars: vec![("s".to_string(), ValueId(5))],
        // init_ids: pre-loop value of `s` is %0 (defined before the loop).
        init_ids: vec![ValueId(0)],
        // exit_ids: the ^while_after block arg the post-loop `s` is rebound to.
        exit_ids: vec![ValueId(6)],
    });
    // Post-loop code reads the exit id %6, which dominates the after-block.
    m.instrs.push(Instr::Output(ValueId(6)));
    m.next_id = 7;

    assert!(
        check_ssa_well_formed(&m).is_ok(),
        "a while loop's post-body live_var and exit_id must be visible to code \
         after the loop: {:?}",
        check_ssa_well_formed(&m)
    );
}

/// A value defined ONLY inside a branch and read AFTER the merge — without being
/// forwarded through a merge id — does NOT dominate the post-branch use and must
/// still be rejected. The dominance relaxation must not turn into acceptance of
/// genuinely ill-formed SSA (a branch-local id escaping its branch).
#[cfg(feature = "std-surface")]
#[test]
fn ssa_branch_local_value_used_after_merge_fails() {
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 1)); // condition
    m.instrs.push(Instr::If {
        cond_id: ValueId(0),
        cond_instrs: vec![],
        // then defines %4, NOT exposed through any merge.
        then_instrs: vec![Instr::ConstI64(ValueId(4), 10)],
        then_result: ValueId(4),
        else_instrs: vec![Instr::ConstI64(ValueId(5), 20)],
        else_result: ValueId(5),
        dst: ValueId(3),
        branch_bindings: vec![],
        merges: vec![],
    });
    // Illegally read %4, a then-branch-local id, after the merge: %4 is not
    // defined on the else path and does not dominate this use.
    m.instrs.push(Instr::Output(ValueId(4)));
    m.next_id = 6;

    let err = check_ssa_well_formed(&m)
        .expect_err("a branch-local id read after the merge must be rejected");
    assert_eq!(err.value, ValueId(4), "violation must name the escaping %4");
    assert_eq!(
        err.rule,
        SsaRule::DefineBeforeUse,
        "violation must be a define-before-use fault"
    );
}

/// A genuine use-before-def INSIDE an if branch (an operand referenced before
/// any instruction on its path defines it) must still be rejected — the branch
/// scope is dominance-checked, not blanket-accepted.
#[cfg(feature = "std-surface")]
#[test]
fn ssa_use_before_def_inside_branch_fails() {
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 1)); // condition
    m.instrs.push(Instr::If {
        cond_id: ValueId(0),
        cond_instrs: vec![],
        // %4 reads %9, which nothing defines on any reaching path.
        then_instrs: vec![Instr::BinOp {
            dst: ValueId(4),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(9), // undefined
        }],
        then_result: ValueId(4),
        else_instrs: vec![Instr::ConstI64(ValueId(5), 20)],
        else_result: ValueId(5),
        dst: ValueId(3),
        branch_bindings: vec![("x".to_string(), ValueId(6))],
        merges: vec![(ValueId(6), ValueId(4), ValueId(5))],
    });
    m.instrs.push(Instr::Output(ValueId(6)));
    m.next_id = 10;

    let err = check_ssa_well_formed(&m)
        .expect_err("an undefined operand inside a branch must be rejected");
    assert_eq!(
        err.value,
        ValueId(9),
        "violation must name the undefined %9"
    );
    assert_eq!(err.rule, SsaRule::DefineBeforeUse);
}

// ---------------------------------------------------------------------------
// CLI-level: mindc verify reports ssa_valid over a mic@3 artifact.
// ---------------------------------------------------------------------------

fn mindc_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    #[cfg(debug_assertions)]
    path.push("debug");
    #[cfg(not(debug_assertions))]
    path.push("release");
    #[cfg(target_os = "windows")]
    path.push("mindc.exe");
    #[cfg(not(target_os = "windows"))]
    path.push("mindc");
    path
}

fn tempfile_path(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("mindc_verify_ssa_test_{name}"));
    p.to_string_lossy().into_owned()
}

#[test]
fn cli_verify_reports_ssa_valid_on_well_formed_artifact() {
    let bin = mindc_binary();
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
    let bin = mindc_binary();
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
