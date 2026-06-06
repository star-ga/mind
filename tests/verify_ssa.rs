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
use libmind::ir::{
    BinOp, IRModule, Instr, SsaRule, ValueId, check_ssa_well_formed,
};

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
    assert_eq!(err.value, ValueId(5), "violation must name the undefined %5");
    assert_eq!(
        err.rule,
        SsaRule::DefineBeforeUse,
        "violation must be a define-before-use fault"
    );
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
