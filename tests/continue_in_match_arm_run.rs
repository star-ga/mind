// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `continue` / `break` inside a `match`-arm body within a loop — RUNTIME gate
//! (regression).
//!
//! A `match` arm whose body is a bare `continue` (or `break`) desugars, through
//! the per-arm if-chain, into an `if` branch whose IR stream ends in
//! `Instr::Continue` / `Instr::Break`. The IR-level if-lowering computed each
//! branch's fall-through predicate by recognizing ONLY `Instr::Return` as a
//! terminator (`!matches!(last, Some(Instr::Return { .. }))`). A branch ending
//! in `continue`/`break` was therefore wrongly deemed to FALL THROUGH, so:
//!   1. the merge-phi placeholder consts were pushed AFTER the terminator
//!      instruction, and
//!   2. because the branch's `.last()` was then a `ConstI64` (not the
//!      `Continue`/`Break`), the MLIR if-lowering's terminator check
//!      (`instr_is_block_terminator(.last())`, which DOES recognize
//!      Break/Continue) missed and appended a second `cf.br ^if_after`.
//!
//! That yielded a block with a mid-block `cf.br` plus a trailing `cf.br`, which
//! `mlir-opt` rejects:
//! `error: operation with block successors must terminate its parent block`.
//!
//! This is a FAILS-TO-BUILD gap (not a silent miscompile): pre-fix the
//! `--emit-shared` subprocess errors out; post-fix it compiles and the
//! function returns the correct value. The qualified-final-arm form
//! (`R.Skip => continue`) dodged the bug by luck of its merge-value layout —
//! the BARE `_ => continue` / bare-variant form is what exposes it.
//!
//! Fix: `branch_terminates(&[Instr])` in `src/eval/lower.rs` recognizes
//! `Break`/`Continue` (std-surface) as terminators alongside `Return`, so the
//! dead placeholder consts are never pushed and the branch's `.last()` stays
//! the terminator. ADDITIVE under std-surface — keystone stays byte-identical.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test continue_in_match_arm_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
enum R { Keep(i64), Skip }

// Bare `_ => continue` arm: odd `i` produces R.Skip and the wildcard arm
// `continue`s, skipping the add. Sum of even i in 1..=6 = 2 + 4 + 6 = 12.
pub fn accumulate() -> i64 {
    let mut acc = 0
    let mut i = 0
    while i < 6 {
        i = i + 1
        let r = if i % 2 == 0 { R.Keep(i) } else { R.Skip }
        match r {
            Keep(p) => acc = acc + p,
            _ => continue,
        }
    }
    return acc
}

// `break` inside a match arm: stop accumulating at i == 4. 1 + 2 + 3 = 6.
pub fn stop_at_four() -> i64 {
    let mut acc = 0
    let mut i = 0
    while i < 10 {
        i = i + 1
        let r = if i == 4 { R.Skip } else { R.Keep(i) }
        match r {
            Keep(p) => acc = acc + p,
            _ => break,
        }
    }
    return acc
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn continue_in_match_arm_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("continue-in-match-arm-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_continue_in_match_arm_run.mind");
    let so = dir.join("mind_continue_in_match_arm_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("continue-in-match-arm-run: needs mlir-build; skipping");
            return;
        }
        panic!("continue-in-match-arm-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for fn, exp in (('accumulate', 12), ('stop_at_four', 6)):\n\
         \x20   f = getattr(lib, fn); f.restype = ctypes.c_int64; f.argtypes = []\n\
         \x20   got = f()\n\
         \x20   assert got == exp, fn + ': got=' + str(got) + ' expected=' + str(exp)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "continue-in-match-arm-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
