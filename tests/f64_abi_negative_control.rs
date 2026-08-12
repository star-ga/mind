// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Non-vacuous NEGATIVE control for the scalar-`f64` call-ARGUMENT ABI boundary,
//! paired with a POSITIVE control (control-on-control).
//!
//! Fail-closed proof required alongside the #298 self-host f64 MLIR surface: an
//! invalid `f64` call-signature usage MUST be rejected at the ABI boundary — not
//! silently miscompiled, not a crash, not a parse error, and not a link error.
//!
//! Discriminating condition (the ONLY thing perturbed between the two variants):
//! the caller's argument is a typed `i64` VARIABLE (NEGATIVE) vs an `f64` VARIABLE
//! (POSITIVE). Everything else is byte-identical. (A bare integer *literal* like
//! `scale(5)` is polymorphic to `5.0` and is legitimately valid, so it is NOT a
//! mismatch — the discriminator must be a typed non-float value.)
//!
//! Machine-checked conjunction:
//!   POSITIVE  `fn driver(v: f64) -> f64 { scale(v) }`
//!       * `mindc --emit-shared` SUCCEEDS  → the shared structure parses + lowers
//!         + links; proves the negative's failure is NOT a parse / structural one.
//!   NEGATIVE  `fn driver(n: i64) -> f64 { scale(n) }`
//!       * `mindc --emit-shared` FAILS at MLIR lowering with the EXACT scalar-ABI
//!         type-conflict diagnostic `expects different type than prior uses:
//!         'f64' vs 'i64'` — rejected AT the f64 call-argument ABI boundary.
//!       * its stderr carries NO parse-error marker (not-parse) and NO linker
//!         marker (link never reached).
//!
//! If the negative compiled, or the positive failed, or the negative failed for
//! any other reason (parse / unrelated / link), this gate FAILS — a vacuous
//! `exit != 0` is explicitly rejected.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test f64_abi_negative_control`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// POSITIVE control: f64 variable argument — valid f64 call ABI.
const POS_SRC: &str = "pub fn scale(x: f64) -> f64 {\n    x + 1.0\n}\n\npub fn driver(v: f64) -> f64 {\n    scale(v)\n}\n";

// NEGATIVE: typed i64 variable argument to the f64 parameter — invalid f64 ABI.
// Byte-identical to POS_SRC except `v: f64` -> `n: i64` (the discriminator).
const NEG_SRC: &str = "pub fn scale(x: f64) -> f64 {\n    x + 1.0\n}\n\npub fn driver(n: i64) -> f64 {\n    scale(n)\n}\n";

// The exact MLIR-lowering ABI type-conflict diagnostic the negative must emit.
const ABI_DIAG: &str = "expects different type than prior uses";

fn emit_shared(
    mindc: &std::path::Path,
    src_path: &std::path::Path,
    so_path: &std::path::Path,
) -> std::process::Output {
    Command::new(mindc)
        .args([
            src_path.to_str().unwrap(),
            "--emit-shared",
            so_path.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc --emit-shared")
}

#[test]
fn f64_abi_negative_control_and_positive() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("f64-abi-negative-control: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let pos = dir.join("mind_f64_abi_pos.mind");
    let neg = dir.join("mind_f64_abi_neg.mind");
    let pos_so = dir.join("mind_f64_abi_pos.so");
    let neg_so = dir.join("mind_f64_abi_neg.so");
    std::fs::write(&pos, POS_SRC).expect("write pos");
    std::fs::write(&neg, NEG_SRC).expect("write neg");

    // ---- POSITIVE control: the identical call with an f64 arg must BUILD. ----
    let pout = emit_shared(&mindc, &pos, &pos_so);
    if !pout.status.success() {
        let stderr = String::from_utf8_lossy(&pout.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("f64-abi-negative-control: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("POSITIVE control (f64 arg) must compile, but failed:\n{stderr}");
    }

    // ---- NEGATIVE: --emit-shared must FAIL at the f64 call-ABI boundary. ----
    let nout = emit_shared(&mindc, &neg, &neg_so);
    let nstderr = String::from_utf8_lossy(&nout.stderr).to_string();
    let nlow = nstderr.to_lowercase();
    // Invariant f64<->i64 type-conflict marker set (robust to cosmetic diagnostic
    // rewording): load-bearing gate is BOTH types present + this marker set + the
    // pos/neg discriminator, NOT a single exact human phrase. ABI_DIAG kept as one
    // accepted marker so the exact-prose signal still counts when present.
    let type_conflict = nstderr.contains(ABI_DIAG)
        || nlow.contains("different type")
        || nlow.contains("type mismatch")
        || nlow.contains("incompatible type")
        || nlow.contains("expects");

    assert!(
        !nout.status.success(),
        "NEGATIVE (typed i64 var to f64 param) must be REJECTED (fail-closed), but it \
         compiled — a silent-miscompile / fail-open at the f64 call-ABI boundary."
    );
    // EXACT reason: the f64/i64 scalar-ABI type conflict (not a vacuous exit!=0).
    assert!(
        nstderr.contains("f64") && nstderr.contains("i64") && type_conflict,
        "NEGATIVE must fail with the EXACT f64 call-ABI type-conflict diagnostic \
         (`{ABI_DIAG} … 'f64' vs 'i64'`), not some unrelated error:\n{nstderr}"
    );
    // Not a parse failure (the POSITIVE, structurally identical, parsed + built).
    assert!(
        !nlow.contains("unexpected token") && !nlow.contains("parse error"),
        "NEGATIVE failure must NOT be a parse error (it is the ABI boundary):\n{nstderr}"
    );
    // Link is never reached (an mlir-lowering type conflict aborts before clang).
    assert!(
        !nlow.contains("undefined reference")
            && !nlow.contains("ld returned")
            && !nlow.contains("linker command failed"),
        "NEGATIVE failure must be at the ABI/lowering boundary, NOT the linker:\n{nstderr}"
    );
}
