// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Libm NAME-CAPTURE runtime regression gate.
//!
//! A MIND function that happens to share its name with a C library routine
//! (`pow`, `sin`, `exp`, ...) used to be compiled as if it WERE that routine:
//! the build hands LLVM IR to `clang -O3 -x ir`, LLVM recognises the libm name,
//! ignores the MIND body entirely, and constant-folds every literal-argument
//! call site to the value the HOST toolchain's folder picks. The MIND kernel
//! never ran, and the folded value is toolchain-dependent (two folders disagree
//! on `pow(-2, 0.5)`), which is a cross-substrate byte-identity violation.
//!
//! The probe bodies below CANNOT produce any libm value: `pow` only ever
//! returns `0.0` or `1.0`, `sin` is the constant `0.125`, `exp` the constant
//! `0.25`. A literal-argument call must therefore return the MIND body's value.
//! The `*_param` probes take the argument at run time (no fold is possible)
//! and serve as the positive control that the body itself is what the literal
//! probes should match.
//!
//! Gate: `cargo test --release --features "mlir-build std-surface cross-module-imports"
//!                   --test libm_name_capture_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use libloading::{Library, Symbol};
use std::process::Command;

const SRC: &str = r#"
pub fn pow(x: f64, y: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    return 1.0;
}

pub fn sin(x: f64) -> f64 {
    return 0.125;
}

pub fn exp(x: f64) -> f64 {
    return 0.25;
}

pub fn probe_pow_literal() -> i64 {
    return __mind_f64_to_bits(pow(0.0 - 2.0, 0.5));
}

pub fn probe_sin_literal() -> i64 {
    return __mind_f64_to_bits(sin(1.0));
}

pub fn probe_exp_literal() -> i64 {
    return __mind_f64_to_bits(exp(1.0));
}

pub fn probe_pow_param(x: f64, y: f64) -> i64 {
    return __mind_f64_to_bits(pow(x, y));
}

pub fn probe_sin_param(x: f64) -> i64 {
    return __mind_f64_to_bits(sin(x));
}
"#;

const BITS_0_0: i64 = 0; // 0.0
const BITS_0_125: i64 = 0x3FC0_0000_0000_0000; // 0.125
const BITS_0_25: i64 = 0x3FD0_0000_0000_0000; // 0.25

#[test]
fn libm_named_mind_fns_run_their_own_body_at_literal_call_sites() {
    let mindc = common::require_mindc();
    let dir = tempfile::tempdir().expect("libm-capture scratch");
    let source = dir.path().join("probe.mind");
    let shared = dir.path().join("probe.so");
    std::fs::write(&source, SRC).expect("write probe source");

    // `--emit-mlir` is the inspection surface and stays the plain lowering
    // text (the self-host MLIR parity gate diffs it byte-for-byte); the
    // closed-world `nobuiltin` marking is applied by the build driver only.
    // Positive control first: the text really carries the libm-named fns.
    let mlir = Command::new(&mindc)
        .arg(&source)
        .arg("--emit-mlir")
        .output()
        .expect("run mindc --emit-mlir");
    assert!(mlir.status.success(), "--emit-mlir failed");
    let mlir_text = String::from_utf8_lossy(&mlir.stdout);
    assert!(
        mlir_text.contains("func.func @sin(") && mlir_text.contains("func.func @pow("),
        "positive control: --emit-mlir must contain the probe definitions:\n{mlir_text}"
    );
    assert!(
        !mlir_text.contains("nobuiltin"),
        "--emit-mlir must stay the plain lowering text (self-host parity)"
    );

    let out = Command::new(&mindc)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(&source)
        .arg("--emit-shared")
        .arg(&shared)
        .output()
        .expect("run mindc --emit-shared");
    if !common::gate::compiled("libm_name_capture_run", &out) {
        return;
    }
    assert!(shared.is_file(), "no shared object produced");

    unsafe {
        let lib = Library::new(&shared).expect("load probe shared library");
        let noarg = |name: &[u8]| -> i64 {
            let f: Symbol<unsafe extern "C" fn() -> i64> = lib.get(name).expect("probe symbol");
            f()
        };
        let pow_param: Symbol<unsafe extern "C" fn(f64, f64) -> i64> =
            lib.get(b"probe_pow_param").expect("probe_pow_param");
        let sin_param: Symbol<unsafe extern "C" fn(f64) -> i64> =
            lib.get(b"probe_sin_param").expect("probe_sin_param");

        // Positive control: with a run-time argument the MIND body runs.
        assert_eq!(pow_param(-2.0, 0.5), BITS_0_0, "pow body (run-time arg)");
        assert_eq!(sin_param(1.0), BITS_0_125, "sin body (run-time arg)");

        // The regression: a literal-argument call must run the SAME body, not
        // a host-libm constant fold (qNaN 0x7FF8..., sin(1) = 0x3FEAED54...,
        // e = 0x4005BF0A... before the fix).
        assert_eq!(
            noarg(b"probe_pow_literal"),
            BITS_0_0,
            "pow(-2, 0.5) literal call was folded to a host-libm value"
        );
        assert_eq!(
            noarg(b"probe_sin_literal"),
            BITS_0_125,
            "sin(1) literal call was folded to a host-libm value"
        );
        assert_eq!(
            noarg(b"probe_exp_literal"),
            BITS_0_25,
            "exp(1) literal call was folded to a host-libm value"
        );
    }
}
