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

/// A program in ANOTHER module than the callee: the caller sees only a
/// declaration of each `std.detmath` function. `opaque` sends its argument
/// through the heap, so the optimizer cannot fold that call and it is the
/// positive control: the value the std body returns at run time.
#[cfg(feature = "cross-module-imports")]
const CROSS_MODULE_SRC: &str = r#"
import std.detmath;

fn opaque(x: f64) -> f64 {
    let mut v: Vec = vec_new();
    v = vec_push(v, __mind_f64_to_bits(x));
    return __mind_bits_to_f64(vec_get(v, 0));
}

fn same(a: f64, b: f64) -> i64 {
    if __mind_f64_to_bits(a) == __mind_f64_to_bits(b) {
        return 1;
    }
    return 0;
}

fn main() -> i64 {
    let s: i64 = same(sin(1.0e22), sin(opaque(1.0e22)));
    let c: i64 = same(cos(1.0e22), cos(opaque(1.0e22)));
    let e: i64 = same(exp(1.0), exp(opaque(1.0)));
    let l: i64 = same(log(10.0), log(opaque(10.0)));
    let p: i64 = same(pow(0.0 - 2.0, 0.5), pow(opaque(0.0 - 2.0), opaque(0.5)));
    let q: i64 = same(pow(2.0, 0.5), pow(opaque(2.0), opaque(0.5)));
    let z: i64 = if pow(0.0 - 2.0, 0.5) == 0.0 { 1 } else { 0 };
    let y: i64 = if pow(opaque(0.0 - 2.0), opaque(0.5)) == 0.0 { 1 } else { 0 };
    return s + c * 2 + e * 4 + l * 8 + p * 16 + q * 32 + z * 64 + y * 128;
}
"#;

/// MIND functions that have the name of a C library function the runtime
/// support object calls. Each returns a value no C function of that name
/// returns, and does no work.
const C_NAME_SRC: &str = r#"
fn write(fd: i64, buf: i64, n: i64) -> i64 {
    return n + 100;
}

fn memcpy(d: i64, s: i64, n: i64) -> i64 {
    return d + 1;
}

fn main() -> i64 {
    let mut v: Vec = vec_new();
    let mut i: i64 = 0;
    while i < 100 {
        v = vec_push(v, i + 1);
        i = i + 1;
    }
    print("answer", 42);
    let own: i64 = (write(1, 0, 5) - 105) + (memcpy(8, 0, 0) - 9);
    return vec_get(v, 0) + vec_get(v, 1) + vec_get(v, 2) + own;
}
"#;

/// Build `text` as an executable with `mindc build` and run it.
fn build_and_run(target: &str, text: &str) -> Option<std::process::Output> {
    let mindc = common::require_mindc();
    let dir = tempfile::tempdir().expect("scratch");
    let source = common::scratch_src_in(dir.path(), "main.mind", text);
    let exe = dir.path().join("probe_exe");
    let built = common::run_build_captured(
        &mindc,
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
        &[
            source.to_str().expect("utf-8 path"),
            "--out",
            exe.to_str().expect("utf-8 path"),
        ],
    );
    if !common::gate::compiled(target, &built) {
        return None;
    }
    assert!(exe.is_file(), "no executable produced");
    Some(Command::new(&exe).output().expect("run the probe"))
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn std_detmath_literal_calls_run_the_std_body_across_modules() {
    let Some(ran) = build_and_run("libm_name_capture_cross_module", CROSS_MODULE_SRC) else {
        return;
    };
    // Bit 128 is the positive control (the std body returns 0.0 for
    // `pow(-2, 0.5)` at run time). Bits 1..64: each literal-argument call has
    // the bits of the run-time call, and `pow(-2, 0.5)` is 0.0, not a NaN.
    assert_eq!(
        ran.status.code(),
        Some(255),
        "a cross-module literal call to std.detmath did not run the std body \
         (bits: sin 1, cos 2, exp 4, log 8, pow(-2,.5) 16, pow(2,.5) 32, \
         pow(-2,.5)==0.0 64, control 128)"
    );
}

#[test]
fn mind_fns_with_c_library_names_do_not_capture_the_c_functions() {
    let Some(ran) = build_and_run("libm_name_capture_c_names", C_NAME_SRC) else {
        return;
    };
    // The runtime support object calls the C `write` (print) and the C
    // `memcpy` (vector growth). It must get the C functions. `print` ends
    // the line, so the output is the two values and one newline.
    assert_eq!(
        String::from_utf8_lossy(&ran.stdout),
        "answer42\n",
        "the runtime's `write` call ran the MIND fn named `write`"
    );
    // 1 + 2 + 3 from the grown vector, and 0 from the MIND fns' own bodies.
    assert_eq!(
        ran.status.code(),
        Some(6),
        "the runtime's `memcpy` call ran the MIND fn named `memcpy`, or the \
         MIND fns did not run their own bodies"
    );
}

#[cfg(feature = "cross-module-imports")]
const SHARED_CROSS_MODULE_SRC: &str = r#"
import std.detmath;

pub fn probe_pow_literal() -> i64 {
    return __mind_f64_to_bits(pow(0.0 - 2.0, 0.5));
}

pub fn probe_sin_literal() -> i64 {
    return __mind_f64_to_bits(sin(1.0e22));
}

pub fn probe_pow_param(x: f64, y: f64) -> i64 {
    return __mind_f64_to_bits(pow(x, y));
}

pub fn probe_sin_param(x: f64) -> i64 {
    return __mind_f64_to_bits(sin(x));
}
"#;

#[cfg(feature = "cross-module-imports")]
#[test]
fn std_detmath_literal_calls_run_the_std_body_in_a_shared_library() {
    let mindc = common::require_mindc();
    let dir = tempfile::tempdir().expect("scratch");
    let source = common::scratch_src_in(dir.path(), "probe.mind", SHARED_CROSS_MODULE_SRC);
    let shared = dir.path().join("probe.so");
    let out = Command::new(&mindc)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(&source)
        .arg("--emit-shared")
        .arg(&shared)
        .output()
        .expect("run mindc --emit-shared");
    if !common::gate::compiled("libm_name_capture_shared_cross_module", &out) {
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
        // Positive control: the std body, reached with run-time arguments.
        assert_eq!(
            pow_param(-2.0, 0.5),
            BITS_0_0,
            "std pow body (run-time arg)"
        );
        assert_eq!(
            noarg(b"probe_pow_literal"),
            BITS_0_0,
            "cross-module pow(-2, 0.5) literal call was folded to a host-libm value"
        );
        assert_eq!(
            noarg(b"probe_sin_literal"),
            sin_param(1.0e22),
            "cross-module sin(1e22) literal call was folded to a host-libm value"
        );
    }
}
