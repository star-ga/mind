//! #231 regression — f64 literal round-trip envelope (end-to-end, executable-witnessed).
//!
//! Each source literal is compiled via `mindc --emit-shared`; the exported
//! `pub fn f() -> f64` is dlopen-called and its exact IEEE-754 bits MUST equal the
//! bits of the same literal parsed by Rust. This catches any point in the pipeline
//! (parse -> IR -> MLIR literal render -> mlir-opt -> clang) that changes the value.
//!
//! NON-VACUOUS: on the pre-fix compiler the sub-EPSILON cases (`1e-16`, `2e-16`) are
//! destroyed to `0.0` by `format_number`'s `(n.fract()).abs() < f64::EPSILON`
//! misclassification, so this test FAILS before the fix and PASSES after it.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test f64_literal_envelope`
#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;
use std::process::Command;

/// Source literals spanning the envelope. Expected value = Rust's parse of the SAME
/// string, so the test asserts the compiled artifact preserves the intended IEEE-754
/// bits. Every literal here is in MIND's positional-decimal grammar (dot present).
const CASES: &[&str] = &[
    // --- sub-EPSILON magnitudes (destroyed by the pre-fix compiler) ---
    "0.0000000000000001", // 1e-16  (< f64::EPSILON)
    "0.0000000000000002", // 2e-16  (< f64::EPSILON)
    "0.0000000000000003", // 3e-16  (> f64::EPSILON, but adjacent)
    "0.0000000000000005", // 5e-16  (> f64::EPSILON)
    // --- values around / below the EPSILON boundary and small normals ---
    "0.0001",
    "0.00000001", // 1e-8
    "0.0000001",  // 1e-7
    // --- exactly f64::EPSILON (2.220446049250313e-16) — the old boundary ---
    "0.0000000000000002220446049250313",
    // --- ordinary decimal fractions (shortest-vs-exact divergence classes) ---
    "0.1",
    "0.2",
    "0.3",
    "0.25", // exact (power-of-two denominator)
    "0.5",
    "1.5",
    "2.0",
    "3.14159",
    "2.718281828459045",
    // --- exact integer-valued f64 (must print WITH a decimal point) ---
    "0.0",
    "1.0",
    "5.0",
    "100.0",
    "1000000.0",
    "123456789.0",
    "9007199254740992.0", // 2^53 (exact)
];

#[test]
fn f64_literal_envelope_roundtrip() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        eprintln!("f64-literal-envelope: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let mut failures = Vec::new();

    for (i, src) in CASES.iter().enumerate() {
        let expected: f64 = src.parse().expect("test literal must parse in Rust");
        let mp = dir.join(format!("mind_f64env_{i}.mind"));
        let sp = dir.join(format!("mind_f64env_{i}.so"));
        std::fs::write(&mp, format!("pub fn f() -> f64 {{\n    {src}\n}}\n")).expect("write src");

        let out = Command::new(&mindc)
            .args([mp.to_str().unwrap(), "--emit-shared", sp.to_str().unwrap()])
            .output()
            .expect("run mindc --emit-shared");
        if !out.status.success() {
            let e = String::from_utf8_lossy(&out.stderr);
            if e.contains("mlir-build") && e.contains("requires") {
                eprintln!("f64-literal-envelope: mindc --emit-shared needs mlir-build; skipping");
                return;
            }
            failures.push(format!("[{src}] build FAILED: {e}"));
            continue;
        }

        // dlopen + call `f() -> f64`, read the exact returned bits.
        let got: f64 = unsafe {
            let lib = libloading::Library::new(&sp).expect("dlopen emitted .so");
            let f: libloading::Symbol<unsafe extern "C" fn() -> f64> =
                lib.get(b"f").expect("symbol f");
            f()
        };

        if got.to_bits() != expected.to_bits() {
            failures.push(format!(
                "[{src}] executable f() returned bits {:#018x} ({got:?}), want {:#018x} ({expected:?})",
                got.to_bits(),
                expected.to_bits()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "f64 literal envelope round-trip miscompiles (#231):\n{}",
        failures.join("\n")
    );
}
