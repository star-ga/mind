// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Phase 17.9 — `const [f64; N]` EXECUTABLE lowering + const-dense name-routing
//! regression gate (`CONST_DENSE_NAME_ROUTING` / `STALE_MODULE_SSA_BYPASS`).
//!
//! A named `const W: [f64; N] = [...]` lowers to a TYPED dense blob
//! (`const_dense_defs` registry + `Instr::ConstDenseTensor`, carrying the
//! element `DType` + per-element exact IEEE-754 bits), NOT the i64-only
//! `ConstArray` path (which would silently zero every float element).
//!
//! Permanent regression: the fn-env builder (`src/eval/lower.rs`) filtered out
//! `const_array_defs` names — whose module SSA ids are invalid in a fn body —
//! but did NOT filter `const_dense_defs` names, so a const-f64 symbol inherited
//! its stale MODULE SSA id and the following `ArrayLoad` base was untyped
//! ("missing type information ... array load base"). This compiles the construct
//! to a real `.so`, dlopen-calls it, and asserts EXACT 64-bit IEEE-754 patterns
//! (incl. `-0.0` sign preservation and a runtime/dynamic index) so no
//! stale-SSA / i64-reinterpretation route can pass. IR-shape-only tests miss it
//! (they never query the lowering type map); this runs the emitted code.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test const_f64_array_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::process::Command;

const SRC: &str = r#"
const W: [f64; 4] = [1.5, -0.0, 3.5, 2.5];

pub fn w0_bits() -> i64 {
    return __mind_f64_to_bits(W[0]);
}
pub fn w1_bits() -> i64 {
    return __mind_f64_to_bits(W[1]);
}
pub fn w3_bits() -> i64 {
    return __mind_f64_to_bits(W[3]);
}

// Dynamic (loop-computed) index into a const f64 LUT: read W[k] with a runtime
// `k`, returning its exact bits — exercises the ArrayLoad base typing under a
// non-constant subscript, the shape that first exposed the stale-SSA bug.
pub fn w_dyn_bits(k: i64) -> i64 {
    let mut j: i64 = 0;
    let mut acc: i64 = 0;
    while j <= k {
        acc = __mind_f64_to_bits(W[j]);
        j = j + 1;
    }
    return acc;
}
"#;

#[test]
fn const_f64_array_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("const-f64-array-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_const_f64_array_run.mind");
    let so = dir.join("mind_const_f64_array_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("const-f64-array-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("const-f64-array-run: mindc --emit-shared failed:\n{stderr}");
    }

    // Expected bits are computed via `struct` (never hand-derived) so the check
    // is on the true binary64 patterns; `-0.0` must keep its sign bit.
    let py = format!(
        "import ctypes, struct\n\
         b = lambda v: struct.unpack('<q', struct.pack('<d', v))[0]\n\
         lib = ctypes.CDLL(r'{}')\n\
         for n in ('w0_bits','w1_bits','w3_bits'):\n\
         \x20   f = getattr(lib, n); f.restype = ctypes.c_int64; f.argtypes = []\n\
         lib.w_dyn_bits.restype = ctypes.c_int64\n\
         lib.w_dyn_bits.argtypes = [ctypes.c_int64]\n\
         assert lib.w0_bits() == b(1.5), 'W0=' + repr(lib.w0_bits())\n\
         assert lib.w1_bits() == b(-0.0), 'W1(-0.0)=' + repr(lib.w1_bits())\n\
         assert lib.w1_bits() != b(0.0), 'W1 lost -0.0 sign'\n\
         assert lib.w3_bits() == b(2.5), 'W3=' + repr(lib.w3_bits())\n\
         assert lib.w_dyn_bits(2) == b(3.5), 'Wdyn2=' + repr(lib.w_dyn_bits(2))\n\
         assert lib.w_dyn_bits(1) == b(-0.0), 'Wdyn1=' + repr(lib.w_dyn_bits(1))\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "const-f64-array-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
