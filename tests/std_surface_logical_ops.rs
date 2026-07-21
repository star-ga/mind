// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Logical `&&` / `||` lowering — short-circuit desugar to `Node::If`.
//!
//! `&&`/`||` parse to a dedicated `Node::Logical` that had NO arm in
//! `lower_expr`, so it fell through the master catch-all and silently lowered
//! to `const.i64 0` (a release-silent miscompile: `x > 0 && x < 10` always
//! returned 0). The fix desugars to the existing, keystone-stable `Node::If`
//! lowering with literal 0/1 branch results, matching the interpreter's
//! short-circuit semantics.
//!
//! This covers determinism (the wedge) cheaply, plus a real functional
//! `.so` round-trip (gated on `mlir-build`) that proves correct runtime values
//! across chaining, precedence, and short-circuit.

#![cfg(feature = "std-surface")]

mod common;

use libmind::ir::compact::emit_mic3;
use libmind::{CompileOptions, compile_source};

#[test]
fn logical_ops_lower_deterministically() {
    // `&&` and `||`, chained and mixed with comparison precedence.
    let src = "fn mid(x: i64) -> i64 { if x > 0 && x < 10 { 1 } else { 0 } }\n\
               fn either(x: i64) -> i64 { if x < 0 || x > 100 { 1 } else { 0 } }\n\
               let y = mid(5)\ny";

    let a = compile_source(src, &CompileOptions::default())
        .expect("a && / || program should compile (not silently collapse)");
    let b = compile_source(src, &CompileOptions::default())
        .expect("a && / || program should compile (not silently collapse)");

    // The wedge: identical source -> byte-identical mic@3.
    assert_eq!(
        emit_mic3(&a.ir),
        emit_mic3(&b.ir),
        "logical-op (desugared-to-If) lowering must be deterministic"
    );
}

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    /// Compile `&&`/`||` programs to a `.so`, dlopen, and check the runtime
    /// values — the ground-truth that the desugar executes correctly (and is
    /// NOT the old silent const-0 miscompile).
    #[test]
    fn logical_ops_round_trip_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("logical_ops_round_trip: mindc not found; skipping");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_logical_ops");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let src = r#"
pub fn r_and(x: i64) -> i64 { if x > 0 && x < 10 { 1 } else { 0 } }
pub fn r_or(x: i64) -> i64 { if x < 0 || x > 100 { 1 } else { 0 } }
pub fn r_chain(x: i64) -> i64 { if x > 0 && x < 100 && x != 50 { 1 } else { 0 } }
pub fn r_mixed(x: i64) -> i64 { if x == 0 || x > 5 && x < 8 { 1 } else { 0 } }
"#;
        let driver = out_dir.join("logical_ops.mind");
        let so = out_dir.join("liblogical_ops.so");
        std::fs::write(&driver, src).expect("write driver");

        let status = Command::new(&mindc)
            .args([
                driver.to_str().unwrap(),
                "--emit-shared",
                so.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        assert!(status.success(), "mindc must compile && / || programs");

        unsafe {
            let lib = libloading::Library::new(&so).expect("dlopen liblogical_ops.so");
            type F = unsafe extern "C" fn(i64) -> i64;
            let call = |name: &[u8], x: i64| -> i64 {
                let f: libloading::Symbol<F> = lib.get(name).unwrap();
                f(x)
            };
            // r_and: 0 < x < 10
            assert_eq!(call(b"r_and\0", 5), 1);
            assert_eq!(call(b"r_and\0", 0), 0);
            assert_eq!(call(b"r_and\0", 10), 0);
            assert_eq!(call(b"r_and\0", -1), 0);
            // r_or: x < 0 || x > 100
            assert_eq!(call(b"r_or\0", -5), 1);
            assert_eq!(call(b"r_or\0", 50), 0);
            assert_eq!(call(b"r_or\0", 200), 1);
            // r_chain: 0 < x < 100 && x != 50
            assert_eq!(call(b"r_chain\0", 49), 1);
            assert_eq!(call(b"r_chain\0", 50), 0);
            assert_eq!(call(b"r_chain\0", 100), 0);
            // r_mixed: x == 0 || (x > 5 && x < 8)  (precedence: && binds tighter)
            assert_eq!(call(b"r_mixed\0", 0), 1);
            assert_eq!(call(b"r_mixed\0", 6), 1);
            assert_eq!(call(b"r_mixed\0", 8), 0);
            assert_eq!(call(b"r_mixed\0", 5), 0);
        }
    }
}
