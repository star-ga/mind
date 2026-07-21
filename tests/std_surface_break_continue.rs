// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `break` / `continue` loop control — SSA-correct codegen.
//!
//! `break`/`continue` were documented-unimplemented: they parsed as bare
//! identifiers and silently lowered to `const.i64 0` (a no-op miscompile —
//! e.g. `std/cli.mind`'s `continue` fell through instead of restarting the
//! loop). They now lower to a `cf.br` to the innermost enclosing loop's
//! `^while_after` (break) / `^while_header` (continue), forwarding each
//! loop-carried variable's CURRENT mid-iteration value (the silent-miscompile
//! risk — using the back-edge's post-body value would be an SSA dominance
//! error). This test is the ground truth: compile to a `.so`, dlopen, and
//! check loop-carried correctness across `continue`, `break`, body-level
//! break, and nested loops (innermost scoping).

#![cfg(feature = "std-surface")]

mod common;

use libmind::ir::compact::emit_mic3;
use libmind::{CompileOptions, compile_source};

#[test]
fn break_continue_lower_deterministically() {
    let src = "fn f(n: i64) -> i64 { let mut i = 0; let mut s = 0; \
               while i < n { i = i + 1; if (i % 2) == 0 { continue } s = s + i } s }\n\
               let y = f(5)\ny";
    let a = compile_source(src, &CompileOptions::default()).expect("break/continue compiles");
    let b = compile_source(src, &CompileOptions::default()).expect("break/continue compiles");
    assert_eq!(
        emit_mic3(&a.ir),
        emit_mic3(&b.ir),
        "break/continue lowering must be deterministic (the wedge)"
    );
}

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    /// Compile break/continue programs to a `.so` and check loop-carried
    /// correctness — the SSA proof that mid-iteration carried values are
    /// forwarded correctly.
    #[test]
    fn break_continue_round_trip_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("break_continue_round_trip: mindc not found; skipping");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_break_continue");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let src = r#"
pub fn sum_skip_evens(n: i64) -> i64 {
    let mut i = 0;
    let mut s = 0;
    while i < n { i = i + 1; if (i % 2) == 0 { continue } s = s + i; }
    s
}
pub fn sum_until(n: i64, cap: i64) -> i64 {
    let mut i = 0;
    let mut s = 0;
    while i < n { i = i + 1; s = s + i; if s >= cap { break } }
    s
}
pub fn first_div(n: i64, d: i64) -> i64 {
    let mut i = 1;
    while i < n { if (i % d) == 0 { break } i = i + 1; }
    i
}
pub fn nested(a: i64, b: i64) -> i64 {
    let mut i = 0;
    let mut acc = 0;
    while i < a {
        i = i + 1;
        let mut j = 0;
        while j < b { j = j + 1; if j == 2 { continue } acc = acc + 1; }
    }
    acc
}
"#;
        let driver = out_dir.join("bc.mind");
        let so = out_dir.join("libbc.so");
        std::fs::write(&driver, src).expect("write driver");

        let status = Command::new(&mindc)
            .args([
                driver.to_str().unwrap(),
                "--emit-shared",
                so.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");
        assert!(
            status.success(),
            "mindc must compile break/continue programs"
        );

        unsafe {
            let lib = libloading::Library::new(&so).expect("dlopen libbc.so");
            type F1 = unsafe extern "C" fn(i64) -> i64;
            type F2 = unsafe extern "C" fn(i64, i64) -> i64;
            let f1 = |name: &[u8], x: i64| -> i64 {
                let f: libloading::Symbol<F1> = lib.get(name).unwrap();
                f(x)
            };
            let f2 = |name: &[u8], x: i64, y: i64| -> i64 {
                let f: libloading::Symbol<F2> = lib.get(name).unwrap();
                f(x, y)
            };
            // continue skips even i: sum of odds <= n
            assert_eq!(f1(b"sum_skip_evens\0", 5), 9); // 1+3+5
            assert_eq!(f1(b"sum_skip_evens\0", 7), 16); // 1+3+5+7
            assert_eq!(f1(b"sum_skip_evens\0", 10), 25); // 1+3+5+7+9
            assert_eq!(f1(b"sum_skip_evens\0", 0), 0);
            // break once running sum hits the cap
            assert_eq!(f2(b"sum_until\0", 100, 10), 10); // 1+2+3+4=10
            assert_eq!(f2(b"sum_until\0", 100, 7), 10); // hits cap at 10
            assert_eq!(f2(b"sum_until\0", 3, 100), 6); // never hits cap: 1+2+3
            // break returns the loop variable
            assert_eq!(f2(b"first_div\0", 20, 5), 5);
            assert_eq!(f2(b"first_div\0", 20, 7), 7);
            // nested: continue scopes to the INNER loop only
            assert_eq!(f2(b"nested\0", 3, 4), 9); // 3 * (4-1 skipped)
            assert_eq!(f2(b"nested\0", 2, 1), 2);
        }
    }
}
