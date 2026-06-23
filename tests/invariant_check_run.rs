// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `invariant NAME { check(...) { ... } }` callable-predicate RUNTIME gate.
//!
//! An `invariant` block's `check(...)` predicate is now lowered to a free
//! function `<invariant>_<predicate>` (e.g. `my_inv_check`), and a call of the
//! shape `<invariant>.check(args)` desugars to it — so governance invariants
//! whose predicates are actually invoked at runtime compile to native code
//! instead of hitting the #306 fail-closed unresolved-method panic. The keystone
//! source declares no invariants, so its emit stays byte-identical.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test invariant_check_run`

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
invariant my_inv {
    description: "n must be zero"
    check(n: u64): bool {
        return n == 0
    }
}

pub fn run() -> i64 {
    if my_inv.check(0) {
        return 1
    }
    return 0
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn invariant_check_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("invariant-check-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_invariant_check_run.mind");
    let so = dir.join("mind_invariant_check_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("invariant-check-run: needs mlir-build; skipping");
            return;
        }
        panic!("invariant-check-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 1, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "invariant-check-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
