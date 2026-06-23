// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Standalone `mindc file.mind --emit-shared` std-import resolution gate.
//!
//! REGRESSION for the reported `import std.map` / `import std.json` asymmetry.
//!
//! The reported bug — "standalone `mindc file.mind` resolving `import std.map`
//! gives `E2003 unsupported call to map_new`" — is the stale/wrong-FEATURE
//! binary footgun, NOT a code divergence between the test harness and the
//! standalone single-file CLI. Std-call resolution flows through the
//! UNCONDITIONAL std-surface name extend in
//! `type_checker::resolve::collect_module_syms` (it extends the resolvable set
//! with the whole bundled `stdlib_exports()` surface), which is populated iff
//! the binary was built with `std-surface` or `cross-module-imports`. With
//! either feature on, a single-file `import std.map` resolves `map_new` /
//! `map_insert` / `map_get` standalone and the `.so` runs; with NEITHER feature
//! a featureless binary's std surface is empty and every std call is `E2003`.
//!
//! This test shells out to `CARGO_BIN_EXE_mindc` — the binary built for THIS
//! test invocation's exact feature set (staleness-free, issue #42) — and proves
//! the GREEN side end-to-end: a standalone single-file compile of `import
//! std.map` + `map_new`/`map_insert`/`map_get`/`map_contains_key` resolves
//! (no E2003), emits a `.so`, and the dlopen-called value is correct. It is the
//! exact RED-pre (featureless binary → E2003) / GREEN-post (prescribed-feature
//! binary → resolves) regression the std-import resolution fix is gated on.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test std_import_standalone_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    any(feature = "std-surface", feature = "cross-module-imports")
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// Single file, single `import std.map`, called the way a real user invokes the
// standalone CLI: `mindc file.mind --emit-shared out.so`. No project, no
// Mind.toml, no STD_ROOT, no --std-root — std resolution is entirely the
// bundled-surface name extend. get(9)=900 + insert chain + contains(7)=1 +
// contains(5)=0 + get(3)=300 = 1201.
const SRC: &str = r#"
import std.map

pub fn lookups() -> i64 {
    let m = map_new()
    let m = map_insert(m, 7, 700)
    let m = map_insert(m, 9, 900)
    let m = map_insert(m, 3, 300)
    return map_get(m, 9) + map_contains_key(m, 7) + map_contains_key(m, 5) + map_get(m, 3)
}
"#;

#[test]
fn std_import_map_resolves_standalone_and_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("std-import-standalone-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_std_import_standalone.mind");
    let so = dir.join("mind_std_import_standalone.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("std-import-standalone-run: needs mlir-build; skipping");
            return;
        }
        // The exact failure this test guards against: a featureless binary
        // sprays `E2003 unsupported call to map_new` (one per std call) because
        // its bundled std surface is empty. With the prescribed features this
        // path must NOT be hit — surface the stderr so a regression is obvious.
        panic!(
            "std-import-standalone-run: `import std.map` failed to resolve \
             standalone (this is the E2003 stale-binary regression):\n{stderr}"
        );
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.lookups.restype = ctypes.c_int64\n\
         r = lib.lookups(); assert r == 1201, 'lookups=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "std-import-standalone-run ctypes check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
