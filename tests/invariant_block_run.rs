// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `invariant NAME { ... }` governance-declaration RUNTIME gate.
//!
//! An `invariant` block is a 512-mind governance/DIFC contract declaration (a
//! `description` plus one or more `check(...)` predicates). It produces NO
//! executable code: the native compiler accepts the whole `{ ... }` body as a
//! transparent marker (brace-balanced, string/comment-aware) and emits an empty
//! block, so a module that declares invariants still compiles to native code.
//! The keystone source declares no invariants, so its emit stays byte-identical.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test invariant_block_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
invariant evidence_per_edge {
    description: "trace length equals executed node count }"
    check(trace_len: u64, executed_nodes: u64): bool {
        // a `}` in a comment must not close the block
        return trace_len == executed_nodes
    }
}

pub fn run() -> i64 {
    return 42
}
"#;

fn mindc_bin() -> PathBuf {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d = m.join("target").join("debug").join("mindc");
    if d.exists() {
        d
    } else {
        m.join("target").join("release").join("mindc")
    }
}

#[test]
fn invariant_block_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("invariant-block-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_invariant_block_run.mind");
    let so = dir.join("mind_invariant_block_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("invariant-block-run: needs mlir-build; skipping");
            return;
        }
        panic!("invariant-block-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 42, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "invariant-block-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
