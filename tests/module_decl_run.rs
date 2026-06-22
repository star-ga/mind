// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Dotted file-level `module a.b.c` declaration RUNTIME gate.
//!
//! A braceless file-level module header with a DOTTED path
//! (`module backends.tool`) is a transparent marker — the file IS the module.
//! The parser must consume the full `.segment` path (it previously stopped at
//! the first `.`, leaving `.tool` for the next item parse → "expected
//! expression"). This shape is pervasive in real multi-module programs.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test module_decl_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
module backends.tool.inner

fn add(a: i64, b: i64) -> i64 {
    return a + b
}

pub fn run() -> i64 {
    return add(40, 2)
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
fn module_decl_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("module-decl-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_module_decl_run.mind");
    let so = dir.join("mind_module_decl_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("module-decl-run: needs mlir-build; skipping");
            return;
        }
        panic!("module-decl-run: mindc --emit-shared failed:\n{stderr}");
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
        "module-decl-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
