// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Module-level `const` RUNTIME gate — scalar, collection, and const-references-
//! const inlining.
//!
//! A top-level `const NAME = value` is inlined at each `Lit(Ident(NAME))` use
//! site (a lowering-time substitution table that never serialises into mic@3, so
//! the keystone — which declares no module consts — stays byte-identical). This
//! exercises a scalar const, a `map<K,V>` const reached through `map_get`, and a
//! const defined in terms of another const (`DOUBLE_LIMIT = LIMIT + LIMIT`).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test module_const_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
const LIMIT: i64 = 42
const DOUBLE_LIMIT: i64 = LIMIT + LIMIT
const M: map<i64, i64> = {1: 10, 2: 20, 3: 30}

fn lookup(k: i64) -> i64 {
    return map_get(M, k)
}

fn use_local_shadow(LIMIT: i64) -> i64 {
    // A param named like the const must shadow it (env is checked first).
    return LIMIT
}

pub fn run() -> i64 {
    let a = lookup(1)          // 10
    let b = lookup(3)          // 30
    let s = use_local_shadow(7) // 7, not 42
    return a + b + LIMIT + DOUBLE_LIMIT + s
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
fn module_const_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("module-const-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_module_const_run.mind");
    let so = dir.join("mind_module_const_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("module-const-run: needs mlir-build; skipping");
            return;
        }
        panic!("module-const-run: mindc --emit-shared failed:\n{stderr}");
    }

    // a=10, b=30, LIMIT=42, DOUBLE_LIMIT=84, s=7 → 10+30+42+84+7 = 173.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 173, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "module-const-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
