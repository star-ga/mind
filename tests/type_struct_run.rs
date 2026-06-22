// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `type X { … }` record-definition RUNTIME gate.
//!
//! A record type defined with the `type` keyword and a brace body
//! (`type TernaryMatrix { rows: u32, … }`) is a struct definition, not a
//! `type X = Y` alias. The parser now routes `type NAME {` through the shared
//! struct-body parser (parse_struct_body) and produces a `Node::StructDef`.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test type_struct_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
type Point { x: i64, y: i64 }

fn sum(p: Point) -> i64 {
    return p.x + p.y
}

pub fn run() -> i64 {
    return sum(Point { x: 40, y: 2 })
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
fn type_struct_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("type-struct-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_type_struct_run.mind");
    let so = dir.join("mind_type_struct_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("type-struct-run: needs mlir-build; skipping");
            return;
        }
        panic!("type-struct-run: mindc --emit-shared failed:\n{stderr}");
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
        "type-struct-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
