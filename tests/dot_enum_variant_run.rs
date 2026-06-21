// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Dot enum-variant access RUNTIME gate (`Enum.Variant`).
//!
//! MIND canonically writes a variant `Enum::Variant`, but Rust-ish code (e.g.
//! mind-flow's parser) uses the DOT form `Enum.Variant` — both as a value
//! (`Color.Red`), as a payload constructor (`Expr.IntLit(x)`), and in a pattern
//! (`Color.Red => …`). The parser tracks declared enum names and normalises a
//! dotted reference whose first segment is an enum to the canonical `::` form; a
//! non-enum receiver stays a struct field access. This compiles a program using
//! the dot form to a `.so`, dlopen-calls it, and asserts the values.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test dot_enum_variant_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
enum Color { Red, Green, Blue }

// Dot value `Color.Green` + dot patterns `Color.Red`/`Color.Green`/`Color.Blue`.
pub fn pick() -> i64 {
    let c = Color.Green
    match c {
        Color.Red => 1,
        Color.Green => 2,
        Color.Blue => 3,
    }
}

enum Expr { IntLit(i64), Nothing }

// Dot PAYLOAD constructor `Expr.IntLit(42)` + dot payload pattern.
pub fn lit() -> i64 {
    let e = Expr.IntLit(42)
    match e {
        Expr.IntLit(v) => v,
        Expr.Nothing => 0,
    }
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
fn dot_enum_variant_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("dot-enum-variant-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_dot_enum_variant_run.mind");
    let so = dir.join("mind_dot_enum_variant_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("dot-enum-variant-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("dot-enum-variant-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('pick','lit'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.pick(); assert r == 2, 'pick=' + str(r)\n\
         r = lib.lit(); assert r == 42, 'lit=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "dot-enum-variant-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
