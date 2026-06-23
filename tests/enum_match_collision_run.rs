// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-enum variant-name COLLISION `match` RUNTIME gate (silent miscompile).
//!
//! A BARE variant pattern in a `match` arm (`Foo(v)`) carries no enum
//! qualification, so the desugar must resolve it to the SCRUTINEE's enum. The
//! prior code resolved each bare arm INDEPENDENTLY to the first lexicographic
//! `Enum::V` in the registry. When a variant name was declared by two enums
//! (`Foo` in both `Alpha` and `Zeta`), a `Zeta` scrutinee tested its tag against
//! `Alpha::Foo`'s ordinal — a WRONG-ARM silent miscompile that compiled and ran
//! cleanly (exit 0) while returning the wrong value:
//!
//!   enum Zeta  { Foo(i64), Bar(i64) }   // Foo=0, Bar=1
//!   enum Alpha { Bar(i64), Foo(i64) }   // Bar=0, Foo=1  (Alpha sorts first)
//!   match z { Foo(v) => 1000 + v, Zeta::Bar(v) => 2000 + v }
//!
//! Pre-fix: `gz(Zeta::Foo(7))` → 0 (tag 0 matched neither Alpha::Foo's tag 1 nor
//! Zeta::Bar's tag 1), `gz(Zeta::Bar(3))` → 1003 (tag 1 == Alpha::Foo tag 1 →
//! took the wrong `Foo` arm). Both WRONG; the fix returns 1007 / 2003.
//!
//! The fix anchors bare-variant resolution to the match's OWNING enum: a
//! qualified arm (`Zeta::Bar`) pins the enum, so every bare arm (`Foo`) resolves
//! within it. `main.mind` uses no enums, so the keystone is untouched.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test enum_match_collision_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

// `Alpha` sorts lexicographically before `Zeta`, so `Alpha::Foo` (tag 1) is the
// FIRST registry hit for a bare `Foo` — the wrong enum for a `Zeta` scrutinee.
// One arm (`Zeta::Bar`) is qualified, pinning the owning enum to `Zeta`.
const SRC: &str = r#"
enum Zeta {
    Foo(i64),
    Bar(i64),
}

enum Alpha {
    Bar(i64),
    Foo(i64),
}

fn gz(z: Zeta) -> i64 {
    match z {
        Foo(v) => 1000 + v,
        Zeta::Bar(v) => 2000 + v,
    }
}

pub fn t_z_foo() -> i64 { gz(Zeta::Foo(7)) }
pub fn t_z_bar() -> i64 { gz(Zeta::Bar(3)) }
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
fn cross_enum_bare_variant_collision_takes_correct_arm() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("enum-match-collision-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_enum_collision_run.mind");
    let so = dir.join("mind_enum_collision_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("enum-match-collision-run: needs mlir-build; skipping");
            return;
        }
        panic!("enum-match-collision-run: mindc --emit-shared failed:\n{stderr}");
    }

    // Flat top-level Python only (Rust `\`-continuation strips leading
    // whitespace, so no indented blocks). The asserts pin the CORRECT arm.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('t_z_foo','t_z_bar'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.t_z_foo(); assert r == 1007, 't_z_foo=' + str(r)\n\
         r = lib.t_z_bar(); assert r == 2003, 't_z_bar=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "enum-match-collision-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
