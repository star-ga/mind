// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct-variant enum RUNTIME gate — declaration, construction, named match.
//!
//! A struct variant `enum E { V { f: T, g: U } }` lowers to the SAME boxed heap
//! record as a tuple variant `V(T, U)` — `[tag @ +0, f @ +8, g @ +16]` — with
//! the field NAMES recorded so a construction `E.V { g: y, f: x }` and a match
//! `E.V { f, g }` resolve each name to its declared positional slot (independent
//! of the order written). A `{ field, .. }` rest pattern binds the listed fields
//! and ignores the rest. This compiles a program to a `.so`, dlopen-calls it, and
//! asserts the values — including deliberately REORDERED construction/match
//! fields, so a name->slot mapping bug cannot pass.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test enum_struct_variant_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
enum Shape {
    Circle { r: i64 },
    Rect { w: i64, h: i64 }
}

// Construct + match by name, fields in declared order.
pub fn area_rect() -> i64 {
    let s = Shape.Rect { w: 6, h: 7 }
    match s {
        Shape.Circle { r } => 0,
        Shape.Rect { w, h } => w * h,
    }
}

pub fn circle_radius() -> i64 {
    let s = Shape.Circle { r: 42 }
    match s {
        Shape.Circle { r } => r,
        Shape.Rect { w, h } => 0,
    }
}

// Construction fields written in REVERSE order — `w` must still land in slot 0.
pub fn ctor_reorder() -> i64 {
    let s = Shape.Rect { h: 7, w: 6 }
    match s {
        Shape.Circle { r } => 0,
        Shape.Rect { w, h } => w * 100 + h,
    }
}

// MATCH fields written in REVERSE order — `w`/`h` must bind by NAME, not text
// position.
pub fn match_reorder() -> i64 {
    let s = Shape.Rect { w: 6, h: 7 }
    match s {
        Shape.Circle { r } => 0,
        Shape.Rect { h, w } => w * 100 + h,
    }
}

enum Tagged {
    A { v: i64, tag: i64 },
    B { x: i64 }
}

// `{ field, .. }` rest pattern — bind `v`, ignore `tag`.
pub fn rest_pattern() -> i64 {
    let t = Tagged.A { v: 9, tag: 5 }
    match t {
        Tagged.A { v, .. } => v,
        Tagged.B { .. } => 0,
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
fn enum_struct_variant_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("enum-struct-variant-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_enum_struct_variant_run.mind");
    let so = dir.join("mind_enum_struct_variant_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("enum-struct-variant-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("enum-struct-variant-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('area_rect','circle_radius','ctor_reorder','match_reorder','rest_pattern'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.area_rect(); assert r == 42, 'area_rect=' + str(r)\n\
         r = lib.circle_radius(); assert r == 42, 'circle_radius=' + str(r)\n\
         r = lib.ctor_reorder(); assert r == 607, 'ctor_reorder=' + str(r)\n\
         r = lib.match_reorder(); assert r == 607, 'match_reorder=' + str(r)\n\
         r = lib.rest_pattern(); assert r == 9, 'rest_pattern=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "enum-struct-variant-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
