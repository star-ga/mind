// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Array handles stay valid across `push` (the "pointer invalidated by
//! push_back" class of bug).
//!
//! A dynamic `array<T>` is an i64 handle to the `[addr | len | cap]` record;
//! the native `vec_push` grows the record IN PLACE and returns the same handle.
//! An alias taken before the push therefore sees the grown array, never a
//! freed buffer. A local slice binding (the direct `int* p = &v[0]` shape) is
//! refused outright. If either property changes, these tests must go red.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const ALIAS_AFTER_PUSH: &str = r#"
fn grow_then_read(xs: array<i64>, view: &[i64]) -> i64 {
    xs.push(5)
    let other: array<i64> = [777, 777, 777, 777]
    return view.length * 10 + view[0] + 0 * other.length
}

pub fn alias_control() -> i64 {
    let xs: array<i64> = [1, 2, 3, 4]
    let view = xs
    xs[0] = 7
    return view[0]
}

pub fn alias_after_push() -> i64 {
    let xs: array<i64> = [1, 2, 3, 4]
    let blocker: array<i64> = [9, 9, 9, 9]
    let view = xs
    xs.push(5)
    let other: array<i64> = [777, 777, 777, 777]
    return view.length * 100 + xs.length * 10 + view[0] + 0 * blocker.length + 0 * other.length
}

pub fn slice_param_after_owner_push() -> i64 {
    let xs: array<i64> = [1, 2, 3, 4]
    return grow_then_read(xs, xs)
}
"#;

fn compile(src: &str, tag: &str, dir: &Path) -> (Output, PathBuf) {
    let source = dir.join(format!("{tag}.mind"));
    let artifact = dir.join(format!("{tag}.so"));
    std::fs::write(&source, src).expect("write source");
    let out = Command::new(common::mindc_bin())
        .args([
            source.to_str().unwrap(),
            "--emit-shared",
            artifact.to_str().unwrap(),
        ])
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc");
    (out, artifact)
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn alias_taken_before_push_sees_the_grown_array() {
    let dir = common::scratch_dir("array_alias_push_run");
    let (out, so) = compile(ALIAS_AFTER_PUSH, "alias_after_push", &dir);
    assert!(out.status.success(), "compile failed:\n{}", combined(&out));

    // A stale handle would report length 4 (or read a reused buffer: 777).
    let py = format!(
        r#"import ctypes
lib=ctypes.CDLL(r'{}')
expected={{'alias_control':7,'alias_after_push':551,'slice_param_after_owner_push':51}}
for name,want in expected.items():
    fn=getattr(lib,name); fn.restype=ctypes.c_int64
    got=fn(); assert got==want, f'{{name}}={{got}}, expected {{want}}'
"#,
        so.display()
    );
    let ran = Command::new("python3")
        .args(["-I", "-c", &py])
        .output()
        .expect("run native artifact");
    assert!(
        ran.status.success(),
        "native execution failed:\n{}",
        combined(&ran)
    );
}

#[test]
fn local_slice_view_then_push_is_refused() {
    let dir = common::scratch_dir("array_alias_push_refuse");
    let src = "pub fn run() -> i64 {\n    let xs: array<i64> = [1, 2, 3, 4]\n    let view: &[i64] = xs\n    xs.push(5)\n    return view[0]\n}\n";
    let (out, so) = compile(src, "local_slice_view", &dir);
    let text = combined(&out);
    assert!(!out.status.success(), "must be refused:\n{text}");
    assert!(text.contains("E2032"), "expected E2032, got:\n{text}");
    assert!(
        text.contains("slice binding `view`"),
        "refusal must name the slice binding:\n{text}"
    );
    assert!(!so.exists(), "a refused program must not leave an artifact");
}
