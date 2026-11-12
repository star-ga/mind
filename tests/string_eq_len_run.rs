// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native string `==` / `!=` / `.len()` RUNTIME gate (issue #245).
//!
//! Before this gate, `==` between two strings compared the two `__mind_alloc`
//! heap-record POINTERS — never equal for two distinct literals — so
//! `"abc" == "abc"` was FALSE, and an annotated `let s: string` was tracked by
//! nothing, so `s.len()` missed the `string_<method>` dispatch and lowered to
//! `const.i64 0`. Both built clean to a real ELF, exited 0, and silently
//! computed the wrong answer with no diagnostic and no JIT-fallback banner.
//!
//! This asserts VALUES through the real runtime, not that the source parses.
//! The integer arm is the control: `==` on i64 must be untouched.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_eq_len_run`

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
// Two EQUAL string literals must compare equal (was 0 — pointer compare).
pub fn lit_eq() -> i64 {
    if "abc" == "abc" { return 1 }
    return 0
}

// Two DIFFERENT literals must still compare unequal.
pub fn lit_neq() -> i64 {
    if "abc" == "xyz" { return 1 }
    return 0
}

// `.len()` on an annotated string local (was 0 — unresolved receiver).
pub fn ann_len() -> i64 {
    let s: string = "abcd"
    return s.len()
}

// Equality through VARIABLES, not just literals.
pub fn var_eq() -> i64 {
    let a: string = "hello"
    let b: string = "hello"
    if a == b { return 1 }
    return 0
}

// `!=` is the negation of string_eq, not a pointer compare.
pub fn ne_diff() -> i64 {
    let a: string = "hello"
    let b: string = "world"
    if a != b { return 1 }
    return 0
}

pub fn distinct_alloc_eq() -> i64 {
    let a: string = "separate"
    let b: string = "separate"
    if a == b { return 1 }
    return 0
}

pub fn ne_equal() -> i64 {
    let a: string = "hello"
    let b: string = "hello"
    if a != b { return 1 }
    return 0
}

// Keep the public std.string function source-compatible.
pub fn public_string_eq_compat() -> i64 {
    return string_eq("compat", "compat")
}

// CONTROL: i64 `==` must be byte-for-byte unaffected by the string routing.
pub fn int_eq_unchanged() -> i64 {
    let a: i64 = 3
    if a == 3 { return 7 }
    return 0
}

pub fn int_ne_unchanged() -> i64 {
    if 3 != 4 { return 9 }
    return 0
}
"#;

#[test]
fn string_eq_and_len_run() {
    let mindc = mindc_bin();
    assert!(mindc.exists(), "string_eq_len_run requires the built mindc");
    let dir = common::scratch_dir("string_eq_len_run");
    let src = dir.join("mind_string_eq_len_run.mind");
    let so = dir.join("mind_string_eq_len_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !common::gate::compiled("string_eq_len_run", &out) {
        return;
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('lit_eq','lit_neq','ann_len','var_eq','ne_diff','distinct_alloc_eq','ne_equal','public_string_eq_compat','int_eq_unchanged','int_ne_unchanged'):\n\
         \x20   getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.lit_eq(); assert r == 1, 'lit_eq=' + str(r)\n\
         r = lib.lit_neq(); assert r == 0, 'lit_neq=' + str(r)\n\
         r = lib.ann_len(); assert r == 4, 'ann_len=' + str(r)\n\
         r = lib.var_eq(); assert r == 1, 'var_eq=' + str(r)\n\
         r = lib.ne_diff(); assert r == 1, 'ne_diff=' + str(r)\n\
         r = lib.distinct_alloc_eq(); assert r == 1, 'distinct_alloc_eq=' + str(r)\n\
         r = lib.ne_equal(); assert r == 0, 'ne_equal=' + str(r)\n\
         r = lib.public_string_eq_compat(); assert r == 1, 'public_string_eq_compat=' + str(r)\n\
         r = lib.int_eq_unchanged(); assert r == 7, 'int_eq_unchanged=' + str(r)\n\
         r = lib.int_ne_unchanged(); assert r == 9, 'int_ne_unchanged=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-eq-len-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn user_string_eq_cannot_hijack_operator_equality() {
    let dir = common::scratch_dir("string_eq_operator_shadow");
    let src = dir.join("shadow.mind");
    let so = dir.join("shadow.so");
    std::fs::write(
        &src,
        r#"
fn string_eq(a: string, b: string) -> i64 { return 0 }
pub fn operator_eq() -> i64 { if "same" == "same" { return 17 }; return 0 }
pub fn direct_public_call() -> i64 { return string_eq("same", "same") }
"#,
    )
    .expect("write shadow source");
    let out = Command::new(mindc_bin())
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("compile shadow source");
    assert!(
        out.status.success(),
        "shadow compile failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\n\
         lib.operator_eq.restype=ctypes.c_int64\n\
         lib.direct_public_call.restype=ctypes.c_int64\n\
         assert lib.operator_eq()==17, lib.operator_eq()\n\
         assert lib.direct_public_call()==0, lib.direct_public_call()\n",
        so.display()
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run shadow artifact");
    assert!(
        ran.status.success(),
        "shadow execution failed:\n{}{}",
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );
}

#[test]
fn reserved_string_equality_entry_is_fail_closed() {
    let dir = common::scratch_dir("string_eq_reserved_entry");
    let (tag, source, expected) = (
        "definition",
        "fn __mind_string_eq(a: i64, b: i64) -> i64 { return 0 }\npub fn run() -> i64 { return 0 }\n",
        "E2023",
    );
    let src = dir.join(format!("{tag}.mind"));
    let so = dir.join(format!("{tag}.so"));
    std::fs::write(&src, source).expect("write refusal source");
    let out = Command::new(mindc_bin())
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("compile refusal source");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "{tag} unexpectedly compiled");
    assert!(text.contains(expected), "{tag} missed {expected}:\n{text}");
    assert!(!text.contains("panicked at"), "{tag} panicked:\n{text}");
    assert!(!so.exists(), "{tag} emitted an artifact");
}

#[test]
fn imported_public_string_eq_cannot_substitute_for_operator_runtime() {
    let root = common::scratch_dir("string_eq_import_shadow");
    std::fs::create_dir_all(root.join("src")).expect("create source directory");
    std::fs::write(
        root.join("Mind.toml"),
        r#"[package]
name = "string_eq_import_shadow"
version = "0.1.0"

[build]
entry = "src/main.mind"
output = "string_eq_import_shadow"

[targets.cpu]
backend = "cpu"

[exports]
c_abi = ["operator_eq", "direct_import"]
"#,
    )
    .expect("write manifest");
    std::fs::write(
        root.join("src/main.mind"),
        r#"use crate.helper
pub fn operator_eq() -> i64 { if "same" == "same" { return 17 }; return 0 }
pub fn direct_import() -> i64 { return string_eq("same", "same") }
"#,
    )
    .expect("write entry source");
    let helper = root.join("src/helper.mind");
    std::fs::write(
        &helper,
        "pub fn string_eq(a: string, b: string) -> i64 { return 0 }\n",
    )
    .expect("write helper source");
    let so = root.join("shadow.so");
    let out = Command::new(mindc_bin())
        .current_dir(&root)
        .args([
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("build imported shadow project");
    assert!(
        out.status.success(),
        "imported shadow build failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let py = format!(
        "import ctypes\nlib=ctypes.CDLL(r'{}')\n\
         lib.operator_eq.restype=ctypes.c_int64\n\
         lib.direct_import.restype=ctypes.c_int64\n\
         assert lib.operator_eq()==17, lib.operator_eq()\n\
         assert lib.direct_import()==0, lib.direct_import()\n",
        so.display()
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run imported shadow artifact");
    assert!(
        ran.status.success(),
        "imported shadow execution failed:\n{}{}",
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );

    std::fs::write(
        &helper,
        "pub fn __mind_string_eq(a: i64, b: i64) -> i64 { return 0 }\n",
    )
    .expect("write reserved helper source");
    let refused = Command::new(mindc_bin())
        .current_dir(&root)
        .args([
            "build",
            "--backend=mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            root.join("reserved.so").to_str().unwrap(),
        ])
        .output()
        .expect("build reserved import project");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&refused.stdout),
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(!refused.status.success(), "reserved import compiled");
    assert!(
        text.contains("E2023"),
        "missing reserved-name diagnostic:\n{text}"
    );
    assert!(
        !text.contains("panicked at"),
        "reserved import panicked:\n{text}"
    );
    assert!(!root.join("reserved.so").exists());
}
