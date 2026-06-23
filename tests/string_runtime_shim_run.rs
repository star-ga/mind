// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Runtime-support std.string gate for standalone `--emit-shared`.
//!
//! The single-file shared-library path lowers imported std.string calls as
//! external `string_*` symbols and relies on `runtime-support/mind_intrinsics.c`
//! to make the `.so` self-contained. Pre-fix, that shim defined only part of
//! the std.string surface: a program using `string_eq`, `string_slice_from`,
//! `string_starts_with`, `string_push_str`, or `string_push_i64` compiled with
//! EXIT=0 but failed at `dlopen` with an undefined `string_*` symbol. This gate
//! proves the artifact is dlopen-able and returns byte-oracle values.
//!
//! Gate: `cargo test --release --features "std-surface mlir-build cross-module-imports"
//!                   --test string_runtime_shim_run`

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
import std.string

fn mk_n(n: i64) -> string {
    let mut s = string_new()
    let mut i = 0
    while i < n {
        s = string_push_byte(s, 65 + i)
        i = i + 1
    }
    return s
}

pub fn push17_sum() -> i64 {
    let s = mk_n(17)
    let mut i = 0
    let mut sum = string_len(s) * 1000
    while i < string_len(s) {
        sum = sum + string_get_byte(s, i)
        i = i + 1
    }
    return sum
}

pub fn slice_push_alias() -> i64 {
    let s = mk_n(3)
    let v = string_slice_from(s, 1)
    let w = string_push_byte(v, 90)
    return string_get_byte(s, 1) * 10000 + string_get_byte(w, 0) * 100 + string_get_byte(w, 2)
}

pub fn eq_diff() -> i64 {
    let mut a = string_new()
    a = string_push_byte(a, 65)
    a = string_push_byte(a, 66)
    let mut b = string_new()
    b = string_push_byte(b, 65)
    b = string_push_byte(b, 67)
    return string_eq(a, b)
}

pub fn starts() -> i64 {
    let a = mk_n(4)
    let n = mk_n(2)
    return string_starts_with(a, n)
}

pub fn concat_sum() -> i64 {
    let a = mk_n(18)
    let b = mk_n(3)
    let c = string_push_str(a, b)
    let mut i = 0
    let mut sum = string_len(c) * 1000
    while i < string_len(c) {
        sum = sum + string_get_byte(c, i)
        i = i + 1
    }
    return sum
}

pub fn itoa_probe() -> i64 {
    let s = string_push_i64(string_new(), -42)
    return string_len(s) * 1000000 + string_get_byte(s, 0) * 10000 + string_get_byte(s, 1) * 100 + string_get_byte(s, 2)
}
"#;

#[test]
fn standalone_shared_links_full_string_runtime_surface() {
    let mindc = mindc_bin();
    let dir = std::env::temp_dir();
    let src = dir.join("mind_string_runtime_shim_run.mind");
    let so = dir.join("mind_string_runtime_shim_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    assert!(
        out.status.success(),
        "string-runtime-shim-run: mindc --emit-shared failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let py = format!(
        r#"import ctypes
lib = ctypes.CDLL(r'{}')
expect = {{
    'push17_sum': 18241,
    'slice_push_alias': 666690,
    'eq_diff': 0,
    'starts': 1,
    'concat_sum': 22521,
    'itoa_probe': 3455250,
}}
bad = []
for name, want in expect.items():
    f = getattr(lib, name)
    f.restype = ctypes.c_int64
    got = f()
    if got != want:
        bad.append((name, got, want))
assert not bad, 'mismatches: ' + str(bad)
print('ok')
"#,
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-runtime-shim-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
