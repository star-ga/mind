// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Bug #38 — a fixed-size `bytes[N]` buffer handle flowing into a growable
//! `bytes` vec parameter is a silent miscompile (the callee reads the raw
//! payload's bytes at offset +8 as the vec length). The type checker now
//! refuses it FAIL-LOUD with diagnostic E2006 rather than emitting the
//! mismatch (consistent with the #306 fail-closed philosophy).
//!
//! The check is NARROW by construction: it fires ONLY when a `bytes`-typed
//! parameter receives an argument that is itself a fixed-`bytes[N]` value form
//! (`bytes[N].zero()` or a call whose declared return type is `bytes[N]`), so a
//! legitimate growable-`bytes` argument (e.g. `bytes.empty()`) and a legitimate
//! `bytes[N]` value used AS `bytes[N]` both still compile.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test bytes_fixed_into_vec_run`

#![cfg(all(unix, feature = "std-surface", feature = "cross-module-imports"))]

mod common;
use common::mindc_bin;

use std::process::Command;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Compile `src` single-file and return (success, combined stdout+stderr).
fn compile(src: &str, tag: &str) -> (bool, String) {
    let mindc = mindc_bin();
    let dir = std::env::temp_dir();
    let path = dir.join(format!("mind_bug38_{tag}.mind"));
    let so = dir.join(format!("mind_bug38_{tag}.so"));
    std::fs::write(&path, src).expect("write src");
    let out = Command::new(&mindc)
        .args([
            path.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc");
    let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), combined)
}

/// The miscompile must be REFUSED: `bytes[32].zero()` (a fixed buffer) into a
/// growable `bytes` parameter is a compile error carrying code E2006.
#[test]
fn fixed_bytes_into_vec_param_is_rejected() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug38-run: mindc not found; skipping");
        return;
    }
    let src = "pub fn take_bytes(input: bytes) -> i64 { return input.length }\n\
               pub fn run() -> i64 { return take_bytes(bytes[32].zero()) }\n";
    let (ok, output) = compile(src, "reject");
    assert!(
        !ok,
        "bug38: fixed bytes[N] into a growable `bytes` param must FAIL to compile, but it succeeded\n{output}"
    );
    assert!(
        output.contains("E2006") && output.contains("fixed-size `bytes[N]` buffer handle"),
        "bug38: expected the E2006 fixed-buffer diagnostic, got:\n{output}"
    );
}

/// No false positive: a legitimate growable `bytes` argument still compiles.
#[test]
fn growable_bytes_arg_still_compiles() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug38-run: mindc not found; skipping");
        return;
    }
    let src = "pub fn take_bytes(input: bytes) -> i64 { return input.length }\n\
               pub fn run() -> i64 { return take_bytes(bytes.empty()) }\n";
    let (ok, output) = compile(src, "growable_ok");
    // Either it compiles, or it fails for a reason OTHER than the #38 check.
    assert!(
        !output.contains("E2006"),
        "bug38: a growable `bytes` argument must NOT trigger the fixed-buffer check:\n{output}"
    );
    let _ = ok;
}

/// No false positive: a `bytes[N]` value used AS `bytes[N]` (not flowed into a
/// `bytes` param) still compiles.
#[test]
fn fixed_bytes_used_as_fixed_compiles() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug38-run: mindc not found; skipping");
        return;
    }
    let src = "pub fn run() -> i64 { let h = bytes[32].zero(); return 1 }\n";
    let (_ok, output) = compile(src, "fixed_ok");
    assert!(
        !output.contains("E2006"),
        "bug38: `bytes[N]` used as a fixed buffer must NOT trigger the check:\n{output}"
    );
}

/// #38 follow-up — the SAME miscompile flowed through a LOCAL BINDING.
/// `let buf: bytes[32] = bytes[32].zero(); take_bytes(buf)` keeps the identical
/// i64 handle (binding is NOT a copy into a growable vec), so `take_bytes`'s
/// `bytes` param reads the buffer's raw payload at +8 as the vec `.length` — the
/// exact silent miscompile #38 fails loud on, but via a one-line alias. The
/// original guard only matched the SYNTACTIC `bytes[N].zero()` / `bytes[N]`-
/// returning-call argument forms, so it silently accepted the aliased value.
/// Must now be REFUSED with E2006 too.
#[test]
fn fixed_bytes_local_into_vec_param_is_rejected() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug38-run: mindc not found; skipping");
        return;
    }
    let src = "fn take_bytes(input: bytes) -> i64 { return __mind_load_i64(input + 8) }\n\
               pub fn run() -> i64 {\n\
               \x20   let buf: bytes[32] = bytes[32].zero()\n\
               \x20   return take_bytes(buf)\n\
               }\n";
    let (ok, output) = compile(src, "local_reject");
    assert!(
        !ok,
        "bug38-follow-up: a `bytes[N]` LOCAL flowing into a growable `bytes` param \
         must FAIL to compile (same miscompile as the direct form), but it succeeded\n{output}"
    );
    assert!(
        output.contains("E2006") && output.contains("fixed-size `bytes[N]` buffer handle"),
        "bug38-follow-up: expected the E2006 fixed-buffer diagnostic on the aliased \
         buffer, got:\n{output}"
    );
}

/// No false positive for the follow-up: a `bytes[N]` LOCAL used AS a `bytes[N]`
/// parameter (not flowed into a growable `bytes` param) still compiles. Guards
/// against the local-tracking over-firing on the legitimate fixed-as-fixed flow.
#[test]
fn fixed_bytes_local_into_fixed_param_compiles() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bug38-run: mindc not found; skipping");
        return;
    }
    let src = "fn first(buf: bytes[32]) -> i64 { return __mind_load_i64(buf) }\n\
               pub fn run() -> i64 {\n\
               \x20   let h: bytes[32] = bytes[32].zero()\n\
               \x20   return first(h)\n\
               }\n";
    let (_ok, output) = compile(src, "local_fixed_ok");
    assert!(
        !output.contains("E2006"),
        "bug38-follow-up: a `bytes[N]` local used AS a `bytes[N]` param must NOT \
         trigger the check:\n{output}"
    );
}
