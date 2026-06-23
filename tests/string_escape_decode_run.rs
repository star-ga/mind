// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! String-literal escape-decoding RUNTIME gate (regression for the parser
//! silent miscompile fixed in `parser::parse_string_lit`).
//!
//! Before the fix, `parse_string_lit` retained escape sequences VERBATIM: the
//! literal `"\n"` was stored as the two bytes `\` (92) and `n` (110), so the
//! materialized `String` had `string_len("\n") == 2` and
//! `string_get_byte("\n", 0) == 92`. That is a wrong-but-RUNNABLE artifact — a
//! silent miscompile — and it disagreed with `parse_char_lit`, which correctly
//! decodes `'\n'` to the single byte 10. This gate compiles a program that
//! builds real `String` values from escaped literals, dlopen-calls it, and
//! asserts the DECODED byte length and bytes for `\n \t \r \0 \\ \"` plus a
//! mixed literal — proving the escapes lower to their byte values.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test string_escape_decode_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
use std.string

// `\n` decodes to a 1-byte string whose byte is 10 (NOT 2 bytes `\`,`n`).
pub fn nl_len() -> i64 { let s: String = "\n"; return string_len(s) }
pub fn nl_b0() -> i64 { let s: String = "\n"; return string_get_byte(s, 0) }
// `\t` -> 9
pub fn tab_len() -> i64 { let s: String = "\t"; return string_len(s) }
pub fn tab_b0() -> i64 { let s: String = "\t"; return string_get_byte(s, 0) }
// `\r` -> 13
pub fn cr_b0() -> i64 { let s: String = "\r"; return string_get_byte(s, 0) }
// `\0` -> a 1-byte string whose byte is 0
pub fn nul_len() -> i64 { let s: String = "\0"; return string_len(s) }
pub fn nul_b0() -> i64 { let s: String = "\0"; return string_get_byte(s, 0) }
// `\\` -> a single backslash (92); also proves `"\\"` no longer corrupts parse.
pub fn bs_len() -> i64 { let s: String = "\\"; return string_len(s) }
pub fn bs_b0() -> i64 { let s: String = "\\"; return string_get_byte(s, 0) }
// `\"` -> a single double-quote (34) without terminating the literal early.
pub fn q_len() -> i64 { let s: String = "\""; return string_len(s) }
pub fn q_b0() -> i64 { let s: String = "\""; return string_get_byte(s, 0) }
// Mixed literal `a\nb` -> 3 bytes, with the middle byte 10.
pub fn mixed_len() -> i64 { let s: String = "a\nb"; return string_len(s) }
pub fn mixed_b1() -> i64 { let s: String = "a\nb"; return string_get_byte(s, 1) }
// An unescaped literal is untouched (regression guard for the common path).
pub fn plain_len() -> i64 { let s: String = "hello"; return string_len(s) }
pub fn plain_b0() -> i64 { let s: String = "hello"; return string_get_byte(s, 0) }
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
fn string_escapes_decode_to_bytes() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("string-escape-decode-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_string_escape_decode_run.mind");
    let so = dir.join("mind_string_escape_decode_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("string-escape-decode-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("string-escape-decode-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         expect = {{'nl_len':1,'nl_b0':10,'tab_len':1,'tab_b0':9,'cr_b0':13,'nul_len':1,'nul_b0':0,'bs_len':1,'bs_b0':92,'q_len':1,'q_b0':34,'mixed_len':3,'mixed_b1':10,'plain_len':5,'plain_b0':104}}\n\
         funcs = {{n: getattr(lib,n) for n in expect}}\n\
         [setattr(funcs[n], 'restype', ctypes.c_int64) for n in expect]\n\
         bad = [(n, funcs[n](), e) for n,e in expect.items() if funcs[n]() != e]\n\
         assert not bad, 'mismatches: ' + str(bad)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-escape-decode-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
