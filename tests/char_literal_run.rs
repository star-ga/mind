// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Character-literal RUNTIME gate.
//!
//! A char literal `'c'` / `'\n'` is an integer constant equal to the character's
//! Unicode scalar value (the byte, for ASCII), with the usual escapes. It lowers
//! exactly like the equivalent integer literal — fully deterministic. This
//! compiles a program to a `.so`, dlopen-calls it, and asserts the values,
//! including arithmetic on char literals (the lexer idiom `c - '0'`).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test char_literal_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
pub fn newline() -> i64 { return '\n' }
pub fn tab() -> i64 { return '\t' }
pub fn zero_digit() -> i64 { return '0' }
pub fn letter_a() -> i64 { return 'a' }
pub fn backslash() -> i64 { return '\\' }
pub fn squote() -> i64 { return '\'' }
// Lexer idiom: digit value of a char.
pub fn digit_val(c: i64) -> i64 { return c - '0' }
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
fn char_literal_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("char-literal-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_char_literal_run.mind");
    let so = dir.join("mind_char_literal_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("char-literal-run: mindc --emit-shared needs mlir-build; skipping");
            return;
        }
        panic!("char-literal-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('newline','tab','zero_digit','letter_a','backslash','squote'): getattr(lib,_n).restype = ctypes.c_int64\n\
         lib.digit_val.restype = ctypes.c_int64; lib.digit_val.argtypes = [ctypes.c_int64]\n\
         r = lib.newline(); assert r == 10, 'newline=' + str(r)\n\
         r = lib.tab(); assert r == 9, 'tab=' + str(r)\n\
         r = lib.zero_digit(); assert r == 48, 'zero_digit=' + str(r)\n\
         r = lib.letter_a(); assert r == 97, 'letter_a=' + str(r)\n\
         r = lib.backslash(); assert r == 92, 'backslash=' + str(r)\n\
         r = lib.squote(); assert r == 39, 'squote=' + str(r)\n\
         r = lib.digit_val(55); assert r == 7, 'digit_val(55)=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "char-literal-run value check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
