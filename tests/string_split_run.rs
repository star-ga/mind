// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! String method RUNTIME gate — `.split` / `.trim` (incl. through for-each).
//!
//! A method on a `String` receiver — an Ident bound to a string, a struct FIELD
//! of type `string`, or a for-each element typed from its collection — routes to
//! the `string_<method>` std free functions. `string_split(s, sep)` returns an
//! `array<string>` (a std.vec of String handles); `string_trim` strips ASCII
//! whitespace. This exercises the full chain mind-flow uses: a string struct
//! field `.split("+")`, a for-each over the split result (its elements typed as
//! String), and `.trim()` on each element.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test string_split_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
import std.string

struct Dec {
    arg_string: string,
}

// Split a string FIELD, iterate the result (elements typed String), trim each.
pub fn process(d: Dec) -> i64 {
    let mut n = 0
    let mut total = 0
    for part in d.arg_string.split("+") {
        let t = part.trim()
        n = n + 1
        total = total + string_len(t)
    }
    return n * 100 + total
}

fn mkstr(a: i64, b: i64, c: i64) -> string {
    let s = string_new()
    let s = string_push_byte(s, a)
    let s = string_push_byte(s, 43)
    let s = string_push_byte(s, b)
    let s = string_push_byte(s, 43)
    let s = string_push_byte(s, c)
    return s
}

pub fn run() -> i64 {
    let d = Dec { arg_string: mkstr(65, 66, 67) }
    return process(d)
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
fn string_split_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("string-split-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_string_split_run.mind");
    let so = dir.join("mind_string_split_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("string-split-run: needs mlir-build; skipping");
            return;
        }
        panic!("string-split-run: mindc --emit-shared failed:\n{stderr}");
    }

    // "A+B+C".split("+") = 3 parts; each trims to length 1 → 3*100 + 3 = 303.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 303, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-split-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
