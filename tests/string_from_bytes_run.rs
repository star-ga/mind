// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Static-type string call RUNTIME gate — `string.from_utf8_bytes(buf)`.
//!
//! `string.from_utf8_bytes(buf)` is a STATIC/associated call: the receiver is the
//! bare TYPE name `string`, not a value, so it routes to the `string_from_utf8_bytes`
//! runtime fn (new C shim: builds a String from a std.vec of byte values) with NO
//! receiver arg. The type-checker (infer_expr + the resolve pass) recognises the
//! type-name receiver instead of flagging "unknown identifier `string`".
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test string_from_bytes_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
fn build() -> string {
    let mut buf: bytes = []
    buf.push(72)
    buf.push(73)
    buf.push(74)
    buf.push(75)
    return string.from_utf8_bytes(buf)
}

pub fn run() -> i64 {
    let s = build()
    return string_len(s)
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
fn string_from_bytes_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("string-from-bytes-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_string_from_bytes_run.mind");
    let so = dir.join("mind_string_from_bytes_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("string-from-bytes-run: needs mlir-build; skipping");
            return;
        }
        panic!("string-from-bytes-run: mindc --emit-shared failed:\n{stderr}");
    }

    // four bytes pushed → the built String has length 4.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 4, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "string-from-bytes-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
