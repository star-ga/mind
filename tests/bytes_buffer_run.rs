// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Dynamic `bytes` buffer RUNTIME gate — `let mut buf: bytes = []` used as a
//! growable `Vec<u8>` (`buf.push(b)`), then passed to a free function.
//!
//! A `bytes` binding INITIALISED from an array literal `[..]` is a freshly-built
//! growable buffer, so it lowers onto the std.vec runtime (push grows it, the
//! handle persists via the existing rebind, and the handle passes through a
//! function call). A raw byte VIEW (`bytes` struct field / param read from data)
//! is deliberately NOT routed this way — the `[..]` initialiser is the
//! discriminator, so the view's indexing is unaffected.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test bytes_buffer_run`

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
fn build() -> i64 {
    let mut buf: bytes = []
    buf.push(72)
    buf.push(73)
    buf.push(74)
    return vec_len(buf)
}

pub fn run() -> i64 {
    return build()
}
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn bytes_buffer_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bytes-buffer-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_bytes_buffer_run.mind");
    let so = dir.join("mind_bytes_buffer_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("bytes-buffer-run: needs mlir-build; skipping");
            return;
        }
        panic!("bytes-buffer-run: mindc --emit-shared failed:\n{stderr}");
    }

    // three pushes onto an empty bytes buffer → vec_len == 3.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.run.restype = ctypes.c_int64\n\
         r = lib.run(); assert r == 3, 'run=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "bytes-buffer-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
