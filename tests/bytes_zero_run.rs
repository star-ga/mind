// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `bytes[N].zero()` RUNTIME gate — a zeroed N-byte heap buffer.
//!
//! mind-flow uses fixed-size byte buffers for hashes (`bytes[32]`, `bytes[8]`).
//! `bytes[N].zero()` lowers to `__mind_calloc(N)` (a zeroed N-byte heap buffer,
//! i64 handle). Compiles a program that allocates a zeroed buffer, writes into
//! it, and reads back, asserting both the zero-init and that it is real
//! writable memory.
//!
//! Gate: `cargo test --features "std-surface mlir-build" --test bytes_zero_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
// A freshly zeroed 32-byte buffer reads back all zero.
pub fn zeroed() -> i64 {
    let h = bytes[32].zero()
    return __mind_load_i64(h) + __mind_load_i64(h + 8) + __mind_load_i64(h + 24)
}

// It is real writable memory: store then load.
pub fn writable() -> i64 {
    let h = bytes[16].zero()
    let _ = __mind_store_i64(h + 8, 12345)
    return __mind_load_i64(h + 8) + __mind_load_i64(h)
}

// `bytes[N]` as a PARAMETER and RETURN type (the i64-handle ABI). The `[N]`
// suffix on a Named type must parse in signature position (mind-flow uses
// `-> bytes[32]` for hashes).
fn first_word(buf: bytes[32]) -> i64 {
    return __mind_load_i64(buf)
}

fn make() -> bytes[32] {
    let h = bytes[32].zero()
    let _ = __mind_store_i64(h, 777)
    return h
}

// `bytes[N]` as a STRUCT FIELD type.
struct Hashed {
    tag: i64,
    digest: bytes[32],
}

pub fn typed_roundtrip() -> i64 {
    let h = make()
    let r = first_word(h)
    let rec = Hashed { tag: 5, digest: bytes[8].zero() }
    return r + rec.tag
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
fn bytes_zero_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("bytes-zero-run: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_bytes_zero_run.mind");
    let so = dir.join("mind_bytes_zero_run.so");
    std::fs::write(&src, SRC).expect("write src");

    let out = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("run mindc");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("bytes-zero-run: needs mlir-build; skipping");
            return;
        }
        panic!("bytes-zero-run: mindc --emit-shared failed:\n{stderr}");
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         for _n in ('zeroed','writable','typed_roundtrip'): getattr(lib,_n).restype = ctypes.c_int64\n\
         r = lib.zeroed(); assert r == 0, 'zeroed=' + str(r)\n\
         r = lib.writable(); assert r == 12345, 'writable=' + str(r)\n\
         r = lib.typed_roundtrip(); assert r == 782, 'typed_roundtrip=' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "bytes-zero-run check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
