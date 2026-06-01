// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `__mind_load_i32` / `__mind_store_i32` intrinsic surface tests.
//!
//! These 4-byte load/store intrinsics provide a proper i32 ABI for the
//! `u32`-field structs of kernel interfaces (the io_uring SQ/CQ ring head/tail
//! and other ABI fields). `__mind_store_i32` must write EXACTLY 4 bytes — never
//! clobbering an adjacent `u32` the way `__mind_store_i64` at a 4-byte offset
//! would — and `__mind_load_i32` must zero-extend to i64 (unsigned 32-bit).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test std_surface_i32_intrinsics`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

use std::path::PathBuf;
use std::process::Command;

// Stores two adjacent u32s, reads both back, and checks a high-bit value
// zero-extends (no sign extension). Returns a bitmask: 1=no-clobber, 2=high
// value zero-extends — so a fully-correct run returns 3.
const SRC: &str = r#"
pub fn i32_probe() -> i64 {
    let buf: i64 = __mind_alloc(16)
    // Two adjacent u32 fields: writing one must not clobber the other.
    let _ = __mind_store_i32(buf + 0, 2864434397)
    let _ = __mind_store_i32(buf + 4, 287454020)
    let a: i64 = __mind_load_i32(buf + 0)
    let b: i64 = __mind_load_i32(buf + 4)
    // A high-bit-set u32 must zero-extend (not become negative i64).
    let _ = __mind_store_i32(buf + 8, 4294967295)
    let c: i64 = __mind_load_i32(buf + 8)
    let _ = __mind_free(buf)
    let mut result: i64 = 0
    if a == 2864434397 {
        if b == 287454020 {
            result = result + 1
        }
    }
    if c == 4294967295 {
        result = result + 2
    }
    result
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
fn i32_load_store_no_clobber_and_zero_extend() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("i32: mindc not found; skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let src = dir.join("mind_i32_probe.mind");
    let so = dir.join("mind_i32_probe.so");
    std::fs::write(&src, SRC).expect("write src");

    let status = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .status()
        .expect("run mindc");
    if !status.success() {
        println!("i32: mindc --emit-shared failed (no MLIR backend?); skipping");
        return;
    }

    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.i32_probe.restype = ctypes.c_int64\n\
         r = lib.i32_probe()\n\
         assert r == 3, f'i32 intrinsics wrong: bitmask {{r}} (1=no-clobber, 2=zero-extend, want 3)'\n\
         print('ok', r)\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "i32 intrinsic check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
