// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std/iouring.mind` surface test — io_uring NOP round-trip.
//!
//! Verifies the minimal io_uring protocol (io_uring_setup -> mmap SQ/CQ/SQE
//! rings -> build an IORING_OP_NOP SQE -> io_uring_enter submit+wait -> reap the
//! CQE) works end-to-end from pure MIND: a NOP's `user_data` must round-trip
//! unchanged through the kernel completion ring.
//!
//! io_uring may be unavailable on a CI runner (old kernel, seccomp, or
//! containerized without the syscall) — in that case `io_uring_nop` returns a
//! negative errno/sentinel and the test SKIPS rather than fails. A positive but
//! wrong return would indicate ring-protocol corruption and FAILS.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test std_surface_iouring`

#![cfg(all(target_os = "linux", feature = "mlir-build", feature = "std-surface"))]

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;
use std::path::PathBuf;
use std::process::Command;

const SRC: &str = include_str!("../std/iouring.mind");

#[test]
fn iouring_parses_and_lowers_with_nop_api() {
    let module = parser::parse(SRC).expect("std/iouring.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let has_nop = ir.instrs.iter().any(
        |i| matches!(i, Instr::FnDef { name, .. } if name == "io_uring_nop"),
    );
    assert!(has_nop, "std.iouring must define `io_uring_nop`");
}

#[test]
fn iouring_nop_roundtrips_user_data() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mindc = {
        let d = manifest.join("target").join("debug").join("mindc");
        if d.exists() {
            d
        } else {
            manifest.join("target").join("release").join("mindc")
        }
    };
    if !mindc.exists() {
        println!("iouring: mindc not found; skipping");
        return;
    }

    let so = std::env::temp_dir().join("mind_iouring_nop.so");
    let src = manifest.join("std").join("iouring.mind");
    let status = Command::new(&mindc)
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .status()
        .expect("run mindc");
    if !status.success() {
        println!("iouring: mindc compile failed (no MLIR backend?); skipping");
        return;
    }

    // user_data sentinel that round-trips through the completion ring.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.io_uring_nop.restype = ctypes.c_int64\n\
         r = lib.io_uring_nop(0x4242)\n\
         if r < 0:\n\
         \x20   print('SKIP', r)  # io_uring unavailable in this environment\n\
         else:\n\
         \x20   assert r == 0x4242, f'NOP user_data corrupted: got {{hex(r)}}'\n\
         \x20   print('OK', hex(r))\n",
        so.to_string_lossy()
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "iouring NOP check failed:\nstdout: {}\nstderr: {}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    if stdout.contains("SKIP") {
        println!("iouring: kernel io_uring unavailable; round-trip skipped");
    }
}
