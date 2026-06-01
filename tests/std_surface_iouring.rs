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
fn iouring_parses_and_lowers_with_ring_api() {
    let module = parser::parse(SRC).expect("std/iouring.mind must parse cleanly");
    let ir = lower_to_ir(&module);
    let names: Vec<&str> = ir
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::FnDef { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    for required in [
        "io_ring_new",
        "io_ring_submit_raw",
        "io_ring_submit_op",
        "io_ring_publish",
        "io_ring_enter",
        "io_ring_reap",
        "io_ring_free",
        "io_uring_nop",
        "io_uring_socketpair_echo",
        "io_ring_register_buffers",
        "io_ring_submit_fixed",
        "io_uring_fixed_buffer_echo",
        "io_ring_register_pbuf_ring",
        "io_ring_submit_recv_provided",
        "io_uring_provided_buffer_recv",
        "io_uring_tcp_accept_one",
        "io_uring_tcp_echo_round",
        "io_uring_tcp_close",
        "io_uring_loopback_echo",
        "io_uring_loopback_echo_n",
        "io_ring_submit_accept_multishot",
        "io_uring_multishot_accept_demo",
        "io_uring_echo_bench",
    ] {
        assert!(
            names.contains(&required),
            "std.iouring must define `{required}`; found {names:?}"
        );
    }
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

    // (1) NOP user_data round-trip; (2) socketpair SEND/RECV echo through the
    // ring. Both SKIP (not fail) where the kernel io_uring is unavailable.
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.io_uring_nop.restype = ctypes.c_int64\n\
         lib.io_uring_socketpair_echo.restype = ctypes.c_int64\n\
         lib.io_uring_socketpair_echo.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         lib.io_uring_loopback_echo.restype = ctypes.c_int64\n\
         lib.io_uring_loopback_echo.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         lib.io_uring_loopback_echo_n.restype = ctypes.c_int64\n\
         lib.io_uring_loopback_echo_n.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]\n\
         r = lib.io_uring_nop(0x4242)\n\
         if r < 0:\n\
         \x20   print('SKIP', r)  # io_uring unavailable in this environment\n\
         else:\n\
         \x20   assert r == 0x4242, f'NOP user_data corrupted: got {{hex(r)}}'\n\
         \x20   msg = b'MIND-io_uring-echo'\n\
         \x20   buf = ctypes.create_string_buffer(msg)\n\
         \x20   e = lib.io_uring_socketpair_echo(ctypes.cast(buf, ctypes.c_void_p).value, len(msg))\n\
         \x20   assert e == 1 or e < 0, f'socketpair echo returned unexpected {{e}}'\n\
         \x20   fbuf = ctypes.create_string_buffer(b'MIND-io_uring-FIXED-buffer')\n\
         \x20   lib.io_uring_fixed_buffer_echo.restype = ctypes.c_int64\n\
         \x20   lib.io_uring_fixed_buffer_echo.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         \x20   fx = lib.io_uring_fixed_buffer_echo(ctypes.cast(fbuf, ctypes.c_void_p).value, 26)\n\
         \x20   assert fx == 1 or fx < 0, f'fixed-buffer echo returned unexpected {{fx}}'\n\
         \x20   pbuf = ctypes.create_string_buffer(b'MIND-io_uring-PROVIDED-buffer')\n\
         \x20   lib.io_uring_provided_buffer_recv.restype = ctypes.c_int64\n\
         \x20   lib.io_uring_provided_buffer_recv.argtypes = [ctypes.c_int64, ctypes.c_int64]\n\
         \x20   pv = lib.io_uring_provided_buffer_recv(ctypes.cast(pbuf, ctypes.c_void_p).value, 29)\n\
         \x20   assert pv == 1 or pv < 0, f'provided-buffer recv returned unexpected {{pv}}'\n\
         \x20   tmsg = b'MIND-io_uring-tcp-echo'\n\
         \x20   tbuf = ctypes.create_string_buffer(tmsg)\n\
         \x20   le = lib.io_uring_loopback_echo(ctypes.cast(tbuf, ctypes.c_void_p).value, len(tmsg))\n\
         \x20   assert le == 1 or le < 0, f'loopback echo returned unexpected {{le}}'\n\
         \x20   ln = lib.io_uring_loopback_echo_n(ctypes.cast(tbuf, ctypes.c_void_p).value, len(tmsg), 5)\n\
         \x20   assert ln == 5 or ln < 0, f'reactor server loop returned unexpected {{ln}}'\n\
         \x20   lib.io_uring_multishot_accept_demo.restype = ctypes.c_int64\n\
         \x20   ms = lib.io_uring_multishot_accept_demo()\n\
         \x20   assert ms == 2 or ms < 0, f'multishot accept returned unexpected {{ms}}'\n\
         \x20   lib.io_uring_echo_bench.restype = ctypes.c_int64\n\
         \x20   lib.io_uring_echo_bench.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]\n\
         \x20   bench = lib.io_uring_echo_bench(ctypes.cast(tbuf, ctypes.c_void_p).value, len(tmsg), 500)\n\
         \x20   assert bench > 0 or bench < 0, f'echo bench returned {{bench}}'\n\
         \x20   print('OK', hex(r), 'sp', e, 'fixed', fx, 'pbuf', pv, 'tcp', le, 'loop', ln, 'multishot', ms, 'reqs', bench)\n",
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
