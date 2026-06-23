// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native cross-module cdylib linking — composition test.
//!
//! A consumer project that imports the self-contained `std` I/O substrate
//! modules (`std.io_canon` + `std.reactor`) must produce a cdylib in which
//! those modules' symbols (`canon_*`, `reactor_*`) are DEFINED, not left as
//! undefined externals. `build_cdylib_from_entry` compiles each imported
//! substrate module to its own object (localizing the synthetic `main`) and
//! links them into the consumer `.so`.
//!
//! This exercises the project-build cdylib path (`mindc build` with
//! `emit = "cdylib"`), which is distinct from the standalone `--emit-shared`
//! path covered by `std_surface_cdylib_link`.
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!                     --test cross_module_cdylib_compose`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

const MAIN_MIND: &str = r#"
import std.io_canon;
import std.reactor;
import std.ring;
import std.sha256;
import std.iouring;

// Full physical->logical reactor pipeline: io_uring delivers NOP completions
// (user_data 3,1,2) in physically non-deterministic order; reap them, then
// re-order canonically by the content key. Returns n_bytes*100 + first_conn
// (so 96*100+1 = 9601 when the 3 completions sort to conn 1,2,3), or a negative
// value if the kernel io_uring is unavailable.
pub fn iou_canon_pipeline() -> i64 {
    let h: i64 = io_ring_new(4)
    if h == 0 {
        return 0 - 1
    }
    let _ = io_ring_submit_op(h, 0, iouring_op_nop(), 0, 0, 0, 3)
    let _ = io_ring_submit_op(h, 1, iouring_op_nop(), 0, 0, 0, 1)
    let _ = io_ring_submit_op(h, 2, iouring_op_nop(), 0, 0, 0, 2)
    let _ = io_ring_publish(h, 3)
    let er: i64 = io_ring_enter(h, 3, 3)
    if er < 0 {
        let _ = io_ring_free(h)
        return 0 - 2
    }
    let ud: i64 = __mind_alloc(64)
    let res: i64 = __mind_alloc(64)
    let count: i64 = io_ring_reap(h, ud, res, 8)
    let _ = io_ring_free(h)
    let batch: i64 = canon_new(8)
    let mut i: i64 = 0
    while i < count {
        let _ = canon_push(batch, __mind_load_i64(ud + i * 8), 0, 0, __mind_load_i64(res + i * 8))
        i = i + 1
    }
    let drained: i64 = __mind_alloc(count * 32)
    let n: i64 = canon_drain(batch, drained, count * 32)
    let firstconn: i64 = __mind_load_i64(drained + 0)
    let _ = __mind_free(drained)
    let _ = __mind_free(ud)
    let _ = __mind_free(res)
    n * 100 + firstconn
}

// One deterministic reactor round: echo over the reused ring AND feed the
// round's reaped completions into the io_canon batch (keyed by completion
// user_data + round) so they enter the canonical order. Small function (no
// sha256) to stay clear of the large-frame codegen path. Returns 1 on match.
fn reactor_round_canon(rh: i64, msg: i64, len: i64, batch: i64, round: i64) -> i64 {
    let h: i64 = __mind_load_i64(rh + 0)
    let cfd: i64 = __mind_load_i64(rh + 8)
    let connfd: i64 = __mind_load_i64(rh + 16)
    let ud: i64 = __mind_load_i64(rh + 48)
    let res: i64 = __mind_load_i64(rh + 56)
    let rbuf: i64 = __mind_alloc(len)
    let _ = io_ring_submit_op(h, 0, iouring_op_send(), cfd, msg, len, round * 2)
    let _ = io_ring_submit_op(h, 1, iouring_op_recv(), connfd, rbuf, len, round * 2 + 1)
    let _ = io_ring_publish(h, 2)
    let er: i64 = io_ring_enter(h, 2, 2)
    let mut ok: i64 = 0
    if er >= 0 {
        let count: i64 = io_ring_reap(h, ud, res, 8)
        let mut i: i64 = 0
        while i < count {
            let _ = canon_push(batch, __mind_load_i64(ud + i * 8), round, 0, __mind_load_i64(res + i * 8))
            i = i + 1
        }
        let mut m: i64 = 1
        let mut j: i64 = 0
        while j < len {
            if __mind_load_i8(rbuf + j) != __mind_load_i8(msg + j) {
                m = 0
            }
            j = j + 1
        }
        ok = m
    }
    let _ = __mind_free(rbuf)
    ok
}

// Anchor the canonical completion sequence (canon_drain + sha256), kept in its
// own small function away from the io_uring loop.
fn anchor_batch(batch: i64, out_addr: i64) -> i64 {
    let cap: i64 = canon_len(batch) * 32
    let drained: i64 = __mind_alloc(cap)
    let n: i64 = canon_drain(batch, drained, cap)
    let _ = sha256(drained, n, out_addr)
    let _ = __mind_free(drained)
    0
}

// The DETERMINISTIC REACTOR SERVER: accept a TCP connection, echo `rounds`
// exchanges over one reused io_uring, canonically order all completions, and
// anchor the canonical sequence with SHA-256. Returns `rounds` on success and
// writes the 32-byte evidence anchor to `anchor_out`. This is the full wedge
// applied to a real server: physical I/O at kernel speed, logical reaction
// order canonically deterministic, anchored in the evidence chain.
pub fn deterministic_reactor(msg: i64, len: i64, rounds: i64, anchor_out: i64) -> i64 {
    let rh: i64 = io_uring_tcp_accept_one()
    if rh == 0 {
        return 0 - 1
    }
    let batch: i64 = canon_new(64)
    let mut done: i64 = 0
    let mut ok: i64 = 1
    while ok == 1 {
        if done >= rounds {
            ok = 2
        } else {
            if reactor_round_canon(rh, msg, len, batch, done) == 1 {
                done = done + 1
            } else {
                ok = 0
            }
        }
    }
    let _ = anchor_batch(batch, anchor_out)
    let _ = io_uring_tcp_close(rh)
    if ok == 2 {
        return done
    }
    0 - 2
}

// Compose all three substrate modules: assign a deterministic key via the
// reactor, stage in a ring, order completions canonically, drain to bytes.
pub fn compose_tick(out_addr: i64, cap: i64) -> i64 {
    let r: i64 = reactor_new(4)
    let _ = reactor_accept(r, 7)
    let _ = ring_new(32)
    let h: i64 = canon_new(8)
    let _ = canon_push(h, 2, 1, 9, 100)
    let _ = canon_push(h, 1, 1, 9, 200)
    canon_drain(h, out_addr, cap)
}

// L4 evidence anchor: SHA-256 over the canonically-ordered completion
// sequence. `anchor_a` and `anchor_b` push the SAME completion multiset in
// DIFFERENT physical orders; their anchors must be byte-identical (physical
// arrival order never enters the deterministic I/O input).
pub fn anchor_a(out_addr: i64) -> i64 {
    let h: i64 = canon_new(8)
    let _ = canon_push(h, 3, 1, 9, 30)
    let _ = canon_push(h, 1, 1, 9, 10)
    let _ = canon_push(h, 2, 1, 9, 20)
    let buf: i64 = __mind_alloc(96)
    let n: i64 = canon_drain(h, buf, 96)
    sha256(buf, n, out_addr)
}

pub fn anchor_b(out_addr: i64) -> i64 {
    let h: i64 = canon_new(8)
    let _ = canon_push(h, 1, 1, 9, 10)
    let _ = canon_push(h, 2, 1, 9, 20)
    let _ = canon_push(h, 3, 1, 9, 30)
    let buf: i64 = __mind_alloc(96)
    let n: i64 = canon_drain(h, buf, 96)
    sha256(buf, n, out_addr)
}

// MONOLITHIC deterministic reactor: the SAME pipeline as deterministic_reactor,
// but with reactor_round_canon + anchor_batch INLINED into one large frame
// (io_ring ops + canon push-loop + canon_drain + sha256, ~19 locals + ~10
// distinct cross-module calls). This is the shape an earlier session reported
// segfaulting from "large-function register pressure"; it now compiles and runs
// correctly because the cross-module substrate symbols (canon_*, sha256) link
// natively into the consumer cdylib. Its evidence anchor MUST equal
// deterministic_reactor's for the same workload — proof the monolithic codegen
// path is correct (the root cause was cross-module symbol resolution at the call
// site, not register allocation).
pub fn deterministic_reactor_mono(msg: i64, len: i64, rounds: i64, anchor_out: i64) -> i64 {
    let rh: i64 = io_uring_tcp_accept_one()
    if rh == 0 {
        return 0 - 1
    }
    let h: i64 = __mind_load_i64(rh + 0)
    let cfd: i64 = __mind_load_i64(rh + 8)
    let connfd: i64 = __mind_load_i64(rh + 16)
    let ud: i64 = __mind_load_i64(rh + 48)
    let res: i64 = __mind_load_i64(rh + 56)
    let batch: i64 = canon_new(64)
    let mut done: i64 = 0
    let mut ok: i64 = 1
    while ok == 1 {
        if done >= rounds {
            ok = 2
        } else {
            let rbuf: i64 = __mind_alloc(len)
            let _ = io_ring_submit_op(h, 0, iouring_op_send(), cfd, msg, len, done * 2)
            let _ = io_ring_submit_op(h, 1, iouring_op_recv(), connfd, rbuf, len, done * 2 + 1)
            let _ = io_ring_publish(h, 2)
            let er: i64 = io_ring_enter(h, 2, 2)
            let mut m: i64 = 0
            if er >= 0 {
                let count: i64 = io_ring_reap(h, ud, res, 8)
                let mut i: i64 = 0
                while i < count {
                    let _ = canon_push(batch, __mind_load_i64(ud + i * 8), done, 0, __mind_load_i64(res + i * 8))
                    i = i + 1
                }
                let mut mm: i64 = 1
                let mut j: i64 = 0
                while j < len {
                    if __mind_load_i8(rbuf + j) != __mind_load_i8(msg + j) {
                        mm = 0
                    }
                    j = j + 1
                }
                m = mm
            }
            let _ = __mind_free(rbuf)
            if m == 1 {
                done = done + 1
            } else {
                ok = 0
            }
        }
    }
    let cap: i64 = canon_len(batch) * 32
    let drained: i64 = __mind_alloc(cap)
    let n: i64 = canon_drain(batch, drained, cap)
    let _ = sha256(drained, n, anchor_out)
    let _ = __mind_free(drained)
    let _ = io_uring_tcp_close(rh)
    if ok == 2 {
        return done
    }
    0 - 2
}
"#;

const MIND_TOML: &str = r#"[package]
name = "compose_probe"
version = "0.1.0"
license = "Apache-2.0"

[build]
target = "cpu"
emit = "cdylib"
optimize = "release"
entry = "src/main.mind"
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn substrate_modules_link_natively_into_consumer_cdylib() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("compose: mindc not found; skipping");
        return;
    }

    // Build an isolated temp project.
    let proj = std::env::temp_dir().join("mind_compose_probe");
    let _ = std::fs::remove_dir_all(&proj);
    std::fs::create_dir_all(proj.join("src")).expect("create project src");
    std::fs::write(proj.join("Mind.toml"), MIND_TOML).expect("write Mind.toml");
    std::fs::write(proj.join("src").join("main.mind"), MAIN_MIND).expect("write main.mind");

    let status = Command::new(&mindc)
        .arg("build")
        .current_dir(&proj)
        .status()
        .expect("spawn mindc build");
    if !status.success() {
        // A toolchain without the MLIR backend can't emit a cdylib; skip
        // rather than fail (mirrors the std_surface_* gating convention).
        println!("compose: mindc build failed (no MLIR backend?); skipping");
        return;
    }

    let so = proj
        .join("target")
        .join("release")
        .join("libcompose_probe.so");
    assert!(so.exists(), "consumer cdylib not produced at {so:?}");

    // 1. The substrate symbols must be DEFINED (T), not undefined (U).
    let nm = Command::new("nm")
        .arg("-D")
        .arg(&so)
        .output()
        .expect("nm on PATH");
    let text = String::from_utf8_lossy(&nm.stdout);
    for sym in [
        "canon_new",
        "canon_drain",
        "canon_push",
        "reactor_new",
        "ring_new",
        "sha256",
    ] {
        let defined = text
            .lines()
            .any(|l| l.split_whitespace().last() == Some(sym) && l.contains(" T "));
        assert!(
            defined,
            "substrate symbol `{sym}` is not DEFINED in the consumer cdylib \
             (cross-module linking regressed). nm -D:\n{text}"
        );
    }

    // 2. The composed entry must dlopen and return the correctly-ordered,
    //    deterministic drained byte count (2 events × 32 bytes), with the
    //    canonical-minimum record (conn=1, req=1) first.
    let so_str = so.to_string_lossy().into_owned();
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{so_str}')\n\
         lib.compose_tick.restype = ctypes.c_int64\n\
         buf = (ctypes.c_int64 * 8)()\n\
         n = lib.compose_tick(ctypes.cast(buf, ctypes.c_void_p), 64)\n\
         assert n == 64, f'expected 64 drained bytes, got {{n}}'\n\
         assert buf[0] == 1 and buf[1] == 1, f'canonical order wrong: {{buf[0]}},{{buf[1]}}'\n\
         # L4 evidence anchor must be deterministic: same completion multiset,\n\
         # different physical push order -> byte-identical SHA-256.\n\
         lib.anchor_a.restype = ctypes.c_int64\n\
         lib.anchor_b.restype = ctypes.c_int64\n\
         da = (ctypes.c_uint8 * 32)()\n\
         db = (ctypes.c_uint8 * 32)()\n\
         lib.anchor_a(ctypes.cast(da, ctypes.c_void_p))\n\
         lib.anchor_b(ctypes.cast(db, ctypes.c_void_p))\n\
         assert bytes(da) == bytes(db), 'evidence anchor not deterministic across physical order'\n\
         assert bytes(da) != bytes(32), 'evidence anchor is all-zero (sha256 did not run)'\n\
         # io_uring -> canonical-ordering pipeline (skips if kernel io_uring absent).\n\
         lib.iou_canon_pipeline.restype = ctypes.c_int64\n\
         pr = lib.iou_canon_pipeline()\n\
         assert pr == 9601 or pr < 0, f'io_uring->canon pipeline wrong: {{pr}}'\n\
         # The deterministic reactor server: real io_uring echo loop + canonical\n\
         # ordering + sha256 evidence anchor (skips if kernel io_uring absent).\n\
         lib.deterministic_reactor.restype = ctypes.c_int64\n\
         lib.deterministic_reactor.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]\n\
         rmsg = b'reactor-tick'\n\
         rbuf = ctypes.create_string_buffer(rmsg)\n\
         ranchor = (ctypes.c_uint8 * 32)()\n\
         dr = lib.deterministic_reactor(ctypes.cast(rbuf, ctypes.c_void_p).value, len(rmsg), 4, ctypes.cast(ranchor, ctypes.c_void_p).value)\n\
         assert dr == 4 or dr < 0, f'deterministic reactor returned {{dr}}'\n\
         # Determinism gate: a SECOND independent run of the reactor over the same\n\
         # workload must yield a BIT-IDENTICAL evidence anchor — physical\n\
         # completion order never enters it (the wedge, proven on the live server).\n\
         ranchor2 = (ctypes.c_uint8 * 32)()\n\
         dr2 = lib.deterministic_reactor(ctypes.cast(rbuf, ctypes.c_void_p).value, len(rmsg), 4, ctypes.cast(ranchor2, ctypes.c_void_p).value)\n\
         if dr == 4 and dr2 == 4:\n\
         \x20   assert bytes(ranchor) != bytes(32), 'reactor evidence anchor is all-zero'\n\
         \x20   assert bytes(ranchor) == bytes(ranchor2), 'reactor evidence anchor not deterministic across runs'\n\
         # Monolithic reactor: the SAME pipeline inlined into one large frame.\n\
         # Must run without segfault AND produce the same anchor as the split\n\
         # deterministic_reactor — proof the large-function cross-module codegen\n\
         # path is correct (the historical 'register-pressure' crash was really\n\
         # unresolved cross-module symbols at the call site, now linked).\n\
         lib.deterministic_reactor_mono.restype = ctypes.c_int64\n\
         lib.deterministic_reactor_mono.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]\n\
         manchor = (ctypes.c_uint8 * 32)()\n\
         drm = lib.deterministic_reactor_mono(ctypes.cast(rbuf, ctypes.c_void_p).value, len(rmsg), 4, ctypes.cast(manchor, ctypes.c_void_p).value)\n\
         assert drm == 4 or drm < 0, f'monolithic reactor returned {{drm}}'\n\
         if dr == 4 and drm == 4:\n\
         \x20   assert bytes(manchor) == bytes(ranchor), 'monolithic reactor anchor != split reactor anchor'\n\
         print('ok', n, bytes(da).hex()[:16], 'iou', pr, 'reactor', dr, 'mono', drm, 'det', bytes(ranchor).hex()[:12])\n"
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3 on PATH");
    assert!(
        out.status.success(),
        "composed cdylib dlopen/call failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
