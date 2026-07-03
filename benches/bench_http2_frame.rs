// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Part of the MIND project (Machine Intelligence Native Design).

//! HTTP/2 binary framing throughput — pure-MIND `std/http2_frame.mind`
//! (RFC 9113 §3.4 / §4.1 / §6; HTTP-provable-stack Phase 2.2).
//!
//! ## What is timed
//!
//! Four hot-path framing primitives, exactly the ABI
//! `tests/http2_frame_driver.py` exercises (every argument is an `i64` address
//! or length; every return is an `i64`):
//!
//! ```text
//! pub fn http2_frame_write_header(len, type, flags, sid, out) -> 9
//! pub fn http2_frame_parse(buf, buf_len, out_fields) -> 9+len | -1   (fail-closed)
//! pub fn http2_settings_write(pairs, count, out) -> 9 + 6*count
//! pub fn http2_settings_parse(payload, payload_len, out_pairs, cap) -> count | -1
//! ```
//!
//!   * `write_header` / `frame_parse` are the fixed 9-byte §4.1 frame envelope
//!     (24-bit length ‖ 8-bit type ‖ 8-bit flags ‖ 1-bit R + 31-bit stream id).
//!     Both are O(1): the timed closure touches only the 9 header bytes, so both
//!     report `Throughput::Bytes(9)` (header bytes processed per second — for an
//!     O(1) op this reads as an ops/sec proxy, not an inflated MiB/s).
//!   * `settings_write` / `settings_parse` are O(N) over the §6.5 payload
//!     (6-byte entries: 16-bit identifier ‖ 32-bit value). They are swept over
//!     1 / 8 / 64 entries and report `Throughput::Bytes` of the wire bytes
//!     produced (`9 + 6N`) / consumed (`6N`) — an honest bytes/sec axis that
//!     scales with the work.
//!
//! The `.so` is compiled once (lazily); all input buffers and output scratch are
//! allocated once, outside the measured region; the timed closure calls **only**
//! the framing function in `black_box`.
//!
//! ## Build recipe (self-contained — no combine chain)
//!
//! `std/http2_frame.mind` has no `import`s (only `__mind_*` intrinsics), so
//! `mindc --emit-shared` exports every `pub fn` directly — the module is compiled
//! in one step exactly as `tests/http2_frame_driver.py` and `bench_sha256.rs` /
//! `bench_aes_gcm.rs` do, with **no** `cat`/`grep -v '^import'` composition
//! (contrast the HKDF/TLS drivers, which compose over `std/sha256.mind`).
//!
//! ## Correctness gate before any timing
//!
//! Before the throughput sweep runs, the bench asserts byte-identity against the
//! hand-derived RFC 9113 §4.1 wire layout (the same independent oracle
//! `http2_frame_driver.py` cross-checks hyperframe against), so a lowering
//! regression that changed the framed bytes can never be reported as a clean
//! throughput number:
//!
//!   1. `write_header` for the driver's large-DATA case (length `0xABCDEF`,
//!      type `0`, flags `1`, the max 31-bit stream id) equals the hand-derived
//!      §4.1 bytes; the reserved bit is sent as zero (`sid 0xffffffff` masked).
//!   2. Every benched SETTINGS size round-trips: `settings_write` →
//!      `frame_parse` (header fields + total) → `settings_parse` == the input
//!      pairs, with the serialized header matching the §4.1 layout.
//!   3. Fail-closed: a frame whose declared 24-bit length exceeds the buffer is
//!      rejected (`-1`) — the §4.1 bounds contract that must never read past the
//!      buffer.
//!
//! Panics (fails the bench run) on any mismatch.
//!
//! ## Self-skipping
//!
//! Like the gated test/bench harnesses, it self-skips (registers no benchmarks,
//! exits clean) when the MLIR toolchain (`mlir-opt` / `mlir-translate` /
//! `clang`) is absent from PATH or `mindc` is not built, because it must shell
//! out to `mindc --emit-shared`. The whole file compiles unconditionally (cargo
//! always builds bench targets); the skip is a runtime check.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_http2_frame --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// SETTINGS entry counts swept (RFC 9113 §6.5): a minimal 1-entry frame, a
/// realistic ~6-entry connection preface neighbourhood (8), and a 64-entry
/// stress upper bound.
const PAIR_COUNTS: &[usize] = &[1, 8, 64];

/// `http2_frame_write_header(length, ftype, flags, stream_id, out_addr) -> 9`.
/// Writes the 9-byte §4.1 frame header; length masked to 24 bits, type/flags to
/// 8, stream id to 31 (R bit sent as zero). Driver arg count: 5.
type WriteHeaderFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

/// `http2_frame_parse(buf_addr, buf_len, out_fields_addr) -> 9+length | -1`.
/// Buffer-bounded, fail-closed header parse; writes 4 i64 fields
/// [length, type, flags, stream_id] to `out_fields_addr`. Driver arg count: 3.
type FrameParseFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// `http2_settings_write(pairs_addr, count, out_addr) -> 9 + 6*count`.
/// Serializes a complete SETTINGS frame; `pairs_addr` holds `count` entries of
/// two consecutive native i64 [identifier, value] (16 bytes each). Driver arg
/// count: 3.
type SettingsWriteFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// `http2_settings_parse(payload_addr, payload_len, out_pairs_addr, out_pairs_cap)
/// -> count | -1`. Parses a §6.5 SETTINGS payload; writes each entry as two
/// consecutive native i64 [identifier, value]. Driver arg count: 4.
type SettingsParseFn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn mindc_path() -> Option<PathBuf> {
    let dbg = manifest_dir().join("target").join("debug").join("mindc");
    if dbg.exists() {
        return Some(dbg);
    }
    let rel = manifest_dir().join("target").join("release").join("mindc");
    if rel.exists() { Some(rel) } else { None }
}

/// Compile `std/http2_frame.mind` to a temp `.so` once. The module is
/// self-contained (0 imports, only `__mind_*` intrinsics), so this is a direct
/// single-file `mindc std/http2_frame.mind --emit-shared out.so` — no dependency
/// combine chain. Returns `None` (self-skip) if the MLIR toolchain is shadowed
/// or `mindc` is not built — same contract as the gated test/bench harnesses.
fn build_http2_frame_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_http2_frame: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_http2_frame: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let src_path = manifest_dir().join("std").join("http2_frame.mind");
        if !src_path.exists() {
            eprintln!("bench_http2_frame: std/http2_frame.mind not found at {src_path:?}; skipping");
            return None;
        }
        let so_path = std::env::temp_dir().join("mind_bench_http2_frame.so");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => Some(so_path),
            _ => {
                eprintln!("bench_http2_frame: mindc --emit-shared failed for std/http2_frame.mind; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Hand-derived RFC 9113 §4.1 frame header: 24-bit Length (big-endian) ‖ 8-bit
/// Type ‖ 8-bit Flags ‖ 1-bit R (zero) + 31-bit Stream Identifier (big-endian).
/// Byte-for-byte `http2_frame_driver.py::rfc9113_header` — the independent
/// oracle the driver cross-checks hyperframe against.
fn rfc9113_header(length: u64, ftype: u8, flags: u8, sid: u32) -> [u8; 9] {
    let l = (length & 0x00FF_FFFF) as u32;
    let s = sid & 0x7FFF_FFFF;
    [
        (l >> 16) as u8,
        (l >> 8) as u8,
        l as u8,
        ftype,
        flags,
        (s >> 24) as u8,
        (s >> 16) as u8,
        (s >> 8) as u8,
        s as u8,
    ]
}

/// Deterministic SETTINGS entries for `n` pairs, as the flat native-i64
/// [identifier, value, …] layout the module reads (`__mind_load_i64` at 8-byte
/// strides). Identifiers are masked to 16 bits and values to 32 bits so the
/// wire round-trip is exact (the parser reconstructs a 16-bit id and 32-bit
/// value). Seeded off `n` for a reproducible, non-trivial distribution.
fn make_pairs(n: usize) -> Vec<i64> {
    let mut v = Vec::with_capacity(2 * n);
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15 ^ (n as u64);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let ident = ((s >> 48) & 0xFFFF) as i64;
        let value = ((s >> 16) & 0xFFFF_FFFF) as i64;
        v.push(ident);
        v.push(value);
    }
    v
}

/// Correctness gate — RFC 9113 §4.1 hand-derived layout KAT + SETTINGS
/// round-trip over every benched size + fail-closed bounds check. Panics (fails
/// the bench run) on any mismatch so a throughput number for miscompiled framing
/// can never be reported as a clean result.
fn assert_correctness(lib: &Library, sizes: &[(usize, Vec<i64>)]) {
    let write_header: Symbol<WriteHeaderFn> = unsafe {
        lib.get(b"http2_frame_write_header\0")
            .expect("http2_frame_write_header symbol")
    };
    let frame_parse: Symbol<FrameParseFn> = unsafe {
        lib.get(b"http2_frame_parse\0")
            .expect("http2_frame_parse symbol")
    };
    let settings_write: Symbol<SettingsWriteFn> = unsafe {
        lib.get(b"http2_settings_write\0")
            .expect("http2_settings_write symbol")
    };
    let settings_parse: Symbol<SettingsParseFn> = unsafe {
        lib.get(b"http2_settings_parse\0")
            .expect("http2_settings_parse symbol")
    };

    // (1) frame-header write KAT — driver case (b): large DATA header, length
    // 0xABCDEF, type 0, flags 1 (END_STREAM), the max 31-bit stream id.
    let mut hdr = [0u8; 9];
    let rc = unsafe { write_header(0x00AB_CDEF, 0x0, 0x1, 0x7FFF_FFFF, hdr.as_mut_ptr() as i64) };
    assert_eq!(
        rc, 9,
        "http2_frame_write_header must return 9 (header size)"
    );
    assert_eq!(
        hdr,
        rfc9113_header(0x00AB_CDEF, 0x0, 0x1, 0x7FFF_FFFF),
        "frame header write drifted from the hand-derived RFC 9113 §4.1 layout"
    );
    // Reserved bit MUST be sent as zero (§4.1): sid 0xffffffff -> 0x7fffffff.
    let rc = unsafe { write_header(8, 0x6, 0x0, 0xFFFF_FFFF, hdr.as_mut_ptr() as i64) };
    assert_eq!(rc, 9);
    assert_eq!(
        hdr,
        rfc9113_header(8, 0x6, 0x0, 0x7FFF_FFFF),
        "R bit not zeroed on send (sid 0xffffffff must serialize as 0x7fffffff)"
    );

    // (2) SETTINGS round-trip over every benched size: write -> frame_parse
    // (header fields + total) -> settings_parse == the input pairs.
    for (n, pairs_in) in sizes {
        let n = *n;
        let total_len = 9 + 6 * n;
        let mut frame = vec![0u8; total_len];
        let total = unsafe {
            settings_write(
                pairs_in.as_ptr() as i64,
                n as i64,
                frame.as_mut_ptr() as i64,
            )
        };
        assert_eq!(
            total, total_len as i64,
            "http2_settings_write size for {n} pairs (expected 9 + 6*{n})"
        );
        // Serialized header matches the §4.1 layout: SETTINGS type 0x4, flags 0,
        // stream id 0, length 6N (connection-level frame, §6.5).
        assert_eq!(
            &frame[..9],
            &rfc9113_header((6 * n) as u64, 0x4, 0x0, 0)[..],
            "SETTINGS frame header layout drifted ({n} pairs)"
        );
        let mut fields = [0i64; 4];
        let consumed =
            unsafe { frame_parse(frame.as_ptr() as i64, total, fields.as_mut_ptr() as i64) };
        assert_eq!(consumed, total, "http2_frame_parse consumed ({n} pairs)");
        assert_eq!(
            fields,
            [(6 * n) as i64, 4, 0, 0],
            "http2_frame_parse header fields ({n} pairs)"
        );
        let mut pairs_out = vec![0i64; 2 * n];
        let cnt = unsafe {
            settings_parse(
                frame.as_ptr() as i64 + 9,
                fields[0],
                pairs_out.as_mut_ptr() as i64,
                n as i64,
            )
        };
        assert_eq!(
            cnt, n as i64,
            "http2_settings_parse entry count ({n} pairs)"
        );
        assert_eq!(
            &pairs_out, pairs_in,
            "SETTINGS round-trip pairs mismatch ({n})"
        );
    }

    // (3) fail-closed: declared 24-bit length exceeds the buffer -> -1 (§4.1
    // bounds contract; driver case (e)). Header claims a 100-byte payload; only
    // 11 payload bytes exist in the 20-byte buffer.
    let mut over = rfc9113_header(100, 0x0, 0x0, 1).to_vec();
    over.extend_from_slice(&[0u8; 11]);
    let mut sink = [0i64; 4];
    let rc = unsafe {
        frame_parse(
            over.as_ptr() as i64,
            over.len() as i64,
            sink.as_mut_ptr() as i64,
        )
    };
    assert_eq!(
        rc, -1,
        "over-length frame must be rejected (-1, fail-closed)"
    );

    eprintln!(
        "bench_http2_frame: correctness VERIFIED (RFC 9113 §4.1 header KAT + \
         SETTINGS round-trip on all sizes + fail-closed bounds)"
    );
}

fn bench_http2_frame(c: &mut Criterion) {
    let Some(so) = build_http2_frame_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_http2_frame: framing kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen http2_frame .so") };

    // Build the timed inputs once (outside any measured region), seeded per size.
    let inputs: Vec<(usize, Vec<i64>)> = PAIR_COUNTS.iter().map(|&n| (n, make_pairs(n))).collect();

    // Correctness gate first: a throughput number for miscompiled framing is a lie.
    assert_correctness(&lib, &inputs);

    let write_header: Symbol<WriteHeaderFn> = unsafe {
        lib.get(b"http2_frame_write_header\0")
            .expect("http2_frame_write_header symbol")
    };
    let frame_parse: Symbol<FrameParseFn> = unsafe {
        lib.get(b"http2_frame_parse\0")
            .expect("http2_frame_parse symbol")
    };
    let settings_write: Symbol<SettingsWriteFn> = unsafe {
        lib.get(b"http2_settings_write\0")
            .expect("http2_settings_write symbol")
    };
    let settings_parse: Symbol<SettingsParseFn> = unsafe {
        lib.get(b"http2_settings_parse\0")
            .expect("http2_settings_parse symbol")
    };

    let mut group = c.benchmark_group("http2_frame");

    // --- 1. frame-header write (§4.1) — O(1), 9 header bytes produced. ---
    {
        group.throughput(Throughput::Bytes(9));
        let mut hdr = [0u8; 9];
        group.bench_function("write_header", |bencher| {
            bencher.iter(|| {
                let rc = unsafe {
                    write_header(
                        black_box(0x00AB_CDEFi64),
                        black_box(0x0i64),
                        black_box(0x1i64),
                        black_box(0x7FFF_FFFFi64),
                        black_box(hdr.as_mut_ptr() as i64),
                    )
                };
                black_box(rc);
            });
        });
    }

    // --- 2. frame parse (§4.1, buffer-bounded, fail-closed) — O(1): only the
    // 9-byte header is decoded, so throughput is reported on those 9 bytes. The
    // probe frame is a realistic 8-entry SETTINGS frame the writer produced. ---
    {
        let probe_pairs = make_pairs(8);
        let mut probe_frame = vec![0u8; 9 + 6 * 8];
        let total = unsafe {
            settings_write(
                probe_pairs.as_ptr() as i64,
                8,
                probe_frame.as_mut_ptr() as i64,
            )
        };
        assert_eq!(total, probe_frame.len() as i64);
        let probe_len = probe_frame.len() as i64;
        let mut fields = [0i64; 4];
        group.throughput(Throughput::Bytes(9));
        group.bench_function("frame_parse", |bencher| {
            bencher.iter(|| {
                let rc = unsafe {
                    frame_parse(
                        black_box(probe_frame.as_ptr() as i64),
                        black_box(probe_len),
                        black_box(fields.as_mut_ptr() as i64),
                    )
                };
                black_box(rc);
            });
        });
    }

    // --- 3. SETTINGS write (§6.5) — O(N), 9 + 6N wire bytes produced. ---
    for (n, pairs) in &inputs {
        let n = *n;
        group.throughput(Throughput::Bytes((9 + 6 * n) as u64));
        let mut out = vec![0u8; 9 + 6 * n];
        group.bench_with_input(
            BenchmarkId::new("settings_write", format!("{n}p")),
            pairs,
            |bencher, p| {
                bencher.iter(|| {
                    let rc = unsafe {
                        settings_write(
                            black_box(p.as_ptr() as i64),
                            black_box(n as i64),
                            black_box(out.as_mut_ptr() as i64),
                        )
                    };
                    black_box(rc);
                });
            },
        );
    }

    // --- 4. SETTINGS parse (§6.5) — O(N), 6N payload bytes consumed. The wire
    // payload is the writer's output sliced past the 9-byte header. ---
    for (n, pairs) in &inputs {
        let n = *n;
        let mut frame = vec![0u8; 9 + 6 * n];
        unsafe { settings_write(pairs.as_ptr() as i64, n as i64, frame.as_mut_ptr() as i64) };
        let payload: Vec<u8> = frame[9..].to_vec();
        let payload_len = payload.len() as i64; // == 6N
        let mut out_pairs = vec![0i64; 2 * n];
        group.throughput(Throughput::Bytes((6 * n) as u64));
        group.bench_with_input(
            BenchmarkId::new("settings_parse", format!("{n}p")),
            &payload,
            |bencher, pl| {
                bencher.iter(|| {
                    let rc = unsafe {
                        settings_parse(
                            black_box(pl.as_ptr() as i64),
                            black_box(payload_len),
                            black_box(out_pairs.as_mut_ptr() as i64),
                            black_box(n as i64),
                        )
                    };
                    black_box(rc);
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = bench_http2_frame_group;
    // Real measurement, not a 5s quick run: warm up + ample sampling so the
    // sub-microsecond framing ops land with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_http2_frame
}
criterion_main!(bench_http2_frame_group);
