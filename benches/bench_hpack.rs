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

//! HPACK header-block decode throughput — pure-MIND `std/hpack.mind` (RFC 7541).
//!
//! ## What is timed
//!
//! `hpack_decode(block, len, state, out, out_cap)` decompressing one complete
//! HTTP/2 header block. The reported axis is `Throughput::Bytes(block_len)` —
//! wire bytes of the compressed header block consumed per second (the natural
//! "bytes decompressed-from" metric for a codec). A small size sweep repeats a
//! fixed representative field group so the same code path is measured at 32 B,
//! 128 B, 512 B and 2 KiB block sizes. A secondary micro-bench times the
//! Appendix-B Huffman string decoder (`hpack_huff_decode`), the compute-heavy
//! inner routine, also on a bytes axis.
//!
//! ## Why the chosen block is safe to reuse across timed iterations
//!
//! `hpack_decode` mutates the caller-owned dynamic-table state **only** for
//! field representations that carry incremental indexing (RFC 7541 §6.2.1) or a
//! §6.3 size update. The benched block is built exclusively from
//! **non-mutating** representations drawn verbatim from RFC 7541 Appendix C.2:
//!   - C.2.2  `04 …`  Literal Header Field *without* Indexing (§6.2.2)
//!   - C.2.3  `10 …`  Literal Header Field *Never* Indexed   (§6.2.3)
//!   - C.2.4  `82`     Indexed Header Field, static entry 2   (§6.1)
//!
//! None of these insert into, evict from, or resize the dynamic table, so a
//! single `hpack_dyn_init`'d state buffer stays byte-for-byte constant and the
//! decode is idempotent — every timed iteration decodes identical input from an
//! identical starting state, exactly like a fresh connection would. Larger
//! sizes simply concatenate this 3-field group (a valid HPACK block is the
//! concatenation of field representations), so they remain non-mutating too.
//!
//! ## Correctness gate before any timing
//!
//! Mirroring `det_matmul_q16.rs::assert_byte_identity`, the bench first decodes
//! the base block and the Huffman vector and **panics** unless the output
//! exactly matches the RFC 7541 published values — a lowering regression that
//! corrupted the decode can never be reported as a clean throughput number.
//!
//! ## Additive, self-skipping
//!
//! Adds nothing to `src/`. Shells out to `mindc std/hpack.mind --emit-shared`
//! (the module is self-contained — no imports, only `__mind_*` intrinsics — so
//! there is **no** combine chain). Self-skips (prints + registers no
//! benchmarks) when the MLIR toolchain (`mlir-opt`/`mlir-translate`/`clang`) is
//! shadowed or `mindc` is not built — the same fail-safe contract as the gated
//! test/bench harnesses; it never panics the run on a missing toolchain.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface" --bin mindc
//! cargo bench --bench bench_hpack --no-default-features
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::Library;

/// Dynamic-table state header size in bytes (see `std/hpack.mind` layout: five
/// i64 fields at offsets 0/8/16/24/32, then the arena). The caller allocates
/// `STATE_HDR + max_size`.
const STATE_HDR: usize = 40;

/// Dynamic-table maximum size the state buffer is sized for. The benched block
/// never inserts, so the arena stays empty; this only sets the `hpack_dyn_init`
/// ceiling.
const MAX_SIZE: usize = 4096;

/// Output buffer capacity for decoded `(name_len u16, name, value_len u16,
/// value)` records — generous vs the largest benched block's ~3.4 KiB of pairs.
const OUT_CAP: usize = 64 * 1024;

/// One non-mutating RFC 7541 Appendix C.2 field group, concatenated:
///   C.2.2 `04 0c …`  (:path, /sample/path)   literal WITHOUT indexing
///   C.2.3 `10 08 …`  (password, secret)       literal NEVER indexed
///   C.2.4 `82`       (:method, GET)            indexed static entry 2
/// 32 wire bytes → 3 header fields, zero dynamic-table mutation.
const BASE_BLOCK_HEX: &str = concat!(
    "040c2f73616d706c652f70617468",       // C.2.2  :path: /sample/path
    "100870617373776f726406736563726574", // C.2.3  password: secret
    "82"                                  // C.2.4  :method: GET
);

/// Expected decode of one base group (RFC 7541 Appendix C.2.2-C.2.4).
const BASE_EXPECTED: &[(&str, &str)] = &[
    (":path", "/sample/path"),
    ("password", "secret"),
    (":method", "GET"),
];

/// Repeat counts for the block-size sweep (× the 32-byte base group).
const REPEATS: &[usize] = &[1, 4, 16, 64];

/// RFC 7541 C.4.1 Huffman string "www.example.com" (12 wire bytes → 15 out).
const HUFF_HEX: &str = "f1e3c2e5f23a6ba0ab90f4ff";
const HUFF_EXPECTED: &str = "www.example.com";

/// FFI ABI — every argument and the return are `i64`, exactly as the official
/// driver (`tests/hpack_driver.py`) declares: `argtypes = [c_int64]*nargs`,
/// `restype = c_int64`. Buffer arguments are raw addresses (`ptr as i64`).
///
/// `hpack_dyn_init(state_addr, max_size) -> 0`
type HpackInitFn = unsafe extern "C" fn(i64, i64) -> i64;
/// `hpack_decode(block_addr, block_len, state_addr, out_addr, out_cap)
///   -> pairs_decoded, or -1 on malformed input / overflow (fail closed)`
type HpackDecodeFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
/// `hpack_huff_decode(src_addr, src_len, dst_addr, dst_cap)
///   -> decoded_byte_count, or -1 on invalid padding / EOS / overflow`
type HpackHuffFn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;

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

/// Decode an ASCII hex string (whitespace ignored) to bytes.
fn hex_bytes(s: &str) -> Vec<u8> {
    let clean: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert!(clean.len() % 2 == 0, "hpack bench: odd-length hex literal");
    clean
        .chunks(2)
        .map(|c| {
            let hi = (c[0] as char).to_digit(16).expect("hex digit") as u8;
            let lo = (c[1] as char).to_digit(16).expect("hex digit") as u8;
            (hi << 4) | lo
        })
        .collect()
}

/// Compile the self-contained `std/hpack.mind` to a temp `.so` **once**.
/// Returns `None` (self-skip) if the MLIR toolchain is shadowed or `mindc` is
/// not built — the same contract as the gated harnesses. No combine chain:
/// `hpack.mind` has no imports, so it emits a shared object directly.
fn build_hpack_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_hpack: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_hpack: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface\" --bin mindc`; skipping"
            );
            return None;
        };
        let src = manifest_dir().join("std").join("hpack.mind");
        if !src.exists() {
            eprintln!("bench_hpack: std/hpack.mind not found; skipping");
            return None;
        }
        let so_path = std::env::temp_dir().join("mind_bench_hpack.so");
        let status = Command::new(&mindc)
            .args([
                src.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => Some(so_path),
            _ => {
                eprintln!("bench_hpack: mindc --emit-shared std/hpack.mind failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// A freshly-initialised dynamic-table state buffer (`STATE_HDR + MAX_SIZE`
/// bytes, `hpack_dyn_init`'d). Kept out of every timed region.
fn new_state(init: HpackInitFn) -> Vec<u8> {
    let mut state = vec![0u8; STATE_HDR + MAX_SIZE];
    let rc = unsafe { init(state.as_mut_ptr() as i64, MAX_SIZE as i64) };
    assert_eq!(rc, 0, "hpack_dyn_init returned {rc} (expected 0)");
    state
}

/// Parse `npairs` `(name_len u16 LE, name, value_len u16 LE, value)` records
/// from a decode output buffer — the same wire layout the driver's `parse_out`
/// walks, used here only in the correctness gate.
fn parse_pairs(raw: &[u8], npairs: usize) -> Vec<(String, String)> {
    let mut pairs = Vec::with_capacity(npairs);
    let mut p = 0usize;
    for _ in 0..npairs {
        let nl = u16::from_le_bytes([raw[p], raw[p + 1]]) as usize;
        p += 2;
        let name = String::from_utf8_lossy(&raw[p..p + nl]).into_owned();
        p += nl;
        let vl = u16::from_le_bytes([raw[p], raw[p + 1]]) as usize;
        p += 2;
        let value = String::from_utf8_lossy(&raw[p..p + vl]).into_owned();
        p += vl;
        pairs.push((name, value));
    }
    pairs
}

/// Correctness gate: decode the base block and the Huffman vector and PANIC
/// unless the output matches RFC 7541 exactly. A throughput number for a
/// corrupted decode would be a lie — this makes a lowering regression fail the
/// bench run rather than post a clean-looking measurement.
fn assert_decode_correct(init: HpackInitFn, decode: HpackDecodeFn, huff: HpackHuffFn) {
    // (1) full-block decode vs the published (name, value) list.
    let mut state = new_state(init);
    let block = hex_bytes(BASE_BLOCK_HEX);
    let mut out = vec![0u8; OUT_CAP];
    let n = unsafe {
        decode(
            block.as_ptr() as i64,
            block.len() as i64,
            state.as_mut_ptr() as i64,
            out.as_mut_ptr() as i64,
            out.len() as i64,
        )
    };
    assert_eq!(
        n,
        BASE_EXPECTED.len() as i64,
        "hpack_decode base block: expected {} pairs, got {n}",
        BASE_EXPECTED.len()
    );
    let pairs = parse_pairs(&out, n as usize);
    let got: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    assert_eq!(
        got, BASE_EXPECTED,
        "hpack_decode base block diverged from RFC 7541 Appendix C.2"
    );

    // (2) Huffman string decode vs the published Appendix-B string.
    let hin = hex_bytes(HUFF_HEX);
    let mut hout = vec![0u8; 256];
    let hn = unsafe {
        huff(
            hin.as_ptr() as i64,
            hin.len() as i64,
            hout.as_mut_ptr() as i64,
            hout.len() as i64,
        )
    };
    assert_eq!(
        hn,
        HUFF_EXPECTED.len() as i64,
        "hpack_huff_decode: expected {} bytes, got {hn}",
        HUFF_EXPECTED.len()
    );
    assert_eq!(
        &hout[..hn as usize],
        HUFF_EXPECTED.as_bytes(),
        "hpack_huff_decode diverged from RFC 7541 Appendix B"
    );

    eprintln!("bench_hpack: correctness VERIFIED (RFC 7541 C.2 header block + Appendix B Huffman)");
}

fn bench_hpack(c: &mut Criterion) {
    let Some(so) = build_hpack_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_hpack: hpack .so unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen hpack .so") };

    // Resolve the three entry points once as plain (Copy) fn pointers, so the
    // timed closures capture no borrow of `lib`.
    let init: HpackInitFn = unsafe { *lib.get(b"hpack_dyn_init").expect("hpack_dyn_init symbol") };
    let decode: HpackDecodeFn = unsafe { *lib.get(b"hpack_decode").expect("hpack_decode symbol") };
    let huff: HpackHuffFn = unsafe {
        *lib.get(b"hpack_huff_decode")
            .expect("hpack_huff_decode symbol")
    };

    // Correctness gate first — panics on any drift from the RFC vectors.
    assert_decode_correct(init, decode, huff);

    let base = hex_bytes(BASE_BLOCK_HEX);

    let mut group = c.benchmark_group("bench_hpack");

    // --- hpack_decode throughput sweep over block size (bytes decompressed-from) ---
    for &r in REPEATS {
        // Concatenate the non-mutating base group `r` times → a valid, larger
        // header block. Built once, outside the timed region.
        let mut block = Vec::with_capacity(base.len() * r);
        for _ in 0..r {
            block.extend_from_slice(&base);
        }
        // One state + one out buffer per shape, both allocated/initialised
        // outside the loop; the block never mutates the table so the state is
        // constant across every timed iteration.
        let mut state = new_state(init);
        let mut out = vec![0u8; OUT_CAP];

        group.throughput(Throughput::Bytes(block.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("decode_block", format!("{}B", block.len())),
            &block,
            |bencher, blk| {
                bencher.iter(|| {
                    let n = unsafe {
                        decode(
                            black_box(blk.as_ptr() as i64),
                            black_box(blk.len() as i64),
                            black_box(state.as_mut_ptr() as i64),
                            black_box(out.as_mut_ptr() as i64),
                            black_box(out.len() as i64),
                        )
                    };
                    black_box(n);
                });
            },
        );
    }

    // --- hpack_huff_decode throughput (compute-heavy inner routine, bytes in) ---
    {
        let hin = hex_bytes(HUFF_HEX);
        let mut hout = vec![0u8; 256];
        group.throughput(Throughput::Bytes(hin.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("huff_decode", HUFF_EXPECTED),
            &hin,
            |bencher, src| {
                bencher.iter(|| {
                    let n = unsafe {
                        huff(
                            black_box(src.as_ptr() as i64),
                            black_box(src.len() as i64),
                            black_box(hout.as_mut_ptr() as i64),
                            black_box(hout.len() as i64),
                        )
                    };
                    black_box(n);
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = bench_hpack_group;
    // Real measurement window (not the 5s quick default): HPACK decode of these
    // blocks is sub-microsecond to a few microseconds, so warm up + sample amply
    // for tight CIs.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_hpack
}
criterion_main!(bench_hpack_group);
