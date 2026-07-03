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

//! Pure-MIND SHA3-256 (FIPS 202) — hash-throughput benchmark for the
//! self-contained `std/keccak.mind` sponge, with an embedded known-answer
//! correctness gate so a throughput number can never be reported for a build
//! that computes the wrong digest.
//!
//! ## What is timed
//!
//! `keccak_sha3_256(in_addr, in_len, out32) -> 0` — the pure-MIND FIPS 202
//! sponge (rate 136, domain-separation byte 0x06) absorbing `in_len` message
//! bytes at `in_addr` and writing the 32-byte digest at `out32`. The sweep
//! measures **hash throughput** (`Throughput::Bytes`) at 1 KiB (single-block
//! plus a partial block) and 64 KiB (deep multi-block absorb, ~482 rate blocks)
//! so criterion reports MiB/s directly.
//!
//! The ABI is byte-for-byte the one exercised by `tests/keccak_driver.py`
//! (three `i64` args — input address, input length, output address — matching
//! that driver's `argtypes = [c_int64] * 3` and its
//! `K.keccak_sha3_256(addr(ib), len(msg), addr(ob))` call). `std/keccak.mind`
//! is self-contained (`// Self-contained: no imports, only __mind_* intrinsics`,
//! line 3), so the `.so` is built directly with a single
//! `mindc std/keccak.mind --emit-shared out.so` — no dependency-combine chain.
//!
//! ## Additive, self-skipping, correctness-gated
//!
//! This bench adds **nothing** to `src/`. Like the gated GEMM/FFT harnesses it
//! self-skips (registers no benchmarks, exits clean) when the MLIR toolchain
//! (`mlir-opt` / `mlir-translate` / `clang`) is shadowed or `mindc` is not
//! built, because it must shell out to `mindc --emit-shared`. The whole file
//! compiles unconditionally (cargo always builds bench targets); the skip is a
//! runtime check.
//!
//! Before any timing runs, `assert_kat` hashes the empty string and `"abc"`
//! and compares against the published FIPS 202 SHA3-256 known-answer vectors
//! (the same constants cross-checked against Python `hashlib` in
//! `keccak_driver.py`). A digest mismatch panics — so a lowering regression
//! that corrupted the sponge could never surface as a clean MiB/s figure.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_keccak --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// Message sizes the throughput sweep exercises (bytes). 1 KiB spans a full
/// rate block (136 B) plus a partial tail; 64 KiB is a deep multi-block absorb.
const INPUT_SIZES: &[(usize, &str)] = &[(1024, "1KiB"), (65536, "64KiB")];

/// Digest length of SHA3-256, in bytes.
const DIGEST_LEN: usize = 32;

/// Kernel ABI: `keccak_sha3_256(in_addr, in_len, out32) -> 0`. Reads `in_len`
/// message bytes at `in_addr` (one byte per offset) and writes the 32-byte
/// digest at `out32`. Three `i64` args — byte-for-byte the driver's argtypes.
type Sha3_256Fn = unsafe extern "C" fn(i64, i64, i64) -> i64;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Path to the self-contained pure-MIND keccak module in the repo.
fn keccak_src() -> PathBuf {
    manifest_dir().join("std").join("keccak.mind")
}

fn mindc_path() -> Option<PathBuf> {
    let dbg = manifest_dir().join("target").join("debug").join("mindc");
    if dbg.exists() {
        return Some(dbg);
    }
    let rel = manifest_dir().join("target").join("release").join("mindc");
    if rel.exists() { Some(rel) } else { None }
}

/// Compile `std/keccak.mind` to a temp `.so` once. Returns `None` (self-skip)
/// if the MLIR toolchain is shadowed or `mindc` is not built — same fail-safe
/// contract as the gated GEMM/FFT harnesses. The module is self-contained, so
/// this is a single `--emit-shared` of the source file, no combine chain.
fn build_keccak_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!(
                    "bench_keccak: {tool} not on PATH; skipping (toolchain shadowed)"
                );
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_keccak: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let src = keccak_src();
        if !src.exists() {
            eprintln!("bench_keccak: {} not found; skipping", src.display());
            return None;
        }
        let so_path = std::env::temp_dir().join("mind_bench_keccak.so");
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
                eprintln!("bench_keccak: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Hash `msg` with the loaded pure-MIND SHA3-256 and return the 32-byte digest.
/// Mirrors the driver: input buffer is padded to >= 1 byte so the pointer is
/// always valid even for the empty message (`in_len == 0`, no bytes read).
fn sha3_256(lib: &Library, msg: &[u8]) -> [u8; DIGEST_LEN] {
    let f: Symbol<Sha3_256Fn> =
        unsafe { lib.get(b"keccak_sha3_256").expect("keccak_sha3_256 symbol") };
    // >= 1 byte so `as_ptr()` is a valid, non-dangling address (driver parity).
    let input: Vec<u8> = if msg.is_empty() {
        vec![0u8]
    } else {
        msg.to_vec()
    };
    let mut out = [0u8; DIGEST_LEN];
    let rc = unsafe {
        f(
            input.as_ptr() as i64,
            msg.len() as i64,
            out.as_mut_ptr() as i64,
        )
    };
    assert_eq!(rc, 0, "keccak_sha3_256 returned {rc} (expected 0)");
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// FIPS 202 SHA3-256 known-answer gate. Runs before any timing: hashes the
/// empty string and `"abc"` and compares to the published digests (identical to
/// the vectors `keccak_driver.py` cross-checks against Python `hashlib`). Panics
/// on mismatch so a wrong-digest build can never be reported as a clean number.
fn assert_kat(lib: &Library) {
    const CASES: &[(&[u8], &str)] = &[
        (
            b"",
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
        ),
        (
            b"abc",
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532",
        ),
    ];
    for &(msg, expected) in CASES {
        let got = hex(&sha3_256(lib, msg));
        assert_eq!(
            got,
            expected,
            "bench_keccak: SHA3-256 KAT mismatch for {:?}\n computed={got}\n expected={expected}\n\
             The pure-MIND keccak sponge is miscomputing — refusing to report throughput.",
            String::from_utf8_lossy(msg)
        );
    }
    eprintln!("bench_keccak: SHA3-256 known-answer vectors VERIFIED (empty, \"abc\")");
}

/// Deterministic message of `n` bytes (content is timing-irrelevant; a fixed
/// repeating pattern keeps runs reproducible — mirrors the driver's 0xa3 fill).
fn make_input(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(0xa3))
        .collect()
}

fn bench_keccak(c: &mut Criterion) {
    let Some(so) = build_keccak_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_keccak: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen keccak .so") };

    // Correctness gate first: a throughput number for a wrong digest is a lie.
    assert_kat(&lib);

    let f: Symbol<Sha3_256Fn> =
        unsafe { lib.get(b"keccak_sha3_256").expect("keccak_sha3_256 symbol") };

    let mut group = c.benchmark_group("keccak_sha3_256");
    for &(n, label) in INPUT_SIZES {
        // Throughput = message bytes absorbed per hash → criterion reports MiB/s.
        group.throughput(Throughput::Bytes(n as u64));

        // Allocate the input and output buffers ONCE, outside the timed loop.
        let input = make_input(n);
        let mut out = [0u8; DIGEST_LEN];

        group.bench_with_input(BenchmarkId::new("hash", label), &n, |bencher, &len| {
            bencher.iter(|| {
                let rc = unsafe {
                    f(
                        black_box(input.as_ptr() as i64),
                        black_box(len as i64),
                        black_box(out.as_mut_ptr() as i64),
                    )
                };
                black_box(rc);
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = keccak;
    // Real measurement time, not a 5s quick run: warm up + ample sampling so the
    // hashes land with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_keccak
}
criterion_main!(keccak);
