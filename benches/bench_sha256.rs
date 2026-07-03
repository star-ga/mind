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

//! Pure-MIND SHA-256 (`std/sha256.mind`) — hash-throughput benchmark with an
//! embedded byte-identity correctness gate.
//!
//! ## What is timed
//!
//! `std/sha256.mind` exports one entry point (FIPS 180-4):
//!
//! ```text
//! pub fn sha256(in_addr: i64, in_len: i64, out_addr: i64) -> i64
//! ```
//!
//! It reads `in_len` bytes at `in_addr` and writes the 32-byte big-endian
//! digest to `out_addr` (returns 0). The timed closure calls **only** that
//! function on a pre-allocated input buffer and a fixed 32-byte output buffer;
//! the `.so` is compiled once (lazily) and the input buffers are filled once,
//! outside the measured region. Throughput is reported in `Throughput::Bytes`
//! (message bytes hashed per second) for a 1 KiB and a 64 KiB message.
//!
//! ## Build recipe (self-contained — no combine chain)
//!
//! `std/sha256.mind` has no `import`s: `mindc --emit-shared` exports every
//! `pub fn` in the source file, so the module is compiled **directly**, exactly
//! as `tests/sha256_smoke.rs` does — no `cat`/`grep -v '^import'` composition is
//! needed (contrast the HKDF/AES-GCM/TLS drivers, which compose over sha256).
//!
//! ## Correctness gate before any timing
//!
//! Before the throughput sweep runs, the bench asserts byte-identity against
//! two independent references, so a lowering regression that changed the digest
//! bytes can never be reported as a clean throughput number:
//!
//!   1. The FIPS 180-4 known-answer `SHA-256("abc")` (hardcoded).
//!   2. Every benched buffer re-hashed by the `sha2` crate (RustCrypto) — an
//!      independent oracle over the *exact* bytes that were timed.
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
//! cargo bench --bench bench_sha256 --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// Message sizes swept, in bytes: 1 KiB and 64 KiB.
const SIZES: &[usize] = &[1024, 64 * 1024];

/// Kernel ABI: `sha256(in_addr, in_len, out_addr) -> 0`. Reads `in_len` bytes
/// at `in_addr`, writes the 32-byte big-endian digest to `out_addr`. Matches
/// `std/sha256.mind` and `tests/sha256_smoke.rs::Sha256Fn` exactly (three i64
/// addresses/length, i64 return).
type Sha256Fn = unsafe extern "C" fn(i64, i64, i64) -> i64;

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

/// Compile `std/sha256.mind` to a temp `.so` once. Returns `None` (self-skip)
/// if the MLIR toolchain is shadowed or `mindc` is not built — same contract as
/// the gated test/bench harnesses. Self-contained: the module has no imports,
/// so it is fed to `mindc --emit-shared` directly (no combine chain).
fn build_sha256_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_sha256: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_sha256: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        // Compile std/sha256.mind directly — no wrapper, no composition needed
        // (the module has no `import`s; mindc --emit-shared exports all pub fns).
        let src_path = manifest_dir().join("std").join("sha256.mind");
        if !src_path.exists() {
            eprintln!("bench_sha256: std/sha256.mind not found at {src_path:?}; skipping");
            return None;
        }
        let so_path = std::env::temp_dir().join("mind_bench_sha256.so");
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
                eprintln!("bench_sha256: mindc --emit-shared failed for std/sha256.mind; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Deterministic pseudo-random input fill (LCG) so the hashed message is a
/// representative byte distribution rather than all-zeros, and is reproducible
/// across runs. Not cryptographic — just a stable, non-trivial input.
fn make_input(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect()
}

/// One MIND SHA-256 call over `input`, returning the 32-byte digest. `rc` must
/// be 0 (the module's success contract).
fn run_sha256(lib: &Library, input: &[u8]) -> [u8; 32] {
    let f: Symbol<Sha256Fn> = unsafe { lib.get(b"sha256\0").expect("sha256 symbol") };
    let mut out = [0u8; 32];
    let rc = unsafe {
        f(
            input.as_ptr() as i64,
            input.len() as i64,
            out.as_mut_ptr() as i64,
        )
    };
    assert_eq!(rc, 0, "sha256 kernel returned {rc} (expected 0)");
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Byte-identity gate: FIPS known-answer + independent `sha2`-crate oracle over
/// every buffer that will be timed. Panics on any mismatch so a throughput
/// number for a drifted digest can never be reported as a clean result.
fn assert_byte_identity(lib: &Library, inputs: &[(usize, Vec<u8>)]) {
    // (1) FIPS 180-4 known answer: SHA-256("abc").
    let abc = run_sha256(lib, b"abc");
    let expected_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        hex(&abc),
        expected_abc,
        "bench_sha256: SHA-256(\"abc\") drifted from FIPS 180-4 known answer"
    );

    // (2) Independent oracle over the exact bytes to be timed.
    for (len, input) in inputs {
        let got = run_sha256(lib, input);
        let mut h = Sha256::new();
        h.update(input);
        let oracle: [u8; 32] = h.finalize().into();
        assert_eq!(
            got, oracle,
            "bench_sha256: MIND digest for {len}-byte input diverged from the sha2-crate oracle"
        );
    }
    eprintln!(
        "bench_sha256: byte-identity VERIFIED (FIPS \"abc\" + sha2-crate oracle on all sizes)"
    );
}

fn bench_sha256(c: &mut Criterion) {
    let Some(so) = build_sha256_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_sha256: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen sha256 .so") };

    // Build the timed inputs once (outside the measured region), seeded per size.
    let inputs: Vec<(usize, Vec<u8>)> = SIZES
        .iter()
        .map(|&len| (len, make_input(len, 0x5EED_0000_0000_0000 ^ len as u64)))
        .collect();

    // Correctness gate first: a throughput number for a wrong digest is a lie.
    assert_byte_identity(&lib, &inputs);

    let f: Symbol<Sha256Fn> = unsafe { lib.get(b"sha256\0").expect("sha256 symbol") };

    let mut group = c.benchmark_group("sha256");
    for (len, input) in &inputs {
        group.throughput(Throughput::Bytes(*len as u64));
        let label = if *len >= 1024 {
            format!("{}KiB", len / 1024)
        } else {
            format!("{len}B")
        };
        let mut out = [0u8; 32];
        group.bench_with_input(BenchmarkId::new("hash", label), input, |bencher, inp| {
            bencher.iter(|| {
                let rc = unsafe {
                    f(
                        black_box(inp.as_ptr() as i64),
                        black_box(inp.len() as i64),
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
    name = bench_sha256_group;
    // Real measurement, not a 5s quick run: warm up + ample sampling so the
    // microsecond-scale hashes land with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_sha256
}
criterion_main!(bench_sha256_group);
