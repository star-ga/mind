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

//! AES-128-GCM AEAD seal throughput — pure-MIND `std/aes_gcm.mind`.
//!
//! ## What this measures
//!
//! `aes128_gcm_encrypt` is the authenticated-encryption seal: AES-128 in
//! counter mode over the plaintext plus a GHASH tag over AAD ‖ ciphertext,
//! all implemented byte-for-byte in pure MIND (`std/aes_gcm.mind`, 0 imports —
//! a genuinely self-contained module, so the `.so` is a single
//! `mindc std/aes_gcm.mind --emit-shared out.so` with no combine chain). The
//! sweep times the seal on a 1 KiB and a 64 KiB plaintext and reports
//! `Throughput::Bytes` so the result reads directly in MiB/s — the natural
//! axis for a stream cipher / AEAD (bytes processed per second), the analogue
//! of the GMAC/s axis the GEMM bench uses.
//!
//! ## Additive, self-skipping, correctness-gated
//!
//! Adds **nothing** to `src/` — it shells out to the already-built `mindc`
//! `--emit-shared` exactly like `det_matmul_q16.rs` and the gated crypto
//! drivers (`tests/crypto_vectors_driver.py`). Same fail-safe contract: it
//! self-skips (prints, registers no benchmarks, exits clean) when the MLIR
//! toolchain (`mlir-opt` / `mlir-translate` / `clang`) is shadowed or `mindc`
//! is not built — it never panics the bench for a missing toolchain.
//!
//! Before any timing it runs a **known-answer gate**: McGrew/Viega GCM spec
//! Test Case 2 (all-zero 128-bit key, all-zero 96-bit IV, one 16-byte zero
//! block, no AAD) whose ciphertext/tag are pinned in the spec and cross-checked
//! against pyca in `crypto_vectors_driver.py`. If the seal output has drifted
//! from those bytes the bench panics — a throughput number for a wrong AEAD
//! output would be a lie.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_aes_gcm --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// Plaintext sizes swept, in bytes: 1 KiB and 64 KiB.
const SIZES: &[usize] = &[1024, 65536];

/// AES-128-GCM seal ABI (byte-for-byte the driver's argtypes: 8 × c_int64,
/// restype c_int64):
/// `aes128_gcm_encrypt(key, iv, pt, pt_len, aad, aad_len, ct_out, tag_out) -> 0`.
/// `key` is 16 bytes, `iv` 12 bytes, `pt`/`ct_out` `pt_len` bytes, `tag_out`
/// 16 bytes. All args are addresses (`ptr as i64`) or lengths.
type GcmEncFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64;

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

/// Compile `std/aes_gcm.mind` to a temp `.so` once. The module is
/// self-contained (0 imports), so this is a direct single-file
/// `mindc std/aes_gcm.mind --emit-shared out.so` — no dependency combine chain.
/// Returns `None` (self-skip) if the MLIR toolchain is shadowed or `mindc` is
/// not built — same contract as the gated test harnesses.
fn build_aes_gcm_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_aes_gcm: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_aes_gcm: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let src_path = manifest_dir().join("std").join("aes_gcm.mind");
        if !src_path.exists() {
            eprintln!("bench_aes_gcm: std/aes_gcm.mind not found; skipping");
            return None;
        }
        let so_path = std::env::temp_dir().join("mind_bench_aes_gcm.so");
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
                eprintln!("bench_aes_gcm: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Thin wrapper over the seal. `aad` is passed with a valid (>=1 byte) backing
/// pointer even when `aad_len == 0`, mirroring the driver's `buf()` semantics
/// so the address argument is never a dangling/null pointer.
fn seal(
    f: &Symbol<GcmEncFn>,
    key: &[u8],
    iv: &[u8],
    pt: &[u8],
    aad: &[u8],
    ct: &mut [u8],
    tag: &mut [u8],
) -> i64 {
    unsafe {
        (**f)(
            key.as_ptr() as i64,
            iv.as_ptr() as i64,
            pt.as_ptr() as i64,
            pt.len() as i64,
            aad.as_ptr() as i64,
            aad.len() as i64,
            ct.as_mut_ptr() as i64,
            tag.as_mut_ptr() as i64,
        )
    }
}

/// Known-answer gate — McGrew/Viega GCM spec Test Case 2 (also verified against
/// pyca in `tests/crypto_vectors_driver.py`): all-zero 128-bit key, all-zero
/// 96-bit IV, a single 16-byte zero plaintext block, no AAD. Panics (fails the
/// bench run) if the seal output has drifted from the pinned bytes, so a
/// throughput number can never be reported for a wrong AEAD output.
fn assert_gcm_kat(lib: &Library) {
    let f: Symbol<GcmEncFn> = unsafe {
        lib.get(b"aes128_gcm_encrypt")
            .expect("aes128_gcm_encrypt symbol")
    };

    let key = [0u8; 16];
    let iv = [0u8; 12];
    let pt = [0u8; 16];
    // aad_len == 0, but keep a valid 1-byte backing buffer for the address arg.
    let aad = [0u8; 1];
    let mut ct = [0u8; 16];
    let mut tag = [0u8; 16];

    let rc = seal(&f, &key, &iv, &pt, &aad[..0], &mut ct, &mut tag);
    assert_eq!(rc, 0, "aes128_gcm_encrypt returned {rc} (expected 0)");

    // GCM spec TC2 pinned ciphertext + tag.
    let exp_ct: [u8; 16] = [
        0x03, 0x88, 0xda, 0xce, 0x60, 0xb6, 0xa3, 0x92, 0xf3, 0x28, 0xc2, 0xb9, 0x71, 0xb2, 0xfe,
        0x78,
    ];
    let exp_tag: [u8; 16] = [
        0xab, 0x6e, 0x47, 0xd4, 0x2c, 0xec, 0x13, 0xbd, 0xf5, 0x3a, 0x67, 0xb2, 0x12, 0x57, 0xbd,
        0xdf,
    ];
    assert_eq!(
        ct, exp_ct,
        "GCM-TC2 ciphertext drifted from the pinned reference — AES-GCM seal is miscompiling"
    );
    assert_eq!(
        tag, exp_tag,
        "GCM-TC2 tag drifted from the pinned reference — GHASH/tag path is miscompiling"
    );
    eprintln!("bench_aes_gcm: known-answer gate VERIFIED (GCM spec TC2 ciphertext+tag)");
}

fn bench_aes_gcm(c: &mut Criterion) {
    let Some(so) = build_aes_gcm_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_aes_gcm: seal unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen aes_gcm .so") };

    // Correctness gate first: a throughput number for a wrong AEAD output is a lie.
    assert_gcm_kat(&lib);

    let f: Symbol<GcmEncFn> = unsafe {
        lib.get(b"aes128_gcm_encrypt")
            .expect("aes128_gcm_encrypt symbol")
    };

    // Fixed 128-bit key / 96-bit IV; empty AAD (still a valid backing pointer).
    let key = [0u8; 16];
    let iv = [0u8; 12];
    let aad = [0u8; 1];

    let mut group = c.benchmark_group("aes128_gcm_encrypt");
    for &n in SIZES {
        group.throughput(Throughput::Bytes(n as u64));

        // Deterministic, non-trivial plaintext (byte ramp) so the seal does real
        // work across the whole buffer; ciphertext/tag scratch allocated once,
        // outside the timed closure.
        let pt: Vec<u8> = (0..n).map(|i| (i & 0xff) as u8).collect();
        let mut ct = vec![0u8; n];
        let mut tag = [0u8; 16];

        group.bench_with_input(
            BenchmarkId::new("seal", format!("{n}B")),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let rc = unsafe {
                        f(
                            black_box(key.as_ptr() as i64),
                            black_box(iv.as_ptr() as i64),
                            black_box(pt.as_ptr() as i64),
                            black_box(pt.len() as i64),
                            black_box(aad.as_ptr() as i64),
                            black_box(0i64),
                            black_box(ct.as_mut_ptr() as i64),
                            black_box(tag.as_mut_ptr() as i64),
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
    name = bench_aes_gcm_group;
    // Real measurement, not a 5s quick run: warm up + ample sampling so the
    // seal throughput lands with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_aes_gcm
}
criterion_main!(bench_aes_gcm_group);
