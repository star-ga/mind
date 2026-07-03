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

//! ML-KEM-768 (FIPS 203, "Kyber") ops/sec benchmark for the pure-MIND
//! `std/mlkem768.mind` module — KeyGen / Encaps / Decaps.
//!
//! ## What is timed
//!
//! Three post-quantum KEM primitives, each an ops/sec (elem/s) measurement —
//! plain criterion iters, no throughput byte-count (a KEM operation is one
//! discrete unit of work, not a stream of bytes):
//!
//!   * `mlkem768_keygen(rand64, ek, dk)` — 64-byte `d‖z` randomness in;
//!     1184-byte encapsulation key + 2400-byte decapsulation key out.
//!   * `mlkem768_encaps(ek, m, ct, ss)` — 1184-byte `ek` + 32-byte `m` in;
//!     1088-byte ciphertext + 32-byte shared secret out.
//!   * `mlkem768_decaps(dk, ct, ss)` — 2400-byte `dk` + 1088-byte `ct` in;
//!     32-byte shared secret out (includes the FO re-encrypt, so it does an
//!     encaps-equivalent internally — the heaviest of the three).
//!
//! The ABI is byte-for-byte the one `tests/mlkem768_driver.py` drives: every
//! argument is an `i64` address (buffer pointer), every function returns `i64`
//! (always 0). Input/output buffers are allocated ONCE, outside the timed
//! region; the timed closure calls only the crypto op inside `black_box`.
//!
//! ## Build recipe — COMBINE CHAIN (not self-contained)
//!
//! `std/mlkem768.mind` begins with `import std.keccak;` (G/H/J/XOF/PRF are all
//! SHA3/SHAKE from `std/keccak.mind`). `std.keccak` is **not** one of the
//! `include_str!`-bundled stdlib modules in `src/project/stdlib.rs` (only
//! `std.sha256` is), so a standalone `mindc std/mlkem768.mind --emit-shared`
//! cannot resolve the keccak bodies and would emit unresolved `keccak_*`
//! references. This bench therefore reproduces the driver-style combine chain
//!
//! ```text
//! cat std/keccak.mind                       > combined.mind   # dependency first
//! grep -v '^import' std/mlkem768.mind       >> combined.mind   # importer, imports stripped
//! mindc combined.mind --emit-shared out.so
//! ```
//!
//! in Rust: read both sources from `$CARGO_MANIFEST_DIR/std`, concatenate
//! `keccak.mind` verbatim followed by `mlkem768.mind` with every line that
//! starts with `import` removed, write the combined source to a temp file, and
//! `mindc --emit-shared` it once. keccak's helpers are all `keccak_*` and
//! mlkem's are `mlkem_*` / `mlkem768_*`, so nothing name-collides.
//!
//! ## Self-skipping, correctness-gated (mirrors `det_matmul_q16.rs`)
//!
//! Like the gated harnesses this self-skips (registers no benchmarks, exits
//! clean) when the MLIR toolchain (`mlir-opt` / `mlir-translate` / `clang`) is
//! shadowed or `mindc` is not built — it must shell out to `mindc
//! --emit-shared`. The whole file compiles unconditionally (cargo always builds
//! bench targets); the skip is a runtime check.
//!
//! When the `.so` DID build, a self-contained round-trip correctness gate runs
//! before any timing: KeyGen → Encaps → Decaps and assert the decapsulated
//! secret equals the encapsulated secret (the FIPS 203 KEM invariant), and that
//! it is not all-zero. This needs no external KAT vectors, yet a lowering
//! regression that silently broke the KEM could never be reported as a clean
//! ops/sec number.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_mlkem768 --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use libloading::Library;

/// ML-KEM-768 wire-format sizes (FIPS 203 Table 3, k = 3).
const RAND64_LEN: usize = 64; // keygen randomness d‖z
const M_LEN: usize = 32; // encaps randomness m
const EK_LEN: usize = 1184; // encapsulation key
const DK_LEN: usize = 2400; // decapsulation key
const CT_LEN: usize = 1088; // ciphertext
const SS_LEN: usize = 32; // shared secret K

/// `mlkem768_keygen(rand64, ek, dk) -> 0` — all args are `i64` buffer
/// addresses. rand64: 64-byte `d‖z` in; ek: 1184-byte out; dk: 2400-byte out.
type KeygenFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
/// `mlkem768_encaps(ek, m, ct, ss) -> 0`. ek: 1184-byte in; m: 32-byte in;
/// ct: 1088-byte out; ss: 32-byte out.
type EncapsFn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
/// `mlkem768_decaps(dk, ct, ss) -> 0`. dk: 2400-byte in; ct: 1088-byte in;
/// ss: 32-byte out (FO implicit-reject folded in).
type DecapsFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

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

/// Combine `std/keccak.mind` (dependency, no imports) with
/// `std/mlkem768.mind` (imports stripped), compile the result to a temp `.so`
/// once. Returns `None` (self-skip) if the MLIR toolchain is shadowed or
/// `mindc` is not built — same contract as the gated test harnesses.
fn build_mlkem_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_mlkem768: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_mlkem768: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };

        // Combine chain: keccak first (defines the SHA3/SHAKE bodies mlkem768
        // calls), then mlkem768 with its `import std.keccak;` line stripped —
        // std.keccak is NOT an include_str!-bundled stdlib module, so its source
        // must be inlined into the same compilation unit. Byte-for-byte the
        // `cat keccak > c; grep -v '^import' mlkem768 >> c` recipe.
        let std_dir = manifest_dir().join("std");
        let keccak_src = match std::fs::read_to_string(std_dir.join("keccak.mind")) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bench_mlkem768: cannot read std/keccak.mind ({e}); skipping");
                return None;
            }
        };
        let mlkem_src = match std::fs::read_to_string(std_dir.join("mlkem768.mind")) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bench_mlkem768: cannot read std/mlkem768.mind ({e}); skipping");
                return None;
            }
        };
        let mut combined = String::with_capacity(keccak_src.len() + mlkem_src.len() + 2);
        combined.push_str(&keccak_src);
        combined.push('\n');
        for line in mlkem_src.lines() {
            // `grep -v '^import'`: drop lines that start (no leading space) with
            // `import`; comments (`// ...import...`) and code are kept verbatim.
            if line.starts_with("import") {
                continue;
            }
            combined.push_str(line);
            combined.push('\n');
        }

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_mlkem768.mind");
        let so_path = dir.join("mind_bench_mlkem768.so");
        if std::fs::write(&src_path, &combined).is_err() {
            eprintln!("bench_mlkem768: could not write combined source; skipping");
            return None;
        }
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
                eprintln!("bench_mlkem768: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Deterministic KAT-style inputs (mirror the driver's `d = 0..32`,
/// `z = 32..64`, `m = 64..96`).
fn rand64_bytes() -> Vec<u8> {
    (0u16..RAND64_LEN as u16).map(|i| i as u8).collect()
}
fn m_bytes() -> Vec<u8> {
    (RAND64_LEN as u16..(RAND64_LEN + M_LEN) as u16)
        .map(|i| i as u8)
        .collect()
}

/// Self-contained round-trip correctness gate — no external vectors. KeyGen →
/// Encaps → Decaps and assert the two shared secrets match (the FIPS 203 KEM
/// invariant), and the secret is not all-zero. Panics (fails the bench run) on
/// any mismatch, so an ops/sec number for a miscompiled KEM can never be
/// reported as clean.
fn assert_roundtrip(keygen: KeygenFn, encaps: EncapsFn, decaps: DecapsFn) {
    let rand64 = rand64_bytes();
    let m = m_bytes();
    let mut ek = vec![0u8; EK_LEN];
    let mut dk = vec![0u8; DK_LEN];
    unsafe {
        keygen(
            rand64.as_ptr() as i64,
            ek.as_mut_ptr() as i64,
            dk.as_mut_ptr() as i64,
        );
    }
    let mut ct = vec![0u8; CT_LEN];
    let mut ss_encaps = vec![0u8; SS_LEN];
    unsafe {
        encaps(
            ek.as_ptr() as i64,
            m.as_ptr() as i64,
            ct.as_mut_ptr() as i64,
            ss_encaps.as_mut_ptr() as i64,
        );
    }
    let mut ss_decaps = vec![0u8; SS_LEN];
    unsafe {
        decaps(
            dk.as_ptr() as i64,
            ct.as_ptr() as i64,
            ss_decaps.as_mut_ptr() as i64,
        );
    }
    assert_eq!(
        ss_encaps, ss_decaps,
        "ML-KEM-768 round-trip failed: decapsulated secret != encapsulated secret \
         (a KEM-invariant break — do NOT trust any throughput number for this build)"
    );
    assert!(
        ss_encaps.iter().any(|&b| b != 0),
        "ML-KEM-768 shared secret is all-zero — kernel is a no-op / did not write output"
    );
}

fn bench_mlkem768(c: &mut Criterion) {
    let Some(so) = build_mlkem_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_mlkem768: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen ML-KEM-768 .so") };
    // Deref each Symbol to a plain Copy fn pointer; the pointers stay valid for
    // as long as `lib` is loaded (the whole function), so the timed closures can
    // capture them freely without holding a Symbol borrow across three benches.
    let keygen: KeygenFn = unsafe { *lib.get::<KeygenFn>(b"mlkem768_keygen").expect("keygen sym") };
    let encaps: EncapsFn = unsafe { *lib.get::<EncapsFn>(b"mlkem768_encaps").expect("encaps sym") };
    let decaps: DecapsFn = unsafe { *lib.get::<DecapsFn>(b"mlkem768_decaps").expect("decaps sym") };

    // Correctness gate first: an ops/sec number for a KEM whose round-trip is
    // broken would be a lie. Panics on any mismatch.
    assert_roundtrip(keygen, encaps, decaps);

    let rand64 = rand64_bytes();
    let m = m_bytes();

    // Pre-build a valid key pair + ciphertext OUTSIDE every timed region, so the
    // encaps/decaps benches feed real, well-formed inputs and each timed closure
    // calls JUST the one crypto op.
    let mut ek = vec![0u8; EK_LEN];
    let mut dk = vec![0u8; DK_LEN];
    unsafe {
        keygen(
            rand64.as_ptr() as i64,
            ek.as_mut_ptr() as i64,
            dk.as_mut_ptr() as i64,
        );
    }
    let mut ct = vec![0u8; CT_LEN];
    let mut ss = vec![0u8; SS_LEN];
    unsafe {
        encaps(
            ek.as_ptr() as i64,
            m.as_ptr() as i64,
            ct.as_mut_ptr() as i64,
            ss.as_mut_ptr() as i64,
        );
    }

    let mut group = c.benchmark_group("mlkem768");

    // KeyGen — (rand64) -> (ek, dk).
    {
        let mut ek_out = vec![0u8; EK_LEN];
        let mut dk_out = vec![0u8; DK_LEN];
        group.bench_function("keygen", |bn| {
            bn.iter(|| {
                let rc = unsafe {
                    keygen(
                        black_box(rand64.as_ptr() as i64),
                        ek_out.as_mut_ptr() as i64,
                        dk_out.as_mut_ptr() as i64,
                    )
                };
                black_box(rc);
            });
        });
    }

    // Encaps — (ek, m) -> (ct, ss).
    {
        let mut ct_out = vec![0u8; CT_LEN];
        let mut ss_out = vec![0u8; SS_LEN];
        group.bench_function("encaps", |bn| {
            bn.iter(|| {
                let rc = unsafe {
                    encaps(
                        black_box(ek.as_ptr() as i64),
                        black_box(m.as_ptr() as i64),
                        ct_out.as_mut_ptr() as i64,
                        ss_out.as_mut_ptr() as i64,
                    )
                };
                black_box(rc);
            });
        });
    }

    // Decaps — (dk, ct) -> ss (includes the FO re-encryption check).
    {
        let mut ss_out = vec![0u8; SS_LEN];
        group.bench_function("decaps", |bn| {
            bn.iter(|| {
                let rc = unsafe {
                    decaps(
                        black_box(dk.as_ptr() as i64),
                        black_box(ct.as_ptr() as i64),
                        ss_out.as_mut_ptr() as i64,
                    )
                };
                black_box(rc);
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = mlkem768;
    // Real measurement, not a 5s quick run: KEM ops are tens-to-hundreds of µs
    // (decaps does a full FO re-encrypt), so warm up + ample sampling for tight
    // CIs and a stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_mlkem768
}
criterion_main!(mlkem768);
