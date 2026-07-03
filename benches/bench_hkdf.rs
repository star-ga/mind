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

//! Deterministic pure-MIND crypto throughput benchmark for `std/hkdf.mind`:
//! **HMAC-SHA256** (RFC 2104) and **HKDF-Expand** (RFC 5869 §2.3), both compiled
//! to native code by `mindc --emit-shared` and driven over the C ABI.
//!
//! ## What is timed
//!
//!   1. `hmac_sha256(key, key_len, msg, msg_len, out)` — one HMAC-SHA256 over a
//!      swept message length. `Throughput::Bytes(msg_len)` → criterion reports
//!      MiB/s of message hashed (two SHA-256 compressions of fixed overhead plus
//!      the message-length-dependent inner hash).
//!   2. `hkdf_expand(prk, prk_len, info, info_len, length, okm)` — expand a
//!      32-byte PRK to a swept OKM length. `Throughput::Bytes(length)` → MiB/s of
//!      key material produced (`ceil(L/32)` chained HMAC-SHA256 blocks).
//!
//! `hkdf_extract` is just a single `hmac_sha256` call (RFC 5869 §2.2) so it is not
//! separately benched — its cost is exactly the `hmac_sha256` curve at
//! `msg_len = len(IKM)`.
//!
//! ## Combined-source build (dependency order, imports stripped)
//!
//! `std/hkdf.mind` is NOT self-contained: it `import std.sha256;` and calls the
//! `sha256` entry point in `std/sha256.mind`. The `.so` is built exactly as the
//! official-vector driver `tests/tls13_keyschedule_driver.py` documents — cat the
//! dependency first, then append the importer with its `import` line stripped:
//! ```text
//! cat std/sha256.mind                          >  combined.mind
//! grep -v '^import std.sha256;' std/hkdf.mind  >> combined.mind
//! mindc combined.mind --emit-shared out.so     # needs the mlir-build toolchain
//! ```
//! This bench performs that same concatenation in Rust (read both modules from
//! the repo, strip the `import std.sha256` line from hkdf, join), writes the
//! combined source to a temp file, and compiles it once.
//!
//! ## Correctness belt (a throughput number for wrong bytes would be a lie)
//!
//! Before any timing, the compiled `.so` is checked against:
//!   * **RFC 4231** Test Case 2 known-answer HMAC-SHA256 (key `"Jefe"`,
//!     msg `"what do ya want for nothing?"`), and
//!   * **RFC 5869** Test Case 1 known-answer HKDF-Expand (published PRK, info,
//!     L = 42), and
//!   * an independent from-scratch Rust HMAC/HKDF oracle (built only on
//!     `sha2::Sha256`) over the exact benched inputs.
//! Any mismatch panics the bench run — mirroring the byte-identity gate in
//! `det_matmul_q16.rs`.
//!
//! ## Additive, self-skipping
//!
//! Adds nothing to `src/`. Like the other `--emit-shared` benches it self-skips
//! (registers no benchmarks, exits clean) when the MLIR toolchain
//! (`mlir-opt` / `mlir-translate` / `clang`) is not on PATH or `mindc` is not
//! built — it never panics for a missing toolchain. The whole file compiles
//! unconditionally; the skip is a runtime check.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_hkdf --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds the
//! debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// HMAC-SHA256 message-length sweep (bytes hashed). All >= 64 so the timed
/// closure never dereferences a zero-length buffer.
const HMAC_MSG_SIZES: &[usize] = &[64, 256, 1024, 4096, 16384];

/// HKDF-Expand OKM-length sweep (bytes of key material produced). 8160 = 255*32
/// is the RFC 5869 maximum (`L <= 255*HashLen`).
const HKDF_OKM_SIZES: &[usize] = &[32, 64, 512, 4096, 8160];

// ---------------------------------------------------------------------------
// C-ABI entry points of the compiled std/hkdf.mind (+ std/sha256.mind) .so.
// Every arg is an i64 address or length; every fn returns 0 and writes its
// result into the caller-provided out buffer (matches the ctypes signatures in
// tests/tls13_keyschedule_driver.py).
// ---------------------------------------------------------------------------

/// `hmac_sha256(key_addr, key_len, msg_addr, msg_len, out_addr) -> 0`.
/// Writes the 32-byte MAC to `out_addr`.
type HmacFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

/// `hkdf_expand(prk_addr, prk_len, info_addr, info_len, length, okm_out) -> 0`.
/// Writes `length` bytes of OKM to `okm_out`.
type HkdfExpandFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

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

/// Build the combined `std/sha256.mind` + `std/hkdf.mind` source and compile it
/// to a temp `.so` once. Returns `None` (self-skip) if the MLIR toolchain is
/// shadowed, `mindc` is not built, the module sources cannot be read, or the
/// compile fails — same never-panic contract as `det_matmul_q16.rs`.
fn build_hkdf_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_hkdf: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_hkdf: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };

        // Combine deps in dependency order, stripping the hkdf->sha256 import —
        // byte-for-byte the recipe in tests/tls13_keyschedule_driver.py:
        //   cat std/sha256.mind                          >  combined
        //   grep -v '^import std.sha256;' std/hkdf.mind  >> combined
        let std_dir = manifest_dir().join("std");
        let Ok(sha_src) = std::fs::read_to_string(std_dir.join("sha256.mind")) else {
            eprintln!("bench_hkdf: cannot read std/sha256.mind; skipping");
            return None;
        };
        let Ok(hkdf_src) = std::fs::read_to_string(std_dir.join("hkdf.mind")) else {
            eprintln!("bench_hkdf: cannot read std/hkdf.mind; skipping");
            return None;
        };
        let hkdf_stripped: String = hkdf_src
            .lines()
            .filter(|l| !l.trim_start().starts_with("import std.sha256"))
            .collect::<Vec<_>>()
            .join("\n");
        let combined = format!("{sha_src}\n{hkdf_stripped}\n");

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_hkdf.mind");
        let so_path = dir.join("mind_bench_hkdf.so");
        if std::fs::write(&src_path, combined).is_err() {
            eprintln!("bench_hkdf: could not write combined source; skipping");
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
                eprintln!("bench_hkdf: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

// ---------------------------------------------------------------------------
// Independent from-scratch reference (oracle): HMAC-SHA256 (RFC 2104) and
// HKDF-Expand (RFC 5869), built ONLY on sha2::Sha256 — no `hmac`/`hkdf` crate.
// This is the byte-for-byte analogue of the Python composition (b) the official
// driver validates against the published RFC constants.
// ---------------------------------------------------------------------------

fn ref_sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

fn ref_hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    // K0: key zero-padded to 64 bytes (or SHA-256(key) padded if key > 64).
    let mut k0 = [0u8; 64];
    if key.len() > 64 {
        k0[..32].copy_from_slice(&ref_sha256(key));
    } else {
        k0[..key.len()].copy_from_slice(key);
    }
    let mut inner = Vec::with_capacity(64 + msg.len());
    for &b in &k0 {
        inner.push(b ^ 0x36);
    }
    inner.extend_from_slice(msg);
    let inner_hash = ref_sha256(&inner);
    let mut outer = Vec::with_capacity(96);
    for &b in &k0 {
        outer.push(b ^ 0x5c);
    }
    outer.extend_from_slice(&inner_hash);
    ref_sha256(&outer)
}

fn ref_hkdf_expand(prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
    let n = length.div_ceil(32);
    let mut okm = Vec::with_capacity(n * 32);
    let mut tprev: Vec<u8> = Vec::new();
    for i in 1..=n {
        let mut input = Vec::with_capacity(tprev.len() + info.len() + 1);
        input.extend_from_slice(&tprev);
        input.extend_from_slice(info);
        input.push((i & 0xff) as u8);
        let t = ref_hmac_sha256(prk, &input);
        okm.extend_from_slice(&t);
        tprev = t.to_vec();
    }
    okm.truncate(length);
    okm
}

/// Minimal hex decoder (avoids adding the `hex` crate for a few KAT literals).
fn hexb(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    assert!(bytes.len() % 2 == 0, "odd-length hex literal");
    let nib = |c: u8| -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("bad hex nibble {c}"),
        }
    };
    bytes
        .chunks(2)
        .map(|p| (nib(p[0]) << 4) | nib(p[1]))
        .collect()
}

/// Deterministic byte generator — the same LCG the cross-substrate gate uses, so
/// benched inputs are reproducible across builds/machines.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u32
    }
}
fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut g = Lcg::new(seed);
    (0..n).map(|_| (g.next_u32() & 0xff) as u8).collect()
}

// ---------------------------------------------------------------------------
// Thin FFI call helpers.
// ---------------------------------------------------------------------------

fn call_hmac(lib: &Library, key: &[u8], msg: &[u8]) -> [u8; 32] {
    let f: Symbol<HmacFn> = unsafe { lib.get(b"hmac_sha256").expect("hmac_sha256 symbol") };
    let mut out = [0u8; 32];
    let rc = unsafe {
        f(
            key.as_ptr() as i64,
            key.len() as i64,
            msg.as_ptr() as i64,
            msg.len() as i64,
            out.as_mut_ptr() as i64,
        )
    };
    assert_eq!(rc, 0, "hmac_sha256 returned {rc} (expected 0)");
    out
}

fn call_hkdf_expand(lib: &Library, prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
    let f: Symbol<HkdfExpandFn> = unsafe { lib.get(b"hkdf_expand").expect("hkdf_expand symbol") };
    let mut out = vec![0u8; length];
    let rc = unsafe {
        f(
            prk.as_ptr() as i64,
            prk.len() as i64,
            info.as_ptr() as i64,
            info.len() as i64,
            length as i64,
            out.as_mut_ptr() as i64,
        )
    };
    assert_eq!(rc, 0, "hkdf_expand returned {rc} (expected 0)");
    out
}

/// Correctness gate: published RFC known-answer vectors + an independent Rust
/// oracle over the benched inputs. Panics (fails the run) on any mismatch, so a
/// throughput number can never be reported for wrong output bytes.
fn assert_crypto_correct(lib: &Library) {
    // (1) RFC 4231 Test Case 2 — HMAC-SHA256 known answer.
    let kat_hmac = call_hmac(lib, b"Jefe", b"what do ya want for nothing?");
    let kat_hmac_exp = hexb("5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    assert_eq!(
        kat_hmac.as_slice(),
        kat_hmac_exp.as_slice(),
        "hmac_sha256 failed RFC 4231 Test Case 2 known-answer vector"
    );

    // (2) RFC 5869 Test Case 1 — HKDF-Expand known answer (PRK given, L = 42).
    let rfc_prk = hexb("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5");
    let rfc_info = hexb("f0f1f2f3f4f5f6f7f8f9");
    let rfc_okm = hexb(
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865",
    );
    let kat_okm = call_hkdf_expand(lib, &rfc_prk, &rfc_info, 42);
    assert_eq!(
        kat_okm, rfc_okm,
        "hkdf_expand failed RFC 5869 Test Case 1 known-answer vector"
    );

    // (3) Independent Rust oracle over the exact benched inputs.
    let key = lcg_bytes(32, 0x0000_00A5);
    for &sz in HMAC_MSG_SIZES {
        let msg = lcg_bytes(sz, 0x1234_5678 ^ sz as u64);
        let got = call_hmac(lib, &key, &msg);
        let exp = ref_hmac_sha256(&key, &msg);
        assert_eq!(
            got,
            exp,
            "hmac_sha256(.so) != oracle for msg_len={sz}\n  got={}\n  exp={}",
            hex_of(&got),
            hex_of(&exp)
        );
    }
    let prk = lcg_bytes(32, 0x0000_BEEF);
    let info = lcg_bytes(16, 0x0000_CAFE);
    for &len in HKDF_OKM_SIZES {
        let got = call_hkdf_expand(lib, &prk, &info, len);
        let exp = ref_hkdf_expand(&prk, &info, len);
        assert_eq!(got, exp, "hkdf_expand(.so) != oracle for okm_len={len}");
    }

    eprintln!(
        "bench_hkdf: correctness VERIFIED — RFC 4231 TC2 (HMAC) + RFC 5869 TC1 \
         (HKDF-Expand) + Rust oracle over all benched sizes."
    );
}

fn hex_of(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for &x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn bench_hkdf(c: &mut Criterion) {
    let Some(so) = build_hkdf_so() else {
        eprintln!("bench_hkdf: crypto module unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen hkdf .so") };

    // Correctness gate first — a throughput number for wrong bytes would be a lie.
    assert_crypto_correct(&lib);

    // ---- HMAC-SHA256 throughput over message length (bytes hashed) ----------
    let key = lcg_bytes(32, 0x0000_00A5);
    let mut hgroup = c.benchmark_group("hkdf_hmac_sha256");
    for &sz in HMAC_MSG_SIZES {
        hgroup.throughput(Throughput::Bytes(sz as u64));
        let msg = lcg_bytes(sz, 0x1234_5678 ^ sz as u64);
        let mut out = [0u8; 32];
        let f: Symbol<HmacFn> = unsafe { lib.get(b"hmac_sha256").expect("hmac_sha256 symbol") };
        hgroup.bench_with_input(BenchmarkId::new("msg_bytes", sz), &sz, |bencher, _| {
            bencher.iter(|| {
                let rc = unsafe {
                    f(
                        black_box(key.as_ptr() as i64),
                        black_box(key.len() as i64),
                        black_box(msg.as_ptr() as i64),
                        black_box(msg.len() as i64),
                        black_box(out.as_mut_ptr() as i64),
                    )
                };
                black_box(rc);
            });
        });
    }
    hgroup.finish();

    // ---- HKDF-Expand throughput over OKM length (bytes produced) ------------
    let prk = lcg_bytes(32, 0x0000_BEEF);
    let info = lcg_bytes(16, 0x0000_CAFE);
    let mut egroup = c.benchmark_group("hkdf_expand");
    for &len in HKDF_OKM_SIZES {
        egroup.throughput(Throughput::Bytes(len as u64));
        let mut out = vec![0u8; len];
        let f: Symbol<HkdfExpandFn> =
            unsafe { lib.get(b"hkdf_expand").expect("hkdf_expand symbol") };
        egroup.bench_with_input(BenchmarkId::new("okm_bytes", len), &len, |bencher, _| {
            bencher.iter(|| {
                let rc = unsafe {
                    f(
                        black_box(prk.as_ptr() as i64),
                        black_box(prk.len() as i64),
                        black_box(info.as_ptr() as i64),
                        black_box(info.len() as i64),
                        black_box(len as i64),
                        black_box(out.as_mut_ptr() as i64),
                    )
                };
                black_box(rc);
            });
        });
    }
    egroup.finish();
}

criterion_group! {
    name = bench_hkdf_group;
    // Real measurement time, not a 5s quick run: warm-up + ample sampling so the
    // microsecond-scale HMAC/HKDF calls land with tight CIs.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_hkdf
}
criterion_main!(bench_hkdf_group);
