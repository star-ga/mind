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

//! RSASSA-PSS-VERIFY throughput benchmark for the pure-MIND `std/rsa_pss.mind`
//! module — `rsa_pss_verify_sha256` **ops/sec** (RSA-2048 / SHA-256 / MGF1-SHA256,
//! the `rsa_pss_rsae_sha256` scheme TLS 1.3 uses for CertificateVerify,
//! RFC 8446 §4.4.3; verification per RFC 8017 §8.1.2 / §9.1.2).
//!
//! ## What is timed
//!
//! One full `rsa_pss_verify_sha256` call on a **valid** signature — the whole
//! fail-closed path: RSAVP1 (`s^e mod n`, a 130-limb radix-2^16 2048-bit modexp
//! with `e = 65537`), EMSA-PSS-VERIFY (MGF1-SHA256 dbMask, salt recovery, the
//! final `H == H'` SHA-256 comparison) — ending in ACCEPT (`rc == 1`). The accept
//! path is the most expensive branch (an early length/range reject would skip the
//! modexp), so ops/sec on ACCEPT is the honest headline number for a verify.
//!
//! ## Ground truth for the embedded vector
//!
//! The `(n, e, sig, msg, salt_len)` vector below was produced by pyca/cryptography
//! (`rsa.generate_private_key(65537, 2048)` +
//! `padding.PSS(MGF1(SHA256), salt_length=32)` + `SHA256`) and cross-verified to
//! ACCEPT under pyca — the exact procedure `tests/rsa_pss_driver.py` runs, only
//! frozen here because a criterion bench cannot depend on pyca at run time. The
//! bench asserts the compiled MIND module also ACCEPTs it (`rc == 1`) before
//! taking any measurement; if it does not, the module or the build is wrong and
//! an ops/sec for a not-fully-executed path would be a lie, so the bench prints
//! the reason and self-skips (no measurement) rather than publish a bogus number.
//!
//! ## Build recipe (mirrors the driver, RFC-8017 combine chain)
//!
//! `std/rsa_pss.mind` composes `std/sha256.mind` (the `sha256` entry point) and
//! `std/x509.mind` (the 130-limb RSA-2048 bignum: `modexp_2048` etc.). The
//! standalone `--emit-shared` build combines the three sources with the import
//! lines stripped, byte-for-byte the chain in `tests/rsa_pss_driver.py`:
//! ```text
//! cat std/sha256.mind > combined.mind
//! grep -v '^import std.sha256;' std/x509.mind   >> combined.mind
//! grep -v '^import '            std/rsa_pss.mind >> combined.mind
//! mindc combined.mind --emit-shared rsa_pss.so
//! ```
//!
//! Like the gated harnesses it self-skips when the MLIR toolchain
//! (`mlir-opt` / `mlir-translate` / `clang`) is not on PATH or `mindc` is not
//! built, because it must shell out to `mindc --emit-shared`. The whole file
//! compiles unconditionally (cargo always builds bench targets); the skip is a
//! runtime check — the bench never panics.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_rsa_pss --no-default-features
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// ABI (RFC 8017 §8.1.2): `rsa_pss_verify_sha256(n_addr, n_len, e, sig_addr,
/// sig_len, msg_addr, msg_len, salt_len) -> i64` — returns 1 (valid) or 0
/// (invalid). All eight arguments are `i64` (addresses passed as `ptr as i64`,
/// lengths/`e`/`salt_len` by value), matching `tests/rsa_pss_driver.py`'s
/// `argtypes = [c_int64] * 8`, `restype = c_int64`.
type VerifyFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64;

// ---------------------------------------------------------------------------
// Fixed valid vector — pyca-generated RSA-2048, PSS(MGF1(SHA256), salt_length=32)
// + SHA256, over the same message the driver signs. `n` and `sig` are 256 bytes
// big-endian; `e = 65537`; `salt_len = 32`. pyca ACCEPTs this signature; the
// MIND module must too (asserted before any timing below).
// ---------------------------------------------------------------------------
const N_HEX: &str = "86b55d28bab57be273669f41e6401f308f4c83e91fd69eac8027f865b4afea7628aa686b869a8220aeb089ee005e6a63d45907ac44e54ff60ed54b57b91b824a59764ad99fef3196478c7406942b515153a721981baa0ef7e8e411bffc680b01d9793825e99bb7c1337679008ee2a19cb82b505732cffe9db6c3090ad8dc6231ab22f9be04ada5ce695cb7c1b03d08cf3be398d54e0903756b393992e9044cc51da336c8737d0632820040273f7d4ebe6994835ea472ed225af1226b7ab58e982bf2ca6c313f1b8a8249834dc23edcc5273178c17a776171e02aed8870eeca235a3d87de9f09ef0c5c012355b5fd80f311872cf5e9dfb93bc0c4f23d1d231127";
const SIG_HEX: &str = "418c65b02127821ca2f9720d38df1cd6cffc7471c0ce1e8d123c2b7326bb690f2a7506054f4d0ad6d9f4368e9c407eb19e511651f8c9f407bfd0b516cf84201aad4f71e7bee807c1c39a67ffe4d42f68ad2d085cf382b37f14094a408cd4c525e3945b24fad2888dbc8503a8ed3ce45ff830be44d1675b5500e62de4b1b6ef5fd33f778d61c786f59ea81ce9e431f3c0f0b99979e885b1fa4eae23f211bca6338a58dc68185bdf0788626f06d0c1f1944e852daeb6f13d4dcec1c7472a8b60cd90b8d1232b502865e454356b8b2157d5b5c23d527c1ca67158ee5025d54549b246aa05435c93351251e2921b15da688df423ad14288e043b544835c72d218823";

/// The raw signed message (RFC 8446 §4.4.3 CertificateVerify content is composed
/// by the handshake caller; here the message is signed directly, as the driver
/// does). Byte-for-byte `tests/rsa_pss_driver.py`'s `msg`.
const MSG: &[u8] = b"MIND provable stack: TLS 1.3 CertificateVerify uses rsa_pss_rsae_sha256";

const RSA_PUBLIC_EXPONENT: i64 = 65537;
const RSA_MODULUS_LEN: i64 = 256; // RSA-2048: k = 256 octets.
const PSS_SALT_LEN: i64 = 32; // rsa_pss_rsae_sha256: salt = hLen = 32 (RFC 8446 §4.2.3).

/// Decode a lowercase hex string to bytes (no external dep; the vectors are
/// well-formed even-length constants above).
fn hex_to_bytes(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "hex string must have even length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

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

/// Combine `std/sha256.mind` + `std/x509.mind` + `std/rsa_pss.mind` into one
/// source with the import lines stripped — byte-for-byte the shell combine chain
/// `tests/rsa_pss_driver.py` documents:
///   cat std/sha256.mind > combined
///   grep -v '^import std.sha256;' std/x509.mind   >> combined
///   grep -v '^import '            std/rsa_pss.mind >> combined
/// Returns `None` (→ self-skip) if any source is unreadable.
fn combined_source() -> Option<String> {
    let std_dir = manifest_dir().join("std");
    let sha256 = std::fs::read_to_string(std_dir.join("sha256.mind")).ok()?;
    let x509 = std::fs::read_to_string(std_dir.join("x509.mind")).ok()?;
    let rsa_pss = std::fs::read_to_string(std_dir.join("rsa_pss.mind")).ok()?;

    // grep -v '^import std.sha256;' — x509 only pulls in sha256.
    let x509_stripped: String = x509
        .lines()
        .filter(|l| !l.starts_with("import std.sha256;"))
        .collect::<Vec<_>>()
        .join("\n");
    // grep -v '^import ' — rsa_pss pulls in both sha256 and x509.
    let rsa_stripped: String = rsa_pss
        .lines()
        .filter(|l| !l.starts_with("import "))
        .collect::<Vec<_>>()
        .join("\n");

    Some(format!("{sha256}\n{x509_stripped}\n{rsa_stripped}\n"))
}

/// Build the combined crypto module to a temp `.so` once. Returns `None`
/// (self-skip) if the MLIR toolchain is shadowed, `mindc` is not built, a source
/// is unreadable, or `mindc --emit-shared` fails — same fail-safe contract as the
/// gated test harnesses; never panics the bench.
fn build_rsa_pss_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_rsa_pss: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_rsa_pss: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let Some(src) = combined_source() else {
            eprintln!("bench_rsa_pss: could not read std/{{sha256,x509,rsa_pss}}.mind; skipping");
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_rsa_pss_combined.mind");
        let so_path = dir.join("mind_bench_rsa_pss.so");
        if std::fs::write(&src_path, src).is_err() {
            eprintln!("bench_rsa_pss: could not write combined source; skipping");
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
                eprintln!("bench_rsa_pss: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

fn bench_rsa_pss_verify(c: &mut Criterion) {
    let Some(so) = build_rsa_pss_so() else {
        // Toolchain shadowed / mindc unbuilt / build failed — register nothing.
        eprintln!("bench_rsa_pss: module unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen rsa_pss .so") };
    let verify: Symbol<VerifyFn> = unsafe {
        lib.get(b"rsa_pss_verify_sha256")
            .expect("rsa_pss_verify_sha256 symbol")
    };

    // Decode the embedded valid vector once, outside the timed region.
    let n = hex_to_bytes(N_HEX);
    let sig = hex_to_bytes(SIG_HEX);
    let msg = MSG;
    assert_eq!(
        n.len(),
        RSA_MODULUS_LEN as usize,
        "n must be 256 bytes (RSA-2048)"
    );
    assert_eq!(
        sig.len(),
        RSA_MODULUS_LEN as usize,
        "sig must be 256 bytes (RSA-2048)"
    );

    let n_ptr = n.as_ptr() as i64;
    let sig_ptr = sig.as_ptr() as i64;
    let msg_ptr = msg.as_ptr() as i64;
    let msg_len = msg.len() as i64;

    // Correctness gate: the pyca-valid signature MUST verify (rc == 1). If it
    // does not, the compiled module or the build is wrong — reporting ops/sec for
    // a path that skipped the modexp+MGF1+compare would be a lie. Self-skip
    // (print + return, no measurement) rather than panic or publish a bogus number.
    let rc = unsafe {
        verify(
            n_ptr,
            RSA_MODULUS_LEN,
            RSA_PUBLIC_EXPONENT,
            sig_ptr,
            RSA_MODULUS_LEN,
            msg_ptr,
            msg_len,
            PSS_SALT_LEN,
        )
    };
    if rc != 1 {
        eprintln!(
            "bench_rsa_pss: embedded valid vector did NOT verify (rc={rc}); \
             module/build mismatch — skipping (no measurement)."
        );
        return;
    }
    eprintln!(
        "bench_rsa_pss: accept-path gate VERIFIED (rc==1) — timing full RSA-2048 \
         PSS(MGF1-SHA256, salt=32)+SHA256 verify."
    );

    let mut group = c.benchmark_group("bench_rsa_pss");
    // ops/sec for one full verify — plain iters (not Throughput::Bytes; this is a
    // fixed-work signature check, not a byte stream).
    group.bench_function("rsa_pss_verify_sha256/accept", |b| {
        b.iter(|| {
            let rc = unsafe {
                verify(
                    black_box(n_ptr),
                    black_box(RSA_MODULUS_LEN),
                    black_box(RSA_PUBLIC_EXPONENT),
                    black_box(sig_ptr),
                    black_box(RSA_MODULUS_LEN),
                    black_box(msg_ptr),
                    black_box(msg_len),
                    black_box(PSS_SALT_LEN),
                )
            };
            black_box(rc);
        });
    });
    group.finish();
}

criterion_group! {
    name = bench_rsa_pss;
    // RSA-2048 modexp is heavier than the matmul kernels; keep the wall-clock
    // bounded with a modest sample size while still landing a stable p50.
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(50);
    targets = bench_rsa_pss_verify
}
criterion_main!(bench_rsa_pss);
