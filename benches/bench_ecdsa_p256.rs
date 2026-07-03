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

//! ECDSA P-256 / SHA-256 signature **verification** throughput — ops/sec for the
//! pure-MIND `std/ecdsa_p256.mind` `ecdsa_p256_verify` entry point (FIPS 186-5
//! §6.4.2 / SEC1 §4.1.4, the `ecdsa_secp256r1_sha256` scheme TLS 1.3 uses for
//! CertificateVerify, RFC 8446 §4.4.3).
//!
//! ## What is timed
//!
//! One `ecdsa_p256_verify(qx, qy, r, s, msg, msg_len) -> {0,1}` call per
//! iteration over a **fixed** known-answer vector — the RFC 6979 A.2.5 P-256 /
//! SHA-256 deterministic-ECDSA vector for the message `"sample"` (public
//! constants; `(r, s)` are fixed). The verify path runs the full SHA-256 of the
//! message plus two P-256 scalar multiplications (`u1*G + u2*Q`) in pure-MIND
//! bignum, so this is an ops/sec (verifications/sec) number, not a bytes/sec one.
//! The signature and public key are decoded and the buffers laid out **exactly**
//! as `tests/ecdsa_p256_driver.py` does: qx/qy/r/s are 32-byte big-endian scalars
//! passed by address (`ptr as i64`), the message by `(addr, len)`.
//!
//! ## Build recipe (combine chain — NOT self-contained)
//!
//! `std/ecdsa_p256.mind` composes `std/sha256.mind` via `import std.sha256;`, so
//! the two sources are combined with the import line stripped, byte-for-byte the
//! chain the driver documents:
//! ```text
//! cat std/sha256.mind                    >  combined.mind
//! grep -v '^import ' std/ecdsa_p256.mind >> combined.mind
//! mindc combined.mind --emit-shared ecdsa_p256.so
//! ```
//! The combined source is written to a temp file first, then compiled once.
//!
//! ## Additive, self-skipping
//!
//! Adds **nothing** to `src/`. Like the gated bench/test harnesses it self-skips
//! (registers no benchmarks, exits clean) when the MLIR toolchain
//! (`mlir-opt` / `mlir-translate` / `clang`) is shadowed or `mindc` is not built,
//! because it must shell out to `mindc --emit-shared`. The whole file compiles
//! unconditionally (cargo always builds bench targets); the skip is a runtime
//! check. After the `.so` loads, a correctness belt asserts the vector VERIFIES
//! (accepts, rc==1) and that a bit-flipped `r` is REJECTED (rc==0) — so a broken
//! lowering can never be reported as a clean verifications/sec number.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_ecdsa_p256 --no-default-features
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// ABI: `ecdsa_p256_verify(qx, qy, r, s, msg, msg_len) -> 0|1`. qx/qy/r/s are
/// addresses of 32-byte big-endian scalars; msg is `(addr, len)`. Returns 1 to
/// ACCEPT, 0 to REJECT (byte-for-byte the contract `ecdsa_p256_driver.py` binds:
/// `restype = c_int64`, `argtypes = [c_int64] * 6`).
type VerifyFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

// ---------------------------------------------------------------------------
// Fixed known-answer vector: RFC 6979 A.2.5, P-256 / SHA-256, message "sample".
// Deterministic ECDSA, so (r, s) are fixed constants. Cross-checked against pyca
// in the driver BEFORE MIND ever sees it; MIND must ACCEPT it (rc==1).
// ---------------------------------------------------------------------------
const KAT_QX: &str = "60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6";
const KAT_QY: &str = "7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299";
const KAT_R: &str = "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716";
const KAT_S: &str = "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8";
const KAT_MSG: &[u8] = b"sample";

/// Decode a 64-char hex string into a 32-byte big-endian buffer (matches the
/// driver's `int.to_bytes(32, "big")` big-endian scalar encoding).
fn hex32(s: &str) -> Vec<u8> {
    assert_eq!(s.len(), 64, "expected 64 hex chars (32 bytes)");
    (0..32)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("valid hex"))
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

/// Combine `std/sha256.mind` + import-stripped `std/ecdsa_p256.mind` into one
/// temp source and compile it to a temp `.so` once. Returns `None` (self-skip)
/// if the MLIR toolchain is shadowed, `mindc` is not built, or the combine /
/// compile fails — same fail-safe contract as the gated harnesses (never panic).
fn build_verify_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_ecdsa_p256: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_ecdsa_p256: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };

        // Combine chain, byte-for-byte the driver's:
        //   cat std/sha256.mind                    >  combined
        //   grep -v '^import ' std/ecdsa_p256.mind >> combined
        let std_dir = manifest_dir().join("std");
        let Ok(sha) = std::fs::read_to_string(std_dir.join("sha256.mind")) else {
            eprintln!("bench_ecdsa_p256: cannot read std/sha256.mind; skipping");
            return None;
        };
        let Ok(ecdsa_raw) = std::fs::read_to_string(std_dir.join("ecdsa_p256.mind")) else {
            eprintln!("bench_ecdsa_p256: cannot read std/ecdsa_p256.mind; skipping");
            return None;
        };
        let ecdsa_stripped: String = ecdsa_raw
            .lines()
            .filter(|l| !l.starts_with("import "))
            .collect::<Vec<_>>()
            .join("\n");
        let combined = format!("{sha}\n{ecdsa_stripped}\n");

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_ecdsa_p256_combined.mind");
        let so_path = dir.join("mind_bench_ecdsa_p256.so");
        if std::fs::write(&src_path, combined).is_err() {
            eprintln!("bench_ecdsa_p256: could not write combined source; skipping");
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
                eprintln!("bench_ecdsa_p256: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// One verify call over already-laid-out buffers. qx/qy/r/s point at 32-byte
/// big-endian scalars; msg at `msg_len` bytes. `f` is the raw `Copy` fn pointer
/// resolved from the loaded `.so`.
#[inline]
fn verify(f: VerifyFn, qx: &[u8], qy: &[u8], r: &[u8], s: &[u8], msg: &[u8]) -> i64 {
    unsafe {
        f(
            qx.as_ptr() as i64,
            qy.as_ptr() as i64,
            r.as_ptr() as i64,
            s.as_ptr() as i64,
            msg.as_ptr() as i64,
            msg.len() as i64,
        )
    }
}

/// Correctness belt (mirrors the driver's accept + reject cross-checks): the KAT
/// vector must ACCEPT (rc==1) and a bit-flipped `r` must REJECT (rc==0). Panics
/// on either failure so a broken build can't be reported as a clean ops/sec.
fn assert_verify_correct(f: VerifyFn, qx: &[u8], qy: &[u8], r: &[u8], s: &[u8]) {
    let accept = verify(f, qx, qy, r, s, KAT_MSG);
    assert_eq!(
        accept, 1,
        "ecdsa_p256_verify REJECTED the RFC 6979 A.2.5 known-good vector \
         (rc={accept}, expected 1) — verify path is broken; refusing to benchmark"
    );
    let mut bad_r = r.to_vec();
    bad_r[31] ^= 1; // flip the LSB (driver's `sig_r ^ 1`)
    let reject = verify(f, qx, qy, &bad_r, s, KAT_MSG);
    assert_eq!(
        reject, 0,
        "ecdsa_p256_verify ACCEPTED a bit-flipped r (rc={reject}, expected 0) — \
         verify path is broken; refusing to benchmark"
    );
    eprintln!(
        "bench_ecdsa_p256: correctness VERIFIED (RFC 6979 A.2.5 accepts, bit-flipped r rejects)"
    );
}

fn bench_ecdsa_p256(c: &mut Criterion) {
    let Some(so) = build_verify_so() else {
        eprintln!("bench_ecdsa_p256: verify kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen ecdsa_p256 .so") };
    let sym: Symbol<VerifyFn> = unsafe {
        lib.get(b"ecdsa_p256_verify")
            .expect("ecdsa_p256_verify symbol")
    };
    // Copy the raw fn pointer out of the Symbol (fn pointers are Copy); `lib` and
    // `sym` outlive every call below, keeping the .so mapped.
    let f: VerifyFn = *sym;

    // Fixed inputs, laid out once (32-byte big-endian scalars + the message).
    let qx = hex32(KAT_QX);
    let qy = hex32(KAT_QY);
    let r = hex32(KAT_R);
    let s = hex32(KAT_S);

    // Correctness gate first: an ops/sec number for a broken verifier is a lie.
    assert_verify_correct(f, &qx, &qy, &r, &s);

    let mut group = c.benchmark_group("ecdsa_p256");
    group.bench_function("verify", |b| {
        b.iter(|| {
            let rc = verify(
                f,
                black_box(&qx),
                black_box(&qy),
                black_box(&r),
                black_box(&s),
                black_box(KAT_MSG),
            );
            black_box(rc);
        });
    });
    group.finish();
}

criterion_group! {
    name = ecdsa_p256;
    // ECDSA verify is millisecond-scale in pure-MIND bignum (SHA-256 + two P-256
    // scalar mults): warm up, then sample amply for a stable verifications/sec.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(50);
    targets = bench_ecdsa_p256
}
criterion_main!(ecdsa_p256);
