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

//! X.509 certificate parse / verify throughput benchmark for the pure-MIND
//! `std/x509.mind` module (DER parsing + RSA PKCS#1 v1.5 / SHA-256 signature
//! verification).
//!
//! ## What is benched
//!
//!   * `x509_parse`               — walk the Certificate DER and record every
//!     extracted field (tbsCertificate, serial, issuer/subject, validity,
//!     modulus, exponent, sigAlg OID, signature). **Bytes/sec** over the DER.
//!   * `x509_verify_self_signed`  — the full stack: MIND parses the cert AND
//!     verifies its own self-signature end to end. **Ops/sec** (plain iters).
//!   * `rsa_pkcs1_sha256_verify`  — the bare RSA-2048 / SHA-256 verify primitive
//!     (modexp + PKCS#1 v1.5 unpad + SHA-256 compare). **Ops/sec**.
//!
//! ## Ground truth / build recipe (mirrors `tests/x509_vectors_driver.py`)
//!
//! `std/x509.mind` does `import std.sha256;`, which the standalone
//! `--emit-shared` path leaves undefined. The driver combines the two sources
//! into one translation unit — `cat std/sha256.mind` first, then `std/x509.mind`
//! with its `import std.sha256;` line stripped — and compiles that. This bench
//! reproduces exactly that combine chain (`build_x509_so`) before `--emit-shared`.
//! `std/sha256.mind` itself imports nothing, so the combined unit is closed.
//!
//! ## Sample vector
//!
//! `CERT_DER_HEX` is a fixed, real RSA-2048 / SHA-256 self-signed X.509v3
//! certificate (CN=mind-provable-stack.test, O=STARGA, serial 0x0123456789ABCDEF,
//! validity 2026..2036), generated with the same builder shape the driver uses.
//! Embedding it keeps the bench self-contained and byte-reproducible. The RSA
//! modulus / exponent / signature / tbsCertificate fed to the bare
//! `rsa_pkcs1_sha256_verify` are extracted from the cert at setup time via one
//! `x509_parse` call (reading the recorded field offset/length slots), exactly
//! as the driver's `slice_field` does — no separate vector needed.
//!
//! ## Self-skipping, correctness-gated
//!
//! Like the other kernel benches it self-skips (registers nothing, exits clean)
//! when the MLIR toolchain (`mlir-opt` / `mlir-translate` / `clang`) is shadowed
//! or `mindc` is not built — it must shell out to `mindc --emit-shared`. Before
//! timing anything it asserts the embedded cert *parses (rc==0)* and both verify
//! entry points *accept (rc==1)*; a throughput number for a broken verify would
//! be a lie, so a semantic mismatch panics the run rather than reporting a clean
//! number (same discipline as `det_matmul_q16`'s byte-identity gate). Only
//! toolchain/build/load failures self-skip.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench bench_x509 --no-default-features
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// `x509_parse(cert_addr, cert_len, out_fields) -> 0 on valid / 1 on reject`.
/// `out_fields` is a caller-allocated buffer of 22 i64 slots (offset/length
/// pairs into the cert DER — the field table the driver reads with `slice_field`).
type ParseFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// `x509_verify_self_signed(cert_addr, cert_len) -> 1 accept / 0 reject`.
type VerifySelfFn = unsafe extern "C" fn(i64, i64) -> i64;

/// `rsa_pkcs1_sha256_verify(n_addr, n_len, e, sig_addr, sig_len, msg_addr,
/// msg_len) -> 1 accept / 0 reject`. `e` is the public exponent passed **by
/// value** as i64 (the driver's `argtypes = [c_int64] * 7`); every `_addr`/`_len`
/// is a pointer/length pair.
type RsaVerifyFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;

/// Number of i64 field slots `x509_parse` writes (offset/length pairs). The
/// driver allocates `out(22 * 8)`.
const FIELD_SLOTS: usize = 22;

/// A fixed real RSA-2048 / SHA-256 self-signed X.509v3 certificate (748 DER
/// bytes), same builder shape as `tests/x509_vectors_driver.py`. Whitespace is
/// stripped by `decode_hex`.
const CERT_DER_HEX: &str = "\
308202e8308201d0a00302010202080123456789abcdef300d06092a864886f70d01010b050030343121301f06035504\
030c186d696e642d70726f7661626c652d737461636b2e74657374310f300d060355040a0c06535441524741301e170d\
3236303130313030303030305a170d3336303130313030303030305a30343121301f06035504030c186d696e642d7072\
6f7661626c652d737461636b2e74657374310f300d060355040a0c0653544152474130820122300d06092a864886f70d\
01010105000382010f003082010a0282010100a020ba97caaf4de101fd13984e856ebf98d515ad1fcc3f5b2e53d2e2bd\
023b3514b9f07433e090baf70035473cc82eba5622738685f6880a310aca5a5c6db09834c9e58c14a4ce9d64712d8b33\
1320d4b0e9c019c7588084907c7f4b42b7306830063509b1a60a06c4db90a106006aa1007c0d52f4ec104fd4de808cf2\
3cd7cd2953e1e558db83f496fd685688669552f2782f95be0771dbf6ae4bed42650a92868208d1fe6ee22a4f4986e366\
4c4dde28e6415d51579df8738827788289d71e4360139f46331e000782d281b10071c1731ca9a119f50e8fe946ed95c8\
59f5a9cc6860714442cbb3dee825eff3297e51a111c0622625f39337017a7e69ebd4310203010001300d06092a864886\
f70d01010b050003820101003e70396e422fde4b77b500bd1db21b7c0d81a9bc4bc646ae0506efbfbed78050c2e2157e\
cb9855cb925a5248fce5889fb94470a82becf5dd4f75d48c381974922480cf3f7ed741f15387226b40dd3bd018e34f70\
a24d40b61ab1d2911dbc895c80e6c9baac66e667febbfcdd8132737c11c71ffd038480e2e1f9ae3c5ee4232ead5b1172\
e16cdf10259232f8464af62afafc5e0f9c29bff6eec0341b468cfe93267d5f2ae672648cc986aedd9caa897f675efab3\
bdfae9559f33d8861cf780e9df379e656adcfcf9f38401731b75250cb75c2c9eef11b6031c0136b3d380cb1da8bc1885\
8f201b352c5c816cdcf36810c4953082f00e89bf660a7ad3e8815582";

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

/// Decode a hex string (ignoring any embedded whitespace) into bytes.
fn decode_hex(s: &str) -> Vec<u8> {
    let digits: Vec<u8> = s.bytes().filter(|b| b.is_ascii_hexdigit()).collect();
    assert!(
        digits.len() % 2 == 0,
        "CERT_DER_HEX has an odd number of hex digits"
    );
    digits
        .chunks(2)
        .map(|c| {
            let hi = (c[0] as char).to_digit(16).unwrap() as u8;
            let lo = (c[1] as char).to_digit(16).unwrap() as u8;
            (hi << 4) | lo
        })
        .collect()
}

/// Compile the combined `std/sha256.mind` + `std/x509.mind` (import-stripped)
/// translation unit to a temp `.so` once. Returns `None` (self-skip) if the MLIR
/// toolchain is shadowed, `mindc` is not built, the sources are unreadable, or
/// the compile fails — same fail-safe contract as the other kernel benches.
fn build_x509_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_x509: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_x509: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };

        // Combine chain, byte-for-byte the driver's `cat sha256 > c; grep -v
        // '^import std.sha256;' x509 >> c`: sha256 first (it imports nothing),
        // then x509 with its `import std.sha256;` line dropped.
        let std_dir = manifest_dir().join("std");
        let sha = match std::fs::read_to_string(std_dir.join("sha256.mind")) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("bench_x509: cannot read std/sha256.mind; skipping");
                return None;
            }
        };
        let x509 = match std::fs::read_to_string(std_dir.join("x509.mind")) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("bench_x509: cannot read std/x509.mind; skipping");
                return None;
            }
        };
        let mut combined = String::with_capacity(sha.len() + x509.len() + 2);
        combined.push_str(&sha);
        combined.push('\n');
        for line in x509.lines() {
            if line.trim_start().starts_with("import std.sha256") {
                continue;
            }
            combined.push_str(line);
            combined.push('\n');
        }

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_x509_combined.mind");
        let so_path = dir.join("mind_bench_x509.so");
        if std::fs::write(&src_path, &combined).is_err() {
            eprintln!("bench_x509: could not write combined source; skipping");
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
                eprintln!("bench_x509: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Read a `(offset, length)` field pair out of the parsed slot table and return
/// the corresponding slice of the cert DER — the Rust twin of the driver's
/// `slice_field(off_slot)`.
fn slice_field<'a>(cert: &'a [u8], slots: &[i64; FIELD_SLOTS], off_slot: usize) -> &'a [u8] {
    let off = slots[off_slot] as usize;
    let len = slots[off_slot + 1] as usize;
    &cert[off..off + len]
}

fn bench_x509(c: &mut Criterion) {
    let Some(so) = build_x509_so() else {
        eprintln!("bench_x509: x509 module unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen x509 .so") };

    let parse: Symbol<ParseFn> = unsafe { lib.get(b"x509_parse").expect("x509_parse symbol") };
    let verify_self: Symbol<VerifySelfFn> = unsafe {
        lib.get(b"x509_verify_self_signed")
            .expect("x509_verify_self_signed symbol")
    };
    let rsa_verify: Symbol<RsaVerifyFn> = unsafe {
        lib.get(b"rsa_pkcs1_sha256_verify")
            .expect("rsa_pkcs1_sha256_verify symbol")
    };

    let cert = decode_hex(CERT_DER_HEX);

    // --- setup: parse once, extract fields, and gate on correctness ---------
    // A throughput number for a cert that does not parse, or a verify that
    // rejects a signature we know is valid, would be a lie — so these panic
    // (they are not the toolchain/build self-skip path).
    let mut slots = [0i64; FIELD_SLOTS];
    let rc_parse = unsafe {
        parse(
            cert.as_ptr() as i64,
            cert.len() as i64,
            slots.as_mut_ptr() as i64,
        )
    };
    assert_eq!(
        rc_parse, 0,
        "bench_x509: x509_parse rejected the embedded valid cert (rc={rc_parse}) — \
         the module changed or the sample vector is stale"
    );

    // tbsCertificate (slot 0/1), RSA modulus n (12/13), exponent e (14/15),
    // signatureValue (18/19) — same slot layout the driver reads.
    let tbs = slice_field(&cert, &slots, 0).to_vec();
    let n: Vec<u8> = {
        // DER INTEGER content carries a leading 0x00 when the high bit is set;
        // strip leading zero bytes to the canonical big-endian magnitude, exactly
        // the 256-byte modulus the driver passes (`ref_n.to_bytes(256)`).
        let mut s: &[u8] = slice_field(&cert, &slots, 12);
        while s.len() > 1 && s[0] == 0 {
            s = &s[1..];
        }
        s.to_vec()
    };
    let e: i64 = slice_field(&cert, &slots, 14)
        .iter()
        .fold(0i64, |acc, &b| (acc << 8) | b as i64);
    let sig = slice_field(&cert, &slots, 18).to_vec();

    let rc_full = unsafe { verify_self(cert.as_ptr() as i64, cert.len() as i64) };
    assert_eq!(
        rc_full, 1,
        "bench_x509: x509_verify_self_signed rejected the valid self-signed cert (rc={rc_full})"
    );
    let rc_rsa = unsafe {
        rsa_verify(
            n.as_ptr() as i64,
            n.len() as i64,
            e,
            sig.as_ptr() as i64,
            sig.len() as i64,
            tbs.as_ptr() as i64,
            tbs.len() as i64,
        )
    };
    assert_eq!(
        rc_rsa, 1,
        "bench_x509: rsa_pkcs1_sha256_verify rejected the valid signature (rc={rc_rsa})"
    );
    eprintln!(
        "bench_x509: correctness gate PASSED — parse rc=0, verify_self_signed rc=1, \
         rsa_pkcs1_sha256_verify rc=1 ({} DER bytes, {}-byte modulus, e={e})",
        cert.len(),
        n.len()
    );

    // --- parse throughput (bytes/sec over the DER) --------------------------
    {
        let mut group = c.benchmark_group("x509_parse");
        group.throughput(Throughput::Bytes(cert.len() as u64));
        // Fresh out-buffer for the timed calls; the parse only writes into it.
        let mut fields = [0i64; FIELD_SLOTS];
        group.bench_function("rsa2048_sha256_der", |b| {
            b.iter(|| {
                let rc = unsafe {
                    parse(
                        black_box(cert.as_ptr() as i64),
                        black_box(cert.len() as i64),
                        black_box(fields.as_mut_ptr() as i64),
                    )
                };
                black_box(rc);
            });
        });
        group.finish();
    }

    // --- verify throughput (ops/sec, plain iters) ---------------------------
    {
        let mut group = c.benchmark_group("x509_verify");

        group.bench_function("verify_self_signed", |b| {
            b.iter(|| {
                let rc = unsafe {
                    verify_self(
                        black_box(cert.as_ptr() as i64),
                        black_box(cert.len() as i64),
                    )
                };
                black_box(rc);
            });
        });

        group.bench_function("rsa_pkcs1_sha256_verify", |b| {
            b.iter(|| {
                let rc = unsafe {
                    rsa_verify(
                        black_box(n.as_ptr() as i64),
                        black_box(n.len() as i64),
                        black_box(e),
                        black_box(sig.as_ptr() as i64),
                        black_box(sig.len() as i64),
                        black_box(tbs.as_ptr() as i64),
                        black_box(tbs.len() as i64),
                    )
                };
                black_box(rc);
            });
        });

        group.finish();
    }
}

criterion_group! {
    name = bench_x509_group;
    // Verify is modexp-bound (millisecond-ish); give it enough warm-up and
    // measurement time for tight CIs without a marathon run.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_x509
}
criterion_main!(bench_x509_group);
