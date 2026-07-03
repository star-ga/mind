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

//! TLS 1.3 record-layer seal/open throughput — pure-MIND `std/tls13_record.mind`
//! (RFC 8446 §5.1-5.3), compiled to native code by `mindc --emit-shared` and
//! driven over the C ABI.
//!
//! ## What is timed
//!
//!   1. `tls13_record_seal(key, iv, seq, content, content_len, inner_type,
//!      out_record)` — protect one application-data record: build the
//!      TLSInnerPlaintext (`content || inner_type`), the 5-byte header/AAD, the
//!      per-record nonce (`iv XOR seq_be8`), then AES-128-GCM-seal it. Returns
//!      the total wire length (`content_len + 22`). Timed over a sweep of
//!      content lengths; `Throughput::Bytes(content_len)` → criterion reports
//!      MiB/s of application data sealed.
//!   2. `tls13_record_open(key, iv, seq, record, record_len, out_content)` — the
//!      inverse deprotection (header/length checks, nonce, AEAD open with tag
//!      verify, inner-padding strip). Returns `(content_len << 8) | inner_type`
//!      on success, negative on any failure. Each size is sealed **once** outside
//!      the timed closure to produce a valid record, then `open` is timed on it;
//!      `Throughput::Bytes(content_len)` for a directly comparable MiB/s axis.
//!
//! ## Combined-source build — the FULL 5-module chain (dependency order, imports stripped)
//!
//! `std/tls13_record.mind` is NOT self-contained. The `.so` is built exactly as
//! the official-vector driver `tests/tls13_record_driver.py` documents — cat each
//! dependency in dependency order, stripping the intra-`std` `import` lines:
//! ```text
//! cat std/sha256.mind                                   >  combined.mind
//! grep -v '^import std.sha256;' std/hkdf.mind           >> combined.mind
//! grep -vE '^import std\.(sha256|hkdf);' std/tls13_keyschedule.mind >> combined.mind
//! cat std/aes_gcm.mind                                  >> combined.mind
//! grep -vE '^import std\.' std/tls13_record.mind        >> combined.mind
//! mindc combined.mind --emit-shared out.so              # needs the mlir-build toolchain
//! ```
//! (`tls13_record.mind` itself only `import std.aes_gcm;`, but the driver's
//! canonical recipe co-resides the whole sha256 → hkdf → keyschedule → aes_gcm →
//! record stack in one `.so`; this bench mirrors that verified-to-compile recipe
//! byte-for-byte rather than shortcutting the chain.) This bench performs the same
//! concatenation in Rust (read all five modules, strip the intra-`std` imports per
//! the greps above, join), writes the combined source to a temp file, and compiles
//! it once.
//!
//! ## Correctness belt (a throughput number for wrong bytes would be a lie)
//!
//! Before any timing the compiled `.so` is checked against the driver's published
//! **RFC 8448 §3** "Simple 1-RTT Handshake" server flight:
//!   * `seal(657-octet payload, seq=0)` reproduces the 679-octet on-the-wire
//!     record byte-for-byte (and returns length 679),
//!   * `open(RFC record, seq=0)` recovers the payload with inner content type
//!     `0x16` (handshake),
//!   * a tampered ciphertext byte is **rejected** (negative return) and never
//!     leaks plaintext (fail-closed), and
//!   * `nonce(seq=1) != nonce(seq=0)`, both matching the RFC 8446 §5.3
//!     construction (`iv XOR left-padded seq_be8`, only the low 8 bytes vary).
//! Any mismatch panics the bench run — mirroring the byte-identity gate in
//! `det_matmul_q16.rs` and the KAT gates in the sibling crypto benches.
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
//! cargo bench --bench bench_tls13_record --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds the
//! debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

/// Application-data content lengths swept (bytes of plaintext sealed/opened).
/// All >= 64 so the timed closures never dereference a zero-length buffer.
const SIZES: &[usize] = &[256, 1024, 4096, 16384, 65536];

/// Inner content type used for the swept records — 0x16 (handshake), the same
/// inner type the RFC 8448 §3 KAT payload carries.
const INNER_TYPE: i64 = 0x16;

// ---------------------------------------------------------------------------
// C-ABI entry points of the compiled std/tls13_record.mind (+ the full chain)
// .so. Byte-for-byte the ctypes signatures in tests/tls13_record_driver.py:
// every arg is an i64 address / length / value; every fn returns an i64.
// ---------------------------------------------------------------------------

/// `tls13_record_nonce(iv_addr, seq, nonce_out) -> 0`. Writes 12 nonce bytes.
type NonceFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// `tls13_record_seal(key_addr, iv_addr, seq, content_addr, content_len,
/// inner_type, out_record_addr) -> total_record_len (= content_len + 22)`.
/// Writes the complete wire record (5B header + ciphertext + 16B tag) to
/// `out_record_addr`.
type SealFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;

/// `tls13_record_open(key_addr, iv_addr, seq, record_addr, record_len,
/// out_content_addr) -> (content_len << 8) | inner_type` on success, negative on
/// failure. Writes recovered plaintext to `out_content_addr` (only on success).
type OpenFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

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

/// Strip the intra-`std` `import` lines a module declares, mirroring the driver's
/// `grep -v` / `grep -vE` filters. A line is dropped when (ignoring leading
/// whitespace) it begins with any of `prefixes`.
fn strip_import_lines(src: &str, prefixes: &[&str]) -> String {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !prefixes.iter().any(|p| t.starts_with(p))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the combined `sha256 + hkdf + tls13_keyschedule + aes_gcm +
/// tls13_record` source (the FULL chain the driver documents) and compile it to a
/// temp `.so` once. Returns `None` (self-skip) if the MLIR toolchain is shadowed,
/// `mindc` is not built, a module source cannot be read, or the compile fails —
/// same never-panic contract as `det_matmul_q16.rs` and the sibling crypto benches.
fn build_tls13_record_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("bench_tls13_record: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "bench_tls13_record: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };

        // Combine deps in dependency order, stripping the intra-std imports —
        // byte-for-byte the recipe in tests/tls13_record_driver.py:
        //   cat std/sha256.mind                                   >  combined
        //   grep -v '^import std.sha256;' std/hkdf.mind           >> combined
        //   grep -vE '^import std\.(sha256|hkdf);' std/tls13_keyschedule.mind >> combined
        //   cat std/aes_gcm.mind                                  >> combined
        //   grep -vE '^import std\.' std/tls13_record.mind        >> combined
        let std_dir = manifest_dir().join("std");
        let read = |name: &str| -> Option<String> {
            match std::fs::read_to_string(std_dir.join(name)) {
                Ok(s) => Some(s),
                Err(_) => {
                    eprintln!("bench_tls13_record: cannot read std/{name}; skipping");
                    None
                }
            }
        };
        let sha_src = read("sha256.mind")?;
        let hkdf_src = read("hkdf.mind")?;
        let ks_src = read("tls13_keyschedule.mind")?;
        let aes_src = read("aes_gcm.mind")?;
        let rec_src = read("tls13_record.mind")?;

        let hkdf_stripped = strip_import_lines(&hkdf_src, &["import std.sha256"]);
        let ks_stripped =
            strip_import_lines(&ks_src, &["import std.sha256", "import std.hkdf"]);
        // tls13_record: strip ALL its intra-std imports (grep -vE '^import std\.').
        let rec_stripped = strip_import_lines(&rec_src, &["import std."]);

        let combined = format!(
            "{sha_src}\n{hkdf_stripped}\n{ks_stripped}\n{aes_src}\n{rec_stripped}\n"
        );

        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_tls13_record.mind");
        let so_path = dir.join("mind_bench_tls13_record.so");
        if std::fs::write(&src_path, combined).is_err() {
            eprintln!("bench_tls13_record: could not write combined source; skipping");
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
                eprintln!("bench_tls13_record: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Minimal hex decoder (avoids adding the `hex` crate for the RFC 8448 literals).
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

/// RFC 8446 §5.3 per-record nonce reference: `iv XOR left-zero-padded seq_be8`
/// (bytes 0..3 = iv unchanged; bytes 4..11 = iv XOR the 8-byte big-endian seq).
/// The independent oracle for the MIND `tls13_record_nonce`.
fn ref_nonce(iv: &[u8], seq: u64) -> [u8; 12] {
    let mut n = [0u8; 12];
    n.copy_from_slice(&iv[..12]);
    let sb = seq.to_be_bytes(); // 8 bytes
    for k in 0..8 {
        n[4 + k] ^= sb[k];
    }
    n
}

// ---------------------------------------------------------------------------
// Thin FFI call helpers (used by the correctness gate; the timed closures hoist
// the symbol lookup out of the loop and call the fn pointer directly).
// ---------------------------------------------------------------------------

fn call_nonce(lib: &Library, iv: &[u8], seq: i64) -> [u8; 12] {
    let f: Symbol<NonceFn> = unsafe {
        lib.get(b"tls13_record_nonce")
            .expect("tls13_record_nonce symbol")
    };
    let mut out = [0u8; 12];
    let rc = unsafe { f(iv.as_ptr() as i64, seq, out.as_mut_ptr() as i64) };
    assert_eq!(rc, 0, "tls13_record_nonce returned {rc} (expected 0)");
    out
}

fn call_seal(
    lib: &Library,
    key: &[u8],
    iv: &[u8],
    seq: i64,
    content: &[u8],
    inner_type: i64,
    out_record: &mut [u8],
) -> i64 {
    let f: Symbol<SealFn> = unsafe {
        lib.get(b"tls13_record_seal")
            .expect("tls13_record_seal symbol")
    };
    unsafe {
        f(
            key.as_ptr() as i64,
            iv.as_ptr() as i64,
            seq,
            content.as_ptr() as i64,
            content.len() as i64,
            inner_type,
            out_record.as_mut_ptr() as i64,
        )
    }
}

fn call_open(
    lib: &Library,
    key: &[u8],
    iv: &[u8],
    seq: i64,
    record: &[u8],
    out_content: &mut [u8],
) -> i64 {
    let f: Symbol<OpenFn> = unsafe {
        lib.get(b"tls13_record_open")
            .expect("tls13_record_open symbol")
    };
    unsafe {
        f(
            key.as_ptr() as i64,
            iv.as_ptr() as i64,
            seq,
            record.as_ptr() as i64,
            record.len() as i64,
            out_content.as_mut_ptr() as i64,
        )
    }
}

/// Correctness gate: the driver's published RFC 8448 §3 server-flight KAT plus
/// a tamper-reject and a nonce check. Panics (fails the run) on any mismatch, so
/// a throughput number can never be reported for wrong record-layer bytes.
fn assert_tls13_record_correct(lib: &Library) {
    // RFC 8448 §3 server_handshake_traffic_secret write key/iv.
    let key = hexb("3fce516009c21727d0f2e4e86ee403bc");
    let iv = hexb("5d313eb2671276ee13000b30");

    // "{server} send handshake record: payload (657 octets)".
    let payload = hexb(concat!(
        "080000240022000a00140012001d00170018001901000101010201030104001c",
        "00024001000000000b0001b9000001b50001b0308201ac30820115a003020102",
        "020102300d06092a864886f70d01010b0500300e310c300a0603550403130372",
        "7361301e170d3136303733303031323335395a170d3236303733303031323335",
        "395a300e310c300a0603550403130372736130819f300d06092a864886f70d01",
        "0101050003818d0030818902818100b4bb498f8279303d980836399b36c6988c",
        "0c68de55e1bdb826d3901a2461eafd2de49a91d015abbc9a95137ace6c1af19e",
        "aa6af98c7ced43120998e187a80ee0ccb0524b1b018c3e0b63264d449a6d38e2",
        "2a5fda430846748030530ef0461c8ca9d9efbfae8ea6d1d03e2bd193eff0ab9a",
        "8002c47428a6d35a8d88d79f7f1e3f0203010001a31a301830090603551d1304",
        "023000300b0603551d0f0404030205a0300d06092a864886f70d01010b050003",
        "81810085aad2a0e5b9276b908c65f73a7267170618a54c5f8a7b337d2df7a594",
        "365417f2eae8f8a58c8f8172f9319cf36b7fd6c55b80f21a03015156726096fd",
        "335e5e67f2dbf102702e608ccae6bec1fc63a42a99be5c3eb7107c3c54e9b9eb",
        "2bd5203b1c3b84e0a8b2f759409ba3eac9d91d402dcc0cc8f8961229ac9187b4",
        "2b4de100000f000084080400805a747c5d88fa9bd2e55ab085a61015b7211f82",
        "4cd484145ab3ff52f1fda8477b0b7abc90db78e2d33a5c141a078653fa6bef78",
        "0c5ea248eeaaa785c4f394cab6d30bbe8d4859ee511f602957b15411ac027671",
        "459e46445c9ea58c181e818e95b8c3fb0bf3278409d3be152a3da5043e063dda",
        "65cdf5aea20d53dfacd42f74f3140000209b9b141d906337fbd2cbdce71df4de",
        "da4ab42c309572cb7fffee5454b78f0718",
    ));
    assert_eq!(payload.len(), 657, "RFC 8448 payload length");

    // "{server} send handshake record: complete record (679 octets)" at seq 0.
    let record = hexb(concat!(
        "17030302a2d1ff334a56f5bff6594a07cc87b580233f500f45e489e7f33af35e",
        "df7869fcf40aa40aa2b8ea73f848a7ca07612ef9f945cb960b4068905123ea78",
        "b111b429ba9191cd05d2a389280f526134aadc7fc78c4b729df828b5ecf7b13b",
        "d9aefb0e57f271585b8ea9bb355c7c79020716cfb9b1183ef3ab20e37d57a6b9",
        "d7477609aee6e122a4cf51427325250c7d0e509289444c9b3a648f1d71035d2e",
        "d65b0e3cdd0cbae8bf2d0b227812cbb360987255cc744110c453baa4fcd61092",
        "8d809810e4b7ed1a8fd991f06aa6248204797e36a6a73b70a2559c09ead68694",
        "5ba246ab66e5edd8044b4c6de3fcf2a89441ac66272fd8fb330ef8190579b368",
        "4596c960bd596eea520a56a8d650f563aad27409960dca63d3e688611ea5e22f",
        "4415cf9538d51a200c27034272968a264ed6540c84838d89f72c24461aad6d26",
        "f59ecaba9acbbb317b66d902f4f292a36ac1b639c637ce343117b65962224531",
        "7b49eeda0c6258f100d7d961ffb138647e92ea330faeea6dfa31c7a84dc3bd7e",
        "1b7a6c7178af36879018e3f252107f243d243dc7339d5684c8b0378bf30244da",
        "8c87c843f5e56eb4c5e8280a2b48052cf93b16499a66db7cca71e4599426f7d4",
        "61e66f99882bd89fc50800becca62d6c74116dbd2972fda1fa80f85df881edbe",
        "5a37668936b335583b599186dc5c6918a396fa48a181d6b6fa4f9d62d513afbb",
        "992f2b992f67f8afe67f76913fa388cb5630c8ca01e0c65d11c66a1e2ac4c859",
        "77b7c7a6999bbf10dc35ae69f5515614636c0b9b68c19ed2e31c0b3b66763038",
        "ebba42f3b38edc0399f3a9f23faa63978c317fc9fa66a73f60f0504de93b5b84",
        "5e275592c12335ee340bbc4fddd502784016e4b3be7ef04dda49f4b440a30cb5",
        "d2af939828fd4ae3794e44f94df5a631ede42c1719bfdabf0253fe5175be898e",
        "750edc53370d2b",
    ));
    assert_eq!(record.len(), 679, "RFC 8448 record length");

    // (a) seal(RFC payload, seq=0) == RFC 8448 complete record, byte-for-byte.
    let mut rec_out = vec![0u8; record.len() + 8];
    let ret = call_seal(lib, &key, &iv, 0, &payload, INNER_TYPE, &mut rec_out);
    assert_eq!(
        ret,
        record.len() as i64,
        "tls13_record_seal returned {ret} (expected {})",
        record.len()
    );
    assert_eq!(
        &rec_out[..record.len()],
        record.as_slice(),
        "seal output != RFC 8448 §3 complete record (record layer is miscompiling)"
    );

    // (b) open(RFC record, seq=0) == payload, inner content type 0x16.
    let mut ct_out = vec![0u8; record.len()];
    let ret = call_open(lib, &key, &iv, 0, &record, &mut ct_out);
    assert!(
        ret >= 0,
        "open(RFC record) rejected a valid record (ret={ret})"
    );
    assert_eq!(
        ret & 0xff,
        INNER_TYPE,
        "open recovered wrong inner content type"
    );
    assert_eq!(
        ret >> 8,
        payload.len() as i64,
        "open recovered wrong content length"
    );
    assert_eq!(
        &ct_out[..payload.len()],
        payload.as_slice(),
        "open recovered plaintext != RFC 8448 payload"
    );

    // (c) tampered ciphertext byte -> REJECTED, fail-closed (no plaintext leak).
    let mut tampered = record.clone();
    tampered[5] ^= 0x01;
    let mut ct_out = vec![0u8; record.len()];
    let ret = call_open(lib, &key, &iv, 0, &tampered, &mut ct_out);
    assert!(ret < 0, "tampered ciphertext not rejected (ret={ret})");
    assert_ne!(
        &ct_out[..payload.len()],
        payload.as_slice(),
        "tampered record leaked plaintext (NOT fail-closed)"
    );

    // (d) nonce(seq=1) != nonce(seq=0); both match RFC 8446 §5.3 (iv XOR seq_be8).
    let n0 = call_nonce(lib, &iv, 0);
    let n1 = call_nonce(lib, &iv, 1);
    assert_ne!(
        n0, n1,
        "nonce(seq=0) == nonce(seq=1) — nonce is not sequence-dependent"
    );
    assert_eq!(n0[..4], n1[..4], "nonce zero-pad region changed with seq");
    assert_eq!(
        n0,
        ref_nonce(&iv, 0),
        "nonce(seq=0) != RFC 8446 §5.3 reference"
    );
    assert_eq!(
        n1,
        ref_nonce(&iv, 1),
        "nonce(seq=1) != RFC 8446 §5.3 reference"
    );

    eprintln!(
        "bench_tls13_record: correctness VERIFIED — RFC 8448 §3 seal/open KAT \
         (657B payload -> 679B record), tamper-reject fail-closed, RFC 8446 §5.3 nonce."
    );
}

fn bench_tls13_record(c: &mut Criterion) {
    let Some(so) = build_tls13_record_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("bench_tls13_record: record layer unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen tls13_record .so") };

    // Correctness gate first: a throughput number for wrong record bytes is a lie.
    assert_tls13_record_correct(&lib);

    // Fixed 128-bit key / 96-bit IV, sequence number 0 for the timed records.
    let key = hexb("3fce516009c21727d0f2e4e86ee403bc");
    let iv = hexb("5d313eb2671276ee13000b30");

    // ---- seal throughput over content length (bytes of app data sealed) ------
    let mut sgroup = c.benchmark_group("tls13_record_seal");
    for &n in SIZES {
        sgroup.throughput(Throughput::Bytes(n as u64));

        // Deterministic byte-ramp plaintext; record scratch (n + 22 + slack)
        // allocated once, outside the timed closure.
        let content: Vec<u8> = (0..n).map(|i| (i & 0xff) as u8).collect();
        let mut rec_out = vec![0u8; n + 22 + 16];
        let f: Symbol<SealFn> = unsafe {
            lib.get(b"tls13_record_seal")
                .expect("tls13_record_seal symbol")
        };

        sgroup.bench_with_input(
            BenchmarkId::new("seal", format!("{n}B")),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let rc = unsafe {
                        f(
                            black_box(key.as_ptr() as i64),
                            black_box(iv.as_ptr() as i64),
                            black_box(0i64),
                            black_box(content.as_ptr() as i64),
                            black_box(content.len() as i64),
                            black_box(INNER_TYPE),
                            black_box(rec_out.as_mut_ptr() as i64),
                        )
                    };
                    black_box(rc);
                });
            },
        );
    }
    sgroup.finish();

    // ---- open throughput over content length (bytes of app data recovered) ---
    let mut ogroup = c.benchmark_group("tls13_record_open");
    for &n in SIZES {
        ogroup.throughput(Throughput::Bytes(n as u64));

        // Seal ONCE outside the timed closure to obtain a valid record, then time
        // open() against it. record scratch and recovered-content scratch both
        // allocated once.
        let content: Vec<u8> = (0..n).map(|i| (i & 0xff) as u8).collect();
        let mut rec_buf = vec![0u8; n + 22 + 16];
        let rec_len = call_seal(&lib, &key, &iv, 0, &content, INNER_TYPE, &mut rec_buf);
        assert_eq!(
            rec_len,
            (n + 22) as i64,
            "seal for open-bench setup returned {rec_len} (expected {})",
            n + 22
        );
        let rec_len = rec_len as usize;
        let mut ct_out = vec![0u8; n + 22];
        let f: Symbol<OpenFn> = unsafe {
            lib.get(b"tls13_record_open")
                .expect("tls13_record_open symbol")
        };

        ogroup.bench_with_input(
            BenchmarkId::new("open", format!("{n}B")),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let rc = unsafe {
                        f(
                            black_box(key.as_ptr() as i64),
                            black_box(iv.as_ptr() as i64),
                            black_box(0i64),
                            black_box(rec_buf.as_ptr() as i64),
                            black_box(rec_len as i64),
                            black_box(ct_out.as_mut_ptr() as i64),
                        )
                    };
                    black_box(rc);
                });
            },
        );
    }
    ogroup.finish();
}

criterion_group! {
    name = bench_tls13_record_group;
    // Real measurement, not a 5s quick run: warm-up + ample sampling so the
    // seal/open throughput lands with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_tls13_record
}
criterion_main!(bench_tls13_record_group);
