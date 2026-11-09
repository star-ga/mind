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

//! std/x509.mind authenticated chain-validation smoke test.
//!
//! Compiles std/x509.mind and the std crypto modules it imports, concatenated
//! with their import lines stripped (the X509_SET of
//! scripts/run_crypto_vectors.sh), via `mindc --emit-shared`, dlopen-s the
//! resulting `.so`, and replays the full offline fixture matrix
//! `tests/x509/testdata/cases_native.bin` (committed openssl-generated DER
//! chains): positive EC / RSA-PKCS1 / RSA-PSS / P-384 chains, wildcard and
//! A-label matching, DER and PEM anchor bundles, and every negative class
//! (wrong host, unknown anchor, expired, not-yet-valid, broken signature,
//! non-CA intermediate, pathLen, missing serverAuth EKU, missing SAN, truncated
//! DER, unknown critical extension, depth > 8).
//!
//! Gate: `cargo test --features "mlir-build std-surface cross-module-imports"
//!        --test x509_chain_smoke`.
//! Self-skips when mlir-opt / mlir-translate / clang are absent, exactly like
//! `sha256_smoke.rs`.
//!
//! `tests/std_tls_x509_import.rs` runs the same verifier through
//! `import std.x509;` under `mindc test` and in a `mindc build` executable.

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]

mod common;
use common::x509_so::{ChainFn, Lib, TD, blob, blob_of, now_unix, read_der, run};

use std::path::PathBuf;

use libloading::Symbol;

type FlagFn = unsafe extern "C" fn() -> i64;

#[test]
fn x509_peer_auth_supported_is_one() {
    let Some(lib) = Lib::get() else { return };
    let f: Symbol<FlagFn> = unsafe { lib.0.get(b"x509_peer_auth_supported\0").unwrap() };
    assert_eq!(unsafe { f() }, 1, "peer-auth capability must be 1");
}

// Replay the serialized case matrix.
fn take_i64(data: &[u8], pos: &mut usize) -> i64 {
    let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn take_blob(data: &[u8], pos: &mut usize) -> Vec<u8> {
    let count = take_i64(data, pos) as usize;
    let mut out = vec![0u8; 8 + 16 * count];
    out[..8].copy_from_slice(&(count as i64).to_le_bytes());
    for i in 0..count {
        let len = take_i64(data, pos) as usize;
        let off = out.len() as i64;
        out[8 + 16 * i..16 + 16 * i].copy_from_slice(&off.to_le_bytes());
        out[16 + 16 * i..24 + 16 * i].copy_from_slice(&(len as i64).to_le_bytes());
        out.extend_from_slice(&data[*pos..*pos + len]);
        *pos += (len + 7) & !7;
    }
    out
}

#[test]
fn x509_chain_matrix() {
    let lib = Lib::get().expect("MLIR library must build for acceptance gate");
    let data = std::fs::read(format!("{TD}/cases_native.bin")).unwrap();
    let f: Symbol<ChainFn> = unsafe { lib.0.get(b"x509_verify_server_chain\0").unwrap() };
    let load: Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
        unsafe { lib.0.get(b"x509_load_anchors\0").unwrap() };
    let bundles: Vec<i64> = ["der", "pem"]
        .iter()
        .map(|ext| {
            let raw = std::fs::read(format!("{TD}/bundle_anchors.{ext}")).unwrap();
            let ptr = unsafe { load(raw.as_ptr() as i64, raw.len() as i64) };
            assert!(ptr > 0);
            ptr
        })
        .collect();
    let mut pos = 0;
    let count = take_i64(&data, &mut pos);
    for i in 0..count {
        let want = take_i64(&data, &mut pos);
        let now = take_i64(&data, &mut pos);
        let mode = take_i64(&data, &mut pos);
        let dns = &data[pos..pos + 256];
        pos += 256;
        let chain = take_blob(&data, &mut pos);
        let anchors = take_blob(&data, &mut pos);
        let ap = if mode == 0 {
            anchors.as_ptr() as i64
        } else {
            bundles[mode as usize - 1]
        };
        let got = unsafe { f(chain.as_ptr() as i64, ap, dns.as_ptr() as i64, now) };
        assert_eq!(got, want, "manifest case {i}");
    }
    assert_eq!(pos, data.len());
    eprintln!("x509 shared matrix: {count}/{count} passed");
}

#[test]
fn x509_in_memory_bundle_loading() {
    let Some(lib) = Lib::get() else { return };
    let now = now_unix();
    type LoadFn = unsafe extern "C" fn(i64, i64) -> i64;
    let load_f: Symbol<LoadFn> = unsafe { lib.0.get(b"x509_load_anchors\0").unwrap() };
    let verify_f: Symbol<ChainFn> = unsafe { lib.0.get(b"x509_verify_server_chain\0").unwrap() };

    // 1. Valid DER bundle
    let der_bundle = std::fs::read(format!("{TD}/bundle_anchors.der")).unwrap();
    let der_blob = unsafe { load_f(der_bundle.as_ptr() as i64, der_bundle.len() as i64) };
    assert!(der_blob > 0, "load DER bundle must succeed");
    let ec_chain = blob(&["ec_leaf", "ec_inter"]);
    let dnsb = b"www.example.com\0".to_vec();
    let res = unsafe {
        verify_f(
            ec_chain.as_ptr() as i64,
            der_blob,
            dnsb.as_ptr() as i64,
            now,
        )
    };
    assert_eq!(res, 0, "verify EC chain against DER bundle");
    let p384_chain = blob(&["p384_leaf"]);
    let p384_dns = b"www.p384-test.example.com\0".to_vec();
    let res = unsafe {
        verify_f(
            p384_chain.as_ptr() as i64,
            der_blob,
            p384_dns.as_ptr() as i64,
            now,
        )
    };
    assert_eq!(res, 0, "verify P-384 chain against DER bundle");

    // 2. Valid PEM bundle
    let pem_bundle = std::fs::read(format!("{TD}/bundle_anchors.pem")).unwrap();
    let pem_blob = unsafe { load_f(pem_bundle.as_ptr() as i64, pem_bundle.len() as i64) };
    assert!(pem_blob > 0, "load PEM bundle must succeed");
    let rsa_chain = blob(&["rsa_leaf"]);
    let rsa_dns = b"www.rsa-test.example.com\0".to_vec();
    let res = unsafe {
        verify_f(
            rsa_chain.as_ptr() as i64,
            pem_blob,
            rsa_dns.as_ptr() as i64,
            now,
        )
    };
    assert_eq!(res, 0, "verify RSA chain against PEM bundle");
    let res = unsafe {
        verify_f(
            p384_chain.as_ptr() as i64,
            pem_blob,
            p384_dns.as_ptr() as i64,
            now,
        )
    };
    assert_eq!(res, 0, "verify P-384 chain against PEM bundle");

    // 3. Malformed PEM bundle must fail closed
    let bad_pem = std::fs::read(format!("{TD}/bundle_bad.pem")).unwrap();
    let bad_res = unsafe { load_f(bad_pem.as_ptr() as i64, bad_pem.len() as i64) };
    assert!(
        bad_res < 0,
        "malformed PEM bundle must return negative error code, got {bad_res}"
    );
}

#[test]
fn x509_module_line_count_within_budget() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let std_dir = manifest_dir.join("std");
    let mut over_budget = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&std_dir)
        .expect("read std dir")
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("x509") && name.ends_with(".mind") {
            let content = std::fs::read_to_string(entry.path()).expect("read mind file");
            let line_count = content.lines().count();
            if line_count > 800 {
                over_budget.push((name, line_count));
            }
        }
    }
    if !over_budget.is_empty() {
        panic!("std/x509*.mind files exceeding 800-line budget: {over_budget:?}");
    }
}

// ── Regression cases for the fixes carried with the validator ──────────────
// Fixtures: tests/x509/testdata/adv/ (fixed clock now.txt).

/// `(case, chain, anchors, host, seconds after now.txt, verdict)`.
type Row = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
    &'static str,
    i64,
    i64,
);

fn check_rows(lib: &Lib, rows: &[Row]) {
    let now = now_unix();
    let failed: Vec<String> = rows
        .iter()
        .filter_map(|(name, chain, anchors, host, shift, want)| {
            let got = run(lib, chain, anchors, host, now + shift);
            (got != *want).then(|| format!("{name}: got {got}, want {want}"))
        })
        .collect();
    assert!(
        failed.is_empty(),
        "x509 regression cases failed:\n{}",
        failed.join("\n")
    );
}

#[rustfmt::skip]
const RSA_SIZES: &[Row] = &[
    ("rsa4096_root", &["adv/rsa4096_leaf", "adv/rsa4096_inter2048"], &["adv/rsa4096_root"], "adv.example.com", 0, 0),
    ("rsa3072_root", &["adv/rsa3072_leaf"], &["adv/rsa3072_root"], "adv.example.com", 0, 0),
    ("pss4096_root", &["adv/pss4096_leaf", "adv/pss4096_inter2048"], &["adv/pss4096_root"], "adv.example.com", 0, 0),
];

/// RSA-3072/4096 roots and an RSA-PSS 4096 root verify through 2048-bit
/// intermediates; a 2048-bit-only bignum refused them.
#[test]
fn x509_rsa_3072_and_4096_issuers_verify() {
    let Some(lib) = Lib::get() else { return };
    check_rows(lib, RSA_SIZES);
    // A modulus length outside the supported sizes is refused, never verified.
    type Pkcs1 = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
    type Pss = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64;
    let pkcs1: Symbol<Pkcs1> = unsafe { lib.0.get(b"rsa_pkcs1_sha256_verify\0").unwrap() };
    let pss: Symbol<Pss> = unsafe { lib.0.get(b"rsa_pss_verify_sha256\0").unwrap() };
    let mut n = [0u8; 1024];
    n[0] = 128;
    let sig = [1u8; 1024];
    let msg = [2u8; 32];
    for len in [128i64, 500, 1024] {
        let (np, sp, mp) = (n.as_ptr() as i64, sig.as_ptr() as i64, msg.as_ptr() as i64);
        assert_eq!(
            unsafe { pkcs1(np, len, 65537, sp, len, mp, 32) },
            0,
            "PKCS#1 {len}-byte modulus"
        );
        assert_eq!(
            unsafe { pss(np, len, 65537, sp, len, mp, 32, 32) },
            0,
            "PSS {len}-byte modulus"
        );
    }
}

/// An anchor's own signature algorithm (SHA-1, or a malformed anchor next to
/// a good one) does not poison verification under the good anchor.
#[test]
fn x509_anchor_self_signature_does_not_poison_verification() {
    let Some(lib) = Lib::get() else { return };
    check_rows(
        lib,
        &[(
            "sha1_selfsigned_anchor",
            &["adv/sha1root_sha256_leaf"],
            &["adv/sha1_selfsigned_root"],
            "adv.example.com",
            0,
            0,
        )],
    );
    let verify: Symbol<ChainFn> = unsafe { lib.0.get(b"x509_verify_server_chain\0").unwrap() };
    let load: Symbol<unsafe extern "C" fn(i64, i64) -> i64> =
        unsafe { lib.0.get(b"x509_load_anchors\0").unwrap() };
    let chain = blob(&["ec_leaf", "ec_inter"]);
    let host = b"www.example.com\0";
    let now = now_unix();
    for bundle in ["bundle_sha1_plus_ec.der", "bundle_ec_plus_sha1.pem"] {
        let raw = std::fs::read(format!("{TD}/adv/{bundle}")).unwrap();
        let anchors = unsafe { load(raw.as_ptr() as i64, raw.len() as i64) };
        assert!(anchors > 0, "{bundle}: the bundle must load");
        let got = unsafe { verify(chain.as_ptr() as i64, anchors, host.as_ptr() as i64, now) };
        assert_eq!(
            got, 0,
            "{bundle}: the EC chain must verify under the bundle's EC anchor"
        );
    }
    let anchors = blob_of(&[
        b"\x30\x05\x02\x01\x00\x00\x00".to_vec(),
        read_der("ec_root"),
    ]);
    let got = unsafe {
        verify(
            chain.as_ptr() as i64,
            anchors.as_ptr() as i64,
            host.as_ptr() as i64,
            now,
        )
    };
    assert_eq!(
        got, 0,
        "a malformed anchor must not hide the valid anchor beside it"
    );
}

#[rustfmt::skip]
const ANCHOR_SIG_AND_LEAF_EXPIRY: &[Row] = &[
    ("anchor_wrong_key", &["adv/xs_leaf", "adv/xs_inter"], &["adv/xs_rootA"], "adv.example.com", 0, -3),
    ("leaf_expired_only", &["path_leaf_expired", "path_ec_inter"], &["path_ec_root"], "path.example.com", 6 * 86_400, -6),
    ("leaf_valid_control", &["path_leaf_expired", "path_ec_inter"], &["path_ec_root"], "path.example.com", 0, 0),
    // ec_leaf_ip's notBefore is 172799 s before now.txt, one second after
    // ec_inter's: at -172800 only the leaf is not yet valid.
    ("leaf_not_yet_valid_only", &["ec_leaf_ip", "ec_inter"], &["ec_root"], "127.0.0.1", -172_800, -5),
    ("leaf_valid_from_control", &["ec_leaf_ip", "ec_inter"], &["ec_root"], "127.0.0.1", -172_799, 0),
];

/// The signature from the last certificate to the anchor is checked: an
/// anchor with the issuer's name and another key is refused (-3), a leaf
/// that alone has expired is refused (-6), and a leaf that alone is not yet
/// valid (its intermediate and root are) is refused (-5).
#[test]
fn x509_anchor_signature_and_leaf_expiry_are_enforced() {
    let Some(lib) = Lib::get() else { return };
    check_rows(lib, ANCHOR_SIG_AND_LEAF_EXPIRY);
}

/// RFC 6125 §6.4.3: a wildcard matches only as the complete left-most label
/// followed by non-empty labels.
#[test]
fn x509_dns_wildcard_rules() {
    let Some(lib) = Lib::get() else { return };
    type DnsFn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
    let dns: Symbol<DnsFn> = unsafe { lib.0.get(b"x509_dns_match\0").unwrap() };
    let cases: &[(&str, &str, i64)] = &[
        ("*.example.com", "www.example.com", 1),
        ("*.example.com", "a.example.com", 1),
        ("*.Example.COM", "www.EXAMPLE.com", 1),
        ("*.a.example.com", "x.a.example.com", 1),
        ("*.example.com", "example.com", 0),
        ("*.example.com", "a.b.example.com", 0),
        ("a.*.com", "z.*.com", 0),
        ("x.example.*", "b.example.*", 0),
        ("a.*.com", "z.foo.com", 0),
        ("x.example.*", "b.example.org", 0),
        ("*", "anything.example.com", 0),
        ("*", "", 0),
        ("*.*.com", "a.b.com", 0),
        ("a.*.*.com", "z.b.c.com", 0),
        ("f*o.example.com", "foo.example.com", 0),
        ("f*o.example.com", "fXo.example.com", 0),
        ("foo*.example.com", "foobar.example.com", 0),
        ("xn--*.example.com", "xn--foo.example.com", 0),
        ("a.xn--*.com", "z.xn--*.com", 0),
        ("*.example.com", ".example.com", 0),
        ("*.", "com.", 0),
        ("*..example.com", "a..example.com", 0),
        ("*.example.com.", "a.example.com.", 0),
    ];
    let failed: Vec<String> = cases
        .iter()
        .filter_map(|(pat, host, want)| {
            let (p, h) = (pat.as_bytes(), host.as_bytes());
            let got = unsafe {
                dns(
                    p.as_ptr() as i64,
                    p.len() as i64,
                    h.as_ptr() as i64,
                    h.len() as i64,
                )
            };
            (got != *want).then(|| format!("pattern {pat:?} host {host:?}: got {got}, want {want}"))
        })
        .collect();
    assert!(failed.is_empty(), "wildcard rules:\n{}", failed.join("\n"));
}

#[rustfmt::skip]
const ANCHOR_POLICY: &[Row] = &[
    ("expired_anchor_still_trusted", &["path_exp_leaf", "path_exp_inter"], &["path_anchor_expired"], "path.example.com", 0, 0),
    ("expired_intermediate_rejected", &["path_ec_leaf", "path_expired"], &["path_ec_root"], "path.example.com", 0, -6),
    ("valid_intermediate_accepted", &["path_ec_leaf", "path_ec_inter"], &["path_ec_root"], "path.example.com", 0, 0),
    ("unsupported_anchor_sigparams_skipped", &["adv/a_leaf_pss384_root", "adv/a_inter_pss384_root"], &["adv/a_root_pss384"], "anc.example.com", 0, -7),
    ("supported_anchor_sigalg_accepted", &["adv/a_leaf_rsa384_root", "adv/a_inter_rsa384_root"], &["adv/a_root_rsa384"], "anc.example.com", 0, 0),
];

/// Anchors are trusted key bindings: an expired anchor still anchors (0), an
/// expired intermediate under a valid anchor fails closed (-6), and an anchor
/// that does not parse under the signature-algorithm rules is skipped (-7)
/// while its well-formed twin verifies (0).
#[test]
fn x509_trust_anchor_policy() {
    let Some(lib) = Lib::get() else { return };
    check_rows(lib, ANCHOR_POLICY);
}

// ── Path-search work bound ─────────────────────────────────────────────────
// The peer controls the certificate bag. A repeated certificate (same DER) is
// one candidate issuer, and one path search spends at most 64 signature
// verifications, anchors and bag issuers alike, failing -13 past that.

type BudgetFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

const RSA_HOST: &str = "www.rsa-test.example.com";
const PATH_HOST: &str = "path.example.com";
const UNLIMITED: i64 = 1 << 40;
/// The RSA leaf and seven copies of its self-signed issuer: every copy
/// verifies every other one, the shape of a bag that cost 13,699 verifications.
const RSA_COPIES: &[&str] = &[
    "rsa_leaf", "rsa_root", "rsa_root", "rsa_root", "rsa_root", "rsa_root", "rsa_root", "rsa_root",
];
const INTER_COPIES: &[&str] = &[
    "path_ec_leaf",
    "path_ec_inter",
    "path_ec_inter",
    "path_ec_inter",
    "path_ec_inter",
    "path_ec_inter",
    "path_ec_inter",
    "path_ec_inter",
];
/// Two certificates named `path_ec root` with the same DER length.
const SAME_LEN_ROOTS: &[&str] = &[
    "path_ec_leaf",
    "path_ec_inter",
    "path_anchor_wrongkey",
    "path_ec_root",
];
/// 1 + 31 + 1 + 31 = 64 verifications: each of 31 wrong-key anchors is tried
/// for the intermediate and for the bag root.
const THREE: &[&str] = &["path_ec_leaf", "path_ec_inter", "path_ec_root"];
/// THREE plus a wrong-key intermediate the leaf tries last: 65 verifications.
const FOUR: &[&str] = &[
    "path_ec_leaf",
    "path_ec_inter",
    "path_ec_root",
    "path_wrongkey",
];
const WRONG_31: &[&str] = &["path_anchor_wrongkey"; 31];
const WRONG_32: &[&str] = &["path_anchor_wrongkey"; 32];

/// Validate through `x509_verify_server_chain_budget` with `budget`
/// verifications allowed; returns (verdict, verifications spent).
fn run_budget(lib: &Lib, chain: &[&str], anchors: &[&str], host: &str, budget: i64) -> (i64, i64) {
    run_budget_blob(lib, &blob(chain), &blob(anchors), host, budget)
}

fn run_budget_blob(lib: &Lib, chain: &[u8], anchors: &[u8], host: &str, budget: i64) -> (i64, i64) {
    let host = format!("{host}\0");
    let f: Symbol<BudgetFn> = unsafe { lib.0.get(b"x509_verify_server_chain_budget\0").unwrap() };
    let mut left = budget;
    let cell = &mut left as *mut i64 as i64;
    let got = unsafe {
        f(
            chain.as_ptr() as i64,
            anchors.as_ptr() as i64,
            host.as_ptr() as i64,
            now_unix(),
            cell,
        )
    };
    (got, budget - left)
}

/// Seven copies of one self-signed issuer cost one verification, not one per
/// copy per path position; the copy kept still completes a valid path, and a
/// different certificate with the same name and DER length is its own
/// candidate.
#[test]
fn x509_repeated_certificate_is_one_candidate_issuer() {
    let Some(lib) = Lib::get() else { return };
    assert_eq!(
        run_budget(lib, RSA_COPIES, &["ec_root"], RSA_HOST, UNLIMITED),
        (-7, 1)
    );
    assert_eq!(run(lib, RSA_COPIES, &["ec_root"], RSA_HOST, now_unix()), -7);
    assert_eq!(
        run_budget(lib, INTER_COPIES, &["path_ec_root"], PATH_HOST, UNLIMITED),
        (0, 2)
    );
    assert_eq!(
        run(lib, INTER_COPIES, &["path_ec_root"], PATH_HOST, now_unix()),
        0
    );
    assert_eq!(
        run_budget(lib, SAME_LEN_ROOTS, &["ec_root"], PATH_HOST, UNLIMITED),
        (-3, 4)
    );
    // The reported shape: a self-signed leaf and seven copies of it. The
    // leaf's identity is used from the start, so no copy is even tried.
    let copies = blob_of(&vec![self_issued_ca_leaf(); 8]);
    let ec_root = blob(&["ec_root"]);
    assert_eq!(
        run_budget_blob(lib, &copies, &ec_root, PATH_HOST, UNLIMITED),
        (-7, 0)
    );
    let f: Symbol<ChainFn> = unsafe { lib.0.get(b"x509_verify_server_chain\0").unwrap() };
    let host = format!("{PATH_HOST}\0");
    let (c, a, h) = (
        copies.as_ptr() as i64,
        ec_root.as_ptr() as i64,
        host.as_ptr() as i64,
    );
    assert_eq!(unsafe { f(c, a, h, now_unix()) }, -7);
}

/// DER element: tag, minimal definite length, body.
fn tlv(tag: u8, body: &[u8]) -> Vec<u8> {
    let n = body.len();
    let mut out = match n {
        0..=127 => vec![tag, n as u8],
        128..=255 => vec![tag, 0x81, n as u8],
        _ => vec![tag, 0x82, (n >> 8) as u8, n as u8],
    };
    out.extend_from_slice(body);
    out
}

/// (header length, total length) of the DER element at `pos`.
fn der_extent(d: &[u8], pos: usize) -> (usize, usize) {
    let b = d[pos + 1] as usize;
    if b < 0x80 {
        return (2, 2 + b);
    }
    let k = b & 0x7f;
    let n = d[pos + 2..pos + 2 + k]
        .iter()
        .fold(0, |a, &x| (a << 8) | x as usize);
    (2 + k, 2 + k + n)
}

/// A self-issued CA leaf for path.example.com (cA, serverAuth EKU, SAN, no
/// keyUsage) with path_ec_leaf's key and a signature that does not verify: a
/// copy of it is a candidate issuer only if one identity can enter a path
/// twice, and that try is a counted verification.
fn self_issued_ca_leaf() -> Vec<u8> {
    let leaf = read_der("path_ec_leaf");
    let mut p = der_extent(&leaf, 0).0;
    p += der_extent(&leaf, p).0;
    for _ in 0..6 {
        // version, serial, signature, issuer, validity, subject
        p += der_extent(&leaf, p).1;
    }
    let spki = leaf[p..p + der_extent(&leaf, p).1].to_vec();
    let ext = |id: u8, critical: bool, value: Vec<u8>| {
        let flag = if critical {
            tlv(0x01, &[0xff])
        } else {
            Vec::new()
        };
        tlv(
            0x30,
            &[tlv(0x06, &[0x55, 0x1d, id]), flag, tlv(0x04, &value)].concat(),
        )
    };
    let alg = tlv(
        0x30,
        &tlv(0x06, &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02]),
    );
    let rdn = [tlv(0x06, &[0x55, 0x04, 0x03]), tlv(0x0c, b"self leaf")].concat();
    let name = tlv(0x30, &tlv(0x31, &tlv(0x30, &rdn)));
    let validity = [tlv(0x17, b"260101000000Z"), tlv(0x17, b"290101000000Z")].concat();
    let exts = [
        ext(0x13, true, tlv(0x30, &tlv(0x01, &[0xff]))),
        ext(
            0x25,
            false,
            tlv(0x30, &tlv(0x06, &[0x2b, 6, 1, 5, 5, 7, 3, 1])),
        ),
        ext(0x11, false, tlv(0x30, &tlv(0x82, PATH_HOST.as_bytes()))),
    ];
    let tbs = [
        tlv(0xa0, &tlv(0x02, &[2])),
        tlv(0x02, &[1]),
        alg.clone(),
        name.clone(),
        tlv(0x30, &validity),
        name,
        spki,
        tlv(0xa3, &tlv(0x30, &exts.concat())),
    ];
    let sig = [
        vec![0u8],
        tlv(0x30, &[tlv(0x02, &[1]), tlv(0x02, &[1])].concat()),
    ]
    .concat();
    tlv(
        0x30,
        &[tlv(0x30, &tbs.concat()), alg, tlv(0x03, &sig)].concat(),
    )
}

#[rustfmt::skip]
const SIG_BUDGET: &[Row] = &[
    ("budget_64_spent", THREE, WRONG_31, PATH_HOST, 0, -3),
    ("budget_65th_refused_in_bag", FOUR, WRONG_31, PATH_HOST, 0, -13),
    ("budget_65th_refused_at_anchor", THREE, WRONG_32, PATH_HOST, 0, -13),
];

/// The search counts every verification, stops at the caller's budget (the
/// boundary is exact) and `x509_verify_server_chain` allows exactly 64.
#[test]
fn x509_path_search_spends_at_most_64_signature_verifications() {
    let Some(lib) = Lib::get() else { return };
    check_rows(lib, SIG_BUDGET);
    assert_eq!(
        run_budget(lib, THREE, WRONG_31, PATH_HOST, UNLIMITED),
        (-3, 64)
    );
    assert_eq!(
        run_budget(lib, FOUR, WRONG_31, PATH_HOST, UNLIMITED),
        (-3, 65)
    );
    assert_eq!(
        run_budget(lib, THREE, WRONG_32, PATH_HOST, UNLIMITED),
        (-3, 66)
    );
    assert_eq!(run_budget(lib, THREE, WRONG_31, PATH_HOST, 64), (-3, 64));
    assert_eq!(run_budget(lib, THREE, WRONG_31, PATH_HOST, 63), (-13, 63));
    assert_eq!(
        run_budget(lib, THREE, &["path_ec_root"], PATH_HOST, 2),
        (0, 2)
    );
    assert_eq!(
        run_budget(lib, THREE, &["path_ec_root"], PATH_HOST, 1),
        (-13, 1)
    );
    // A budget cell pointer <= 0 is refused like the other pointers.
    let f: Symbol<BudgetFn> = unsafe { lib.0.get(b"x509_verify_server_chain_budget\0").unwrap() };
    let (chain, anchors) = (blob(THREE), blob(&["path_ec_root"]));
    let host = format!("{PATH_HOST}\0");
    let got = unsafe {
        f(
            chain.as_ptr() as i64,
            anchors.as_ptr() as i64,
            host.as_ptr() as i64,
            now_unix(),
            0,
        )
    };
    assert_eq!(got, -1, "a null budget cell must be refused");
}
