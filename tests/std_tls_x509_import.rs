// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A program that imports `std.x509` and `std.https` (the TLS 1.3 client) checks, builds,
//! links and runs, and the X.509 and TLS 1.3 known answers hold under
//! `mindc test` and in the executable `mindc build` writes.
//!
//! Each program is generated here from committed fixtures: the DER
//! certificates in `tests/x509/testdata/` (fixed clock `now.txt`), the RFC 8448
//! §3 "Simple 1-RTT" handshake bytes and secrets, and the recorded TLS 1.3
//! flights in `tests/fixtures/tls13_client/`. A fixture is written into the
//! program as stores, so neither engine reads a file. Every program returns
//! the number of failed checks from `main`; a mutated copy with exactly one
//! wrong expectation must report a failure on every engine that runs it, so
//! no check here can pass by asserting nothing.
//!
//! Cost decides which engine runs which set. The tree-walking evaluator takes
//! tens of seconds per RSA-2048 verification and minutes per ECDSA P-256 one,
//! so the `mindc test` sets use RSA chains, the checks that refuse before any
//! signature is verified, the IP literal parsers and the anchor-bundle loader.
//! The build sets add the ECDSA chains that verify and the recorded flights.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;
#[path = "support/tls13_leaf_key_program.rs"]
mod tls13_leaf_key_program;
#[path = "support/x509_literals_program.rs"]
mod x509_literals_program;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const TD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/x509/testdata");
#[cfg(all(unix, feature = "mlir-build"))]
const TLS_FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls13_client");
#[cfg(all(unix, feature = "mlir-build"))]
const TESTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests");

// ── Program assembly ───────────────────────────────────────────────────────

/// A MIND function `name(dst) -> i64` that writes `data` at `dst` with 8-byte
/// stores (byte stores for the tail) and returns its length.
fn stores(name: &str, data: &[u8]) -> String {
    let mut out = format!("fn {name}(dst: i64) -> i64 {{\n");
    for (i, chunk) in data.chunks(8).enumerate() {
        let off = i * 8;
        if chunk.len() == 8 {
            let v = i64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
            if v >= 0 {
                out.push_str(&format!("    __mind_store_i64(dst + {off}, {v});\n"));
                continue;
            }
            if v != i64::MIN {
                let neg = v.unsigned_abs();
                out.push_str(&format!("    __mind_store_i64(dst + {off}, 0 - {neg});\n"));
                continue;
            }
        }
        for (j, b) in chunk.iter().enumerate() {
            out.push_str(&format!("    __mind_store_i8(dst + {}, {b});\n", off + j));
        }
    }
    out.push_str(&format!("    return {};\n}}\n\n", data.len()));
    out
}

fn hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

/// The source of a program that runs `checks()`: `main` returns the number of
/// failed checks and one `#[test]` asserts there are none.
fn program(imports: &[&str], body: &str) -> String {
    let mut src: String = imports
        .iter()
        .map(|m| format!("import std.{m};\n"))
        .collect();
    src.push('\n');
    src.push_str(body);
    src.push_str(
        "#[test]\nfn known_answers() {\n    assert checks() == 0;\n}\n\n\
         fn main() -> i64 {\n    return checks();\n}\n",
    );
    src
}

/// A cpu-executable project in a fresh scratch directory for `case`.
fn project(case: &str, main: &str) -> PathBuf {
    let dir = common::scratch_dir("std_tls_x509_import").join(case);
    std::fs::create_dir_all(&dir).expect("create project dir");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"p\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"p\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(dir.join("main.mind"), main).expect("write main.mind");
    dir
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn mindc_in(dir: &Path, args: &[&str]) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn mindc {args:?}: {e}"))
}

/// `mindc test` must report the one known-answer test as passed (`pass`) or
/// as failed (`!pass`).
fn assert_interp(case: &str, dir: &Path, pass: bool) {
    let out = mindc_in(dir, &["test", "main.mind"]);
    let all = text(&out);
    let want = if pass {
        "1 passed; 0 failed"
    } else {
        "0 passed; 1 failed"
    };
    assert!(
        out.status.success() == pass && all.contains(want),
        "{case}: `mindc test` must report `{want}`:\n{all}"
    );
}

/// Build the project and run the executable; `None` when the build backend is
/// a capability gap the shared gate allows to skip.
#[cfg(all(unix, feature = "mlir-build"))]
fn build_and_run(case: &str, dir: &Path) -> Option<i32> {
    let out = mindc_in(dir, &["build"]);
    if !common::gate::compiled("std_tls_x509_import", &out) {
        return None;
    }
    let run = Command::new(dir.join("target").join("debug").join("p"))
        .output()
        .unwrap_or_else(|e| panic!("{case}: run the built executable: {e}"));
    Some(run.status.code().unwrap_or_else(|| {
        panic!(
            "{case}: the executable ended without an exit code:\n{}",
            text(&run)
        )
    }))
}

/// Build and run each `(case, project, expected exit)`; a no-op where the
/// build path is not compiled in.
#[cfg(all(unix, feature = "mlir-build"))]
fn assert_built(runs: &[(&str, &Path, i32)]) {
    for (case, dir, want) in runs {
        if let Some(code) = build_and_run(case, dir) {
            assert_eq!(
                code, *want,
                "{case}: the executable must report {want} failed check(s)"
            );
        }
    }
}

#[cfg(not(all(unix, feature = "mlir-build")))]
fn assert_built(_: &[(&str, &Path, i32)]) {}

// ── X.509 path validation ──────────────────────────────────────────────────

struct Chain {
    name: &'static str,
    chain: &'static [&'static str],
    anchors: &'static [&'static str],
    host: &'static str,
    /// Seconds added to the fixture clock.
    shift: i64,
    want: i64,
}

const DAY: i64 = 86_400;

const fn case(
    name: &'static str,
    chain: &'static [&'static str],
    anchors: &'static [&'static str],
    host: &'static str,
    shift: i64,
    want: i64,
) -> Chain {
    Chain {
        name,
        chain,
        anchors,
        host,
        shift,
        want,
    }
}

/// Refused before any signature is verified. Short DNS names (7 to 15 bytes,
/// first tried as an IPv4 literal) and IP literals are in this set, so both
/// engines run the literal parsers. The `eku_*` and `dns_ip_leaf` chains are
/// valid under `eku_root` (`eku_root_ok` in the build set): only the refused
/// rule stops them, an EKU without serverAuth or an IP literal against a
/// dNSName. `#outer_sigalg_sha384` changes only the outer signatureAlgorithm.
/// `kp_leaf_badkey` is `kp_leaf` with a third INTEGER in its RSAPublicKey,
/// signed again by `kp_root`: only the key is malformed (`kp_ok` below).
#[rustfmt::skip]
const X509_INTERP_FAST: &[Chain] = &[
    case("expired", &["ec_leaf", "ec_inter"], &["ec_root"], "www.example.com", 3800 * DAY, -6),
    case("not_yet_valid", &["ec_leaf", "ec_inter"], &["ec_root"], "www.example.com", -3800 * DAY, -5),
    case("hostname_mismatch", &["ec_leaf", "ec_inter"], &["ec_root"], "other.example.com", 0, -8),
    case("rsa_hostname_mismatch", &["rsa_leaf"], &["rsa_root"], "other.rsa-test.example.com", 0, -8),
    case("wildcard_bare_parent", &["ec_leaf_wild", "ec_inter"], &["ec_root"], "wild.example.com", 0, -8),
    case("wildcard_two_labels", &["ec_leaf_wild", "ec_inter"], &["ec_root"], "a.b.wild.example.com", 0, -8),
    case("wildcard_star_alone", &["path_wild_star_alone", "path_ec_inter"], &["path_ec_root"], "anything.example.com", 0, -8),
    case("wildcard_ulabel", &["ec_leaf_wild", "ec_inter"], &["ec_root"], "m\u{fc}ller.example.com", 0, -8),
    case("no_ca_path", &["rsa_leaf"], &["ec_root"], "www.rsa-test.example.com", 0, -7),
    case("no_anchor", &["rsa_leaf"], &[], "www.rsa-test.example.com", 0, -7),
    case("no_server_auth", &["ec_leaf_noeku", "ec_inter"], &["ec_root"], "www.example.com", 0, -11),
    case("no_san", &["ec_leaf_nosan", "ec_inter"], &["ec_root"], "www.example.com", 0, -15),
    case("short_hostname_mismatch", &["ec_leaf", "ec_inter"], &["ec_root"], "example.com", 0, -8),
    case("ip_mismatch", &["ec_leaf_ip", "ec_inter"], &["ec_root"], "192.168.1.1", 0, -8),
    case("ipv6_mismatch", &["path_ipv6", "path_ec_inter"], &["path_ec_root"], "2001:db8::2", 0, -8),
    case("dns_name_for_ip", &["ec_leaf", "ec_inter"], &["ec_root"], "127.0.0.1", 0, -8),
    case("ip_literal_in_dns_san", &["dns_ip_leaf"], &["eku_root"], "127.0.0.1", 0, -8),
    case("eku_without_server_auth", &["eku_client_leaf"], &["eku_root"], "eku.example.com", 0, -11),
    case("outer_sigalg_differs", &["ec_leaf#outer_sigalg_sha384", "ec_inter"], &["ec_root"], "www.example.com", 0, -1),
    case("leaf_only_not_yet_valid", &["ec_leaf_ip", "ec_inter"], &["ec_root"], "127.0.0.1", LEAF_IP_NB - 1, -5),
    case("leaf_rsa_key_malformed", &["kp_leaf_badkey"], &["kp_root"], "www.kp-test.example.com", 0, -14),
];

/// Seconds from the fixture clock to `ec_leaf_ip`'s notBefore. One second
/// earlier `ec_inter` and `ec_root` are already valid and only the leaf is
/// not; at notBefore the chain verifies (`leaf_valid_from` in the build set).
const LEAF_IP_NB: i64 = 1 - 2 * DAY;

/// One RSA-2048 PKCS#1 v1.5 verification each. `wrong_key` names the issuer
/// correctly but carries another modulus (see `x509_anchor_der`).
#[rustfmt::skip]
const X509_INTERP_RSA: &[Chain] = &[
    case("rsa_ok", &["rsa_leaf"], &["rsa_root"], "www.rsa-test.example.com", 0, 0),
    case("kp_ok", &["kp_leaf"], &["kp_root"], "www.kp-test.example.com", 0, 0),
    case("rsa_wrong_key", &["rsa_leaf"], &["rsa_root#wrong_key"], "www.rsa-test.example.com", 0, -3),
];

/// The build set adds ECDSA P-256 / P-384 and RSA-PSS chains, intermediates,
/// the IP address positives, name constraints and the wildcard positives.
#[cfg(all(unix, feature = "mlir-build"))]
#[rustfmt::skip]
const X509_BUILD: &[Chain] = &[
    case("ec_ok", &["ec_leaf", "ec_inter"], &["ec_root"], "www.example.com", 0, 0),
    case("ec_case_insensitive", &["ec_leaf", "ec_inter"], &["ec_root"], "WWW.EXAMPLE.COM", 0, 0),
    case("pss_ok", &["pss_leaf"], &["pss_root"], "www.pss-test.example.com", 0, 0),
    case("p384_ok", &["p384_leaf"], &["p384_root"], "www.p384-test.example.com", 0, 0),
    case("wildcard_ok", &["ec_leaf_wild", "ec_inter"], &["ec_root"], "www.wild.example.com", 0, 0),
    case("wildcard_alabel_ok", &["ec_leaf_wild", "ec_inter"], &["ec_root"], "xn--nxasmq6b.example.com", 0, 0),
    case("wildcard_partial", &["path_wild_partial", "path_ec_inter"], &["path_ec_root"], "foo.example.com", 0, -8),
    case("wildcard_double_star", &["path_wild_double_star", "path_ec_inter"], &["path_ec_root"], "a.b.com", 0, -8),
    case("wildcard_mid_label", &["path_m01_mid_star_dot", "path_ec_inter"], &["path_ec_root"], "z.*.com", 0, -8),
    case("wildcard_mid_cert", &["ec_leaf_midwild", "ec_inter"], &["ec_root"], "sub.foo.example.com", 0, -8),
    case("wrong_key_intermediate", &["path_ec_leaf", "path_wrongkey"], &["path_ec_root"], "path.example.com", 0, -3),
    case("wrong_key_anchor", &["path_ec_leaf", "path_ec_inter"], &["path_anchor_wrongkey"], "path.example.com", 0, -3),
    case("broken_signature", &["ec_leaf_badsig", "ec_inter"], &["ec_root"], "www.example.com", 0, -3),
    case("no_ca_path_ec", &["ec_leaf", "ec_inter"], &["pl_root"], "www.example.com", 0, -7),
    case("expired_intermediate", &["path_ec_leaf", "path_expired"], &["path_ec_root"], "path.example.com", 0, -6),
    case("future_intermediate", &["path_ec_leaf", "path_future"], &["path_ec_root"], "path.example.com", 0, -5),
    case("alternate_issuer", &["path_ec_leaf", "path_expired", "path_ec_inter"], &["path_ec_root"], "path.example.com", 0, 0),
    case("non_ca_intermediate", &["noca_leaf", "noca_inter"], &["noca_root"], "noca.example.com", 0, -9),
    case("path_length", &["pl_leaf", "pl_i2", "pl_i1"], &["pl_root"], "pl.example.com", 0, -10),
    case("ip_ok", &["ec_leaf_ip", "ec_inter"], &["ec_root"], "127.0.0.1", 0, 0),
    case("leaf_valid_from", &["ec_leaf_ip", "ec_inter"], &["ec_root"], "127.0.0.1", LEAF_IP_NB, 0),
    case("ipv6_ok", &["path_ipv6", "path_ec_inter"], &["path_ec_root"], "2001:db8::1", 0, 0),
    case("eku_root_ok", &["eku_server_leaf"], &["eku_root"], "eku.example.com", 0, 0),
    case("name_constraint_excluded", &["nc_leaf_bad", "nc_inter"], &["nc_root"], "bad.example.com", 0, -17),
    case("sha1_refused", &["sha1_leaf"], &["sha1_root"], "www.sha1-test.example.com", 0, -2),
    case("rsa3072_leaf_ok", &["kp_rsa3072_leaf"], &["kp_root"], "www.kp-test.example.com", 0, 0),
];

fn read_fixture(name: &str) -> Vec<u8> {
    let path = format!("{TD}/{name}.der");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// `rsa_root` with one byte of its modulus changed: the subject still names
/// the leaf's issuer, but the key is not the one that signed the leaf.
fn wrong_key(root: &[u8]) -> Vec<u8> {
    // INTEGER, two-byte length 257, leading zero: the 2048-bit modulus.
    let tag = [0x02, 0x82, 0x01, 0x01, 0x00];
    let at = root
        .windows(tag.len())
        .position(|w| w == tag)
        .expect("rsa_root carries a 2048-bit modulus");
    let mut out = root.to_vec();
    out[at + tag.len() + 100] ^= 0x01;
    out
}

/// An ECDSA P-256 / SHA-256 certificate whose outer signatureAlgorithm says
/// SHA-384 while tbsCertificate.signature still says SHA-256; the signed
/// bytes are unchanged.
fn outer_sigalg_sha384(cert: &[u8]) -> Vec<u8> {
    // OBJECT IDENTIFIER 1.2.840.10045.4.3.2 (ecdsa-with-SHA256).
    let oid = [0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
    let hits: Vec<usize> = (0..cert.len())
        .filter(|&i| cert[i..].starts_with(&oid))
        .collect();
    assert_eq!(hits.len(), 2, "inner and outer ecdsa-with-SHA256");
    let mut out = cert.to_vec();
    // ...4.3.3 is ecdsa-with-SHA384.
    out[hits[1] + oid.len() - 1] = 0x03;
    out
}

fn x509_anchor_der(name: &str) -> Vec<u8> {
    match name.split_once('#') {
        Some((base, "wrong_key")) => wrong_key(&read_fixture(base)),
        Some((base, "outer_sigalg_sha384")) => outer_sigalg_sha384(&read_fixture(base)),
        Some((_, tweak)) => panic!("unknown fixture tweak `{tweak}`"),
        None => read_fixture(name),
    }
}

fn fixture_fn(name: &str) -> String {
    format!("der_{}", name.replace('#', "_"))
}

/// The blob `x509_verify_server_chain` takes: an i64 count, (offset, length)
/// pairs, then the DER bytes.
fn blob_stmts(var: &str, names: &[&str], ders: &[Vec<u8>]) -> String {
    let size = 8 + 16 * names.len() + ders.iter().map(Vec::len).sum::<usize>();
    let mut s = format!(
        "    let {var}: i64 = __mind_alloc({size});\n    __mind_store_i64({var}, {});\n",
        names.len()
    );
    let mut off = 8 + 16 * names.len();
    for (i, (name, der)) in names.iter().zip(ders).enumerate() {
        s.push_str(&format!(
            "    __mind_store_i64({var} + {}, {off});\n    __mind_store_i64({var} + {}, {}({var} + {off}));\n",
            8 + 16 * i,
            16 + 16 * i,
            fixture_fn(name)
        ));
        off += der.len();
    }
    s
}

/// A program checking every case; `mutate` adds 100 to the expected verdict
/// of that case, which must then be the one failed check.
fn x509_program(cases: &[Chain], mutate: Option<usize>) -> String {
    let now: i64 = std::fs::read_to_string(format!("{TD}/now.txt"))
        .expect("read now.txt")
        .trim()
        .parse()
        .expect("now.txt holds unix seconds");
    let mut fixtures: Vec<&str> = cases
        .iter()
        .flat_map(|c| c.chain.iter().chain(c.anchors))
        .copied()
        .collect();
    fixtures.sort_unstable();
    fixtures.dedup();
    let mut body: String = fixtures
        .iter()
        .map(|n| stores(&fixture_fn(n), &x509_anchor_der(n)))
        .collect();
    for (i, c) in cases.iter().enumerate() {
        let chain: Vec<Vec<u8>> = c.chain.iter().map(|n| x509_anchor_der(n)).collect();
        let anchors: Vec<Vec<u8>> = c.anchors.iter().map(|n| x509_anchor_der(n)).collect();
        let mut host = c.host.as_bytes().to_vec();
        host.push(0);
        let want = c.want + if mutate == Some(i) { 100 } else { 0 };
        body.push_str(&stores(&format!("host_{i}"), &host));
        body.push_str(&format!("// {}\nfn case_{i}() -> i64 {{\n", c.name));
        body.push_str(&blob_stmts("c", c.chain, &chain));
        body.push_str(&blob_stmts("a", c.anchors, &anchors));
        body.push_str(&format!(
            "    let h: i64 = __mind_alloc({});\n    host_{i}(h);\n\
             \x20   let rc: i64 = x509_verify_server_chain(c, a, h, {});\n\
             \x20   if rc == {want} {{\n        return 0;\n    }}\n    return 1;\n}}\n\n",
            host.len(),
            now + c.shift
        ));
    }
    let calls: Vec<String> = (0..cases.len()).map(|i| format!("case_{i}()")).collect();
    body.push_str(&format!(
        "fn checks() -> i64 {{\n    return {};\n}}\n\n",
        calls.join(" + ")
    ));
    program(&["x509"], &body)
}

fn x509_interp_and_build(label: &str, cases: &[Chain]) {
    let dir = project(label, &x509_program(cases, None));
    assert_interp(label, &dir, true);
    let mutant = format!("{label}_mutated");
    let mdir = project(&mutant, &x509_program(cases, Some(0)));
    assert_interp(&mutant, &mdir, false);
    assert_built(&[
        (label, dir.as_path(), 0),
        (mutant.as_str(), mdir.as_path(), 1),
    ]);
}

/// Expiry, not-yet-valid, no CA path, hostname mismatch and the wildcard
/// rules refuse with their verdicts under `mindc test` and in the executable.
#[test]
fn x509_refusals_before_any_signature_agree_in_test_and_build() {
    x509_interp_and_build("x509_fast", X509_INTERP_FAST);
}

/// Two valid RSA chains verify (0) and one of them under an anchor with the
/// issuer's name and another key is refused (-3), on both engines. A debug
/// build of the evaluator needs about 7 minutes per verification, so this
/// runs in release builds only; the build set runs every chain in every
/// profile.
#[cfg_attr(
    debug_assertions,
    ignore = "each RSA-2048 verification takes about 7 minutes in a debug-profile mindc test"
)]
#[test]
fn x509_rsa_chain_and_wrong_key_agree_in_test_and_build() {
    x509_interp_and_build("x509_rsa", X509_INTERP_RSA);
}

/// Every chain of both `mindc test` sets plus the ECDSA, RSA-PSS, IP address,
/// intermediate and name-constraint cases, in one executable.
#[cfg(all(unix, feature = "mlir-build"))]
#[test]
fn x509_full_set_runs_in_the_built_executable() {
    let cases: Vec<Chain> = X509_BUILD
        .iter()
        .chain(X509_INTERP_FAST)
        .chain(X509_INTERP_RSA)
        .map(|c| case(c.name, c.chain, c.anchors, c.host, c.shift, c.want))
        .collect();
    let dir = project("x509_build", &x509_program(&cases, None));
    let mdir = project("x509_build_mutated", &x509_program(&cases, Some(0)));
    assert_built(&[
        ("x509_build", dir.as_path(), 0),
        ("x509_build_mutated", mdir.as_path(), 1),
    ]);
}

/// `differs(a, b, len)`: 1 when the `len` bytes at `a` and `b` differ, else 0.
const DIFFERS: &str = "fn differs(a: i64, b: i64, len: i64) -> i64 {\n    let mut i: i64 = 0;\n\
     \x20   let mut d: i64 = 0;\n    while i < len {\n\
     \x20       d = d | ((__mind_load_i8(a + i) ^ __mind_load_i8(b + i)) & 255);\n\
     \x20       i = i + 1;\n    }\n    if d == 0 {\n        return 0;\n    }\n    return 1;\n}\n\n";

/// The IPv4 and IPv6 literal parsers and the DER and PEM anchor loader give
/// the same answers under `mindc test` as in the executable: their loops end
/// on a flag, not on `break`. One wrong anchor count fails both engines.
#[test]
fn x509_ip_literals_and_anchor_bundles_agree_in_test_and_build() {
    let dir = project("x509_literals", &x509_literals_program::source(false));
    assert_interp("x509_literals", &dir, true);
    let mdir = project(
        "x509_literals_mutated",
        &x509_literals_program::source(true),
    );
    assert_interp("x509_literals_mutated", &mdir, false);
    assert_built(&[
        ("x509_literals", dir.as_path(), 0),
        ("x509_literals_mutated", mdir.as_path(), 1),
    ]);
}

// ── The import, the link and the capability ────────────────────────────────

const CAPABILITY: &str = "fn checks() -> i64 {\n    let st: i64 = __mind_alloc(tls13c_state_size());\n\
     \x20   if st == 0 {\n        return 2;\n    }\n    return 1 - x509_peer_auth_supported();\n}\n\n";

/// `import std.x509; import std.https;` resolves, links and runs, and
/// `x509_peer_auth_supported()` reports the implemented path validation.
#[test]
fn a_program_importing_x509_and_the_tls_client_builds_and_runs() {
    let dir = project("capability", &program(&["x509", "https"], CAPABILITY));
    let out = mindc_in(&dir, &["check", "main.mind"]);
    assert!(
        out.status.success(),
        "`mindc check` must accept the imports:\n{}",
        text(&out)
    );
    assert_interp("capability", &dir, true);
    // `main` returns 1 - x509_peer_auth_supported(), so 0 means it reported 1.
    assert_built(&[("capability", dir.as_path(), 0)]);
}

// ── The CertificateVerify key (tests/support/tls13_leaf_key_program.rs) ────

/// `x509_parse` records n and e only from a strictly decoded rsaEncryption
/// key, leaves them zero for an ECDSA or other key and rejects a malformed
/// one; the TLS client refuses a leaf without an RSA key at CertificateVerify,
/// and a malformed RSA leaf (E_CHAIN) and an RSA-3072 leaf (E_OVF) at
/// Certificate, on both engines.
#[test]
fn the_cv_key_is_a_verified_rsa_key_in_test_and_build() {
    let dir = project("leaf_key", &tls13_leaf_key_program::source(false, false));
    assert_interp("leaf_key", &dir, true);
    let mdir = project(
        "leaf_key_mutated",
        &tls13_leaf_key_program::source(false, true),
    );
    assert_interp("leaf_key_mutated", &mdir, false);
    assert_built(&[
        ("leaf_key", dir.as_path(), 0),
        ("leaf_key_mutated", mdir.as_path(), 1),
    ]);
}

/// Through the handshake state machine and the real chain verifier: an ECDSA
/// leaf whose chain verifies (verdict 0) is refused at CertificateVerify for
/// rsa_pss_rsae_sha256, and a valid RSA chain still verifies and records its
/// key. ECDSA verification runs in the executable only.
#[cfg(all(unix, feature = "mlir-build"))]
#[test]
fn an_ecdsa_leaf_with_a_valid_chain_is_refused_at_certificate_verify() {
    let dir = project(
        "leaf_key_flow",
        &tls13_leaf_key_program::source(true, false),
    );
    let mdir = project(
        "leaf_key_flow_mutated",
        &tls13_leaf_key_program::source(true, true),
    );
    assert_built(&[
        ("leaf_key_flow", dir.as_path(), 0),
        ("leaf_key_flow_mutated", mdir.as_path(), 1),
    ]);
}

// ── TLS 1.3 transcript and key schedule (RFC 8448 §3) ──────────────────────

const RFC_CH: &str = "010000c00303cb34ecb1e78163ba1c38c6dacb196a6dffa21a8d9912ec18a2ef\
6283024dece7000006130113031302010000910000000b0009000006736572766572ff01000100000a00140012001d0017\
001800190100010101020103010400230000003300260024001d002099381de560e4bd43d23d8e435a7dbafeb3c06e\
51c13cae4d5413691e529aaf2c002b0003020304000d0020001e040305030603020308040805080604010501060102010402050206020202002d00020101001c00024001";
const RFC_SH: &str = "020000560303a6af06a4121860dc5e6e60249cd34c95930c8ac5cb1434dac155\
772ed3e2692800130100002e00330024001d0020c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f002b00020304";
const RFC_CLIENT_PRIV: &str = "49af42ba7f7994852d713ef2784bcbcaa7911de26adc5642cb634540e7ea5005";
const RFC_SERVER_PUB: &str = "c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f";

/// `(state offset, RFC 8448 §3 value)`, at the offsets the state layout in
/// std/tls13_client_core.mind documents.
const RFC_HS: &[(usize, &str)] = &[
    (
        5176,
        "860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8",
    ),
    (
        4064,
        "1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac",
    ),
    (
        4096,
        "b3eddb126e067f35a780b3abf45e2d8f3b1a950738f52e9600746a0e27a55a21",
    ),
    (
        4128,
        "b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38",
    ),
    (4160, "dbfaa693d1762c5b666af5d950258d01"),
    (81928, "5bd3c71b836e0b76bb73265f"),
    (4192, "3fce516009c21727d0f2e4e86ee403bc"),
    (81940, "5d313eb2671276ee13000b30"),
];
const RFC_MASTER: (usize, &str) = (
    4224,
    "18df06843d13a08bf2a449844c5f8a478001bc4d4c627984d5a41da8d0402919",
);

/// The client transcript over the RFC ClientHello and ServerHello, then the
/// X25519 handshake secret, both handshake traffic secrets, their keys and
/// IVs, and the master secret. `mutate` flips one expected bit.
fn tls_keyschedule_program(mutate: bool) -> String {
    let mut body = stores("rfc_ch", &hex(RFC_CH));
    body.push_str(&stores("rfc_sh", &hex(RFC_SH)));
    body.push_str(&stores("rfc_priv", &hex(RFC_CLIENT_PRIV)));
    body.push_str(&stores("rfc_pub", &hex(RFC_SERVER_PUB)));
    let mut expect = String::new();
    for (k, (off, v)) in RFC_HS.iter().chain([&RFC_MASTER]).enumerate() {
        let mut want = hex(v);
        if mutate && k == 0 {
            want[0] ^= 1;
        }
        body.push_str(&stores(&format!("want_{k}"), &want));
        expect.push_str(&format!(
            "    want_{k}(w);\n    n = n + differs(s + {off}, w, {});\n",
            want.len()
        ));
        if k == RFC_HS.len() - 1 {
            expect.push_str("    tls13c_derive_app_keys(s);\n");
        }
    }
    body.push_str(DIFFERS);
    body.push_str(&format!(
        "fn checks() -> i64 {{\n    let s: i64 = __mind_alloc(tls13c_state_size());\n\
         \x20   tls13c_zero(s, tls13c_state_size());\n\
         \x20   tls13c_store_i64(s + 40, tls13ci_suite_aes128gcm_sha256());\n\
         \x20   tls13c_store_i64(s + 48, tls13ci_group_x25519());\n\
         \x20   rfc_priv(s + 320);\n    rfc_pub(s + 384);\n    let m: i64 = __mind_alloc(256);\n\
         \x20   tls13c_transcript_append(s, m, rfc_ch(m));\n\
         \x20   tls13c_transcript_append(s, m, rfc_sh(m));\n\
         \x20   if tls13c_derive_hs_keys(s) != 0 {{\n        return 100;\n    }}\n\
         \x20   let w: i64 = __mind_alloc(64);\n    let mut n: i64 = 0;\n{expect}    return n;\n}}\n\n"
    ));
    program(&["https", "tls13_client_core"], &body)
}

/// The RFC 8448 §3 transcript hash and key schedule hold under `mindc test`
/// and in the executable; one flipped expected bit fails both.
#[test]
fn tls13_transcript_and_key_schedule_match_rfc8448_in_test_and_build() {
    let dir = project("tls_keyschedule", &tls_keyschedule_program(false));
    assert_interp("tls_keyschedule", &dir, true);
    let mdir = project("tls_keyschedule_mutated", &tls_keyschedule_program(true));
    assert_interp("tls_keyschedule_mutated", &mdir, false);
    assert_built(&[
        ("tls_keyschedule", dir.as_path(), 0),
        ("tls_keyschedule_mutated", mdir.as_path(), 1),
    ]);
}

// ── TLS 1.3 recorded flights ───────────────────────────────────────────────

/// The std modules the recorded-flight cases and their fixture PKI call.
#[cfg(all(unix, feature = "mlir-build"))]
const FLIGHT_IMPORTS: &[&str] = &[
    "sha256",
    "hkdf",
    "x509",
    "rsa_pss",
    "x509_bignum",
    "tls13_keyschedule",
    "tls13_finished",
    "tls13_handshake",
    "x25519",
    "x25519mlkem768",
    "tls13_client_core",
    "tls13_client_hello",
    "tls13_client_api",
    "https_url",
    "https_http1",
    "https",
];

/// The 24 recorded flights: suite x group x HelloRetryRequest x tamper mode.
#[cfg(all(unix, feature = "mlir-build"))]
fn flight_files() -> Vec<String> {
    let mut out = Vec::new();
    for suite in ["1301", "1303"] {
        for group in ["001d", "11ec"] {
            for hrr in 0..2 {
                for tamper in 0..3 {
                    let t = if tamper == 0 {
                        String::new()
                    } else {
                        format!("-tamper{tamper}")
                    };
                    out.push(format!("{suite}-{group}-hrr{hrr}{t}.bin"));
                }
            }
        }
    }
    out
}

/// The recorded-flight suite of tests/tls13_client_runner.mind and the
/// shipped-verifier suite of tests/tls13_client_shipped.mind; `mutate` flips
/// one byte of the server flight of the first positive recording.
#[cfg(all(unix, feature = "mlir-build"))]
fn flights_program(mutate: bool) -> String {
    let read = |p: String| std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let mut body = String::new();
    for f in [
        "fixture_keys.mind",
        "rfc8448.mind",
        "generate.mind",
        "cvsigs.mind",
    ] {
        body.push_str(&read(format!("{TLS_FIX}/{f}")));
        body.push('\n');
    }
    for f in [
        "tls13_client_cases.mind",
        "tls13_client_runner.mind",
        "tls13_client_shipped.mind",
        "tls13_client_rx.mind",
    ] {
        body.push_str(&read(format!("{TESTS}/{f}")));
        body.push('\n');
    }
    let files = flight_files();
    for (i, f) in files.iter().enumerate() {
        let path = format!("{TLS_FIX}/{f}");
        let mut data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        if mutate && i == 0 {
            // Twelve i64 header fields, then CH1, CH2, HRR and SH; the byte
            // after them is the first byte of the encrypted server flight.
            let len = |k: usize| {
                i64::from_le_bytes(data[k * 8..k * 8 + 8].try_into().expect("header field"))
                    as usize
            };
            let at = 96 + len(3) + len(4) + len(5) + len(6);
            data[at + 8] ^= 1;
        }
        body.push_str(&stores(&format!("tls13ct_frozen_{i}"), &data));
    }
    body.push_str("fn tls13ct_frozen(i: i64, dst: i64) -> i64 {\n");
    for i in 0..files.len() {
        body.push_str(&format!(
            "    if i == {i} {{\n        return tls13ct_frozen_{i}(dst);\n    }}\n"
        ));
    }
    body.push_str("    return 0;\n}\n\n");
    body.push_str(
        "fn checks() -> i64 {\n    let all: i64 = tls13ct_run_all();\n    if all != 0 {\n\
         \x20       return 1;\n    }\n    if tls13ct_run_shipped() != 0 {\n        return 2;\n    }\n\
         \x20   let buf: i64 = __mind_alloc(16384);\n    let n: i64 = tls13ct_frozen(0, buf);\n\
         \x20   if tls13ct_run_rx(buf, n) != 0 {\n        return 3;\n    }\n    return 0;\n}\n\n",
    );
    program(FLIGHT_IMPORTS, &body)
}

/// Eight recorded TLS 1.3 handshakes (AES-128-GCM and ChaCha20-Poly1305,
/// X25519 and X25519MLKEM768, with and without HelloRetryRequest) reach
/// CONNECTED through the real chain verifier, their expired, wrong-name,
/// unknown-anchor and tampered twins are refused, and the refusal latches,
/// in the executable. One flipped byte of a recorded flight fails it.
#[cfg(all(unix, feature = "mlir-build"))]
#[test]
fn tls13_recorded_flights_run_in_the_built_executable() {
    let dir = project("tls_flights", &flights_program(false));
    let mdir = project("tls_flights_mutated", &flights_program(true));
    assert_built(&[
        ("tls_flights", dir.as_path(), 0),
        ("tls_flights_mutated", mdir.as_path(), 1),
    ]);
}
