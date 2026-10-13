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
//! Compiles std/x509.mind (+ the std.crypto modules it imports, concatenated
//! exactly like tests/x509/build_shared.sh) via `mindc --emit-shared`,
//! dlopen-s the resulting `.so`, and replays the full offline fixture matrix
//! (committed openssl-generated DER chains under tests/x509/testdata/):
//! positive EC / RSA-PKCS1 / RSA-PSS chains, wildcard + A-label matching, and
//! every negative class (wrong host, unknown anchor, expired, not-yet-valid,
//! broken signature, non-CA intermediate, pathLen, missing serverAuth EKU,
//! missing SAN, truncated DER, unknown critical extension, depth > 8).
//!
//! Gate: `cargo test --features "mlir-build std-surface cross-module-imports"
//!        --test x509_chain_smoke`.
//! Self-skips when mlir-opt / mlir-translate / clang are absent, exactly like
//! `sha256_smoke.rs`.
//!
//! The native (non-MLIR) backend twin of this matrix lives in
//! tests/x509/build_native.sh; both must agree.

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]

mod common;
use common::mindc_bin;

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

type ChainFn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FlagFn = unsafe extern "C" fn() -> i64;

const TD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/x509/testdata");

fn read_der(name: &str) -> Vec<u8> {
    std::fs::read(format!("{TD}/{name}.der")).expect("read fixture")
}

/// Pack the chain/anchor blob: i64 count, (i64 off, i64 len) pairs, data.
fn blob(names: &[&str]) -> Vec<u8> {
    let ders: Vec<Vec<u8>> = names.iter().map(|n| read_der(n)).collect();
    let n = ders.len();
    let mut off = 8 + 16 * n;
    let mut out = Vec::new();
    out.extend_from_slice(&(n as i64).to_le_bytes());
    for d in &ders {
        out.extend_from_slice(&(off as i64).to_le_bytes());
        out.extend_from_slice(&(d.len() as i64).to_le_bytes());
        off += d.len();
    }
    for d in &ders {
        out.extend_from_slice(d);
    }
    out
}

fn now_unix() -> i64 {
    std::fs::read_to_string(format!("{TD}/now.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

/// Build the combined x509 shared library once.
fn build_x509_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                crate::common::gate::skipped(
                    "x509_chain_smoke",
                    &format!("x509_chain_smoke: {tool} not on PATH; skipping"),
                );
                return None;
            }
        }
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Concatenate the module set exactly as tests/x509/build_shared.sh.
        let mut combined = String::new();
        for m in ["sha256", "sha512", "ecdsa_p256", "x509", "rsa_pss"] {
            let src = std::fs::read_to_string(manifest_dir.join("std").join(format!("{m}.mind")))
                .expect("read std module");
            for line in src.lines() {
                if !line.starts_with("import ") {
                    combined.push_str(line);
                    combined.push('\n');
                }
            }
        }
        let dir = crate::common::scratch_dir("x509_chain_smoke");
        let src_path = dir.join("x509_combined.mind");
        std::fs::write(&src_path, combined).expect("write combined source");
        let so_path = dir.join("mind_x509_chain_smoke.so");
        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .env("MINDC_BUILD_TOOL_TIMEOUT_SECS", "900")
            .status()
            .expect("spawn mindc for x509_chain_smoke");
        if !status.success() {
            crate::common::gate::skipped(
                "x509_chain_smoke",
                "mindc --emit-shared failed for combined std/x509.mind",
            );
            return None;
        }
        Some(so_path)
    })
    .as_ref()
}

struct Lib(&'static Library);

impl Lib {
    fn get() -> Option<&'static Lib> {
        static LIB: OnceLock<Option<Lib>> = OnceLock::new();
        LIB.get_or_init(|| {
            let so = build_x509_so()?;
            let lib = unsafe { Library::new(so).expect("dlopen x509 .so") };
            let lib: &'static Library = Box::leak(Box::new(lib));
            Some(Lib(lib))
        })
        .as_ref()
    }
}

fn run(lib: &Lib, chain: &[&str], anchors: &[&str], dns: &str, now: i64) -> i64 {
    let chain = blob(chain);
    let anchors = blob(anchors);
    let mut dnsb = dns.as_bytes().to_vec();
    dnsb.push(0);
    let f: Symbol<ChainFn> = unsafe { lib.0.get(b"x509_verify_server_chain\0").unwrap() };
    unsafe {
        f(
            chain.as_ptr() as i64,
            anchors.as_ptr() as i64,
            dnsb.as_ptr() as i64,
            now,
        )
    }
}

#[test]
fn x509_peer_auth_supported_is_one() {
    let Some(lib) = Lib::get() else { return };
    let f: Symbol<FlagFn> = unsafe { lib.0.get(b"x509_peer_auth_supported\0").unwrap() };
    assert_eq!(unsafe { f() }, 1, "peer-auth capability must be 1");
}

#[test]
fn x509_chain_matrix() {
    let Some(lib) = Lib::get() else { return };
    let now = now_unix();
    let past = now - 3800 * 86400;
    let future = now + 3800 * 86400;
    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, Vec<&str>, Vec<&str>, &str, i64, i64)> = vec![
        (
            "ec_ok",
            vec!["ec_leaf", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            now,
            0,
        ),
        (
            "ec_casehost",
            vec!["ec_leaf", "ec_inter"],
            vec!["ec_root"],
            "WWW.EXAMPLE.COM",
            now,
            0,
        ),
        (
            "rsa_ok",
            vec!["rsa_leaf"],
            vec!["rsa_root"],
            "www.rsa-test.example.com",
            now,
            0,
        ),
        (
            "pss_ok",
            vec!["pss_leaf"],
            vec!["pss_root"],
            "www.pss-test.example.com",
            now,
            0,
        ),
        (
            "root_in_chain",
            vec!["ec_leaf", "ec_inter", "ec_root"],
            vec!["ec_root"],
            "www.example.com",
            now,
            0,
        ),
        (
            "extra_anchor",
            vec!["ec_leaf", "ec_inter"],
            vec!["pl_root", "ec_root"],
            "www.example.com",
            now,
            0,
        ),
        (
            "wild_ok",
            vec!["ec_leaf_wild", "ec_inter"],
            vec!["ec_root"],
            "www.wild.example.com",
            now,
            0,
        ),
        (
            "wild_bare",
            vec!["ec_leaf_wild", "ec_inter"],
            vec!["ec_root"],
            "wild.example.com",
            now,
            -8,
        ),
        (
            "wild_two_labels",
            vec!["ec_leaf_wild", "ec_inter"],
            vec!["ec_root"],
            "a.b.wild.example.com",
            now,
            -8,
        ),
        (
            "alabel_ok",
            vec!["ec_leaf_wild", "ec_inter"],
            vec!["ec_root"],
            "xn--nxasmq6b.example.com",
            now,
            0,
        ),
        (
            "alabel_ulabel_mismatch",
            vec!["ec_leaf_wild", "ec_inter"],
            vec!["ec_root"],
            "m\u{fc}ller.example.com",
            now,
            -8,
        ),
        (
            "wrong_host",
            vec!["ec_leaf", "ec_inter"],
            vec!["ec_root"],
            "other.example.com",
            now,
            -8,
        ),
        (
            "unknown_anchor",
            vec!["ec_leaf", "ec_inter"],
            vec!["pl_root"],
            "www.example.com",
            now,
            -7,
        ),
        (
            "empty_anchor",
            vec!["ec_leaf", "ec_inter"],
            vec![],
            "www.example.com",
            now,
            -7,
        ),
        (
            "expired",
            vec!["ec_leaf", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            future,
            -6,
        ),
        (
            "not_yet_valid",
            vec!["ec_leaf", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            past,
            -5,
        ),
        (
            "broken_sig",
            vec!["ec_leaf_badsig", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            now,
            -3,
        ),
        (
            "non_ca_inter",
            vec!["noca_leaf", "noca_inter"],
            vec!["noca_root"],
            "noca.example.com",
            now,
            -9,
        ),
        (
            "pathlen",
            vec!["pl_leaf", "pl_i2", "pl_i1"],
            vec!["pl_root"],
            "pl.example.com",
            now,
            -10,
        ),
        (
            "no_eku",
            vec!["ec_leaf_noeku", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            now,
            -11,
        ),
        (
            "no_san",
            vec!["ec_leaf_nosan", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            now,
            -15,
        ),
        (
            "truncated",
            vec!["ec_leaf_trunc", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            now,
            -1,
        ),
        (
            "unknown_crit",
            vec!["ec_leaf_crit", "ec_inter"],
            vec!["ec_root"],
            "www.example.com",
            now,
            -12,
        ),
        (
            "too_deep",
            vec!["ec_leaf"; 9],
            vec!["ec_root"],
            "www.example.com",
            now,
            -13,
        ),
    ];
    for (name, chain, anchors, dns, when, want) in cases {
        let got = run(lib, &chain, &anchors, dns, when);
        assert_eq!(got, want, "case {name}: got {got}, want {want}");
    }
}
