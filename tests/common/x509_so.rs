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

//! Shared fixture and shared-library helpers for the x509 chain tests
//! (`tests/x509_chain_smoke.rs`, `tests/x509_pss_salt.rs`): the std/x509
//! module set built once with `mindc --emit-shared`, the DER chain blob
//! layout, and the verifier call.

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]
#![allow(dead_code)]

use super::mindc_bin;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

pub type ChainFn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;

pub const TD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/x509/testdata");

pub fn read_der(name: &str) -> Vec<u8> {
    std::fs::read(format!("{TD}/{name}.der")).expect("read fixture")
}

/// Pack the chain/anchor blob: i64 count, (i64 off, i64 len) pairs, data.
pub fn blob(names: &[&str]) -> Vec<u8> {
    blob_of(&names.iter().map(|n| read_der(n)).collect::<Vec<_>>())
}

pub fn blob_of(ders: &[Vec<u8>]) -> Vec<u8> {
    let n = ders.len();
    let mut off = 8 + 16 * n;
    let mut out = Vec::new();
    out.extend_from_slice(&(n as i64).to_le_bytes());
    for d in ders {
        out.extend_from_slice(&(off as i64).to_le_bytes());
        out.extend_from_slice(&(d.len() as i64).to_le_bytes());
        off += d.len();
    }
    for d in ders {
        out.extend_from_slice(d);
    }
    out
}

pub fn now_unix() -> i64 {
    std::fs::read_to_string(format!("{TD}/now.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

/// Build the combined x509 shared library once.
pub fn build_x509_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                super::gate::skipped(
                    "x509_chain_smoke",
                    &format!("x509_chain_smoke: {tool} not on PATH; skipping"),
                );
                return None;
            }
        }
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Concatenate the module set exactly as X509_SET in scripts/run_crypto_vectors.sh.
        let mut combined = String::new();
        for m in [
            "sha256",
            "sha512",
            "ecdsa_p256",
            "x509_p384",
            "x509_bignum",
            "rsa_pss",
            "x509_bundle",
            "x509_der",
            "x509_ext",
            "x509_name",
            "x509_sig",
            "x509_path",
            "x509",
        ] {
            let src = std::fs::read_to_string(manifest_dir.join("std").join(format!("{m}.mind")))
                .expect("read std module");
            for line in src.lines() {
                if !line.starts_with("import ") {
                    combined.push_str(line);
                    combined.push('\n');
                }
            }
        }
        let dir = super::scratch_dir("x509_chain_smoke");
        let src_path = dir.join("x509_combined.mind");
        std::fs::write(&src_path, combined).expect("write combined source");
        let so_path = dir.join("mind_x509_chain_smoke.so");
        let out = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .env("MINDC_BUILD_TOOL_TIMEOUT_SECS", "900")
            .output()
            .expect("spawn mindc for x509_chain_smoke");
        // Shared fail-closed helper: a capability gap may skip (never under
        // MIND_BENCH_REQUIRE=1); any other compile failure panics.
        if !super::gate::compiled("x509_chain_smoke", &out) {
            return None;
        }
        Some(so_path)
    })
    .as_ref()
}

pub struct Lib(pub &'static Library);

impl Lib {
    pub fn get() -> Option<&'static Lib> {
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

pub fn run(lib: &Lib, chain: &[&str], anchors: &[&str], dns: &str, now: i64) -> i64 {
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
