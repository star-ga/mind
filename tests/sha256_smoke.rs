// Copyright 2025 STARGA Inc.
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

//! std.sha256 execution smoke test — RFC 0016 evidence_chain.trace_hash
//! prerequisite.
//!
//! Compiles a thin `.mind` wrapper exposing `sha256` via `mindc --emit-shared`,
//! dlopen-s the resulting `.so`, and asserts byte-for-byte correctness against
//! three FIPS 180-4 known-answer vectors:
//!
//!   ""    → e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
//!   "abc" → ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
//!   "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" (56 bytes)
//!           → 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
//!
//! The 56-byte input forces a two-block message (input + 0x80 = 57 bytes,
//! leaving only 7 bytes before the 64-byte block boundary — insufficient for
//! the 8-byte length field, so the padding spills into a second block).
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!         --test sha256_smoke`.
//!
//! Self-skips when mlir-opt / mlir-translate / clang are absent from PATH,
//! exactly like `blas_vec_q16_smoke.rs`.

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

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Build the sha256.mind wrapper to a `.so`, exactly once per test run.
/// Returns `None` if the MLIR toolchain is absent (self-skip).
fn build_sha256_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                println!("sha256_smoke: {tool} not on PATH; skipping");
                return None;
            }
        }

        // Use std/sha256.mind directly — no wrapper needed; mindc --emit-shared
        // exports all `pub fn`s in the source file.
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let src_path = manifest_dir.join("std").join("sha256.mind");
        assert!(
            src_path.exists(),
            "std/sha256.mind not found at {src_path:?}"
        );

        let dir = std::env::temp_dir();
        let so_path = dir.join("mind_sha256_smoke.so");

        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc for sha256");
        assert!(
            status.success(),
            "mindc --emit-shared failed for std/sha256.mind"
        );
        Some(so_path)
    })
    .as_ref()
}

type Sha256Fn = unsafe extern "C" fn(i64, i64, i64) -> i64;

fn call_sha256(lib: &Library, input: &[u8], output: &mut [u8; 32]) -> i64 {
    let f: Symbol<Sha256Fn> = unsafe {
        lib.get(b"sha256\0")
            .expect("symbol 'sha256' missing from sha256 .so")
    };
    unsafe {
        f(
            input.as_ptr() as i64,
            input.len() as i64,
            output.as_mut_ptr() as i64,
        )
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// FIPS 180-4 known-answer: SHA-256("") =
/// e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
#[test]
fn sha256_empty_string() {
    let Some(so) = build_sha256_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen sha256 .so") };
    let mut digest = [0u8; 32];
    let ret = call_sha256(&lib, b"", &mut digest);
    assert_eq!(ret, 0, "sha256 must return 0");

    let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert_eq!(
        hex(&digest),
        expected,
        "SHA-256(\"\") mismatch: got {} expected {expected}",
        hex(&digest)
    );
}

/// FIPS 180-4 known-answer: SHA-256("abc") =
/// ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
#[test]
fn sha256_abc() {
    let Some(so) = build_sha256_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen sha256 .so") };
    let mut digest = [0u8; 32];
    let ret = call_sha256(&lib, b"abc", &mut digest);
    assert_eq!(ret, 0, "sha256 must return 0");

    let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        hex(&digest),
        expected,
        "SHA-256(\"abc\") mismatch: got {} expected {expected}",
        hex(&digest)
    );
}

/// FIPS 180-4 known-answer: SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
/// (56 bytes — forces two-block padding) =
/// 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
#[test]
fn sha256_two_block_padding() {
    let Some(so) = build_sha256_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen sha256 .so") };
    let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(input.len(), 56, "test vector must be exactly 56 bytes");
    let mut digest = [0u8; 32];
    let ret = call_sha256(&lib, input, &mut digest);
    assert_eq!(ret, 0, "sha256 must return 0");

    let expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    assert_eq!(
        hex(&digest),
        expected,
        "SHA-256(56-byte vector) mismatch: got {} expected {expected}",
        hex(&digest)
    );
}
