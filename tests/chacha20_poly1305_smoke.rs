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

//! std.chacha20_poly1305 execution smoke test — the RFC 8439 §2.8.2
//! AEAD_CHACHA20_POLY1305 known-answer test as a runnable check.
//!
//! Compiles std/chacha20_poly1305.mind via `mindc --emit-shared`, dlopen-s the
//! resulting `.so`, and asserts byte-for-byte correctness against the RFC 8439
//! §2.8.2 vector:
//!
//!   key   = 80 81 82 .. 9f (32 bytes)
//!   nonce = 07 00 00 00 40 41 42 43 44 45 46 47 (12 bytes)
//!   aad   = 50 51 52 53 c0 c1 c2 c3 c4 c5 c6 c7 (12 bytes)
//!   pt    = "Ladies and Gentlemen of the class of '99: If I could offer you
//!            only one tip for the future, sunscreen would be it." (114 bytes)
//!   tag   = 1a:e1:0b:59:4f:09:e2:6a:7e:90:2e:cb:d0:60:06:91
//!
//! It also checks the decrypt round-trip (tag verifies, plaintext recovered) and
//! that a single flipped tag byte is rejected fail-closed (return 1, pt zeroed).
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!         --test chacha20_poly1305_smoke`.
//!
//! Self-skips when mlir-opt / mlir-translate / clang are absent, exactly like
//! `sha256_smoke.rs`.

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

/// Build std/chacha20_poly1305.mind to a `.so`, once per test run.
/// Returns `None` if the MLIR toolchain is absent (self-skip).
fn build_chacha_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                println!("chacha20_poly1305_smoke: {tool} not on PATH; skipping");
                return None;
            }
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let src_path = manifest_dir.join("std").join("chacha20_poly1305.mind");
        assert!(
            src_path.exists(),
            "std/chacha20_poly1305.mind not found at {src_path:?}"
        );

        let dir = std::env::temp_dir();
        let so_path = dir.join("mind_chacha20_poly1305_smoke.so");

        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc for chacha20_poly1305");
        assert!(
            status.success(),
            "mindc --emit-shared failed for std/chacha20_poly1305.mind"
        );
        Some(so_path)
    })
    .as_ref()
}

// pub fn chacha20_poly1305_encrypt(key,nonce,pt,pt_len,aad,aad_len,ct_out,tag_out) -> i64
type SealFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64;
// pub fn chacha20_poly1305_decrypt(key,nonce,ct,ct_len,aad,aad_len,tag,pt_out) -> i64
type OpenFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// RFC 8439 §2.8.2 fixture.
fn vector() -> (Vec<u8>, [u8; 12], [u8; 12], Vec<u8>) {
    let key: Vec<u8> = (0x80u8..=0x9f).collect(); // 32 bytes
    let nonce: [u8; 12] = [
        0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    ];
    let aad: [u8; 12] = [
        0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    ];
    let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.".to_vec();
    assert_eq!(pt.len(), 114, "RFC 8439 §2.8.2 plaintext must be 114 bytes");
    (key, nonce, aad, pt)
}

const EXPECTED_CT: &str = "d31a8d34648e60db7b86afbc53ef7ec2\
a4aded51296e08fea9e2b5a736ee62d6\
3dbea45e8ca9671282fafb69da92728b\
1a71de0a9e060b2905d6a5b67ecd3b36\
92ddbd7f2d778b8c9803aee328091b58\
fab324e4fad675945585808b4831d7bc\
3ff4def08e4b7a9de576d26586cec64b\
6116";
const EXPECTED_TAG: &str = "1ae10b594f09e26a7e902ecbd0600691";

/// RFC 8439 §2.8.2 seal: ciphertext + tag are byte-identical to the vector.
#[test]
fn chacha20_poly1305_rfc8439_2_8_2_seal() {
    let Some(so) = build_chacha_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen chacha20_poly1305 .so") };
    let seal: Symbol<SealFn> = unsafe {
        lib.get(b"chacha20_poly1305_encrypt\0")
            .expect("symbol 'chacha20_poly1305_encrypt' missing")
    };

    let (key, nonce, aad, pt) = vector();
    let mut ct = vec![0u8; pt.len()];
    let mut tag = [0u8; 16];
    let ret = unsafe {
        seal(
            key.as_ptr() as i64,
            nonce.as_ptr() as i64,
            pt.as_ptr() as i64,
            pt.len() as i64,
            aad.as_ptr() as i64,
            aad.len() as i64,
            ct.as_mut_ptr() as i64,
            tag.as_mut_ptr() as i64,
        )
    };
    assert_eq!(ret, 0, "encrypt must return 0");
    assert_eq!(
        hex(&ct),
        EXPECTED_CT,
        "ciphertext mismatch:\n got {}\n exp {EXPECTED_CT}",
        hex(&ct)
    );
    assert_eq!(
        hex(&tag),
        EXPECTED_TAG,
        "tag mismatch: got {} expected {EXPECTED_TAG}",
        hex(&tag)
    );
}

/// Decrypt the RFC vector: tag verifies, plaintext is recovered exactly.
#[test]
fn chacha20_poly1305_rfc8439_2_8_2_open_roundtrip() {
    let Some(so) = build_chacha_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen chacha20_poly1305 .so") };
    let seal: Symbol<SealFn> = unsafe { lib.get(b"chacha20_poly1305_encrypt\0").unwrap() };
    let open: Symbol<OpenFn> = unsafe {
        lib.get(b"chacha20_poly1305_decrypt\0")
            .expect("symbol 'chacha20_poly1305_decrypt' missing")
    };

    let (key, nonce, aad, pt) = vector();
    let mut ct = vec![0u8; pt.len()];
    let mut tag = [0u8; 16];
    unsafe {
        seal(
            key.as_ptr() as i64,
            nonce.as_ptr() as i64,
            pt.as_ptr() as i64,
            pt.len() as i64,
            aad.as_ptr() as i64,
            aad.len() as i64,
            ct.as_mut_ptr() as i64,
            tag.as_mut_ptr() as i64,
        );
    }

    let mut back = vec![0u8; pt.len()];
    let ret = unsafe {
        open(
            key.as_ptr() as i64,
            nonce.as_ptr() as i64,
            ct.as_ptr() as i64,
            ct.len() as i64,
            aad.as_ptr() as i64,
            aad.len() as i64,
            tag.as_ptr() as i64,
            back.as_mut_ptr() as i64,
        )
    };
    assert_eq!(ret, 0, "decrypt of a valid tag must return 0");
    assert_eq!(back, pt, "recovered plaintext must equal the original");
}

/// A single flipped tag byte is rejected fail-closed: return 1, output zeroed.
#[test]
fn chacha20_poly1305_tamper_rejected_fail_closed() {
    let Some(so) = build_chacha_so() else {
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen chacha20_poly1305 .so") };
    let seal: Symbol<SealFn> = unsafe { lib.get(b"chacha20_poly1305_encrypt\0").unwrap() };
    let open: Symbol<OpenFn> = unsafe { lib.get(b"chacha20_poly1305_decrypt\0").unwrap() };

    let (key, nonce, aad, pt) = vector();
    let mut ct = vec![0u8; pt.len()];
    let mut tag = [0u8; 16];
    unsafe {
        seal(
            key.as_ptr() as i64,
            nonce.as_ptr() as i64,
            pt.as_ptr() as i64,
            pt.len() as i64,
            aad.as_ptr() as i64,
            aad.len() as i64,
            ct.as_mut_ptr() as i64,
            tag.as_mut_ptr() as i64,
        );
    }
    tag[0] ^= 0x01; // corrupt the tag

    let mut back = vec![0xAAu8; pt.len()];
    let ret = unsafe {
        open(
            key.as_ptr() as i64,
            nonce.as_ptr() as i64,
            ct.as_ptr() as i64,
            ct.len() as i64,
            aad.as_ptr() as i64,
            aad.len() as i64,
            tag.as_ptr() as i64,
            back.as_mut_ptr() as i64,
        )
    };
    assert_eq!(ret, 1, "corrupted tag must be rejected (return 1)");
    assert!(
        back.iter().all(|&b| b == 0),
        "on tag failure the output must be zeroed (no unauthenticated plaintext)"
    );
}
