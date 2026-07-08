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

//! std.sha512 / std.sha384 execution smoke test — FIPS 180-4 known-answer
//! vectors for the 64-bit SHA-2 family.
//!
//! Compiles `std/sha512.mind` via `mindc --emit-shared`, dlopen-s the resulting
//! `.so`, and asserts byte-for-byte correctness of both `sha512` (64-byte
//! digest) and `sha384` (48-byte digest) against three FIPS 180-4 vectors each:
//!
//!   SHA-512("")   → cf83e135…927da3e
//!   SHA-512("abc")→ ddaf35a1…a54ca49f
//!   SHA-512(112B) → 8e959b75…874be909   (two-block message, forces padding spill)
//!   SHA-384("")   → 38b060a7…4898b95b
//!   SHA-384("abc")→ cb00753f…34c825a7
//!   SHA-384(112B) → 09330c33…91746039
//!
//! The 112-byte input ("abcdefgh…nopqrstu") + 0x80 = 113 bytes, leaving 15 bytes
//! before the 128-byte block boundary — insufficient for the 16-byte length
//! field, so the padding spills into a second block.
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!         --test sha512_smoke`.
//!
//! Self-skips when mlir-opt / mlir-translate / clang are absent from PATH,
//! exactly like `sha256_smoke.rs`.

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

/// The 112-byte two-block FIPS 180-4 vector shared by both digests.
const TWO_BLOCK: &[u8] = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

/// Build the sha512.mind module to a `.so`, exactly once per test run.
/// Returns `None` if the MLIR toolchain is absent (self-skip).
fn build_sha512_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                println!("sha512_smoke: {tool} not on PATH; skipping");
                return None;
            }
        }

        // std/sha512.mind exports both `pub fn sha512` and `pub fn sha384`.
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let src_path = manifest_dir.join("std").join("sha512.mind");
        assert!(
            src_path.exists(),
            "std/sha512.mind not found at {src_path:?}"
        );

        let dir = std::env::temp_dir();
        let so_path = dir.join("mind_sha512_smoke.so");

        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc for sha512");
        assert!(
            status.success(),
            "mindc --emit-shared failed for std/sha512.mind"
        );
        Some(so_path)
    })
    .as_ref()
}

type ShaFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

/// Call an exported digest fn (`sha512` or `sha384`) and return the hex digest.
fn digest(lib: &Library, sym: &[u8], input: &[u8], out_len: usize) -> String {
    let f: Symbol<ShaFn> = unsafe {
        lib.get(sym).unwrap_or_else(|_| {
            panic!(
                "symbol {:?} missing from sha512 .so",
                String::from_utf8_lossy(sym)
            )
        })
    };
    let mut out = vec![0u8; out_len];
    let ret = unsafe {
        f(
            input.as_ptr() as i64,
            input.len() as i64,
            out.as_mut_ptr() as i64,
        )
    };
    assert_eq!(ret, 0, "digest fn must return 0");
    out.iter().map(|b| format!("{b:02x}")).collect()
}

// ---- SHA-512 (64-byte digest) --------------------------------------------

#[test]
fn sha512_empty_string() {
    let Some(so) = build_sha512_so() else { return };
    let lib = unsafe { Library::new(so).expect("dlopen sha512 .so") };
    assert_eq!(
        digest(&lib, b"sha512\0", b"", 64),
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
    );
}

#[test]
fn sha512_abc() {
    let Some(so) = build_sha512_so() else { return };
    let lib = unsafe { Library::new(so).expect("dlopen sha512 .so") };
    assert_eq!(
        digest(&lib, b"sha512\0", b"abc", 64),
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
    );
}

#[test]
fn sha512_two_block() {
    let Some(so) = build_sha512_so() else { return };
    let lib = unsafe { Library::new(so).expect("dlopen sha512 .so") };
    assert_eq!(TWO_BLOCK.len(), 112, "two-block vector must be 112 bytes");
    assert_eq!(
        digest(&lib, b"sha512\0", TWO_BLOCK, 64),
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
    );
}

// ---- SHA-384 (48-byte digest, SHA-512 core + SHA-384 IV) ------------------

#[test]
fn sha384_empty_string() {
    let Some(so) = build_sha512_so() else { return };
    let lib = unsafe { Library::new(so).expect("dlopen sha512 .so") };
    assert_eq!(
        digest(&lib, b"sha384\0", b"", 48),
        "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
    );
}

#[test]
fn sha384_abc() {
    let Some(so) = build_sha512_so() else { return };
    let lib = unsafe { Library::new(so).expect("dlopen sha512 .so") };
    assert_eq!(
        digest(&lib, b"sha384\0", b"abc", 48),
        "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7",
    );
}

#[test]
fn sha384_two_block() {
    let Some(so) = build_sha512_so() else { return };
    let lib = unsafe { Library::new(so).expect("dlopen sha512 .so") };
    assert_eq!(
        digest(&lib, b"sha384\0", TWO_BLOCK, 48),
        "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712fcc7c71a557e2db966c3e9fa91746039",
    );
}
