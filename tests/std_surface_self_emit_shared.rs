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
// Part of the MIND project (Machine Intelligence Native Design).

//! Regression: a std *surface* module must compile to a self-contained
//! `--emit-shared` cdylib (the "self-compiling std surface" path).
//!
//! # The bug this guards against
//!
//! The `--emit-shared` link path UNCONDITIONALLY links the bundled C
//! runtime-support stub (`runtime-support/mind_intrinsics.c`) into every
//! cdylib so the `.so` is self-contained.  That stub provides C fallback
//! implementations of the pure-MIND std surface (`vec_*` / `map_*` /
//! `string_*`).  When the module being compiled IS one of those surface
//! modules (`std/string.mind`, `std/map.mind`, `std/vec.mind`), the
//! MIND-compiled functions get the EXACT same external symbol names as the
//! stub's exports, so `ld` saw each symbol defined twice:
//!
//! ```text
//! error[build]: subprocess clang failed: ...
//!   mind_intrinsics.c:(.text+0x7e0): multiple definition of `string_new'
//! ```
//!
//! The fix marks the stub's surface fallbacks `__attribute__((weak))`, so a
//! strong MIND definition overrides the weak C fallback (no collision) while
//! a consumer cdylib that does NOT define the symbol still gets the fallback
//! (self-contained property preserved).
//!
//! This test was RED before the fix (link failure for string/map/vec) and is
//! GREEN after.  `std/json.mind` is the control: its surface (`jv_*`) does not
//! overlap the stub at all, so it always compiled clean — asserting it here
//! pins that the fix does not regress the non-overlapping modules.
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!                     --test std_surface_self_emit_shared`

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::path::PathBuf;
use std::process::Command;

/// Absolute path to a bundled `std/<name>.mind` source.
fn std_src(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("std")
        .join(format!("{name}.mind"))
}

/// Compile `std/<module>.mind` to a temp `.so` via the mindc binary.
/// Returns the `.so` path on success; panics with mindc's stderr on failure
/// (the failure mode this regression guards is exactly a non-zero exit with
/// "multiple definition" in stderr).
fn emit_shared(module: &str) -> PathBuf {
    let src = std_src(module);
    assert!(src.exists(), "missing std source: {src:?}");

    let so_path = std::env::temp_dir().join(format!("std_self_emit_{module}.so"));
    let _ = std::fs::remove_file(&so_path);

    let mindc = common::mindc_bin();
    let out = Command::new(&mindc)
        .args([
            src.to_str().unwrap(),
            "--emit-shared",
            so_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn mindc");

    assert!(
        out.status.success(),
        "mindc --emit-shared FAILED for std/{module}.mind \
         (duplicate-symbol link collision regression).\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        so_path.exists() && std::fs::metadata(&so_path).map(|m| m.len() > 0).unwrap_or(false),
        ".so for std/{module}.mind was not produced or is empty"
    );
    so_path
}

/// dlopen the `.so` and call `symbol() -> i64`, asserting a nonzero handle.
/// Proves the self-contained property is preserved (no missing `__mind_*`
/// intrinsic at load time) and that the symbol resolves to a real definition.
fn assert_dlopen_symbol(so: &PathBuf, symbol: &str) {
    let so_str = so.to_string_lossy().into_owned();
    let py = format!(
        "import ctypes; \
         lib = ctypes.CDLL('{so_str}'); \
         fn = getattr(lib, '{symbol}'); \
         fn.restype = ctypes.c_int64; \
         h = fn(); \
         assert h != 0, '{symbol}() returned null'; \
         print('ok', h)"
    );
    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3 not found");
    assert!(
        out.status.success(),
        "dlopen/call of {symbol} in {so:?} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Assert the cdylib has zero undefined symbols beyond the allowed libc set —
/// i.e. it is self-contained (the weak C fallback supplies `__mind_alloc` etc.).
fn assert_self_contained(so: &PathBuf, module: &str) {
    let nm = Command::new("nm")
        .arg("-D")
        .arg(so.as_os_str())
        .output()
        .expect("nm not found on PATH");
    let text = String::from_utf8_lossy(&nm.stdout);
    // "Self-contained" here means: NO MIND-level symbol is left undefined.
    // The stub legitimately pulls in plain libc functions (malloc, calloc,
    // memcpy, bcmp, read/write, abort, …) which are resolved by the loader
    // from libc at dlopen time — those are fine. The failure mode this
    // regression guards is a MISSING runtime intrinsic (`__mind_alloc` etc.)
    // or a missing surface fn (`vec_*`/`map_*`/`string_*`), so reject exactly
    // those rather than maintaining an ever-growing libc allowlist.
    let is_mind_symbol = |bare: &str| -> bool {
        bare.starts_with("__mind_")
            || bare.starts_with("vec_")
            || bare.starts_with("map_")
            || bare.starts_with("string_")
    };
    for line in text.lines().filter(|l| l.contains(" U ")) {
        let name = line.split_whitespace().last().unwrap_or("");
        let bare = name.split('@').next().unwrap_or(name);
        assert!(
            !is_mind_symbol(bare),
            "std/{module}.mind cdylib is NOT self-contained — \
             undefined MIND symbol: {line}"
        );
    }
}

#[test]
fn string_module_emits_self_contained_shared() {
    let so = emit_shared("string");
    assert_self_contained(&so, "string");
    assert_dlopen_symbol(&so, "string_new");
}

#[test]
fn map_module_emits_self_contained_shared() {
    let so = emit_shared("map");
    assert_self_contained(&so, "map");
    assert_dlopen_symbol(&so, "map_new");
}

#[test]
fn vec_module_emits_self_contained_shared() {
    let so = emit_shared("vec");
    assert_self_contained(&so, "vec");
    assert_dlopen_symbol(&so, "vec_new");
}

/// Control: json's surface (`jv_*`) does not overlap the stub, so it never
/// collided.  Pin that the weak-symbol fix leaves it compiling clean.
#[test]
fn json_module_still_emits_shared() {
    let so = emit_shared("json");
    assert_self_contained(&so, "json");
}
