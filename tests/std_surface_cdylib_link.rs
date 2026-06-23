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

//! RFC 0005 Phase 6.5 Stage 1b — cdylib self-contained linkage test.
//!
//! Verifies that `--emit-shared` produces a `.so` that `dlopen`s cleanly
//! without undefined MIND symbols.  The test compiles a tiny .mind file
//! that uses `std.vec`, emits the cdylib, then asserts:
//!
//! 1. The `.so` file is produced (nonzero size).
//! 2. `nm -D` reports zero undefined symbols beyond the allowed libc set
//!    (`malloc`, `free`, `memcpy`, `realloc` from the runtime-support stub).
//! 3. The `.so` opens via `dlopen(RTLD_LAZY)` without error.
//!
//! Gated: `cargo test --features "mlir-build std-surface cross-module-imports"
//!                     --test std_surface_cdylib_link`

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::path::PathBuf;
use std::process::Command;

// Minimal .mind source that uses std.vec so the cdylib must bundle vec_new /
// vec_push / __mind_load_i64 / __mind_alloc / etc. from the runtime-support
// stub to be self-contained.
const SRC: &str = r#"
use std.vec;

pub fn make_vec() -> Vec {
    vec_new()
}

pub fn make_and_push(v: Vec, x: i64) -> Vec {
    vec_push(v, x)
}
"#;

/// Compile SRC to a temporary .so via the mindc binary and return its path.
///
/// The test binary locates mindc at `target/debug/mindc` relative to the
/// manifest dir; `cargo test` ensures the binary is built before tests run.
fn build_test_so() -> PathBuf {
    let dir = std::env::temp_dir();
    let src_path = dir.join("mind_cdylib_link_test.mind");
    let so_path = dir.join("mind_cdylib_link_test.so");

    std::fs::write(&src_path, SRC).expect("write test .mind source");

    // Binary resolved via tests/common::mindc_bin() (CARGO_BIN_EXE_mindc).
    let mindc = common::mindc_bin();

    let status = Command::new(&mindc)
        .args([
            src_path.to_str().unwrap(),
            "--emit-shared",
            so_path.to_str().unwrap(),
        ])
        .status()
        .expect("spawn mindc");

    assert!(
        status.success(),
        "mindc --emit-shared failed for test source"
    );

    so_path
}

#[test]
fn cdylib_is_produced_and_nonzero() {
    let so = build_test_so();
    let meta = std::fs::metadata(&so).expect("stat .so");
    assert!(meta.len() > 0, ".so file is empty");
}

#[test]
fn cdylib_has_no_undefined_mind_symbols() {
    let so = build_test_so();

    let nm_out = Command::new("nm")
        .arg("-D")
        .arg(so.as_os_str())
        .output()
        .expect("nm not found on PATH");

    let text = String::from_utf8_lossy(&nm_out.stdout);

    let undefined: Vec<&str> = text.lines().filter(|l| l.contains(" U ")).collect();

    // The only undefined symbols allowed are libc symbols that the
    // runtime-support stub itself depends on.  The POSIX I/O calls
    // are pulled in by std.io's __mind_read / __mind_write / pread /
    // pwrite passthrough (see runtime-support/mind_intrinsics.c).
    // `abort` is referenced by the RFC 0010 GenRef region allocator for
    // deterministic OOM / generation-wrap / region-nesting panics.
    let allowed = [
        "malloc", "free", "memcpy", "realloc", "read", "write", "pread", "pwrite", "abort",
    ];

    for sym_line in &undefined {
        let name = sym_line.split_whitespace().last().unwrap_or("");
        let bare = name.split('@').next().unwrap_or(name);
        assert!(
            allowed.contains(&bare),
            "unexpected undefined symbol in cdylib: {sym_line}\n\
             Full undefined list: {undefined:#?}"
        );
    }
}

/// #306 regression: the runtime-support shim must DEFINE the MLIR
/// executable-print helpers `printI64` / `printNewline`.
///
/// `mindc` lowers scalar `print(x)` outputs to `func.call @printI64(%x)` +
/// `func.call @printNewline()` and emits both as `func.func private`
/// (external) declarations (see `src/eval/mlir_export.rs`).  If the shim does
/// not provide the definitions, the native cdylib link fails with an
/// undefined reference to `printI64` and `mindc build --emit=cdylib` of any
/// program that prints (notably the self-host keystone) silently falls back
/// to a launcher-script stub instead of a real ELF.  This compiles the shim
/// in isolation and asserts both symbols are present as defined text symbols.
#[test]
fn runtime_support_defines_print_helpers() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shim = manifest_dir
        .join("runtime-support")
        .join("mind_intrinsics.c");
    assert!(shim.exists(), "runtime-support shim not found at {shim:?}");

    let obj = std::env::temp_dir().join("mind_shim_print_helpers.o");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .args(["-c", "-O2", "-fPIC"])
        .arg(shim.as_os_str())
        .arg("-o")
        .arg(obj.as_os_str())
        .status()
        .expect("C compiler not found on PATH");
    assert!(status.success(), "failed to compile runtime-support shim");

    let nm_out = Command::new("nm")
        .arg(obj.as_os_str())
        .output()
        .expect("nm not found on PATH");
    let text = String::from_utf8_lossy(&nm_out.stdout);

    for sym in ["printI64", "printNewline"] {
        let defined = text
            .lines()
            .any(|l| l.split_whitespace().last() == Some(sym) && l.contains(" T "));
        assert!(
            defined,
            "runtime-support shim does not DEFINE `{sym}` (T) — \
             executable-print cdylib link would fall back to a stub (#306).\n\
             nm output:\n{text}"
        );
    }
}

#[test]
fn cdylib_dlopens_via_python() {
    let so = build_test_so();

    // Use the system Python3 to dlopen the .so.  This is the same
    // mechanism the bootstrap_smoke.py harness uses and verifies
    // the complete RTLD_NOW load path without any Rust-side dependency
    // on the mlir-jit / libloading features.
    let so_str = so.to_string_lossy().into_owned();
    let py_script = format!(
        "import ctypes; \
         lib = ctypes.CDLL('{so_str}'); \
         fn = lib.vec_new; \
         fn.restype = ctypes.c_int64; \
         handle = fn(); \
         assert handle != 0, 'vec_new() returned null'; \
         print('ok', handle)"
    );

    let out = Command::new("python3")
        .args(["-c", &py_script])
        .output()
        .expect("python3 not found");

    assert!(
        out.status.success(),
        "python3 dlopen/call failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
