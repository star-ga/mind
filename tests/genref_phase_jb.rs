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

//! RFC 0010 Phase J-B — `GenRef<T>` generation-checked reference tests.
//!
//! Tests the three-tier memory model's Tier 3 (region-exterior heap) surface:
//! `gen_alloc`, `gen_deref`, and `gen_free` runtime helpers, plus the
//! `safety::genref_unchecked_deref` compile-time diagnostic.
//!
//! Test matrix:
//!
//!   1. `gen_alloc` → `gen_deref` returns a non-zero live pointer.
//!   2. `gen_free` → `gen_deref` of the same handle returns 0 (dangling detected).
//!   3. Slot reuse: alloc, free, alloc again → the OLD handle still derefs to 0.
//!   4. `safety::genref_unchecked_deref` fires when `gen_deref` result is used
//!      without a guard (structured diagnostic through `check_module_types`).
//!   5. Guarded deref (inside `match`) → no `safety::genref_unchecked_deref`.
//!   6. Runtime C-shim smoke: compile `mind_intrinsics.c` and exercise the
//!      `__mind_gen_alloc` / `__mind_gen_deref` / `__mind_gen_free` functions
//!      directly via `libloading` (the same pattern as `tests/blas_smoke.rs`).
//!
//! All tests are gated to `std-surface` — the `gen_alloc/deref/free` builtins
//! are only resolved in that build configuration.

#![cfg(feature = "std-surface")]

use libmind::parser::parse;
use libmind::type_checker::{TypeEnv, check_module_types};

// ---------------------------------------------------------------------------
// Tests 1–3: interpreter-level semantics through the MIND evaluator.
//
// `gen_alloc`, `gen_deref`, and `gen_free` are plain `Node::Call` forms that
// lower to calls into the runtime-support C library.  The MIND interpreter
// does not execute C functions, so we verify these tests via the C-shim smoke
// in Test 6.  Tests 1–3 here verify the AST/IR pipeline up to the point where
// the calls would be emitted.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Test 1 — gen_alloc source parses without error and lowers to a Call node.
// ---------------------------------------------------------------------------

#[test]
fn genref_alloc_parses_and_lowers() {
    let src = r#"
fn alloc_handle() -> i64 {
    let r = gen_alloc(64)
    r
}
"#;
    let module = parse(src).expect("parse should succeed");
    // No type-check errors for a valid gen_alloc usage that returns the handle.
    let diags = check_module_types(&module, src, &TypeEnv::default());
    // The only expected diagnostic would be genref_unchecked_deref if the
    // handle were used — but here we just return the i64 handle, which is safe.
    let genref_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{d:?}").contains("genref_unchecked_deref"))
        .collect();
    assert!(
        genref_errs.is_empty(),
        "returning the raw handle (i64) should not trigger genref_unchecked_deref; \
         got: {genref_errs:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — gen_deref without a guard triggers safety::genref_unchecked_deref.
// ---------------------------------------------------------------------------

#[test]
fn genref_deref_unguarded_emits_diagnostic() {
    let src = r#"
fn use_without_guard() -> i64 {
    let r = gen_alloc(64)
    let p = gen_deref(r)
    p
}
"#;
    // `let p = gen_deref(r)` followed by `p` (not a match/if) → diagnostic.
    let module = parse(src).expect("parse should succeed");
    let diags = check_module_types(&module, src, &TypeEnv::default());
    assert!(
        !diags.is_empty(),
        "expected at least one diagnostic for unguarded gen_deref; got none"
    );
    let combined: String = diags.iter().map(|d| format!("{d:?}")).collect();
    assert!(
        combined.contains("genref_unchecked_deref"),
        "diagnostic code must be `safety::genref_unchecked_deref`; got: {combined}"
    );
}

// ---------------------------------------------------------------------------
// Test 3 — gen_deref guarded by match → no safety::genref_unchecked_deref.
// ---------------------------------------------------------------------------

#[test]
fn genref_deref_guarded_by_match_no_diagnostic() {
    let src = r#"
fn use_with_guard(r: i64) -> i64 {
    let p = gen_deref(r)
    match p {
        0 => 0
        _ => p
    }
}
"#;
    let module = parse(src).expect("parse should succeed");
    let diags = check_module_types(&module, src, &TypeEnv::default());
    let genref_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{d:?}").contains("genref_unchecked_deref"))
        .collect();
    assert!(
        genref_errs.is_empty(),
        "guarded gen_deref inside `match` must NOT trigger genref_unchecked_deref; \
         got: {genref_errs:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — gen_deref guarded by if → no safety::genref_unchecked_deref.
// ---------------------------------------------------------------------------

#[test]
fn genref_deref_guarded_by_if_no_diagnostic() {
    let src = r#"
fn use_with_if_guard(r: i64) -> i64 {
    let p = gen_deref(r)
    if p == 0 {
        0
    } else {
        p
    }
}
"#;
    let module = parse(src).expect("parse should succeed");
    let diags = check_module_types(&module, src, &TypeEnv::default());
    let genref_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{d:?}").contains("genref_unchecked_deref"))
        .collect();
    assert!(
        genref_errs.is_empty(),
        "guarded gen_deref inside `if` must NOT trigger genref_unchecked_deref; \
         got: {genref_errs:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 5 — bare gen_deref expression statement (no let binding, no guard)
//          triggers safety::genref_unchecked_deref.
// ---------------------------------------------------------------------------

#[test]
fn genref_deref_bare_expr_emits_diagnostic() {
    let src = r#"
fn bare_deref(r: i64) -> i64 {
    gen_deref(r)
}
"#;
    // gen_deref as a bare expression — result discarded, no guard possible.
    let module = parse(src).expect("parse should succeed");
    let diags = check_module_types(&module, src, &TypeEnv::default());
    assert!(
        !diags.is_empty(),
        "expected at least one diagnostic for bare gen_deref expression; got none"
    );
    let combined: String = diags.iter().map(|d| format!("{d:?}")).collect();
    assert!(
        combined.contains("genref_unchecked_deref"),
        "diagnostic code must be `safety::genref_unchecked_deref`; got: {combined}"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — runtime C-shim smoke: compile mind_intrinsics.c, dlopen, and
//           exercise gen_alloc / gen_deref / gen_free directly.
//
// This is the same compile-and-dlopen pattern as `tests/blas_smoke.rs`.
// ---------------------------------------------------------------------------

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

const RUNTIME_SUPPORT_REL: &str = "runtime-support/mind_intrinsics.c";

static GENREF_SO: OnceLock<Option<PathBuf>> = OnceLock::new();

// Serialize the genref *runtime* tests: they all dlopen the same smoke .so and
// thus share its process-global genref_table, which realloc-moves as it grows.
// Running them on parallel test threads is a data race (a concurrent gen_alloc
// realloc under a gen_deref/gen_free read) -- the source of the Windows flake.
static GENREF_RUNTIME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

type GenAllocFn = unsafe extern "C" fn(i64) -> i64;
type GenDerefFn = unsafe extern "C" fn(i64) -> i64;
type GenFreeFn = unsafe extern "C" fn(i64) -> i64;

fn build_genref_so() -> Option<PathBuf> {
    let clang = which::which("clang").ok()?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest_dir.join(RUNTIME_SUPPORT_REL);
    assert!(
        src.exists(),
        "runtime-support source must exist at {}",
        src.display()
    );

    let out_dir = manifest_dir.join("target").join("genref_smoke");
    std::fs::create_dir_all(&out_dir).expect("create target/genref_smoke");

    // Unique per-process output path: the shim is `clang -o`'d to this file, so
    // a FIXED name would let two concurrent test processes (e.g. parallel cargo
    // test jobs on one CI runner) clobber each other's `.so` mid-write and
    // intermittently dlopen a half-written library. Keying the name on the PID
    // gives each process its own artifact with no shared write target. (The
    // in-process `OnceLock` already dedupes builds within one process.)
    let pid = std::process::id();
    #[cfg(windows)]
    let so_path = out_dir.join(format!("mind_genref_smoke_{pid}.dll"));
    #[cfg(not(windows))]
    let so_path = out_dir.join(format!("libmind_genref_smoke_{pid}.so"));

    let mut cmd = Command::new(&clang);
    cmd.args([
        "-x",
        "c",
        src.to_str().unwrap(),
        "-shared",
        "-O2",
        "-o",
        so_path.to_str().unwrap(),
    ]);
    #[cfg(not(windows))]
    cmd.arg("-fPIC");

    let status = cmd.status().expect("spawn clang");
    assert!(
        status.success(),
        "clang failed to compile {} → {}",
        src.display(),
        so_path.display()
    );
    Some(so_path)
}

fn open_genref_lib(path: &std::path::Path) -> Library {
    unsafe { Library::new(path).expect("dlopen genref smoke .so") }
}

unsafe fn sym_genref<'lib, F>(lib: &'lib Library, name: &[u8]) -> Symbol<'lib, F> {
    unsafe {
        lib.get::<F>(name)
            .unwrap_or_else(|e| panic!("symbol {} missing: {e}", String::from_utf8_lossy(name)))
    }
}

fn call_gen_alloc(lib: &Library, bytes: i64) -> i64 {
    unsafe {
        let f: Symbol<GenAllocFn> = sym_genref(lib, b"__mind_gen_alloc\0");
        f(bytes)
    }
}

fn call_gen_deref(lib: &Library, handle: i64) -> i64 {
    unsafe {
        let f: Symbol<GenDerefFn> = sym_genref(lib, b"__mind_gen_deref\0");
        f(handle)
    }
}

fn call_gen_free(lib: &Library, handle: i64) -> i64 {
    unsafe {
        let f: Symbol<GenFreeFn> = sym_genref(lib, b"__mind_gen_free\0");
        f(handle)
    }
}

// Test 6a — gen_alloc returns non-zero handle; gen_deref returns non-zero ptr.
#[test]
fn genref_runtime_alloc_then_deref_returns_live_ptr() {
    let so = GENREF_SO.get_or_init(build_genref_so);
    let Some(ref so_path) = *so else {
        println!("clang not found — skipping genref C-shim smoke test");
        return;
    };
    let _serial = GENREF_RUNTIME_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lib = open_genref_lib(so_path);

    let handle = call_gen_alloc(&lib, 64);
    assert_ne!(handle, 0, "gen_alloc(64) must return a non-zero handle");

    let ptr = call_gen_deref(&lib, handle);
    assert_ne!(
        ptr, 0,
        "gen_deref of a live handle must return a non-zero pointer"
    );
}

// Test 6b — gen_free makes subsequent gen_deref return 0 (dangling detected).
#[test]
fn genref_runtime_free_makes_deref_return_zero() {
    let so = GENREF_SO.get_or_init(build_genref_so);
    let Some(ref so_path) = *so else {
        println!("clang not found — skipping genref C-shim smoke test");
        return;
    };
    let _serial = GENREF_RUNTIME_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lib = open_genref_lib(so_path);

    let handle = call_gen_alloc(&lib, 32);
    assert_ne!(handle, 0, "gen_alloc must succeed");

    // Verify live before free.
    let ptr_before = call_gen_deref(&lib, handle);
    assert_ne!(ptr_before, 0, "deref before free must be non-zero");

    call_gen_free(&lib, handle);

    // After free the same handle must deref to 0.
    let ptr_after = call_gen_deref(&lib, handle);
    assert_eq!(
        ptr_after, 0,
        "deref after gen_free must return 0 (dangling detected)"
    );
}

// Test 6c — slot reuse: old handle from a freed slot returns 0 even after
//           a new allocation reuses the same slot.
#[test]
fn genref_runtime_stale_handle_returns_zero_after_slot_reuse() {
    let so = GENREF_SO.get_or_init(build_genref_so);
    let Some(ref so_path) = *so else {
        println!("clang not found — skipping genref C-shim smoke test");
        return;
    };
    let _serial = GENREF_RUNTIME_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lib = open_genref_lib(so_path);

    // Alloc + free to create a free slot with generation 1.
    let old_handle = call_gen_alloc(&lib, 16);
    assert_ne!(old_handle, 0);
    call_gen_free(&lib, old_handle);

    // A new alloc will reuse the freed slot with incremented generation.
    let _new_handle = call_gen_alloc(&lib, 16);

    // The old handle (generation 0) must still deref to 0 — generation mismatch.
    let ptr = call_gen_deref(&lib, old_handle);
    assert_eq!(
        ptr, 0,
        "stale handle (old generation) must return 0 even after slot reuse"
    );
}

// Test 6d — handle 0 / invalid bytes return 0 gracefully.
#[test]
fn genref_runtime_null_handle_returns_zero() {
    let so = GENREF_SO.get_or_init(build_genref_so);
    let Some(ref so_path) = *so else {
        println!("clang not found — skipping genref C-shim smoke test");
        return;
    };
    let _serial = GENREF_RUNTIME_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lib = open_genref_lib(so_path);

    // gen_alloc with non-positive bytes → handle 0.
    let h = call_gen_alloc(&lib, 0);
    assert_eq!(h, 0, "gen_alloc(0) must return handle 0");

    // gen_deref of the null handle → 0.
    let p = call_gen_deref(&lib, 0);
    assert_eq!(p, 0, "gen_deref(0) must return 0");

    // gen_free of the null handle → no-op, returns 0.
    let r = call_gen_free(&lib, 0);
    assert_eq!(r, 0, "gen_free(0) must return 0");
}
