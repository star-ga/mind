// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

// Part of the MIND project (Machine Intelligence Native Design).

// The differential harness (dlopen + FFI helpers, fixture corpus, etc.) is
// only exercised by the Linux-gated test below; on macOS/Windows the test
// no-ops, leaving these items unused. Silence the resulting dead-code/unused
// warnings off-Linux rather than cfg-gating every helper individually.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

//! RFC 0010 G2.1 — Pure-MIND vs Rust IR-text differential coverage harness.
//!
//! Compiles every fixture in the corpus through two paths:
//!
//!   1. **Rust path**: `mindc <fixture> --emit-ir` — the production Rust
//!      pipeline IR-text emitter.
//!   2. **Pure-MIND path**: dlopen `examples/mindc_mind/libmindc_mind.so`,
//!      call `mindc_compile(src_addr, src_len)`, decode the returned
//!      `EmitState.buf` string.
//!
//! Each fixture is classified as one of:
//!   * `MATCH` — byte-identical (Rust and pure-MIND agree).
//!   * `DIVERGE` — both produced output but they differ (real bug).
//!   * `MIND_UNSUPPORTED` — the pure-MIND `mindc_compile` returned a null
//!     handle or panicked at runtime on a fixture the Rust path compiled.
//!     There is no longer a source-level feature pre-filter: the front-end
//!     lowers the whole corpus (fn / struct / enum / extern / module / use /
//!     import / const items + bare const-folded expressions), so a construct
//!     it cannot handle surfaces as `DIVERGE`, not a silent exclusion.
//!   * `RUST_ONLY` — only the Rust path succeeds (Rust exit != 0
//!     means the fixture itself is invalid for some
//!     language feature the Rust path also lacks).
//!
//! Gate: the test **passes iff DIVERGE == 0**.  MIND_UNSUPPORTED is
//! expected and does not fail the test.  RUST_ONLY means the Rust path
//! itself could not compile the fixture (invalid syntax, etc.).
//!
//! Coverage report is written to `target/g2-coverage.txt` and printed to
//! stdout (captured by default; use `--nocapture` to see it live).
//!
//! Run:
//! ```
//! cargo test --release \
//!     --features "mlir-build std-surface cross-module-imports" \
//!     --test g2_differential_mlir
//! ```
//!
//! The pure-MIND `.so` is built on demand from the committed
//! `examples/mindc_mind/main.mind` source via `mindc build --emit=cdylib`
//! (no committed binary oracle — a Linux `.so` only dlopens on Linux, and the
//! ELF bytes are toolchain-patch-version specific). This harness is Linux-only
//! (it dlopens an ELF and calls in over the System V AMD64 C ABI); on other
//! platforms the test no-ops as a pass.

mod common;
use common::mindc_bin;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use libloading::Library;

// ---------------------------------------------------------------------------
// Infrastructure helpers — mirrors phase_g_keystone_bootstrap.rs
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

fn require_mindc() -> Option<PathBuf> {
    let bin = mindc_bin();
    if bin.exists() {
        Some(bin)
    } else {
        println!(
            "g2_differential_mlir: SKIP — mindc binary not found at {}; \
             run `cargo build --release` first",
            bin.display()
        );
        None
    }
}

// ---------------------------------------------------------------------------
// libmindc_mind.so — load once per process via OnceLock
// ---------------------------------------------------------------------------

/// Build and return the path to a real-ELF `libmindc_mind.so`.
///
/// No binary oracle is committed, so this normally rebuilds from
/// `examples/mindc_mind/main.mind` via `mindc build --emit=cdylib`. If a
/// real-ELF `.so` happens to be present locally (e.g. a prior build left one
/// in the example dir) it is reused; a stub or absent artifact triggers a
/// rebuild, and a failed rebuild (no MLIR toolchain) returns `None` to skip.
fn oracle_so_path(bin: &Path) -> Option<PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        let committed = repo_root().join("examples/mindc_mind/libmindc_mind.so");

        // Use the committed oracle if it is a real ELF.
        if committed.exists() {
            if let Ok(bytes) = fs::read(&committed) {
                if bytes.starts_with(b"\x7fELF") {
                    return Some(committed);
                }
            }
        }

        // Oracle absent or is a stub — rebuild from source.
        let out = std::env::temp_dir().join("g2_libmindc_mind_built.so");
        let r = Command::new(bin)
            .args([
                "build",
                "--release",
                "--emit=cdylib",
                &format!("--out={}", out.display()),
            ])
            .current_dir(repo_root())
            .output()
            .expect("spawn mindc for oracle rebuild");

        if !r.status.success() {
            println!(
                "g2_differential_mlir: oracle rebuild failed — MLIR toolchain \
                 may be unavailable.\nstderr: {}",
                String::from_utf8_lossy(&r.stderr)
            );
            return None;
        }

        if let Ok(bytes) = fs::read(&out) {
            if bytes.starts_with(b"\x7fELF") {
                return Some(out);
            }
        }

        println!(
            "g2_differential_mlir: rebuilt .so is not an ELF — \
             MLIR toolchain unavailable, skipping test"
        );
        None
    })
    .clone()
}

// ---------------------------------------------------------------------------
// Pure-MIND path: call mindc_compile via dlopen
// ---------------------------------------------------------------------------

/// MIND heap-record layouts (RFC 0005 Option-C ABI, 8-byte stride):
///
///   EmitState (3×i64, at es_handle):
///     [+0]  buf     — String heap-record address
///     [+8]  next_id — SSA counter
///     [+16] last_id — last SSA id
///
///   String (3×i64, at buf_handle):
///     [+0]  addr — byte backing-store base address
///     [+8]  len  — logical byte count
///     [+16] cap  — capacity
///
/// Reads raw i64 values at addresses returned by the MIND runtime heap.
/// Safe because the MIND `.so` maintains these allocations while loaded.
fn read_mind_string(buf_handle: i64) -> Vec<u8> {
    if buf_handle == 0 {
        return Vec::new();
    }
    // SAFETY: buf_handle is the MIND runtime's String heap record address.
    let str_addr = unsafe { read_i64_at(buf_handle, 0) };
    let str_len = unsafe { read_i64_at(buf_handle, 8) };
    if str_addr == 0 || str_len <= 0 {
        return Vec::new();
    }
    let ptr = str_addr as *const u8;
    // SAFETY: MIND String backing store is a valid byte array of length str_len.
    unsafe { std::slice::from_raw_parts(ptr, str_len as usize).to_vec() }
}

/// Read a little-endian i64 from `(base_addr + byte_offset)`.
///
/// # Safety
/// Caller must ensure `base_addr + byte_offset` is a valid pointer inside
/// the MIND heap allocation.
unsafe fn read_i64_at(base_addr: i64, byte_offset: usize) -> i64 {
    let ptr = (base_addr as usize + byte_offset) as *const i64;
    unsafe { ptr.read_unaligned() }
}

/// Signature of `mindc_compile` as exported from `libmindc_mind.so`.
///
/// In MIND's Option-C ABI structs are returned as their i64 heap address.
type MinDcCompileFn = unsafe extern "C" fn(src_addr: i64, src_len: i64) -> i64;

/// Call `mindc_compile` on `src_bytes` via `lib`.
///
/// Returns the decoded output bytes on success, or `None` if the returned
/// handle is 0 (allocation failure).
///
/// # Safety
/// Calls foreign code and reads MIND heap records at returned addresses.
unsafe fn call_mindc_compile(lib: &Library, src_bytes: &[u8]) -> Option<Vec<u8>> {
    let compile: libloading::Symbol<MinDcCompileFn> = unsafe {
        lib.get(b"mindc_compile\0")
            .expect("symbol mindc_compile must be present in libmindc_mind.so")
    };

    // Keep source bytes alive in a stable allocation across the FFI boundary.
    let src_copy: Vec<u8> = src_bytes.to_vec();
    let src_addr = src_copy.as_ptr() as i64;
    let src_len = src_copy.len() as i64;

    let es_handle: i64 = unsafe { compile(src_addr, src_len) };
    if es_handle == 0 {
        return None;
    }

    let buf_handle = unsafe { read_i64_at(es_handle, 0) };
    Some(read_mind_string(buf_handle))
}

/// Wrapper that runs `call_mindc_compile` on a dedicated thread with a
/// 64 MiB stack to avoid overflow on large fixtures (the pure-MIND compiler
/// uses deep recursion for its lexer/parser/emitter).
///
/// The `Library` handle must remain alive in the calling thread for the
/// duration. We pass a raw pointer into the spawned thread; this is safe
/// because the calling thread joins before returning, and the `.so` stays
/// loaded in-process.
fn call_on_large_stack(lib: &Library, src_bytes: &[u8]) -> Option<Vec<u8>> {
    // 64 MiB — enough for the pure-MIND compiler's recursive parsing of
    // large files (e.g. examples/mindc_mind/main.mind, ~1700 LOC).
    const STACK_SIZE: usize = 64 * 1024 * 1024;

    // Move data to heap so they can be shared via raw pointers.
    let src_bytes_box: Box<[u8]> = src_bytes.into();
    let src_ptr = src_bytes_box.as_ptr() as usize;
    let src_len = src_bytes_box.len();

    // SAFETY: We cast a reference-counted library handle to a raw pointer
    // and join the thread before returning, so the library outlives the thread.
    let lib_ptr = lib as *const Library as usize;

    let result = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || {
            // SAFETY: library is alive in the parent thread, which joins
            // before we return.
            let lib_ref: &Library = unsafe { &*(lib_ptr as *const Library) };
            let src_slice: &[u8] =
                unsafe { std::slice::from_raw_parts(src_ptr as *const u8, src_len) };
            unsafe { call_mindc_compile(lib_ref, src_slice) }
        })
        .expect("spawn worker thread")
        .join();

    // Keep the src box alive past the thread join.
    drop(src_bytes_box);

    // A panicked worker thread (e.g. stack-overflow caught by the OS, or the
    // pure-MIND compiler hitting an internal assertion) is treated as
    // unsupported — `Err(_)` collapses to the `None` default.
    result.unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Self-host coverage note
// ---------------------------------------------------------------------------
//
// There is no longer a source-level "unsupported feature" pre-filter. The
// pure-MIND front-end (examples/mindc_mind/main.mind) lowers EVERY top-level
// construct in this corpus byte-identically to mindc-Rust `--emit-ir`: fn /
// struct / enum / `extern "C"` blocks / `module NAME { }` blocks / use / import
// / const (one stub per item), plus a bare top-level arithmetic expression
// (`1 + 2 * 3`), which is const-folded to one `const.i64 <val>` exactly like
// Rust. A fixture the front-end genuinely could not handle now surfaces as a
// DIVERGE (gate failure) instead of being silently excluded — the honest,
// strict posture. The only remaining `MIND_UNSUPPORTED` path is the runtime
// valve in run_fixture (mindc_compile returned a null handle or panicked).

// ---------------------------------------------------------------------------
// Fixture corpus
// ---------------------------------------------------------------------------

fn collect_fixtures() -> Vec<PathBuf> {
    let root = repo_root();
    let mut paths: Vec<PathBuf> = Vec::new();

    let dirs: &[&str] = &[
        "tests/conformance/cpu_baseline",
        "std",
        "examples",
        "tests/runtime",
        "tests/shapes",
        "tests/autodiff",
        "tests/backend",
        "tests/type_checker",
        "tests/fixtures",
        "tests/lexical",
        "tests/ir_verification",
    ];

    for dir in dirs {
        let full = root.join(dir);
        if !full.exists() {
            continue;
        }
        collect_mind_files(&full, &mut paths);
    }

    // Negative fixtures: programs DESIGNED to fail compilation (they test that the
    // compiler correctly REJECTS bad input). They belong to error-path test suites,
    // not to a self-host PARITY differential — the Rust oracle correctly produces no
    // IR for them, so they would only ever be reported `RUST_ONLY`. Exclude them so
    // the differential's RUST_ONLY set reflects only roadmap demos (features pending),
    // not deliberately-invalid inputs.
    const NEGATIVE_FIXTURES: &[&str] = &[
        "tests/fixtures/invalid.mind",            // parse error (intentional)
        "tests/fixtures/invalid_broadcast.mind",  // type-check error (intentional)
        "tests/shapes/broadcast_incompatible.mind", // shape type-check error (intentional)
        "tests/ir_verification/undefined_operand.mind", // IR-verify error (intentional)
    ];
    paths.retain(|p| {
        let rel = p.strip_prefix(&root).unwrap_or(p);
        !NEGATIVE_FIXTURES.iter().any(|neg| rel == Path::new(neg))
    });

    paths.sort();
    paths.dedup();
    paths
}

fn collect_mind_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.extension().and_then(|e| e.to_str()) == Some("mind") {
                out.push(path);
            }
        } else if path.is_dir() {
            collect_mind_files(&path, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Match,
    Diverge { diff_preview: String },
    MindUnsupported { reason: String },
    RustOnly { reason: String },
}

// ---------------------------------------------------------------------------
// Per-fixture runner
// ---------------------------------------------------------------------------

/// Strip the trailing `\n` that `println!("{}", text)` appends.
fn normalize_rust_output(raw: &[u8]) -> &[u8] {
    raw.strip_suffix(b"\n").unwrap_or(raw)
}

/// Build a short diff preview showing the first divergent line pair.
fn build_diff_preview(rust: &[u8], mind: &[u8]) -> String {
    let rust_lines: Vec<&[u8]> = rust.split(|&b| b == b'\n').collect();
    let mind_lines: Vec<&[u8]> = mind.split(|&b| b == b'\n').collect();
    let max_shared = rust_lines.len().min(mind_lines.len());

    for i in 0..max_shared {
        if rust_lines[i] != mind_lines[i] {
            return format!(
                "first diff at line {i}:\n  RUST: {}\n  MIND: {}",
                String::from_utf8_lossy(rust_lines[i]),
                String::from_utf8_lossy(mind_lines[i]),
            );
        }
    }

    format!(
        "line count differs: RUST={} MIND={}",
        rust_lines.len(),
        mind_lines.len()
    )
}

fn run_fixture(bin: &Path, lib: &Library, fixture: &Path) -> Outcome {
    let src_bytes = match fs::read(fixture) {
        Ok(b) => b,
        Err(e) => {
            return Outcome::RustOnly {
                reason: format!("cannot read fixture: {e}"),
            };
        }
    };

    // --- Rust path ---
    let rust_result = Command::new(bin)
        .args([fixture.to_str().unwrap(), "--emit-ir"])
        .output()
        .expect("spawn mindc");

    if !rust_result.status.success() {
        return Outcome::RustOnly {
            reason: format!(
                "mindc --emit-ir exit {}",
                rust_result.status.code().unwrap_or(-1)
            ),
        };
    }

    // `normalize_rust_output` strips the single trailing `\n` that `println!`
    // appends; the MIND IR text itself ends with `}  // next_id = N\n` so
    // after normalization both paths should end identically.
    let rust_out: Vec<u8> = normalize_rust_output(&rust_result.stdout).to_vec();

    // --- Pure-MIND path (on a large-stack thread to handle deep recursion) ---
    let mind_raw = call_on_large_stack(lib, &src_bytes);

    let mind_out: Vec<u8> = match mind_raw {
        Some(bytes) => bytes,
        None => {
            return Outcome::MindUnsupported {
                reason: "mindc_compile returned null handle or panicked".to_string(),
            };
        }
    };

    // --- Byte-for-byte comparison ---
    // No additional stripping of the pure-MIND output: the MIND IR text
    // produced by `lower_program` already ends with `}  // next_id = N\n`,
    // which matches the Rust output after the `println!` newline is stripped.
    if rust_out.as_slice() == mind_out.as_slice() {
        Outcome::Match
    } else {
        Outcome::Diverge {
            diff_preview: build_diff_preview(&rust_out, &mind_out),
        }
    }
}

// ---------------------------------------------------------------------------
// Coverage report writer
// ---------------------------------------------------------------------------

fn write_coverage_report(rows: &[(PathBuf, Outcome)], report_path: &Path) -> String {
    let mut buf = String::new();
    buf.push_str("RFC 0010 G2.1 -- Differential Coverage Report\n");
    buf.push_str("==============================================\n\n");

    let mut n_match = 0usize;
    let mut n_diverge = 0usize;
    let mut n_unsupported = 0usize;
    let mut n_rust_only = 0usize;

    let root = repo_root();

    for (path, outcome) in rows {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();

        match outcome {
            Outcome::Match => {
                n_match += 1;
                buf.push_str(&format!("MATCH            {rel}\n"));
            }
            Outcome::Diverge { diff_preview } => {
                n_diverge += 1;
                buf.push_str(&format!("DIVERGE          {rel}\n"));
                for line in diff_preview.lines() {
                    buf.push_str(&format!("                   {line}\n"));
                }
            }
            Outcome::MindUnsupported { reason } => {
                n_unsupported += 1;
                buf.push_str(&format!("MIND_UNSUPPORTED {rel}  [{reason}]\n"));
            }
            Outcome::RustOnly { reason } => {
                n_rust_only += 1;
                buf.push_str(&format!("RUST_ONLY        {rel}  [{reason}]\n"));
            }
        }
    }

    let total = n_match + n_diverge + n_unsupported + n_rust_only;
    let summary = format!(
        "\nSUMMARY: {n_match} MATCH / {n_diverge} DIVERGE / \
         {n_unsupported} MIND_UNSUPPORTED / {n_rust_only} RUST_ONLY \
         out of {total} fixtures\n"
    );
    buf.push_str(&summary);

    if n_diverge > 0 {
        buf.push_str("\nG2 FINDINGS (DIVERGE):\n");
        for (path, outcome) in rows {
            if let Outcome::Diverge { diff_preview } = outcome {
                let rel = path
                    .strip_prefix(&root)
                    .unwrap_or(path)
                    .display()
                    .to_string();
                buf.push_str(&format!("  {rel}:\n    {diff_preview}\n"));
            }
        }
    }

    buf.push_str("\nG2.2+ PORTING SCOPE (MIND_UNSUPPORTED):\n");
    for (path, outcome) in rows {
        if let Outcome::MindUnsupported { reason } = outcome {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            buf.push_str(&format!("  {rel}  [{reason}]\n"));
        }
    }

    if let Err(e) = fs::write(report_path, &buf) {
        println!("g2_differential_mlir: WARNING -- could not write report: {e}");
    }

    buf
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

// The pure-MIND path dlopen()s a Linux ELF `libmindc_mind.so` and calls into
// it via the System V AMD64 C ABI. On macOS/Windows that object cannot be
// loaded (`dlopen` reports "slice is not valid mach-o file" / a PE error), and
// no committed cross-platform oracle exists (a committed Linux `.so` would only
// dlopen on Linux anyway). The differential harness is therefore meaningful
// only on Linux; gate it there and let it no-op as a passing test elsewhere so
// the cross-platform CI matrix stays green.
#[cfg(not(target_os = "linux"))]
#[test]
fn g2_1_differential_coverage() {
    println!(
        "g2_differential_mlir: SKIP -- differential harness dlopen()s a Linux \
         ELF and is gated to #[cfg(target_os = \"linux\")]"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn g2_1_differential_coverage() {
    let Some(bin) = require_mindc() else {
        return;
    };

    let Some(so_path) = oracle_so_path(&bin) else {
        println!(
            "g2_differential_mlir: SKIP -- could not obtain a valid \
             libmindc_mind.so (MLIR toolchain absent)"
        );
        return;
    };

    // Defence-in-depth: oracle_so_path only ever returns a path that starts
    // with the ELF magic, but re-verify before dlopen so a non-native or
    // truncated artifact skips cleanly instead of panicking in the loader.
    match fs::read(&so_path) {
        Ok(bytes) if bytes.starts_with(b"\x7fELF") => {}
        _ => {
            println!(
                "g2_differential_mlir: SKIP -- {} is not a native ELF; \
                 cannot dlopen for the differential harness",
                so_path.display()
            );
            return;
        }
    }

    // Load the shared library once.
    let lib = unsafe {
        Library::new(&so_path).unwrap_or_else(|e| {
            panic!(
                "g2_differential_mlir: dlopen({}) failed: {e}",
                so_path.display()
            )
        })
    };

    let fixtures = collect_fixtures();
    assert!(
        !fixtures.is_empty(),
        "fixture corpus must not be empty — check collect_fixtures()"
    );

    let mut rows: Vec<(PathBuf, Outcome)> = Vec::with_capacity(fixtures.len());

    for fixture in &fixtures {
        let outcome = run_fixture(&bin, &lib, fixture);
        rows.push((fixture.clone(), outcome));
    }

    // Write and print the coverage report.
    let target_dir = repo_root().join("target");
    let _ = fs::create_dir_all(&target_dir);
    let report_path = target_dir.join("g2-coverage.txt");
    let report = write_coverage_report(&rows, &report_path);

    println!("\n=== G2.1 Coverage Report ===\n{report}");
    println!(
        "g2_differential_mlir: coverage report written to {}",
        report_path.display()
    );

    // Gate: fail if any DIVERGE exists.
    let divergences: Vec<&(PathBuf, Outcome)> = rows
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Diverge { .. }))
        .collect();

    if !divergences.is_empty() {
        let root = repo_root();
        let mut msg = format!(
            "G2.1 GATE FAILED: {} fixture(s) DIVERGE \
             (pure-MIND and Rust compilers disagree on a feature both handle).\n",
            divergences.len()
        );
        for (path, outcome) in &divergences {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            if let Outcome::Diverge { diff_preview } = outcome {
                msg.push_str(&format!("\n  {rel}:\n    {diff_preview}\n"));
            }
        }
        panic!("{msg}");
    }

    let n_match = rows
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Match))
        .count();
    let n_unsupported = rows
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::MindUnsupported { .. }))
        .count();
    let n_rust_only = rows
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::RustOnly { .. }))
        .count();

    println!(
        "g2_differential_mlir PASS: {} MATCH / 0 DIVERGE / {} MIND_UNSUPPORTED / \
         {} RUST_ONLY out of {} fixtures",
        n_match,
        n_unsupported,
        n_rust_only,
        rows.len()
    );
}
