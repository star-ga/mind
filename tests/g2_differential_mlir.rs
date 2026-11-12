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
//!   * `MIND_UNSUPPORTED` — the pure-MIND `mindc_compile` explicitly refused
//!     the input (next_id == -1, empty output) that the Rust path compiled.
//!     There is no longer a source-level feature pre-filter: the front-end
//!     lowers the whole corpus (fn / struct / enum / extern / module / use /
//!     import / const items + bare const-folded expressions), so a construct
//!     it cannot handle without an explicit refusal surfaces as `DIVERGE`.
//!   * `REJECT_MATCH` — both paths reject a bimap or NERVE policy violation.
//!   * `RUST_ONLY` — Rust rejected another input; the self-host path was not
//!     compared. This bucket is not evidence of self-host correctness.
//!
//!   * `MIND_CRASH` — the pure-MIND compiler terminated ABNORMALLY on a
//!     fixture (panic / internal assertion / worker death). A DEFECT, never a
//!     porting-backlog entry.
//!
//! Gate: the test **passes iff DIVERGE == 0 AND MIND_CRASH == 0**.
//! MIND_UNSUPPORTED is expected and does not fail the test — declining a
//! construct is legitimate. CRASHING on one is not, and the two used to share
//! a bucket: a null handle and a dead worker both collapsed to `None`, so the
//! pure-MIND compiler segfaulted 21 times over four days inside runs that
//! reported pass. RUST_ONLY means the Rust path itself could not compile the
//! fixture (invalid syntax, etc.).
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

#[path = "support/g2_capability_ratchet.rs"]
mod capability_ratchet;
mod common;
#[path = "support/g2_compile_result.rs"]
mod compile_result;
use compile_result::{MindCall, classify_compiler_output};
#[path = "support/g2_policy_rejections.rs"]
mod policy_rejections;
use common::require_mindc;

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

// require_mindc() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Emit a machine-readable `SDLC-GATE ... ran=N fail=K` marker that SURVIVES
/// libtest's stdout capture.
///
/// `println!` goes through `std::io::_print`, whose sink libtest swaps per test:
/// a PASSING test's output is buffered and thrown away unless the run asks for
/// `--show-output`/`--nocapture`. A skip that returns `Ok` is exactly the case
/// this marker exists for, so a `println!`-only marker can never appear in the
/// tier log — documentation, not evidence. Writing to the process stdout handle
/// bypasses the capture shim (measured: the direct write lands in the log, the
/// `println!` next to it does not).
///
/// Turning on `--show-output` for the whole tier would work too, and is worse:
/// it splices every passing test's captured stdout into the log that
/// scripts/exec_semantics_gate.sh counts `^test result:` lines in, and some of
/// this repo's tests print a subprocess `mindc test` summary that starts with
/// exactly that prefix — inflating the very floors the gate defends.
fn emit_gate_marker(line: &str) {
    use std::io::Write;
    let mut out = std::io::stdout();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
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
/// True when `so` is older than any input that determines its bytes, or when that
/// cannot be established. Covers the compiler files of the Python resolver's stamp
/// (`examples/mindc_mind/_selfhost_so.py`): every top-level compiler-dir `.mind`.
///
/// Fails STALE on any unreadable mtime: an oracle we cannot prove current is exactly
/// the one that must not be trusted.
fn is_stale(so: &Path, compiler: &Path) -> bool {
    let Ok(so_mtime) = fs::metadata(so).and_then(|m| m.modified()) else {
        return true;
    };
    let root = repo_root();
    // The COMPILER is the caller's ACTUAL binary, not a hardcoded
    // `target/release/mindc`. Hardcoding it forced an exception -- "an unreadable
    // mindc is normal, so do not count it" -- which made the stated rule ("fails
    // stale on any input it cannot read") FALSE for one of the four inputs, and left
    // a real hole: if the compiler that produced the artifact cannot be compared
    // against it, staleness is undecidable and the artifact must not be trusted.
    // Using the path the caller already holds removes the exception rather than
    // documenting it.
    // main.mind, the driver and every part: each top-level compiler-dir .mind, plus Mind.toml.
    let entries: std::io::Result<Vec<fs::DirEntry>> =
        fs::read_dir(root.join("examples/mindc_mind")).and_then(Iterator::collect);
    let Ok(entries) = entries else {
        return true; // Unreadable input -> undecidable -> stale. No exceptions.
    };
    let mind = entries.iter().map(fs::DirEntry::path);
    let inputs = mind.filter(|p| p.extension().is_some_and(|x| x == "mind"));
    let mut inputs = inputs.chain([root.join("Mind.toml")]);
    let src_stale = inputs.any(|p| match fs::metadata(&p).and_then(|m| m.modified()) {
        Ok(input_mtime) => input_mtime > so_mtime,
        Err(_) => true,
    });
    let compiler_stale = match fs::metadata(compiler).and_then(|m| m.modified()) {
        Ok(bin_mtime) => bin_mtime > so_mtime,
        Err(_) => true,
    };
    src_stale || compiler_stale
}

fn oracle_so_path(bin: &Path) -> Option<PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        let committed = repo_root().join("examples/mindc_mind/libmindc_mind.so");

        // Use the in-tree oracle only if it is a real ELF **and is not STALE**.
        //
        // This file is an UNTRACKED build artifact, not a committed one, so its age
        // is whatever the last local build left behind — on CI it is absent and gets
        // rebuilt, on a dev box it can be arbitrarily old. Accepting it on existence
        // alone made this gate compare the Rust oracle against a pure-MIND compiler
        // built from source that no longer exists.
        //
        // Measured: a 35-hour-old artifact reported 32 fixtures as SIGSEGV crashes
        // that the then-current source did not crash on at all. The same hole runs
        // the other way and is far worse — a bug newly introduced into main.mind is
        // INVISIBLE here, because the gate never loads the code under test. A gate
        // that cannot see the change it exists to gate is not a gate.
        if committed.exists() && !is_stale(&committed, bin) {
            if let Ok(bytes) = fs::read(&committed) {
                if bytes.starts_with(b"\x7fELF") {
                    return Some(committed);
                }
            }
        }
        // Oracle absent or a stub: rebuild from source (the MLIR self-host build).
        let out =
            std::env::temp_dir().join(format!("g2_libmindc_mind_built_{}.so", std::process::id()));
        let r = Command::new(bin)
            .args([
                "build",
                "--backend=mlir",
                "--release",
                "--emit=cdylib",
                &format!("--out={}", out.display()),
            ])
            .current_dir(repo_root())
            .output()
            .expect("spawn mindc for oracle rebuild");

        // A rebuild failure used to print and return None, so the whole
        // differential comparison was skipped and the gate PASSED on a real
        // oracle-build regression. Only a genuine capability gap may skip now.
        if !common::gate::compiled("g2_differential_mlir", &r) {
            return None;
        }

        if let Ok(bytes) = fs::read(&out) {
            if bytes.starts_with(b"\x7fELF") {
                return Some(out);
            }
        }

        common::gate::skipped(
            "g2_differential_mlir",
            "rebuilt oracle is not an ELF (MLIR toolchain unavailable)",
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
/// A null handle is a runtime failure. Only the compiler's explicit refusal
/// sentinel with empty output is an unsupported construct.
///
/// # Safety
/// Calls foreign code and reads MIND heap records at returned addresses.
unsafe fn call_mindc_compile(lib: &Library, src_bytes: &[u8]) -> MindCall {
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
        return MindCall::Crashed("null compiler result handle".to_string());
    }

    let buf_handle = unsafe { read_i64_at(es_handle, 0) };
    if buf_handle == 0 {
        return MindCall::Crashed("null compiler output buffer".to_string());
    }
    let next_id = unsafe { read_i64_at(es_handle, 8) };
    classify_compiler_output(next_id, read_mind_string(buf_handle))
}

/// Re-exec THIS test binary as a single-fixture worker so a hard signal is
/// observable instead of fatal.
///
/// Why this exists: a SIGSEGV does not unwind. Measured directly — a thread that
/// reads address 8 kills the whole process, `join()` never returns, exit 139. So
/// the in-process `MindCall::Crashed` arm can only ever see a PANIC. A genuine
/// segfault in the pure-MIND compiler would take the harness down with it, and the
/// gate would never get to classify anything.
///
/// Running the call in a child process turns that fatal signal into an exit status
/// the parent can read and attribute. The child does the dlopen itself (a `Library`
/// handle cannot cross a process boundary), writes the compiled bytes to stdout, and
/// signals its verdict through the exit code:
///
///   0   -> compiled; stdout holds the bytes
///   3   -> `mindc_compile` returned an explicit refusal record
///   4   -> the WORKER ITSELF could not get as far as asking the compiler
///          (staged fixture unreadable, `.so` path not passed, dlopen failed,
///          stdout unwritable). Distinct from 3 on purpose: this is a harness
///          fault, and folding it into the null-handle bucket would report a
///          fixture the gate never compiled as "construct not lowered yet" —
///          the exact declined-vs-died collapse this file was fixed for.
///   <0  -> killed by a signal: SIGSEGV / SIGBUS / SIGILL / SIGABRT — a CRASH
///
/// deferred: one process per fixture costs a fork+dlopen per call. If that becomes
/// the harness's bottleneck, upgrade path is a persistent worker fed fixtures over a
/// pipe, respawned on death — same classification, one dlopen.
const WORKER_ENV: &str = "G2_MIND_WORKER_SRC";
/// The parent passes the ALREADY-RESOLVED .so. Re-deriving it in the child could
/// select a different library than the one the parent is testing, which would make
/// every verdict describe the wrong artifact.
const WORKER_SO_ENV: &str = "G2_MIND_WORKER_SO";
/// Where the child writes the compiled bytes.
///
/// NOT stdout: the child re-execs this same TEST binary, so the worker body runs
/// inside a libtest test, and libtest installs an output capture that swallows
/// writes through `std::io::stdout()`. The compiled bytes never reached the parent
/// and every fixture compared as EMPTY — 0 MATCH / 99 DIVERGE, a differential gate
/// that silently compared nothing at all. A plain file has no such interception.
const WORKER_OUT_ENV: &str = "G2_MIND_WORKER_OUT";
const WORKER_EXIT_REFUSED: i32 = 3;
/// The worker could not reach the compiler at all. See the exit-code table above:
/// this must NEVER be `WORKER_EXIT_REFUSED`, because a refusal means the
/// pure-MIND front end looked at the fixture and declined it, and that is a
/// legitimate, gate-passing outcome. A harness fault is not.
const WORKER_EXIT_HARNESS_FAULT: i32 = 4;

/// Bail out of the child with the harness-fault code, naming the fault on stderr so
/// the parent can quote it instead of reporting a bare exit number.
fn worker_fault(reason: &str) -> ! {
    eprintln!("{reason}");
    std::process::exit(WORKER_EXIT_HARNESS_FAULT)
}

/// Runs in the CHILD when `WORKER_ENV` is set. Never returns there; in the
/// PARENT it does nothing. Not `let Ok(..) else { return; }`: a bare early-out
/// on an env read is the shape `fail_open_skip_site_ratchet.rs` forbids.
fn run_as_worker_if_requested() {
    if let Ok(src_path) = std::env::var(WORKER_ENV) {
        // `unwrap_or_default()` here handed the compiler an EMPTY source whenever the
        // staged fixture could not be read, and an empty program is exactly the kind of
        // input the front end declines — so an I/O fault arrived at the parent wearing a
        // null handle and was filed as MIND_UNSUPPORTED. Same collapse as the one this
        // file was fixed for, one layer down.
        let src = match std::fs::read(&src_path) {
            Ok(b) => b,
            Err(e) => worker_fault(&format!("cannot read staged fixture {src_path}: {e}")),
        };
        let Ok(so_path) = std::env::var(WORKER_SO_ENV) else {
            worker_fault("worker was not given the resolved .so path")
        };
        let lib = match unsafe { Library::new(&so_path) } {
            Ok(l) => l,
            // A `.so` that will not dlopen means EVERY fixture is uncompiled. Reported as a
            // null handle that read as "the whole corpus is unsupported yet" — only the
            // MATCH floor stood between that and a green gate, and it named the wrong cause.
            Err(e) => worker_fault(&format!("cannot dlopen {so_path}: {e}")),
        };
        // 64 MiB, the same stack the in-process path used. The pure-MIND compiler
        // parses recursively, so a large fixture can outgrow the default ~8 MiB main
        // stack. Moving the call into a subprocess made the signal OBSERVABLE; it must
        // not also make it more LIKELY. A harness-induced stack overflow arrives as a
        // SIGSEGV and is indistinguishable, at the parent, from a real compiler crash —
        // so dropping this would manufacture the exact fault the gate now reports.
        const WORKER_STACK_SIZE: usize = 64 * 1024 * 1024;

        // A panic inside the compile is NOT caught here: it unwinds, the child exits
        // non-zero, and the parent files it as Crashed with the message quoted.
        let compiled = std::thread::scope(|scope| {
            let handle = std::thread::Builder::new()
                .stack_size(WORKER_STACK_SIZE)
                // SAFETY: same contract as before — `lib` and `src` outlive the scope,
                // which joins the thread before returning.
                .spawn_scoped(scope, || unsafe { call_mindc_compile(&lib, &src) })
                .unwrap_or_else(|e| worker_fault(&format!("cannot spawn compile thread: {e}")));
            match handle.join() {
                Ok(v) => v,
                Err(payload) => std::panic::resume_unwind(payload),
            }
        });

        match compiled {
            MindCall::Ok(bytes) => {
                let Ok(out_path) = std::env::var(WORKER_OUT_ENV) else {
                    worker_fault("worker was not given an output path")
                };
                // A dropped or short write silently truncates the compiled bytes, and the
                // parent then byte-compares a truncated stream against the Rust oracle and
                // reports DIVERGE — a harness fault indicted as a compiler defect. The
                // mirror image of the collapse above, and just as wrong.
                if let Err(e) = std::fs::write(&out_path, &bytes) {
                    worker_fault(&format!("cannot write compiled bytes to {out_path}: {e}"));
                }
                std::process::exit(0);
            }
            MindCall::Refused => std::process::exit(WORKER_EXIT_REFUSED),
            MindCall::Crashed(reason) => panic!("compiler result failure: {reason}"),
        }
    }
}

/// Parent side: run one fixture in a child and classify how it ended.
fn call_in_subprocess(src_bytes: &[u8], so_path: &Path) -> MindCall {
    use std::os::unix::process::ExitStatusExt as _;

    let Ok(exe) = std::env::current_exe() else {
        return MindCall::Crashed("cannot locate test binary to re-exec".to_string());
    };
    let tmp = std::env::temp_dir().join(format!("g2_worker_src_{}.mind", std::process::id()));
    if std::fs::write(&tmp, src_bytes).is_err() {
        return MindCall::Crashed("cannot stage fixture for worker".to_string());
    }

    let out_file = std::env::temp_dir().join(format!("g2_worker_out_{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&out_file);
    let out = std::process::Command::new(exe)
        .env(WORKER_ENV, &tmp)
        .env(WORKER_SO_ENV, so_path)
        .env(WORKER_OUT_ENV, &out_file)
        .stdin(std::process::Stdio::null())
        .output();
    let _ = std::fs::remove_file(&tmp);

    let out = match out {
        Ok(o) => o,
        Err(e) => return MindCall::Crashed(format!("failed to spawn worker: {e}")),
    };

    // A signal means the compiler DIED on this fixture. That is the case the
    // in-process path structurally cannot observe, and the reason this exists.
    if let Some(sig) = out.status.signal() {
        let name = match sig {
            11 => "SIGSEGV",
            7 => "SIGBUS",
            4 => "SIGILL",
            6 => "SIGABRT",
            _ => "signal",
        };
        return MindCall::Crashed(format!(
            "pure-MIND compiler killed by {name} ({sig}) — this is a crash, not an \
             unsupported construct"
        ));
    }
    // The child writes its fault to stderr; quote it so a gate failure names the
    // actual cause rather than a bare exit number.
    let child_err = String::from_utf8_lossy(&out.stderr).trim().to_string();
    match out.status.code() {
        Some(0) => match std::fs::read(&out_file) {
            Ok(bytes) => {
                let _ = std::fs::remove_file(&out_file);
                MindCall::Ok(bytes)
            }
            // Exit 0 promises a result file. Its absence is a harness fault, never
            // an empty compilation — the distinction this whole file exists to keep.
            Err(e) => MindCall::Crashed(format!(
                "worker exited 0 but left no result at {}: {e}",
                out_file.display()
            )),
        },
        Some(c) if c == WORKER_EXIT_REFUSED => MindCall::Refused,
        Some(c) if c == WORKER_EXIT_HARNESS_FAULT => MindCall::Crashed(format!(
            "HARNESS FAULT (the compiler was never asked about this fixture): {child_err}"
        )),
        Some(c) => MindCall::Crashed(format!(
            "worker exited {c} without a verdict{}",
            if child_err.is_empty() {
                String::new()
            } else {
                format!(": {child_err}")
            }
        )),
        None => MindCall::Crashed("worker ended with neither code nor signal".to_string()),
    }
}

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

    // These four legacy error fixtures remain covered by their dedicated suites.
    // Curated selfhost_policy/reject fixtures are included: both compilers must
    // refuse them, and the Rust diagnostic must match the pinned code. Other
    // Rust-only inputs remain unpaired and are not evidence of self-host parity.
    const NEGATIVE_FIXTURES: &[&str] = &[
        "tests/fixtures/invalid.mind",              // parse error (intentional)
        "tests/fixtures/invalid_broadcast.mind",    // type-check error (intentional)
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
    RejectMatch,
    Diverge {
        diff_preview: String,
    },
    MindUnsupported {
        reason: String,
    },
    /// The pure-MIND compiler terminated abnormally on a fixture. This is a DEFECT,
    /// never a porting-backlog entry, and it FAILS the gate. Kept distinct from
    /// MindUnsupported because collapsing the two is what let 21 real crashes ride
    /// inside a green test run over four days.
    MindCrash {
        reason: String,
    },
    RustOnly {
        reason: String,
    },
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

    if let Some(outcome) =
        policy_rejections::check_rejection(&rust_result, fixture, bin, &src_bytes)
    {
        return outcome;
    }
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
    // Subprocess, not a thread: a hard signal must be observable rather than fatal
    // to the harness (a SIGSEGV does not unwind — measured, exit 139, join never
    // returns). `lib` stays loaded in-process for the Rust-side path.
    let _ = lib;
    let Some(so_for_worker) = oracle_so_path(bin) else {
        return Outcome::MindCrash {
            reason: "resolved pure-MIND oracle became unavailable".to_string(),
        };
    };
    let mind_raw = call_in_subprocess(&src_bytes, &so_for_worker);

    let mind_out: Vec<u8> = match mind_raw {
        MindCall::Ok(bytes) => bytes,
        // Declining a construct is legitimate and stays non-fatal.
        MindCall::Refused => {
            return Outcome::MindUnsupported {
                reason: "mindc_compile explicitly refused the input (next_id=-1, empty output)"
                    .to_string(),
            };
        }
        // Dying on a construct is not the same event and must not share its bucket.
        MindCall::Crashed(reason) => {
            return Outcome::MindCrash { reason };
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
    let mut n_reject_match = 0usize;
    let mut n_diverge = 0usize;
    let mut n_crash = 0usize;
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
            Outcome::RejectMatch => {
                n_reject_match += 1;
                buf.push_str(&format!("REJECT_MATCH     {rel}\n"));
            }
            Outcome::Diverge { diff_preview } => {
                n_diverge += 1;
                buf.push_str(&format!("DIVERGE          {rel}\n"));
                for line in diff_preview.lines() {
                    buf.push_str(&format!("                   {line}\n"));
                }
            }
            Outcome::MindCrash { reason } => {
                n_crash += 1;
                buf.push_str(&format!("MIND_CRASH {rel}  [{reason}]\n"));
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

    let total = n_match + n_reject_match + n_diverge + n_crash + n_unsupported + n_rust_only;
    let summary = format!(
        "\nSUMMARY: {n_match} MATCH / {n_reject_match} REJECT_MATCH / {n_diverge} DIVERGE / {n_crash} MIND_CRASH / \
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
    crate::common::gate::skipped(
        "g2_differential_mlir",
        "differential harness dlopen()s a Linux ELF; gated to target_os = linux",
    );
}

#[cfg(target_os = "linux")]
#[test]
fn g2_1_differential_coverage() {
    // When re-exec'd as a single-fixture worker this never returns.
    run_as_worker_if_requested();

    let bin = require_mindc();

    let Some(so_path) = oracle_so_path(&bin) else {
        // A missing oracle is environmental and may legitimately skip — but it must
        // not look like a run that passed. Without a marker the exec-tier gate cannot
        // distinguish "compared every fixture and found no divergence" from "compared
        // nothing", and a harness that silently asserts nothing is the same class of
        // false green as the crash-as-unsupported bug this file was just fixed for.
        //
        // SDLC-GATE naming matches the ran=/fail= convention the other gates print.
        // scripts/exec_semantics_gate.sh CONSUMES this line: it attributes the marker
        // to the enclosing `Running tests/<target>.rs` block and treats ran=0 as a
        // failure to EXECUTE unless that target is named in ENV_TOLERATED_<tier>.
        // g2_differential_mlir is named there, so this skip is reported and counted
        // rather than fatal — but it can no longer read as a green run.
        emit_gate_marker("SDLC-GATE g2_differential ran=0 fail=0 SKIPPED");
        crate::common::gate::skipped(
            "g2_differential_mlir",
            "g2_differential_mlir: SKIP -- could not obtain a valid \
             libmindc_mind.so (MLIR toolchain absent). This asserted NOTHING; do not \
             read it as a pass.",
        );
        return;
    };

    // Defence-in-depth: oracle_so_path only ever returns a path that starts
    // with the ELF magic, but re-verify before dlopen so a non-native or
    // truncated artifact skips cleanly instead of panicking in the loader.
    match fs::read(&so_path) {
        Ok(bytes) if bytes.starts_with(b"\x7fELF") => {}
        _ => {
            // The LIKELIER of the two skips: the stale/truncated in-tree .so that
            // ENV_TOLERATED_exec documents. It emitted BOTH markers by hand and never
            // touched common::gate, so MIND_BENCH_REQUIRE=1 could not turn it into a
            // failure — the exec tier could demand a real backend and be handed a pass.
            emit_gate_marker("SDLC-GATE g2_differential ran=0 fail=0 SKIPPED");
            crate::common::gate::skipped(
                "g2_differential_mlir",
                &format!(
                    "{} is not a native ELF. This asserted NOTHING.",
                    so_path.display()
                ),
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
    // An empty program still emits a module; this also pins next_id's ABI word
    // independently of last_id, which remains -1 for that program.
    assert!(
        matches!(unsafe { call_mindc_compile(&lib, b"") }, MindCall::Ok(bytes) if !bytes.is_empty())
    );
    capability_ratchet::assert_preserved(&repo_root(), &rows);

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

    // Gate part 1: a CRASH fails, always and first.
    //
    // This check did not exist. An abnormal termination was classified as
    // MIND_UNSUPPORTED and the gate looked only at DIVERGE, so the pure-MIND
    // compiler segfaulted 21 times across four days inside test runs that reported
    // pass. A construct the compiler declines is a backlog item; a construct that
    // KILLS it is a defect, and the difference has to be visible to the gate.
    let crashes: Vec<&(PathBuf, Outcome)> = rows
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::MindCrash { .. }))
        .collect();

    if !crashes.is_empty() {
        let root = repo_root();
        let mut msg = format!(
            "G2.1 GATE FAILED: the pure-MIND compiler CRASHED on {} fixture(s).\n\
             This is not a porting gap — an unsupported construct returns an explicit refusal \
             and is reported as MIND_UNSUPPORTED. These terminated abnormally.\n",
            crashes.len()
        );
        for (path, outcome) in &crashes {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            if let Outcome::MindCrash { reason } = outcome {
                msg.push_str(&format!("\n  {rel}:\n    {reason}\n"));
            }
        }
        panic!("{msg}");
    }

    // Gate part 2: fail if any DIVERGE exists.
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

    // Require actual positive byte comparisons as well as zero failures.
    // Keep the existing floor: rejection matches and uncompiled inputs cannot
    // compensate for a collapse of successful comparisons.
    const MIN_MATCH: usize = 90;
    assert!(
        n_match >= MIN_MATCH,
        "G2.1 GATE FAILED: only {n_match} fixture(s) MATCHED (floor {MIN_MATCH}) out of {} \
         — {n_unsupported} MIND_UNSUPPORTED, {n_rust_only} RUST_ONLY. DIVERGE == 0 is \
         vacuous when nothing was compared: this run demonstrated the absence of a \
         comparison, not the absence of a divergence.",
        rows.len()
    );

    let n_reject_match = rows
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::RejectMatch))
        .count();
    println!(
        "g2_differential_mlir PASS: {} MATCH / {n_reject_match} REJECT_MATCH / 0 DIVERGE / 0 MIND_CRASH / {} MIND_UNSUPPORTED / \
         {} RUST_ONLY out of {} fixtures",
        n_match,
        n_unsupported,
        n_rust_only,
        rows.len()
    );
}
