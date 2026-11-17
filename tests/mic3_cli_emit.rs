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

//! Integration tests for RFC 0021 step 3: `--emit-mic3` and `--emit-evidence`
//! CLI flags on the `mindc` binary.
//! ELF note type 1 is not covered here: that note is the seeded compiler module digest, not the user's --emit-mic3 bytes.
//!
//! These tests compile `tests/fixtures/simple.mind`, emit both artifact kinds
//! to temp files, and verify:
//!
//! 1. `--emit-mic3` output matches `compact::emit_mic3(&ir)` for the same source.
//! 2. `--emit-evidence` round-trips through `mic3_evidence_report` with
//!    `trace_hash_valid == true` and non-empty `toolchain` / `substrate` fields.
//! 3. Existing `--emit-mic` (mic@1 text) output is byte-unchanged.
//! 4. Default invocation (no emit flags) still prints IR text to stdout.

mod common;
use common::{require_mindc, scratch_dir};

use std::path::PathBuf;
use std::process::Command;

// require_mindc() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

// ---------------------------------------------------------------------------
// 1.  --emit-mic3 produces valid binary that parses back correctly
// ---------------------------------------------------------------------------

#[test]
fn emit_mic3_writes_parseable_binary() {
    let bin = require_mindc();

    let tmp = tempfile_path("mic3_output.bin");
    let out = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-mic3", &tmp])
        .output()
        .expect("run mindc --emit-mic3");

    assert!(
        out.status.success(),
        "mindc --emit-mic3 failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes =
        std::fs::read(&tmp).unwrap_or_else(|e| panic!("cannot read output file {tmp}: {e}"));

    // Verify magic header: 'M','I','C','3'
    assert!(
        bytes.starts_with(b"MIC3"),
        "--emit-mic3 output must start with MIC3 magic, got {:02X?}",
        &bytes[..bytes.len().min(8)]
    );

    // Must parse without error.
    let parsed = libmind::ir::compact::parse_mic3(&bytes)
        .expect("parse_mic3 must succeed on --emit-mic3 output");

    // Round-trip: re-emit must equal the file bytes.
    let reemitted = libmind::ir::compact::emit_mic3(&parsed);
    assert_eq!(
        bytes, reemitted,
        "--emit-mic3 output must be fixed-point under emit_mic3(parse_mic3(...))"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 2.  --emit-evidence: trace_hash_valid == true; substrate and toolchain set
// ---------------------------------------------------------------------------

#[test]
fn emit_evidence_report_validates() {
    let bin = require_mindc();

    let tmp = tempfile_path("evidence_output.bin");
    let out = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-evidence", &tmp])
        .output()
        .expect("run mindc --emit-evidence");

    assert!(
        out.status.success(),
        "mindc --emit-evidence failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes =
        std::fs::read(&tmp).unwrap_or_else(|e| panic!("cannot read output file {tmp}: {e}"));

    let report = libmind::ir::compact::mic3_evidence_report(&bytes)
        .expect("mic3_evidence_report must succeed on --emit-evidence output");

    assert!(
        report.trace_hash_valid,
        "trace_hash_valid must be true for a freshly emitted artifact"
    );
    assert!(
        !report.toolchain.is_empty(),
        "toolchain field must be non-empty"
    );
    assert!(
        !report.substrate.is_empty(),
        "substrate field must be non-empty"
    );

    // The built-in verifier-core self-check (RFC 0016 Phase B-CLI) must run and
    // pass on the happy path: the success message reports it explicitly so a
    // regression that silently drops the self-check is caught here.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("self-check ok"),
        "--emit-evidence success must report the self-check ran, got stderr: {stderr}"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 3.  --emit-mic3 bytes equal emit_mic3 on the same source (library cross-check)
// ---------------------------------------------------------------------------

#[test]
fn emit_mic3_matches_library_emit() {
    let bin = require_mindc();

    let tmp = tempfile_path("mic3_crosscheck.bin");
    let out = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-mic3", &tmp])
        .output()
        .expect("run mindc --emit-mic3 for crosscheck");

    assert!(out.status.success());

    let cli_bytes = std::fs::read(&tmp).unwrap_or_else(|e| panic!("cannot read {tmp}: {e}"));

    // Reproduce via the library directly.
    let source = std::fs::read_to_string(fixture("simple.mind")).unwrap();
    let opts = libmind::pipeline::CompileOptions::default();
    let products = libmind::pipeline::compile_source_with_name(&source, Some("simple.mind"), &opts)
        .expect("library compile must succeed");
    let lib_bytes = libmind::ir::compact::emit_mic3(&products.ir);

    assert_eq!(
        cli_bytes, lib_bytes,
        "--emit-mic3 output must equal emit_mic3 called with the same IR"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 4.  --emit-mic (mic@1 text) is unchanged — regression guard
// ---------------------------------------------------------------------------

#[test]
fn emit_mic1_unchanged_after_rfc0021_step3() {
    let bin = require_mindc();

    let out = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-mic"])
        .output()
        .expect("run mindc --emit-mic");

    assert!(
        out.status.success(),
        "mindc --emit-mic failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.trim_start().starts_with("mic@1"),
        "--emit-mic output must start with 'mic@1', got: {text}"
    );
}

// ---------------------------------------------------------------------------
// 5.  Default invocation (no emit flags) still prints IR text
// ---------------------------------------------------------------------------

#[test]
fn default_invocation_still_prints_ir() {
    let bin = require_mindc();

    let out = Command::new(&bin)
        .args([&fixture("simple.mind")])
        .output()
        .expect("run mindc with no emit flags");

    assert!(
        out.status.success(),
        "default mindc invocation failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let text = String::from_utf8_lossy(&out.stdout);
    // The IR printer always emits at least one "output" line.
    assert!(
        text.to_lowercase().contains("output"),
        "default stdout must contain IR output line, got: {text}"
    );
}

// ---------------------------------------------------------------------------
// 6.  --emit-evidence: body is prefix-equal to --emit-mic3 for same source
// ---------------------------------------------------------------------------

#[test]
fn evidence_body_prefix_equals_plain_mic3() {
    let bin = require_mindc();

    let tmp_plain = tempfile_path("mic3_plain.bin");
    let tmp_ev = tempfile_path("mic3_ev.bin");

    let out_plain = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-mic3", &tmp_plain])
        .output()
        .expect("run mindc --emit-mic3");
    let out_ev = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-evidence", &tmp_ev])
        .output()
        .expect("run mindc --emit-evidence");

    assert!(out_plain.status.success());
    assert!(out_ev.status.success());

    let plain = std::fs::read(&tmp_plain).unwrap();
    let with_ev = std::fs::read(&tmp_ev).unwrap();

    assert!(
        with_ev.starts_with(&plain),
        "--emit-evidence bytes must start with the --emit-mic3 body"
    );
    assert!(
        with_ev.len() > plain.len(),
        "--emit-evidence must append MAP epilogue bytes beyond the plain body"
    );

    let _ = std::fs::remove_file(&tmp_plain);
    let _ = std::fs::remove_file(&tmp_ev);
}

// ---------------------------------------------------------------------------
// 7.  Stored trace_hash equals an independent SHA-256 of --emit-mic3
// ---------------------------------------------------------------------------
//
// The digest is hashlib and sha256sum of the bytes this run just wrote.
// A stored note is not an oracle. mindc verify is not an oracle.

/// SELECTION: true on the rows whose `mindc` resolves `import std.vec` (the
/// keystone fixture). The same cfg guards the type checker's std-import refusal.
const KEYSTONE_ROW: bool = cfg!(any(
    feature = "std-surface",
    feature = "cross-module-imports"
));

/// COUNT: the exact program count the hash test must run on this row. The
/// literal is stated per row by its own cfg attribute and is never computed
/// from `KEYSTONE_ROW`. Computed from the selector, the count would follow any
/// change of the selection and a dropped keystone would stay green.
#[cfg(any(feature = "std-surface", feature = "cross-module-imports"))]
const EXPECTED_RAN: u32 = 4; // tiny, note_kinds, simple, keystone_fixture
#[cfg(not(any(feature = "std-surface", feature = "cross-module-imports")))]
const EXPECTED_RAN: u32 = 3; // tiny, note_kinds, simple

#[test]
fn emit_trace_hash_matches_independent_sha256() {
    let bin = mindc_for_independent_hash();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = unique_dir("indep_hash");
    let tiny = dir.join("tiny.mind");
    std::fs::write(&tiny, "fn main() -> i64 { 40 + 2 }\n").expect("write tiny");
    // bank_acc8.mind calls ymm intrinsics that the v0.10.2 compiler rejects.
    // The shared corpus is the programs both compilers emit.
    let mut programs = vec![
        ("tiny", tiny),
        (
            "note_kinds",
            root.join("examples/mindc_mind/testdata/native_cf/note_kinds.mind"),
        ),
        ("simple", root.join("tests/fixtures/simple.mind")),
    ];
    // The keystone fixture imports `std.vec`: a row built without the std
    // surface refuses it with E2007, so it joins the corpus only where it emits.
    if KEYSTONE_ROW {
        programs.push((
            "keystone_fixture",
            root.join("examples/mindc_mind/fixture.mind"),
        ));
    }
    let mut ran = 0u32;
    for (label, src) in &programs {
        assert!(
            src.is_file(),
            "corpus file missing ({label}): {}",
            src.display()
        );
        let mic = dir.join(format!("{label}.mic3"));
        let ev = dir.join(format!("{label}.ev"));
        emit_artifact(&bin, src, "--emit-mic3", &mic);
        emit_artifact(&bin, src, "--emit-evidence", &ev);
        let (hashlib_hex, sha256sum_hex) = independent_sha256(&mic);
        assert_eq!(
            hashlib_hex, sha256sum_hex,
            "{label}: hashlib {hashlib_hex} != sha256sum {sha256sum_hex}"
        );
        let stored = stored_trace_hash(&std::fs::read(&ev).expect("read evidence"))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            stored, hashlib_hex,
            "E2 mismatch {label}: stored={stored} independent={hashlib_hex}"
        );
        ran += 1;
    }
    assert_eq!(
        ran, EXPECTED_RAN,
        "ran={ran}, expected {EXPECTED_RAN} on this row; a dropped program, a \
         missing compiler or an empty corpus is a failure"
    );
    // Control for the selection: where the keystone is left out, this row's
    // `mindc` must refuse it with E2007. Where it is left out but the binary
    // emits it, the selection is wrong and this assertion is red.
    if !KEYSTONE_ROW {
        let keystone = root.join("examples/mindc_mind/fixture.mind");
        assert!(
            keystone.is_file(),
            "keystone fixture missing: {}",
            keystone.display()
        );
        let run = run_emit(
            &bin,
            &keystone,
            "--emit-mic3",
            &dir.join("keystone_refused.mic3"),
        );
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success() && stderr.contains("E2007") && stderr.contains("import std.vec"),
            "the keystone is left out of this row's corpus, so its mindc must \
             refuse it with E2007 on `import std.vec`; status={:?} stderr:\n{stderr}",
            run.status.code()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn mindc_for_independent_hash() -> PathBuf {
    if let Some(raw) = std::env::var_os("MINDC_BIN") {
        let path = PathBuf::from(&raw);
        assert!(
            path.is_file(),
            "MINDC_BIN is set but the compiler is missing: {}",
            path.display()
        );
        return path;
    }
    PathBuf::from(require_mindc())
}

fn run_emit(bin: &PathBuf, src: &PathBuf, flag: &str, out: &PathBuf) -> std::process::Output {
    Command::new(bin)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([src.as_os_str(), std::ffi::OsStr::new(flag), out.as_os_str()])
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn mindc {flag}: {e}"))
}

fn emit_artifact(bin: &PathBuf, src: &PathBuf, flag: &str, out: &PathBuf) {
    let run = run_emit(bin, src, flag, out);
    assert!(
        run.status.success(),
        "mindc {flag} {} failed:\n{}",
        src.display(),
        String::from_utf8_lossy(&run.stderr)
    );
}

fn independent_sha256(path: &PathBuf) -> (String, String) {
    let py = Command::new("python3")
        .args([
            "-c",
            "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",
        ])
        .arg(path)
        .output()
        .expect("python3 hashlib");
    assert!(
        py.status.success(),
        "python3 hashlib failed: {}",
        String::from_utf8_lossy(&py.stderr)
    );
    let hashlib_hex = String::from_utf8_lossy(&py.stdout).trim().to_string();
    let sum = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum");
    assert!(
        sum.status.success(),
        "sha256sum failed: {}",
        String::from_utf8_lossy(&sum.stderr)
    );
    let sha256sum_hex = String::from_utf8_lossy(&sum.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    (hashlib_hex, sha256sum_hex)
}

/// Read the stored `evidence_chain.trace_hash` bytes. The walk is the MAP
/// layout: ULEB key length, key, tag 0x02, ULEB 32, then 32 digest bytes.
fn stored_trace_hash(bytes: &[u8]) -> Result<String, String> {
    let key = b"evidence_chain.trace_hash";
    let mut hits = Vec::new();
    let mut start = 0usize;
    while start < bytes.len() {
        let Some(rel) = bytes[start..]
            .windows(key.len())
            .position(|window| window == key)
        else {
            break;
        };
        let at = start + rel;
        start = at + 1;
        if at == 0 || bytes[at - 1] != u8::try_from(key.len()).unwrap_or(0) {
            continue;
        }
        let tag_at = at + key.len();
        if tag_at + 2 + 32 > bytes.len() || bytes[tag_at] != 0x02 || bytes[tag_at + 1] != 32 {
            continue;
        }
        hits.push(to_hex(&bytes[tag_at + 2..tag_at + 2 + 32]));
    }
    match hits.as_slice() {
        [one] => Ok(one.clone()),
        other => Err(format!(
            "expected one stored trace_hash, found {}",
            other.len()
        )),
    }
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

/// A subdirectory of this target's scratch directory, so the caller can remove
/// it whole without touching the files its sibling tests write beside it.
fn unique_dir(label: &str) -> PathBuf {
    let path = scratch_dir("mic3_cli_emit").join(label);
    std::fs::create_dir_all(&path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    path
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tempfile_path(name: &str) -> String {
    scratch_dir("mic3_cli_emit")
        .join(format!("mindc_test_{name}"))
        .to_string_lossy()
        .into_owned()
}
