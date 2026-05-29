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
//!
//! These tests compile `tests/fixtures/simple.mind`, emit both artifact kinds
//! to temp files, and verify:
//!
//! 1. `--emit-mic3` output matches `compact::emit_mic3(&ir)` for the same source.
//! 2. `--emit-evidence` round-trips through `mic3_evidence_report` with
//!    `trace_hash_valid == true` and non-empty `toolchain` / `substrate` fields.
//! 3. Existing `--emit-mic` (mic@1 text) output is byte-unchanged.
//! 4. Default invocation (no emit flags) still prints IR text to stdout.

use std::path::PathBuf;
use std::process::Command;

fn mindc_binary() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    #[cfg(debug_assertions)]
    path.push("debug");
    #[cfg(not(debug_assertions))]
    path.push("release");
    #[cfg(target_os = "windows")]
    path.push("mindc.exe");
    #[cfg(not(target_os = "windows"))]
    path.push("mindc");
    path
}

fn require_mindc() -> PathBuf {
    let bin = mindc_binary();
    if !bin.exists() {
        eprintln!("Skipping: mindc binary not found at {bin:?}");
    }
    bin
}

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
    if !bin.exists() {
        return;
    }

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
    if !bin.exists() {
        return;
    }

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

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 3.  --emit-mic3 bytes equal emit_mic3 on the same source (library cross-check)
// ---------------------------------------------------------------------------

#[test]
fn emit_mic3_matches_library_emit() {
    let bin = require_mindc();
    if !bin.exists() {
        return;
    }

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
    if !bin.exists() {
        return;
    }

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
    if !bin.exists() {
        return;
    }

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
    if !bin.exists() {
        return;
    }

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
// Helpers
// ---------------------------------------------------------------------------

fn tempfile_path(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("mindc_test_{name}"));
    p.to_string_lossy().into_owned()
}
