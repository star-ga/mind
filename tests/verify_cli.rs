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

//! Integration tests for the `mindc verify` subcommand (RFC 0021 §4.2 /
//! #288 / #290 / #309 step 4): consumer-side evidence-chain verification.
//!
//! `mindc verify <artifact>` reads an artifact emitted by
//! `mindc build --emit-evidence`, peels the `evidence_chain.*` MAP, recomputes
//! the canonical mic@3 `trace_hash`, and confirms it matches. These tests
//! exercise the four outcomes:
//!
//! * valid, untampered artifact      → exit 0
//! * tampered body byte              → exit 1 (trace_hash mismatch)
//! * plain mic@3 with no evidence    → exit 1 (unattested)
//! * unreadable artifact path        → exit 2 (I/O error)

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

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

fn tempfile_path(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("mindc_verify_test_{name}"));
    p.to_string_lossy().into_owned()
}

/// Emit an evidence artifact for `simple.mind` to `out`, asserting success.
/// Returns `false` (skip) if the mindc binary is not built.
fn emit_evidence(bin: &PathBuf, out: &str) -> bool {
    if !bin.exists() {
        eprintln!("Skipping: mindc binary not found at {bin:?}");
        return false;
    }
    let res = Command::new(bin)
        .args([&fixture("simple.mind"), "--emit-evidence", out])
        .output()
        .expect("run mindc --emit-evidence");
    assert!(
        res.status.success(),
        "mindc --emit-evidence failed:\n{}",
        String::from_utf8_lossy(&res.stderr)
    );
    true
}

// ---------------------------------------------------------------------------
// 1.  Valid artifact → exit 0, report says trace_hash_valid.
// ---------------------------------------------------------------------------

#[test]
fn verify_valid_artifact_exit_zero() {
    let bin = mindc_binary();
    let tmp = tempfile_path("valid.bin");
    if !emit_evidence(&bin, &tmp) {
        return;
    }

    let out = Command::new(&bin)
        .args(["verify", &tmp])
        .output()
        .expect("run mindc verify");

    assert!(
        out.status.success(),
        "verify must exit 0 for an untampered artifact; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("trace_hash_valid: yes"),
        "human report must confirm validity, got:\n{stdout}"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 2.  --json emits a machine-readable object with trace_hash_valid:true.
// ---------------------------------------------------------------------------

#[test]
fn verify_json_reports_valid() {
    let bin = mindc_binary();
    let tmp = tempfile_path("json.bin");
    if !emit_evidence(&bin, &tmp) {
        return;
    }

    let out = Command::new(&bin)
        .args(["verify", &tmp, "--json"])
        .output()
        .expect("run mindc verify --json");

    assert!(
        out.status.success(),
        "verify --json must exit 0 for valid artifact"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"trace_hash_valid\":true"),
        "JSON must report trace_hash_valid:true, got:\n{stdout}"
    );
    assert!(
        stdout.contains("\"substrate\":\"cpu\""),
        "JSON must carry the substrate field, got:\n{stdout}"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 3.  Tampered body byte → exit 1 (verification fails).
// ---------------------------------------------------------------------------

#[test]
fn verify_tampered_artifact_fails() {
    let bin = mindc_binary();
    let tmp = tempfile_path("tampered.bin");
    if !emit_evidence(&bin, &tmp) {
        return;
    }

    // Flip one bit in the final byte. With today's emitter the
    // `evidence_chain.trace_hash` value ends the artifact, so this mutates the
    // stored hash and the report parses but reports trace_hash_valid = false.
    // The `MISMATCH` stderr assertion below — not the byte position — is the
    // real guard: a Malformed parse error prints a different message, so this
    // test fails loudly if a future emit-order change moves the flipped byte.
    let mut bytes = std::fs::read(&tmp).expect("read emitted artifact");
    assert!(bytes.len() > 8, "artifact unexpectedly tiny");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&tmp, &bytes).expect("write tampered artifact");

    let out = Command::new(&bin)
        .args(["verify", &tmp])
        .output()
        .expect("run mindc verify on tampered artifact");

    assert_eq!(
        out.status.code(),
        Some(1),
        "tampered artifact must exit 1, stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MISMATCH"),
        "error must report a trace_hash mismatch, got:\n{stderr}"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 4.  Plain mic@3 (no evidence chain) → exit 1 (unattested).
// ---------------------------------------------------------------------------

#[test]
fn verify_unattested_artifact_fails() {
    let bin = mindc_binary();
    if !bin.exists() {
        eprintln!("Skipping: mindc binary not found");
        return;
    }

    let tmp = tempfile_path("plain.bin");
    let emit = Command::new(&bin)
        .args([&fixture("simple.mind"), "--emit-mic3", &tmp])
        .output()
        .expect("run mindc --emit-mic3");
    assert!(emit.status.success(), "--emit-mic3 must succeed");

    let out = Command::new(&bin)
        .args(["verify", &tmp])
        .output()
        .expect("run mindc verify on plain mic@3");

    assert_eq!(
        out.status.code(),
        Some(1),
        "unattested artifact must exit 1"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unattested"),
        "error must explain the artifact is unattested, got:\n{stderr}"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// 5.  Unreadable path → exit 2 (I/O error, distinct from verification failure).
// ---------------------------------------------------------------------------

#[test]
fn verify_missing_file_exit_two() {
    let bin = mindc_binary();
    if !bin.exists() {
        eprintln!("Skipping: mindc binary not found");
        return;
    }

    let missing = tempfile_path("definitely_absent.bin");
    let _ = std::fs::remove_file(&missing);

    let out = Command::new(&bin)
        .args(["verify", &missing])
        .output()
        .expect("run mindc verify on missing file");

    assert_eq!(
        out.status.code(),
        Some(2),
        "missing artifact must exit 2 (I/O), not 1 (verification), stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
