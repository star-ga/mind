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

//! Audit rank 1 (SECURITY) regression — a pinned signer (`--signer-pubkey`)
//! MUST fail closed on an UNATTESTED artifact (one carrying no evidence_chain).
//!
//! Before the fix, the `Err(EvidenceError::Missing)` arm of `mindc verify`
//! (src/bin/mindc.rs) gated only `--require-strict-fp` / `--require-deterministic`
//! and never consulted the `trusted` allowlist, so
//! `mindc verify --signer-pubkey KEY evil.mic3` returned 0 on a fully
//! attacker-authored, unsigned artifact — a CI gate of the form
//! `verify --signer-pubkey KEY && deploy` would deploy attacker code. The
//! attested path already rejected a pinned-but-unsigned artifact (mindc.rs
//! ~1866/1887); this closes the stripped-evidence_chain downgrade path.

use std::fs;
use std::process::Command;

use tempfile::tempdir;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// Compile a trivial program to a PLAIN mic@3 (no `--emit-evidence`, so the
/// artifact carries no evidence_chain and verifies as `Missing` / unattested).
fn emit_plain_mic3(dir: &std::path::Path) -> std::path::PathBuf {
    let src = dir.join("evil.mind");
    fs::write(&src, "fn main() -> i64 {\n    return 1337;\n}\n").unwrap();
    let out = dir.join("evil.mic3");
    let res = mindc()
        .arg(src.to_str().unwrap())
        .arg("--emit-mic3")
        .arg(out.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc --emit-mic3");
    assert!(
        res.status.success(),
        "emit-mic3 failed: {}",
        String::from_utf8_lossy(&res.stderr)
    );
    out
}

/// 32-byte (64 hex char) placeholder pubkey — never signed anything here.
const PINNED_KEY: &str = "abababababababababababababababababababababababababababababababab";

#[test]
fn pinned_signer_rejects_unattested_artifact() {
    let dir = tempdir().unwrap();
    let mic3 = emit_plain_mic3(dir.path());
    // A pinned signer on an artifact with NO evidence_chain must fail closed:
    // there is no signature to verify, so the pin cannot be satisfied.
    let out = mindc()
        .arg("verify")
        .arg("--signer-pubkey")
        .arg(PINNED_KEY)
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc verify");
    assert!(
        !out.status.success(),
        "SECURITY REGRESSION (audit rank 1): `mindc verify --signer-pubkey KEY` \
         returned 0 on an unsigned, unattested artifact. stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("pinned") && stderr.contains("no evidence_chain"),
        "expected a pinned-signer fail-closed diagnostic, got: {stderr}"
    );
}

#[test]
fn plain_verify_accepts_unattested_artifact() {
    // Control: WITHOUT a pinned signer, an unattested (but SSA-well-formed)
    // artifact still verifies — attestation is absent, not failed (RFC 0017).
    let dir = tempdir().unwrap();
    let mic3 = emit_plain_mic3(dir.path());
    let out = mindc()
        .arg("verify")
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc verify");
    assert!(
        out.status.success(),
        "plain verify of an unattested artifact should exit 0: stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
}
