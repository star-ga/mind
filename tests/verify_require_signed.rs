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

//! SECURITY regression — `mindc verify --require-signed` MUST fail closed on an
//! UNATTESTED artifact (one carrying no evidence_chain).
//!
//! A signature lives INSIDE the evidence chain (`signature.*`), so an artifact
//! with no evidence_chain carries no signature at all. Before the fix, the
//! `Err(EvidenceError::Missing)` arm of `mindc verify` (src/bin/mindc.rs)
//! consulted only the pinned-signer allowlist, `--require-strict-fp` and
//! `--require-deterministic`; `require_signed` was read ONLY on the attested
//! path, so the Missing arm fell through to a bare `0`. A CI gate of the form
//! `mindc verify --require-signed artifact.mic3 && deploy` therefore DEPLOYED a
//! fully attacker-authored, unsigned, unattested artifact — the exact inverse of
//! the flag's contract, and the same stripped-evidence_chain downgrade path that
//! `verify_pinned_signer.rs` closes for `--signer-pubkey`.
//!
//! RFC 0017 §4.5: a failed `--require-*` gate is exit 1 (2 is reserved for I/O /
//! CLI errors). Plain `verify` with no flag still exits 0 on an unattested
//! artifact — attestation is absent, not failed.
//!
//! NOTE: this file carries NO `#![cfg(feature = ...)]` gate on purpose. A test
//! file compiled out by a missing feature prints "ok. 0 passed" and exits 0 —
//! a false green. Every test here must actually run in every configuration.

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

#[test]
fn require_signed_rejects_unattested_artifact() {
    let dir = tempdir().unwrap();
    let mic3 = emit_plain_mic3(dir.path());
    let out = mindc()
        .arg("verify")
        .arg("--require-signed")
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc verify");
    assert!(
        !out.status.success(),
        "SECURITY REGRESSION: `mindc verify --require-signed` returned 0 on an \
         unsigned, unattested artifact — `verify --require-signed && deploy` \
         would deploy attacker code. stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // RFC 0017 §4.5: a failed --require-* gate is exit 1, not 2 (I/O / CLI error).
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failed --require-signed gate must exit 1 per RFC 0017 §4.5, got {:?}",
        out.status.code(),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--require-signed") && stderr.contains("no evidence_chain"),
        "expected a --require-signed fail-closed diagnostic, got: {stderr}"
    );
}

#[test]
fn require_signed_rejects_unattested_artifact_json() {
    // The `--json` leg must fail closed too: a JSON consumer reads the exit
    // code, and the pre-fix JSON path emitted `{"attested":false}` and exited 0.
    let dir = tempdir().unwrap();
    let mic3 = emit_plain_mic3(dir.path());
    let out = mindc()
        .arg("verify")
        .arg("--json")
        .arg("--require-signed")
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc verify --json");
    assert_eq!(
        out.status.code(),
        Some(1),
        "`verify --json --require-signed` on an unattested artifact must exit 1, \
         got {:?}. stdout={} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn plain_verify_still_accepts_unattested_artifact() {
    // Control (RFC 0017): WITHOUT the flag, an unattested but SSA-well-formed
    // artifact still verifies — attestation is absent, not failed. This is the
    // line the fix must NOT cross: the gate is opt-in, not a new default.
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

#[test]
fn require_signed_rejects_attested_but_unsigned_artifact() {
    // Regression guard on the ATTESTED arm (already correct pre-fix): an
    // artifact WITH an evidence_chain but no signature must also fail closed,
    // so the two arms agree. Signing is opt-in, so `--emit-evidence` alone
    // produces exactly this shape.
    let dir = tempdir().unwrap();
    let src = dir.path().join("attested.mind");
    fs::write(&src, "fn main() -> i64 {\n    return 7;\n}\n").unwrap();
    let mic3 = dir.path().join("attested.mic3");
    let build = mindc()
        .arg(src.to_str().unwrap())
        .arg("--emit-evidence")
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc --emit-evidence");
    assert!(
        build.status.success(),
        "emit-evidence build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    // Confirm the artifact really IS attested — otherwise this test would
    // silently re-test the unattested path above and prove nothing. The
    // unattested JSON report is `{artifact, ssa_valid, ssa_reason, attested, scope}`;
    // only the attested report carries `trace_hash_valid` / `signature`.
    let plain = mindc()
        .arg("verify")
        .arg("--json")
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc verify --json");
    let stdout = String::from_utf8_lossy(&plain.stdout);
    assert!(
        stdout.contains("\"trace_hash_valid\":true") && !stdout.contains("\"attested\":false"),
        "fixture is not attested, so this test would not exercise the attested \
         arm; verify --json said: {stdout}"
    );
    assert!(
        stdout.contains("\"signature\":\"absent\""),
        "fixture must be attested-but-UNSIGNED for this test to mean anything; \
         verify --json said: {stdout}"
    );

    let out = mindc()
        .arg("verify")
        .arg("--require-signed")
        .arg(mic3.to_str().unwrap())
        .output()
        .expect("failed to spawn mindc verify");
    assert_eq!(
        out.status.code(),
        Some(1),
        "`--require-signed` must reject an attested-but-UNSIGNED artifact \
         (exit 1), got {:?}. stdout={} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
