// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Unsigned evidence identity across the PQC feature split, plus the signing
//! smoke the release-profile CI job runs on the binary it packaged.
//!
//! `tests/pqc_hybrid_cli_signing.rs` compares an unsigned artifact with
//! `tests/fixtures/evidence/unsigned_reference.mic3`, but that whole file is
//! compiled only when BOTH signing features are on. No build WITHOUT them was
//! compared with the same bytes, so "compiling signing in does not move unsigned
//! output" rested on a manual cross-check. This file is not gated on the signing
//! features. Its fixture comparison needs `std-surface` (see the twin's note), so
//! the `--no-default-features` row does not run it; the std-surface `cargo test`
//! rows of ci.yml's build_test job run it with PQC compiled OUT, and the
//! release-profile job runs it with the exact release feature set compiled IN.
//! Both compare with the one committed fixture, so the two builds emit
//! byte-identical unsigned evidence or one of them goes red.
//!
//! The binary under test is `CARGO_BIN_EXE_mindc` unless
//! `MIND_RELEASE_SMOKE_MINDC` names another. The release-profile job points it
//! at the binary unpacked from the archive it built the way release.yml builds
//! it, so the smoke runs on what ships, not on a test-profile rebuild. Seeds are
//! the fixed, public TEST values the signing suite uses; no key is read.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The program the committed fixture was emitted from.
const REFERENCE_PROGRAM: &str = "fn main() -> i64 {\n    return 7;\n}\n";
const FIXTURE: &str = "tests/fixtures/evidence/unsigned_reference.mic3";
const SIGNING_SUITE_SUPPORT: &str = "tests/support/pqc_hybrid_cli_support.rs";

/// Deterministic, public, non-production TEST seeds (same as the signing suite).
const TEST_MLDSA87_SEED: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const TEST_SLHDSA_SEED: &str = concat!(
    "222222222222222222222222222222222222222222222222",
    "222222222222222222222222222222222222222222222222",
    "222222222222222222222222222222222222222222222222",
    "222222222222222222222222222222222222222222222222",
);

/// Every variable that can supply a signing key or a trust anchor. Each run
/// starts with all of them removed, so an inherited value cannot sign an
/// "unsigned" emit or redefine what "trusted" means.
const SIGNING_ENV: &[&str] = &[
    "MIND_EVIDENCE_MLDSA87_KEY",
    "MIND_EVIDENCE_SLHDSA_KEY",
    "MIND_EVIDENCE_MLDSA_KEY",
    "MIND_EVIDENCE_ED25519_KEY",
    "MIND_EVIDENCE_VERIFY_PUBKEYS",
];

fn mindc() -> PathBuf {
    std::env::var_os("MIND_RELEASE_SMOKE_MINDC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_mindc")))
}

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

struct Scratch {
    dir: tempfile::TempDir,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("p.mind"), REFERENCE_PROGRAM).expect("write source");
        Self { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn mindc(&self) -> Command {
        let mut cmd = Command::new(mindc());
        cmd.current_dir(self.dir.path());
        for key in SIGNING_ENV {
            cmd.env_remove(key);
        }
        cmd
    }

    fn emit(&self, out: &str, seeds: &[(&str, &str)]) -> Output {
        let mut cmd = self.mindc();
        cmd.arg(self.path("p.mind"))
            .arg("--emit-evidence")
            .arg(self.path(out));
        for (key, value) in seeds {
            cmd.env(key, value);
        }
        cmd.output().expect("spawn mindc")
    }
}

const BOTH_LEGS: &[(&str, &str)] = &[
    ("MIND_EVIDENCE_MLDSA87_KEY", TEST_MLDSA87_SEED),
    ("MIND_EVIDENCE_SLHDSA_KEY", TEST_SLHDSA_SEED),
];

/// The PQC-OFF twin of `the_unsigned_path_is_unchanged_by_the_retirement`:
/// with or without the signing features, unsigned bytes equal the fixture.
///
/// Gated on `std-surface` because the fixture is a std-surface artifact: the
/// same program emitted by a `--no-default-features` build has a different
/// body (266 bytes, not 269, and another trace_hash), so comparing that build
/// with this fixture would measure the frontend split, not the signing split.
#[cfg(feature = "std-surface")]
#[test]
fn unsigned_evidence_matches_the_committed_reference_with_or_without_signing() {
    let s = Scratch::new();
    let out = s.emit("plain.mic3", &[]);
    assert_eq!(out.status.code(), Some(0), "unsigned emit: {}", text(&out));
    let bytes = std::fs::read(s.path("plain.mic3")).expect("read artifact");
    let reference = std::fs::read(repo(FIXTURE)).expect("committed unsigned reference");
    println!(
        "binary={} pqc_compiled_in={} bytes={}",
        mindc().display(),
        cfg!(all(feature = "evidence-mldsa", feature = "evidence-slhdsa")),
        bytes.len()
    );
    assert_eq!(
        bytes, reference,
        "unsigned evidence must be byte-identical to {FIXTURE} with and without the \
         signing features compiled in"
    );
}

/// Drift guard: the twin above compares against the fixture only while it
/// compiles the same program the signing suite does.
#[test]
fn the_reference_program_is_the_one_the_signing_suite_compiles() {
    let support = std::fs::read_to_string(repo(SIGNING_SUITE_SUPPORT)).expect("read support");
    let spelled = format!("{REFERENCE_PROGRAM:?}");
    assert!(
        support.contains(&spelled),
        "{SIGNING_SUITE_SUPPORT} no longer writes {spelled}; both suites must compile \
         the program {FIXTURE} was emitted from"
    );
}

#[test]
fn the_binary_reports_this_package_version() {
    let s = Scratch::new();
    let out = s.mindc().arg("--version").output().expect("spawn mindc");
    assert_eq!(out.status.code(), Some(0), "--version: {}", text(&out));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let first = stdout.lines().next().unwrap_or("").trim_end();
    assert_eq!(first, format!("mind {}", env!("CARGO_PKG_VERSION")));
}

/// A build without both signing features must refuse the seeds, exit 1 and
/// write nothing: it can never emit an artifact that looks signed.
#[cfg(not(all(feature = "evidence-mldsa", feature = "evidence-slhdsa")))]
#[test]
fn without_both_signing_features_the_seeds_are_refused_with_no_artifact() {
    let s = Scratch::new();
    let out = s.emit("signed.mic3", BOTH_LEGS);
    assert_eq!(
        out.status.code(),
        Some(1),
        "seeds on a PQC-OFF build: {}",
        text(&out)
    );
    assert!(
        !s.path("signed.mic3").exists(),
        "a refusal must write no artifact"
    );
    assert!(
        text(&out).contains("signing requires the `evidence-"),
        "the refusal must name the missing feature: {}",
        text(&out)
    );
}

#[cfg(all(feature = "evidence-mldsa", feature = "evidence-slhdsa"))]
mod signing {
    use super::*;
    use libmind::ir::compact::v3::evidence::MAP_SENTINEL;
    use libmind::ir::compact::v3::{mldsa, parse_mic3_prefix, slhdsa};

    const SCHEME: &str = "pqc-hybrid-ml-dsa-87-slh-dsa-256s";

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex seed"))
            .collect()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn uleb_read(bytes: &[u8], at: &mut usize) -> usize {
        let (mut value, mut shift) = (0usize, 0u32);
        loop {
            let byte = bytes[*at];
            *at += 1;
            value |= usize::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
        }
    }

    fn uleb_write(mut value: usize, out: &mut Vec<u8>) {
        loop {
            let low = (value & 0x7f) as u8;
            value >>= 7;
            out.push(if value == 0 { low } else { low | 0x80 });
            if value == 0 {
                return;
            }
        }
    }

    /// (key, the entry's exact wire bytes). Values are skipped by their tag
    /// (0 string, 1 int, 2 bytes), never decoded, so a re-encode can only
    /// change what an edit removes.
    fn map_entries(map: &[u8]) -> Vec<(String, Vec<u8>)> {
        assert_eq!(map.first(), Some(&MAP_SENTINEL), "MAP epilogue sentinel");
        let mut at = 1;
        let count = uleb_read(map, &mut at);
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let start = at;
            let key_len = uleb_read(map, &mut at);
            let key = String::from_utf8(map[at..at + key_len].to_vec()).expect("utf-8 key");
            at += key_len;
            let tag = map[at];
            at += 1;
            let value = uleb_read(map, &mut at);
            match tag {
                0 | 2 => at += value,
                1 => {}
                other => panic!("unknown MAP value tag {other:#04x} for {key}"),
            }
            entries.push((key, map[start..at].to_vec()));
        }
        assert_eq!(at, map.len(), "the MAP epilogue must be consumed exactly");
        entries
    }

    fn encode(body: &[u8], entries: &[(String, Vec<u8>)]) -> Vec<u8> {
        let mut out = body.to_vec();
        out.push(MAP_SENTINEL);
        uleb_write(entries.len(), &mut out);
        for (_, wire) in entries {
            out.extend_from_slice(wire);
        }
        out
    }

    fn verify(s: &Scratch, artifact: &str, extra: &[&str]) -> Output {
        let mut cmd = s.mindc();
        cmd.arg("verify").arg(s.path(artifact)).args(extra);
        cmd.output().expect("spawn mindc verify")
    }

    /// Sign with the TEST seeds, verify under keys derived independently from
    /// those seeds, then strip either leg and require the verifier to refuse.
    #[test]
    fn test_seeds_sign_pinned_keys_verify_and_a_stripped_leg_is_refused() {
        let s = Scratch::new();
        let out = s.emit("signed.mic3", BOTH_LEGS);
        assert_eq!(out.status.code(), Some(0), "signing: {}", text(&out));
        assert!(
            text(&out).contains(SCHEME),
            "hybrid scheme tag: {}",
            text(&out)
        );
        let signed = std::fs::read(s.path("signed.mic3")).expect("read signed");

        let mut seed87 = [0u8; 32];
        seed87.copy_from_slice(&unhex(TEST_MLDSA87_SEED));
        let pk87 = hex(&mldsa::public_key_87(&seed87));
        let pk_slh = hex(&slhdsa::public_key(&unhex(TEST_SLHDSA_SEED)));
        let report = text(&verify(&s, "signed.mic3", &[]));
        assert!(
            report.contains(&pk87),
            "ML-DSA-87 key not the TEST seed's: {report}"
        );
        assert!(
            report.contains(&pk_slh),
            "SLH-DSA key not the TEST seed's: {report}"
        );
        let pinned = [
            "--require-signed",
            "--signer-pubkey",
            pk87.as_str(),
            "--signer-pubkey",
            pk_slh.as_str(),
        ];
        let ok = verify(&s, "signed.mic3", &pinned);
        assert_eq!(ok.status.code(), Some(0), "pinned verify: {}", text(&ok));

        let body_end = parse_mic3_prefix(&signed).expect("signed body").consumed;
        let unsigned = s.emit("plain.mic3", &[]);
        assert_eq!(
            unsigned.status.code(),
            Some(0),
            "unsigned: {}",
            text(&unsigned)
        );
        let plain = std::fs::read(s.path("plain.mic3")).expect("read unsigned");
        let plain_end = parse_mic3_prefix(&plain).expect("unsigned body").consumed;
        assert_eq!(plain_end, body_end, "signing changed the body length");
        assert_eq!(
            signed[..body_end],
            plain[..body_end],
            "signing moved a body byte"
        );
        let entries = map_entries(&signed[body_end..]);
        // Positive control: re-encoding every entry rebuilds the exact bytes.
        assert_eq!(
            encode(&signed[..body_end], &entries),
            signed,
            "re-encode drifted"
        );

        for leg in ["signature.slhdsa", "signature.mldsa87"] {
            let removed = [leg.to_string(), format!("{leg}_pubkey")];
            let kept: Vec<_> = entries
                .iter()
                .filter(|(k, _)| !removed.contains(k))
                .cloned()
                .collect();
            assert_eq!(
                kept.len() + 2,
                entries.len(),
                "{leg}: both entries present first"
            );
            let name = format!("strip_{leg}.mic3");
            std::fs::write(s.path(&name), encode(&signed[..body_end], &kept)).expect("write");
            let unpinned: &[&str] = &[];
            for extra in [unpinned, &pinned[..]] {
                let r = verify(&s, &name, extra);
                let msg = text(&r);
                println!(
                    "stripped {leg} pinned={}: rc={:?}",
                    !extra.is_empty(),
                    r.status.code()
                );
                assert_ne!(r.status.code(), Some(0), "stripped {leg} accepted: {msg}");
                assert!(msg.contains("trace_hash_valid: yes"), "body intact: {msg}");
            }
            let bare = text(&verify(&s, &name, &[]));
            assert!(
                bare.contains("signature is malformed"),
                "named refusal: {bare}"
            );
        }
    }
}
