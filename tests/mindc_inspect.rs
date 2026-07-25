//! Integration test for `mindc inspect` — the mic@3 artifact decoder/differ.
//!
//! `inspect` is the consumer/debug counterpart of `--emit-mic3`: it decodes a
//! mic@3 artifact into a structural summary + the canonical IR + (when present)
//! the evidence chain, and with `--diff` pinpoints the FIRST diverging byte
//! between two artifacts — the tool the self-host byte-identity gates need when
//! a reseed/loop stops being byte-identical. These tests shell out to the built
//! binary so they exercise the real CLI end to end.

use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn emit_evidence(src_text: &str, tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let src = dir.join(format!("mindc_inspect_{tag}.mind"));
    let art = dir.join(format!("mindc_inspect_{tag}.mic3"));
    std::fs::write(&src, src_text).unwrap();
    let out = mindc()
        .arg(&src)
        .arg("--emit-evidence")
        .arg(&art)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "emit-evidence failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    art
}

#[test]
fn inspect_reports_summary_and_evidence() {
    let art = emit_evidence("fn main() -> i64 { 42 }\n", "sum");

    let out = mindc().arg("inspect").arg(&art).output().unwrap();
    assert!(out.status.success(), "inspect exited non-zero");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("instrs:"), "missing structural summary:\n{s}");
    assert!(s.contains("trace_hash:"), "missing evidence chain:\n{s}");
    assert!(
        s.contains("determinism:      deterministic"),
        "missing determinism:\n{s}"
    );
    assert!(s.contains("--- canonical IR ---"), "missing IR body:\n{s}");

    // --json surfaces the same fields, charset-safe.
    let j = mindc()
        .arg("inspect")
        .arg(&art)
        .arg("--json")
        .output()
        .unwrap();
    assert!(j.status.success());
    let js = String::from_utf8_lossy(&j.stdout);
    assert!(
        js.contains("\"attested\":true") && js.contains("\"trace_hash\":\""),
        "bad json: {js}"
    );
}

#[test]
fn inspect_diff_detects_divergence_and_identity() {
    let a = emit_evidence("fn main() -> i64 { 42 }\n", "da");
    let b = emit_evidence("fn main() -> i64 { 43 }\n", "db");

    // Identical artifacts → exit 0, "identical: YES".
    let same = mindc()
        .arg("inspect")
        .arg(&a)
        .arg("--diff")
        .arg(&a)
        .output()
        .unwrap();
    assert_eq!(same.status.code(), Some(0), "self-diff must be identical");
    assert!(String::from_utf8_lossy(&same.stdout).contains("identical:  YES"));

    // Differing artifacts → exit 1, reports the first diverging byte.
    let diff = mindc()
        .arg("inspect")
        .arg(&a)
        .arg("--diff")
        .arg(&b)
        .output()
        .unwrap();
    assert_eq!(
        diff.status.code(),
        Some(1),
        "differing artifacts must exit 1"
    );
    let s = String::from_utf8_lossy(&diff.stdout);
    assert!(
        s.contains("identical:        NO"),
        "missing NO verdict:\n{s}"
    );
    assert!(s.contains("first_diff_byte:"), "missing byte offset:\n{s}");
}

#[test]
fn inspect_malformed_artifact_exits_1_not_panic() {
    let dir = std::env::temp_dir();
    let bad = dir.join("mindc_inspect_bad.mic3");
    std::fs::write(&bad, b"not a mic3 artifact at all").unwrap();
    let out = mindc().arg("inspect").arg(&bad).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "malformed artifact must fail closed, not panic"
    );

    // --json parse-error path still emits a well-formed object to stdout (like verify).
    let j = mindc()
        .arg("inspect")
        .arg(&bad)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(j.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&j.stdout).contains("\"error\":\""),
        "json parse-error must emit an error object to stdout"
    );

    // --diff of a byte-identical GARBAGE artifact must NOT report "identical: YES"
    // (exit 0) — decodability is part of the exit-0 contract, so it fails closed.
    let self_diff = mindc()
        .arg("inspect")
        .arg(&bad)
        .arg("--diff")
        .arg(&bad)
        .output()
        .unwrap();
    assert_eq!(
        self_diff.status.code(),
        Some(1),
        "byte-identical garbage must not exit 0"
    );
}
