// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Signed i64 `/` and `%` compute the SAME value on the MLIR backend that the
//! frozen-profile native ELF computes — over the shared edge corpus, not over a
//! restatement of it.
//!
//! # Why this test exists
//!
//! `src/ir/frozen_profile.rs` admitted signed i64 Div/Mod to the frozen native profile
//! on 2026-09-16, on the strength of the native emitter being edge-correct. "Native is
//! correct" is a claim about ONE backend; the wedge claim is that the backends AGREE.
//! The native leg is asserted by `examples/mindc_mind/ri_d1_frozen_profile_gate.py`
//! (build with `--backend native`, zero toolchain, run, check the exit code). This is
//! the other leg: the same programs, the same expected exit codes, through the MLIR
//! pipeline. Together they are `native == pinned == MLIR`.
//!
//! # Why the corpus is a file and not a list in this source
//!
//! All three legs read `examples/mindc_mind/testdata/signed_div_mod_edge_corpus.tsv`
//! (native gate, this test, and `tests/signed_div_mod_interpreter_corpus_run.rs`). Two
//! hand-maintained lists asserted to agree are a claim about two lists — this repo's
//! most expensive recurring defect class, and one that a passing test suite never
//! catches, because both lists keep passing while they drift apart. A single corpus
//! makes the agreement structural: a fixture can only be edited for every tier at once.
//! (The IR preview evaluator `eval_ir` does not call `main`, so it cannot run these
//! programs; its arms are held to the same contract by its own unit test, which is a
//! separate operand list and is NOT structurally coupled to this corpus.)
//!
//! # The contract under test
//!
//! * truncation toward zero; the remainder takes the sign of the DIVIDEND
//! * `x / 0 == 0` and `x % 0 == 0` — guarded, never a `#DE` / SIGFPE trap
//! * `i64::MIN / -1 == i64::MIN`, `i64::MIN % -1 == 0` — the defined two's-complement
//!   wrap (both backends substitute divisor 1 rather than trapping)
//!
//! A trap is distinguishable from a wrong answer here and is reported as such: a
//! process killed by a signal has no exit status, so the harness reports the signal
//! instead of silently reading it as some integer.
//!
//! Gate: `cargo test --release --features "std-surface mlir-build cross-module-imports"
//!        --test signed_div_mod_cross_backend_parity_run`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::mindc_bin;

use std::os::unix::process::ExitStatusExt as _;
use std::path::PathBuf;
use std::process::Command;

/// Exact row count of the shared corpus, pinned in every reader.
const CORPUS_ROWS: usize = 13;

/// Opt-out for a host that compiles `mlir-build` but has no MLIR toolchain on PATH.
/// Without it a missing toolchain is a FAILURE: a skip that prints to a captured
/// stdout is indistinguishable from a pass, which is the vacuous-green this corpus
/// exists to rule out.
const ALLOW_SKIP_ENV: &str = "MIND_DIVMOD_PARITY_ALLOW_SKIP";

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/mindc_mind/testdata")
}

/// `(name, source, expected_exit)` for every fixture in the shared corpus.
///
/// Parse failures are a hard panic rather than a skip: a corpus this test cannot read
/// is a corpus it is not testing, and "0 fixtures ran" must never be able to look like
/// a pass (see `~/.claude/rules/common/wiring-discipline.md` rule 9).
fn load_corpus() -> Vec<(String, String, i32)> {
    let dir = corpus_dir();
    let path = dir.join("signed_div_mod_edge_corpus.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read the shared corpus at {}: {e}", path.display()));
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = raw.split('\t').collect();
        assert_eq!(
            parts.len(),
            4,
            "{}:{}: expected 4 TAB-separated fields, got {}",
            path.display(),
            i + 1,
            parts.len()
        );
        let name = parts[0].trim().to_string();
        let expect: i32 = parts[1]
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("{}:{}: bad expected exit: {e}", path.display(), i + 1));
        let src = match parts[2].trim() {
            "src" => parts[3].trim().to_string(),
            "file" => {
                let f = dir.join(parts[3].trim());
                std::fs::read_to_string(&f)
                    .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", f.display()))
            }
            other => panic!("{}:{}: unknown kind '{other}'", path.display(), i + 1),
        };
        out.push((name, src, expect));
    }
    assert_eq!(
        out.len(),
        CORPUS_ROWS,
        "shared corpus has {} rows, pinned at {CORPUS_ROWS} — a deleted row must fail, \
         not shrink",
        out.len()
    );
    out
}

/// True when this build can actually drive the MLIR pipeline. Probes with a program
/// containing NO division, so a failure can only mean "no toolchain", never "the thing
/// under test is broken" — a probe that the bug could also fail is not a probe.
fn mlir_build_available(mindc: &PathBuf, dir: &std::path::Path) -> bool {
    if !mindc.exists() {
        return false;
    }
    let s = dir.join("parity_probe.mind");
    if std::fs::write(&s, "fn main()->i64{return 0;}\n").is_err() {
        return false;
    }
    Command::new(mindc)
        .args([
            s.to_str().unwrap(),
            "--emit-shared",
            dir.join("parity_probe.so").to_str().unwrap(),
        ])
        .output()
        .map(|o| {
            let e = String::from_utf8_lossy(&o.stderr).to_string();
            !(e.contains("mlir-build") && e.contains("requires"))
        })
        .unwrap_or(false)
}

#[test]
fn signed_div_mod_edges_agree_between_the_mlir_backend_and_the_pinned_native_values() {
    let mindc = mindc_bin();
    let tmp = std::env::temp_dir().join("mind_divmod_parity");
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    // Load (and length-check) the corpus BEFORE deciding to skip, so even an
    // opted-out run still proves the shared file is intact.
    let corpus = load_corpus();

    if !mlir_build_available(&mindc, &tmp) {
        assert!(
            std::env::var_os(ALLOW_SKIP_ENV).is_some(),
            "div/mod parity: this mindc cannot drive the MLIR pipeline (no mlir-opt / \
             mlir-translate / clang), so 0 of {CORPUS_ROWS} fixtures would run. Refusing to \
             report that as a pass. Install the toolchain, or set {ALLOW_SKIP_ENV}=1 to skip \
             deliberately."
        );
        eprintln!("div/mod parity: SKIPPED by {ALLOW_SKIP_ENV} — 0/{CORPUS_ROWS} fixtures ran");
        return;
    }

    let mut failures = Vec::new();
    for (name, src, expect) in &corpus {
        let s = tmp.join(format!("{name}.mind"));
        let exe = tmp.join(format!("{name}.exe"));
        let _ = std::fs::remove_file(&exe);
        std::fs::write(&s, src).expect("write fixture");

        // The DEFAULT backend — i.e. the MLIR pipeline — driven exactly as the native
        // gate drives its own leg, so the only difference between the two legs is the
        // backend and never the way the program is invoked or its result is read.
        let build = Command::new(&mindc)
            .args(["build", s.to_str().unwrap(), "--out", exe.to_str().unwrap()])
            .output()
            .expect("run mindc build");
        if !build.status.success() || !exe.exists() {
            failures.push(format!(
                "{name}: MLIR build produced no artifact\n    {}",
                String::from_utf8_lossy(&build.stderr).trim()
            ));
            continue;
        }

        let status = Command::new(&exe).status().expect("run artifact");
        match (status.code(), status.signal()) {
            (Some(got), _) if got == *expect => {}
            (Some(got), _) => failures.push(format!(
                "{name}: MLIR artifact exit {got}, expected {expect} (the value the \
                 frozen-profile native ELF is pinned to in ri_d1_frozen_profile_gate.py) \
                 — the backends DISAGREE"
            )),
            (None, Some(sig)) => failures.push(format!(
                "{name}: MLIR artifact DIED BY SIGNAL {sig} (expected exit {expect}) — a \
                 trap is not a value; signal 8 here is the unguarded-idiv #DE this \
                 contract exists to rule out"
            )),
            (None, None) => failures.push(format!("{name}: artifact neither exited nor signalled")),
        }
    }

    assert!(
        failures.is_empty(),
        "signed div/mod cross-backend parity failed for {}/{} fixtures:\n  - {}",
        failures.len(),
        corpus.len(),
        failures.join("\n  - ")
    );
    println!(
        "div/mod parity: {}/{} fixtures agree between the MLIR backend and the pinned \
         native values",
        corpus.len(),
        corpus.len()
    );
}
