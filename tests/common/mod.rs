// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Shared test-harness helpers.
//!
//! # Staleness elimination (issue #42)
//!
//! The historic per-file `mindc_bin()` helper resolved the mindc binary by
//! preferring `target/debug/mindc` over `target/release/mindc`. A stale
//! `target/debug/mindc` — left over from an unrelated `cargo test` run that
//! did NOT compile with `--features mlir-build` — caused integration tests to
//! shell out to a WRONG-VERSION binary, producing pre-fix behaviour, parse
//! errors, or phantom bugs (false positives). Each such false positive
//! triggered a debugging chase against a problem that no longer existed.
//!
//! The fix: use `env!("CARGO_BIN_EXE_mindc")`. Cargo sets this environment
//! variable at *compile time* of the test binary itself, pointing at the
//! `mindc` binary that was built for the SAME invocation, SAME profile, and
//! SAME feature set. There is no staleness possible: the path is baked in
//! during the test binary's own compilation. Cargo rebuilds `mindc` before
//! running tests, but another Cargo invocation sharing that target directory
//! can replace it while tests execute. Keep the target exclusive for the full
//! test lifetime or use an isolated target per concurrent invocation.
//!
//! Reference: <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>
//! The `[[bin]] name = "mindc"` entry in `Cargo.toml` guarantees
//! `CARGO_BIN_EXE_mindc` is available to every integration test in this crate.

pub mod gate;
pub mod xsi_gate;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Return the path to the `mindc` binary for the CURRENT test-profile and
/// feature set.
///
/// Uses `env!("CARGO_BIN_EXE_mindc")` which is set by cargo at test-binary
/// compile time to the binary built alongside this test run. Keep that target
/// directory exclusive until tests finish; the path does not freeze its bytes.
/// No filesystem probing is required or performed.
///
/// Callers that tolerate a missing binary (skip semantics) should pair this
/// with an `.exists()` check; callers that require the binary should
/// `assert!(mindc_bin().exists(), "…build instructions…")`.
#[allow(dead_code)]
pub fn mindc_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
}

/// A scratch directory private to ONE test target in ONE process.
///
/// The std smokes wrote their `.so` to a FIXED shared path
/// (`std::env::temp_dir().join("mind_<x>_smoke.so")`). On a machine running two
/// test processes at once — two agents, two CI jobs on one runner, a `cargo
/// test` beside a preflight — one process's `mindc` truncates the artifact
/// another has just `dlopen`ed, and the flake reads as a compiler regression.
/// The path is also world-writable and predictable, so another user can
/// pre-create it as a symlink and redirect the write.
///
/// The directory carries the target name and the pid; the FILE names inside are
/// left alone so an artifact's identity is unchanged.
///
/// deferred: the remaining shared-root artifact paths and their per-file
/// counts are recorded by the gate below. The two wedge gates are routed:
/// `cross_substrate_identity` (9 sites) and `phase_g_keystone_bootstrap` (8).
/// The gate in
/// `tests/harness_scratch_isolation.rs` records the backlog as a SET with
/// per-file counts and fails in BOTH directions, so a new shared path is red
/// and a row that outlived its debt is red. Upgrade path: route a file through
/// this helper as it is next touched, keeping the file names, and delete its
/// row in that gate — one helper, no per-file policy.
#[allow(dead_code)]
pub fn scratch_dir(target: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mind-{target}-{}", std::process::id()));
    std::fs::create_dir_all(&d)
        .unwrap_or_else(|e| panic!("create scratch dir {}: {e}", d.display()));
    d
}

/// Return the path to the `mind` CLI binary for the CURRENT test-profile and
/// feature set.
///
/// Same staleness argument as [`mindc_bin`], and the same defect it was written
/// for. Nine CLI/exec harnesses each hand-rolled
/// `CARGO_MANIFEST_DIR/target/{debug,release}/mind`, which is wrong twice over:
/// it ignores `CARGO_TARGET_DIR` (so the probe misses entirely whenever the
/// build is directed elsewhere) and it names a profile directory rather than the
/// binary cargo actually built for this run. The miss then landed on a
/// `println!("skipping"); return;`, so nine CLI gates asserted nothing and
/// reported green. `CARGO_BIN_EXE_mind` is baked in at the test binary's own
/// compile time by the `[[bin]] name = "mind"` entry and cannot be stale.
#[allow(dead_code)]
pub fn mind_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mind"))
}

/// The `mindc` binary, or `None` once its absence has been ROUTED through the
/// one fail-closed skip gate for `target`.
///
/// This helper existed with no caller and no routing: it handed back a bare
/// `None` and left every call site to invent its own announcement, which is the
/// fail-open shape `tests/fail_open_skip_site_ratchet.rs` exists to forbid. It
/// is wired now rather than deleted — the capability (tolerate a missing binary)
/// is wanted; the silence was the defect. Under `MIND_BENCH_REQUIRE=1` the skip
/// is a hard failure, and otherwise it prints the `ran=0` marker the tier log
/// reads.
///
/// Use [`require_mindc`] instead wherever absence is not tolerable — which is
/// most places, since `CARGO_BIN_EXE_mindc` is built by cargo for the very test
/// target that reads it.
#[allow(dead_code)]
pub fn mindc_or_skip(target: &str) -> Option<PathBuf> {
    let bin = mindc_bin();
    if bin.exists() {
        return Some(bin);
    }
    gate::skipped(target, &format!("mindc not found at {bin:?}"));
    None
}

/// The `mindc` binary, asserting it is there.
///
/// NO early return on absence. [`mindc_bin`] resolves `CARGO_BIN_EXE_mindc`,
/// which cargo builds for this test target before it runs, so the binary cannot
/// legitimately be missing: an `Option` probe here ANNOUNCED its skip and handed
/// the caller `None`, which every call site turned into a bare `return` — a
/// broken harness graded as a silent pass.
///
/// Five test targets carried this assert verbatim. One owner now, so a change to
/// what "the binary is missing" means cannot land in four files and miss the
/// fifth.
#[allow(dead_code)]
pub fn require_mindc() -> PathBuf {
    let bin = mindc_bin();
    assert!(
        bin.exists(),
        "mindc binary missing at {bin:?}; CARGO_BIN_EXE_mindc is built by cargo \
         for this target, so its absence is a broken gate, not a skip"
    );
    bin
}

/// Run `mindc build` in `dir` with stderr and stdout CAPTURED.
///
/// Capture is the point: a harness that inherits stdio discards the diagnostic
/// and is then only able to treat every failure as "backend unavailable", which
/// is how a real compiler regression came to grade as a skip.
#[allow(dead_code)]
pub fn run_build_captured(mindc: &Path, dir: &Path, extra_args: &[&str]) -> Output {
    run_build_captured_env(mindc, dir, extra_args, &[])
}

/// As [`run_build_captured`], with environment overrides applied to the child.
///
/// The overrides are how a harness DRIVES a host-capability path instead of
/// waiting for a host that happens to be in it. Both of the ones used today are
/// documented compiler inputs, not harness back doors: `MLIR_OPT` is the
/// override `eval::mlir_build::resolve_tools` reads before it searches `PATH`,
/// and `MIND_LIB_DIR` is the runtime-library override `project::find_runtime_lib`
/// reads before `~/.mind/lib`. A path that does not resolve therefore produces
/// the SAME refusal a host genuinely missing the tool produces, by the same code
/// path, rather than a state only a test can be in.
///
/// The child is still spawned in exactly ONE place, so a change to how it is
/// launched (capture, working directory, argument order) cannot land in one
/// helper and miss the other.
#[allow(dead_code)]
pub fn run_build_captured_env(
    mindc: &Path,
    dir: &Path,
    extra_args: &[&str],
    env: &[(&str, &Path)],
) -> Output {
    let mut cmd = Command::new(mindc);
    cmd.arg("build").args(extra_args).current_dir(dir);
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.output().expect("failed to spawn mindc")
}

/// The artifact path the builder ITSELF reported for this run.
///
/// `mindc build` prints `   Finished <target> [<emit>] <path>` naming the file
/// it wrote (`src/bin/mindc.rs`). Reading that back is what keeps a harness from
/// hand-typing a SECOND copy of the artifact-naming rule. There is one rule now
/// — `project::artifact_stem`, pinned by `tests/mindc_artifact_name.rs` — but
/// there were two that disagreed: the native builder named a `binary` after
/// `package.name` while the launcher fallback named it after `[build] output`,
/// whose serde default was `"app"`. Asserting the launcher's name against the
/// native builder's file is exactly how a revived byte-identity check came to
/// panic on every host that HAS the backend — the one tier it exists for.
/// Reading the reported path stays right regardless of which name the rule
/// yields, and it was itself hand-copied into two harnesses before this.
///
/// There is no fallback when the line is absent: a build that exits 0 without
/// naming its artifact is a broken gate, not a skip.
#[allow(dead_code)]
pub fn reported_artifact(out: &Output) -> PathBuf {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let path = stdout
        .lines()
        .find_map(|line| {
            let rest = line.trim_start().strip_prefix("Finished ")?;
            // `Finished <target> [<emit>] <path>`: neither the target nor the
            // emit token contains `]`, so the first `] ` closes the emit bracket
            // and the remainder is the path verbatim.
            rest.split_once("] ").map(|(_, p)| p.trim())
        })
        .unwrap_or_else(|| {
            panic!(
                "`mindc build` exited 0 but printed no `Finished <target> \
                 [<emit>] <path>` line naming its artifact, so the caller has \
                 nothing to read.\n--- stdout ---\n{stdout}"
            )
        });
    PathBuf::from(path)
}
