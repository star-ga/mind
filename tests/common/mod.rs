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
//! during the test binary's own compilation, and `cargo test` always rebuilds
//! `mindc` before running tests if any source changed.
//!
//! Reference: <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>
//! The `[[bin]] name = "mindc"` entry in `Cargo.toml` guarantees
//! `CARGO_BIN_EXE_mindc` is available to every integration test in this crate.

use std::path::PathBuf;

/// Return the path to the `mindc` binary for the CURRENT test-profile and
/// feature set.
///
/// Uses `env!("CARGO_BIN_EXE_mindc")` which is set by cargo at test-binary
/// compile time to the binary built alongside this test run — always correct,
/// never stale. No filesystem probing is required or performed.
///
/// Callers that tolerate a missing binary (skip semantics) should pair this
/// with an `.exists()` check; callers that require the binary should
/// `assert!(mindc_bin().exists(), "…build instructions…")`.
#[allow(dead_code)]
pub fn mindc_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
}

/// Return `Some(path)` if the mindc binary exists, or `None` (suitable for
/// soft-skip inside a `#[test]`) if it does not.
///
/// A missing binary is only expected when the test is invoked without having
/// built mindc first (e.g. `cargo test --no-run` followed by manual deletion).
/// Under normal `cargo test` usage `CARGO_BIN_EXE_mindc` always points at a
/// freshly built binary.
#[allow(dead_code)]
pub fn mindc_or_skip() -> Option<PathBuf> {
    let p = mindc_bin();
    if p.exists() { Some(p) } else { None }
}
