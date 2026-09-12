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

//! `CARGO_TARGET_DIR` must be probed for `mindc` BEFORE the manifest target directory.
//!
//! WHAT THIS REPLACES, AND WHY IT IS A SEPARATE TEST TARGET. The bench's probe looked only
//! under `manifest_dir()/target/{debug,release}`, so a run with `CARGO_TARGET_DIR` set — the
//! normal way to keep build artifacts off a full disk — searched the one place the binary was
//! not. Combined with the bench's then-silent `exit 0` on zero measurements, two attempts to
//! obtain a roofline number produced nothing and read as an environment problem.
//!
//! Two earlier attempts to gate this both failed, and both failures are the reason this file
//! looks the way it does:
//!
//! 1. **A shell mutation test could not discriminate.** Pointing `CARGO_TARGET_DIR` at a
//!    scratch directory also redirects cargo's own build output there, so cargo rebuilds from
//!    scratch instead of finding the planted binary; pointing it at the real target directory
//!    makes both candidate paths the same file. An honest end-to-end demonstration needs a
//!    second complete ~9 GB target directory, which this box's disk budget does not allow.
//!
//! 2. **A `#[test]` inside the bench never ran.** `det_matmul_i16` is declared
//!    `harness = false` in `Cargo.toml`, so cargo DISCARDS `#[test]` functions in it. The test
//!    compiled, reported nothing, and would have been claimed as verification — a vacuous gate
//!    of exactly the kind the repo's own discipline warns about.
//!
//! So the probe order moved into `benches/common/mindc_probe.rs` as a pure function over an
//! injected env value, and this real test target includes it. Bench-only: nothing in `src/`
//! references it, so no emitted byte moves and no canary is touched.

#[allow(dead_code)]
mod probe {
    include!("../benches/common/mindc_probe.rs");
}

use probe::mindc_candidates;
use std::path::{Path, PathBuf};

/// With `CARGO_TARGET_DIR` set, every candidate under it precedes every manifest candidate.
#[test]
fn cargo_target_dir_is_probed_before_the_manifest_target() {
    let manifest = Path::new("/repo");
    let got = mindc_candidates(Some("/elsewhere"), manifest);

    assert_eq!(
        got.first().map(PathBuf::as_path),
        Some(Path::new("/elsewhere/release/mindc")),
        "CARGO_TARGET_DIR/release must be the FIRST candidate; got {got:?}"
    );

    let last_env = got.iter().rposition(|p| p.starts_with("/elsewhere"));
    let first_manifest = got.iter().position(|p| p.starts_with("/repo"));
    assert!(
        last_env < first_manifest,
        "every CARGO_TARGET_DIR candidate must precede every manifest candidate: {got:?}"
    );

    assert!(
        got.contains(&PathBuf::from("/elsewhere/debug/mindc")),
        "the debug profile under CARGO_TARGET_DIR must also be probed: {got:?}"
    );
}

/// NEGATIVE CONTROL. Unset, the manifest paths are the ONLY candidates.
///
/// Without this, the test above would still pass if `mindc_candidates` unconditionally prepended
/// `/elsewhere` regardless of its argument — so this is what makes the assertions above about
/// ORDERING rather than about a string appearing somewhere.
#[test]
fn with_cargo_target_dir_unset_only_the_manifest_paths_are_probed() {
    let manifest = Path::new("/repo");
    let got = mindc_candidates(None, manifest);

    assert!(
        got.iter().all(|p| p.starts_with("/repo")),
        "with CARGO_TARGET_DIR unset, no candidate may come from anywhere else: {got:?}"
    );
    assert_eq!(
        got,
        vec![
            PathBuf::from("/repo/target/debug/mindc"),
            PathBuf::from("/repo/target/release/mindc"),
        ],
        "unset must yield exactly the two manifest candidates, debug first: {got:?}"
    );
}

/// The candidate list must never be empty, and must never contain duplicates.
///
/// A duplicate would make the probe do redundant filesystem work on every call; an empty list
/// would make `mindc_path()` return `None` unconditionally, which is the failure mode that
/// produced "kernel unavailable" in the first place.
#[test]
fn the_candidate_list_is_non_empty_and_duplicate_free() {
    for env in [None, Some("/elsewhere"), Some("/repo/target")] {
        let got = mindc_candidates(env, Path::new("/repo"));
        assert!(!got.is_empty(), "candidate list must never be empty (env={env:?})");
        let mut seen = got.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(
            seen.len(),
            got.len(),
            "candidate list must not repeat a path (env={env:?}): {got:?}"
        );
    }
}
