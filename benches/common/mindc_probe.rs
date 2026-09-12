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

// Where a bench looks for the `mindc` binary, in probe order.
//
// Shared, pure, and bench-only — nothing in `src/` references it, so no emitted byte moves.
// It lives here rather than inside a single bench because every `harness = false` bench
// silently DISCARDS `#[test]` functions, so an ordering test written next to the probe would
// never run. `tests/mindc_probe_order.rs` includes this module and does run.

use std::path::{Path, PathBuf};

/// Every candidate location for `mindc`, most-preferred first.
///
/// Takes the `CARGO_TARGET_DIR` value rather than reading the environment, so the ORDER is
/// testable without standing up a second ~9 GB target directory.
///
/// **CARGO_TARGET_DIR FIRST, and that ordering is the fix.** The probe previously looked only
/// under `manifest_dir()/target/...`, so a run with `CARGO_TARGET_DIR=/dev/shm/...` — the normal
/// way to keep build artifacts off a full disk — put the binary exactly where the function did
/// not look. The bench then reported "kernel unavailable; no measurements taken" and exited 0,
/// so two separate attempts to obtain a roofline number produced nothing and looked like an
/// environment problem rather than a harness defect.
pub fn mindc_candidates(cargo_target_dir: Option<&str>, manifest: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = cargo_target_dir {
        let base = PathBuf::from(dir);
        out.push(base.join("release").join("mindc"));
        out.push(base.join("debug").join("mindc"));
    }
    out.push(manifest.join("target").join("debug").join("mindc"));
    out.push(manifest.join("target").join("release").join("mindc"));
    // Deduplicate, preserving probe order. `CARGO_TARGET_DIR` is very often set to exactly
    // `<manifest>/target` — that IS cargo's default location — in which case every candidate
    // appears twice and the probe pays two redundant `is_file()` stats on the hot path for no
    // new information. Found by this module's own duplicate-free test, which failed on the
    // `Some("/repo/target")` case.
    out.dedup_by(|a, b| a == b);
    let mut seen = Vec::new();
    out.retain(|p| {
        if seen.contains(p) {
            false
        } else {
            seen.push(p.clone());
            true
        }
    });
    out
}
