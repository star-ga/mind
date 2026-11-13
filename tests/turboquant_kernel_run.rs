// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! `bench/turboquant.mind` must BUILD AND RUN, and its own numeric assertions
//! must pass.
//!
//! The kernel states its own standard in its header: "compiling is not passing;
//! the assertions passing is." It checks its numerics at run time against
//! independently derived expected values and returns nonzero on mismatch.
//!
//! Until this file existed, nothing executed it. 4c79ea36 committed the kernel
//! having verified only `--emit-mic3` and `mindc check`; 56c99b47 then ran it by
//! hand once and recorded the result as a dated comment. An audit called that
//! what it is — text-keyed closure of an execution finding: a comment cannot
//! notice the day the kernel's numerics break.
//!
//! It is built in an ISOLATED temp project rather than in-repo on purpose. A repo
//! build compiles every bench source, and `bench/matmul_det_bench.mind` calls
//! `__mind_now_ns` — a deliberately unregistered World intrinsic (it reads a
//! clock) — so it cannot compile natively and the #244 fail-closed guard
//! correctly refuses the whole build. That is a property of the sibling, not of
//! this kernel, and isolating the entry is what separates the two.

#![cfg(feature = "mlir-build")]

mod common;

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The marker the kernel prints only after every assertion has passed.
const PASS_MARKER: &str = "TURBOQUANT PIPELINE PASS";

#[test]
fn turboquant_builds_and_its_own_assertions_pass() {
    let mindc = PathBuf::from(env!("CARGO_BIN_EXE_mindc"));
    let kernel = repo_root().join("bench/turboquant.mind");
    assert!(kernel.is_file(), "missing {}", kernel.display());

    let dir = std::env::temp_dir().join(format!("mind_turboquant_run_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("create temp project");
    std::fs::copy(&kernel, dir.join("src/main.mind")).expect("copy kernel");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"turboquant\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n",
    )
    .expect("write manifest");

    let out_bin = dir.join("turboquant");
    let build = Command::new(&mindc)
        .args([
            "build",
            "--backend=mlir",
            "--release",
            "--emit=binary",
            &format!("--out={}", out_bin.display()),
        ])
        .current_dir(&dir)
        .output()
        .expect("spawn mindc build");

    let bstderr = String::from_utf8_lossy(&build.stderr);
    // Environmental only: no MLIR toolchain on this runner. Anything else is a
    // real failure and must not be swallowed -- this test exists because a
    // silent non-execution is what it is defending against.
    if !build.status.success()
        && !crate::common::gate::enforce_real_backend()
        && crate::common::gate::is_capability_gap(&bstderr)
    {
        crate::common::gate::skipped("turboquant_kernel_run", "mindc lacks the MLIR toolchain");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(
        build.status.success(),
        "turboquant must BUILD.\nstderr: {bstderr}"
    );
    assert!(out_bin.is_file(), "no binary at {}", out_bin.display());

    let run = Command::new(&out_bin).output().expect("run turboquant");
    let stdout = String::from_utf8_lossy(&run.stdout);

    assert_eq!(
        run.status.code(),
        Some(0),
        "turboquant's own numeric assertions FAILED (it returns nonzero on \
         mismatch).\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    // Exit 0 alone is not proof the pipeline ran: the marker is printed only
    // after every stage's assertions have passed.
    // EXACT line, not `contains`. A substring check is satisfied by a CORRUPTED
    // marker -- "TURBOQUANT PIPELINE PASSX" contains "TURBOQUANT PIPELINE PASS" --
    // so mutating the marker slipped through this assertion even though the kernel
    // was genuinely rebuilt and rerun. Caught by mutation-testing this very gate.
    assert!(
        stdout.lines().any(|l| l.trim() == PASS_MARKER),
        "expected a line exactly equal to {PASS_MARKER:?} — exit 0 without it means \
         the pipeline did not reach the end.\nstdout:\n{stdout}"
    );
    for stage in ["stage 1", "stage 2", "stage 3"] {
        assert!(
            stdout.contains(stage),
            "{stage} did not report; the pipeline is not running end to end.\n{stdout}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
