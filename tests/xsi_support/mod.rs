// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Workload helpers of the cross-substrate identity gate.
//!
//! Only `cross_substrate_identity` builds embedded workloads and the
//! json-canon-kat driver, so these live beside it rather than in the harness
//! every integration test compiles.

pub mod json_canon_kat;

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::common::mindc_bin;

/// Write an embedded workload source into `dir`.
pub fn scratch_src_in(dir: &Path, name: &str, text: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
    p
}

/// Compile `src` to `dir/out_name` with `mindc --emit-shared`, quoting
/// mindc's OWN stdout and stderr in the panic when it fails.
///
/// Eight builders each hand-rolled the same `Command…status()` +
/// `assert!(status.success(), "…failed for the <x> workload")` pair. `status()`
/// INHERITS the child's stderr, which bypasses libtest's capture and lands
/// unattributed in the raw run output — never in the panic message, and never
/// beside the test that failed. `.output()` captures both streams and the
/// assertion quotes them, so the next occurrence names itself.
pub fn emit_shared_in(
    dir: &Path,
    workload: &str,
    src: &Path,
    out_name: &str,
    env: &[(&str, &str)],
) -> PathBuf {
    let out = dir.join(out_name);
    let mut cmd = Command::new(mindc_bin());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let r = cmd
        .args([
            src.to_str().unwrap(),
            "--emit-shared",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap_or_else(|e| panic!("spawn mindc --emit-shared for the {workload} workload: {e}"));
    assert!(
        r.status.success(),
        "mindc --emit-shared failed for the {workload} workload\n  \
         src:    {}\n  out:    {}\n  status: {}\n  stdout: {}\n  stderr: {}",
        src.display(),
        out.display(),
        r.status,
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr),
    );
    out
}
