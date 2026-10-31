// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Executable regression gate for `__mind_open(path)` negative error returns (#290).
//!
//! Writes `tests/open_error_kat.mind` into an isolated scratch project, builds
//! both `debug` and `release` binaries through `common::gate::compiled(...)`
//! (fail-closed under `MIND_BENCH_REQUIRE=1`), and runs the compiled artifact
//! against:
//!   1. a null path (`__mind_open(0) < 0`),
//!   2. a missing path under an owned directory (`__mind_open(absent) < 0`),
//!   3. a path whose intermediate component is a regular file (`__mind_open(existing/child) < 0`),
//!   4. an existing 1-byte file (`"M"`) read to EOF (`__mind_open(existing) >= 0`).
//!
//! Before `fcb44b97` (#290), `runtime-support/mind_intrinsics.c` returned the
//! positive sentinel `999` on a failed `open()`, causing `tests/open_error_kat.mind`
//! to fail at `if __mind_open(absent) >= 0` with exit `1` and stdout `2\n`.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::path::Path;
use std::process::Command;

const TARGET: &str = "open_error_kat";

#[test]
fn open_error_kat_preserves_negative_open_errors_in_debug_and_release() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kat_path = repo_root.join("tests").join("open_error_kat.mind");
    let kat_src = std::fs::read_to_string(&kat_path).expect("read tests/open_error_kat.mind");

    let base_dir = common::scratch_dir(TARGET);
    let _ = std::fs::remove_dir_all(&base_dir);
    let fixture_dir = base_dir.join("fixture");
    std::fs::create_dir_all(&fixture_dir).expect("create fixture dir");

    let existing = fixture_dir.join("existing");
    std::fs::write(&existing, b"M").expect("write 1-byte fixture file");
    let missing = fixture_dir.join("missing");
    assert!(!missing.exists(), "missing fixture path must not exist");
    let not_directory = existing.join("child");

    for profile in ["debug", "release"] {
        let project_dir = base_dir.join(profile);
        std::fs::create_dir_all(&project_dir).expect("create profile project dir");
        std::fs::write(project_dir.join("main.mind"), &kat_src).expect("write main.mind");
        std::fs::write(
            project_dir.join("Mind.toml"),
            "[package]\nname = \"open_error_regression\"\nversion = \"0.0.1\"\n\n[build]\nentry = \"main.mind\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
        )
        .expect("write Mind.toml");

        let bin_path = project_dir.join("program");
        let build_out = Command::new(common::mindc_bin())
            .current_dir(&project_dir)
            .env("MINDC_RUNTIME_OBJ_CACHE", "0")
            .args([
                "build",
                "--backend",
                "mlir",
                "--target",
                "cpu",
                "--emit",
                "binary",
                "--optimize",
                profile,
                "--no-cache",
                "--out",
                "program",
            ])
            .output()
            .expect("spawn mindc build");

        if !common::gate::compiled(TARGET, &build_out) {
            return;
        }
        assert!(
            bin_path.is_file(),
            "{TARGET} ({profile}): mindc build succeeded without producing {}",
            bin_path.display()
        );

        // Negative control: wrong argc must return 64.
        let bad_argc = Command::new(&bin_path)
            .output()
            .expect("run compiled binary with no arguments");
        assert_eq!(
            bad_argc.status.code(),
            Some(64),
            "{TARGET} ({profile}): argc != 4 control must exit 64"
        );

        let run = Command::new(&bin_path)
            .arg(&existing)
            .arg(&missing)
            .arg(&not_directory)
            .output()
            .expect("run compiled open_error_kat binary");

        let stdout = String::from_utf8_lossy(&run.stdout);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert_eq!(
            run.status.code(),
            Some(0),
            "{TARGET} ({profile}): expected exit 0, got {:?} (stdout={stdout:?}, stderr={stderr:?})",
            run.status.code()
        );
        assert_eq!(
            stdout.as_ref(),
            "0\n",
            "{TARGET} ({profile}): expected stdout \"0\\n\", got {stdout:?}"
        );
        assert!(
            stderr.is_empty(),
            "{TARGET} ({profile}): expected empty stderr, got {stderr:?}"
        );
    }
}
