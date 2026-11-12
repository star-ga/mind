// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Cross-process coverage for project build transactions.

mod common;

use common::require_mindc;
use libmind::build::cache::{CacheProbe, cache_root, probe};
use libmind::build::{CacheKeyFlags, compile_cache_material};
use libmind::project::{BuildTarget, EmitKind, OptimizeLevel};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::thread;

const SIMPLE_MIND: &str = "fn main() -> i64 { 42 }\n";
const MODIFIED_MIND: &str = "fn main() -> i64 { 99 }\n";

fn make_project(root: &Path) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"concurrent_project\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"src/main.mind\"\n",
    )
    .unwrap();
    fs::write(root.join("src/main.mind"), SIMPLE_MIND).unwrap();
}

#[test]
fn same_project_builds_serialize_without_manifest_or_cache_corruption() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    make_project(root);
    let manifest_before = fs::read(root.join("Mind.toml")).unwrap();
    let mindc = require_mindc();

    let spawn = || {
        let binary = mindc.clone();
        let project = root.to_path_buf();
        thread::spawn(move || {
            Command::new(binary)
                .args(["build", "--backend", "mlir"])
                .current_dir(project)
                .output()
                .expect("spawn project build")
        })
    };
    let first = spawn();
    let second = spawn();
    let first = first.join().expect("first build panicked");
    let second = second.join().expect("second build panicked");

    assert_eq!(fs::read(root.join("Mind.toml")).unwrap(), manifest_before);
    assert!(root.join(".mind-build.lock").is_file());
    assert!(!root.join("target/.mind-build.lock").exists());
    let capable_first = common::gate::compiled("mindc_project_lock", &first);
    let capable_second = common::gate::compiled("mindc_project_lock", &second);
    assert_eq!(capable_first, capable_second);
    if !capable_first {
        return;
    }

    let material = compile_cache_material(
        SIMPLE_MIND.as_bytes(),
        CacheKeyFlags {
            target: BuildTarget::Cpu,
            optimize: OptimizeLevel::Debug,
            emit: EmitKind::Binary,
            edition: 2024,
        },
        &mindc,
        root,
        &[root.join("src/main.mind")],
    )
    .expect("compiler identity");
    let cache = cache_root(root, BuildTarget::Cpu, OptimizeLevel::Debug);
    let meta = libmind::build::cache::meta_path(&cache, material.key());
    let parsed: serde_json::Value = serde_json::from_slice(&fs::read(meta).unwrap()).unwrap();
    assert!(parsed.is_object());
    assert!(matches!(probe(&cache, &material), CacheProbe::Hit { .. }));
}

#[test]
fn standalone_builds_leave_no_temporary_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let source_a = root.join("alpha.mind");
    let source_b = root.join("beta.mind");
    fs::write(&source_a, SIMPLE_MIND).unwrap();
    fs::write(&source_b, MODIFIED_MIND).unwrap();
    let mindc = require_mindc();

    let spawn = |source: &Path, output: &Path| {
        let binary = mindc.clone();
        let project = root.to_path_buf();
        let source = source.to_path_buf();
        let output = output.to_path_buf();
        thread::spawn(move || {
            Command::new(binary)
                .args(["build", "--backend", "mlir", "--emit=cdylib", "--out"])
                .arg(output)
                .arg(source)
                .current_dir(project)
                .output()
                .expect("spawn standalone build")
        })
    };
    let first = spawn(&source_a, &root.join("alpha.so"));
    let second = spawn(&source_b, &root.join("beta.so"));
    let first = first.join().expect("alpha build panicked");
    let second = second.join().expect("beta build panicked");

    let capable_first = common::gate::compiled("mindc_project_lock", &first);
    let capable_second = common::gate::compiled("mindc_project_lock", &second);
    assert_eq!(capable_first, capable_second);
    assert!(source_a.is_file());
    assert!(source_b.is_file());
    assert!(!root.join("Mind.toml").exists());
}

/// Run `args` in two directories holding `files`: `control` has no manifest,
/// `killed` holds the leftover temporary manifest of a killed build, planted
/// with the exact text the build writes for `entry`. The leftover must be gone
/// afterwards. Returns (control, killed) outputs.
fn leftover_vs_control(
    name: &str,
    files: &[(&str, &str)],
    entry: &str,
    args: &[&str],
) -> [Output; 2] {
    let tmp = tempfile::tempdir().unwrap();
    let mindc = require_mindc();
    let run_in = |dir: &Path| {
        Command::new(&mindc)
            .args(args)
            .current_dir(dir)
            .output()
            .expect("spawn mindc")
    };
    let dirs = ["control", "killed"].map(|leaf| {
        let dir = tmp.path().join(leaf);
        fs::create_dir(&dir).unwrap();
        for (file, text) in files {
            fs::write(dir.join(file), text).unwrap();
        }
        dir
    });
    let leftover = dirs[1].join("Mind.toml");
    fs::write(&leftover, libmind::build::temporary_manifest(name, entry)).unwrap();
    // Positive control: the planted text is a loadable manifest, so only the
    // marker keeps it from being a project.
    assert!(libmind::project::load_manifest(&dirs[1]).is_ok());
    let outputs = dirs.map(|dir| run_in(&dir));
    assert!(
        !leftover.exists(),
        "{args:?}: the leftover temporary manifest was not removed"
    );
    assert!(
        !outputs[0].status.success(),
        "{args:?}: control succeeded with no Mind.toml"
    );
    assert_eq!(
        outputs[1].status.code(),
        outputs[0].status.code(),
        "{args:?}: a leftover temporary manifest changed the exit status"
    );
    outputs
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// `mindc build` with no path, `mindc run` and `mindc test` follow the one
/// rule a named build does: a leftover temporary manifest of a killed build is
/// removed under the lock and is never a project, so each command ends exactly
/// as in a directory that never had a `Mind.toml`.
#[test]
fn no_path_build_treats_a_leftover_temporary_manifest_as_absent() {
    let files = [("alpha.mind", SIMPLE_MIND)];
    let [control, killed] = leftover_vs_control(
        "alpha",
        &files,
        "alpha.mind",
        &["build", "--backend", "mlir", "--emit=cdylib"],
    );
    assert_eq!(stderr_of(&killed), stderr_of(&control));
}

#[test]
fn run_treats_a_leftover_temporary_manifest_as_absent() {
    let files = [("alpha.mind", SIMPLE_MIND)];
    let [control, killed] =
        leftover_vs_control("alpha", &files, "alpha.mind", &["run", "--backend", "mlir"]);
    assert_eq!(stderr_of(&killed), stderr_of(&control));
}

/// The test file imports a sibling, so import discovery looks for a project:
/// with the leftover read as one, the test would run and pass.
#[cfg(feature = "cross-module-imports")]
#[test]
fn test_treats_a_leftover_temporary_manifest_as_absent() {
    let files = [
        (
            "helper.mind",
            "export { seven }\nfn seven() -> i64 { return 7; }\n",
        ),
        (
            "t.mind",
            "import helper;\n\n#[test]\nfn seven_is_seven() {\n    assert helper::seven() == 7, \"seven\";\n}\n",
        ),
    ];
    let [control, killed] = leftover_vs_control("t", &files, "t.mind", &["test"]);
    let refusal = |output: &Output| -> Vec<String> {
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            stderr_of(output)
        );
        text.lines()
            .filter(|line| line.contains("require an enclosing Mind.toml"))
            .map(str::to_string)
            .collect()
    };
    assert!(
        !refusal(&control).is_empty(),
        "control lost its no-project refusal"
    );
    assert_eq!(refusal(&killed), refusal(&control));
}
