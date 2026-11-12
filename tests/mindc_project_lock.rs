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

/// `mindc lock`, `fetch` and `clean` find their project by the rule of the
/// build lock: a leftover temporary manifest is no project, so none of them
/// writes a lock file into, fetches for, or cleans a standalone directory.
#[test]
fn package_commands_treat_a_leftover_temporary_manifest_as_absent() {
    let files = [("alpha.mind", SIMPLE_MIND)];
    for command in ["lock", "fetch", "clean"] {
        let [control, killed] = leftover_vs_control("alpha", &files, "alpha.mind", &[command]);
        assert_eq!(stderr_of(&killed), stderr_of(&control), "{command}");
    }
}

/// `mindc fmt` reads its config from the project of the current directory by
/// the rule of the build lock: a leftover temporary manifest below a real
/// project does not hide that project's `[mindcraft.format]`.
#[test]
fn fmt_config_is_never_read_from_a_leftover_temporary_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let mindc = require_mindc();
    fs::write(
        tmp.path().join("Mind.toml"),
        "[package]\nname = \"formatted\"\nversion = \"0.1.0\"\n\n\
         [mindcraft.format]\nindent_width = 2\n",
    )
    .unwrap();
    let sub = tmp.path().join("sub");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("alpha.mind"),
        "fn main() -> i64 {\n    return 42;\n}\n",
    )
    .unwrap();
    let fmt = |dir: &Path, args: &[&str]| {
        Command::new(&mindc)
            .arg("fmt")
            .args(args)
            .arg("alpha.mind")
            .current_dir(dir)
            .output()
            .expect("spawn mindc fmt")
    };
    let formatted = fmt(&sub, &[]);
    assert!(formatted.status.success(), "{}", stderr_of(&formatted));
    // Positive control: under the default config the formatted file differs.
    let bare = tempfile::tempdir().unwrap();
    fs::copy(sub.join("alpha.mind"), bare.path().join("alpha.mind")).unwrap();
    let default_config = fmt(bare.path(), &["--check"]);
    assert_eq!(
        default_config.status.code(),
        Some(1),
        "the project config changed nothing"
    );
    let leftover = sub.join("Mind.toml");
    fs::write(
        &leftover,
        libmind::build::temporary_manifest("alpha", "alpha.mind"),
    )
    .unwrap();
    let checked = fmt(&sub, &["--check"]);
    assert!(checked.status.success(), "{}", stderr_of(&checked));
    assert!(
        !leftover.exists(),
        "the leftover temporary manifest was not removed"
    );
}

/// A search for the project made before the build lock meets a LIVE standalone
/// build: its lock held and its temporary manifest written (or still empty,
/// between creation and write). The writer removes that file, then releases
/// the lock. Each test plays that build, so the window the CI race needed is
/// held open: `/proc/locks` shows when the build under test waits on the lock.
#[cfg(target_os = "linux")]
mod live_standalone_build {
    use super::{MODIFIED_MIND, common, require_mindc};
    use libmind::build::{BuildOpts, run_build, temporary_manifest};
    use libmind::project::EmitKind;
    use std::fs::{self, File, OpenOptions};
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::process::ExitStatusExt;
    use std::path::Path;
    use std::process::{Command, ExitStatus, Output, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Take `root`'s project build lock the way a build takes it.
    fn hold_build_lock(root: &Path) -> File {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(".mind-build.lock"))
            .unwrap();
        fs2::FileExt::lock_exclusive(&file).unwrap();
        file
    }

    /// True once process `pid` waits on the lock `held` (a `->` line of
    /// `/proc/locks`), false when `ended` says it finished without waiting.
    fn waits_on(pid: u32, held: &File, mut ended: impl FnMut() -> bool) -> bool {
        let inode = held.metadata().unwrap().ino().to_string();
        let pid = pid.to_string();
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            let locks = fs::read_to_string("/proc/locks").unwrap();
            let waiting = locks.lines().any(|line| {
                let fields: Vec<&str> = line.split_whitespace().collect();
                fields.contains(&"->")
                    && fields.contains(&pid.as_str())
                    && fields
                        .iter()
                        .any(|f| f.contains(':') && f.rsplit(':').next() == Some(inode.as_str()))
            });
            if waiting {
                return true;
            }
            if ended() {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("process {pid} neither waited on the build lock nor ended");
    }

    /// The backend of `mindc build <file>` is selected before the build takes
    /// its lock. That search found the live manifest and read it after its
    /// writer removed it (`manifest error: Failed to read <dir>/Mind.toml`).
    /// It must wait the live build out: no route is chosen while the manifest
    /// exists, and the route is never read from it.
    #[test]
    fn route_selection_waits_out_a_live_standalone_build() {
        let mindc = require_mindc();
        for text in [temporary_manifest("alpha", "alpha.mind"), String::new()] {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path().canonicalize().unwrap();
            let source = root.join("beta.mind");
            fs::write(&source, MODIFIED_MIND).unwrap();
            let held = hold_build_lock(&root);
            let manifest = root.join("Mind.toml");
            fs::write(&manifest, &text).unwrap();
            let log = tempfile::NamedTempFile::new().unwrap();
            let mut build = Command::new(&mindc)
                .args(["build", "--backend", "mlir", "--emit=cdylib", "--out"])
                .arg(root.join("beta.so"))
                .arg(&source)
                .current_dir(&root)
                .stdout(Stdio::null())
                .stderr(log.reopen().unwrap())
                .spawn()
                .unwrap();
            let waited = waits_on(build.id(), &held, || build.try_wait().unwrap().is_some());
            let early = fs::read_to_string(log.path()).unwrap();
            fs::remove_file(&manifest).unwrap();
            drop(held);
            let status = build.wait().unwrap();
            let stderr = fs::read(log.path()).unwrap();
            let all = String::from_utf8_lossy(&stderr).into_owned();
            assert!(waited, "{text:?}: the build did not wait: {all}");
            assert!(
                !early.contains("backend:"),
                "{text:?}: a route was chosen while the live manifest existed: {early}"
            );
            assert!(!all.contains("manifest error"), "{text:?}: {all}");
            assert!(all.contains("backend: mlir (from flag)"), "{text:?}: {all}");
            assert!(!manifest.exists());
            let output = Output {
                status,
                stdout: Vec::new(),
                stderr,
            };
            common::gate::compiled("mindc_project_lock", &output);
        }
    }

    /// A named build one directory below a live standalone build, inside one
    /// repository, probes its root before it takes a lock. The probe found the
    /// live manifest, waited on its lock, and then read the removed file:
    /// `manifest error: Failed to read <root>/Mind.toml`. The probe follows the
    /// rule of the lock, so the build is standalone in its own directory.
    #[test]
    fn a_named_build_below_a_live_standalone_build_is_standalone() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        let source = root.join("sub/beta.mind");
        fs::write(&source, MODIFIED_MIND).unwrap();
        let held = hold_build_lock(&root);
        let manifest = root.join("Mind.toml");
        fs::write(&manifest, temporary_manifest("alpha", "alpha.mind")).unwrap();
        let opts = BuildOpts {
            paths: vec![source],
            emit: Some(EmitKind::Cdylib),
            out: Some(root.join("sub/beta.so")),
            no_cache: true,
            ..BuildOpts::default()
        };
        let build = thread::spawn(move || match run_build(&opts) {
            Ok(_) => (0, String::new()),
            Err(error) => (error.exit_code(), error.render()),
        });
        let waited = waits_on(std::process::id(), &held, || build.is_finished());
        fs::remove_file(&manifest).unwrap();
        drop(held);
        let (code, rendered) = build.join().expect("the build panicked");
        assert!(waited, "the build did not wait: {rendered}");
        assert!(!rendered.contains("manifest error"), "{rendered}");
        assert!(!root.join("sub/Mind.toml").exists());
        let output = Output {
            status: ExitStatus::from_raw(code << 8),
            stdout: Vec::new(),
            stderr: rendered.into_bytes(),
        };
        common::gate::compiled("mindc_project_lock", &output);
    }

    /// The search `open_explicit` makes again after a wait follows the same
    /// rule. Here the entry's own project loses its manifest while the build
    /// waits on that project's lock, so the build searches again and meets a
    /// live standalone build one directory up. Read as a project, that root
    /// failed after its writer removed the file (`manifest error`).
    #[test]
    fn the_search_after_a_wait_waits_out_a_live_standalone_build() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let sub = root.join("sub");
        fs::create_dir(root.join(".git")).unwrap();
        fs::create_dir(&sub).unwrap();
        let source = sub.join("beta.mind");
        fs::write(&source, MODIFIED_MIND).unwrap();
        let own = sub.join("Mind.toml");
        fs::write(
            &own,
            "[package]\nname = \"beta\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"beta.mind\"\n",
        )
        .unwrap();
        let held_root = hold_build_lock(&root);
        let live = root.join("Mind.toml");
        fs::write(&live, temporary_manifest("alpha", "alpha.mind")).unwrap();
        let held_sub = hold_build_lock(&sub);
        let opts = BuildOpts {
            paths: vec![source],
            emit: Some(EmitKind::Cdylib),
            out: Some(sub.join("beta.so")),
            no_cache: true,
            ..BuildOpts::default()
        };
        let build = thread::spawn(move || match run_build(&opts) {
            Ok(_) => (0, String::new()),
            Err(error) => (error.exit_code(), error.render()),
        });
        let waited_sub = waits_on(std::process::id(), &held_sub, || build.is_finished());
        fs::remove_file(&own).unwrap();
        drop(held_sub);
        let waited_root = waits_on(std::process::id(), &held_root, || build.is_finished());
        fs::remove_file(&live).unwrap();
        drop(held_root);
        let (code, rendered) = build.join().expect("the build panicked");
        assert!(
            waited_sub && waited_root,
            "the build did not wait: {rendered}"
        );
        assert!(!rendered.contains("manifest error"), "{rendered}");
        let output = Output {
            status: ExitStatus::from_raw(code << 8),
            stdout: Vec::new(),
            stderr: rendered.into_bytes(),
        };
        common::gate::compiled("mindc_project_lock", &output);
    }
}
