//! Shared harness for the native module-bridge integration controls.
//!
//! Split out so neither control file exceeds the 800-line ceiling
//! `tests/module_size_ratchet.rs` pins. This is a `tests/<dir>/mod.rs` module,
//! not a test target, so it compiles into each consumer rather than running as
//! its own binary. The real stage1 image is executed only on Linux x86-64;
//! other hosts use the host-native drain fixture for admission and transport.

#![allow(dead_code)]

// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

use fs2::FileExt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

pub const MINDC: &str = env!("CARGO_BIN_EXE_mindc");

pub struct Project {
    dir: tempfile::TempDir,
}

impl Project {
    /// A project with a manifest and a git root, which the bounded project-root
    /// scan requires before it will resolve siblings.
    pub fn new(name: &str) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("src")).expect("src");
        std::fs::write(
            root.join("Mind.toml"),
            format!(
                "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n\
                 [build]\ntarget = \"cpu\"\nemit = \"binary\"\nentry = \"src/main.mind\"\n"
            ),
        )
        .expect("manifest");
        // `find_project_root_for_file_checked` is git-root bounded.
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(root)
                .status()
                .expect("git init")
                .success(),
            "git init must succeed or every case degrades to MissingProject"
        );
        Self { dir }
    }

    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// Read the image consumed by the host-native transport fixture. This is
    /// available only on non-Linux/x86 hosts; the real stage1 image does not
    /// use this test-only capture path.
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    pub fn captured_source_image(&self) -> Vec<u8> {
        std::fs::read(self.root().join("native-image.bin"))
            .expect("host-native fixture must capture the source image")
    }

    pub fn write(&self, rel: &str, body: &str) -> PathBuf {
        let p = self.root().join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&p, body).expect("write");
        p
    }

    /// Build through the native backend. Returns (exit code, stderr, artifact bytes).
    pub fn build_native(&self, entry_rel: &str) -> (i32, String, Option<Vec<u8>>) {
        self.build_native_emit(entry_rel, "binary")
    }

    /// Build with NO source argument: the manifest entry must drive it.
    pub fn build_native_no_path(&self) -> (i32, String, Option<Vec<u8>>) {
        self.build_native_emit("", "binary")
    }

    pub fn build_native_emit(&self, entry_rel: &str, emit: &str) -> (i32, String, Option<Vec<u8>>) {
        self.build_with(entry_rel, emit, native_compiler())
    }

    /// Build through the historical (unmarked) compiler on every host: the
    /// drain fixture carries no lane marker, so admission takes the
    /// historical branch. For refusal controls only; it runs no MIND code.
    pub fn build_native_historical(&self, entry_rel: &str) -> (i32, String, Option<Vec<u8>>) {
        self.build_with(entry_rel, "binary", historical_compiler())
    }

    fn build_with(
        &self,
        entry_rel: &str,
        emit: &str,
        compiler: PathBuf,
    ) -> (i32, String, Option<Vec<u8>>) {
        let out = self.root().join("out.elf");
        let _ = std::fs::remove_file(&out);
        let mut args: Vec<String> = vec!["build".into()];
        if !entry_rel.is_empty() {
            args.push(entry_rel.into());
        }
        let res0 = args;
        let res = Command::new(MINDC)
            .args(res0)
            .args([
                "--release".to_string(),
                "--backend".to_string(),
                "native".to_string(),
                format!("--emit={emit}"),
                "--out".to_string(),
            ])
            .arg(&out)
            .current_dir(self.root())
            .env("MINDC_STD_DIR", repo_path("std"))
            .env("MINDC_NATIVE_ELF", compiler)
            .env(
                "MIND_NATIVE_TEST_CAPTURE",
                self.root().join("native-image.bin"),
            )
            .output()
            .expect("spawn mindc");
        let bytes = std::fs::read(&out).ok();
        (
            res.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&res.stderr).into_owned(),
            bytes,
        )
    }
}

/// The committed compiler image is Linux x86-64.  Admission and source-image
/// transport are still portable controls, so other CI targets use a host-native
/// executable that drains stdin and emits a deterministic non-executable anchor.
/// Semantic native execution remains covered by `assert_native_result` on the
/// Linux x86-64 target where the real image can run.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn native_compiler() -> PathBuf {
    repo_path("examples/mindc_mind/testdata/selfhost_loop/stage1.elf")
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn native_compiler() -> PathBuf {
    historical_compiler()
}

/// The host-native drain fixture, built once per test binary on any host.
fn historical_compiler() -> PathBuf {
    static FIXTURE: OnceLock<PathBuf> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let dir = tempfile::tempdir().expect("native fixture directory");
            let root = dir.keep();
            let output = root.join(if cfg!(windows) {
                "draining_native_compiler.exe"
            } else {
                "draining_native_compiler"
            });
            let status = Command::new("rustc")
                .args([
                    "--edition",
                    "2024",
                    "--crate-name",
                    "draining_native_compiler",
                ])
                .arg(repo_path(
                    "tests/native_bridge_support/draining_native_compiler.rs",
                ))
                .arg("-o")
                .arg(&output)
                .status()
                .expect("compile host-native drain fixture");
            assert!(
                status.success(),
                "host-native drain fixture compilation failed: {status}"
            );
            output
        })
        .clone()
}

/// Assert a semantic result where the real Linux x86-64 image is available.
/// Other targets assert that the host fixture received a nonempty source image
/// and emitted its exact deterministic anchor; dedicated composition controls
/// prove source completeness and identity. They do not pretend that a fake
/// anchor executed MIND semantics.
pub fn assert_native_result(project: &Project, bytes: &[u8], expected: i32, context: &str) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let _ = project;
        assert_eq!(run(bytes), expected, "{context}");
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let captured = project.captured_source_image();
        assert!(
            !captured.is_empty(),
            "{context}: host-native fixture received no source image"
        );
        assert_eq!(
            bytes.len(),
            397,
            "{context}: host-native transport fixture must emit its fixed anchor"
        );
        assert_eq!(
            &bytes[..4],
            b"\x7fELF",
            "{context}: host-native transport fixture must preserve ELF framing"
        );
        assert_eq!(
            &bytes[4..],
            &[0; 393],
            "{context}: host-native transport fixture must preserve its exact anchor"
        );
        let _ = expected;
    }
}

pub fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Run an emitted artifact and return its exit code.
pub fn run(bytes: &[u8]) -> i32 {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("run.elf");
    // A concurrent Command::spawn can fork while this writer is open and
    // inherit its writable descriptor.  On Linux that makes the later execve
    // fail with ETXTBSY even after this thread closes its copy.  Hold an
    // exclusive flock before closing the writer, then acquire an exclusive
    // lock through a read-only reopen.  The read-only lock waits for inherited
    // writer descriptors to close and proves that subsequent forks inherit
    // only a read-only descriptor. See rust-lang/rust#114554 for this handoff.
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&p)
        .expect("create elf");
    writer.write_all(bytes).expect("write elf");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = writer.metadata().expect("meta").permissions();
        perm.set_mode(0o755);
        writer.set_permissions(perm).expect("chmod");
    }
    writer.lock_exclusive().expect("lock writable elf");
    drop(writer);
    let reader = OpenOptions::new().read(true).open(&p).expect("reopen elf");
    reader.lock_exclusive().expect("lock readable elf");
    drop(reader);
    Command::new(&p)
        .status()
        .expect("run artifact")
        .code()
        .unwrap_or(-1)
}
