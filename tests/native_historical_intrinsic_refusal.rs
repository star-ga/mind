// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A native compiler runs only when it carries `nblane02`.
//!
//! An unmarked compiler is refused before admission, with one message that
//! names that mark. The host does not judge the program: an intrinsic call, a
//! call through a helper and a missing user function all get that same
//! refusal, and the compiler does not run. Intrinsic binding belongs to the
//! compiler that carries the mark.
//!
//! Every build goes through a private recording shim, so the result does not
//! depend on whether the embedded seeds carry the mark. On a refusal the shim
//! must never run: the refusal comes before the compiler sees an image.

#![cfg(unix)]
#![cfg(all(
    unix,
    feature = "std-surface",
    target_os = "linux",
    target_arch = "x86_64"
))]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MINDC: &str = env!("CARGO_BIN_EXE_mindc");
const MARK: &[u8] = b"\x48\xb8nblane02";
const MARK_REFUSAL: &str = "does not carry the native-lane-v1 mark `nblane02`";

struct Case {
    dir: tempfile::TempDir,
    compiler: PathBuf,
}

impl Case {
    fn new(marked: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let compiler = dir.path().join("recording-compiler.sh");
        let mut bytes = b"#!/bin/sh\nset -eu\ncat > \"$MIND_NATIVE_TEST_CAPTURE\"\nexec \"$MIND_NATIVE_TEST_SEED\" < \"$MIND_NATIVE_TEST_CAPTURE\"\n# ".to_vec();
        if marked {
            bytes.extend_from_slice(MARK);
        }
        bytes.push(b'\n');
        std::fs::write(&compiler, bytes).unwrap();
        std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self { dir, compiler }
    }

    fn capture(&self) -> PathBuf {
        self.dir.path().join("image.bin")
    }

    fn program(&self) -> PathBuf {
        self.dir.path().join("program")
    }

    /// `extra` carries profile flags, e.g. `--native-profile bounded`.
    fn build(&self, source: &str, extra: &[&str]) -> Output {
        let source_path = self.dir.path().join("main.mind");
        std::fs::write(&source_path, source).unwrap();
        Command::new(MINDC)
            .args(["build"])
            .arg(source_path)
            .args(["--backend", "native", "--emit", "binary"])
            .args(extra)
            .arg("--out")
            .arg(self.program())
            .env("MINDC_NATIVE_ELF", &self.compiler)
            .env(
                "MINDC_STD_DIR",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("std"),
            )
            .env("MIND_NATIVE_TEST_CAPTURE", self.capture())
            .env(
                "MIND_NATIVE_TEST_SEED",
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("examples/mindc_mind/testdata/selfhost_loop/stage1.elf"),
            )
            .env_remove("MINDC_SO")
            .current_dir(self.dir.path())
            .output()
            .unwrap()
    }

    /// Refused before the compiler ran, because the shim carries no mark.
    fn mark_refusal(&self, source: &str, extra: &[&str]) -> String {
        let build = self.build(source, extra);
        let stderr = String::from_utf8_lossy(&build.stderr).into_owned();
        assert_eq!(build.status.code(), Some(3), "{stderr}");
        assert!(
            !self.program().exists(),
            "a refusal must not emit an artifact"
        );
        assert!(
            !self.capture().exists(),
            "the refusal must precede the compiler: {stderr}"
        );
        assert!(stderr.contains(MARK_REFUSAL), "{stderr}");
        assert!(stderr.contains("(native.compiler_mark)"), "{stderr}");
        stderr
    }
}

const MEMORY: &str = "fn main() -> i64 { let p: i64 = __mind_alloc(8); \
                      __mind_store_i64(p, 37); return __mind_load_i64(p); }";

/// Both memory profiles: the downstream consumer's builds use `bounded`.
const PROFILES: [&[&str]; 2] = [&[], &["--native-profile", "bounded"]];

#[test]
fn an_unmarked_compiler_is_refused_with_the_mark_on_both_profiles() {
    for extra in PROFILES {
        let stderr = Case::new(false).mark_refusal(MEMORY, extra);
        assert!(stderr.contains("nblane02"), "{extra:?}: {stderr}");
    }
}

/// Positive control: the mark alone changes the outcome.
/// Same compiler binary, same source, same delegated seed.
#[test]
fn the_mark_is_what_selects_the_compiler() {
    let marked = Case::new(true);
    let build = marked.build(MEMORY, &[]);
    assert_eq!(
        build.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(marked.program()).output().unwrap();
    assert_eq!(run.status.code(), Some(37), "{run:?}");
    assert!(run.stdout.is_empty() && run.stderr.is_empty(), "{run:?}");
}

/// The refusal does not depend on the callee. A helper, a missing user
/// function, an unknown `__mind_` name and a std name all stop at the mark.
#[test]
fn the_mark_refusal_does_not_depend_on_the_callee() {
    for source in [
        "fn helper() -> i64 { return __mind_load_i64(0); }\n\
         fn main() -> i64 { return helper(); }",
        "fn main() -> i64 { return absent(); }",
        "fn main() -> i64 { return __mind_forged(8); }",
        "fn main() -> i64 { return sha256(0, 0, 0); }",
        "fn g(s: String) -> i64 { return 1; }\nfn main() -> i64 { return g(5); }",
    ] {
        let stderr = Case::new(false).mark_refusal(source, &[]);
        assert!(
            !stderr.contains("closure.unresolved_edge"),
            "{source}: {stderr}"
        );
        assert!(
            !stderr.contains("frozen native profile"),
            "{source}: {stderr}"
        );
    }
}
