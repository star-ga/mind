// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The historical native path refuses compiler intrinsics, and it says why.
//!
//! An unmarked native compiler selects the historical frozen profile, which
//! binds no compiler intrinsic. The closure refuses an intrinsic edge as
//! unresolved, and the frozen fence would refuse the same call next as
//! `call.undefined_or_builtin`. That refusal is deliberate and stays exactly as
//! strict: the unmarked seeds' intrinsic lowering is not proven in every value
//! position. Intrinsic binding belongs to native-lane-v1, which only the
//! compiler's `nblane01` mark selects.
//!
//! The defect pinned here is the diagnostic. It used to read only "has no
//! definition in the closure", as if the user had forgotten a function. A
//! qualification round read that as a missing binding in the historical
//! closure and filed it there, when the cause was that the selected compiler
//! did not carry the mark. The refusal must name the selection, keep its kind
//! label and first line, and never attach that note to a callee that is not a
//! compiler intrinsic.
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
const MARK: &[u8] = b"\x48\xb8nblane01";
const NOTE: &str = "is a compiler intrinsic. The selected native compiler does not carry \
                    the native-lane-v1 compatibility mark";

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

    /// Refused on the historical path before the compiler ran; returns stderr.
    fn historical_refusal(&self, source: &str, extra: &[&str]) -> String {
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
        assert!(stderr.contains("(closure.unresolved_edge)"), "{stderr}");
        stderr
    }
}

const MEMORY: &str = "fn main() -> i64 { let p: i64 = __mind_alloc(8); \
                      __mind_store_i64(p, 37); return __mind_load_i64(p); }";

/// Both memory profiles: the downstream consumer's builds use `bounded`.
const PROFILES: [&[&str]; 2] = [&[], &["--native-profile", "bounded"]];

#[test]
fn historical_intrinsic_refusal_names_the_profile_selection() {
    for extra in PROFILES {
        let stderr = Case::new(false).historical_refusal(MEMORY, extra);
        // The first line and the kind label are unchanged.
        assert!(
            stderr.contains(
                "error[backend-native]: `main` calls `__mind_alloc`, which has no \
                 definition in the closure (closure.unresolved_edge)\n"
            ),
            "{extra:?}: {stderr}"
        );
        assert!(
            stderr.contains(&format!("`__mind_alloc` {NOTE}")),
            "{extra:?}: {stderr}"
        );
        assert!(stderr.contains("binds no compiler intrinsic"), "{stderr}");
        assert!(stderr.contains("--backend mlir"), "{stderr}");
    }
}

/// Positive control for the note's claim: the mark alone changes the outcome.
/// Same compiler binary, same source, same delegated seed.
#[test]
fn the_mark_is_the_selection_the_note_names() {
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

#[test]
fn the_note_names_the_intrinsic_reached_through_a_helper() {
    let stderr = Case::new(false).historical_refusal(
        "fn helper() -> i64 { return __mind_load_i64(0); }\n\
         fn main() -> i64 { return helper(); }",
        &[],
    );
    assert!(
        stderr.contains("`helper` calls `__mind_load_i64`, which has no definition"),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("`__mind_load_i64` {NOTE}")),
        "{stderr}"
    );
}

/// The note is keyed on the callee: a missing user function, an unknown
/// `__mind_` name and a std function the closure cannot see keep the bare
/// refusal.
#[test]
fn a_callee_that_is_not_a_compiler_intrinsic_gets_no_note() {
    for (source, callee) in [
        ("fn main() -> i64 { return absent(); }", "absent"),
        (
            "fn main() -> i64 { return __mind_forged(8); }",
            "__mind_forged",
        ),
        ("fn main() -> i64 { return sha256(0, 0, 0); }", "sha256"),
    ] {
        let stderr = Case::new(false).historical_refusal(source, &[]);
        assert!(
            stderr.contains(&format!(
                "`main` calls `{callee}`, which has no definition in the closure"
            )),
            "{stderr}"
        );
        assert!(!stderr.contains(NOTE), "{source}: {stderr}");
        assert!(!stderr.contains("native-lane-v1"), "{source}: {stderr}");
    }
}
