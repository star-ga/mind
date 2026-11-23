// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Candidate selector/namespace controls, not release qualification.
//!
//! A private recording shim receives a synthetic compatibility marker and
//! delegates to the historical seed. This tests scalar programs the seed
//! already executes. It does not attest that this seed implements the
//! complete candidate lane. Final qualification uses a newly built compiler
//! with the marker emitted from its committed source and the full lane corpus.

#![cfg(unix)]
#![cfg(all(
    unix,
    feature = "std-surface",
    target_os = "linux",
    target_arch = "x86_64"
))]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const MINDC: &str = env!("CARGO_BIN_EXE_mindc");
const MARK: &[u8] = b"\x48\xb8nblane02";
/// The lane mark of the 0.10.3 seeds. The host no longer selects the lane for it.
const OLD_MARK: &[u8] = b"\x48\xb8nblane01";

const SEED: &str = "examples/mindc_mind/testdata/selfhost_loop/stage1.elf";

struct Case {
    dir: tempfile::TempDir,
    compiler: PathBuf,
    seed: PathBuf,
}

impl Case {
    fn new(marked: bool) -> Self {
        Self::with_mark(marked.then_some(MARK))
    }

    fn with_mark(mark: Option<&[u8]>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let compiler = dir.path().join("recording-compiler.sh");
        let mut bytes = b"#!/bin/sh\nset -eu\ncat > \"$MIND_NATIVE_TEST_CAPTURE\"\nexec \"$MIND_NATIVE_TEST_SEED\" < \"$MIND_NATIVE_TEST_CAPTURE\"\n# ".to_vec();
        if let Some(mark) = mark {
            bytes.extend_from_slice(mark);
        }
        bytes.push(b'\n');
        std::fs::write(&compiler, bytes).unwrap();
        std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700)).unwrap();
        let seed = Path::new(env!("CARGO_MANIFEST_DIR")).join(SEED);
        Self {
            dir,
            compiler,
            seed,
        }
    }

    /// A case whose seed is a temporary copy of the committed seed with the
    /// 0.10.3 lane mark in place of the current one (a seed with the old mark
    /// whatever the advance has embedded), recorded under the old mark.
    fn with_old_mark_seed() -> Self {
        let mut case = Self::with_mark(Some(OLD_MARK));
        let mut bytes = std::fs::read(&case.seed).unwrap();
        let at = bytes.windows(MARK.len()).position(|w| w == MARK);
        if let Some(at) = at {
            bytes[at..at + MARK.len()].copy_from_slice(OLD_MARK);
        }
        assert!(
            bytes.windows(OLD_MARK.len()).any(|w| w == OLD_MARK),
            "the seed copy carries no nblane01 mark"
        );
        assert!(!bytes.windows(MARK.len()).any(|w| w == MARK));
        case.seed = case.dir.path().join("old-mark-seed.elf");
        std::fs::write(&case.seed, bytes).unwrap();
        std::fs::set_permissions(&case.seed, std::fs::Permissions::from_mode(0o700)).unwrap();
        case
    }

    fn build(&self, source: &str) -> (std::process::Output, PathBuf) {
        let source_path = self.dir.path().join("main.mind");
        let output_path = self.dir.path().join("program");
        std::fs::write(&source_path, source).unwrap();
        let output = Command::new(MINDC)
            .args(["build"])
            .arg(source_path)
            .args(["--backend", "native", "--emit", "binary", "--out"])
            .arg(&output_path)
            .env("MINDC_NATIVE_ELF", &self.compiler)
            .env(
                "MINDC_STD_DIR",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("std"),
            )
            .env(
                "MIND_NATIVE_TEST_CAPTURE",
                self.dir.path().join("image.bin"),
            )
            .env("MIND_NATIVE_TEST_SEED", &self.seed)
            .env_remove("MINDC_SO")
            .current_dir(self.dir.path())
            .output()
            .unwrap();
        (output, output_path)
    }

    fn runs(&self, source: &str, exit: i32) {
        let (build, binary) = self.build(source);
        assert_eq!(
            build.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(binary).output().unwrap();
        assert_eq!(run.status.code(), Some(exit), "{run:?}");
        assert!(run.stdout.is_empty(), "{run:?}");
        assert!(run.stderr.is_empty(), "{run:?}");
    }

    fn refuses(&self, source: &str, label: &str) {
        let (build, binary) = self.build(source);
        assert_eq!(build.status.code(), Some(3), "{build:?}");
        assert!(!binary.exists(), "a refusal must not emit an artifact");
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert!(stderr.contains(label), "{stderr}");
    }
}

const COLLIDING: &str =
    "fn sha_bsig0(x: i64) -> i64 { return x + 3; }\nfn main() -> i64 { return sha_bsig0(4); }\n";

#[test]
fn marker_selects_closed_private_namespace_and_an_unmarked_compiler_is_refused() {
    Case::new(false).refuses(COLLIDING, "native.compiler_mark");
    let candidate = Case::new(true);
    candidate.runs(COLLIDING, 7);
    let image = std::fs::read(candidate.dir.path().join("image.bin")).unwrap();
    assert_eq!(i64::from_le_bytes(image[..8].try_into().unwrap()), 0);
    assert_eq!(&image[16..], COLLIDING.as_bytes());
}

#[test]
fn an_unmarked_compiler_refuses_a_scalar_and_the_marked_compiler_runs_it() {
    let source = "fn main() -> i64 { return 7; }\n";
    Case::new(false).refuses(source, "native.compiler_mark");
    let candidate = Case::new(true);
    candidate.runs(source, 7);
    let image = std::fs::read(candidate.dir.path().join("image.bin")).unwrap();
    assert!(i64::from_le_bytes(image[..8].try_into().unwrap()) > 0);
}

#[test]
fn public_collision_and_hidden_std_dependency_still_refuse() {
    Case::new(true).refuses(
        &COLLIDING.replacen("fn sha_bsig0", "pub fn sha_bsig0", 1),
        "closure.std_symbol_collision",
    );
    let dead = format!("{COLLIDING}\nfn dead() -> i64 {{ return sha256(0, 0, 0); }}");
    Case::new(true).refuses(&dead, "native-lane-v1");
}

#[test]
fn candidate_intrinsic_memory_roundtrip_is_executed() {
    Case::new(true).runs("fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 37); return __mind_load_i64(p); }", 37);
}

/// The `__mind_store_i16` FrozenNative contract row (R10/N5): before the row
/// the lane refuses the call as `lane.call.unknown`; with the row the 2-byte
/// store runs and the low byte is read back.
#[test]
fn candidate_store_i16_contract_row_is_admitted() {
    Case::new(true).runs(
        "fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i16(p, 42); return __mind_load_i8(p); }",
        42,
    );
}

#[test]
fn candidate_cannot_accept_forged_intrinsics_or_unsupported_abis() {
    for source in [
        "fn main() -> i64 { return __mind_forged(8); }",
        "fn main() -> i64 { return __mind_alloc(); }",
        "fn main() -> i64 { return __mind_alloc(1.5); }",
        "extern \"C\" { fn __mind_alloc(x: f64) -> f64; }\nfn main() -> i64 { return 7; }",
        "extern \"C\" { fn open(x: i64) -> i64; }\nfn main() -> i64 { return open(0); }",
    ] {
        Case::new(true).refuses(source, "native-lane-v1");
    }
}

/// The host runs a native compiler only when it carries `nblane02`. A compiler
/// with the 0.10.3 mark (`nblane01`), or with no mark, is refused before any
/// profile judges the program, and the message names `nblane02`.
///
/// These four programs used to build on that old mark: the frozen profile
/// admitted a String parameter called with an integer, a String const literal,
/// a String result and `let s: String = 5`. None of them may build, and the
/// compiler must not run.
#[test]
fn a_compiler_without_the_current_mark_is_refused() {
    let int_in_string = "fn g(s: String) -> i64 { return 1; }\nfn main() -> i64 { return g(5); }\n";
    let string_const = "const S: String = \"hi\";\nfn main() -> i64 { return 0; }\n";
    let string_result = "fn g(s: String) -> String { return s; }\nfn main() -> i64 { return 7; }\n";
    let string_let = "fn main() -> i64 { let s: String = 5; return 5; }\n";
    let programs = [int_in_string, string_const, string_result, string_let];
    for source in programs {
        for case in [Case::with_old_mark_seed(), Case::new(false)] {
            let (build, binary) = case.build(source);
            let stderr = String::from_utf8_lossy(&build.stderr);
            assert_eq!(build.status.code(), Some(3), "{stderr}");
            assert!(
                !binary.exists(),
                "a refusal must not emit an artifact: {stderr}"
            );
            assert!(
                stderr.contains("does not carry the native-lane-v1 mark `nblane02`"),
                "{stderr}"
            );
            assert!(stderr.contains("(native.compiler_mark)"), "{stderr}");
            assert!(
                !case.dir.path().join("image.bin").exists(),
                "the compiler must not run: {stderr}"
            );
        }
    }
    // The mark the host selects does not take this refusal. The host check has
    // no string rule left; the compiler decides. The recording shim writes
    // image.bin only when the host starts the compiler, so a build that the host
    // stops for any other reason is red here too.
    let marked = Case::new(true);
    let (build, _) = marked.build(int_in_string);
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !stderr.contains("does not carry the native-lane-v1 mark"),
        "{stderr}"
    );
    assert!(
        marked.dir.path().join("image.bin").is_file(),
        "the host must start the marked compiler: status={:?} {stderr}",
        build.status.code()
    );
}
