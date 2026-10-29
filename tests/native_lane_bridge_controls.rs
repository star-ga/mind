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
const MARK: &[u8] = b"\x48\xb8nblane01";

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
            .env(
                "MIND_NATIVE_TEST_SEED",
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("examples/mindc_mind/testdata/selfhost_loop/stage1.elf"),
            )
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
    "fn bsig0(x: i64) -> i64 { return x + 3; }\nfn main() -> i64 { return bsig0(4); }\n";

#[test]
fn marker_selects_closed_private_namespace_and_old_seed_stays_strict() {
    Case::new(false).refuses(COLLIDING, "closure.std_symbol_collision");
    let candidate = Case::new(true);
    candidate.runs(COLLIDING, 7);
    let image = std::fs::read(candidate.dir.path().join("image.bin")).unwrap();
    assert_eq!(i64::from_le_bytes(image[..8].try_into().unwrap()), 0);
    assert_eq!(&image[16..], COLLIDING.as_bytes());
}

#[test]
fn noncolliding_scalar_image_is_identical_across_profile_selection() {
    let source = "fn main() -> i64 { return 7; }\n";
    let historical = Case::new(false);
    let candidate = Case::new(true);
    historical.runs(source, 7);
    candidate.runs(source, 7);
    let first = std::fs::read(historical.dir.path().join("image.bin")).unwrap();
    let second = std::fs::read(candidate.dir.path().join("image.bin")).unwrap();
    assert_eq!(first, second);
    assert!(i64::from_le_bytes(first[..8].try_into().unwrap()) > 0);
    assert_eq!(
        std::fs::read(historical.dir.path().join("program")).unwrap(),
        std::fs::read(candidate.dir.path().join("program")).unwrap()
    );
}

#[test]
fn public_collision_and_hidden_std_dependency_still_refuse() {
    Case::new(true).refuses(
        &COLLIDING.replacen("fn bsig0", "pub fn bsig0", 1),
        "closure.std_symbol_collision",
    );
    let dead = format!("{COLLIDING}\nfn dead() -> i64 {{ return sha256(0, 0, 0); }}");
    Case::new(true).refuses(&dead, "native-lane-v1");
}

#[test]
fn candidate_intrinsic_memory_roundtrip_is_executed() {
    Case::new(true).runs("fn main() -> i64 { let p: i64 = __mind_alloc(8); __mind_store_i64(p, 37); return __mind_load_i64(p); }", 37);
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
