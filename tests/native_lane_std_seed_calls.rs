// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Calls from a native-lane-v1 program into the std seed, end to end.
//!
//! Every fixture in `tests/fixtures/native_lane_std/` is built with the
//! embedded (lane-marked) native compiler AND with the MLIR backend. A
//! positive fixture must build and run on both with the same exit code and
//! the same stdout bytes, and those must equal the value pinned here (for the
//! SHA-256 fixtures, the FIPS 180-4 digest computed independently). A negative
//! fixture must refuse natively with exit 3, no artifact, and the named kind
//! and site.
//!
//! Before the seed tier existed every positive fixture refused natively as
//! `lane.call.unknown` while MLIR ran it, and an unused non-seed import was
//! silently nothing.

#![cfg(all(
    feature = "std-surface",
    feature = "mlir-build",
    target_os = "linux",
    target_arch = "x86_64"
))]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MINDC: &str = env!("CARGO_BIN_EXE_mindc");

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/native_lane_std")
        .join(format!("{name}.mind"))
}

/// Build `name` with one backend in its own temporary directory (the MLIR
/// build writes `target/` next to the source, never into the repository).
fn build(name: &str, native: bool) -> (tempfile::TempDir, Output, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("main.mind");
    std::fs::copy(fixture(name), &source).unwrap();
    let artifact = dir.path().join("program");
    let mut cmd = Command::new(MINDC);
    cmd.arg("build").arg(&source);
    if native {
        cmd.args(["--backend", "native", "--emit", "binary", "--out"]);
        cmd.arg(&artifact);
    } else {
        cmd.args(["--backend", "mlir", "--emit=binary"])
            .arg(format!("--out={}", artifact.display()));
    }
    let output = cmd
        .env(
            "MINDC_STD_DIR",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("std"),
        )
        .env_remove("MINDC_NATIVE_ELF")
        .env_remove("MINDC_SO")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    (dir, output, artifact)
}

/// The (exit code, stdout) of running `name` built by one backend.
fn run(name: &str, native: bool) -> (i32, Vec<u8>) {
    let (_dir, built, artifact) = build(name, native);
    assert_eq!(
        built.status.code(),
        Some(0),
        "{name} (native={native}) did not build:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&artifact).output().unwrap();
    assert!(ran.stderr.is_empty(), "{name} (native={native}): {ran:?}");
    (ran.status.code().expect("exited"), ran.stdout)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Native and MLIR agree, and both equal the pinned result.
fn parity(name: &str, exit: i32, stdout: &[u8]) {
    let native = run(name, true);
    let mlir = run(name, false);
    assert_eq!(native, mlir, "{name}: native != MLIR");
    assert_eq!(native.0, exit, "{name}");
    assert_eq!(hex(&native.1), hex(stdout), "{name}");
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn sha256_runs_natively_with_the_fips_digest() {
    let empty = unhex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    parity("sha256_empty", 0xe3, &empty);
    // 1000 bytes of (i * 7 + 3) & 255: sixteen blocks, the padding in the last.
    let long = unhex("1e9bc38cbf860b9ec31918b065f9b52476c549a782e0e7990bed8ce3868d2371");
    parity("sha256_1000", 0x71, &long);
}

#[test]
fn pure_i64_seed_helpers_and_their_private_callees_run_natively() {
    parity("io_print_bytes", 43, b"Hi\n");
    parity("ring_roundtrip", 131, b"");
    parity("string_push_capacity", 8, b"");
}

#[test]
fn a_seed_function_calling_another_seed_module_runs_natively() {
    // std.io_canon's anchor hashes the canonical event bytes with std.sha256.
    let (exit, stdout) = run("io_canon_anchor", false);
    assert_eq!((exit, stdout.len()), (2, 32));
    parity("io_canon_anchor", exit, &stdout);
}

/// Refuses natively with exit 3 and no artifact, naming every `labels` entry.
fn refuses(name: &str, labels: &[&str]) {
    let (_dir, built, artifact) = build(name, true);
    let stderr = String::from_utf8_lossy(&built.stderr);
    assert_eq!(built.status.code(), Some(3), "{name}: {stderr}");
    assert!(!artifact.exists(), "{name}: a refusal wrote an artifact");
    for label in labels {
        assert!(
            stderr.contains(label),
            "{name}: missing {label:?} in {stderr}"
        );
    }
}

/// The source-set resolver refuses the import first, naming the module
/// (`scope.std_module_not_in_native_seed`, std_bundled_module_table.rs);
/// `lane.import.not_seed` stays the lane's own backstop behind it.
#[test]
fn a_non_seed_std_import_refuses_used_or_not() {
    for name in ["import_not_seed_unused", "import_not_seed_used"] {
        refuses(
            name,
            &[
                "(scope.std_module_not_in_native_seed)",
                "imports `std.hkdf`, which ships with mindc but is not in the std seed",
            ],
        );
    }
    // Positive controls: MLIR ignores the unused import, and an unused SEED
    // import is admitted natively.
    assert_eq!(run("import_not_seed_unused", false), (7, vec![]));
    parity("import_seed_unused", 7, b"");
}

#[test]
fn a_reached_seed_body_outside_the_lane_refuses_in_its_own_module() {
    refuses(
        "cli_args_get",
        &["(lane.type.named) in std/cli.mind fn `args_get` at line 63, column 5 of std/cli.mind"],
    );
    refuses(
        "vec_zeroed",
        &["(lane.call.unknown) in std/vec.mind fn `vec_zeroed` at line 105"],
    );
    // The default profile reclaims nothing: std.sha256 frees its scratch, so
    // `sha256` reached from a loop refuses the free inside the seed body it
    // calls (measured: `sha256_checked`).
    refuses(
        "std_sha256_in_loop",
        &["(lane.alloc.free_in_loop) in std/sha256.mind fn `sha256_checked`"],
    );
}

#[test]
fn a_user_name_colliding_with_the_seed_is_refused_never_preferred() {
    refuses(
        "user_shadows_seed_helper",
        &["(lane.call.ambiguous) in std/sha256.mind fn `sha_compress_block`"],
    );
    // A closed image omits the whole seed, so a private user fn named like a
    // PUBLIC seed function builds and runs its own body: 1 + 2 + 3.
    assert_eq!(run("user_defines_sha256", true), (6, vec![]));
    // The same name in the user image and in an imported seed module: both
    // tiers answer the call, so it is ambiguous at the call site.
    refuses(
        "user_defines_imported_seed_function",
        &["(lane.call.ambiguous) in fn `main` at line 8, column 12"],
    );
    // A private collision may omit an UNUSED seed; a seed call keeps it, so
    // the collision refuses instead of dropping the called function.
    refuses(
        "private_collision_with_seed_call",
        &[
            "source call `sha256` is not closed",
            "closure.std_symbol_collision",
        ],
    );
}

/// Vendored identical copies beside divergent private collisions. Omission
/// runs before the identical-copy guard: the closed image drops the seed, so
/// the divergent `sha_compress_block`, `toml_copy_bytes` and `sha_w_set` bind nothing and
/// the vendored core hashes "abc" (exit 0). The same image plus a seed call
/// is open and still refuses.
#[test]
fn a_closed_vendoring_image_omits_the_seed_and_runs() {
    parity("naes_shape_closed_vendored", 0, b"");
    refuses(
        "naes_shape_open_divergent",
        &[
            "`sha_compress_block` collides with a std seed-blob symbol (closure.std_symbol_collision)",
        ],
    );
    // The NAES `decode` shape: a private fn named like std.json's public
    // `decode`. Closed, the seed is omitted and the user body runs (argc 1,
    // exit 2); open, the refusal names the symbol and its std module.
    assert_eq!(run("naes_shape_public_name_closed", true), (2, vec![]));
    refuses(
        "naes_shape_public_name_open",
        &[
            "`decode` collides with public std symbol `decode` defined in std/json.mind \
           (closure.std_symbol_collision)",
        ],
    );
}

#[test]
fn a_user_call_binds_only_to_a_public_function_of_an_imported_seed_module() {
    refuses(
        "sha256_without_import",
        &["(lane.call.unknown) in fn `main`"],
    );
    refuses("private_seed_helper", &["(lane.call.unknown) in fn `main`"]);
}
