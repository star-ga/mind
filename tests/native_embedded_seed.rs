// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc build --backend native` from outside the repository, with a clean
//! environment.
//!
//! The bridge used to read the std seed from `./std` and the compiler from a
//! repo-relative `stage1.elf`, so outside a checkout it failed with "cannot
//! read std module 'std/arena.mind'" unless `MINDC_STD_DIR` and
//! `MINDC_NATIVE_ELF` were set. Both are now embedded in `mindc` and verified
//! against their committed pins before use. These controls build from a fresh
//! temp directory with every variable cleared except `PATH` and `TMPDIR`, and
//! require the result to equal, byte for byte, the build that names the
//! in-repo seed and std explicitly.
#![cfg(all(feature = "std-surface", target_os = "linux", target_arch = "x86_64"))]

mod native_bridge_support;
use native_bridge_support::{MINDC, repo_path, run};
use std::path::Path;
use std::process::Command;

const PROGRAM: &str = "fn twice(x: i64) -> i64 { return x + x; }\n\
                       fn main() -> i64 { return twice(3) + 1; }\n";

/// Build `prog.mind` in `dir` natively. `explicit` names the in-repo seed and
/// std through the environment; otherwise the environment is cleared.
fn build(dir: &Path, tmp: &Path, profile: &str, explicit: bool) -> (i32, String, Option<Vec<u8>>) {
    std::fs::write(dir.join("prog.mind"), PROGRAM).expect("write program");
    let out = dir.join(format!("prog-{profile}-{explicit}.elf"));
    let mut cmd = Command::new(MINDC);
    cmd.env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("TMPDIR", tmp)
        .current_dir(dir)
        .args([
            "build",
            "prog.mind",
            "--backend",
            "native",
            "--native-profile",
        ])
        .arg(profile)
        .arg("--out")
        .arg(&out);
    if explicit {
        let seed = if profile == "bounded" {
            "examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf"
        } else {
            "examples/mindc_mind/testdata/selfhost_loop/stage1.elf"
        };
        cmd.env("MINDC_STD_DIR", repo_path("std"))
            .env("MINDC_NATIVE_ELF", repo_path(seed));
    }
    let res = cmd.output().expect("spawn mindc");
    (
        res.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&res.stderr).into_owned(),
        std::fs::read(&out).ok(),
    )
}

fn check_profile(profile: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let tmp = tempfile::tempdir().expect("private TMPDIR");
    assert!(
        !dir.path().starts_with(env!("CARGO_MANIFEST_DIR")),
        "the control must run outside the repository"
    );
    let (code, stderr, embedded) = build(dir.path(), tmp.path(), profile, false);
    assert_eq!(
        code, 0,
        "{profile}: clean-env native build failed:\n{stderr}"
    );
    let embedded = embedded.expect("artifact");
    assert_eq!(run(&embedded), 7, "{profile}: embedded-seed artifact exit");
    // The unpacked seed is removed once the compiler has run.
    let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("read TMPDIR")
        .collect();
    assert!(
        leftovers.is_empty(),
        "{profile}: unpacked seed left behind: {leftovers:?}"
    );

    let (code, stderr, explicit) = build(dir.path(), tmp.path(), profile, true);
    assert_eq!(
        code, 0,
        "{profile}: explicit-seed native build failed:\n{stderr}"
    );
    assert_eq!(
        Some(embedded),
        explicit,
        "{profile}: the embedded seed must produce the in-repo seed's exact bytes"
    );
}

#[test]
fn a_clean_env_native_build_outside_the_repo_uses_the_embedded_default_seed() {
    check_profile("default");
}

#[test]
fn a_clean_env_native_build_outside_the_repo_uses_the_embedded_bounded_seed() {
    check_profile("bounded");
}

/// The exact flag set a consumer qualification build passes. Every program in
/// that build refused with "the embedded std seed blob hashes to ..., but the
/// manifest pins SEED_BLOB_SHA256 ..." after seed-member std modules changed
/// without the pin moving, while the shorter command line above was the only
/// control and nothing in the local gates ran it.
fn consumer_build(profile: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let tmp = tempfile::tempdir().expect("private TMPDIR");
    std::fs::write(dir.path().join("main.mind"), PROGRAM).expect("write program");
    let out = dir.path().join("program");
    let res = Command::new(MINDC)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("TMPDIR", tmp.path())
        .current_dir(dir.path())
        .args([
            "build",
            "main.mind",
            "--backend",
            "native",
            "--target",
            "cpu",
            "--emit",
            "binary",
            "--optimize",
            "debug",
            "--no-cache",
            "--native-profile",
            profile,
            "--out",
        ])
        .arg(&out)
        .output()
        .expect("spawn mindc");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        !stderr.contains("SEED_BLOB_SHA256"),
        "{profile}: the embedded std blob was refused against its pin:\n{stderr}"
    );
    assert_eq!(
        res.status.code(),
        Some(0),
        "{profile}: consumer command line failed:\n{stderr}"
    );
    let elf = std::fs::read(&out).expect("the build must write the artifact");
    assert_eq!(run(&elf), 7, "{profile}: consumer artifact exit");
}

#[test]
fn the_consumer_command_line_builds_with_the_embedded_bounded_seed() {
    consumer_build("bounded");
}

#[test]
fn the_consumer_command_line_builds_with_the_embedded_default_seed() {
    consumer_build("default");
}

/// The downstream consumer `w_set` shape (a store intrinsic's result returned and summed)
/// plus a full-width `i64 as f64` / `f64 as i64` round trip. Only the
/// native-lane-v1 admission path binds the intrinsics and admits these casts;
/// the historical path refuses the program at the closure fence
/// (`__mind_alloc` has no definition, `closure.unresolved_edge`). The bridge
/// selects the lane only for a compiler carrying the `nblane01` mark, so with
/// the embedded seeds and no override this builds exactly when the reseeded
/// seeds carry it. Expected: every store result is the defined 0, so `bad` is
/// 0; slot 7 holds 7 * 3 + 1 = 22, and (22.0 + 0.5) as i64 truncates to 22.
const LANE_PROGRAM: &str = "fn lane_get(w_addr: i64, i: i64) -> i64 { __mind_load_i64(w_addr + i * 8) }\n\
     fn lane_set(w_addr: i64, i: i64, v: i64) -> i64 { __mind_store_i64(w_addr + i * 8, v) }\n\
     fn main() -> i64 {\n\
     \x20   let w: i64 = __mind_alloc(64);\n\
     \x20   let mut i: i64 = 0;\n\
     \x20   let mut bad: i64 = 0;\n\
     \x20   while i < 8 { bad = bad + lane_set(w, i, i * 3 + 1); i = i + 1; }\n\
     \x20   let f: f64 = lane_get(w, 7) as f64;\n\
     \x20   ((f + 0.5) as i64) + bad\n\
     }\n";

fn lane_build(profile: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let tmp = tempfile::tempdir().expect("private TMPDIR");
    std::fs::write(dir.path().join("main.mind"), LANE_PROGRAM).expect("write program");
    let out = dir.path().join("program");
    let res = Command::new(MINDC)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("TMPDIR", tmp.path())
        .current_dir(dir.path())
        .args([
            "build",
            "main.mind",
            "--backend",
            "native",
            "--target",
            "cpu",
            "--emit",
            "binary",
            "--optimize",
            "debug",
            "--no-cache",
            "--native-profile",
            profile,
            "--out",
        ])
        .arg(&out)
        .output()
        .expect("spawn mindc");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert_eq!(
        res.status.code(),
        Some(0),
        "{profile}: the embedded seed did not select native-lane-v1 for a lane-only \
         program:\n{stderr}"
    );
    let elf = std::fs::read(&out).expect("the build must write the artifact");
    assert_eq!(run(&elf), 22, "{profile}: lane program exit");
}

#[test]
fn the_embedded_bounded_seed_selects_native_lane_v1() {
    lane_build("bounded");
}

#[test]
fn the_embedded_default_seed_selects_native_lane_v1() {
    lane_build("default");
}

#[test]
fn an_explicit_missing_compiler_override_is_still_refused() {
    // The override keeps its meaning: a named compiler that does not exist is
    // an error, never a silent switch to the embedded copy.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("prog.mind"), PROGRAM).expect("write");
    let res = Command::new(MINDC)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("MINDC_NATIVE_ELF", dir.path().join("no-such-stage1.elf"))
        .current_dir(dir.path())
        .args([
            "build",
            "prog.mind",
            "--backend",
            "native",
            "--out",
            "x.elf",
        ])
        .output()
        .expect("spawn mindc");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert_eq!(res.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("MINDC_NATIVE_ELF is set"), "{stderr}");
    assert!(!dir.path().join("x.elf").exists());
}

#[test]
fn an_explicit_missing_std_override_is_still_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("prog.mind"), PROGRAM).expect("write");
    let res = Command::new(MINDC)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("MINDC_STD_DIR", dir.path().join("no-such-std"))
        .current_dir(dir.path())
        .args([
            "build",
            "prog.mind",
            "--backend",
            "native",
            "--out",
            "x.elf",
        ])
        .output()
        .expect("spawn mindc");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert_eq!(res.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("MINDC_STD_DIR is set"), "{stderr}");
    assert!(!dir.path().join("x.elf").exists());
}
