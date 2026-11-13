// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! Cache-key regressions for manifest exports and native C sources.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports",
    feature = "ffi-c-user"
))]

mod common;

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use common::mindc_bin;

fn run_build(mindc: &Path, root: &Path, out: &str) -> Output {
    Command::new(mindc)
        .current_dir(root)
        .args(["build", "--backend", "mlir", "--verbose", "--out", out])
        .output()
        .expect("run mindc build")
}

fn transcript(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_build(output: &Output) {
    assert!(
        output.status.success(),
        "mindc build failed:\n{}",
        transcript(output)
    );
}

unsafe fn chosen_value(path: &Path) -> i64 {
    let lib = unsafe { libloading::Library::new(path) }.unwrap();
    let chosen: libloading::Symbol<unsafe extern "C" fn() -> i64> =
        unsafe { lib.get(b"chosen\0") }.unwrap();
    unsafe { chosen() }
}

#[test]
fn explicit_entry_is_the_compiled_and_cached_entry() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        common::gate::skipped(
            "mindc_cache_build_inputs",
            "mindc binary unavailable for effective-entry cache regression",
        );
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir(root.join("src")).unwrap();
    fs::write(root.join("src/main.mind"), "fn chosen() -> i64 { 11 }\n").unwrap();
    fs::write(root.join("src/other.mind"), "fn chosen() -> i64 { 22 }\n").unwrap();
    let manifest = "[package]\nname=\"effective_entry\"\nversion=\"0.1.0\"\n\n\
                    [build]\nentry=\"src/main.mind\"\nemit=\"cdylib\"\n\n\
                    [targets.cpu]\nbackend=\"cpu\"\n\
                    sources=[\"src/main.mind\",\"src/other.mind\"]\n\n\
                    [exports]\nc_abi=[\"chosen\"]\n";
    fs::write(root.join("Mind.toml"), manifest).unwrap();

    let build = |out: &str, no_cache: bool, release: bool| {
        let mut command = Command::new(&mindc);
        command
            .current_dir(root)
            .args([
                "build",
                "--backend",
                "mlir",
                "src/other.mind",
                "--emit=cdylib",
                "--verbose",
            ])
            .args(["--out", out]);
        if no_cache {
            command.arg("--no-cache");
        }
        if release {
            command.arg("--release");
        }
        command.output().expect("run explicit-entry build")
    };

    let bypass = build("bypass.so", true, false);
    assert_build(&bypass);
    assert!(!transcript(&bypass).contains("[CACHE HIT]"));
    assert_eq!(unsafe { chosen_value(&root.join("bypass.so")) }, 22);
    assert_eq!(
        fs::read_to_string(root.join("Mind.toml")).unwrap(),
        manifest
    );

    let cold = build("cold.so", false, true);
    assert_build(&cold);
    assert!(!transcript(&cold).contains("[CACHE HIT]"));
    assert_eq!(unsafe { chosen_value(&root.join("cold.so")) }, 22);

    let warm = build("warm.so", false, true);
    assert_build(&warm);
    assert!(transcript(&warm).contains("[CACHE HIT]"));
    assert_eq!(unsafe { chosen_value(&root.join("warm.so")) }, 22);
    assert_eq!(
        fs::read(root.join("cold.so")).unwrap(),
        fs::read(root.join("warm.so")).unwrap()
    );
    assert_eq!(
        fs::read_to_string(root.join("Mind.toml")).unwrap(),
        manifest
    );
}

#[test]
fn changed_manifest_exports_never_restore_the_old_shared_library() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        common::gate::skipped(
            "mindc_cache_build_inputs",
            "mindc binary unavailable for export cache regression",
        );
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("src/main.mind"),
        "fn export_a() -> i64 { 11 }\nfn export_b() -> i64 { 22 }\n",
    )
    .unwrap();
    let manifest = |export: &str| {
        format!(
            "[package]\nname=\"export_cache\"\nversion=\"0.1.0\"\n\n\
             [build]\nentry=\"src/main.mind\"\nemit=\"cdylib\"\n\n\
             [targets.cpu]\nbackend=\"cpu\"\n\n[exports]\nc_abi=[\"{export}\"]\n"
        )
    };

    fs::write(root.join("Mind.toml"), manifest("export_a")).unwrap();
    let first = run_build(&mindc, root, "first.so");
    assert_build(&first);
    unsafe {
        let lib = libloading::Library::new(root.join("first.so")).unwrap();
        let export: libloading::Symbol<
            unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> i32,
        > = lib.get(b"mind_fn_export_a_v1_invoke\0").unwrap();
        assert_eq!(export(std::ptr::null(), 0, std::ptr::null_mut(), 0), 38);
    }

    fs::write(root.join("Mind.toml"), manifest("export_b")).unwrap();
    let second = run_build(&mindc, root, "second.so");
    assert_build(&second);
    assert!(
        !transcript(&second).contains("[CACHE HIT]"),
        "changed [exports].c_abi restored an artifact under the old key:\n{}",
        transcript(&second)
    );
    unsafe {
        let lib = libloading::Library::new(root.join("second.so")).unwrap();
        let export: libloading::Symbol<
            unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> i32,
        > = lib.get(b"mind_fn_export_b_v1_invoke\0").unwrap();
        assert_eq!(export(std::ptr::null(), 0, std::ptr::null_mut(), 0), 38);
    }

    let third = run_build(&mindc, root, "third.so");
    assert_build(&third);
    assert!(
        transcript(&third).contains("[CACHE HIT]"),
        "unchanged captured exports should produce a warm hit:\n{}",
        transcript(&third)
    );
    assert_eq!(
        fs::read(root.join("second.so")).unwrap(),
        fs::read(root.join("third.so")).unwrap(),
        "the warm export-cache restore must preserve exact artifact bytes"
    );
}

#[test]
fn native_source_builds_do_not_publish_a_stale_whole_artifact_key() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        common::gate::skipped(
            "mindc_cache_build_inputs",
            "mindc binary unavailable for native-source cache regression",
        );
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("src/main.mind"),
        "fn main() -> i64 { __mind_nerve_lut_exp_h() }\n",
    )
    .unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname=\"native_cache\"\nversion=\"0.1.0\"\n\n\
         [build]\nentry=\"src/main.mind\"\n\n\
         [targets.cpu]\nbackend=\"cpu\"\nnative_sources=[\"shim.c\"]\n",
    )
    .unwrap();

    fs::write(
        root.join("shim.c"),
        "long __mind_nerve_lut_exp_h(void) { return 1; }\n",
    )
    .unwrap();
    let first = run_build(&mindc, root, "first");
    assert_build(&first);
    assert_eq!(
        Command::new(root.join("first")).status().unwrap().code(),
        Some(1)
    );

    fs::write(
        root.join("shim.c"),
        "long __mind_nerve_lut_exp_h(void) { return 2; }\n",
    )
    .unwrap();
    let second = run_build(&mindc, root, "second");
    assert_build(&second);
    assert!(
        !transcript(&second).contains("[CACHE HIT]"),
        "native source edit reused the old whole artifact:\n{}",
        transcript(&second)
    );
    assert_eq!(
        Command::new(root.join("second")).status().unwrap().code(),
        Some(2)
    );

    let third = run_build(&mindc, root, "third");
    assert_build(&third);
    assert!(
        !transcript(&third).contains("[CACHE HIT]"),
        "path-fed native sources must remain cache-ineligible until their full input graph is bound"
    );
    assert_eq!(
        Command::new(root.join("third")).status().unwrap().code(),
        Some(2)
    );
    assert_eq!(
        fs::read(root.join("second")).unwrap(),
        fs::read(root.join("third")).unwrap(),
        "two uncached native-source builds from unchanged inputs must be deterministic"
    );
}
