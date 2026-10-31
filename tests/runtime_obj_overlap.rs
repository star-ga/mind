// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The runtime-support object behind `mindc --emit-shared`.
//!
//! The object is compiled on a thread that overlaps the mlir-opt /
//! mlir-translate leg and is joined before the shared link. Two contracts:
//!
//! * A clang failure in that overlapped compile still fails the build with the
//!   `clang` subprocess error. It is never swallowed into a silent success.
//! * `MINDC_RUNTIME_OBJ_CACHE=0` bypasses the disk cache (no read, no write)
//!   and changes no output byte. A poisoned cache entry proves both halves: it
//!   breaks the cached build and leaves the disabled build untouched.

#![cfg(unix)]

mod common;
use common::mindc_bin;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const TARGET: &str = "runtime_obj_overlap";
const CACHE_SWITCH: &str = "MINDC_RUNTIME_OBJ_CACHE";
const RUNTIME_SYMBOL: &[u8] = b"__mind_load_i64";
const INJECTED: &str = "runtime-obj-overlap: injected -x c failure";

const SRC: &str = "pub fn add(a: i64, b: i64) -> i64 {\n    return a + b\n}\n";

struct Case {
    dir: PathBuf,
    src: PathBuf,
}

impl Case {
    fn new(name: &str) -> Case {
        let dir = common::scratch_dir(TARGET).join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create case dir");
        let src = dir.join("add.mind");
        std::fs::write(&src, SRC).expect("write source");
        Case { dir, src }
    }

    fn cache_home(&self, name: &str) -> PathBuf {
        let d = self.dir.join(name);
        std::fs::create_dir_all(&d).expect("create cache home");
        d
    }

    /// `mindc add.mind --emit-shared <out>` with an isolated XDG cache home.
    fn emit(&self, out: &str, cache_home: &Path, env: &[(&str, &str)]) -> (Output, PathBuf) {
        let so = self.dir.join(out);
        let mut cmd = Command::new(mindc_bin());
        cmd.arg(&self.src)
            .arg("--emit-shared")
            .arg(&so)
            .env("XDG_CACHE_HOME", cache_home)
            .env_remove(CACHE_SWITCH);
        for (k, v) in env {
            cmd.env(k, v);
        }
        (cmd.output().expect("spawn mindc"), so)
    }
}

fn cached_objects(cache_home: &Path) -> Vec<PathBuf> {
    let dir = cache_home.join("mind").join("runtime-obj");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut objs: Vec<PathBuf> = entries.map(|e| e.expect("dir entry").path()).collect();
    objs.sort();
    objs
}

fn read_so(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.starts_with(b"\x7fELF"),
        "{} is not an ELF",
        path.display()
    );
    assert!(
        bytes
            .windows(RUNTIME_SYMBOL.len())
            .any(|w| w == RUNTIME_SYMBOL),
        "{} lacks the runtime-support object",
        path.display()
    );
    bytes
}

fn assert_ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cache_switch_is_byte_neutral_and_bypasses_the_cache() {
    let case = Case::new("switch");
    let home = case.cache_home("xdg-on");
    let (cold, cold_so) = case.emit("cold.so", &home, &[]);
    if !common::gate::compiled(TARGET, &cold) {
        return;
    }
    let entries = cached_objects(&home);
    assert_eq!(entries.len(), 1, "cold build must write one cache entry");

    let (warm, warm_so) = case.emit("warm.so", &home, &[]);
    assert_ok(&warm, "warm build");
    assert_eq!(
        cached_objects(&home),
        entries,
        "warm build rewrote the cache"
    );

    let off_home = case.cache_home("xdg-off");
    let (off, off_so) = case.emit("off.so", &off_home, &[(CACHE_SWITCH, "0")]);
    assert_ok(&off, "cache-disabled build");
    assert!(
        cached_objects(&off_home).is_empty(),
        "MINDC_RUNTIME_OBJ_CACHE=0 wrote a cache entry"
    );

    let cold_bytes = read_so(&cold_so);
    assert!(
        read_so(&warm_so) == cold_bytes,
        "warm build differs from cold"
    );
    assert!(
        read_so(&off_so) == cold_bytes,
        "cache-disabled build differs"
    );

    // Poison the one entry the key resolves to. The cached build must read it
    // (and fail to link garbage); the disabled build must not read it.
    let poisoned_home = case.cache_home("xdg-poisoned");
    let rel = entries[0]
        .strip_prefix(&home)
        .expect("entry under cache home");
    let poisoned = poisoned_home.join(rel);
    std::fs::create_dir_all(poisoned.parent().expect("entry parent")).expect("mkdir");
    std::fs::write(&poisoned, b"not an object file").expect("write poison");

    let (read_it, _) = case.emit("poisoned-on.so", &poisoned_home, &[]);
    assert!(
        !read_it.status.success(),
        "positive control: the cached build did not read the poisoned entry"
    );
    let (skipped_it, skipped_so) =
        case.emit("poisoned-off.so", &poisoned_home, &[(CACHE_SWITCH, "0")]);
    assert_ok(&skipped_it, "cache-disabled build over a poisoned cache");
    assert!(
        read_so(&skipped_so) == cold_bytes,
        "disabled build read the cache"
    );
}

/// A clang wrapper that fails ONLY the runtime-support compile (the one
/// invocation that passes `-x c`) when `RTO_FAIL=1`, and otherwise execs the
/// real clang unchanged.
fn write_clang_wrapper(dir: &Path, real_clang: &str) -> PathBuf {
    let script = dir.join("clang-wrapper.sh");
    let body = format!(
        "#!/bin/sh\nprev=\nfor a in \"$@\"; do\n  if [ \"$RTO_FAIL\" = 1 ] && [ \"$prev\" = -x ] && [ \"$a\" = c ]; then\n    echo '{INJECTED}' >&2\n    exit 3\n  fi\n  prev=$a\ndone\nexec '{real_clang}' \"$@\"\n"
    );
    std::fs::write(&script, body).expect("write clang wrapper");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    script
}

#[test]
fn runtime_compile_failure_surfaces_as_a_clang_error() {
    let case = Case::new("clang-fail");
    let home = case.cache_home("xdg-direct");
    let (direct, direct_so) = case.emit("direct.so", &home, &[(CACHE_SWITCH, "0")]);
    if !common::gate::compiled(TARGET, &direct) {
        return;
    }
    let tools = libmind::eval::mlir_build::resolve_tools().expect("toolchain resolved");
    let wrapper = write_clang_wrapper(&case.dir, &tools.clang);
    let wrapper = wrapper.to_str().expect("utf-8 wrapper path");

    // Positive control: the wrapper itself is a faithful clang.
    let (passthrough, passthrough_so) = case.emit(
        "passthrough.so",
        &case.cache_home("xdg-pass"),
        &[("CLANG", wrapper), (CACHE_SWITCH, "0")],
    );
    assert_ok(&passthrough, "build through the pass-through wrapper");
    assert!(
        read_so(&passthrough_so) == read_so(&direct_so),
        "the clang wrapper changed the output"
    );

    let (failed, failed_so) = case.emit(
        "failed.so",
        &case.cache_home("xdg-fail"),
        &[("CLANG", wrapper), (CACHE_SWITCH, "0"), ("RTO_FAIL", "1")],
    );
    let stderr = String::from_utf8_lossy(&failed.stderr);
    assert!(
        !failed.status.success(),
        "a failed runtime-support compile was reported as success"
    );
    assert!(
        stderr.contains("subprocess clang failed") && stderr.contains(INJECTED),
        "expected the clang subprocess error, got: {stderr}"
    );
    assert!(!failed_so.exists(), "a failed build left an output behind");
}
