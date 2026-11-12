// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg(all(unix, target_os = "linux", feature = "mlir-build"))]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(stem: &str, source: &str, backend: &str) -> (PathBuf, PathBuf) {
    let root = common::scratch_dir(stem);
    let user_dir = root.join("isolated-user");
    let empty_lib = root.join("empty-runtime-lib");
    fs::create_dir_all(root.join("src")).expect("create source directory");
    fs::create_dir_all(&user_dir).expect("create isolated user directory");
    fs::create_dir_all(&empty_lib).expect("create empty runtime directory");
    fs::write(root.join("src/main.mind"), source).expect("write source");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{stem}\"\nversion = \"0.1.0\"\n\n\
             [build]\nentry = \"src/main.mind\"\n\n\
             [targets.cpu]\nbackend = \"{backend}\"\n"
        ),
    )
    .expect("write manifest");
    (root, empty_lib)
}

fn build(root: &Path, empty_lib: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(["build", "--backend", "mlir", "--no-cache"])
        .current_dir(root)
        // These are subprocess-only test inputs. Together they prove the build
        // cannot see an ambient ~/.mind installation or a private link path.
        .env("HOME", root.join("isolated-user"))
        .env("MIND_LIB_DIR", empty_lib)
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("LIBRARY_PATH")
        .output()
        .expect("spawn mindc build")
}

#[test]
fn cpu_binary_needs_no_installed_runtime_and_is_deterministic() {
    let (root, empty_lib) = project(
        "public_cpu_native_link",
        "fn main() -> i64 { return 0; }\n",
        "cpu",
    );

    let first = build(&root, &empty_lib);
    assert!(
        first.status.success(),
        "public CPU build failed:\n{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let artifact = root.join("target/debug/public_cpu_native_link");
    let first_bytes = fs::read(&artifact).expect("read first artifact");
    assert_eq!(
        &first_bytes[..4],
        b"\x7fELF",
        "artifact is not a native ELF"
    );

    let run = Command::new(&artifact)
        .output()
        .expect("run native artifact");
    assert_eq!(run.status.code(), Some(0), "native artifact failed");

    let dynamic = Command::new("readelf")
        .args(["-d"])
        .arg(&artifact)
        .output()
        .expect("run readelf");
    assert!(dynamic.status.success(), "readelf rejected native artifact");
    let dynamic = String::from_utf8_lossy(&dynamic.stdout).to_ascii_lowercase();
    assert!(
        !dynamic.contains("libmind") && !dynamic.contains(".mind/lib"),
        "native ELF retained a private MIND runtime dependency:\n{dynamic}"
    );

    let second = build(&root, &empty_lib);
    assert!(
        second.status.success(),
        "repeat public CPU build failed:\n{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(
        first_bytes,
        fs::read(&artifact).expect("read repeated artifact"),
        "two --no-cache builds produced different native ELF bytes"
    );
}

#[test]
fn unresolved_native_symbol_fails_without_writing_a_launcher() {
    let (root, empty_lib) = project(
        "public_cpu_missing_symbol",
        "extern \"C\" { safe fn public_cpu_symbol_that_does_not_exist() -> i64; }\n\
         fn main() -> i64 { return public_cpu_symbol_that_does_not_exist(); }\n",
        "cpu",
    );
    let artifact = root.join("target/debug/public_cpu_missing_symbol");
    fs::create_dir_all(artifact.parent().expect("artifact parent")).expect("create target dir");
    fs::write(&artifact, b"#!/bin/sh\nexit 0\n").expect("seed stale launcher");
    let out = build(&root, &empty_lib);
    assert!(
        !out.status.success(),
        "unresolved native symbol linked successfully"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("public_cpu_symbol_that_does_not_exist"),
        "link failure did not identify the unresolved symbol:\n{stderr}"
    );
    assert!(
        !artifact.exists(),
        "failed native link left a stale artifact that could be mistaken for a binary"
    );
}

#[test]
fn invalid_module_fails_before_any_native_artifact() {
    let (root, empty_lib) = project(
        "public_cpu_invalid_module",
        "fn main( -> i64 { return 0; }\n",
        "cpu",
    );
    let artifact = root.join("target/debug/public_cpu_invalid_module");
    fs::create_dir_all(artifact.parent().expect("artifact parent")).expect("create target dir");
    fs::write(&artifact, b"\x7fELF stale").expect("seed stale executable");
    let out = build(&root, &empty_lib);
    assert!(!out.status.success(), "invalid module built successfully");
    assert!(
        !artifact.exists(),
        "invalid module left a stale runnable artifact"
    );
}

#[test]
fn accelerator_still_requires_its_installed_runtime() {
    let (root, empty_lib) = project(
        "public_cpu_accelerator_control",
        "fn main() -> i64 { return 0; }\n",
        "cuda",
    );
    let out = build(&root, &empty_lib);
    assert!(
        !out.status.success(),
        "CUDA build bypassed its runtime requirement"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("MIND runtime not found for backend 'cuda'"),
        "CUDA build no longer checked its installed runtime requirement:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---- Spec "Extern bindings" (bind clause, MLIR route) --------------------

/// Build `source` as a CPU project with `extra` build arguments; return the
/// build output, the artifact path and the run output when it built.
fn bind_case(stem: &str, source: &str, extra: &[&str]) -> (Output, PathBuf, Option<Output>) {
    let (root, empty_lib) = project(stem, source, "cpu");
    let out = Command::new(common::mindc_bin())
        .args(["build", "--backend", "mlir", "--no-cache"])
        .args(extra)
        .current_dir(&root)
        .env("HOME", root.join("isolated-user"))
        .env("MIND_LIB_DIR", &empty_lib)
        .output()
        .expect("spawn mindc build");
    let profile = if extra.contains(&"--release") {
        "release"
    } else {
        "debug"
    };
    let artifact = root.join("target").join(profile).join(stem);
    let run = out
        .status
        .success()
        .then(|| Command::new(&artifact).output().expect("run"));
    (out, artifact, run)
}

fn needed(artifact: &Path) -> Vec<String> {
    let out = Command::new("readelf")
        .arg("-d")
        .arg(artifact)
        .output()
        .expect("readelf");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split("Shared library: [").nth(1))
        .map(|l| l.trim_end_matches(']').to_string())
        .collect()
}

/// A NUL-terminated name in a fresh buffer (no string-literal layout assumed).
fn c_name(var: &str, name: &str) -> String {
    let mut s = format!("let {var}: i64 = __mind_alloc({});\n", name.len() + 1);
    for (i, b) in name.bytes().chain([0]).enumerate() {
        s += &format!("__mind_store_i8({var} + {i}, {b});\n");
    }
    s
}

#[test]
fn a_call_through_a_null_address_exits_78_and_writes_nothing() {
    let src = "extern \"C\" bind(.address) { fn add2(a: i64, b: i64) -> i64; }\n\
               fn main() -> i64 { let p: i64 = 0; return add2(p, 1, 2); }\n";
    for extra in [&[][..], &["--release"][..]] {
        let (out, _, run) = bind_case("bind_null_addr", src, extra);
        let run = run.unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(&out.stderr)));
        assert_eq!(run.status.code(), Some(78), "{extra:?}");
        assert!(run.stdout.is_empty() && run.stderr.is_empty(), "{extra:?}");
    }
}

#[test]
fn a_call_through_a_dlsym_address_reaches_the_c_function() {
    let src = format!(
        "extern \"C\" {{ fn dlsym(h: *mut u8, name: *const u8) -> *mut u8; }}\n\
         extern \"C\" bind(.address) {{ fn labs(x: i64) -> i64; fn abs(x: i32) -> i32; }}\n\
         fn main() -> i64 {{\n{}{}let f: i64 = dlsym(0, n);\nlet g: i64 = dlsym(0, m);\n\
         if f == 0 {{ return 90; }}\nif g == 0 {{ return 91; }}\n\
         return labs(f, 0 - 5) + abs(g, 0 - 7) * 10;\n}}\n",
        c_name("n", "labs"),
        c_name("m", "abs")
    );
    let (out, artifact, run) = bind_case("bind_dlsym_addr", &src, &[]);
    let run = run.unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(&out.stderr)));
    assert_eq!(run.status.code(), Some(75), "labs(-5) + abs(-7) * 10");
    assert!(!needed(&artifact).iter().any(|n| n.contains("labs")));
}

const CRC32: &str = "fn crc32(crc: u64, buf: *const u8, len: u32) -> u64;";

fn crc_main() -> String {
    format!(
        "fn main() -> i64 {{\n{}let c: i64 = crc32(0, b, 9);\n\
         if c == 3421780262 {{ return 42; }}\nreturn c & 255;\n}}\n",
        c_name("b", "123456789")
    )
}

#[test]
fn a_library_binding_links_the_named_library_after_libc() {
    let src = format!(
        "extern \"C\" bind(\"libz.so.1\") {{ {CRC32} }}\n{}",
        crc_main()
    );
    let (out, artifact, run) = bind_case("bind_libz", &src, &[]);
    let run = run.unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(&out.stderr)));
    assert_eq!(
        run.status.code(),
        Some(42),
        "crc32(\"123456789\") == 0xCBF43926"
    );
    let libs = needed(&artifact);
    let (c, z) = (
        libs.iter().position(|l| l == "libc.so.6"),
        libs.iter().position(|l| l == "libz.so.1"),
    );
    assert!(
        c.is_some() && z.is_some() && c < z,
        "DT_NEEDED order: {libs:?}"
    );
    // The same program with no clause: E6012, exit 1, no artifact.
    let bare = format!("extern \"C\" {{ {CRC32} }}\n{}", crc_main());
    let (out, artifact, run) = bind_case("bind_libz_missing", &bare, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(run.is_none() && out.status.code() == Some(1), "{stderr}");
    assert!(
        stderr.contains("E6012") && stderr.contains("crc32"),
        "{stderr}"
    );
    assert!(!artifact.exists());
}

#[test]
fn declared_libraries_link_in_ascending_name_order() {
    let src = format!(
        "extern \"C\" bind(\"libz.so.1\") {{ {CRC32} }}\n\
         extern \"C\" bind(\"libgcc_s.so.1\") {{ fn __popcountdi2(x: i64) -> i32; }}\n{}",
        crc_main()
    );
    let (out, artifact, run) = bind_case("bind_two_libs", &src, &[]);
    let run = run.unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(&out.stderr)));
    assert_eq!(run.status.code(), Some(42));
    let libs = needed(&artifact);
    let at = |n: &str| libs.iter().position(|l| l == n);
    assert!(
        at("libc.so.6") < at("libgcc_s.so.1") && at("libgcc_s.so.1") < at("libz.so.1"),
        "{libs:?}"
    );
    assert!(
        at("libgcc_s.so.1").is_some(),
        "a declared library is needed with no call: {libs:?}"
    );
}

#[test]
fn route_a_refuses_a_bind_clause_by_name() {
    let src = "extern \"C\" bind(\"libz.so.1\") { fn close(fd: i32) -> i32; }\n\
               fn main() -> i64 { return 0; }\n";
    let root = common::scratch_dir("bind_route_a");
    fs::create_dir_all(&root).expect("dir");
    let file = root.join("bind_route_a.mind");
    fs::write(&file, src).expect("write");
    let exe = root.join("bind_route_a.out");
    let out = Command::new(common::mindc_bin())
        .args(["build", "--backend", "native"])
        .arg(&file)
        .arg("--out")
        .arg(&exe)
        .output()
        .expect("spawn mindc build");
    let text =
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3), "{text}");
    assert!(text.contains("lane.extern.bind_clause"), "{text}");
    assert!(!exe.exists());
}

#[test]
fn a_loader_call_needs_allow_nondeterministic() {
    let root = common::scratch_dir("bind_dlopen_label");
    fs::create_dir_all(&root).expect("dir");
    let file = root.join("d.mind");
    fs::write(
        &file,
        "extern \"C\" { fn dlopen(p: *const u8, f: i32) -> *mut u8; }\n\
         fn main() -> i64 { let h: i64 = dlopen(0, 2); return h; }\n",
    )
    .expect("write");
    let obj = root.join("d.o");
    let run = |extra: &[&str]| {
        Command::new(common::mindc_bin())
            .arg(&file)
            .arg("--emit-obj")
            .arg(&obj)
            .args(extra)
            .output()
            .expect("spawn mindc")
    };
    let refused = run(&[]);
    assert!(!refused.status.success() && !obj.exists());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("dlopen"));
    assert!(run(&["--allow-nondeterministic"]).status.success() && obj.exists());
}
