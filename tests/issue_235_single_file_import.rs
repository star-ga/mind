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

#![cfg(unix)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const HELPER: &str = "pub fn graph_anchor(x: i64) -> i64 {\n    return x + 1;\n}\n";
const MAIN: &str = "import helper;\n\npub fn entry(x: i64) -> i64 {\n    return graph_anchor(x);\n}\n\nfn main() -> i32 {\n    return entry(1);\n}\n";

fn project(name: &str) -> PathBuf {
    let root = common::scratch_dir("issue-235-single-file").join(name);
    fs::create_dir_all(&root).expect("create project");
    fs::write(root.join("main.mind"), MAIN).expect("write entry");
    fs::write(root.join("helper.mind"), HELPER).expect("write helper");
    fs::write(root.join("stray.mind"), "fn stray( -> i64 {\n").expect("write hostile sibling");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n\
             [build]\nentry = \"main.mind\"\nemit = \"cdylib\"\n\n\
             [targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"helper.mind\"]\n"
        ),
    )
    .expect("write manifest");
    root
}

fn project_with_sources(
    name: &str,
    entry: &str,
    sources: &[&str],
    files: &[(&str, &str)],
) -> PathBuf {
    let root = common::scratch_dir("issue-235-single-file").join(name);
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    for (relative, source) in files {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("source parent")).expect("create source dir");
        fs::write(path, source).expect("write source");
    }
    let declared = sources
        .iter()
        .map(|source| format!("\"{source}\""))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n\
             [build]\nentry = \"{entry}\"\nemit = \"cdylib\"\n\n\
             [targets.cpu]\nbackend = \"cpu\"\nsources = [{declared}]\n"
        ),
    )
    .expect("write manifest");
    root
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(root)
        .output()
        .expect("spawn mindc")
}

fn assert_success(output: &Output, operation: &str) {
    assert!(
        output.status.success(),
        "{operation} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

unsafe fn call_entry(path: &Path) -> i64 {
    let library = unsafe { libloading::Library::new(path) }.expect("load emitted shared library");
    let entry: libloading::Symbol<'_, unsafe extern "C" fn(i64) -> i64> =
        unsafe { library.get(b"entry") }.expect("resolve entry");
    unsafe { entry(1) }
}

unsafe fn call_entry0(path: &Path) -> i64 {
    let library = unsafe { libloading::Library::new(path) }.expect("load emitted shared library");
    let entry: libloading::Symbol<'_, unsafe extern "C" fn() -> i64> =
        unsafe { library.get(b"entry") }.expect("resolve entry");
    unsafe { entry() }
}

#[test]
fn explicit_check_uses_the_enclosing_manifest_scope() {
    let root = project("issue235_check");
    let output = run(&root, &["check", "main.mind"]);
    assert_success(&output, "single-file check");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("E2003"), "{stderr}");
    assert!(!stderr.contains("unused import"), "{stderr}");
}

#[test]
fn flat_shared_emit_links_and_executes_the_imported_body_deterministically() {
    let root = project("issue235_flat_shared");
    let first = root.join("first.so");
    let second = root.join("second.so");

    let first_output = run(&root, &["main.mind", "--emit-shared", "first.so"]);
    assert_success(&first_output, "first flat shared emit");
    let first_bytes = fs::read(&first).expect("read first shared library");
    assert_eq!(unsafe { call_entry(&first) }, 2);

    let second_output = run(&root, &["main.mind", "--emit-shared", "second.so"]);
    assert_success(&second_output, "second flat shared emit");
    assert_eq!(unsafe { call_entry(&second) }, 2);
    assert_eq!(
        first_bytes,
        fs::read(second).expect("read second shared library"),
        "canonical import order did not produce deterministic shared-library bytes"
    );
}

#[test]
fn manifest_build_remains_the_positive_control() {
    let root = project("issue235_build_control");
    let output = run(
        &root,
        &[
            "build",
            "--backend=mlir",
            "main.mind",
            "--emit",
            "cdylib",
            "--out",
            "project.so",
            "--no-cache",
        ],
    );
    assert_success(&output, "manifest build control");
    assert_eq!(unsafe { call_entry(&root.join("project.so")) }, 2);
}

#[test]
fn nested_qualified_import_emits_a_usable_native_library() {
    let main = "import crate.modules.math.helper;\n\
                pub fn entry(x: i64) -> i64 { return graph_anchor(x); }\n\
                fn main() -> i32 { return entry(1); }\n";
    let root = project_with_sources(
        "issue235_nested_qualified",
        "main.mind",
        &["main.mind", "modules/math/helper.mind"],
        &[("main.mind", main), ("modules/math/helper.mind", HELPER)],
    );
    let artifact = root.join("qualified.so");
    let output = run(&root, &["main.mind", "--emit-shared", "qualified.so"]);
    assert_success(&output, "nested qualified shared emit");
    assert_eq!(unsafe { call_entry(&artifact) }, 2);
}

#[test]
fn explicit_src_scope_mixes_std_and_qualified_local_imports() {
    let main = "import std.vec;\n\
                import crate.src.helper;\n\
                pub fn entry() -> i64 {\n\
                    let values: Vec = vec_new();\n\
                    return graph_anchor(vec_len(values));\n\
                }\n\
                fn main() -> i32 { return entry(); }\n";
    let root = project_with_sources(
        "issue235_src_mixed",
        "src/main.mind",
        &["src/main.mind", "src/helper.mind"],
        &[("src/main.mind", main), ("src/helper.mind", HELPER)],
    );
    let artifact = root.join("mixed.so");
    let output = run(&root, &["src/main.mind", "--emit-shared", "mixed.so"]);
    assert_success(&output, "explicit src mixed-import shared emit");
    assert_eq!(unsafe { call_entry0(&artifact) }, 1);
}

#[test]
fn duplicate_basenames_link_distinct_objects_and_execute() {
    let main = "import crate.a.util;\n\
                import crate.b.util;\n\
                pub fn entry() -> i64 { return from_a() + from_b(); }\n\
                fn main() -> i32 { return entry(); }\n";
    let root = project_with_sources(
        "issue235_duplicate_basenames",
        "main.mind",
        &["main.mind", "a/util.mind", "b/util.mind"],
        &[
            ("main.mind", main),
            ("a/util.mind", "pub fn from_a() -> i64 { return 20; }\n"),
            ("b/util.mind", "pub fn from_b() -> i64 { return 22; }\n"),
        ],
    );
    let artifact = root.join("duplicate.so");
    let output = run(
        &root,
        &[
            "build",
            "--backend=mlir",
            "main.mind",
            "--emit",
            "cdylib",
            "--out",
            "duplicate.so",
            "--no-cache",
        ],
    );
    assert_success(&output, "duplicate-basename manifest build");
    assert_eq!(unsafe { call_entry0(&artifact) }, 42);
    assert!(root.join("__mod_5_crate_1_a_4_util.o").is_file());
    assert!(root.join("__mod_5_crate_1_b_4_util.o").is_file());
}

#[test]
fn no_manifest_import_is_explicitly_unsupported() {
    let root = common::scratch_dir("issue-235-single-file").join("issue235_no_manifest");
    fs::create_dir_all(&root).expect("create no-manifest fixture");
    fs::write(root.join("main.mind"), MAIN).expect("write entry");
    fs::write(root.join("helper.mind"), HELPER).expect("write helper");

    let output = run(&root, &["check", "main.mind"]);
    assert!(
        !output.status.success(),
        "no-manifest import unexpectedly resolved"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    // With import discovery the missing project is named at the import
    // (E2042); without it the unresolved call is all that can be refused.
    let code = if cfg!(feature = "cross-module-imports") {
        "E2042"
    } else {
        "E2003"
    };
    assert!(stderr.contains(code), "{stderr}");
    assert!(stderr.contains("Mind.toml"), "{stderr}");
}

#[test]
fn a_reachable_malformed_sibling_fails_before_artifact_emission() {
    let root = project("issue235_bad_import");
    fs::write(root.join("helper.mind"), "pub fn graph_anchor( -> i64 {\n")
        .expect("break imported helper");
    let artifact = root.join("bad.so");
    let output = run(&root, &["main.mind", "--emit-shared", "bad.so"]);
    assert!(
        !output.status.success(),
        "malformed imported sibling was ignored"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("helper.mind does not parse"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!artifact.exists(), "failed build left an artifact");
}
