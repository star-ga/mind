// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Manifest-captured import ownership must be shared by check and emit.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest(entry: &str, output: &str, export: &str) -> String {
    format!(
        "[package]\nname = \"{output}\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"{entry}\"\noutput = \"{output}\"\n\n\
         [targets.cpu]\nbackend = \"cpu\"\n\n\
         [exports]\nc_abi = [\"{export}\"]\n"
    )
}

fn check(mindc: &Path, root: &Path, source: &str) -> std::process::Output {
    Command::new(mindc)
        .current_dir(root)
        .args(["check", source])
        .output()
        .expect("run mindc check")
}

fn emit_and_read(mindc: &Path, root: &Path, source: &str, output: &Path) {
    let out = Command::new(mindc)
        .current_dir(root)
        .args([
            "--emit-shared",
            output.to_str().expect("shared output path"),
            source,
        ])
        .output()
        .expect("run mindc emit-shared");
    assert!(
        out.status.success() && output.is_file(),
        "captured scope emit failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         class Widget(ctypes.Structure):\n\
         \x20   _fields_ = [('value', ctypes.c_int64)]\n\
         f = lib.read\n\
         f.argtypes = [ctypes.POINTER(Widget)]\n\
         f.restype = ctypes.c_int64\n\
         assert f(ctypes.byref(Widget(42))) == 42\n\
         print('ok')\n",
        output.to_string_lossy()
    );
    let run = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run emitted shared library");
    assert!(
        run.status.success(),
        "captured scope artifact returned the wrong value:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn nested_project_import_scope_controls() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("nested import scope scratch");

    // The manifest entry is at the project root while its imported sibling is
    // under src/. Captured resolution must preserve crate.src.types_a.
    let root_entry = temp.path().join("root_entry");
    fs::create_dir_all(root_entry.join("src")).expect("root-entry src");
    fs::write(
        root_entry.join("Mind.toml"),
        manifest("main.mind", "root_entry_scope", "read"),
    )
    .expect("root-entry manifest");
    fs::write(
        root_entry.join("main.mind"),
        "fn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("root-entry main");
    fs::write(
        root_entry.join("src/types_a.mind"),
        "struct Widget {\n    value: i64,\n}\n",
    )
    .expect("root-entry sibling");
    fs::write(
        root_entry.join("src/consumer.mind"),
        "import types_a;\nfn read(w: types_a.Widget) -> i64 {\n    return w.value;\n}\n",
    )
    .expect("root-entry consumer");
    let out = check(&mindc, &root_entry, "src/consumer.mind");
    assert!(
        out.status.success(),
        "captured root-entry scope check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    emit_and_read(
        &mindc,
        &root_entry,
        "src/consumer.mind",
        &root_entry.join("root-entry.so"),
    );

    // Existing src-entry projects retain their exact crate.types_a path.
    let src_entry = temp.path().join("src_entry");
    fs::create_dir_all(src_entry.join("src")).expect("src-entry src");
    fs::write(
        src_entry.join("Mind.toml"),
        manifest("src/main.mind", "src_entry_scope", "read"),
    )
    .expect("src-entry manifest");
    fs::write(
        src_entry.join("src/main.mind"),
        "fn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("src-entry main");
    fs::write(
        src_entry.join("src/types_a.mind"),
        "struct Widget {\n    value: i64,\n}\n",
    )
    .expect("src-entry sibling");
    fs::write(
        src_entry.join("src/consumer.mind"),
        "import types_a;\nfn read(w: types_a.Widget) -> i64 {\n    return w.value;\n}\n",
    )
    .expect("src-entry consumer");
    let out = check(&mindc, &src_entry, "src/consumer.mind");
    assert!(
        out.status.success(),
        "src-entry scope control failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Duplicate stems are not resolved by first match. The qualified type
    // must fail closed when the captured owner is ambiguous.
    let ambiguous = temp.path().join("ambiguous");
    fs::create_dir_all(ambiguous.join("src/a")).expect("ambiguous a");
    fs::create_dir_all(ambiguous.join("src/b")).expect("ambiguous b");
    fs::write(
        ambiguous.join("Mind.toml"),
        manifest("main.mind", "ambiguous_scope", "read"),
    )
    .expect("ambiguous manifest");
    fs::write(
        ambiguous.join("main.mind"),
        "fn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("ambiguous main");
    for path in ["src/a/types_a.mind", "src/b/types_a.mind"] {
        fs::write(
            ambiguous.join(path),
            "struct Widget {\n    value: i64,\n}\n",
        )
        .expect("ambiguous sibling");
    }
    fs::write(
        ambiguous.join("src/consumer.mind"),
        "import types_a;\nfn read(w: types_a.Widget) -> i64 {\n    return w.value;\n}\n",
    )
    .expect("ambiguous consumer");
    let out = check(&mindc, &ambiguous, "src/consumer.mind");
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "ambiguous stem compiled: {rendered}");
    assert!(
        rendered.contains("E2002"),
        "ambiguous stem lost E2002: {rendered}"
    );
    assert!(
        !rendered.contains("panicked at"),
        "ambiguous stem panicked: {rendered}"
    );

    // An exact owner can be selected while a non-exported type remains
    // inaccessible to the consumer.
    let private = temp.path().join("private");
    fs::create_dir_all(private.join("src")).expect("private src");
    fs::write(
        private.join("Mind.toml"),
        manifest("main.mind", "private_scope", "read"),
    )
    .expect("private manifest");
    fs::write(
        private.join("main.mind"),
        "fn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("private main");
    fs::write(
        private.join("src/types_a.mind"),
        "export { helper }\nstruct Widget {\n    value: i64,\n}\nfn helper() -> i64 {\n    return 0;\n}\n",
    )
    .expect("private sibling");
    fs::write(
        private.join("src/consumer.mind"),
        "import types_a;\nfn read(w: types_a.Widget) -> i64 {\n    return w.value;\n}\n",
    )
    .expect("private consumer");
    let out = check(&mindc, &private, "src/consumer.mind");
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "private type compiled: {rendered}");
    assert!(
        rendered.contains("E2002"),
        "private type lost E2002: {rendered}"
    );
    assert!(
        !rendered.contains("panicked at"),
        "private type panicked: {rendered}"
    );
}

const LEDGER_LIB: &str = "import std.io;\nexport { lib_fn }\n\npub fn lib_fn() -> i64 {\n    \
                          print_bytes(string_addr(\"lib ok\\n\"), 7);\n    return 0;\n}\n";
const LEDGER_MAIN: &str =
    "import std.io;\nimport ledger_lib;\n\nfn main() -> i64 {\n    return lib_fn();\n}\n";
const LEDGER_PROJECT: &str = "[package]\nname = \"p3\"\nversion = \"0.1.0\"\n\n\
                              [build]\ntarget = \"cpu\"\nentry = \"main.mind\"\n";

/// `mindc build <args> --release --emit=binary --out=<dir>/../<name>` in `dir`.
fn build_binary(
    mindc: &Path,
    dir: &Path,
    paths: &[&str],
    name: &str,
) -> (std::process::Output, PathBuf) {
    let out = dir.with_file_name(name);
    let output = Command::new(mindc)
        .current_dir(dir)
        .args(["build", "--backend", "mlir"])
        .args(paths)
        .args(["--release", "--emit=binary"])
        .arg(format!("--out={}", out.display()))
        .output()
        .expect("run mindc build");
    (output, out)
}

fn ledger_dir(root: &Path, name: &str, main: &str, manifest: Option<&str>) -> PathBuf {
    let dir = root.join(name);
    // A repository boundary keeps the bounded project scan inside this row.
    fs::create_dir_all(dir.join(".git")).expect("row dir");
    fs::write(dir.join("ledger_lib.mind"), LEDGER_LIB).expect("sibling module");
    fs::write(dir.join("main.mind"), main).expect("entry");
    if let Some(text) = manifest {
        fs::write(dir.join("Mind.toml"), text).expect("manifest");
    }
    dir
}

fn assert_refused(
    row: &str,
    output: &std::process::Output,
    out: &Path,
    code: i32,
    needles: &[&str],
) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(code),
        "{row}: wrong exit status: {stderr}"
    );
    for needle in needles {
        assert!(stderr.contains(needle), "{row}: no `{needle}` in: {stderr}");
    }
    assert!(
        !stderr.contains("unsupported call"),
        "{row}: refused at a call: {stderr}"
    );
    assert!(!out.exists(), "{row}: a refused build wrote an artifact");
}

fn assert_prints_lib_ok(row: &str, out: &Path) {
    let run = Command::new(out).output().expect("run built binary");
    assert_eq!(run.status.code(), Some(0), "{row}: built binary failed");
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "lib ok\n",
        "{row}: wrong output"
    );
}

/// One build rule for a named entry. P1: no project, a local import is refused
/// at the import (E2042) and names the missing Mind.toml. P2: a second path is a usage
/// error, never ignored. P3: an entry named inside a project builds that
/// project's module graph and runs. P3b: the no-path build is unchanged.
#[test]
fn named_entry_build_rule_rows() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("build rule scratch");

    let p1 = ledger_dir(temp.path(), "p1", LEDGER_MAIN, None);
    let (output, out) = build_binary(&mindc, &p1, &["main.mind"], "p1.bin");
    assert_refused(
        "P1",
        &output,
        &out,
        1,
        &["error[build][E2042]", "ledger_lib", "add a Mind.toml"],
    );

    let lone_main = "import std.io;\n\nfn main() -> i64 {\n    return lib_fn();\n}\n";
    let p2 = ledger_dir(temp.path(), "p2", lone_main, None);
    let (output, out) = build_binary(&mindc, &p2, &["main.mind", "ledger_lib.mind"], "p2.bin");
    assert_refused(
        "P2",
        &output,
        &out,
        2,
        &["at most one entry path", "ledger_lib.mind"],
    );

    let p3 = ledger_dir(temp.path(), "p3", LEDGER_MAIN, Some(LEDGER_PROJECT));
    let (output, out) = build_binary(&mindc, &p3, &["main.mind"], "p3.bin");
    if !common::gate::compiled("nested_project_import_scope_run", &output) {
        return;
    }
    assert_prints_lib_ok("P3", &out);
    // Cold, so P3b proves the no-path build compiles, not a P3 cache hit.
    let (output, out) = build_binary(&mindc, &p3, &["--no-cache"], "p3b.bin");
    assert!(
        common::gate::compiled("nested_project_import_scope_run", &output),
        "P3b lost its capability after P3"
    );
    assert_prints_lib_ok("P3b", &out);
    assert_eq!(
        fs::read(p3.join("Mind.toml")).expect("manifest"),
        LEDGER_PROJECT.as_bytes(),
        "the project manifest was not restored"
    );
}

/// A killed build's leftover temporary manifest in a subdirectory of a real
/// project is never a project: the search continues upward past it, so the
/// named entry next to it builds the real project above and runs.
#[test]
fn leftover_below_a_real_project_builds_the_real_project() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("leftover scratch");
    let top = temp
        .path()
        .canonicalize()
        .expect("canonical scratch")
        .join("top");
    fs::create_dir_all(top.join(".git")).expect("repo boundary");
    fs::write(
        top.join("Mind.toml"),
        "[package]\nname = \"toprow\"\nversion = \"0.1.0\"\n\n\
         [build]\ntarget = \"cpu\"\nentry = \"sub/main.mind\"\n",
    )
    .expect("real manifest");
    // No repository boundary in `sub`: the search must climb to `top`.
    let sub = top.join("sub");
    fs::create_dir(&sub).expect("sub");
    fs::write(sub.join("ledger_lib.mind"), LEDGER_LIB).expect("sibling module");
    fs::write(sub.join("main.mind"), LEDGER_MAIN).expect("entry");
    let leftover = sub.join("Mind.toml");
    fs::write(
        &leftover,
        libmind::build::temporary_manifest("main", "main.mind"),
    )
    .expect("leftover");
    let output = Command::new(&mindc)
        .current_dir(&sub)
        .args([
            "build",
            "--backend",
            "mlir",
            "main.mind",
            "--release",
            "--emit=binary",
        ])
        .output()
        .expect("run mindc build");
    assert!(
        !leftover.exists(),
        "the leftover temporary manifest was not removed"
    );
    if !common::gate::compiled("nested_project_import_scope_run", &output) {
        return;
    }
    let artifact = common::reported_artifact(&output);
    assert!(
        artifact.starts_with(top.join("target")),
        "built under the leftover, not the real project: {}",
        artifact.display()
    );
    assert_prints_lib_ok("leftover below a project", &artifact);
}
