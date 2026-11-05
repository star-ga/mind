// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! A qualified enum variant resolves with and without a Mind.toml sources list.
//!
//! `src/config.mind` under an explicit sources list is module `crate.src.config`.
//! `config.Boundary.Reflective` must still resolve. The same program with no
//! sources list already resolved. Both checks must pass.
//!
//! A bare enum head that two imports export is ambiguous however each import
//! reaches its owner: directly (`crate.a.T`) or through the sources-list
//! fallback (`crate.src.b.T`).

#![cfg(all(unix, feature = "std-surface", feature = "cross-module-imports"))]

use std::fs;
use std::path::Path;
use std::process::Command;

fn mindc() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
}

fn write_sources(root: &Path) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/config.mind"),
        "module config {\n    enum Boundary {\n        Reflective = 0,\n        Zero = 1,\n    }\n    export enum Boundary\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("src/main.mind"),
        "import config;\nfn tag(bc: config.Boundary) -> i64 {\n    match bc {\n        config.Boundary.Reflective => { return 1 },\n        config.Boundary.Zero => { return 2 },\n    }\n}\nfn main() -> i64 {\n    return tag(config.Boundary.Reflective)\n}\n",
    )
    .unwrap();
}

fn check(root: &Path) -> (bool, String) {
    let out = Command::new(mindc())
        .current_dir(root)
        .args(["check", "--no-fmt", "--no-lint", "src/main.mind"])
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[test]
fn qualified_variant_resolves_without_a_sources_list() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    write_sources(root);
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-src\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/main.mind\"\n",
    )
    .unwrap();
    let (ok, text) = check(root);
    assert!(ok, "no sources list:\n{text}");
}

#[test]
fn qualified_variant_resolves_with_a_sources_list() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    write_sources(root);
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-src\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/main.mind\"\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"src/config.mind\", \"src/main.mind\"]\n",
    )
    .unwrap();
    let (ok, text) = check(root);
    assert!(
        ok,
        "explicit sources list must resolve config.Boundary.Reflective:\n{text}"
    );
    assert!(
        !text.contains("unknown module-qualified enum variant"),
        "{text}"
    );
}

fn enum_t(module: &str) -> String {
    format!(
        "module {module} {{\n    enum T {{\n        X = 0,\n        Y = 1,\n    }}\n    export enum T\n}}\n"
    )
}

fn two_owner_project(root: &Path, main: &str) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("a.mind"), enum_t("a")).unwrap();
    fs::write(root.join("src/b.mind"), enum_t("b")).unwrap();
    fs::write(root.join("src/main.mind"), main).unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-two\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/main.mind\"\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"a.mind\", \"src/b.mind\", \"src/main.mind\"]\n",
    )
    .unwrap();
}

#[test]
fn a_bare_head_exported_by_a_direct_and_a_fallback_import_is_ambiguous() {
    // `a.mind` is `crate.a`, reached by `import a` directly; `src/b.mind` is
    // `crate.src.b`, reached by `import b` only through the fallback.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    two_owner_project(
        root,
        "import a;\nimport b;\nfn main() -> i64 {\n    let v = T::Y;\n    match v {\n        T::X => { return 1 },\n        T::Y => { return 2 },\n    }\n}\n",
    );
    let (ok, text) = check(root);
    assert!(
        !ok,
        "a bare T named by two imports must not resolve to one:\n{text}"
    );
    assert!(text.contains("ambiguous enum `T`"), "{text}");
}

#[test]
fn the_qualified_head_resolves_in_the_same_layout() {
    // Positive control for the refusal above: the same two owners, each named
    // through its module, check clean.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    two_owner_project(
        root,
        "import a;\nimport b;\nfn main() -> i64 {\n    let v = a.T.Y;\n    let w = b.T.X;\n    match v {\n        a.T.X => { return 1 },\n        a.T.Y => { return 2 },\n    }\n}\n",
    );
    let (ok, text) = check(root);
    assert!(ok, "qualified a.T / b.T must resolve:\n{text}");
}
