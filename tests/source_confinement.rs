// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! End-to-end project-boundary regression for `mindc check`.

mod common;

use std::fs;
use std::process::Command;

#[cfg(unix)]
use std::os::unix::fs::symlink;

#[test]
fn check_rejects_parent_escape_in_manifest_entry() {
    let root = common::scratch_dir("source_confinement").join("entry/project");
    let outside = root
        .parent()
        .expect("scratch parent")
        .join("source_confinement-outside");
    fs::create_dir_all(root.join("tests")).expect("project tests");
    fs::create_dir_all(root.join(".git")).expect("project boundary");
    fs::create_dir_all(&outside).expect("outside source dir");
    fs::write(
        root.join("Mind.toml"),
        concat!(
            "[package]\nname = \"source_confinement\"\n",
            "version = \"0.1.0\"\n\n[build]\n",
            "entry = \"../source_confinement-outside/main.mind\"\n"
        ),
    )
    .expect("manifest");
    fs::write(
        outside.join("main.mind"),
        "fn main() -> i32 { return 0; }\n",
    )
    .expect("outside entry");
    fs::write(
        outside.join("dep.mind"),
        "export { escaped }\nfn escaped() -> i64 { return 73; }\n",
    )
    .expect("outside dependency");
    fs::write(
        root.join("tests/escape.mind"),
        concat!(
            "import dep;\n#[test]\nfn outside_must_not_execute() { ",
            "assert dep.escaped() == 73, \"escape\"; }\n"
        ),
    )
    .expect("test source");

    let test_path = root.join("tests/escape.mind");
    let output = Command::new(common::mindc_bin())
        .current_dir(&root)
        .args(["check", test_path.to_str().expect("test path")])
        .output()
        .expect("run mindc check");
    assert!(
        !output.status.success(),
        "escaped manifest entry was accepted: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostics.contains("outside project root")
            || diagnostics.contains("project-root-relative")
            || diagnostics.contains("manifest entry"),
        "missing confinement diagnostic: {diagnostics}"
    );
}

#[cfg(unix)]
#[test]
fn check_rejects_symlink_escape_in_manifest_entry() {
    let root = common::scratch_dir("source_confinement_symlink").join("entry/project");
    let outside = root
        .parent()
        .expect("scratch parent")
        .join("source_confinement-symlink-outside");
    fs::create_dir_all(root.join("tests")).expect("project tests");
    fs::create_dir_all(root.join(".git")).expect("project boundary");
    fs::create_dir_all(&outside).expect("outside source dir");
    fs::write(
        root.join("Mind.toml"),
        concat!(
            "[package]\nname = \"source_confinement_symlink\"\n",
            "version = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n"
        ),
    )
    .expect("manifest");
    fs::create_dir_all(root.join("src")).expect("src");
    fs::write(
        outside.join("main.mind"),
        "fn main() -> i32 { return 0; }\n",
    )
    .expect("outside entry");
    symlink(outside.join("main.mind"), root.join("src/main.mind")).expect("entry link");
    fs::write(
        root.join("tests/escape.mind"),
        "import absent;\n#[test]\nfn refused() { assert false, \"must not run\"; }\n",
    )
    .expect("test source");

    let test_path = root.join("tests/escape.mind");
    let output = Command::new(common::mindc_bin())
        .current_dir(&root)
        .args(["check", test_path.to_str().expect("test path")])
        .output()
        .expect("run mindc check");
    assert!(
        !output.status.success(),
        "escaped entry symlink was accepted"
    );
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostics.contains("outside project root") || diagnostics.contains("manifest entry"),
        "missing symlink confinement diagnostic: {diagnostics}"
    );
}

#[test]
fn check_uses_the_validated_project_entry() {
    let root = common::scratch_dir("source_confinement_valid_check").join("project");
    fs::create_dir_all(root.join("src")).expect("src");
    fs::create_dir_all(root.join("tests")).expect("tests");
    fs::create_dir_all(root.join(".git")).expect("project boundary");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"source_confinement_valid_check\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n",
    )
    .expect("manifest");
    fs::write(
        root.join("src/main.mind"),
        "fn main() -> i64 { return 42; }\n",
    )
    .expect("main");
    fs::write(
        root.join("src/helper.mind"),
        "export { value }\nfn value() -> i64 { return 1; }\n",
    )
    .expect("helper");
    fs::write(
        root.join("tests/check.mind"),
        "import helper;\n\nfn entry() -> i64 {\n    return value();\n}\n",
    )
    .expect("check source");

    let source = root.join("tests/check.mind");
    let output = Command::new(common::mindc_bin())
        .current_dir(&root)
        .args(["check", source.to_str().expect("source path")])
        .output()
        .expect("run mindc check");
    assert!(
        output.status.success(),
        "valid project check was rejected: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(feature = "mlir-build")]
#[test]
fn build_uses_the_validated_project_entry() {
    let root = common::scratch_dir("source_confinement_valid_entry").join("project");
    fs::create_dir_all(root.join("src")).expect("src");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"source_confinement_valid\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n",
    )
    .expect("manifest");
    fs::write(
        root.join("src/main.mind"),
        "fn main() -> i64 { return 42; }\n",
    )
    .expect("main");
    fs::write(
        root.join("src/helper.mind"),
        "export { value }\nfn value() -> i64 { return 1; }\n",
    )
    .expect("helper");
    let build = Command::new(common::mindc_bin())
        .current_dir(&root)
        .args(["build", "--backend", "mlir", "--no-cache"])
        .output()
        .expect("run mindc build");
    assert!(
        build.status.success(),
        "valid project build was rejected: {}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
}

#[cfg(feature = "mlir-build")]
#[test]
fn build_refuses_an_escaping_manifest_entry() {
    let root = common::scratch_dir("source_confinement_entry_commands").join("project");
    let outside = root
        .parent()
        .expect("scratch parent")
        .join("source_confinement_entry_commands-outside");
    fs::create_dir_all(&root).expect("project");
    fs::create_dir_all(&outside).expect("outside");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"source_confinement_entry_commands\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"../source_confinement_entry_commands-outside/main.mind\"\n",
    )
    .expect("manifest");
    fs::write(
        outside.join("main.mind"),
        "fn main() -> i64 { return 99; }\n",
    )
    .expect("outside");
    let build = Command::new(common::mindc_bin())
        .current_dir(&root)
        .args(["build", "--backend", "mlir", "--no-cache"])
        .output()
        .expect("run mindc build");
    let diagnostics = format!(
        "{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        !build.status.success(),
        "build accepted an escaping manifest entry: {diagnostics}"
    );
    assert!(
        diagnostics.contains("outside project root")
            || diagnostics.contains("project-root-relative")
            || diagnostics.contains("manifest entry"),
        "build lacked a confinement diagnostic: {diagnostics}"
    );
}

/// A project with a broken `.mind` file in `docs/` that nothing imports, and
/// two `docs/` files on one module key (`crate.docs.notes`). `main` is the
/// entry text.
fn stray_docs_project(name: &str, main: &str) -> std::path::PathBuf {
    let root = common::scratch_dir(name).join("project");
    fs::create_dir_all(root.join("docs/notes")).expect("docs");
    fs::create_dir_all(root.join(".git")).expect("project boundary");
    let manifest = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
    );
    fs::write(root.join("Mind.toml"), manifest).expect("manifest");
    fs::write(root.join("main.mind"), main).expect("entry");
    fs::write(
        root.join("helper.mind"),
        "export { value }\nfn value() -> i64 { return 41; }\n",
    )
    .expect("helper");
    fs::write(
        root.join("docs/stray.mind"),
        "export { stray_value }\nfn stray_value() -> i64 { return missing_name; }\n",
    )
    .expect("stray");
    fs::write(
        root.join("docs/notes.mind"),
        "fn note() -> i64 { return 1; }\n",
    )
    .expect("notes");
    fs::write(
        root.join("docs/notes/mod.mind"),
        "fn note() -> i64 { return 2; }\n",
    )
    .expect("mod");
    root
}

/// Run `mindc <args>` in `root`; return success and the combined output.
fn mindc_in(root: &std::path::Path, args: &[&str]) -> (bool, String) {
    let output = Command::new(common::mindc_bin())
        .current_dir(root)
        .args(args)
        .output()
        .expect("run mindc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

/// `mindc check` (type-check pass only) of the project's entry.
fn check_entry(root: &std::path::Path) -> (bool, String) {
    let entry = root.join("main.mind");
    let entry = entry.to_str().expect("entry path");
    mindc_in(root, &["check", "--no-fmt", "--no-lint", entry])
}

#[test]
fn check_and_build_ignore_a_stray_docs_file() {
    let main = "import helper;\n\nfn main() -> i64 {\n    return value() + 1;\n}\n";
    let root = stray_docs_project("stray_docs_ignored", main);
    let (ok, out) = check_entry(&root);
    assert!(ok, "check analysed a stray docs file: {out}");
    #[cfg(feature = "mlir-build")]
    {
        let (ok, out) = mindc_in(&root, &["build", "--backend", "mlir", "--no-cache"]);
        assert!(ok, "build compiled a stray docs file: {out}");
        assert!(!out.contains("stray.mind"), "build named the stray: {out}");
    }
}

#[test]
fn check_and_build_see_an_imported_docs_file() {
    // Without the import the stray's function is not in scope.
    let main = "import helper;\n\nfn main() -> i64 {\n    return value() + stray_value();\n}\n";
    let root = stray_docs_project("stray_docs_unimported", main);
    let (ok, out) = check_entry(&root);
    assert!(
        !ok && out.contains("stray_value"),
        "unimported fn resolved: {out}"
    );

    let main = format!("import docs.stray;\n{main}");
    let root = stray_docs_project("stray_docs_imported", &main);
    let (ok, out) = check_entry(&root);
    assert!(ok, "check did not see the imported docs file: {out}");
    #[cfg(feature = "mlir-build")]
    {
        let (ok, out) = mindc_in(&root, &["build", "--backend", "mlir", "--no-cache"]);
        assert!(!ok, "build skipped the imported broken docs file: {out}");
        assert!(
            out.contains("stray.mind"),
            "build did not name the stray: {out}"
        );
    }
}
