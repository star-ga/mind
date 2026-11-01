// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! mindc test evaluates an imported enum variant that mindc check accepts.
//!
//! Reflective is tag 0 and Zero is tag 1. Comparing a Reflective value to
//! Zero must fail. Both the qualified spelling and the bare spelling run.

#![cfg(all(unix, feature = "std-surface", feature = "cross-module-imports"))]

use std::fs;
use std::path::Path;
use std::process::Command;

fn mindc() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
}

fn project(root: &Path, body: &str) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-eval\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/use.mind\"\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"src/config.mind\", \"src/use.mind\"]\n",
    )
    .unwrap();
    fs::write(
        root.join("src/config.mind"),
        "module config {\n    enum Boundary {\n        Reflective = 0,\n        Zero = 1,\n    }\n    export enum Boundary\n}\n",
    )
    .unwrap();
    fs::write(root.join("src/use.mind"), body).unwrap();
}

fn run_test(root: &Path) -> (bool, String) {
    let out = Command::new(mindc())
        .current_dir(root)
        .args(["test", "src/use.mind"])
        .output()
        .expect("run mindc test");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

const QUAL: &str = "\
import config;\n\
#[test]\n\
fn test_qual() {\n\
    let v = config.Boundary.Reflective\n\
    let z = config.Boundary.Zero\n\
    assert v != z, \"tags differ\"\n\
    assert v == config.Boundary.Reflective, \"reflective\"\n\
}\n";

const BARE: &str = "\
import config;\n\
#[test]\n\
fn test_bare() {\n\
    let v = Boundary.Reflective\n\
    let z = Boundary.Zero\n\
    assert v != z, \"tags differ\"\n\
    assert v == Boundary.Reflective, \"reflective\"\n\
}\n";

const WRONG_TAG: &str = "\
import config;\n\
#[test]\n\
fn test_wrong_tag() {\n\
    let v = config.Boundary.Reflective\n\
    assert v == config.Boundary.Zero, \"other tag\"\n\
}\n";

#[test]
fn qualified_imported_enum_evaluates() {
    let temp = tempfile::tempdir().unwrap();
    project(temp.path(), QUAL);
    let (ok, text) = run_test(temp.path());
    assert!(ok, "qualified enum test failed:\n{text}");
}

#[test]
fn bare_imported_enum_evaluates() {
    let temp = tempfile::tempdir().unwrap();
    project(temp.path(), BARE);
    let (ok, text) = run_test(temp.path());
    assert!(ok, "bare enum test failed:\n{text}");
}

#[test]
fn comparing_to_the_other_tag_is_red() {
    let temp = tempfile::tempdir().unwrap();
    project(temp.path(), WRONG_TAG);
    let (ok, text) = run_test(temp.path());
    assert!(!ok, "the other tag must fail:\n{text}");
    assert!(
        text.contains("other tag") || text.contains("FAILED"),
        "{text}"
    );
}
