// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! mindc test evaluates an imported enum variant that mindc check accepts.
//!
//! Reflective is tag 0 and Zero is tag 1. Comparing a Reflective value to
//! Zero must fail. Both the qualified spelling and the bare spelling run.
//! `==` on an enum with a payload variant is refused: the compiled program
//! compares those values by heap record, so no answer would match it.

#![cfg(all(unix, feature = "std-surface", feature = "cross-module-imports"))]

use std::fs;
use std::path::Path;
use std::process::Command;

fn mindc() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
}

const BOUNDARY: &str = "module config {\n    enum Boundary {\n        Reflective = 0,\n        Zero = 1,\n    }\n    export enum Boundary\n}\n";

fn project(root: &Path, body: &str) {
    project_with(root, BOUNDARY, body);
}

fn project_with(root: &Path, config: &str, body: &str) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-eval\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/use.mind\"\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"src/config.mind\", \"src/use.mind\"]\n",
    )
    .unwrap();
    fs::write(root.join("src/config.mind"), config).unwrap();
    fs::write(root.join("src/use.mind"), body).unwrap();
}

fn run_test(root: &Path) -> (bool, String) {
    run(root, &["test", "src/use.mind"])
}

fn run(root: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(mindc())
        .current_dir(root)
        .args(args)
        .output()
        .expect("run mindc");
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
        text.contains("other tag"),
        "the failure must be the tag assertion, not another error:\n{text}"
    );
}

#[test]
fn bare_spelling_checks_clean() {
    let temp = tempfile::tempdir().unwrap();
    project(temp.path(), BARE);
    let (ok, text) = run(
        temp.path(),
        &["check", "--no-fmt", "--no-lint", "src/use.mind"],
    );
    assert!(ok, "mindc check must accept Boundary.Reflective:\n{text}");
}

const LOCAL_FIELD: &str = "\
struct Cfg {\n\
    Zero: i64,\n\
}\n\
#[test]\n\
fn test_local_field() {\n\
    let Boundary = Cfg { Zero: 7 }\n\
    assert Boundary.Zero == 7, \"field read\"\n\
}\n";

#[test]
fn a_local_named_like_an_unimported_enum_keeps_its_field() {
    // This module does not import config, so `Boundary.Zero` is the field of
    // the local, not the enum variant the project registry also knows.
    let temp = tempfile::tempdir().unwrap();
    project(temp.path(), LOCAL_FIELD);
    let (ok, text) = run_test(temp.path());
    assert!(ok, "the local's field must be read:\n{text}");
}

const SHAPE: &str = "module config {\n    enum Shape {\n        Circle(i64),\n        Empty,\n    }\n    export enum Shape\n}\n";

const BOXED_EQ: &str = "\
import config;\n\
#[test]\n\
fn test_boxed_eq() {\n\
    let a = config.Shape.Empty\n\
    let b = config.Shape.Empty\n\
    assert a == b, \"boxed\"\n\
}\n";

#[test]
fn equality_on_an_imported_boxed_enum_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    project_with(temp.path(), SHAPE, BOXED_EQ);
    let (ok, text) = run_test(temp.path());
    assert!(
        !ok,
        "a compiled program compares these by heap record:\n{text}"
    );
    assert!(text.contains("heap record"), "{text}");
}

#[test]
fn single_file_enum_equality_follows_the_declaration() {
    let temp = tempfile::tempdir().unwrap();
    let decls =
        "enum Color {\n    Red,\n    Blue,\n}\nenum Shape {\n    Circle(i64),\n    Empty,\n}\n";
    let unit = format!(
        "{decls}#[test]\nfn test_unit() {{\n    let a = Color::Red\n    assert a == Color::Red, \"same\"\n    assert a != Color::Blue, \"other\"\n}}\n"
    );
    let boxed = format!(
        "{decls}#[test]\nfn test_boxed() {{\n    let a = Shape::Empty\n    assert a == Shape::Empty, \"boxed\"\n}}\n"
    );
    fs::write(temp.path().join("unit.mind"), unit).unwrap();
    fs::write(temp.path().join("boxed.mind"), boxed).unwrap();
    let (ok, text) = run(temp.path(), &["test", "unit.mind"]);
    assert!(ok, "a unit-only enum compares by variant:\n{text}");
    let (ok, text) = run(temp.path(), &["test", "boxed.mind"]);
    assert!(!ok, "a boxed enum is not compared:\n{text}");
    assert!(text.contains("heap record"), "{text}");
}
