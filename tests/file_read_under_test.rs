// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A project must be able to read its own manifest under `mindc test` and
//! `mindc run`. `mindc check` already accepts the program.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const READER: &str = r#"
import std.fs;

fn manifest_path() -> i64 {
    let b: i64 = __mind_alloc(9);
    __mind_store_i8(b + 0, 77);
    __mind_store_i8(b + 1, 105);
    __mind_store_i8(b + 2, 110);
    __mind_store_i8(b + 3, 100);
    __mind_store_i8(b + 4, 46);
    __mind_store_i8(b + 5, 116);
    __mind_store_i8(b + 6, 111);
    __mind_store_i8(b + 7, 109);
    __mind_store_i8(b + 8, 0);
    return b;
}

fn manifest_len() -> i64 {
    let rec: i64 = read_to_string(manifest_path());
    return __mind_load_i64(rec + 8);
}

fn main() -> i64 {
    let n: i64 = manifest_len();
    if n <= 0 {
        return 1;
    }
    return 0;
}
"#;

const TEST_SRC: &str = r#"
import std.fs;

fn manifest_path() -> i64 {
    let b: i64 = __mind_alloc(9);
    __mind_store_i8(b + 0, 77);
    __mind_store_i8(b + 1, 105);
    __mind_store_i8(b + 2, 110);
    __mind_store_i8(b + 3, 100);
    __mind_store_i8(b + 4, 46);
    __mind_store_i8(b + 5, 116);
    __mind_store_i8(b + 6, 111);
    __mind_store_i8(b + 7, 109);
    __mind_store_i8(b + 8, 0);
    return b;
}

#[test]
fn reads_own_manifest() {
    let rec: i64 = read_to_string(manifest_path());
    let n: i64 = __mind_load_i64(rec + 8);
    assert n > 0, "manifest empty";
}
"#;

fn project() -> PathBuf {
    let root = common::scratch_dir("file-read-under-test").join("proj");
    fs::create_dir_all(root.join("tests")).expect("create tests dir");
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"file_read\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n",
    )
    .expect("write manifest");
    fs::write(root.join("main.mind"), READER).expect("write entry");
    fs::write(root.join("tests/read_manifest.mind"), TEST_SRC).expect("write test");
    root
}

fn run(root: &Path, args: &[&str]) -> (i32, String) {
    let out = Command::new(common::mindc_bin())
        .args(args)
        .current_dir(root)
        .output()
        .expect("spawn mindc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), text)
}

#[test]
fn check_accepts_a_manifest_read() {
    let root = project();
    let (code, text) = run(&root, &["check", "main.mind"]);
    assert_eq!(code, 0, "check should accept the program:\n{text}");
}

#[test]
fn test_reads_the_project_manifest() {
    let root = project();
    let (code, text) = run(&root, &["test", "tests/read_manifest.mind"]);
    assert_eq!(
        code, 0,
        "mindc test must read the project manifest:\n{text}"
    );
    assert!(
        text.contains("ok") || text.contains("passed"),
        "expected a passing test line:\n{text}"
    );
}

#[test]
fn run_reads_the_project_manifest() {
    let root = project();
    let (code, text) = run(&root, &["run", "main.mind"]);
    assert_eq!(code, 0, "mindc run must read the project manifest:\n{text}");
}
