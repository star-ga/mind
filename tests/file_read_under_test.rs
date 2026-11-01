// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A project must be able to read its own manifest under `mindc test` and
//! `mindc run`. `mindc check` already accepts the program.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test file_read_under_test`
//!
//! The repro program speaks the `std.fs` surface, so the whole file is gated on
//! `std-surface` (the low-level-only build rejects it by design). The `run` leg
//! additionally needs the native backend; a binary built without `mlir-build`
//! (or a host without the MLIR toolchain) refuses with a coded capability gap,
//! which [`common::gate`] skips legally — and which `MIND_BENCH_REQUIRE=1`
//! turns into a hard failure.

#![cfg(feature = "std-surface")]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const READER: &str = r#"import std.fs;

fn manifest_path() -> i64 {
    let b: i64 = __mind_alloc(10);
    __mind_store_i8(b + 0, 77);
    __mind_store_i8(b + 1, 105);
    __mind_store_i8(b + 2, 110);
    __mind_store_i8(b + 3, 100);
    __mind_store_i8(b + 4, 46);
    __mind_store_i8(b + 5, 116);
    __mind_store_i8(b + 6, 111);
    __mind_store_i8(b + 7, 109);
    __mind_store_i8(b + 8, 108);
    __mind_store_i8(b + 9, 0);
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

const TEST_SRC: &str = r#"import std.fs;

fn manifest_path() -> i64 {
    let b: i64 = __mind_alloc(10);
    __mind_store_i8(b + 0, 77);
    __mind_store_i8(b + 1, 105);
    __mind_store_i8(b + 2, 110);
    __mind_store_i8(b + 3, 100);
    __mind_store_i8(b + 4, 46);
    __mind_store_i8(b + 5, 116);
    __mind_store_i8(b + 6, 111);
    __mind_store_i8(b + 7, 109);
    __mind_store_i8(b + 8, 108);
    __mind_store_i8(b + 9, 0);
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

fn run_out(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(root)
        .output()
        .expect("spawn mindc")
}

fn text_of(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn run(root: &Path, args: &[&str]) -> (i32, String) {
    let out = run_out(root, args);
    (out.status.code().unwrap_or(-1), text_of(&out))
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
    // `mindc run` is project-mode: it reads Mind.toml for the entry (there is
    // no positional source argument). The program returns 1 when the manifest
    // read is empty and 0 when it read bytes, so exit 0 IS the read proof.
    let out = run_out(&root, &["run"]);
    if !common::gate::compiled("file_read_under_test::run", &out) {
        // Legal capability skip (no native backend in this binary / no MLIR
        // toolchain); MIND_BENCH_REQUIRE=1 turns this into a hard failure.
        return;
    }
    let text = text_of(&out);
    assert_eq!(
        out.status.code().unwrap_or(-1),
        0,
        "mindc run must read the project manifest:\n{text}"
    );
}

const MANIFEST_STORES: &str = r#"
    let b: i64 = __mind_alloc(10);
    __mind_store_i8(b + 0, 77);
    __mind_store_i8(b + 1, 105);
    __mind_store_i8(b + 2, 110);
    __mind_store_i8(b + 3, 100);
    __mind_store_i8(b + 4, 46);
    __mind_store_i8(b + 5, 116);
    __mind_store_i8(b + 6, 111);
    __mind_store_i8(b + 7, 109);
    __mind_store_i8(b + 8, 108);
    __mind_store_i8(b + 9, 0);
"#;

const BYTE_FNS: &str = r#"import std.fs;
import std.string;

fn manifest_path() -> i64 {
    let s = string_new();
    let s = string_push_byte(s, 77);
    let s = string_push_byte(s, 105);
    let s = string_push_byte(s, 110);
    let s = string_push_byte(s, 100);
    let s = string_push_byte(s, 46);
    let s = string_push_byte(s, 116);
    let s = string_push_byte(s, 111);
    let s = string_push_byte(s, 109);
    let s = string_push_byte(s, 108);
    let n: i64 = string_len(s);
    let b: i64 = __mind_alloc(n + 1);
    let mut i: i64 = 0;
    while i < n {
        __mind_store_i8(b + i, string_get_byte(s, i));
        i = i + 1;
    }
    __mind_store_i8(b + n, 0);
    return b;
}

fn emit_manifest_bytes() -> i64 {
    let rec: i64 = read_to_string(manifest_path());
    let addr: i64 = __mind_load_i64(rec + 0);
    let n: i64 = __mind_load_i64(rec + 8);
    let cap: i64 = __mind_load_i64(rec + 16);
    if n <= 0 {
        return 1;
    }
    if cap != n + 1 {
        return 1;
    }
    let mut i: i64 = 0;
    while i < n {
        print(__mind_load_i8(addr + i));
        i = i + 1;
    }
    return 0;
}
"#;

fn byte_project() -> PathBuf {
    let root = common::scratch_dir("file-read-under-test").join("bytes");
    fs::create_dir_all(root.join("tests")).expect("create tests dir");
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"file_read_bytes\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n",
    )
    .expect("write manifest");
    fs::write(
        root.join("main.mind"),
        format!("{BYTE_FNS}\nfn main() -> i64 {{\n    return emit_manifest_bytes();\n}}\n"),
    )
    .expect("write entry");
    fs::write(
        root.join("tests/read_manifest_bytes.mind"),
        format!(
            "{BYTE_FNS}\n#[test]\nfn reads_manifest_bytes() {{\n    let code: i64 = emit_manifest_bytes();\n    assert code == 0, \"manifest read failed\";\n}}\n"
        ),
    )
    .expect("write test");
    root
}

fn decode_printed_bytes(raw: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(raw);
    text.lines()
        .map(|line| {
            line.parse::<u8>()
                .unwrap_or_else(|_| panic!("printed line is not a byte: {line}"))
        })
        .collect()
}

#[test]
fn manifest_bytes_match_under_test_and_run() {
    let root = byte_project();
    let expected = fs::read(root.join("Mind.toml")).expect("read manifest bytes");
    let encoded: Vec<u8> = expected
        .iter()
        .flat_map(|byte| format!("{byte}\n").into_bytes())
        .collect();

    let tested = run_out(
        &root,
        &["test", "--threads", "1", "tests/read_manifest_bytes.mind"],
    );
    let test_text = text_of(&tested);
    assert_eq!(
        tested.status.code().unwrap_or(-1),
        0,
        "mindc test must read the manifest bytes:\n{test_text}"
    );
    assert_eq!(
        tested.stderr, encoded,
        "mindc test byte stream diverges from the file:\n{test_text}"
    );
    assert_eq!(decode_printed_bytes(&tested.stderr), expected);

    let ran = run_out(&root, &["run"]);
    if !common::gate::compiled("file_read_under_test::bytes", &ran) {
        return;
    }
    let run_text = text_of(&ran);
    assert_eq!(
        ran.status.code().unwrap_or(-1),
        0,
        "mindc run must read the manifest bytes:\n{run_text}"
    );
    assert_eq!(
        ran.stdout, encoded,
        "mindc run byte stream diverges from the file:\n{run_text}"
    );
    assert_eq!(tested.stderr, ran.stdout, "test and run outputs differ");
    assert_eq!(decode_printed_bytes(&ran.stdout), expected);
}

#[test]
fn write_string_is_refused() {
    let root = project();
    let src = format!(
        "import std.fs;\n\nfn manifest_path() -> i64 {{\n{MANIFEST_STORES}    return b;\n}}\n\n#[test]\nfn write_is_refused() {{\n    let p: i64 = manifest_path();\n    let _ = write_string(p, p, 9);\n}}\n"
    );
    fs::write(root.join("tests/write_refused.mind"), src).expect("write");
    let (code, text) = run(
        &root,
        &["test", "--threads", "1", "tests/write_refused.mind"],
    );
    assert_ne!(code, 0, "write_string must not succeed:\n{text}");
    assert!(
        text.contains("O_RDONLY"),
        "refusal must name O_RDONLY:\n{text}"
    );
    let low = text.to_ascii_lowercase();
    assert!(!low.contains("os error"));
    assert!(!low.contains("no such file"));
}

#[test]
fn open_descriptor_resets_across_tests() {
    let root = project();
    let src = format!(
        "extern \"C\" {{\n    fn open(path: i64, flags: i64, mode: i64) -> i64\n}}\n\nfn manifest_path() -> i64 {{\n{MANIFEST_STORES}    return b;\n}}\n\n#[test]\nfn leaves_a_file_open() {{\n    let fd: i64 = open(manifest_path(), 0, 0);\n    assert fd == 3, \"first descriptor\";\n}}\n\n#[test]\nfn next_test_gets_the_same_descriptor() {{\n    let fd: i64 = open(manifest_path(), 0, 0);\n    assert fd == 3, \"descriptor after the previous test\";\n}}\n"
    );
    fs::write(root.join("tests/fd_reset.mind"), src).expect("write");
    let (code, text) = run(&root, &["test", "--threads", "1", "tests/fd_reset.mind"]);
    assert_eq!(code, 0, "descriptor numbers must restart at 3:\n{text}");
}

#[test]
fn runtime_helpers_have_a_caller() {
    let root = project();
    let src = format!(
        "extern \"C\" {{\n    fn read_to_string(path_addr: i64) -> i64\n    fn string_push_byte(s: i64, b: i64) -> i64\n    fn vec_zeroed(n: i64) -> i64\n}}\n\nfn manifest_path() -> i64 {{\n{MANIFEST_STORES}    return b;\n}}\n\n#[test]\nfn helpers_match_native_records() {{\n    let z: i64 = vec_zeroed(2);\n    assert __mind_load_i64(z) == 0, \"zero slot\";\n    assert __mind_load_i64(z + 8) == 0, \"zero slot 2\";\n    let rec0: i64 = __mind_alloc(24);\n    let s: i64 = string_push_byte(rec0, 65);\n    assert __mind_load_i64(s + 8) == 1, \"len\";\n    assert __mind_load_i8(__mind_load_i64(s)) == 65, \"byte\";\n    let missing: i64 = read_to_string(0);\n    assert __mind_load_i64(missing) == 0, \"null addr\";\n    assert __mind_load_i64(missing + 8) == 0, \"null len\";\n    let rec: i64 = read_to_string(manifest_path());\n    let n: i64 = __mind_load_i64(rec + 8);\n    let cap: i64 = __mind_load_i64(rec + 16);\n    assert n > 0, \"file len\";\n    assert cap == n + 1, \"cap\";\n}}\n"
    );
    fs::write(root.join("tests/runtime_helpers.mind"), src).expect("write");
    let (code, text) = run(
        &root,
        &["test", "--threads", "1", "tests/runtime_helpers.mind"],
    );
    assert_eq!(code, 0, "runtime helpers must run:\n{text}");
}

const PAYLOAD_PATH: &str = r#"
fn payload_path() -> i64 {
    let b: i64 = __mind_alloc(12);
    __mind_store_i8(b + 0, 112);
    __mind_store_i8(b + 1, 97);
    __mind_store_i8(b + 2, 121);
    __mind_store_i8(b + 3, 108);
    __mind_store_i8(b + 4, 111);
    __mind_store_i8(b + 5, 97);
    __mind_store_i8(b + 6, 100);
    __mind_store_i8(b + 7, 46);
    __mind_store_i8(b + 8, 98);
    __mind_store_i8(b + 9, 105);
    __mind_store_i8(b + 10, 110);
    __mind_store_i8(b + 11, 0);
    return b;
}

fn main() -> i64 {
    let rec: i64 = read_to_string(payload_path());
    print(__mind_load_i64(rec + 8));
    print(__mind_load_i64(rec + 16));
    return 0;
}
"#;

fn payload_project(name: &str, header: &str) -> PathBuf {
    let root = common::scratch_dir("file-read-under-test").join(name);
    fs::create_dir_all(root.join(".git")).expect("project boundary");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("manifest");
    fs::write(root.join("payload.bin"), b"abcd").expect("payload");
    fs::write(root.join("main.mind"), format!("{header}\n{PAYLOAD_PATH}")).expect("entry");
    root
}

#[test]
fn c_and_std_read_records_match() {
    let c_root = payload_project(
        "c_payload",
        "extern \"C\" {\n    fn read_to_string(path_addr: i64) -> i64\n}",
    );
    let std_root = payload_project("std_payload", "import std.fs;");
    let c_out = run_out(&c_root, &["run"]);
    if !common::gate::compiled("file_read_under_test::c_record", &c_out) {
        return;
    }
    let std_out = run_out(&std_root, &["run"]);
    if !common::gate::compiled("file_read_under_test::std_record", &std_out) {
        return;
    }
    let c_text = text_of(&c_out);
    let std_text = text_of(&std_out);
    assert_eq!(c_out.status.code().unwrap_or(-1), 0, "c path:\n{c_text}");
    assert_eq!(
        std_out.status.code().unwrap_or(-1),
        0,
        "std path:\n{std_text}"
    );
    assert_eq!(c_out.stdout, b"4\n5\n", "c record fields:\n{c_text}");
    assert_eq!(
        std_out.stdout, c_out.stdout,
        "record fields differ:\nstd:\n{std_text}\nc:\n{c_text}"
    );
}
