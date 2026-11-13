// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A directory read returns the zero record (ptr 0, len 0, cap 0) from every
//! reader. Included by `tests/file_read_under_test.rs`, whose project and run
//! helpers it uses.
//!
//! Two readers answer a MIND `read_to_string` call, and the binding picks one:
//! - `import std.fs` binds the MIND body in std/fs.mind; both engines run it.
//! - an `extern "C"` declaration binds the runtime-support fallback: the
//!   evaluator intrinsic (src/eval/extern_c_intrinsics.rs) under `mindc test`,
//!   the weak C definition (runtime-support/mind_fs_read.h) under `mindc run`.
//!
//! Both fallbacks refuse a directory by file type (fstat S_ISDIR) before they
//! size it. A size check alone already refuses "." on ext4 (lseek(SEEK_END)
//! reports INT64_MAX) and on tmpfs (lseek fails), so only a directory whose
//! reported size is finite proves the type check. The fallback tests read
//! /proc: lseek(SEEK_END) on it reports 0.

use super::*;

/// `extern "C"` binding to the runtime-support reader (no `import std.fs`).
#[cfg(target_os = "linux")]
const READ_DECL: &str = "extern \"C\" {\n    fn read_to_string(path_addr: i64) -> i64\n}\n";

/// The reader plus the descriptor calls the evaluator legs also make.
#[cfg(target_os = "linux")]
const FALLBACK_DECLS: &str = "extern \"C\" {\n    fn read_to_string(path_addr: i64) -> i64\n    fn open(path: i64, flags: i64, mode: i64) -> i64\n    fn close(fd: i64) -> i64\n    fn is_dir_fd(fd: i64) -> i64\n}\n";

/// Evaluator legs. The arena is a bump allocator, so the next allocation lands
/// at `rec + 24` exactly when the reader allocated nothing past its record.
/// The file read is the probe's positive control: a buffer moves `next` on.
#[cfg(target_os = "linux")]
const FALLBACK_EVAL_TESTS: &str = r#"
#[test]
fn directory_is_refused_before_a_buffer() {
    let rec: i64 = read_to_string(dir_path());
    let next: i64 = __mind_alloc(8);
    assert __mind_load_i64(rec + 0) == 0, "dir ptr must be 0";
    assert __mind_load_i64(rec + 8) == 0, "dir len must be 0";
    assert __mind_load_i64(rec + 16) == 0, "dir cap must be 0";
    assert next == rec + 24, "a directory read must allocate its record only";
}

#[test]
fn file_read_allocates_a_buffer_past_the_record() {
    let rec: i64 = read_to_string(manifest_path());
    let next: i64 = __mind_alloc(8);
    assert __mind_load_i64(rec + 8) > 0, "manifest len";
    assert next > rec + 24, "the arena probe must see a file buffer";
}

#[test]
fn is_dir_fd_tells_a_directory_from_a_file() {
    let d: i64 = open(dir_path(), 0, 0);
    let f: i64 = open(manifest_path(), 0, 0);
    assert d >= 3, "open dir";
    assert f >= 3, "open file";
    assert is_dir_fd(d) == 1, "directory descriptor";
    assert is_dir_fd(f) == 0, "file descriptor";
    assert is_dir_fd(99) == 0, "descriptor that is not open";
    let _ = close(d);
    let _ = close(f);
}
"#;

/// Native leg: the record fields of a directory read through the C fallback.
#[cfg(target_os = "linux")]
const FALLBACK_RUN_MAIN: &str = r#"
fn main() -> i64 {
    let rec: i64 = read_to_string(dir_path());
    print(__mind_load_i64(rec + 0));
    print(__mind_load_i64(rec + 8));
    print(__mind_load_i64(rec + 16));
    return 0;
}
"#;

/// A MIND function `name() -> i64` that returns a NUL-terminated copy of `path`.
#[cfg(target_os = "linux")]
fn c_path_fn(name: &str, path: &str) -> String {
    let mut src = format!(
        "fn {name}() -> i64 {{\n    let b: i64 = __mind_alloc({});\n",
        path.len() + 1
    );
    for (i, byte) in path.bytes().enumerate() {
        src.push_str(&format!("    __mind_store_i8(b + {i}, {byte});\n"));
    }
    src.push_str(&format!(
        "    __mind_store_i8(b + {}, 0);\n    return b;\n}}\n",
        path.len()
    ));
    src
}

/// /proc, once the host shows the regime these tests need: a directory whose
/// lseek(SEEK_END) reports 0. Panics otherwise; a host where no size check is
/// fooled cannot tell the type check from its absence.
#[cfg(target_os = "linux")]
fn zero_size_dir() -> &'static str {
    use std::io::{Seek, SeekFrom};
    const DIR: &str = "/proc";
    let mut f = fs::File::open(DIR).unwrap_or_else(|e| panic!("open {DIR}: {e}"));
    let meta = f.metadata().unwrap_or_else(|e| panic!("fstat {DIR}: {e}"));
    assert!(meta.is_dir(), "{DIR} is not a directory");
    let end = f
        .seek(SeekFrom::End(0))
        .unwrap_or_else(|e| panic!("lseek {DIR}: {e}"));
    assert_eq!(end, 0, "lseek(SEEK_END) on {DIR} must report 0");
    DIR
}

#[cfg(target_os = "linux")]
fn fallback_project(name: &str) -> PathBuf {
    let dir = zero_size_dir();
    let root = isolated_dir(name);
    fs::create_dir_all(root.join("tests")).expect("create tests dir");
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("manifest");
    let dir_fn = c_path_fn("dir_path", dir);
    fs::write(
        root.join("main.mind"),
        format!("{READ_DECL}\n{dir_fn}{FALLBACK_RUN_MAIN}"),
    )
    .expect("entry");
    let manifest_fn = c_path_fn("manifest_path", "Mind.toml");
    fs::write(
        root.join("tests/dir_fallback.mind"),
        format!("{FALLBACK_DECLS}\n{dir_fn}\n{manifest_fn}{FALLBACK_EVAL_TESTS}"),
    )
    .expect("test");
    root
}

#[cfg(target_os = "linux")]
#[test]
fn fallback_reader_refuses_a_zero_size_directory_under_test() {
    let root = fallback_project("dir_fallback_test");
    let (code, text) = run(
        &root,
        &["test", "--threads", "1", "tests/dir_fallback.mind"],
    );
    assert_eq!(
        code, 0,
        "the evaluator reader must refuse a directory before it sizes it:\n{text}"
    );
    assert!(
        text.contains("3 passed; 0 failed"),
        "all three fallback tests must run and pass:\n{text}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn fallback_reader_refuses_a_zero_size_directory_under_run() {
    let root = fallback_project("dir_fallback_run");
    let out = run_out(&root, &["run", "--backend", "mlir"]);
    if !common::gate::compiled("file_read_under_test::dir_fallback", &out) {
        // Legal capability skip (no native backend / no MLIR toolchain).
        return;
    }
    let text = text_of(&out);
    assert_eq!(out.status.code().unwrap_or(-1), 0, "{text}");
    assert_eq!(
        out.stdout, b"0\n0\n0\n",
        "the C reader must return the zero record (ptr, len, cap) for a directory:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// std/fs.mind parity: `import std.fs` binds the MIND reader, so both legs below
// run the std/fs.mind body (under the evaluator, over the open/lseek/read
// intrinsics; under `mindc run`, compiled). It reads ".", which a size check
// already refuses on the usual filesystems, so it pins the record shape; the
// fallback tests above pin the type check itself.
// ---------------------------------------------------------------------------

const DIR_READER_STD: &str = r#"import std.fs;

fn dir_path() -> i64 {
    // NUL-terminated "." — the current directory is always a directory.
    let b: i64 = __mind_alloc(2);
    __mind_store_i8(b + 0, 46);
    __mind_store_i8(b + 1, 0);
    return b;
}

fn main() -> i64 {
    let rec: i64 = read_to_string(dir_path());
    let ptr: i64 = __mind_load_i64(rec + 0);
    let len: i64 = __mind_load_i64(rec + 8);
    let cap: i64 = __mind_load_i64(rec + 16);
    // A directory must yield the zero record on every filesystem.
    if ptr != 0 {
        return 1;
    }
    if len != 0 {
        return 2;
    }
    if cap != 0 {
        return 3;
    }
    return 0;
}
"#;

const DIR_TEST_STD: &str = r#"import std.fs;

fn dir_path() -> i64 {
    let b: i64 = __mind_alloc(2);
    __mind_store_i8(b + 0, 46);
    __mind_store_i8(b + 1, 0);
    return b;
}

#[test]
fn directory_returns_the_zero_record() {
    let rec: i64 = read_to_string(dir_path());
    let ptr: i64 = __mind_load_i64(rec + 0);
    let len: i64 = __mind_load_i64(rec + 8);
    let cap: i64 = __mind_load_i64(rec + 16);
    assert ptr == 0, "dir ptr must be 0";
    assert len == 0, "dir len must be 0";
    assert cap == 0, "dir cap must be 0";
}
"#;

fn dir_parity_project(name: &str) -> PathBuf {
    let root = isolated_dir(name);
    fs::create_dir_all(root.join("tests")).expect("create tests dir");
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("manifest");
    fs::write(root.join("main.mind"), DIR_READER_STD).expect("entry");
    fs::write(root.join("tests/dir_parity.mind"), DIR_TEST_STD).expect("test");
    root
}

#[test]
fn directory_read_is_the_zero_record_in_both_engines() {
    // std/fs.mind reader under the mindc test evaluator
    let root = dir_parity_project("dir_parity_test");
    let (code, text) = run(&root, &["test", "tests/dir_parity.mind"]);
    assert_eq!(
        code, 0,
        "mindc test evaluator must refuse a directory with the zero record:\n{text}"
    );
    assert!(
        text.contains("ok") || text.contains("passed"),
        "expected a passing test line:\n{text}"
    );

    // std/fs.mind reader compiled by mindc run
    let root2 = dir_parity_project("dir_parity_run");
    let out = run_out(&root2, &["run", "--backend", "mlir"]);
    if !common::gate::compiled("file_read_under_test::dir_parity", &out) {
        // Legal capability skip (no native backend / no MLIR toolchain).
        return;
    }
    let text2 = text_of(&out);
    assert_eq!(
        out.status.code().unwrap_or(-1),
        0,
        "native runtime must refuse a directory with the zero record:\n{text2}"
    );
}
