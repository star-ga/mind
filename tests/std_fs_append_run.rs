// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `std.fs.append_string` RUNTIME gate: an append-only log without a
//! read-modify-write or a temp file.
//!
//! A native program writes "ab", then appends "cde" twice, the second time to
//! the existing file and once to a path that does not exist yet. The file
//! bytes are read back from Rust: appends land at end-of-file, never truncate,
//! and create a missing file; an unopenable path reports 0.
//!
//! Gate: `cargo test --features "mlir-build std-surface cross-module-imports"
//!                   --test std_fs_append_run`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::process::Command;

fn program(log: &str, fresh: &str, bad: &str) -> String {
    format!(
        r#"import std.fs;
import std.string;

fn cpath(text: String) -> String {{
    return string_push_byte(string_push_str(string_new(), text), 0);
}}

fn main() -> i64 {{
    let log: String = cpath("{log}");
    let fresh: String = cpath("{fresh}");
    let bad: String = cpath("{bad}");
    let head: String = "ab";
    let tail: String = "cde";
    let w0: i64 = write_string(string_addr(log), string_addr(head), string_len(head));
    let a1: i64 = append_string(string_addr(log), string_addr(tail), string_len(tail));
    let a2: i64 = append_string(string_addr(log), string_addr(tail), string_len(tail));
    let a3: i64 = append_string(string_addr(fresh), string_addr(tail), string_len(tail));
    let a4: i64 = append_string(string_addr(bad), string_addr(tail), string_len(tail));
    print(w0);
    print(a1);
    print(a2);
    print(a3);
    print(a4);
    return 0;
}}
"#
    )
}

#[test]
fn append_string_appends_creates_and_reports_failure() {
    let dir = common::scratch_dir("std_fs_append_run");
    let log = dir.join("log.txt");
    let fresh = dir.join("fresh.txt");
    let bad = dir.join("missing-dir").join("x.txt");
    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&fresh);
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"fa\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"fa\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    let source = program(
        &log.display().to_string(),
        &fresh.display().to_string(),
        &bad.display().to_string(),
    );
    std::fs::write(dir.join("main.mind"), source).expect("write main.mind");
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .output()
        .expect("run mindc build");
    if !common::gate::compiled("std_fs_append_run", &out) {
        return;
    }
    let run = Command::new(dir.join("target").join("debug").join("fa"))
        .output()
        .expect("run the built executable");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    assert_eq!(run.status.code(), Some(0), "stdout {stdout:?}");
    assert_eq!(
        stdout, "1\n1\n1\n1\n0\n",
        "write, three appends, one refusal"
    );
    assert_eq!(std::fs::read(&log).expect("read log"), b"abcdecde");
    assert_eq!(std::fs::read(&fresh).expect("read fresh"), b"cde");
    assert!(
        !bad.exists(),
        "a path under a missing directory is not created"
    );
}
