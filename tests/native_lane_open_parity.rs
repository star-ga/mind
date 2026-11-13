// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Native == MLIR execution parity for `__mind_open` (census R9) and for the
//! process-argument rows `__mind_argc` / `__mind_argv`.
//!
//! The census's `__mind_open` shapes (weights:243, naes_t1:732, selftest:65,
//! main:230,367) build natively and the call returns the same fd/values as
//! the MLIR build. A mutant that removes the `__mind_open` row from the
//! native lane's admission table turns this test red. The argv echo runs on
//! both backends with the same arguments; a mutant that drops the runtime's
//! argv store turns it red.

#![cfg(all(
    feature = "std-surface",
    feature = "mlir-build",
    target_os = "linux",
    target_arch = "x86_64"
))]

mod native_lane_extern_support;

use native_lane_extern_support::{Case, MINDC, repo, run};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::process::Command;

/// The MLIR build of `source` in its own temporary directory.
fn mlir_run(label: &str, source: &str) -> (i32, Vec<u8>) {
    let (_dir, artifact) = mlir_build(label, source);
    run(&artifact)
}

/// The MLIR-built program for `source`, kept alive by the returned directory.
fn mlir_build(label: &str, source: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("main.mind");
    let artifact = dir.path().join("program");
    std::fs::write(&source_path, source).unwrap();
    let built = Command::new(MINDC)
        .args(["build", "--backend", "mlir"])
        .arg(&source_path)
        .arg("--emit=binary")
        .arg(format!("--out={}", artifact.display()))
        .env("MINDC_STD_DIR", repo().join("std"))
        .env_remove("MINDC_NATIVE_ELF")
        .env_remove("MINDC_SO")
        .current_dir(repo())
        .output()
        .unwrap();
    assert_eq!(
        built.status.code(),
        Some(0),
        "{label}: the MLIR build failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    (dir, artifact)
}

/// Native and MLIR agree; `exit`, when given, pins the shared exit code.
fn parity(label: &str, source: &str, exit: Option<i32>) {
    let case = Case::new(true);
    let native = run(&case.builds(source));
    let mlir = mlir_run(label, source);
    assert_eq!(native, mlir, "{label}: native != MLIR");
    if let Some(exit) = exit {
        assert_eq!(native.0, exit, "{label}");
    }
}

/// weights:243 — open a path, return fd + 100. The path is "/dev/null"
/// (always exists), so fd >= 0 on both backends and the result is fd + 100.
#[test]
fn open_returns_fd_plus_100_on_both_backends() {
    let source = r#"
fn main() -> i64 {
    let p: i64 = __mind_alloc(16);
    __mind_store_i8(p + 0, 47);
    __mind_store_i8(p + 1, 100);
    __mind_store_i8(p + 2, 101);
    __mind_store_i8(p + 3, 118);
    __mind_store_i8(p + 4, 47);
    __mind_store_i8(p + 5, 110);
    __mind_store_i8(p + 6, 117);
    __mind_store_i8(p + 7, 108);
    __mind_store_i8(p + 8, 108);
    __mind_store_i8(p + 9, 0);
    let fd: i64 = __mind_open(p);
    return fd + 100;
}
"#;
    parity("open_fd_plus_100", source, None);
}

/// naes_t1:732 — open a path, check fd >= 0. "/dev/null" exists, so the
/// program returns 0 on both backends.
#[test]
fn open_nonnegative_check_on_both_backends() {
    let source = r#"
fn main() -> i64 {
    let p: i64 = __mind_alloc(16);
    __mind_store_i8(p + 0, 47);
    __mind_store_i8(p + 1, 100);
    __mind_store_i8(p + 2, 101);
    __mind_store_i8(p + 3, 118);
    __mind_store_i8(p + 4, 47);
    __mind_store_i8(p + 5, 110);
    __mind_store_i8(p + 6, 117);
    __mind_store_i8(p + 7, 108);
    __mind_store_i8(p + 8, 108);
    __mind_store_i8(p + 9, 0);
    let fd: i64 = __mind_open(p);
    if fd < 0 {
        return 1;
    }
    return 0;
}
"#;
    parity("open_nonnegative", source, Some(0));
}

/// selftest:65 — open a path, read one byte. "/dev/null" reads 0 bytes
/// (EOF), so the program returns 0 on both backends.
#[test]
fn open_and_read_one_byte_on_both_backends() {
    let source = r#"
fn main() -> i64 {
    let p: i64 = __mind_alloc(16);
    __mind_store_i8(p + 0, 47);
    __mind_store_i8(p + 1, 100);
    __mind_store_i8(p + 2, 101);
    __mind_store_i8(p + 3, 118);
    __mind_store_i8(p + 4, 47);
    __mind_store_i8(p + 5, 110);
    __mind_store_i8(p + 6, 117);
    __mind_store_i8(p + 7, 108);
    __mind_store_i8(p + 8, 108);
    __mind_store_i8(p + 9, 0);
    let fd: i64 = __mind_open(p);
    if fd < 0 {
        return 1;
    }
    let b: i64 = __mind_alloc(1);
    let n: i64 = __mind_read(fd, b, 1, 0 - 1);
    return n;
}
"#;
    parity("open_and_read", source, Some(0));
}

/// main:230 — open a path, use fd in a condition. "/dev/null" exists,
/// so the program returns 0 on both backends.
#[test]
fn open_condition_on_both_backends() {
    let source = r#"
fn main() -> i64 {
    let p: i64 = __mind_alloc(16);
    __mind_store_i8(p + 0, 47);
    __mind_store_i8(p + 1, 100);
    __mind_store_i8(p + 2, 101);
    __mind_store_i8(p + 3, 118);
    __mind_store_i8(p + 4, 47);
    __mind_store_i8(p + 5, 110);
    __mind_store_i8(p + 6, 117);
    __mind_store_i8(p + 7, 108);
    __mind_store_i8(p + 8, 108);
    __mind_store_i8(p + 9, 0);
    let fd: i64 = __mind_open(p);
    if fd >= 0 {
        return 0;
    }
    return 1;
}
"#;
    parity("open_condition", source, Some(0));
}

/// main:367 — open a path, return fd directly. The fd depends on the
/// process's open descriptors, so only parity is pinned.
#[test]
fn open_returns_fd_on_both_backends() {
    let source = r#"
fn main() -> i64 {
    let p: i64 = __mind_alloc(16);
    __mind_store_i8(p + 0, 47);
    __mind_store_i8(p + 1, 100);
    __mind_store_i8(p + 2, 101);
    __mind_store_i8(p + 3, 118);
    __mind_store_i8(p + 4, 47);
    __mind_store_i8(p + 5, 110);
    __mind_store_i8(p + 6, 117);
    __mind_store_i8(p + 7, 108);
    __mind_store_i8(p + 8, 108);
    __mind_store_i8(p + 9, 0);
    return __mind_open(p);
}
"#;
    parity("open_returns_fd", source, None);
}

/// Missing path: both backends return the hosted sentinel `-1` (`fd + 100`
/// exits 99). A mutant that drops the native fold leaves the raw syscall's
/// `-ENOENT` (`-2`) and the exits diverge (98 vs 99).
#[test]
fn open_missing_path_returns_the_hosted_sentinel() {
    let source = r#"
fn main() -> i64 {
    let p: i64 = __mind_alloc(16);
    __mind_store_i8(p + 0, 47);
    __mind_store_i8(p + 1, 110);
    __mind_store_i8(p + 2, 111);
    __mind_store_i8(p + 3, 47);
    __mind_store_i8(p + 4, 115);
    __mind_store_i8(p + 5, 117);
    __mind_store_i8(p + 6, 99);
    __mind_store_i8(p + 7, 104);
    __mind_store_i8(p + 8, 0);
    let fd: i64 = __mind_open(p);
    return fd + 100;
}
"#;
    parity("open_missing", source, Some(99));
}

/// Null path. The hosted runtime returns `-1` without calling `open`. The raw
/// syscall returns `-EFAULT`. The fold maps every negative to `-1`.
#[test]
fn open_null_path_returns_the_hosted_sentinel() {
    let source = "fn main() -> i64 { return __mind_open(0) + 100; }\n";
    parity("open_null", source, Some(99));
}

/// Mutant control: a program that calls `__mind_open` with the wrong arity
/// must refuse on the native lane. This turns red if the admission is broken.
#[test]
fn wrong_arity_refuses_on_native_lane() {
    let case = Case::new(true);
    // Zero arguments: lane.call.arity.
    let bad0 = "fn main() -> i64 { return __mind_open(); }";
    let (out, _) = case.build(bad0);
    assert_ne!(
        out.status.code(),
        Some(0),
        "zero-arg __mind_open must refuse"
    );
    // Two arguments: lane.call.arity.
    let bad2 = "fn main() -> i64 { return __mind_open(1, 2); }";
    let (out2, _) = case.build(bad2);
    assert_ne!(
        out2.status.code(),
        Some(0),
        "two-arg __mind_open must refuse"
    );
}

/// Echo each argument after argv[0] on its own line, then read past the end.
/// argv[argc] is the NULL terminator (exit 90 if not). The native row reads
/// with no bounds check, so argv[argc + 1] is envp[0]: the program echoes that
/// string (exit 91 if it reads 0, as a bounds-checked accessor would), and
/// argv[argc + 2] is the NULL that ends a one-entry environment (exit 92).
/// `arg_at` and the entry are spliced in by `argv_echo`.
const ARGV_ECHO_BODY: &str = r#"
    let nl: i64 = __mind_alloc(8);
    __mind_store_i8(nl, 10);
    let mut i: i64 = 1;
    while i < argc {
        let p: i64 = arg_at(argv, i);
        __mind_write(1, p, arg_len(p), 0 - 1);
        __mind_write(1, nl, 1, 0 - 1);
        i = i + 1;
    }
    if arg_at(argv, argc) != 0 {
        return 90;
    }
    let e: i64 = arg_at(argv, argc + 1);
    if e == 0 {
        return 91;
    }
    __mind_write(1, e, arg_len(e), 0 - 1);
    __mind_write(1, nl, 1, 0 - 1);
    if arg_at(argv, argc + 2) != 0 {
        return 92;
    }
    return argc;
}
"#;

/// The argv echo for one backend. The MLIR program reads the vector through
/// `__mind_argc` / `__mind_argv`. Native-lane-v1 does not admit those two
/// rows, so the native program reads the same vector through its entry
/// binding `fn main(argc, argv)`, where argv[i] is the word at argv + 8 * i:
/// the address the native `__mind_argv` row reads.
fn argv_echo(native: bool) -> String {
    let (entry, arg_at) = if native {
        (
            "fn main(argc: i64, argv: i64) -> i64 {",
            "__mind_load_i64(argv + 8 * i)",
        )
    } else {
        (
            "fn main() -> i64 {\n    let argc: i64 = __mind_argc();\n    let argv: i64 = 0;",
            "__mind_argv(i) + argv",
        )
    };
    format!(
        "fn arg_len(p: i64) -> i64 {{\n    let mut n: i64 = 0;\n    while __mind_load_i8(p + n) != 0 {{\n        n = n + 1;\n    }}\n    return n;\n}}\n\nfn arg_at(argv: i64, i: i64) -> i64 {{\n    return {arg_at};\n}}\n\n{entry}{ARGV_ECHO_BODY}"
    )
}

/// One argv echo case: the arguments, the exit code and the stdout bytes.
type EchoCase = (&'static [&'static [u8]], i32, &'static [u8]);

/// The one environment entry the argv echo runs with (non-ASCII value).
const ECHO_ENV: (&str, &[u8]) = ("B3", b"\xc3\xa9v");

/// Run `binary` with `args` and exactly the `ECHO_ENV` entry: exit code and
/// stdout.
fn run_with_args(binary: &Path, args: &[&[u8]]) -> (i32, Vec<u8>) {
    let ran = Command::new(binary)
        .args(args.iter().map(|a| OsStr::from_bytes(a)))
        .env_clear()
        .env(ECHO_ENV.0, OsStr::from_bytes(ECHO_ENV.1))
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(ran.stderr.is_empty(), "{}: {ran:?}", binary.display());
    let code = ran
        .status
        .code()
        .unwrap_or_else(|| panic!("{} died on a signal: {:?}", binary.display(), ran.status));
    (code, ran.stdout)
}

/// The MLIR build reads the process arguments as the native build does: the
/// same bytes for no arguments, for non-ASCII bytes and for an argument with a
/// space, and the native contract value past the end (NULL, then envp[0]).
#[test]
fn argv_echo_is_equal_on_both_backends() {
    let case = Case::new(true);
    let native = case.builds(&argv_echo(true));
    let (_dir, mlir) = mlir_build("argv_echo", &argv_echo(false));
    let cases: [EchoCase; 3] = [
        (&[], 1, b""),
        (
            &[b"\xc3\xa9t\xc3\xa9", b"\xff\x01\x7f"],
            3,
            b"\xc3\xa9t\xc3\xa9\n\xff\x01\x7f\n",
        ),
        (&[b"a b", b"", b"-"], 4, b"a b\n\n-\n"),
    ];
    let env_line = [ECHO_ENV.0.as_bytes(), &b"="[..], ECHO_ENV.1, &b"\n"[..]].concat();
    for (args, exit, stdout) in cases {
        let want = (exit, [stdout, &env_line[..]].concat());
        assert_eq!(run_with_args(&native, args), want, "native, args {args:?}");
        assert_eq!(run_with_args(&mlir, args), want, "MLIR, args {args:?}");
    }
}

/// std.io's `read_stdin_bytes` (one `__mind_read(0, ..)`), echoed back.
const STDIN_ECHO: &str = r#"
import std.io;

fn main() -> i64 {
    let buf: i64 = __mind_alloc(64);
    let n: i64 = read_stdin_bytes(buf, 64);
    __mind_write(1, buf, n, 0 - 1);
    return n;
}
"#;

/// Run `binary` with `input` on stdin: exit code and stdout.
fn run_with_stdin(binary: &Path, input: &[u8]) -> (i32, Vec<u8>) {
    use std::io::Write;
    let mut child = Command::new(binary)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let ran = child.wait_with_output().unwrap();
    assert!(ran.stderr.is_empty(), "{}: {ran:?}", binary.display());
    let code = ran
        .status
        .code()
        .unwrap_or_else(|| panic!("{} died on a signal: {:?}", binary.display(), ran.status));
    (code, ran.stdout)
}

/// std.io links on the MLIR backend (it was not a substrate module, so
/// `read_stdin_bytes` was an undefined reference) and reads the same bytes as
/// the native build: empty stdin, and a short non-ASCII input.
#[test]
fn std_io_stdin_echo_is_equal_on_both_backends() {
    let case = Case::new(true);
    let native = case.builds(STDIN_ECHO);
    let (_dir, mlir) = mlir_build("stdin_echo", STDIN_ECHO);
    for input in [&b""[..], &b"\xc3\xa9\xff\x00z\n"[..]] {
        let want = (input.len() as i32, input.to_vec());
        assert_eq!(
            run_with_stdin(&native, input),
            want,
            "native, stdin {input:?}"
        );
        assert_eq!(run_with_stdin(&mlir, input), want, "MLIR, stdin {input:?}");
    }
}
