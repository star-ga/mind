// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! String equality, length and slice give one answer on every backend.
//!
//! A downstream consumer measured two defects:
//!
//! * `==` / `!=` on a `string` parameter or a string-returning call compiled
//!   to a HANDLE compare on the MLIR path: `same("abc", "abc")` returned 0 and
//!   `a != b` returned 1 for equal text, with no diagnostic. Only literals and
//!   immutable literal-bound locals were compared by bytes.
//! * `mindc test` refused every string comparison the build computed, and the
//!   specified std.string API (`string_len`, `string_slice`, `string_eq`)
//!   failed on a `string` parameter or a literal argument, because the record
//!   the interpreter passes was not readable as the std `String` value.
//!
//! The same source runs here through the MLIR shared-library build and through
//! `mindc test`, and every case must produce the SAME value on both. A string
//! compared with an operand whose type is not a string is refused with
//! `E2038` naming `string_eq`, by `mindc check`, `mindc build` and `mindc test`
//! alike, and never compiled to a handle compare.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_ops_backend_parity`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::{mindc_bin, scratch_dir};

use std::process::Command;

const HELPERS: &str = r#"use std.string;

struct Dec {
    name: string,
    n: i64,
}

fn same(a: string, b: string) -> i64 {
    if a == b { return 1 }
    return 0
}

fn differ(a: string, b: string) -> i64 {
    if a != b { return 1 }
    return 0
}

fn is_abc(a: string) -> i64 {
    if a == "abc" { return 1 }
    return 0
}

fn mk() -> string { return "abc" }

fn call_eq() -> i64 {
    let x = mk()
    if x == "abc" { return 1 }
    return 0
}

fn field_eq(d: Dec) -> i64 {
    if d.name == "cache" { return 1 }
    return 0
}

fn len_of(s: string) -> i64 { return string_len(s) }

fn slice_len(s: string) -> i64 { return string_len(string_slice(s, 1, 3)) }

fn slice_eq(s: string) -> i64 {
    if string_slice(s, 1, 3) == "bc" { return 1 }
    return 0
}

fn lib_eq(a: string, b: string) -> i64 { return string_eq(a, b) }

fn first_is_abc(xs: array<string>) -> i64 {
    if xs[0] == "abc" { return 1 }
    return 0
}
"#;

/// `(case, expression, expected)`: each case is a zero-argument `pub fn`.
const CASES: &[(&str, &str, i64)] = &[
    ("same_equal", "same(\"abc\", \"abc\")", 1),
    ("same_unequal", "same(\"abc\", \"abd\")", 0),
    ("differ_equal", "differ(\"abc\", \"abc\")", 0),
    ("differ_unequal", "differ(\"abc\", \"abd\")", 1),
    ("param_literal_equal", "is_abc(\"abc\")", 1),
    ("param_literal_unequal", "is_abc(\"abd\")", 0),
    ("call_literal_equal", "call_eq()", 1),
    (
        "field_literal_equal",
        "field_eq(Dec { name: \"cache\", n: 1 })",
        1,
    ),
    (
        "field_literal_unequal",
        "field_eq(Dec { name: \"cachf\", n: 1 })",
        0,
    ),
    ("literal_literal_equal", "lit_lit()", 1),
    ("len_of_param", "len_of(\"abcd\")", 4),
    ("slice_len_param", "slice_len(\"abcd\")", 2),
    ("slice_literal_equal", "slice_eq(\"abcd\")", 1),
    ("std_string_eq_param", "lib_eq(\"abc\", \"abc\")", 1),
    ("std_string_eq_literals", "string_eq(\"abc\", \"abc\")", 1),
    ("std_string_len_literal", "string_len(\"abcd\")", 4),
    ("array_element_equal", "first_is_abc([\"abc\", \"x\"])", 1),
];

fn source() -> String {
    let mut src = String::from(HELPERS);
    src.push_str(
        "\nfn lit_lit() -> i64 {\n    if \"abc\" == \"abc\" { return 1 }\n    return 0\n}\n",
    );
    for (case, expr, _) in CASES {
        src.push_str(&format!("\npub fn {case}() -> i64 {{ return {expr} }}\n"));
    }
    src
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// The value each case produces under `mindc test` (the interpreter).
fn interpreter_values(dir: &std::path::Path) -> Vec<i64> {
    let mut src = source();
    for (i, (case, _, _)) in CASES.iter().enumerate() {
        // Each test asserts every candidate value, so the failure names the
        // value the interpreter computed instead of only "not expected".
        src.push_str(&format!(
            "\n#[test]\nfn t{i:02}_{case}() {{\n    let v: i64 = {case}()\n"
        ));
        for candidate in 0..=4 {
            src.push_str(&format!(
                "    assert v != {candidate}, \"value of {case} is {candidate}.\"\n"
            ));
        }
        src.push_str("}\n");
    }
    let path = dir.join("parity_interp.mind");
    std::fs::write(&path, src).expect("write interpreter source");
    let out = Command::new(mindc_bin())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    let log = text(&out);
    CASES
        .iter()
        .map(|(case, _, _)| {
            (0..=4)
                .find(|candidate| log.contains(&format!("value of {case} is {candidate}.")))
                .unwrap_or_else(|| panic!("mindc test gave no value for `{case}`:\n{log}"))
        })
        .collect()
}

/// The value each case produces from the MLIR-built shared library.
fn mlir_values(dir: &std::path::Path) -> Option<Vec<i64>> {
    let src = dir.join("parity_mlir.mind");
    let so = dir.join("parity_mlir.so");
    std::fs::write(&src, source()).expect("write build source");
    let out = Command::new(mindc_bin())
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("spawn mindc build");
    if !common::gate::compiled("string_ops_backend_parity", &out) {
        return None;
    }
    let names: Vec<String> = CASES.iter().map(|(c, _, _)| format!("'{c}'")).collect();
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\n\
         for n in [{}]:\n    f = getattr(lib, n)\n    f.restype = ctypes.c_int64\n    print(f())\n",
        so.display(),
        names.join(", ")
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("spawn python3");
    assert!(
        ran.status.success(),
        "shared library run failed:\n{}",
        text(&ran)
    );
    Some(
        String::from_utf8_lossy(&ran.stdout)
            .lines()
            .map(|line| line.trim().parse().expect("an i64 per case"))
            .collect(),
    )
}

#[test]
fn string_cases_agree_between_mindc_test_and_the_mlir_build() {
    let dir = scratch_dir("string_ops_backend_parity");
    let expected: Vec<i64> = CASES.iter().map(|(_, _, v)| *v).collect();
    let interp = interpreter_values(&dir);
    assert_eq!(interp, expected, "mindc test values, in CASES order");
    if let Some(built) = mlir_values(&dir) {
        assert_eq!(built, expected, "MLIR build values, in CASES order");
        assert_eq!(built, interp, "the two backends disagree");
    }
}

/// Text compared with an `i64` parameter and with an `i64` call result. Each
/// would compile to a handle compare, which is false for equal text.
const REFUSED: &str = r#"fn handle() -> i64 { return 7 }

fn by_handle(h: i64) -> i64 {
    if h == "abc" { return 1 }
    return 0
}

fn by_call() -> i64 {
    let h = handle()
    if h != "abc" { return 1 }
    return 0
}

fn null_check(s: string) -> i64 {
    if s == 0 { return 1 }
    return 0
}
"#;

#[test]
fn text_against_an_untyped_operand_is_refused_by_check_build_and_test() {
    let dir = scratch_dir("string_ops_backend_parity_refusal");
    let src = dir.join("refused.mind");
    std::fs::write(&src, REFUSED).expect("write refusal source");

    let check = Command::new(mindc_bin())
        .args(["check", "--no-fmt", src.to_str().unwrap()])
        .output()
        .expect("spawn mindc check");
    let check_text = text(&check);
    assert!(!check.status.success(), "check accepted:\n{check_text}");
    assert_eq!(
        check_text.matches("E2038").count(),
        2,
        "one E2038 per refused comparison, none for the null check:\n{check_text}"
    );
    assert!(check_text.contains("string_eq"), "{check_text}");
    assert!(!check_text.contains("panicked"), "{check_text}");

    let so = dir.join("refused.so");
    let build = Command::new(mindc_bin())
        .args([src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()])
        .output()
        .expect("spawn mindc build");
    let build_text = text(&build);
    assert!(!build.status.success(), "build accepted:\n{build_text}");
    assert!(build_text.contains("E2038"), "{build_text}");
    assert!(!build_text.contains("panicked"), "{build_text}");
    assert!(!so.exists(), "a refused build left an artifact");

    let test_src = dir.join("refused_test.mind");
    std::fs::write(
        &test_src,
        format!(
            "{REFUSED}\n#[test]\nfn refused_handle() {{ assert by_handle(0) == 0, \"handle\" }}\n"
        ),
    )
    .expect("write refusal test source");
    let test = Command::new(mindc_bin())
        .args(["test", test_src.to_str().unwrap()])
        .output()
        .expect("spawn mindc test");
    let test_text = text(&test);
    assert!(!test.status.success(), "mindc test passed:\n{test_text}");
    assert!(test_text.contains("[E2038]"), "{test_text}");
    assert!(test_text.contains("string_eq"), "{test_text}");
}

/// Positive control for the refusal test: the same shapes with string-typed
/// operands check clean, so the refusal is about the operand type alone. A
/// reassigned string binding stays a string.
#[test]
fn typed_string_operands_check_clean() {
    let dir = scratch_dir("string_ops_backend_parity_control");
    let src = dir.join("typed.mind");
    std::fs::write(
        &src,
        "fn by_text(h: string) -> i64 {\n    if h == \"abc\" { return 1 }\n    return 0\n}\n\
         fn null_check(s: string) -> i64 {\n    if s == 0 { return 1 }\n    return 0\n}\n\
         fn reassigned(flag: i64) -> i64 {\n    let mut s: string = \"abc\"\n    \
         if flag == 1 { s = \"abd\" }\n    if s != \"abc\" { return 1 }\n    return 0\n}\n",
    )
    .expect("write control source");
    let check = Command::new(mindc_bin())
        .args(["check", "--no-fmt", src.to_str().unwrap()])
        .output()
        .expect("spawn mindc check");
    assert!(check.status.success(), "control refused:\n{}", text(&check));
}

/// The native backend (the frozen pure-MIND x86-64 compiler) either builds a
/// string comparison and computes the byte-wise answer, or refuses it at
/// build time with no artifact. It must never emit a handle compare.
#[test]
fn native_backend_computes_or_refuses_string_equality() {
    let dir = scratch_dir("string_ops_backend_parity_native");
    let src = dir.join("native_eq.mind");
    let elf = dir.join("native_eq");
    let _ = std::fs::remove_file(&elf);
    std::fs::write(
        &src,
        "fn same(a: string, b: string) -> i64 {\n    if a == b { return 1 }\n    return 0\n}\n\
         fn main() -> i64 {\n    if same(\"abc\", \"abc\") == 1 { return 1 }\n    return 2\n}\n",
    )
    .expect("write native source");
    let repo = env!("CARGO_MANIFEST_DIR");
    let out = Command::new(mindc_bin())
        .arg("build")
        .arg("--backend=native")
        .arg(format!("--out={}", elf.display()))
        .arg(&src)
        .current_dir(repo)
        .env("MINDC_STD_DIR", format!("{repo}/std"))
        .env(
            "MINDC_NATIVE_ELF",
            format!("{repo}/examples/mindc_mind/testdata/selfhost_loop/stage1.elf"),
        )
        .output()
        .expect("spawn mindc build --backend=native");
    let log = text(&out);
    if out.status.success() {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&elf, std::fs::Permissions::from_mode(0o755))
            .expect("mark native artifact executable");
        let ran = Command::new(&elf).output().expect("run native artifact");
        assert_eq!(
            ran.status.code(),
            Some(1),
            "native byte-wise result:\n{log}"
        );
    } else {
        assert!(log.contains("error[backend-native]"), "{log}");
        assert!(!log.contains("panicked"), "{log}");
        assert!(!elf.exists(), "a refused native build left an artifact");
    }
}
