// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Strings in slots declared otherwise, and the null string handle.
//!
//! A review measured three shapes that still compiled:
//!
//! * A string stored into an integer slot by a path the equality pass did not
//!   record: an element store (`xs[0] = "abc"`, `xs.push("abc")`,
//!   `xs.set(0, "abc")`), the payload of an `Option<string>` / `Result<string,
//!   E>` given to an `i64` parameter, field or result, and a string kept by a
//!   std collection or raw memory through an `i64` parameter (`vec_push`,
//!   `map_insert`, `vec_set`, `__mind_store_i64`) and read back through an
//!   `i64` result. `==` on two of them compiled to a handle compare: the MLIR
//!   build returned 0 for equal text with no diagnostic, while `mindc test`
//!   failed with another error. Each is now refused with `E2038` by `mindc
//!   check`, `mindc build` and `mindc test` alike.
//! * A `string` slot holding the null handle `0` crashed `==`: the byte-wise
//!   comparison read a record through it (SIGSEGV on the MLIR build, "memory
//!   access out of bounds" under `mindc test`). The null handle is now equal
//!   only to itself, and no record is read through it, on every backend.
//! * An integer other than the null handle given to a `string` parameter is
//!   no string record either; comparing that parameter with text is refused.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_slots_backend_parity`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::{mindc_bin, scratch_dir};

use std::path::Path;
use std::process::{Command, Output};

/// `(case, source)`: each source stores a string where the declaration says
/// otherwise, and `probe()` compares two such values with `==`.
const REFUSED: &[(&str, &str)] = &[
    (
        "element_index_assign",
        "pub fn probe() -> i64 {\n    let mut xs: [i64; 2] = [0, 0]\n    xs[0] = \"abc\"\n    \
         xs[1] = \"abc\"\n    if xs[0] == xs[1] { return 1 }\n    return 0\n}\n",
    ),
    (
        "element_push",
        "pub fn probe() -> i64 {\n    let mut xs: array<i64> = array<i64>.new()\n    \
         xs.push(\"abc\")\n    xs.push(\"abc\")\n    if xs[0] == xs[1] { return 1 }\n    \
         return 0\n}\n",
    ),
    (
        "element_set",
        "pub fn probe() -> i64 {\n    let mut xs: array<i64> = array<i64>.new()\n    \
         xs.push(0)\n    xs.push(0)\n    xs.set(0, \"abc\")\n    xs.set(1, \"abc\")\n    \
         if xs[0] == xs[1] { return 1 }\n    return 0\n}\n",
    ),
    (
        "element_of_a_field",
        "struct S { xs: [i64; 2] }\n\npub fn probe() -> i64 {\n    \
         let mut s = S { xs: [0, 0] }\n    s.xs[0] = \"abc\"\n    s.xs[1] = \"abc\"\n    \
         if s.xs[0] == s.xs[1] { return 1 }\n    return 0\n}\n",
    ),
    (
        "option_payload_into_param",
        "fn eqh(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    return 0\n}\n\n\
         fn get() -> Option<string> { return Some(\"abc\") }\n\n\
         pub fn probe() -> i64 {\n    match get() {\n        Some(x) => {\n            \
         match get() {\n                Some(y) => { return eqh(x, y) },\n                \
         None => { return 2 },\n            }\n        },\n        None => { return 3 },\n    \
         }\n}\n",
    ),
    (
        "result_payload_into_param",
        "fn eqh(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    return 0\n}\n\n\
         fn get() -> Result<string, i64> { return Ok(\"abc\") }\n\n\
         pub fn probe() -> i64 {\n    match get() {\n        Ok(x) => {\n            \
         match get() {\n                Ok(y) => { return eqh(x, y) },\n                \
         Err(e) => { return 2 },\n            }\n        },\n        Err(e) => { return 3 },\n    \
         }\n}\n",
    ),
    (
        "option_payload_into_field",
        "struct Box { v: i64 }\n\nfn get() -> Option<string> { return Some(\"abc\") }\n\n\
         fn unbox() -> Box {\n    match get() {\n        Some(x) => { return Box { v: x } },\n        \
         None => { return Box { v: 0 } },\n    }\n}\n\n\
         pub fn probe() -> i64 {\n    let a = unbox()\n    let b = unbox()\n    \
         if a.v == b.v { return 1 }\n    return 0\n}\n",
    ),
    (
        "option_payload_returned",
        "fn get() -> Option<string> { return Some(\"abc\") }\n\n\
         fn unwrap_i(o: Option<string>) -> i64 {\n    match o {\n        Some(x) => x,\n        \
         None => 0,\n    }\n}\n\n\
         pub fn probe() -> i64 {\n    if unwrap_i(get()) == unwrap_i(get()) { return 1 }\n    \
         return 0\n}\n",
    ),
    (
        "std_vec_push",
        "use std.vec;\n\npub fn probe() -> i64 {\n    let v0 = vec_new()\n    \
         let v1 = vec_push(v0, \"abc\")\n    let v2 = vec_push(v1, \"abc\")\n    \
         if vec_get(v2, 0) == vec_get(v2, 1) { return 1 }\n    return 0\n}\n",
    ),
    (
        "std_vec_set",
        "use std.vec;\n\npub fn probe() -> i64 {\n    let v0 = vec_new()\n    \
         let v1 = vec_push(v0, 0)\n    let v = vec_push(v1, 0)\n    vec_set(v, 0, \"abc\")\n    \
         vec_set(v, 1, \"abc\")\n    if vec_get(v, 0) == vec_get(v, 1) { return 1 }\n    \
         return 0\n}\n",
    ),
    (
        "std_map_insert",
        "use std.map;\n\npub fn probe() -> i64 {\n    let m0 = map_new()\n    \
         let m1 = map_insert(m0, 1, \"abc\")\n    let m2 = map_insert(m1, 2, \"abc\")\n    \
         if map_get(m2, 1) == map_get(m2, 2) { return 1 }\n    return 0\n}\n",
    ),
    (
        "raw_memory",
        "pub fn probe() -> i64 {\n    let p = __mind_alloc(16)\n    \
         __mind_store_i64(p, \"abc\")\n    __mind_store_i64(p + 8, \"abc\")\n    \
         if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n",
    ),
    (
        "integer_into_string_param",
        "fn is_abc(s: string) -> i64 {\n    if s == \"abc\" { return 1 }\n    return 0\n}\n\n\
         fn h(n: i64) -> i64 { return is_abc(n) }\n\npub fn probe() -> i64 { return h(5) }\n",
    ),
];

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn mindc(args: &[&str]) -> Output {
    Command::new(mindc_bin())
        .args(args)
        .output()
        .expect("spawn mindc")
}

#[test]
fn strings_stored_in_slots_declared_otherwise_are_refused_by_check_build_and_test() {
    let dir = scratch_dir("string_slots_backend_parity_refused");
    for (case, src) in REFUSED {
        let path = dir.join(format!("{case}.mind"));
        std::fs::write(&path, src).expect("write case source");
        let check = mindc(&["check", "--no-fmt", path.to_str().unwrap()]);
        let check_text = text(&check);
        assert!(
            !check.status.success(),
            "{case}: check accepted:\n{check_text}"
        );
        assert_eq!(
            check_text.matches("E2038").count(),
            1,
            "{case}: one E2038 for the one comparison:\n{check_text}"
        );
        assert!(check_text.contains("string_eq"), "{case}:\n{check_text}");
        assert!(!check_text.contains("panicked"), "{case}:\n{check_text}");

        let so = dir.join(format!("{case}.so"));
        let build = mindc(&[
            path.to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ]);
        let build_text = text(&build);
        assert!(
            !build.status.success(),
            "{case}: build accepted:\n{build_text}"
        );
        assert!(build_text.contains("E2038"), "{case}:\n{build_text}");
        assert!(!build_text.contains("panicked"), "{case}:\n{build_text}");
        assert!(!so.exists(), "{case}: a refused build left an artifact");

        let test_path = dir.join(format!("{case}_test.mind"));
        let test_src =
            format!("{src}\n#[test]\nfn probed() {{ assert probe() == 1, \"value\" }}\n");
        std::fs::write(&test_path, test_src).expect("write case test source");
        let test = mindc(&["test", test_path.to_str().unwrap()]);
        let test_text = text(&test);
        assert!(
            !test.status.success(),
            "{case}: mindc test passed:\n{test_text}"
        );
        assert!(test_text.contains("[E2038]"), "{case}:\n{test_text}");
        assert!(test_text.contains("string_eq"), "{case}:\n{test_text}");
    }
}

/// The null handle `0` in a `string` slot, and positive controls: integer
/// slots given integers and typed strings keep their comparisons.
const DEFINED: &str = r#"struct D { name: string }

fn same(a: string, b: string) -> i64 {
    if a == b { return 1 }
    return 0
}

fn find(c: i64) -> string {
    if c == 1 { return "abc" }
    return 0
}

fn null_name() -> i64 {
    let d = D { name: 0 }
    if d.name == "abc" { return 1 }
    return 0
}

fn ints() -> i64 {
    let mut xs: [i64; 2] = [0, 0]
    xs[0] = 7
    xs[1] = 7
    if xs[0] == xs[1] { return 1 }
    return 0
}

fn eqh(a: i64, b: i64) -> i64 {
    if a == b { return 1 }
    return 0
}

fn opt() -> Option<i64> { return Some(7) }

fn payload_of_ints() -> i64 {
    match opt() {
        Some(x) => { return eqh(x, 7) },
        None => { return 3 },
    }
}

fn found(c: i64) -> i64 {
    if find(c) == "abc" { return 1 }
    return 0
}
"#;

/// `(case, expression, expected)`: each case is a zero-argument `pub fn`.
const DEFINED_CASES: &[(&str, &str, i64)] = &[
    ("null_vs_text", "same(0, \"abc\")", 0),
    ("text_vs_null", "same(\"abc\", 0)", 0),
    ("null_vs_empty", "same(0, \"\")", 0),
    ("null_vs_null", "same(0, 0)", 1),
    ("text_vs_text", "same(\"abc\", \"abc\")", 1),
    ("null_result_vs_text", "found(0)", 0),
    ("text_result_vs_text", "found(1)", 1),
    ("null_field_vs_text", "null_name()", 0),
    ("int_elements", "ints()", 1),
    ("int_payload", "payload_of_ints()", 1),
];

fn defined_source() -> String {
    let mut src = String::from(DEFINED);
    for (case, expr, _) in DEFINED_CASES {
        src.push_str(&format!("\npub fn {case}() -> i64 {{ return {expr} }}\n"));
    }
    src
}

/// The value each case produces under `mindc test` (the interpreter).
fn interpreter_values(dir: &Path) -> Vec<i64> {
    let mut src = defined_source();
    for (i, (case, _, _)) in DEFINED_CASES.iter().enumerate() {
        src.push_str(&format!(
            "\n#[test]\nfn t{i:02}_{case}() {{\n    let v: i64 = {case}()\n"
        ));
        for candidate in 0..=3 {
            src.push_str(&format!(
                "    assert v != {candidate}, \"value of {case} is {candidate}.\"\n"
            ));
        }
        src.push_str("}\n");
    }
    let path = dir.join("defined_interp.mind");
    std::fs::write(&path, src).expect("write interpreter source");
    let log = text(&mindc(&["test", path.to_str().unwrap()]));
    DEFINED_CASES
        .iter()
        .map(|(case, _, _)| {
            (0..=3)
                .find(|candidate| log.contains(&format!("value of {case} is {candidate}.")))
                .unwrap_or_else(|| panic!("mindc test gave no value for `{case}`:\n{log}"))
        })
        .collect()
}

/// The value each case produces from the MLIR-built shared library, run in
/// one child process per case so a crash names its case.
fn mlir_values(dir: &Path) -> Option<Vec<i64>> {
    let src = dir.join("defined_mlir.mind");
    let so = dir.join("defined_mlir.so");
    std::fs::write(&src, defined_source()).expect("write build source");
    let check = mindc(&["check", "--no-fmt", src.to_str().unwrap()]);
    assert!(check.status.success(), "check refused:\n{}", text(&check));
    let out = mindc(&[src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()]);
    if !common::gate::compiled("string_slots_backend_parity", &out) {
        return None;
    }
    let values = DEFINED_CASES
        .iter()
        .map(|(case, _, _)| {
            let py = format!(
                "import ctypes\nlib = ctypes.CDLL(r'{}')\nf = lib.{case}\n\
                 f.restype = ctypes.c_int64\nprint(f())\n",
                so.display()
            );
            let ran = Command::new("python3")
                .args(["-c", &py])
                .output()
                .expect("spawn python3");
            assert!(ran.status.success(), "`{case}` crashed:\n{}", text(&ran));
            String::from_utf8_lossy(&ran.stdout)
                .trim()
                .parse()
                .expect("an i64")
        })
        .collect();
    Some(values)
}

#[test]
fn the_null_string_handle_compares_without_a_crash_on_every_backend() {
    let dir = scratch_dir("string_slots_backend_parity_defined");
    let expected: Vec<i64> = DEFINED_CASES.iter().map(|(_, _, v)| *v).collect();
    let interp = interpreter_values(&dir);
    assert_eq!(
        interp, expected,
        "mindc test values, in DEFINED_CASES order"
    );
    if let Some(built) = mlir_values(&dir) {
        assert_eq!(built, expected, "MLIR build values, in DEFINED_CASES order");
    }
}
