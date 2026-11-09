// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Strings that reached an integer slot without a recorded store, string
//! literal `match` patterns, and string length / slice / trim.
//!
//! A review measured these shapes on the MLIR build and under `mindc test`:
//!
//! * A string reached an `i64` slot through a value whose type the equality
//!   pass did not establish (a tuple destructured from a tuple result, a value
//!   block that reassigns a local, a local assigned in a loop), through raw
//!   reads of a std collection's storage, or through another module of a
//!   project. `mindc check` accepted `==` on two such values and the MLIR build
//!   returned 0 for equal text. Each is now refused with `E2038` by `mindc
//!   check`, `mindc build` and `mindc test` (the project shapes by `mindc
//!   check` of the importing module and by `mindc build`).
//! * `match s { "abc" => 1, _ => 0 }` returned 1 on the MLIR build and 0
//!   under `mindc test`, and on an integer scrutinee it crashed the MLIR build.
//!   Both backends now compare bytes; a scrutinee not known to be a string is
//!   refused at check time.
//! * `.len()`, `.slice(a, b)` and `.trim()` on a string computed a value on the
//!   MLIR build and failed under `mindc test`; `.length` passed `mindc check`
//!   and was refused only by the build. The methods now give the same value
//!   on both backends, and `.length` is refused at check time.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_flows_backend_parity`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::{mindc_bin, scratch_dir};

use std::path::Path;
use std::process::{Command, Output};

const EQH: &str =
    "fn eqh(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    return 0\n}\n\n";

/// A value block whose result is a local it reassigns.
const BLOCK: &str = "if true { let mut t = \"x\"; t = \"abc\"; t } else { let mut u = \"y\"; \
                     u = \"abc\"; u }";

/// `(case, source, code)`: each source compares two strings held in integer
/// slots (or reads a member no string has) in `probe()`.
fn refused_cases() -> Vec<(&'static str, String, &'static str)> {
    vec![
        (
            "tuple_destructured",
            format!(
                "{EQH}fn pair() -> (i64, i64) {{ return (\"abc\", \"abc\") }}\n\n\
                 pub fn probe() -> i64 {{\n    let (a, b) = pair()\n    return eqh(a, b)\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_into_param",
            format!(
                "{EQH}pub fn probe() -> i64 {{\n    let a = {BLOCK}\n    let b = {BLOCK}\n    \
                 return eqh(a, b)\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_returned",
            format!(
                "fn mk() -> i64 {{\n    let a = {BLOCK}\n    return a\n}}\n\n\
                 pub fn probe() -> i64 {{\n    if mk() == mk() {{ return 1 }}\n    return 0\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_in_field",
            format!(
                "struct Box {{ v: i64 }}\n\npub fn probe() -> i64 {{\n    \
                 let x = Box {{ v: {BLOCK} }}\n    let y = Box {{ v: {BLOCK} }}\n    \
                 if x.v == y.v {{ return 1 }}\n    return 0\n}}\n"
            ),
            "E2038",
        ),
        (
            "loop_local_into_param",
            format!(
                "{EQH}pub fn probe() -> i64 {{\n    let mut a = 0\n    let mut b = 0\n    \
                 let mut i = 0\n    while i < 1 {{\n        let s = \"abc\"\n        a = s\n        \
                 b = s\n        i = i + 1\n    }}\n    return eqh(a, b)\n}}\n"
            ),
            "E2038",
        ),
        (
            "block_value_pushed",
            format!(
                "use std.vec;\n\npub fn probe() -> i64 {{\n    let a = {BLOCK}\n    \
                 let b = {BLOCK}\n    let v = vec_push(vec_push(vec_new(), a), b)\n    \
                 if vec_get(v, 0) == vec_get(v, 1) {{ return 1 }}\n    return 0\n}}\n"
            ),
            "E2038",
        ),
        (
            "vec_storage_read_raw",
            "use std.vec;\n\npub fn probe() -> i64 {\n    \
             let v = vec_push(vec_push(vec_new(), \"abc\"), \"abc\")\n    let p = vec_addr(v)\n    \
             if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "vec_set_storage_read_raw",
            "use std.vec;\n\npub fn probe() -> i64 {\n    \
             let v = vec_push(vec_push(vec_new(), 0), 0)\n    vec_set(v, 0, \"abc\")\n    \
             vec_set(v, 1, \"abc\")\n    let p = vec_addr(v)\n    \
             if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "map_storage_read_raw",
            "use std.map;\n\npub fn probe() -> i64 {\n    \
             let m = map_insert(map_insert(map_new(), 1, \"abc\"), 2, \"abc\")\n    \
             let p = map_vals_addr(m)\n    \
             if __mind_load_i64(p) == __mind_load_i64(p + 8) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "raw_store_read_by_vec_get",
            "use std.vec;\n\npub fn probe() -> i64 {\n    \
             let v = vec_push(vec_push(vec_new(), 0), 0)\n    let p = vec_addr(v)\n    \
             __mind_store_i64(p, \"abc\")\n    __mind_store_i64(p + 8, \"abc\")\n    \
             if vec_get(v, 0) == vec_get(v, 1) { return 1 }\n    return 0\n}\n"
                .to_string(),
            "E2038",
        ),
        (
            "literal_pattern_on_an_integer",
            "fn is_abc(s: i64) -> i64 {\n    match s {\n        \"abc\" => 1,\n        \
             _ => 0,\n    }\n}\n\npub fn probe() -> i64 { return is_abc(5) }\n"
                .to_string(),
            "E2038",
        ),
        (
            "length_member",
            "fn slen(s: string) -> i64 { return s.length }\n\n\
             pub fn probe() -> i64 { return slen(\"abc\") }\n"
                .to_string(),
            "E6009",
        ),
    ]
}

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
fn strings_reaching_integer_slots_are_refused_by_check_build_and_test() {
    let dir = scratch_dir("string_flows_backend_parity_refused");
    for (case, src, code) in refused_cases() {
        let path = dir.join(format!("{case}.mind"));
        std::fs::write(&path, &src).expect("write case source");
        let check = mindc(&["check", "--no-fmt", path.to_str().unwrap()]);
        let check_text = text(&check);
        assert!(
            !check.status.success(),
            "{case}: check accepted:\n{check_text}"
        );
        assert_eq!(
            check_text.matches(code).count(),
            1,
            "{case}: one {code} for the one refused form:\n{check_text}"
        );
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
        assert!(build_text.contains(code), "{case}:\n{build_text}");
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
        assert!(
            test_text.contains(&format!("[{code}]")),
            "{case}:\n{test_text}"
        );
    }
}

/// `(case, provider, main)`: a two-module project whose `main` stores strings
/// into `provider`'s integer slots, or reads them from its integer result.
const PROJECT_CASES: &[(&str, &str, &str)] = &[
    (
        "param",
        "export { eqh }\n\nfn eqh(a: i64, b: i64) -> i64 {\n    if a == b { return 1 }\n    \
         return 0\n}\n",
        "import provider;\n\npub fn probe() -> i64 { return provider.eqh(\"abc\", \"abc\") }\n",
    ),
    (
        "result",
        "export { mk }\n\nfn mk() -> i64 { return \"abc\" }\n",
        "import provider;\n\npub fn probe() -> i64 {\n    \
         if provider.mk() == provider.mk() { return 1 }\n    return 0\n}\n",
    ),
];

const MANIFEST: &str = "[package]\nname = \"xmod\"\nversion = \"0.1.0\"\n\n[build]\n\
                        entry = \"main.mind\"\nemit = \"cdylib\"\n\n[targets.cpu]\n\
                        backend = \"cpu\"\nsources = [\"main.mind\", \"provider.mind\"]\n";

#[cfg(feature = "cross-module-imports")]
#[test]
fn strings_stored_across_project_modules_are_refused_by_check_and_build() {
    for (case, provider, main) in PROJECT_CASES {
        let dir = scratch_dir(&format!("string_flows_backend_parity_project_{case}"));
        std::fs::write(dir.join("Mind.toml"), MANIFEST).expect("write manifest");
        std::fs::write(dir.join("provider.mind"), provider).expect("write provider");
        let main_path = dir.join("main.mind");
        std::fs::write(&main_path, main).expect("write main");
        let check = mindc(&["check", "--no-fmt", main_path.to_str().unwrap()]);
        let check_text = text(&check);
        assert!(
            !check.status.success(),
            "{case}: check of the importing module accepted:\n{check_text}"
        );
        assert!(check_text.contains("E2038"), "{case}:\n{check_text}");
        let so = dir.join("x.so");
        let build = Command::new(mindc_bin())
            .args(["build", "--emit", "cdylib", "--out", so.to_str().unwrap()])
            .current_dir(&dir)
            .output()
            .expect("spawn mindc build");
        let build_text = text(&build);
        assert!(
            !build.status.success(),
            "{case}: project build accepted:\n{build_text}"
        );
        assert!(build_text.contains("E2038"), "{case}:\n{build_text}");
        assert!(!so.exists(), "{case}: a refused build left an artifact");
    }
}

/// A string literal arm lowers to the runtime byte-wise comparison, which
/// reads no record through the null handle; an inline record decode read
/// the length at address 8 for it.
#[test]
fn a_string_literal_arm_lowers_to_the_byte_wise_comparison() {
    let dir = scratch_dir("string_flows_backend_parity_arm");
    let src = dir.join("arm.mind");
    std::fs::write(
        &src,
        "fn is_abc(s: string) -> i64 {\n    match s {\n        \"abc\" => 1,\n        \
         _ => 0,\n    }\n}\n",
    )
    .expect("write arm source");
    let out = mindc(&[src.to_str().unwrap(), "--emit-mlir"]);
    let ir = text(&out);
    assert!(out.status.success(), "lowering failed:\n{ir}");
    assert!(ir.contains("__mind_string_eq"), "no byte-wise test:\n{ir}");
    assert!(
        !ir.contains("__mind_load_i8"),
        "inline record decode:\n{ir}"
    );
}

/// String match arms and string methods: each case is a
/// zero-argument `pub fn`, and both backends must give the expected value.
const DEFINED: &str = r#"fn is_abc(s: string) -> i64 {
    match s {
        "abc" => 1,
        _ => 0,
    }
}

fn tlen(s: string) -> i64 { return string_len(s.trim()) }

fn slen(s: string) -> i64 { return string_len(s.slice(1, 3)) }

fn mlen(s: string) -> i64 { return s.len() }

fn tails(a: string, b: string) -> i64 {
    if a.slice(1, 3) == b.slice(1, 3) { return 1 }
    return 0
}

"#;

const DEFINED_CASES: &[(&str, &str, i64)] = &[
    ("match_equal", "is_abc(\"abc\")", 1),
    ("match_unequal", "is_abc(\"abd\")", 0),
    ("match_built", "is_abc(string_slice(\"xabc\", 1, 4))", 1),
    ("match_null", "is_abc(0)", 0),
    ("trim", "tlen(\" ab \")", 2),
    ("trim_all_space", "tlen(\"  \")", 0),
    ("slice", "slen(\"abcd\")", 2),
    ("len_method", "mlen(\"abc\")", 3),
    ("literal_len_method", "\"abcd\".len()", 4),
    ("bare_string_len", "string_len(\"abc\")", 3),
    ("slices_equal", "tails(\"xbcd\", \"ybce\")", 1),
    ("slices_unequal", "tails(\"xbcd\", \"ybde\")", 0),
];

fn defined_source() -> String {
    let mut src = String::from(DEFINED);
    for (case, expr, _) in DEFINED_CASES {
        src.push_str(&format!("\npub fn {case}() -> i64 {{ return {expr} }}\n"));
    }
    src
}

/// The value each case produces under `mindc test` (the interpreter), one
/// module per case. Integer controls of these shapes are unit tests
/// (`string_equality::flow_tests`): a `mindc test` module that holds these
/// helpers and a loop or tuple helper calling `eqh` overflows the
/// interpreter's stack on the previous head too, which this test does not
/// measure.
fn interpreter_values(dir: &Path) -> Vec<i64> {
    DEFINED_CASES
        .iter()
        .map(|(case, expr, _)| {
            let mut src = format!(
                "{DEFINED}\npub fn {case}() -> i64 {{ return {expr} }}\n\n#[test]\n\
                 fn probed() {{\n    let v: i64 = {case}()\n"
            );
            for candidate in 0..=4 {
                src.push_str(&format!(
                    "    assert v != {candidate}, \"value of {case} is {candidate}.\"\n"
                ));
            }
            src.push_str("}\n");
            let path = dir.join(format!("interp_{case}.mind"));
            std::fs::write(&path, src).expect("write interpreter source");
            let log = text(&mindc(&["test", path.to_str().unwrap()]));
            (0..=4)
                .find(|candidate| log.contains(&format!("value of {case} is {candidate}.")))
                .unwrap_or_else(|| panic!("mindc test gave no value for `{case}`:\n{log}"))
        })
        .collect()
}

/// The value each case produces from the MLIR-built shared library, one
/// child process per case so a crash names its case.
fn mlir_values(dir: &Path) -> Option<Vec<i64>> {
    let src = dir.join("defined_mlir.mind");
    let so = dir.join("defined_mlir.so");
    std::fs::write(&src, defined_source()).expect("write build source");
    let check = mindc(&["check", "--no-fmt", src.to_str().unwrap()]);
    assert!(check.status.success(), "check refused:\n{}", text(&check));
    let out = mindc(&[src.to_str().unwrap(), "--emit-shared", so.to_str().unwrap()]);
    if !common::gate::compiled("string_flows_backend_parity", &out) {
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
fn string_matches_and_methods_agree_between_mindc_test_and_the_mlir_build() {
    let dir = scratch_dir("string_flows_backend_parity_defined");
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
