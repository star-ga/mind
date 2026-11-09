// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! String-equality edges, each pinned on `mindc check`, the MLIR build and
//! `mindc test`.
//!
//! A review measured four shapes on which the backends disagreed:
//!
//! * Integer code that calls std.vec / std.map (`vec_get(v, 0) ==
//!   vec_get(v, 1)` over pushed integers). `mindc check` accepted it and the
//!   build computed 1, but `mindc test` refused it with `E2038`: its merged
//!   module took the std functions without the std `struct` declarations,
//!   so `v.cap` inside std `vec_push` had no type there. `mindc test` now
//!   gives the built program's value.
//! * A string whose type the pass did not establish (here 70 parentheses
//!   deep, past the analysis depth) handed to `vec_push` or
//!   `__mind_store_i64` recorded no escape, so `vec_get` / `__mind_load_i64`
//!   compared the two handles: the build returned 0 for equal text. Such a
//!   value counts as maybe a string where it is handed over too, and the
//!   comparison is refused with `E2038`.
//! * `(5 as string) == "abc"` passed `mindc check`; the build crashed
//!   (SIGSEGV) reading a string record at address 5. A cast of an integer
//!   literal other than the null handle `0` to `string` is refused with
//!   `E2038`.
//! * `string_eq(null, "abc")` gave 0 on the build and failed under `mindc
//!   test` ("no field .len for Int(0)"). Both give 0 now, as `==` does.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports" \
//!        --test string_edges_backend_parity`

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;
use common::{mindc_bin, scratch_dir};

use std::path::Path;
use std::process::{Command, Output};

/// `(case, source, expected)`: each source defines `probe()`; `mindc test`
/// and the MLIR build must both give `expected`.
const VALUES: &[(&str, &str, i64)] = &[
    (
        "vec_get_pair",
        "use std.vec;\n\npub fn probe() -> i64 {\n    let v0 = vec_new()\n    \
         let v1 = vec_push(v0, 7)\n    let v2 = vec_push(v1, 7)\n    \
         if vec_get(v2, 0) == vec_get(v2, 1) { return 1 }\n    return 0\n}\n",
        1,
    ),
    (
        "vec_len_of_push",
        "use std.vec;\n\npub fn probe() -> i64 { return vec_len(vec_push(vec_new(), 7)) }\n",
        1,
    ),
    (
        "vec_get_vs_literal",
        "use std.vec;\n\npub fn probe() -> i64 {\n    let v = vec_push(vec_new(), 7)\n    \
         if vec_get(v, 0) == 7 { return 1 }\n    return 0\n}\n",
        1,
    ),
    (
        "map_get_pair",
        "use std.map;\n\npub fn probe() -> i64 {\n    let m = map_insert(map_new(), 1, 7)\n    \
         if map_get(m, 1) == map_get(m, 1) { return 1 }\n    return 0\n}\n",
        1,
    ),
    (
        "null_string_eq",
        "fn null_name() -> string { return 0 }\n\n\
         pub fn probe() -> i64 { return string_eq(null_name(), \"abc\") }\n",
        0,
    ),
    (
        "null_string_eq_imported",
        "import std.string;\n\nfn null_name() -> string { return 0 }\n\n\
         pub fn probe() -> i64 { return string_eq(null_name(), \"abc\") }\n",
        0,
    ),
    (
        "null_eq_form",
        "fn null_name() -> string { return 0 }\n\npub fn probe() -> i64 {\n    \
         if null_name() == \"abc\" { return 1 }\n    return 0\n}\n",
        0,
    ),
    (
        "text_string_eq",
        "pub fn probe() -> i64 { return string_eq(\"abc\", \"abc\") }\n",
        1,
    ),
    (
        "null_cast",
        "pub fn probe() -> i64 {\n    if (0 as string) == \"abc\" { return 1 }\n    return 0\n}\n",
        0,
    ),
    (
        "string_cast",
        "fn same(s: string) -> i64 {\n    if (s as string) == \"abc\" { return 1 }\n    \
         return 0\n}\n\npub fn probe() -> i64 { return same(\"abc\") }\n",
        1,
    ),
];

/// A string 70 parentheses deep: past the depth the pass analyses.
fn deep_string() -> String {
    format!("{}\"abc\"{}", "(".repeat(70), ")".repeat(70))
}

/// `(case, source, marker)`: each source is refused with `E2038` by every
/// backend; `marker` is a phrase of the refusal.
fn refused() -> Vec<(&'static str, String, &'static str)> {
    let d = deep_string();
    vec![
        (
            "deep_string_into_vec",
            format!(
                "use std.vec;\n\npub fn probe() -> i64 {{\n    let v0 = vec_new()\n    \
                 let v1 = vec_push(v0, {d})\n    let v2 = vec_push(v1, {d})\n    \
                 if vec_get(v2, 0) == vec_get(v2, 1) {{ return 1 }}\n    return 0\n}}\n"
            ),
            "string_eq",
        ),
        (
            "deep_string_into_raw_memory",
            format!(
                "pub fn probe() -> i64 {{\n    let p = __mind_alloc(16)\n    \
                 __mind_store_i64(p, {d})\n    __mind_store_i64(p + 8, {d})\n    \
                 if __mind_load_i64(p) == __mind_load_i64(p + 8) {{ return 1 }}\n    \
                 return 0\n}}\n"
            ),
            "string_eq",
        ),
        (
            "integer_cast_to_string",
            "pub fn probe() -> i64 {\n    if (5 as string) == \"abc\" { return 1 }\n    \
             return 0\n}\n"
                .to_string(),
            "as string",
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

/// `src` with a `#[test]` that fails with `value is N.` for the value of
/// `probe()` when it is 0 or 1.
fn with_value_test(src: &str) -> String {
    format!(
        "{src}\n#[test]\nfn probed() {{\n    let v: i64 = probe()\n    \
         assert v != 0, \"value is 0.\"\n    assert v != 1, \"value is 1.\"\n}}\n"
    )
}

/// The value of `probe()` under `mindc test` (the interpreter).
fn interpreter_value(dir: &Path, case: &str, src: &str) -> i64 {
    let path = dir.join(format!("{case}_test.mind"));
    std::fs::write(&path, with_value_test(src)).expect("write test source");
    let log = text(&mindc(&["test", path.to_str().unwrap()]));
    (0..=1)
        .find(|v| log.contains(&format!("value is {v}.")))
        .unwrap_or_else(|| panic!("{case}: mindc test gave no value:\n{log}"))
}

/// The value of `probe()` from the MLIR-built shared library, run in its own
/// child process so a crash names its case; `None` without the toolchain.
fn mlir_value(dir: &Path, case: &str, src: &str) -> Option<i64> {
    let path = dir.join(format!("{case}.mind"));
    let so = dir.join(format!("{case}.so"));
    std::fs::write(&path, src).expect("write build source");
    let check = mindc(&["check", "--no-fmt", path.to_str().unwrap()]);
    assert!(
        check.status.success(),
        "{case}: check refused:\n{}",
        text(&check)
    );
    let out = mindc(&[
        path.to_str().unwrap(),
        "--emit-shared",
        so.to_str().unwrap(),
    ]);
    if !common::gate::compiled("string_edges_backend_parity", &out) {
        return None;
    }
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\nf = lib.probe\n\
         f.restype = ctypes.c_int64\nprint(f())\n",
        so.display()
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("spawn python3");
    assert!(
        ran.status.success(),
        "{case}: probe() crashed:\n{}",
        text(&ran)
    );
    let value = String::from_utf8_lossy(&ran.stdout).trim().parse();
    Some(value.expect("an i64"))
}

#[test]
fn integer_std_collections_null_strings_and_casts_agree_on_every_backend() {
    let dir = scratch_dir("string_edges_backend_parity_values");
    for (case, src, expected) in VALUES {
        assert_eq!(
            interpreter_value(&dir, case, src),
            *expected,
            "{case}: mindc test value"
        );
        if let Some(built) = mlir_value(&dir, case, src) {
            assert_eq!(built, *expected, "{case}: MLIR build value");
        }
    }
}

#[test]
fn unestablished_strings_in_std_slots_and_integer_casts_are_refused_everywhere() {
    let dir = scratch_dir("string_edges_backend_parity_refused");
    for (case, src, marker) in refused() {
        let path = dir.join(format!("{case}.mind"));
        std::fs::write(&path, &src).expect("write case source");
        let check = mindc(&["check", "--no-fmt", path.to_str().unwrap()]);
        let check_text = text(&check);
        assert!(
            !check.status.success(),
            "{case}: check accepted:\n{check_text}"
        );
        assert!(check_text.contains("E2038"), "{case}:\n{check_text}");
        assert!(check_text.contains(marker), "{case}:\n{check_text}");
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
        std::fs::write(&test_path, with_value_test(&src)).expect("write test source");
        let test = mindc(&["test", test_path.to_str().unwrap()]);
        let test_text = text(&test);
        assert!(
            !test.status.success(),
            "{case}: mindc test passed:\n{test_text}"
        );
        assert!(
            !test_text.contains("value is"),
            "{case}: a value:\n{test_text}"
        );
        // A debug `mindc test` overflows its worker thread's stack in the
        // parser on 70 nested parentheses, before the pass runs (as on
        // `main`); an optimised one parses them and refuses the comparison.
        if case.starts_with("deep_") && cfg!(debug_assertions) {
            continue;
        }
        assert!(test_text.contains("[E2038]"), "{case}:\n{test_text}");
        assert!(test_text.contains(marker), "{case}:\n{test_text}");
    }
}
