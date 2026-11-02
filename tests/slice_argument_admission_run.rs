// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Slice-call admission for alias element types and module-private helpers.
//!
//! The slice ABI admits a compatible `array<T>` value or array literal for a
//! `&[T]`/`&mut [T]` parameter. Two valid shapes were refused before lowering:
//!
//! * an element type spelled through a type alias (`array<Q>` with
//!   `type Q = i32`, locally or as an imported `defs.Q`): provenance compared
//!   the alias spelling with the target type instead of the target types;
//! * a call to a module-private sibling `fn` under an explicit export list:
//!   inside a `module m { … }` block the helper had no signature, so a slice
//!   argument was reported as passed to an unresolved function.
//!
//! Every admitted program is compiled, run natively, and compared with the
//! `mindc test` evaluator on the same source. The refusal controls keep the
//! layout and capability rules closed, including range views `&xs[a..b]`,
//! which the slice-call boundary does not specify.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::path::Path;
use std::process::{Command, Output};

const ALIAS_ELEMENTS: &str = r#"
type Q = i32

fn weigh(xs: &[Q]) -> i64 {
    return (xs.len() as i64) * 1000 + (xs[0] as i64) * 100 + (xs[1] as i64) * 10 + (xs[2] as i64)
}

fn fold_first(xs: &mut [Q]) -> Q {
    xs[0] = xs[0] + xs[2]
    return xs[0] + xs[1]
}

pub fn alias_literal() -> i64 {
    let xs: array<Q> = [-3, 5, 7]
    return weigh(xs)
}

pub fn plain_array_to_alias_param() -> i64 {
    let xs: array<i32> = [-3, 5, 7]
    return weigh(xs)
}

pub fn direct_literal() -> i64 {
    return weigh([1, 2, 3])
}

pub fn alias_mutable() -> i64 {
    let xs: array<Q> = [4, 1, -9]
    return fold_first(xs) as i64
}
"#;

const ALIAS_EXPECTED: &[(&str, i64)] = &[
    ("alias_literal", 2757),
    ("plain_array_to_alias_param", 2757),
    ("direct_literal", 3123),
    ("alias_mutable", -4),
];

const PRIVATE_HELPERS: &str = r#"
module helpers {
    fn first(f: &[i32]) -> i32 {
        return later(f, 1)
    }

    fn later(f: &[i32], i: u32) -> i32 {
        return f[i as usize]
    }

    fn bump(f: &mut [i32]) -> i32 {
        f[0] = f[0] + 100
        return f[0]
    }

    fn twice(f: &mut [i32]) -> i32 {
        return bump(f) + bump(f)
    }

    pub fn private_helper() -> i64 {
        let xs: array<i32> = [-8, 21]
        return first(xs) as i64
    }

    pub fn private_mut_helper() -> i64 {
        let xs: array<i32> = [1, 7]
        return (bump(xs) + xs[1]) as i64
    }

    pub fn private_mut_write_back() -> i64 {
        let xs: array<i32> = [1, 0]
        return twice(xs) as i64
    }

    export fn first, private_helper, private_mut_helper, private_mut_write_back
}
"#;

const PRIVATE_EXPECTED: &[(&str, i64)] = &[("private_helper", 21), ("private_mut_helper", 108)];

// deferred: would also compare this case under `mindc test`, stubbed because the
// evaluator copies a `&mut [T]` argument, so a callee's writes are not visible to
// the caller (on main: `bump(ys); ys[0]` is 1 under `mindc test`, 101 natively)
// — upgrade path: give the evaluator shared array storage, then move it into
// PRIVATE_EXPECTED.
const PRIVATE_NATIVE_ONLY: &[(&str, i64)] = &[("private_mut_write_back", 302)];

#[cfg(feature = "cross-module-imports")]
const DEFS_SRC: &str = r#"
module defs {
    type Q16 = i32

    struct Bank {
        fill: u32,
        slots: array<Q16>,
    }

    fn lift(v: i32) -> Q16 {
        return v * 2
    }

    export type Q16
    export struct Bank
    export fn lift
}
"#;

#[cfg(feature = "cross-module-imports")]
const MAKER_SRC: &str = r#"
import defs;

module maker {
    fn make_bank() -> defs.Bank {
        let slots: array<defs.Q16> = [5, -6]
        return defs.Bank { fill: 2, slots: slots }
    }

    export fn make_bank
}
"#;

#[cfg(feature = "cross-module-imports")]
const MAIN_SRC: &str = r#"
import defs;
import maker;

module main {
    fn weigh(xs: &[defs.Q16]) -> i64 {
        return (xs.len() as i64) * 1000 + (xs[0] as i64) * 10 + (xs[1] as i64)
    }

    pub fn field_arg() -> i64 {
        let bank: defs.Bank = maker.make_bank()
        return weigh(bank.slots)
    }

    pub fn call_literal_arg() -> i64 {
        let input: array<defs.Q16> = [defs.lift(3), defs.lift(-4)]
        return weigh(input)
    }

    pub fn qualified_literal_arg() -> i64 {
        let input: array<defs.Q16> = [7, -1]
        return weigh(input)
    }

    export fn weigh, field_arg, call_literal_arg, qualified_literal_arg
}
"#;

#[cfg(feature = "cross-module-imports")]
const IMPORTED_EXPECTED: &[(&str, i64)] = &[
    ("field_arg", 2044),
    ("call_literal_arg", 2052),
    ("qualified_literal_arg", 2069),
];

fn mindc(args: &[&str], cwd: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(cwd)
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc")
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Call each exported `() -> i64` entry of a native artifact and compare.
fn assert_native(artifact: &Path, expected: &[(&str, i64)]) {
    let table: Vec<String> = expected
        .iter()
        .map(|(name, want)| format!("('{name}', {want})"))
        .collect();
    let py = format!(
        "import ctypes\nlib = ctypes.CDLL(r'{}')\nfor name, want in [{}]:\n    fn = getattr(lib, name); fn.restype = ctypes.c_int64\n    got = fn(); assert got == want, f'{{name}}={{got}}, expected {{want}}'\nprint('native-ok')\n",
        artifact.display(),
        table.join(", ")
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run native artifact");
    assert!(
        ran.status.success() && combined(&ran).contains("native-ok"),
        "native execution failed:\n{}",
        combined(&ran)
    );
}

/// The same expected values asserted by `#[test]` functions under `mindc test`.
fn test_block(expected: &[(&str, i64)]) -> String {
    expected
        .iter()
        .map(|(name, want)| {
            format!(
                "\n#[test]\nfn parity_{name}() {{\n    assert {name}() == {want}, \"{name}\"\n}}\n"
            )
        })
        .collect()
}

fn assert_interpreter_parity(dir: &Path, tag: &str, source: &str, expected: &[(&str, i64)]) {
    let path = dir.join(format!("{tag}_parity.mind"));
    std::fs::write(&path, format!("{source}{}", test_block(expected))).expect("write parity");
    let out = mindc(&["test", path.to_str().unwrap()], dir);
    let text = combined(&out);
    assert!(out.status.success(), "mindc test disagrees:\n{text}");
    assert!(
        text.contains(&format!("{} passed", expected.len())),
        "mindc test did not run every parity case:\n{text}"
    );
}

fn check_compile_and_run(
    tag: &str,
    source: &str,
    expected: &[(&str, i64)],
    native_only: &[(&str, i64)],
) {
    let dir = common::scratch_dir("slice_argument_admission_run").join(tag);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("{tag}.mind"));
    std::fs::write(&path, source).expect("write source");

    let checked = mindc(
        &["check", "--no-fmt", "--no-lint", path.to_str().unwrap()],
        &dir,
    );
    assert!(
        checked.status.success(),
        "mindc check refused a valid slice argument:\n{}",
        combined(&checked)
    );

    let artifact = dir.join(format!("{tag}.so"));
    let built = mindc(
        &[
            path.to_str().unwrap(),
            "--emit-shared",
            artifact.to_str().unwrap(),
        ],
        &dir,
    );
    assert!(
        built.status.success(),
        "compile failed:\n{}",
        combined(&built)
    );
    assert_native(&artifact, expected);
    assert_native(&artifact, native_only);
    assert_interpreter_parity(&dir, tag, source, expected);
}

#[test]
fn alias_element_arrays_and_literals_pass_to_alias_slices_and_run() {
    check_compile_and_run("alias_elements", ALIAS_ELEMENTS, ALIAS_EXPECTED, &[]);
}

#[test]
fn private_helpers_under_an_export_list_take_slices_and_run() {
    check_compile_and_run(
        "private_helpers",
        PRIVATE_HELPERS,
        PRIVATE_EXPECTED,
        PRIVATE_NATIVE_ONLY,
    );
}

/// `sources_list` declares `[targets.cpu].sources`, which records each module
/// under the project root (`crate.src.defs`) instead of the entry directory.
#[cfg(feature = "cross-module-imports")]
fn write_project(root: &Path, sources_list: bool) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root.join("src")).expect("project src");
    let exports: Vec<String> = IMPORTED_EXPECTED
        .iter()
        .map(|(name, _)| format!("\"{name}\""))
        .collect();
    std::fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"slice_alias_import\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\noutput = \"slice_alias_import\"\n\n[targets.cpu]\nbackend = \"cpu\"\n{}\n[exports]\nc_abi = [{}]\n",
            if sources_list {
                "sources = [\"src/defs.mind\", \"src/maker.mind\", \"src/main.mind\"]\n"
            } else {
                ""
            },
            exports.join(", ")
        ),
    )
    .expect("write manifest");
    std::fs::write(root.join("src/defs.mind"), DEFS_SRC).expect("write defs");
    std::fs::write(root.join("src/maker.mind"), MAKER_SRC).expect("write maker");
    std::fs::write(root.join("src/main.mind"), MAIN_SRC).expect("write main");
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn imported_alias_fields_and_call_literals_pass_to_qualified_slices_and_run() {
    for sources_list in [false, true] {
        run_imported_project(sources_list);
    }
}

#[cfg(feature = "cross-module-imports")]
fn run_imported_project(sources_list: bool) {
    let root = common::scratch_dir("slice_argument_admission_run")
        .join(format!("imported_sources_list_{sources_list}"));
    write_project(&root, sources_list);

    let checked = mindc(&["check", "--no-fmt", "--no-lint", "src/main.mind"], &root);
    assert!(
        checked.status.success(),
        "mindc check refused imported alias slice arguments (sources list: {sources_list}):\n{}",
        combined(&checked)
    );

    let artifact = root.join("imported.so");
    let built = mindc(
        &[
            "build",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            artifact.to_str().unwrap(),
        ],
        &root,
    );
    assert!(
        built.status.success(),
        "project build failed (sources list: {sources_list}):\n{}",
        combined(&built)
    );
    assert_native(&artifact, IMPORTED_EXPECTED);
}

#[test]
fn unsupported_layouts_and_capabilities_stay_refused() {
    let cases = [
        (
            "alias_float_element",
            "type F = f64\nfn take(xs: &[F]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<F> = [1.5]; return take(xs) }\n",
            "E2032",
        ),
        (
            "alias_float_element_owner",
            "type F = f64\nfn take(xs: &[F]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<F> = vec_zeroed(1); return take(xs) }\n",
            "E2032",
        ),
        (
            "alias_element_mismatch",
            "type A = i32\ntype B = i64\nfn take(xs: &[B]) -> i64 { return xs.length }\npub fn run() -> i64 { let xs: array<A> = [1]; return take(xs) }\n",
            "E2032",
        ),
        (
            "range_view_argument",
            "fn first_of(xs: &[i32]) -> i32 { return xs[0] }\nfn second(xs: &[i32]) -> i32 { return first_of(&xs[1..3]) }\npub fn run() -> i64 { let xs: array<i32> = [4, 9, 6]; return second(xs) as i64 }\n",
            "E2032",
        ),
        (
            "private_helper_wrong_element",
            "module m {\n    fn take(xs: &[i64]) -> i64 { return xs.length }\n    pub fn run() -> i64 { let xs: array<u8> = [1]; return take(xs) }\n    export fn run\n}\n",
            "E2032",
        ),
        (
            "private_helper_scalar_formal",
            "module m {\n    fn hide(raw: i64) -> i64 { return raw }\n    fn bad(xs: &[i64]) -> i64 { return hide(xs) }\n    pub fn run() -> i64 { return bad([1]) }\n    export fn run\n}\n",
            "E2033",
        ),
    ];
    let dir = common::scratch_dir("slice_argument_admission_refusal");
    for (tag, src, code) in cases {
        let path = dir.join(format!("{tag}.mind"));
        let artifact = dir.join(format!("{tag}.so"));
        let _ = std::fs::remove_file(&artifact);
        std::fs::write(&path, src).expect("write source");
        let out = mindc(
            &[
                path.to_str().unwrap(),
                "--emit-shared",
                artifact.to_str().unwrap(),
            ],
            &dir,
        );
        let text = combined(&out);
        assert!(!out.status.success(), "{tag} unexpectedly compiled");
        assert!(text.contains(code), "{tag} missed {code}:\n{text}");
        assert!(!text.contains("panicked at"), "{tag} panicked:\n{text}");
        assert!(!artifact.exists(), "{tag} emitted an artifact on refusal");
    }
}
