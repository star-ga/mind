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

fn inner_sum(xs: &[i32]) -> i64 {
    return (xs[0] as i64) * 10 + (xs[1] as i64)
}

fn forward_alias(xs: &[Q]) -> i64 {
    return inner_sum(xs)
}

fn inner_bump(xs: &mut [i32]) -> i64 {
    xs[0] = xs[0] + 10
    return (xs[0] as i64) * 10 + (xs[1] as i64)
}

fn forward_alias_mut(xs: &mut [Q]) -> i64 {
    return inner_bump(xs)
}

pub fn alias_param_forward() -> i64 {
    let a: array<Q> = [3, -4]
    return forward_alias(a)
}

pub fn alias_param_forward_mut() -> i64 {
    let a: array<Q> = [3, -4]
    return forward_alias_mut(a)
}

struct Pair {
    left: Q,
    right: i32,
}

pub fn read_elements() -> i64 {
    let a: array<Q> = [6, -2, 9]
    let b: array<i32> = [a[2], a[0], a[1]]
    return weigh(b)
}

pub fn field_elements() -> i64 {
    let p: Pair = Pair { left: 4, right: -1 }
    return weigh([p.left, p.right, p.left])
}
"#;

const ALIAS_EXPECTED: &[(&str, i64)] = &[
    ("alias_literal", 2757),
    ("plain_array_to_alias_param", 2757),
    ("direct_literal", 3123),
    ("alias_mutable", -4),
    ("alias_param_forward", 26),
    ("alias_param_forward_mut", 126),
    ("read_elements", 3958),
    ("field_elements", 3394),
];

const CONST_ELEMENTS: &str = r#"
type Q = i32

const Z: i32 = 7
const QZ: Q = -2
const DERIVED = Z * 3

fn weigh(xs: &[i32]) -> i64 {
    return (xs.len() as i64) * 1000 + (xs[0] as i64) * 100 + (xs[1] as i64) * 10 + (xs[2] as i64)
}

pub fn local_const_literal() -> i64 {
    let xs: array<i32> = [Z, Z, QZ]
    return weigh(xs)
}

pub fn const_expression_elements() -> i64 {
    return weigh([Z + 1, -QZ, DERIVED])
}
"#;

const CONST_EXPECTED: &[(&str, i64)] = &[
    ("local_const_literal", 3768),
    ("const_expression_elements", 3841),
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
    type B = u8
    type R = f64

    const ONE: Q16 = 65536
    const PONE: i32 = 3
    const RH: R = 2

    struct Bank {
        fill: u32,
        slots: array<Q16>,
    }

    struct Raw {
        fill: u32,
        raw: array<i32>,
    }

    struct Pack {
        fill: u32,
        bytes: array<B>,
    }

    fn lift(v: i32) -> Q16 {
        return v * 2
    }

    fn make_raw() -> Raw {
        let raw: array<i32> = [7, -8]
        return Raw { fill: 2, raw: raw }
    }

    fn make_bank_here() -> Bank {
        let slots: array<Q16> = [5, -6]
        return Bank { fill: 2, slots: slots }
    }

    fn make_pack() -> Pack {
        let bytes: array<B> = [2, 255]
        return Pack { fill: 2, bytes: bytes }
    }

    export type Q16, B, R
    export const ONE, PONE, RH
    export struct Bank, Raw, Pack
    export fn lift, make_raw, make_bank_here, make_pack
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

    fn total(xs: &[defs.Q16]) -> i64 {
        return (xs.len() as i64) * 1000 + (xs[0] as i64) * 10 + (xs[1] as i64)
    }

    export fn make_bank, total
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

    fn weigh_i32(xs: &[i32]) -> i64 {
        return (xs.len() as i64) * 1000 + (xs[0] as i64) * 10 + (xs[1] as i64)
    }

    fn weigh_bytes(xs: &[defs.B]) -> i64 {
        return (xs.len() as i64) * 100000 + (xs[0] as i64) * 1000 + (xs[1] as i64)
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

    pub fn unqualified_return_field() -> i64 {
        let raw: defs.Raw = defs.make_raw()
        return weigh_i32(raw.raw)
    }

    pub fn unannotated_return_field() -> i64 {
        let raw = defs.make_raw()
        return weigh_i32(raw.raw)
    }

    pub fn unqualified_return_alias_field() -> i64 {
        let bank: defs.Bank = defs.make_bank_here()
        return weigh(bank.slots)
    }

    pub fn u8_alias_field() -> i64 {
        let pack: defs.Pack = defs.make_pack()
        return weigh_bytes(pack.bytes)
    }

    pub fn imported_const_plain() -> i64 {
        let xs: array<i32> = [defs.PONE, defs.PONE]
        return weigh_i32(xs)
    }

    pub fn imported_const_alias() -> i64 {
        let xs: array<defs.Q16> = [defs.ONE, defs.ONE]
        return weigh(xs)
    }

    pub fn imported_const_direct() -> i64 {
        return weigh([defs.ONE, defs.ONE])
    }

    pub fn imported_const_mixed() -> i64 {
        return weigh([defs.lift(1), defs.ONE])
    }

    pub fn imported_const_expression() -> i64 {
        return weigh([defs.ONE + 1, -defs.PONE])
    }

    pub fn imported_callee_const_literal() -> i64 {
        let input: array<defs.Q16> = [defs.ONE, defs.PONE]
        return maker.total(input)
    }

    export fn weigh, field_arg, call_literal_arg, qualified_literal_arg
    export fn unqualified_return_field, unannotated_return_field
    export fn unqualified_return_alias_field, u8_alias_field
    export fn imported_const_plain, imported_const_alias, imported_const_direct
    export fn imported_const_mixed, imported_const_expression, imported_callee_const_literal
}
"#;

#[cfg(feature = "cross-module-imports")]
const IMPORTED_EXPECTED: &[(&str, i64)] = &[
    ("field_arg", 2044),
    ("call_literal_arg", 2052),
    ("qualified_literal_arg", 2069),
    ("unqualified_return_field", 2062),
    ("unannotated_return_field", 2062),
    ("unqualified_return_alias_field", 2044),
    ("u8_alias_field", 202255),
    ("imported_const_plain", 2033),
    ("imported_const_alias", 722896),
    ("imported_const_direct", 722896),
    ("imported_const_mixed", 67556),
    ("imported_const_expression", 657367),
    ("imported_callee_const_literal", 657363),
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
fn constant_elements_pass_to_slices_and_run() {
    check_compile_and_run("const_elements", CONST_ELEMENTS, CONST_EXPECTED, &[]);
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

/// A struct the checked module declares keeps its own field types: a local
/// `Raw` is not the imported `defs.Raw` of the same name. An imported constant
/// keeps its declared type: `RH: R` with `type R = f64` is not an integer.
#[cfg(feature = "cross-module-imports")]
#[test]
fn owner_scoped_fields_and_constant_types_refuse_mismatches() {
    let root = common::scratch_dir("slice_argument_admission_run").join("imported_name_collision");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).expect("project src");
    std::fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"slice_name_collision\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\noutput = \"slice_name_collision\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\n[exports]\nc_abi = [\"run\"]\n",
    )
    .expect("write manifest");
    std::fs::write(root.join("src/defs.mind"), DEFS_SRC).expect("write defs");
    std::fs::write(
        root.join("src/main.mind"),
        "import defs;\n\nmodule main {\n    struct Raw {\n        raw: array<i64>,\n    }\n\n    fn weigh_i32(xs: &[i32]) -> i64 {\n        return xs.len() as i64\n    }\n\n    pub fn run() -> i64 {\n        let imported: defs.Raw = defs.make_raw()\n        let mine: Raw = Raw { raw: [1, 2] }\n        let a: i64 = weigh_i32(imported.raw)\n        let b: i64 = weigh_i32(mine.raw)\n        return a + b + weigh_i32([defs.RH])\n    }\n\n    export fn run\n}\n",
    )
    .expect("write main");
    let checked = mindc(&["check", "--no-fmt", "--no-lint", "src/main.mind"], &root);
    let text = combined(&checked);
    assert!(
        !checked.status.success(),
        "mismatched slice arguments passed:\n{text}"
    );
    assert_eq!(
        text.matches("E2032").count(),
        2,
        "the local field and the float constant must be refused, not the imported field:\n{text}"
    );
    for line in ["main.mind:16:", "main.mind:17:"] {
        assert!(text.contains(line), "no refusal at {line}:\n{text}");
    }
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
            "array_alias_binding",
            "type V = array<i32>\nfn get(xs: &[i32]) -> i64 { return xs.length }\npub fn run() -> i64 { let v: V = [3, 4]; return get(v) }\n",
            "E2032",
        ),
        (
            "float_const_element",
            "const F: f64 = 1.5\nfn take(xs: &[i64]) -> i64 { return xs.length }\npub fn run() -> i64 { return take([F, F]) }\n",
            "E2032",
        ),
        (
            "const_alias_type_disagrees_with_value",
            "type Q = f64\nconst H: Q = 2\nfn take(xs: &[i64]) -> i64 { return xs[0] }\npub fn run() -> i64 { return take([H, H]) }\n",
            "E2032",
        ),
        (
            "alias_slice_parameter",
            "type SL = &[i32]\nfn g(xs: SL) -> i64 { return xs.len() as i64 }\npub fn run() -> i64 { let a: array<i32> = [5, 6]; return g(a) }\n",
            "E2032",
        ),
        (
            "alias_slice_binding",
            "type SL = &[i32]\nfn g(xs: &[i32]) -> i64 { let s: SL = xs; return 3 }\npub fn run() -> i64 { let a: array<i32> = [5, 6]; return g(a) }\n",
            "E2032",
        ),
        (
            "alias_slice_struct_field",
            "type SL = &[i32]\nstruct S { n: i64, xs: SL }\npub fn run() -> i64 { return 1 }\n",
            "E2033",
        ),
        (
            "float_read_element",
            "fn take(xs: &[i64]) -> i64 { return xs[0] }\npub fn run() -> i64 { let f: array<f64> = [1.5]; return take([f[0]]) }\n",
            "E2032",
        ),
        (
            "binding_shadows_const_element",
            "const SCALE: i64 = 4\nstruct P { v: i64 }\nfn take(xs: &[i64]) -> i64 { return xs[0] }\nfn run_with(SCALE: P) -> i64 { return take([SCALE]) }\npub fn run() -> i64 { return run_with(P { v: 1 }) }\n",
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
