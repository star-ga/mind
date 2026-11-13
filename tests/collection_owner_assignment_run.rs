// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Collection owners stored in fields or indexed owner slots must retain their
//! declared collection type. Replacing one with a mutator status, scalar, or a
//! different owner type previously passed both `check` and `build`, then could
//! crash when the invalid handle was used.

#![cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn source_path(src: &str, tag: &str, dir: &Path) -> PathBuf {
    let path = dir.join(format!("{tag}.mind"));
    std::fs::write(&path, src).expect("write source");
    path
}

fn check(source: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(["check", "--no-fmt", source.to_str().unwrap()])
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc check")
}

fn build(source: &Path, artifact: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args([
            source.to_str().unwrap(),
            "--emit-shared",
            artifact.to_str().unwrap(),
        ])
        .env(
            "MINDC_STD_DIR",
            format!("{}/std", env!("CARGO_MANIFEST_DIR")),
        )
        .output()
        .expect("run mindc build")
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn assert_refused(tag: &str, src: &str, code: &str, dir: &Path) {
    let source = source_path(src, tag, dir);
    let artifact = dir.join(format!("{tag}.so"));

    let checked = check(&source);
    let check_text = combined(&checked);
    assert_eq!(
        checked.status.code(),
        Some(1),
        "{tag} check result:\n{check_text}"
    );
    assert!(
        check_text.contains(code),
        "{tag} check missed {code}:\n{check_text}"
    );
    if tag == "field_element_type" {
        assert!(
            check_text.contains("array<i64>") && check_text.contains("array<f64>"),
            "{tag} omitted concrete owner types:\n{check_text}"
        );
    }
    assert!(!check_text.contains("panicked at"), "{tag} check panicked");

    let built = build(&source, &artifact);
    let build_text = combined(&built);
    assert_eq!(
        built.status.code(),
        Some(1),
        "{tag} build result:\n{build_text}"
    );
    assert!(
        build_text.contains(code),
        "{tag} build missed {code}:\n{build_text}"
    );
    assert!(!build_text.contains("panicked at"), "{tag} build panicked");
    assert!(!artifact.exists(), "{tag} refusal emitted an artifact");
}

#[test]
fn incompatible_collection_owner_replacements_are_refused_before_emission() {
    let dir = common::scratch_dir("collection_owner_assignment_refusal");
    for (tag, src) in [
        (
            "field_status",
            "struct Holder { xs: array<i64> }\npub fn run() -> i64 { let h: Holder = Holder { xs: [1] }; h.xs = h.xs.set(0, 9); return h.xs[0] }\n",
        ),
        (
            "field_scalar",
            "struct Holder { xs: array<i64> }\npub fn run() -> i64 { let h: Holder = Holder { xs: [1] }; h.xs = 7; return h.xs[0] }\n",
        ),
        (
            "field_owner_kind",
            "struct Holder { xs: array<i64> }\npub fn run() -> i64 { let h: Holder = Holder { xs: [1] }; let m: map<i64, i64> = {}; h.xs = m; return h.xs[0] }\n",
        ),
        (
            "field_element_type",
            "struct Holder { xs: array<i64> }\npub fn run() -> i64 { let h: Holder = Holder { xs: [1] }; let ys: array<f64> = [1.5]; h.xs = ys; return h.xs[0] }\n",
        ),
        (
            "map_literal_key_type",
            "struct Holder { ids: map<i64, i64> }\npub fn run() -> i64 { let h: Holder = Holder { ids: {} }; h.ids = {\"x\": 1}; return 0 }\n",
        ),
        (
            "set_literal_element_type",
            "struct Holder { flags: set<i64> }\npub fn run() -> i64 { let h: Holder = Holder { flags: {} }; h.flags = {\"x\"}; return 0 }\n",
        ),
        (
            "indexed_owner_status",
            "struct Holder { xss: array<array<i64>> }\npub fn run() -> i64 { let h: Holder = Holder { xss: [[1]] }; h.xss[0] = h.xss[0].set(0, 9); return h.xss[0][0] }\n",
        ),
        (
            "fixed_array_indexed_owner_status",
            "fn bad(slots: [array<i64>; 1]) -> i64 { slots[0] = slots[0].set(0, 9); return 0 }\n",
        ),
        (
            "nested_literal_element_type",
            "struct Holder { xss: array<array<i64>> }\npub fn run(h: Holder) -> i64 { h.xss = [[1], [1.5]]; return 0 }\n",
        ),
        (
            "forward_declared_field",
            "pub fn run(h: Holder) -> i64 { h.xs = 7; return 0 }\nstruct Holder { xs: array<i64> }\n",
        ),
        (
            "module_wrapped_field",
            "module m { struct Holder { xs: array<i64> } pub fn run(h: Holder) -> i64 { h.xs = 7; return 0 } }\n",
        ),
    ] {
        assert_refused(tag, src, "E2034", &dir);
    }
}

#[test]
fn borrowed_slice_cannot_escape_through_a_scalar_field() {
    let dir = common::scratch_dir("collection_owner_assignment_borrow");
    assert_refused(
        "borrowed_field",
        "struct Holder { raw: i64 }\nfn bad(view: &[i64], h: Holder) -> i64 { h.raw = view; return 0 }\n",
        "E2033",
        &dir,
    );
}

#[test]
fn compatible_field_call_and_scalar_element_replacements_run_natively() {
    let src = r#"
struct Holder { xs: array<i64> }
struct Bag { ids: map<i64, i64>, flags: set<i64> }
fn make_array() -> array<i64> { return [9] }
fn make_map() -> map<i64, i64> {
    let ids: map<i64, i64> = {}
    ids.insert(1, 9)
    return ids
}
fn make_set() -> set<i64> {
    let flags: set<i64> = {}
    flags.insert(9)
    return flags
}

pub fn from_field() -> i64 {
    let h: Holder = Holder { xs: [1] }
    let other: Holder = Holder { xs: [9] }
    h.xs = other.xs
    return h.xs[0]
}

pub fn from_call() -> i64 {
    let h: Holder = Holder { xs: [1] }
    h.xs = make_array()
    return h.xs[0]
}

pub fn scalar_element() -> i64 {
    let h: Holder = Holder { xs: [1] }
    h.xs[0] = 9
    return h.xs[0]
}

pub fn map_return() -> i64 {
    let ids: map<i64, i64> = {}
    let flags: set<i64> = {}
    let b: Bag = Bag { ids: ids, flags: flags }
    b.ids = make_map()
    return b.ids.get(1)
}

pub fn set_return() -> i64 {
    let ids: map<i64, i64> = {}
    let flags: set<i64> = {}
    let b: Bag = Bag { ids: ids, flags: flags }
    b.flags = make_set()
    return b.flags.contains(9)
}
"#;
    let dir = common::scratch_dir("collection_owner_assignment_native");
    let source = source_path(src, "valid", &dir);
    let artifact = dir.join("valid.so");
    let out = build(&source, &artifact);
    assert!(
        out.status.success(),
        "valid build failed:\n{}",
        combined(&out)
    );

    let py = format!(
        r#"import ctypes
lib = ctypes.CDLL(r'{}')
expected = {{'from_field': 9, 'from_call': 9, 'scalar_element': 9, 'map_return': 9, 'set_return': 1}}
for name, want in expected.items():
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int64
    got = fn()
    assert got == want, f'{{name}}={{got}}, expected {{want}}'
"#,
        artifact.display()
    );
    let ran = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("run native controls");
    assert!(
        ran.status.success(),
        "native controls failed:\n{}",
        combined(&ran)
    );
}

#[test]
fn compatible_literal_and_nested_owner_replacements_pass_type_checking() {
    let src = r#"
struct Holder { xs: array<i64>, xss: array<array<i64>> }
pub fn run() -> i64 {
    let h: Holder = Holder { xs: [1], xss: [[1], [9]] }
    h.xs = ([9])
    h.xss[0] = h.xss[1]
    return h.xs[0]
}

"#;
    let dir = common::scratch_dir("collection_owner_assignment_literals");
    let source = source_path(src, "valid", &dir);
    let out = check(&source);
    assert!(
        out.status.success(),
        "valid check failed:\n{}",
        combined(&out)
    );
}

#[test]
fn string_name_synonyms_remain_compatible_inside_owner_types() {
    let src = r#"
struct Text {
    words: array<String>,
    counts: map<String, i64>,
    flags: set<String>,
}
fn replace(
    text: Text,
    words: array<string>,
    counts: map<string, i64>,
    flags: set<string>,
) -> i64 {
    text.words = words
    text.counts = counts
    text.flags = flags
    return 0
}
"#;
    let dir = common::scratch_dir("collection_owner_assignment_string_names");
    let source = source_path(src, "valid", &dir);
    let out = check(&source);
    assert!(
        out.status.success(),
        "string synonym check failed:\n{}",
        combined(&out)
    );
}

#[test]
fn compatible_rhs_does_not_hide_mutation_in_the_receiver_path() {
    let cases = [
        r#"
struct Holder { xs: array<i64> }
fn bad(holders: array<Holder>, indexes: array<i64>, other: Holder) -> i64 {
    holders[indexes.push(0)].xs = other.xs
    return 0
}
"#,
        r#"
fn bad(slots: array<array<i64>>, indexes: array<i64>, rhs: array<i64>) -> i64 {
    slots[indexes.push(0)] = rhs
    return 0
}
"#,
    ];
    let dir = common::scratch_dir("collection_owner_assignment_receiver");
    for (index, src) in cases.into_iter().enumerate() {
        let source = source_path(src, &format!("receiver_{index}"), &dir);
        let out = check(&source);
        let text = combined(&out);
        assert_eq!(
            out.status.code(),
            Some(1),
            "unexpected check result:\n{text}"
        );
        assert!(
            text.contains("E2300"),
            "missing receiver diagnostic:\n{text}"
        );
        assert!(
            !text.contains("E2034"),
            "compatible RHS was rejected:\n{text}"
        );
    }
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn sibling_struct_owner_type_is_enforced_by_project_check_and_build() {
    let project = common::scratch_dir("collection_owner_assignment_xmod").join("project");
    std::fs::create_dir_all(project.join("src")).expect("create project");
    std::fs::write(
        project.join("Mind.toml"),
        "[package]\nname = \"owner_xmod\"\nversion = \"0.1.0\"\n\n[build]\nemit = \"cdylib\"\n",
    )
    .expect("write manifest");
    std::fs::write(
        project.join("src/types.mind"),
        "struct Holder { xs: array<i64> }\n",
    )
    .expect("write owner module");
    std::fs::write(
        project.join("src/main.mind"),
        "pub fn run(h: Holder) -> i64 { h.xs = 7; return 0 }\n",
    )
    .expect("write consumer module");

    for command in ["check", "build"] {
        let mut invocation = Command::new(common::mindc_bin());
        invocation.arg(command).current_dir(&project);
        if command == "check" {
            invocation.args(["--no-fmt", "--no-lint"]);
        } else {
            invocation.args(["--backend", "mlir"]);
        }
        let out = invocation
            .output()
            .unwrap_or_else(|error| panic!("run project {command}: {error}"));
        let text = combined(&out);
        assert_eq!(out.status.code(), Some(1), "project {command}:\n{text}");
        assert!(text.contains("E2034"), "project {command}:\n{text}");
        assert!(!text.contains("panicked at"), "project {command} panicked");
    }
    assert!(
        !project.join("target/debug/libowner_xmod.so").exists(),
        "project refusal emitted an artifact"
    );
}

#[cfg(feature = "cross-module-imports")]
fn write_collision_project(project: &Path, imports: &str, invalid: bool) {
    std::fs::create_dir_all(project.join("src")).expect("create collision project");
    std::fs::write(
        project.join("Mind.toml"),
        "[package]\nname = \"owner_collision\"\nversion = \"0.1.0\"\n\n[build]\nemit = \"cdylib\"\n",
    )
    .expect("write collision manifest");
    std::fs::write(
        project.join("src/a.mind"),
        "pub struct Holder { xs: array<i64>, marker: i64 }\n",
    )
    .expect("write module a");
    std::fs::write(
        project.join("src/b.mind"),
        "pub struct Holder { scalar: i64, ys: array<i64> }\n",
    )
    .expect("write module b");
    let body = if invalid {
        "pub fn bad(h: a.Holder) -> i64 { h.xs = 7; return 0 }\n"
    } else {
        "pub fn read_a(h: a.Holder) -> i64 { h.xs = h.xs; return h.marker }\n\
         pub fn read_b(h: b.Holder) -> i64 { h.ys = h.ys; return h.scalar }\n"
    };
    std::fs::write(project.join("src/main.mind"), format!("{imports}\n{body}"))
        .expect("write collision consumer");
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn qualified_colliding_struct_owners_are_order_independent_and_run_natively() {
    let root = common::scratch_dir("collection_owner_assignment_collision");
    for (tag, imports) in [
        ("ab", "import a;\nimport b;"),
        ("ba", "import b;\nimport a;"),
    ] {
        let valid = root.join(format!("valid_{tag}"));
        write_collision_project(&valid, imports, false);
        let built = Command::new(common::mindc_bin())
            .args(["build", "--backend", "mlir", "--no-cache"])
            .current_dir(&valid)
            .output()
            .expect("build valid collision project");
        assert!(
            built.status.success(),
            "valid {tag} build:\n{}",
            combined(&built)
        );
        let artifact = valid.join("target/debug/libowner_collision.so");
        let py = format!(
            r#"import ctypes
lib = ctypes.CDLL(r'{}')
class A(ctypes.Structure):
    _fields_ = [('xs', ctypes.c_int64), ('marker', ctypes.c_int64)]
class B(ctypes.Structure):
    _fields_ = [('scalar', ctypes.c_int64), ('ys', ctypes.c_int64)]
for name, cls, value, want in (('read_a', A, A(0, 41), 41), ('read_b', B, B(73, 0), 73)):
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.POINTER(cls)]
    got = fn(ctypes.byref(value))
    assert got == want, f'{{name}}={{got}}, expected {{want}}'
"#,
            artifact.display()
        );
        let ran = Command::new("python3")
            .args(["-c", &py])
            .output()
            .expect("run collision exports");
        assert!(
            ran.status.success(),
            "valid {tag} native:\n{}",
            combined(&ran)
        );

        let invalid = root.join(format!("invalid_{tag}"));
        write_collision_project(&invalid, imports, true);
        let checked = Command::new(common::mindc_bin())
            .args(["check", "--no-fmt", "--no-lint"])
            .current_dir(&invalid)
            .output()
            .expect("check invalid collision project");
        let text = combined(&checked);
        assert_eq!(checked.status.code(), Some(1), "invalid {tag}:\n{text}");
        assert!(text.contains("E2034"), "invalid {tag}:\n{text}");
    }
}
