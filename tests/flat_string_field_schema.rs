// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Field reads and writes on a `String` value in a single-file compile.
//!
//! A project build installs the std `String` schema from the bundled
//! `std/string.mind`. A single translation unit (`mindc <file>.mind
//! --emit-mic3 / --emit-evidence / --emit-ir`) has no project registry, and
//! `name.len` on a `String` parameter used to fail lowering with E6009 although
//! `mindc check` accepted the file. These tests pin that the single-file path
//! loads the same offsets as a project build, from the same bundled
//! declaration, and that a program declaring its own `String` keeps its layout
//! or is refused.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use libmind::ir::compact::parse_mic3;
use libmind::ir::{BinOp, Instr, ValueId};

const FIELDS: &str = "\
fn str_addr(name: String) -> i64 {
    name.addr
}

fn str_len(name: String) -> i64 {
    name.len
}

fn str_cap(name: String) -> i64 {
    name.cap
}
";

/// Every receiver shape whose field resolves against the `String` schema.
const SHAPES: &str = "\
struct Rec {
    id: i64,
    name: String,
    label: string,
}

fn str_len(name: String) -> i64 {
    name.len
}

fn lower_len(name: string) -> i64 {
    name.len
}

fn let_len() -> i64 {
    let s: string = \"abc\";
    s.len
}

fn set_len(name: String) -> i64 {
    name.len = 2;
    name.len
}

fn rec_name_len(r: Rec) -> i64 {
    r.name.len
}

fn rec_label_len(r: Rec) -> i64 {
    r.label.len
}

fn echo(x: String) -> String {
    x
}

fn call_len(x: String) -> i64 {
    echo(x).len
}

fn literal_len() -> i64 {
    \"abc\".len
}
";

fn scratch(name: &str) -> PathBuf {
    let dir = common::scratch_dir("flat-string-field-schema").join(name);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn mindc(args: &[&str], cwd: &Path, env: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(common::require_mindc());
    cmd.args(args)
        .current_dir(cwd)
        .env_remove("MIND_STDLIB_PATH");
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.output().expect("spawn mindc")
}

fn assert_ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} failed ({:?}):\nstdout: {}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `mindc <file> --emit-mic3` of a single translation unit.
fn flat_mic3(dir: &Path, source: &str, env: &[(&str, &Path)]) -> Vec<u8> {
    fs::write(dir.join("flat.mind"), source).expect("write flat source");
    let out = mindc(&["flat.mind", "--emit-mic3", "flat.mic3"], dir, env);
    assert_ok(&out, "flat --emit-mic3");
    fs::read(dir.join("flat.mic3")).expect("read flat mic3")
}

/// The same source as the entry of a Mind.toml project whose entry imports a
/// local sibling, so discovery installs the whole-project registry.
fn project_mic3(dir: &Path, source: &str, env: &[(&str, &Path)]) -> Vec<u8> {
    let root = dir.join("project");
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"strfields\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"main.mind\"\nemit = \"cdylib\"\n\n\
         [targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"helper.mind\"]\n",
    )
    .expect("write manifest");
    fs::write(
        root.join("helper.mind"),
        "pub fn helper_k() -> i64 {\n    7\n}\n",
    )
    .expect("write helper");
    fs::write(
        root.join("main.mind"),
        format!("import helper;\n\n{source}"),
    )
    .expect("write entry");
    let out = mindc(&["main.mind", "--emit-mic3", "main.mic3"], &root, env);
    assert_ok(&out, "project --emit-mic3");
    fs::read(root.join("main.mic3")).expect("read project mic3")
}

/// Every function body of a mic@3 module, by name, in a comparable form.
fn bodies(bytes: &[u8]) -> BTreeMap<String, Vec<Instr>> {
    let module = parse_mic3(bytes).expect("mic@3 parses");
    module
        .instrs
        .into_iter()
        .filter_map(|instr| match instr {
            Instr::FnDef { name, body, .. } => Some((name, body)),
            _ => None,
        })
        .collect()
}

/// The byte offset, from parameter 0, of the one field load in `body`.
fn load_offset(body: &[Instr]) -> i64 {
    let param = body
        .iter()
        .find_map(|i| match i {
            Instr::Param { dst, index: 0, .. } => Some(*dst),
            _ => None,
        })
        .expect("parameter 0");
    let loads: Vec<ValueId> = body
        .iter()
        .filter_map(|i| match i {
            Instr::Call { name, args, .. } if name == "__mind_load_i64" => Some(args[0]),
            _ => None,
        })
        .collect();
    assert_eq!(loads.len(), 1, "one field load expected in {body:?}");
    if loads[0] == param {
        return 0;
    }
    let (lhs, rhs) = body
        .iter()
        .find_map(|i| match i {
            Instr::BinOp {
                dst,
                op: BinOp::Add,
                lhs,
                rhs,
            } if *dst == loads[0] => Some((*lhs, *rhs)),
            _ => None,
        })
        .expect("field address is base + offset");
    assert_eq!(lhs, param, "field address must start at the parameter");
    body.iter()
        .find_map(|i| match i {
            Instr::ConstI64(dst, k) if *dst == rhs => Some(*k),
            _ => None,
        })
        .expect("constant field offset")
}

fn offsets(bytes: &[u8]) -> [i64; 3] {
    let fns = bodies(bytes);
    ["str_addr", "str_len", "str_cap"].map(|name| load_offset(&fns[name]))
}

fn assert_refused_at(out: &Output, operation: &str, source: &str, snippet: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let start = source.find(snippet).expect("snippet in source");
    let expected = format!(
        "error[materialization][E6009]: unsupported lowering operation `{operation}` at source span {start}..{}",
        start + snippet.len()
    );
    assert!(
        !out.status.success() && stderr.contains(&expected),
        "expected `{expected}`; got status {:?}, stderr:\n{stderr}",
        out.status.code()
    );
}

#[test]
fn flat_string_fields_load_the_std_string_offsets() {
    let dir = scratch("offsets");
    assert_eq!(offsets(&flat_mic3(&dir, FIELDS, &[])), [0, 8, 16]);
    assert_ok(
        &mindc(&["flat.mind", "--emit-ir"], &dir, &[]),
        "flat --emit-ir",
    );
}

#[test]
fn flat_evidence_of_string_fields_verifies() {
    let dir = scratch("evidence");
    fs::write(dir.join("fields.mind"), FIELDS).expect("write source");
    let out = mindc(
        &["fields.mind", "--emit-evidence", "fields.evidence.mic3"],
        &dir,
        &[],
    );
    assert_ok(&out, "flat --emit-evidence");
    let bytes = fs::read(dir.join("fields.evidence.mic3")).expect("read evidence");
    assert_eq!(offsets(&bytes), [0, 8, 16]);
    let verify = mindc(&["verify", "fields.evidence.mic3", "--json"], &dir, &[]);
    assert_ok(&verify, "mindc verify");
    let report = String::from_utf8_lossy(&verify.stdout);
    assert!(
        report.contains("\"trace_hash_valid\":true") && report.contains("\"ssa_valid\":true"),
        "evidence must verify: {report}"
    );
}

#[test]
fn flat_and_project_lower_string_fields_identically() {
    let dir = scratch("parity");
    let flat = bodies(&flat_mic3(&dir, SHAPES, &[]));
    let project = bodies(&project_mic3(&dir, SHAPES, &[]));
    assert_eq!(flat.len(), 9, "every function of the source is lowered");
    for (name, body) in &flat {
        let other = project
            .get(name)
            .unwrap_or_else(|| panic!("project build has no fn {name}"));
        assert_eq!(
            format!("{body:?}"),
            format!("{other:?}"),
            "fn {name}: single-file and project lowering differ"
        );
    }
}

/// The schema comes from the std sources the compiler loads, never from a
/// copy: with a std directory whose `String` puts `len` first, both paths load
/// `len` at offset 0.
#[test]
fn both_paths_take_the_string_layout_from_the_loaded_std() {
    let dir = scratch("std-override");
    let std_dir = dir.join("std");
    fs::create_dir_all(&std_dir).expect("create std override");
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("std");
    for entry in fs::read_dir(&shipped).expect("read std") {
        let path = entry.expect("std entry").path();
        if path.extension().is_some_and(|e| e == "mind") {
            fs::copy(&path, std_dir.join(path.file_name().unwrap())).expect("copy std");
        }
    }
    let string = fs::read_to_string(std_dir.join("string.mind")).expect("read string.mind");
    let declared = "struct String {\n    addr: i64,\n    len: i64,\n    cap: i64,\n}";
    assert!(string.contains(declared), "std/string.mind declares String");
    let permuted = "struct String {\n    len: i64,\n    addr: i64,\n    cap: i64,\n}";
    fs::write(
        std_dir.join("string.mind"),
        string.replace(declared, permuted),
    )
    .expect("write permuted string.mind");
    let env = [("MIND_STDLIB_PATH", std_dir.as_path())];
    assert_eq!(offsets(&flat_mic3(&dir, FIELDS, &env)), [8, 0, 16]);
    assert_eq!(offsets(&project_mic3(&dir, FIELDS, &env)), [8, 0, 16]);
}

#[test]
fn a_string_declared_before_the_access_keeps_its_layout() {
    let dir = scratch("user-first");
    let source =
        format!("struct String {{\n    len: i64,\n    addr: i64,\n    cap: i64,\n}}\n\n{FIELDS}");
    assert_eq!(offsets(&flat_mic3(&dir, &source, &[])), [8, 0, 16]);
}

#[test]
fn a_string_declared_after_the_access_is_refused() {
    for (case, tail) in [
        (
            "after",
            "\nstruct String {\n    len: i64,\n    addr: i64,\n    cap: i64,\n}\n",
        ),
        (
            "nested",
            "\nmodule text {\n    struct String {\n        len: i64,\n        addr: i64,\n        cap: i64,\n    }\n}\n",
        ),
    ] {
        let dir = scratch(&format!("user-{case}"));
        let source = format!("{FIELDS}{tail}");
        fs::write(dir.join("late.mind"), &source).expect("write source");
        let out = mindc(&["late.mind", "--emit-mic3", "late.mic3"], &dir, &[]);
        assert_refused_at(&out, "struct field read", &source, "name.addr");
    }
}

#[test]
fn an_unknown_string_field_is_refused() {
    let dir = scratch("unknown-field");
    let source = "fn f(name: String) -> i64 {\n    name.size\n}\n";
    fs::write(dir.join("unknown.mind"), source).expect("write source");
    let out = mindc(&["unknown.mind", "--emit-mic3", "unknown.mic3"], &dir, &[]);
    assert_refused_at(&out, "struct field read", source, "name.size");
}

/// Only a `String` value takes the `String` schema: a struct declared after
/// its use still has no schema at the access, so the read stays refused.
#[test]
fn a_non_string_receiver_never_takes_the_string_schema() {
    let dir = scratch("non-string");
    let source = "fn f(p: P) -> i64 {\n    p.len\n}\n\nstruct P {\n    len: i64,\n}\n";
    fs::write(dir.join("p.mind"), source).expect("write source");
    let out = mindc(&["p.mind", "--emit-mic3", "p.mic3"], &dir, &[]);
    assert_refused_at(&out, "struct field read", source, "p.len");
}
