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
//! declaration, for owned and borrowed (`&String`, `&mut String`) strings, that
//! a `String` literal is laid out in that order, and that a program declaring
//! its own `String` keeps its layout or is refused.

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

struct Borrowed {
    name: &String,
}

fn ref_len(name: &String) -> i64 {
    name.len
}

fn ref_mut_len(name: &mut String) -> i64 {
    name.len
}

fn ref_set_len(name: &mut String) -> i64 {
    name.len = 2;
    name.len
}

fn ref_lower_len(name: &string) -> i64 {
    name.len
}

fn ref_len_call(name: &String) -> i64 {
    name.len()
}

fn let_ref_len(s: String) -> i64 {
    let r: &String = &s;
    r.len
}

fn let_borrow_len(s: String) -> i64 {
    let r = &s;
    r.len
}

fn let_ref_if_len(c: bool, a: String, b: String) -> i64 {
    let r: &String = if c { &a } else { &b };
    r.len
}

fn let_paren_len(s: String) -> i64 {
    let r = (s);
    r.len
}

fn borrow(x: &String) -> &String {
    x
}

fn borrow_len(s: String) -> i64 {
    borrow(&s).len
}

fn let_borrow_call_len(s: String) -> i64 {
    let r = borrow(&s);
    r.len
}

fn borrowed_field_len(b: Borrowed) -> i64 {
    b.name.len
}

fn borrowed_elements_len(v: array<&String>) -> i64 {
    let mut t = 0;
    for s in v {
        t = t + s.len;
    }
    t
}

fn paren_len(s: String) -> i64 {
    (s).len
}

fn paren_borrow_len(s: String) -> i64 {
    (&s).len
}

fn struct_literal_len() -> i64 {
    String { addr: 0, len: 3, cap: 0 }.len
}

fn permuted_literal_len() -> i64 {
    let s = String { len: 3, addr: 0, cap: 0 };
    s.len
}
";

/// `FIELDS` read through a shared and through a mutable borrow.
const BORROWED_FIELDS: [&str; 2] = [
    "\
fn str_addr(name: &String) -> i64 {
    name.addr
}

fn str_len(name: &String) -> i64 {
    name.len
}

fn str_cap(name: &String) -> i64 {
    name.cap
}
",
    "\
fn str_addr(name: &mut String) -> i64 {
    name.addr
}

fn str_len(name: &mut String) -> i64 {
    name.len
}

fn str_cap(name: &mut String) -> i64 {
    name.cap
}
",
];

/// A `String` literal whose fields are listed out of declaration order.
const LITERAL: &str = "\
fn make() -> String {
    String { len: 3, addr: 5, cap: 7 }
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

/// `body` as text, with each for-each index name made offset-free: the name
/// carries the loop's source offset, and a project entry starts with an import.
fn comparable(body: &[Instr]) -> String {
    const INDEX: &str = "__fe_i_";
    let text = format!("{body:?}");
    let mut out = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(at) = rest.find(INDEX) {
        out.push_str(&rest[..at + INDEX.len()]);
        rest = rest[at + INDEX.len()..].trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
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

/// The byte offset of `addr` from `base`: `base` itself, or `base + k`.
fn offset_from(body: &[Instr], base: ValueId, addr: ValueId) -> Option<i64> {
    if addr == base {
        return Some(0);
    }
    let (lhs, rhs) = body.iter().find_map(|i| match i {
        Instr::BinOp {
            dst,
            op: BinOp::Add,
            lhs,
            rhs,
        } if *dst == addr => Some((*lhs, *rhs)),
        _ => None,
    })?;
    (lhs == base).then_some(())?;
    body.iter().find_map(|i| match i {
        Instr::ConstI64(dst, k) if *dst == rhs => Some(*k),
        _ => None,
    })
}

/// Each constant a struct literal in `body` stores, by its byte offset.
fn literal_stores(body: &[Instr]) -> BTreeMap<i64, i64> {
    let base = body
        .iter()
        .find_map(|i| match i {
            Instr::Call { dst, name, .. } if name == "__mind_alloc" => Some(*dst),
            _ => None,
        })
        .expect("the literal allocates its record");
    body.iter()
        .filter_map(|i| match i {
            Instr::Call { name, args, .. } if name == "__mind_store_i64" => {
                let value = body.iter().find_map(|j| match j {
                    Instr::ConstI64(dst, k) if *dst == args[1] => Some(*k),
                    _ => None,
                })?;
                Some((offset_from(body, base, args[0])?, value))
            }
            _ => None,
        })
        .collect()
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
    assert_eq!(flat.len(), 27, "every function of the source is lowered");
    for (name, body) in &flat {
        let other = project
            .get(name)
            .unwrap_or_else(|| panic!("project build has no fn {name}"));
        assert_eq!(
            comparable(body),
            comparable(other),
            "fn {name}: single-file and project lowering differ"
        );
    }
}

/// A borrow lowers to the string it borrows, so `&String` and `&mut String`
/// read the same offsets as an owned `String`, and a zero-arg `.len()` on a
/// borrow calls `string_len` with it, as on an owned string, never a constant.
#[test]
fn flat_borrowed_string_fields_load_the_std_string_offsets() {
    for (case, source) in ["shared", "mutable"].into_iter().zip(BORROWED_FIELDS) {
        let dir = scratch(&format!("borrowed-{case}"));
        assert_eq!(offsets(&flat_mic3(&dir, source, &[])), [0, 8, 16], "{case}");
    }
    let dir = scratch("borrowed-call");
    let fns = bodies(&flat_mic3(&dir, SHAPES, &[]));
    let body = &fns["ref_len_call"];
    let param = body.iter().find_map(|i| match i {
        Instr::Param { dst, index: 0, .. } => Some(*dst),
        _ => None,
    });
    assert!(
        body.iter().any(|i| matches!(i,
            Instr::Call { name, args, .. } if name == "string_len" && args.first().copied() == param)),
        "`name.len()` on a borrowed string must call string_len(name): {body:?}"
    );
}

/// A `String` literal is laid out in the declared order, whatever order it
/// lists its fields in, as a project build lays it out and as every field
/// access reads it; with no field access in the module the artifact still
/// carries no `String` schema of its own.
#[test]
fn a_string_literal_is_laid_out_in_the_bundled_order() {
    let dir = scratch("literal-order");
    let flat = flat_mic3(&dir, LITERAL, &[]);
    let expected = BTreeMap::from([(0, 5), (8, 3), (16, 7)]);
    assert_eq!(literal_stores(&bodies(&flat)["make"]), expected);
    assert_eq!(
        literal_stores(&bodies(&project_mic3(&dir, LITERAL, &[]))["make"]),
        expected
    );
    let module = parse_mic3(&flat).expect("mic@3 parses");
    assert!(
        !module.struct_defs.contains_key("String"),
        "a literal alone must not add a schema to the artifact"
    );
}

/// A `String` the program declares, even after the literal, is the one the
/// literal builds: its fields stay in that declaration's order.
#[test]
fn a_string_literal_of_a_user_string_keeps_its_order() {
    let dir = scratch("literal-user");
    let source =
        format!("{LITERAL}\nstruct String {{\n    len: i64,\n    addr: i64,\n    cap: i64,\n}}\n");
    let stores = literal_stores(&bodies(&flat_mic3(&dir, &source, &[]))["make"]);
    assert_eq!(stores, BTreeMap::from([(0, 3), (8, 5), (16, 7)]));
}

/// The schema comes from the std sources the compiler loads, never from a
/// copy: with a std directory whose `String` puts `len` first, both paths load
/// `len` at offset 0.
#[test]
fn both_paths_take_the_string_layout_from_the_loaded_std() {
    let dir = scratch("std-override");
    let std_dir = std_with_string(&dir, "len: i64,\n    addr: i64,\n    cap: i64,");
    let env = [("MIND_STDLIB_PATH", std_dir.as_path())];
    assert_eq!(offsets(&flat_mic3(&dir, FIELDS, &env)), [8, 0, 16]);
    assert_eq!(offsets(&project_mic3(&dir, FIELDS, &env)), [8, 0, 16]);
}

/// A loaded std `String` whose fields are not all 8-byte cells: a literal
/// stores them with the widths and offsets the reads use, as a project build
/// does, whether the literal or a read comes first.
#[test]
fn a_string_literal_takes_the_widths_of_the_loaded_std() {
    let dir = scratch("std-narrow");
    let std_dir = std_with_string(&dir, "len: i32,\n    addr: i32,\n    cap: i64,");
    let env = [("MIND_STDLIB_PATH", std_dir.as_path())];
    let source =
        "fn f() -> i64 {\n    let s = String { cap: 7, addr: 5, len: 3 };\n    s.addr\n}\n";
    let flat = bodies(&flat_mic3(&dir, source, &env));
    let project = bodies(&project_mic3(&dir, source, &env));
    assert_eq!(format!("{:?}", flat["f"]), format!("{:?}", project["f"]));
}

/// A copy of the shipped std whose `String` declares `fields` instead.
fn std_with_string(dir: &Path, fields: &str) -> PathBuf {
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
    let replaced = format!("struct String {{\n    {fields}\n}}");
    fs::write(
        std_dir.join("string.mind"),
        string.replace(declared, &replaced),
    )
    .expect("write std/string.mind");
    std_dir
}

#[test]
fn a_string_declared_before_the_access_keeps_its_layout() {
    let dir = scratch("user-first");
    let source =
        format!("struct String {{\n    len: i64,\n    addr: i64,\n    cap: i64,\n}}\n\n{FIELDS}");
    assert_eq!(offsets(&flat_mic3(&dir, &source, &[])), [8, 0, 16]);
}

/// A string value has the bundled layout, so a program's own `String` with
/// another layout, in field order or in field types, is never used to read
/// one: the access is refused.
#[test]
fn a_string_value_is_never_read_through_a_user_layout() {
    for (case, fields) in [
        ("order", "len: i64,\n    addr: i64,\n    cap: i64,"),
        ("types", "addr: i32,\n    len: i32,\n    cap: i32,"),
    ] {
        let dir = scratch(&format!("user-layout-{case}"));
        let source = format!(
            "struct String {{\n    {fields}\n}}\n\nfn f(name: string) -> i64 {{\n    name.len\n}}\n"
        );
        fs::write(dir.join("value.mind"), &source).expect("write source");
        let out = mindc(&["value.mind", "--emit-mic3", "value.mic3"], &dir, &[]);
        assert_refused_at(&out, "struct field read", &source, "name.len");
    }
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
        ("alias", "\ntype String = i64;\n"),
        ("enum", "\nenum String {\n    A,\n    B,\n}\n"),
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
