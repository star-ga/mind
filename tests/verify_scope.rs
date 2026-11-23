// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc verify --json` prints a `scope` key (RFC 0017 §4.7).
//!
//! `scope` relays a fact `verify` already computes from the decoded body: whether
//! the body calls a function it does not hold (`ir::reach::calls_outside_body`).
//! `whole_program`: every call resolves to a function the body defines, an `extern`
//! symbol the body declares, or an intrinsic row. `partial`: some call does not.
//! Only the decoded body sets the value; nothing stored in the artifact does.
//!
//! Every case compiles a program with the top-level `--emit-evidence` (or
//! `--emit-mic3`), decodes the artifact with the library, and requires the printed
//! value to equal the library's answer AND the value expected for that shape. The
//! callee names are read back from the body, so a case whose call was compiled
//! away fails instead of passing with no call in it.

#![cfg(all(feature = "std-surface", feature = "cross-module-imports"))]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use libmind::ir::compact::{emit_mic3, parse_mic3_envelope};
use libmind::ir::reach::{called_names, calls_outside_body};
use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use serde_json::Value;

/// Defines the one function it calls.
const SELF_CONTAINED: &str =
    "fn double(x: i64) -> i64 {\n    x + x\n}\n\npub fn f(p: i64) -> i64 {\n    double(p)\n}\n";

/// Calls a std function by name; that function's body is not in the unit.
const STD_CALL: &str =
    "import std.entropy;\n\npub fn f() -> i64 {\n    entropy.entropy_max_len()\n}\n";

/// The same call, spelled bare.
const STD_CALL_BARE: &str =
    "import std.entropy;\n\npub fn f() -> i64 {\n    entropy_max_len()\n}\n";

/// The text of that std function, as `std/entropy.mind` holds it.
const ENTROPY_MAX_LEN: &str = "pub fn entropy_max_len() -> i64 {\n    1048576\n}\n";

/// Calls one intrinsic row and one symbol it declares `extern`.
const INTRINSIC_AND_EXTERN: &str = "extern \"C\" {\n    fn labs(x: i64) -> i64\n}\n\n\
     pub fn f(p: i64) -> i64 {\n    let q: i64 = __mind_alloc(64);\n    q + labs(p)\n}\n";

/// A declared extern does not excuse a std call beside it.
const EXTERN_AND_STD_CALL: &str = "import std.entropy;\n\nextern \"C\" {\n    fn labs(x: i64) -> i64\n}\n\n\
     pub fn f(p: i64) -> i64 {\n    labs(p) + entropy.entropy_max_len()\n}\n";

fn mindc(args: &[&str]) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .output()
        .expect("spawn mindc")
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

fn output_text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Compile `program` in `dir` with `flag` (`--emit-evidence` or `--emit-mic3`) and
/// the `extra` arguments.
fn emit(dir: &Path, tag: &str, program: &str, flag: &str, ext: &str, extra: &[&str]) -> PathBuf {
    let src = common::scratch_src_in(dir, &format!("{tag}.mind"), program);
    let artifact = dir.join(format!("{tag}.{ext}"));
    let mut args = vec![path_str(&src), flag, path_str(&artifact)];
    args.extend_from_slice(extra);
    let out = mindc(&args);
    assert!(
        out.status.success(),
        "`mindc {tag}.mind {flag}` failed\n{}",
        output_text(&out)
    );
    artifact
}

/// An artifact with the `evidence_chain.*` MAP.
fn attested(dir: &Path, tag: &str, program: &str) -> PathBuf {
    emit(dir, tag, program, "--emit-evidence", "ev", &[])
}

/// A plain mic@3 body: no MAP, so `verify` reports it unattested.
fn unattested(dir: &Path, tag: &str, program: &str) -> PathBuf {
    emit(dir, tag, program, "--emit-mic3", "mic3", &[])
}

struct Verified {
    code: Option<i32>,
    stderr: String,
    json: Value,
}

/// `mindc verify <artifact> --json`, with the JSON line parsed.
fn verify_json(artifact: &Path) -> Verified {
    let out = mindc(&["verify", path_str(artifact), "--json"]);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let json = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("`verify --json` did not print one JSON object ({e})\nstdout: {stdout}\nstderr: {stderr}")
    });
    Verified {
        code: out.status.code(),
        stderr,
        json,
    }
}

fn scope(json: &Value) -> &str {
    json["scope"]
        .as_str()
        .unwrap_or_else(|| panic!("the report has no `scope` key: {json}"))
}

/// The names the decoded body calls, and whether one of them has no body there.
fn decoded(artifact: &Path) -> (BTreeSet<String>, bool) {
    let bytes = std::fs::read(artifact).expect("read artifact");
    let module = parse_mic3_envelope(&bytes).expect("decode artifact");
    (called_names(&module.instrs), calls_outside_body(&module))
}

/// `verify --json` on an intact artifact: exit 0, the report of the expected kind,
/// and `scope` equal to the library's answer for the decoded body. Returns the
/// printed scope and the names the body calls.
fn checked_scope(artifact: &Path, is_attested: bool) -> (String, BTreeSet<String>) {
    let v = verify_json(artifact);
    assert_eq!(v.code, Some(0), "verify must exit 0\n{}", v.stderr);
    if is_attested {
        assert_eq!(
            v.json["trace_hash_valid"], true,
            "the chain must be intact: {}",
            v.json
        );
        assert!(
            v.json.get("attested").is_none(),
            "an attested report has no `attested` key: {}",
            v.json
        );
    } else {
        assert_eq!(
            v.json["attested"], false,
            "the report must be the unattested kind: {}",
            v.json
        );
        assert!(
            v.json.get("trace_hash_valid").is_none(),
            "an unattested report has no `trace_hash_valid`: {}",
            v.json
        );
    }
    let (calls, outside) = decoded(artifact);
    let printed = scope(&v.json).to_string();
    let relayed = if outside { "partial" } else { "whole_program" };
    assert_eq!(
        printed, relayed,
        "scope must relay `calls_outside_body` ({outside}); calls: {calls:?}"
    );
    (printed, calls)
}

#[test]
fn a_unit_that_holds_every_function_it_calls_is_whole_program() {
    let dir = common::scratch_dir("verify_scope");
    let (printed, calls) = checked_scope(&attested(&dir, "self_contained", SELF_CONTAINED), true);
    assert!(
        calls.contains("double"),
        "the body must call `double`, else this proves nothing: {calls:?}"
    );
    assert_eq!(printed, "whole_program");
}

#[test]
fn a_std_function_called_by_name_is_partial() {
    let dir = common::scratch_dir("verify_scope");
    for (tag, program) in [
        ("std_qualified", STD_CALL),
        ("std_bare", STD_CALL_BARE),
        ("extern_and_std", EXTERN_AND_STD_CALL),
    ] {
        let (printed, calls) = checked_scope(&attested(&dir, tag, program), true);
        assert!(
            calls.iter().any(|name| name.ends_with("entropy_max_len")),
            "{tag}: the body must call `entropy_max_len`: {calls:?}"
        );
        assert_eq!(
            printed, "partial",
            "{tag}: no body of the std function is held"
        );
    }
}

#[test]
fn the_same_function_pasted_into_the_unit_is_whole_program() {
    assert!(
        include_str!("../std/entropy.mind").contains(ENTROPY_MAX_LEN),
        "the pasted text must be the std function as std/entropy.mind holds it"
    );
    let dir = common::scratch_dir("verify_scope");
    let program = format!("{ENTROPY_MAX_LEN}\npub fn f() -> i64 {{\n    entropy_max_len()\n}}\n");
    let (printed, calls) = checked_scope(&attested(&dir, "pasted", &program), true);
    assert!(
        calls.contains("entropy_max_len"),
        "the body must call the pasted function: {calls:?}"
    );
    assert_eq!(printed, "whole_program");
}

#[test]
fn an_intrinsic_and_a_declared_extern_are_whole_program() {
    let dir = common::scratch_dir("verify_scope");
    let art = attested(&dir, "intrinsic_extern", INTRINSIC_AND_EXTERN);
    let (printed, calls) = checked_scope(&art, true);
    assert!(
        calls.contains("__mind_alloc") && calls.contains("labs"),
        "the body must call the intrinsic and the extern: {calls:?}"
    );
    assert_eq!(printed, "whole_program");
}

/// A project whose entry calls `w.<callee>(p)` and whose `w.mind` defines `callee`;
/// the attested artifact of the entry. The body of `callee` is not in that unit.
fn project_artifact(tag: &str, callee: &str, extra: &[&str]) -> PathBuf {
    let dir = common::scratch_dir("verify_scope_project").join(tag);
    std::fs::create_dir_all(dir.join(".git")).expect("create project boundary");
    let manifest = "[package]\nname = \"pw\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"pw\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\", \"w.mind\"]\n";
    common::scratch_src_in(&dir, "Mind.toml", manifest);
    common::scratch_src_in(
        &dir,
        "main.mind",
        &format!("import w;\n\npub fn f(p: i64) -> i64 {{\n    w.{callee}(p)\n}}\n"),
    );
    common::scratch_src_in(
        &dir,
        "w.mind",
        &format!("pub fn {callee}(p: i64) -> i64 {{\n    p + 1\n}}\n"),
    );
    let out = Command::new(common::mindc_bin())
        .args(["main.mind", "--emit-evidence", "x.ev"])
        .args(extra)
        .current_dir(&dir)
        .output()
        .expect("spawn mindc");
    assert!(
        out.status.success(),
        "project emit failed\n{}",
        output_text(&out)
    );
    dir.join("x.ev")
}

#[test]
fn a_project_module_called_by_name_is_partial() {
    let (printed, calls) = checked_scope(&project_artifact("plain", "g", &[]), true);
    assert!(
        calls.iter().any(|name| name.ends_with('g')),
        "the entry must call `g`: {calls:?}"
    );
    assert_eq!(
        printed, "partial",
        "the body of `w.g` is not in the entry's unit"
    );
}

/// The check is by the bare name the body keeps for a call. Where that name is a row
/// of the intrinsic table, the callee resolves although its body is not in the
/// artifact: the table holds libc and POSIX symbol names (`readdir`, `connect`, ...),
/// and `std.posix` and a project module define functions of those names.
#[test]
fn a_callee_whose_bare_name_is_an_intrinsic_row_is_whole_program() {
    assert!(
        include_str!("../std/posix.mind").contains("pub fn readdir(d: i64) -> i64 {"),
        "`std.posix` must define `readdir`, a function whose body the unit does not hold"
    );
    let dir = common::scratch_dir("verify_scope");
    let std_art = emit(
        &dir,
        "row_std",
        "import std.posix;\n\npub fn run(d: i64) -> i64 {\n    readdir(d)\n}\n",
        "--emit-evidence",
        "ev",
        &["--allow-nondeterministic"],
    );
    let bytes = std::fs::read(&std_art).expect("read artifact");
    let module = parse_mic3_envelope(&bytes).expect("decode artifact");
    assert!(
        !module
            .instrs
            .iter()
            .any(|i| matches!(i, Instr::FnDef { name, .. } if name == "readdir")),
        "the artifact must not hold a body for `readdir`, else this proves nothing"
    );
    let (printed, calls) = checked_scope(&std_art, true);
    assert!(
        calls.contains("readdir"),
        "the body must call `readdir`: {calls:?}"
    );
    assert_eq!(printed, "whole_program", "`readdir` is an intrinsic row");

    let flags = ["--allow-nondeterministic"];
    for (callee, want) in [("connect", "whole_program"), ("twice", "partial")] {
        let art = project_artifact(&format!("row_{callee}"), callee, &flags);
        let (printed, calls) = checked_scope(&art, true);
        assert!(
            calls.iter().any(|name| name.ends_with(callee)),
            "the entry must call `{callee}`: {calls:?}"
        );
        assert_eq!(printed, want, "project function `{callee}`");
    }
    let rfc = include_str!("../docs/rfcs/0017-mindc-verify.md");
    assert!(
        rfc.contains("`readdir`"),
        "RFC 0017 section 4.7 must state this case"
    );
    for loader in ["`dlopen`", "`dlsym`", "`dlclose`", "`dlerror`"] {
        assert!(
            rfc.contains(loader),
            "RFC 0017 section 4.7 must name the loader row {loader}; std does not declare it"
        );
    }
}

#[test]
fn the_unattested_report_carries_the_same_scope() {
    let dir = common::scratch_dir("verify_scope");
    for (tag, program, want) in [
        ("plain_self", SELF_CONTAINED, "whole_program"),
        ("plain_std", STD_CALL, "partial"),
        ("plain_extern", INTRINSIC_AND_EXTERN, "whole_program"),
    ] {
        let (printed, _) = checked_scope(&unattested(&dir, tag, program), false);
        assert_eq!(printed, want, "{tag}: unattested report");
        let (with_chain, _) = checked_scope(&attested(&dir, &format!("{tag}_ev"), program), true);
        assert_eq!(with_chain, printed, "{tag}: attested and unattested differ");
    }
}

fn sorted_keys(json: &Value) -> Vec<String> {
    let mut keys: Vec<String> = json
        .as_object()
        .expect("the report is a JSON object")
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}

fn sorted(mut keys: Vec<&str>) -> Vec<String> {
    keys.sort_unstable();
    keys.into_iter().map(String::from).collect()
}

#[test]
fn both_reports_carry_exactly_the_documented_keys() {
    let dir = common::scratch_dir("verify_scope");
    let attested_report = verify_json(&attested(&dir, "keys_ev", SELF_CONTAINED)).json;
    assert_eq!(
        sorted_keys(&attested_report),
        sorted(vec![
            "artifact",
            "substrate",
            "determinism",
            "toolchain",
            "parent",
            "trace_hash",
            "trace_hash_kind",
            "trace_hash_valid",
            "fp_mode",
            "ssa_valid",
            "ssa_reason",
            "signature",
            "signature_ed25519_pubkey",
            "signature_mldsa_pubkey",
            "signature_mldsa87_pubkey",
            "signature_slhdsa_pubkey",
            "provenance_authenticated",
            "scope",
        ]),
        "attested report"
    );
    let plain_report = verify_json(&unattested(&dir, "keys_plain", SELF_CONTAINED)).json;
    assert_eq!(
        sorted_keys(&plain_report),
        sorted(vec![
            "artifact",
            "ssa_valid",
            "ssa_reason",
            "attested",
            "scope"
        ]),
        "unattested report"
    );
}

#[test]
fn the_human_readable_report_does_not_print_scope() {
    let dir = common::scratch_dir("verify_scope");
    for (kind, art, marker) in [
        (
            "attested",
            attested(&dir, "human_ev", STD_CALL),
            "trace_hash_valid: yes",
        ),
        (
            "unattested",
            unattested(&dir, "human_plain", STD_CALL),
            "attested:         no",
        ),
    ] {
        let out = mindc(&["verify", path_str(&art)]);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert_eq!(out.status.code(), Some(0), "{kind}: {}", output_text(&out));
        assert!(
            stdout.contains(marker),
            "{kind}: positive control, the report is the expected kind: {stdout}"
        );
        assert!(
            !stdout
                .lines()
                .any(|line| line.trim_start().starts_with("scope")),
            "{kind}: the human-readable report has a scope line: {stdout}"
        );
    }
}

#[test]
fn the_scope_is_not_stored_in_the_artifact() {
    let dir = common::scratch_dir("verify_scope");
    let chain = std::fs::read(attested(&dir, "stored_ev", STD_CALL)).expect("read attested");
    let plain = std::fs::read(unattested(&dir, "stored_plain", STD_CALL)).expect("read plain");
    let has = |bytes: &[u8], needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        has(&chain, b"evidence_chain.determinism"),
        "positive control: the MAP is in the attested artifact"
    );
    for (name, bytes) in [("attested", &chain), ("plain", &plain)] {
        assert!(!has(bytes, b"scope"), "{name} artifact stores a scope");
        assert!(
            !has(bytes, b"whole_program"),
            "{name} artifact stores a value"
        );
        assert!(!has(bytes, b"partial"), "{name} artifact stores a value");
    }
}

/// Replace the string `from` (one-byte length prefix, as the MAP and the string
/// table write it) by `to`, with its own prefix. Panics when `from` is absent.
fn replace_string(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
    let (from, to) = (from.as_bytes(), to.as_bytes());
    let at = bytes
        .windows(from.len() + 1)
        .position(|w| usize::from(w[0]) == from.len() && &w[1..] == from)
        .expect("the string to replace is in the artifact");
    let parts: [&[u8]; 4] = [
        &bytes[..at],
        &[u8::try_from(to.len()).expect("short string")],
        to,
        &bytes[at + 1 + from.len()..],
    ];
    parts.concat()
}

#[test]
fn a_flipped_trace_hash_still_fails_and_the_scope_follows_the_body() {
    let dir = common::scratch_dir("verify_scope");
    let art = attested(&dir, "flip", STD_CALL);
    let pristine = verify_json(&art);
    assert_eq!(
        pristine.code,
        Some(0),
        "pristine artifact: {}",
        pristine.stderr
    );
    assert_eq!(scope(&pristine.json), "partial");

    let mut bytes = std::fs::read(&art).expect("read artifact");
    let key = b"evidence_chain.trace_hash";
    let key_pos = (0..=bytes.len() - key.len())
        .find(|&i| &bytes[i..i + key.len()] == key && bytes.get(i + key.len()) != Some(&b'_'))
        .expect("trace_hash key in the artifact");
    bytes[key_pos + key.len() + 8] ^= 0x01;
    let flipped = dir.join("flip_forged.ev");
    std::fs::write(&flipped, &bytes).expect("write forged artifact");

    let v = verify_json(&flipped);
    assert_eq!(
        v.code,
        Some(1),
        "a flipped trace_hash must exit 1: {}",
        v.stderr
    );
    assert!(v.stderr.contains("MISMATCH"), "stderr: {}", v.stderr);
    assert_eq!(v.json["trace_hash_valid"], false, "report: {}", v.json);
    assert_eq!(scope(&v.json), "partial", "the body is untouched");
}

#[test]
fn a_forged_determinism_label_still_fails_and_does_not_move_the_scope() {
    let dir = common::scratch_dir("verify_scope");
    let art = attested(&dir, "label", SELF_CONTAINED);
    let pristine = verify_json(&art);
    assert_eq!(
        pristine.code,
        Some(0),
        "pristine artifact: {}",
        pristine.stderr
    );
    assert_eq!(pristine.json["determinism"], "deterministic");

    let bytes = std::fs::read(&art).expect("read artifact");
    let forged = dir.join("label_forged.ev");
    std::fs::write(
        &forged,
        replace_string(&bytes, "deterministic", "nondeterministic"),
    )
    .expect("write forged artifact");

    let v = verify_json(&forged);
    assert_eq!(v.code, Some(1), "a forged label must exit 1: {}", v.stderr);
    assert!(v.stderr.contains("TAMPERED"), "stderr: {}", v.stderr);
    assert_eq!(
        scope(&v.json),
        scope(&pristine.json),
        "scope moved: {}",
        v.json
    );
}

#[test]
fn an_edit_of_the_unhashed_map_does_not_move_the_scope() {
    let dir = common::scratch_dir("verify_scope");
    let art = attested(&dir, "map", STD_CALL);
    let pristine = verify_json(&art);
    assert_eq!(
        pristine.code,
        Some(0),
        "pristine artifact: {}",
        pristine.stderr
    );

    let bytes = std::fs::read(&art).expect("read artifact");
    let forged = dir.join("map_forged.ev");
    std::fs::write(
        &forged,
        replace_string(&bytes, env!("CARGO_PKG_VERSION"), "forged-toolchain"),
    )
    .expect("write forged artifact");

    let v = verify_json(&forged);
    assert_eq!(v.code, Some(0), "the MAP is not hashed: {}", v.stderr);
    assert_eq!(
        v.json["toolchain"], "forged-toolchain",
        "positive control: the edit landed: {}",
        v.json
    );
    assert_eq!(v.json["trace_hash_valid"], true, "report: {}", v.json);
    assert_eq!(scope(&v.json), "partial");
}

#[test]
fn an_ssa_fault_report_carries_no_scope() {
    let dir = common::scratch_dir("verify_scope");
    // Two instructions define %1: the same fault `tests/verify_ssa.rs` uses.
    let mut m = IRModule::new();
    m.instrs.push(Instr::ConstI64(ValueId(0), 2));
    m.instrs.push(Instr::ConstI64(ValueId(1), 3));
    m.instrs.push(Instr::BinOp {
        dst: ValueId(1),
        op: BinOp::Add,
        lhs: ValueId(0),
        rhs: ValueId(1),
    });
    m.instrs.push(Instr::Output(ValueId(1)));
    m.next_id = 2;
    let art = dir.join("ssa_fault.mic3");
    std::fs::write(&art, emit_mic3(&m)).expect("write artifact");

    let v = verify_json(&art);
    assert_eq!(v.code, Some(1), "an SSA fault must exit 1: {}", v.stderr);
    assert_eq!(v.json["ssa_valid"], false, "positive control: {}", v.json);
    assert!(
        v.json.get("scope").is_none(),
        "the SSA-fault report has no `scope`: {}",
        v.json
    );
    assert!(
        v.stderr.contains("single-assignment"),
        "positive control: this body decoded and failed the SSA check: {}",
        v.stderr
    );
}

#[test]
fn a_body_that_does_not_decode_carries_no_scope() {
    let dir = common::scratch_dir("verify_scope");
    let art = dir.join("garbage.mic3");
    std::fs::write(&art, b"this is not a mic@3 artifact").expect("write artifact");

    let v = verify_json(&art);
    assert_eq!(
        v.code,
        Some(1),
        "an undecodable body must exit 1: {}",
        v.stderr
    );
    assert_eq!(
        v.json["ssa_reason"], "mic@3 body did not parse for SSA check",
        "positive control: this is the not-decoded arm: {}",
        v.json
    );
    assert!(
        v.json.get("scope").is_none(),
        "the SSA-fault report has no `scope`: {}",
        v.json
    );
}
