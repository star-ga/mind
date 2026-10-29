// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Every `lane.*` refusal key of the candidate native lane, pinned by its exact key.
//!
//! Each row names one key, its fixture in `tests/fixtures/native_lane_refusals/`
//! (the key with dots as underscores) and a positive control: the one edit that
//! removes the refused construct. The fixture must refuse with exactly
//! `outside candidate native-lane-v1 (<key>)` and write no artifact; the control
//! must build. A fixture that refuses for any other reason, or a control that
//! does not build, fails its row.
//!
//! The ratchet reads every `"lane.<...>"` literal in the lane profile and the
//! native bridge sources. A key with no row and no `UNREACHABLE` entry fails, and
//! so does a row or entry whose key is gone from the source. A change that adds
//! a key therefore lands with its row.

// The build rows write a POSIX shell shim (tests/harness_portability.rs).
#![cfg(unix)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

struct Row {
    key: &'static str,
    /// Exact text, present once in the fixture, that makes it refuse.
    refused: &'static str,
    /// Replacement for `refused` that leaves an admitted program.
    control: &'static str,
}

const fn row(key: &'static str, refused: &'static str, control: &'static str) -> Row {
    Row {
        key,
        refused,
        control,
    }
}

const ROWS: &[Row] = &[
    row(
        "lane.alloc.size_positive",
        "__mind_alloc(0)",
        "__mind_alloc(8)",
    ),
    row("lane.alloc.free_in_loop", "__mind_free(p); ", ""),
    row("lane.binop.non_scalar", "h() + h()", "1 + 1"),
    row("lane.bool.width", "let b: bool = 1;", "let b: i64 = 1;"),
    row("lane.call.arity", "__mind_alloc()", "__mind_alloc(8)"),
    row("lane.call.unknown", "__mind_forged(8)", "8"),
    row(
        "lane.entry.f64_result",
        "-> f64 { return 1.0; }",
        "-> i64 { return 1; }",
    ),
    row(
        "lane.extern.unimplemented",
        "extern \"C\" { fn close(fd: i64) -> i64; }",
        "",
    ),
    row("lane.source.binding_type", "let y = h();", "h();"),
    row("lane.source.cast", "1 as i32", "1"),
    row("lane.source.const_attributes", "#[inline] ", ""),
    row("lane.source.const_type", "const U = h();", "const U = 1;"),
    row(
        "lane.source.duplicate_parameter",
        "fn f(a: i64, a: i64)",
        "fn f(a: i64, b: i64)",
    ),
    row(
        "lane.source.float_neg_lowering",
        "let y = -x;",
        "let y = x;",
    ),
    row("lane.source.function_contract", "fn id<T>(", "fn id("),
    row("lane.source.loop_attributes", "#[collapse] ", ""),
    row("lane.source.loop_exit_scope", "break; ", ""),
    row("lane.source.neg_type", "let y = -h();", "h();"),
    row(
        "lane.source.nested_function",
        "fn inner() -> i64 { return 1; } ",
        "",
    ),
    row("lane.source.print", "print(1); ", ""),
    row("lane.source.string", "\"hi\"", "1"),
    row("lane.source.top_level", "struct P { x: i64 }", ""),
    row("lane.source.unbound", "h = 2; ", ""),
    row("lane.source.unsupported", "let t = (1, 2); ", ""),
    row("lane.type.branch_mismatch", "else { 2.0 }", "else { 2 }"),
    row("lane.type.mismatch", "let x: f64 = 1;", "let x: f64 = 1.0;"),
    row("lane.type.non_f64_float", "let x: f32", "let x: f64"),
    row("lane.type.unsupported", "let x: i32", "let x: i64"),
];

/// Keys no source program reaches through `mindc build --backend native`: an
/// earlier refusal or a lowering invariant always comes first. Each stays in
/// the source as defence in depth; the unit tests in the profile reach the IR
/// ones by editing lowered IR. A key moves to `ROWS` once a program reaches it.
const UNREACHABLE: &[(&str, &str)] = &[
    (
        "lane.source.duplicate_function",
        "the bridge's ambiguity guard refuses a duplicate definition first (closure.duplicate_definition)",
    ),
    (
        "lane.ir.aggregate_metadata",
        "module value types come only from aggregates, which the source walker refuses",
    ),
    (
        "lane.ir.function_binding",
        "lowering emits one function per admitted top-level fn, with its declared parameters",
    ),
    (
        "lane.ir.loop_arity",
        "lowering builds a loop's live, init and exit lists from one list",
    ),
    (
        "lane.ir.loop_binding",
        "lowering carries each loop variable once and passes all of them at every exit",
    ),
    (
        "lane.ir.loop_exit_scope",
        "lane.source.loop_exit_scope refuses a break or continue outside a loop first",
    ),
    (
        "lane.ir.nested_function",
        "lane.source.nested_function refuses a nested fn first",
    ),
    (
        "lane.ir.parameter_binding",
        "lowering binds parameters in declared order; duplicates refuse as lane.source.duplicate_parameter",
    ),
    (
        "lane.ir.semantic_type",
        "a non-scalar semantic type needs an annotation the source walker refuses (lane.type.*)",
    ),
    (
        "lane.ir.ssa",
        "lowering of an admitted program is well-formed SSA; only a lowering defect reaches it",
    ),
    (
        "lane.ir.unsupported",
        "every construct lowering to these instructions refuses at the source first",
    ),
    (
        "lane.ir.untyped_value",
        "every value an admitted program's lowering reads is defined before use",
    ),
];

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_dir() -> PathBuf {
    repo().join("tests/fixtures/native_lane_refusals")
}

fn fixture_path(key: &str) -> PathBuf {
    fixture_dir().join(format!("{}.mind", key.replace('.', "_")))
}

fn rust_files(dir: &Path, skip: &[&str]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .filter(|p| !skip.iter().any(|s| p.file_name().is_some_and(|n| n == *s)))
        .collect();
    files.sort();
    files
}

/// The lane profile (every module but its unit tests) and the native bridge.
fn key_sources() -> Vec<PathBuf> {
    let profile = repo().join("src/ir/native_lane_profile");
    let mut files = rust_files(&profile, &["tests.rs"]);
    for required in ["admit.rs", "types.rs"] {
        assert!(
            files.contains(&profile.join(required)),
            "the ratchet must read src/ir/native_lane_profile/{required}"
        );
    }
    files.push(repo().join("src/build/native_bridge.rs"));
    files.extend(rust_files(&repo().join("src/build/native_bridge"), &[]));
    files
}

/// Every `"lane.<...>"` literal with the places it occurs. A literal that is
/// not a plain dotted key (a format string, say) cannot be checked by key, so
/// it is returned as malformed instead of being skipped.
fn source_keys() -> (BTreeMap<String, Vec<String>>, Vec<String>) {
    let mut keys: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut malformed = Vec::new();
    for file in key_sources() {
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        let name = file
            .strip_prefix(repo())
            .unwrap_or(&file)
            .display()
            .to_string();
        for (at, _) in text.match_indices("\"lane.") {
            let line = text[..at].matches('\n').count() + 1;
            let site = format!("{name}:{line}");
            let body = &text[at + 1..];
            let key = body.find('"').map(|end| &body[..end]).filter(|k| {
                !k.ends_with('.')
                    && k.bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_.".contains(&b))
            });
            match key {
                Some(key) => keys.entry(key.to_string()).or_default().push(site),
                None => malformed.push(site),
            }
        }
    }
    (keys, malformed)
}

#[test]
fn every_lane_key_has_a_fixture_row_or_an_unreachable_entry() {
    let (keys, malformed) = source_keys();
    let mut errors: Vec<String> = malformed
        .iter()
        .map(|site| format!("{site}: a \"lane.\" literal that is not a plain dotted key"))
        .collect();
    let mut tabled = BTreeSet::new();
    let entries = ROWS
        .iter()
        .map(|r| (r.key, "row"))
        .chain(UNREACHABLE.iter().map(|(k, _)| (*k, "UNREACHABLE entry")));
    for (key, kind) in entries {
        if !tabled.insert(key) {
            errors.push(format!("`{key}` is tabled more than once"));
        }
        if !keys.contains_key(key) {
            errors.push(format!(
                "{kind} `{key}` names a key that is no longer in the source"
            ));
        }
    }
    for (key, sites) in &keys {
        if !tabled.contains(key.as_str()) {
            errors.push(format!(
                "`{key}` ({}) has no fixture row and no UNREACHABLE entry",
                sites.join(", ")
            ));
        }
    }
    for (key, reason) in UNREACHABLE {
        if reason.trim().is_empty() {
            errors.push(format!("UNREACHABLE entry `{key}` gives no reason"));
        }
    }
    let mut fixtures = BTreeSet::new();
    for row in ROWS {
        let path = fixture_path(row.key);
        fixtures.insert(path.clone());
        let Ok(text) = std::fs::read_to_string(&path) else {
            errors.push(format!(
                "row `{}`: fixture {} is missing",
                row.key,
                path.display()
            ));
            continue;
        };
        let found = text.matches(row.refused).count();
        if found != 1 || row.refused == row.control {
            errors.push(format!(
                "row `{}`: its refused text {:?} occurs {found} times in the fixture \
                 (want 1, with a different control)",
                row.key, row.refused
            ));
        }
    }
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if !fixtures.contains(&path) {
            errors.push(format!("{} belongs to no row", path.display()));
        }
    }
    println!(
        "native_lane_refusal_keys: {} lane.* keys in source = {} fixture rows + {} unreachable",
        keys.len(),
        ROWS.len(),
        UNREACHABLE.len()
    );
    assert!(
        errors.is_empty(),
        "refusal-key ratchet:\n{}",
        errors.join("\n")
    );
    assert_eq!(keys.len(), ROWS.len() + UNREACHABLE.len());
}

/// Runs the committed seed behind a private shim carrying the lane marker, as
/// `native_lane_bridge_controls.rs` does, so the candidate profile is selected.
#[cfg(all(feature = "std-surface", target_os = "linux", target_arch = "x86_64"))]
mod build {
    use super::{ROWS, fixture_path, repo};
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    const MARK: &[u8] = b"\x48\xb8nblane01";

    struct Built {
        code: Option<i32>,
        stderr: String,
        artifact: Option<Vec<u8>>,
    }

    fn build(source: &str) -> Built {
        let dir = tempfile::tempdir().unwrap();
        let compiler = dir.path().join("marked-compiler.sh");
        let mut shim = b"#!/bin/sh\nexec \"$MIND_NATIVE_TEST_SEED\"\n# ".to_vec();
        shim.extend_from_slice(MARK);
        shim.push(b'\n');
        std::fs::write(&compiler, shim).unwrap();
        std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700)).unwrap();
        let source_path = dir.path().join("main.mind");
        let artifact = dir.path().join("program");
        std::fs::write(&source_path, source).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_mindc"))
            .arg("build")
            .arg(&source_path)
            .args(["--backend", "native", "--emit", "binary", "--out"])
            .arg(&artifact)
            .env("MINDC_NATIVE_ELF", &compiler)
            .env("MINDC_STD_DIR", repo().join("std"))
            .env(
                "MIND_NATIVE_TEST_SEED",
                repo().join("examples/mindc_mind/testdata/selfhost_loop/stage1.elf"),
            )
            .env_remove("MINDC_SO")
            .current_dir(dir.path())
            .output()
            .unwrap();
        Built {
            code: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            artifact: std::fs::read(&artifact).ok(),
        }
    }

    #[test]
    fn every_row_refuses_with_its_exact_key_and_its_control_builds() {
        let mut errors = Vec::new();
        let mut pinned = 0;
        for row in ROWS {
            let before = errors.len();
            let path = fixture_path(row.key);
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let refused = build(&source);
            let exact = format!("outside candidate native-lane-v1 ({})", row.key);
            if refused.artifact.is_some() || refused.code == Some(0) {
                errors.push(format!(
                    "row `{}`: expected refusal, got artifact (exit {:?})",
                    row.key, refused.code
                ));
            } else if refused.code != Some(3)
                || !refused.stderr.contains(&exact)
                || !refused.stderr.contains("No artifact written.")
            {
                errors.push(format!(
                    "row `{}`: want exit 3 with `{exact}` and `No artifact written.`, got \
                     exit {:?}: {}",
                    row.key,
                    refused.code,
                    refused.stderr.trim()
                ));
            }
            let control = build(&source.replacen(row.refused, row.control, 1));
            let elf = control
                .artifact
                .as_deref()
                .is_some_and(|b| b.starts_with(b"\x7fELF"));
            if control.code != Some(0) || !elf {
                errors.push(format!(
                    "row `{}`: control (refused construct removed) did not build: exit {:?}: {}",
                    row.key,
                    control.code,
                    control.stderr.trim()
                ));
            }
            if errors.len() == before {
                pinned += 1;
            }
        }
        println!(
            "native_lane_refusal_keys: {pinned} of {} rows refuse with their exact key and \
             their control builds",
            ROWS.len()
        );
        assert!(errors.is_empty(), "{}", errors.join("\n"));
    }
}
