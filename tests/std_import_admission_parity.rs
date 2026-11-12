// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc test` admits from std exactly what `mindc build` links for the same
//! program, and resolves a std function's calls the way the link does.
//!
//! Each case is ONE program. The interpreter leg runs it under `mindc test`;
//! the native leg builds the same source with `fn main() -> i64 { return
//! score(); }` and runs the artifact (or, for a refused call, expects the link
//! to fail on that symbol). Every passing interpreter leg also runs with a
//! WRONG expected value and must FAIL, so a runner that passes everything
//! cannot pass this file.
//!
//! * An import of a std module that itself imports `std.vec` / `std.map`
//!   (`std.toml`, `std.cli`, `std.http`, `std.tui`) does not admit the
//!   ambient module in full: `vec_free` and `map_str_key_eq` are not in the
//!   runtime support the build links, so `mindc test` refuses them as the
//!   linker does (`undefined reference`). (`vec_zeroed` is unlinkable too, but
//!   the evaluator answers a bare `vec_zeroed` through its extern-C runtime
//!   intrinsic whatever std admits, with or without an import; that path is
//!   not std admission and is not covered here.)
//! * The refusal holds in every import order and next to an import whose
//!   graph is several levels deep: admission is a membership test on the
//!   build's link set, not the order of a walk.
//! * A file fn named like a std helper (`sha_ch`, reached through `std.hkdf` or
//!   imported directly through `std.sha256`) or like a std function another
//!   std module calls (`sha256`, called by `std.hkdf`) does not replace the
//!   std definition inside std code: the build keeps the std one, and so does
//!   `mindc test`.
//!
//! The native leg needs the MLIR backend; without it the step is a coded
//! capability skip ([`common::gate`]), which `MIND_BENCH_REQUIRE=1` turns
//! into a failure.
//!
//! Gate: `cargo test --release --test std_import_admission_parity`

#![cfg(feature = "std-surface")]

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// `import std.toml` (which imports std.vec, std.map and std.string) plus a
/// call to an ambient std.vec fn the runtime support does not link.
const TOML_VEC_FREE: &str = r#"import std.toml;

fn score() -> i64 {
    let v: Vec = vec_new();
    return vec_free(v) + 5;
}
"#;

/// The linkable control: `vec_len` is a runtime-support name.
const TOML_VEC_LEN: &str = r#"import std.toml;

fn score() -> i64 {
    let v: Vec = vec_new();
    return vec_len(v) + 5;
}
"#;

/// HMAC-SHA256('', '')[0] = 0xb6 = 182 through `import std.hkdf`, with a file
/// fn named like std.sha256's private helper `sha_ch`.
const HKDF_FILE_CH: &str = r#"import std.hkdf;

fn sha_ch(x: i64, y: i64, z: i64) -> i64 {
    return 0;
}

fn score() -> i64 {
    let key: i64 = __mind_alloc(1);
    let msg: i64 = __mind_alloc(1);
    let out: i64 = __mind_alloc(32);
    hmac_sha256(key, 0, msg, 0, out);
    return __mind_load_i8(out);
}
"#;

/// SHA-256('')[0] = 0xe3 = 227 through a DIRECT `import std.sha256`, with a
/// file fn named like its private helper `sha_ch`.
const SHA256_FILE_CH: &str = r#"import std.sha256;

fn sha_ch(x: i64, y: i64, z: i64) -> i64 {
    return 0;
}

fn score() -> i64 {
    let msg: i64 = __mind_alloc(1);
    let out: i64 = __mind_alloc(32);
    sha256(msg, 0, out);
    return __mind_load_i8(out);
}
"#;

/// HMAC-SHA256('', '')[0] = 182 through `import std.hkdf`, with a file fn
/// named like the std.sha256 function std.hkdf calls.
const HKDF_FILE_SHA256: &str = r#"import std.hkdf;

fn sha256(in_addr: i64, in_len: i64, out_addr: i64) -> i64 {
    return 0;
}

fn score() -> i64 {
    let key: i64 = __mind_alloc(1);
    let msg: i64 = __mind_alloc(1);
    let out: i64 = __mind_alloc(32);
    hmac_sha256(key, 0, msg, 0, out);
    return __mind_load_i8(out);
}
"#;

/// One case, one directory. The cases run on parallel threads; two sharing a
/// path raced (one `mindc test` read the other's truncated write and found 0
/// tests), so a label already taken in this process is refused, not raced.
fn scratch(label: &str) -> PathBuf {
    static TAKEN: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    let fresh = TAKEN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(label.to_owned());
    assert!(fresh, "two cases share the scratch label {label}");
    let dir = common::scratch_dir("std_import_admission_parity").join(label);
    fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Run `mindc test` on `program` plus a `parity` test asserting
/// `score() == expected`; returns `(exit code, output)`.
fn mindc_test(stem: &str, program: &str, expected: i64) -> (i32, String) {
    let path = scratch(&format!("{stem}-expect{expected}")).join(format!("{stem}.mind"));
    let src = format!(
        "{program}\n#[test]\nfn parity() {{\n    let got: i64 = score();\n    \
         assert got == {expected}, \"score\";\n}}\n"
    );
    fs::write(&path, src).expect("write test source");
    let out = Command::new(common::require_mindc())
        .arg("test")
        .arg(&path)
        .output()
        .expect("spawn mindc test");
    (out.status.code().unwrap_or(-1), text(&out))
}

/// The interpreter leg: the right score passes, a wrong one fails its assert.
fn assert_interpreter(stem: &str, program: &str, expected: i64) {
    let (code, out) = mindc_test(stem, program, expected);
    assert_eq!(
        code, 0,
        "mindc test must pass with score {expected}:\n{out}"
    );
    assert!(
        out.contains(&format!("test {stem}::parity ... ok")) && out.contains("1 passed; 0 failed"),
        "expected the parity test to run and pass:\n{out}"
    );
    let (code, out) = mindc_test(stem, program, expected + 1);
    assert_ne!(code, 0, "a wrong expected score must fail:\n{out}");
    assert!(
        out.contains(&format!("test {stem}::parity ... FAILED")) && out.contains("score"),
        "expected the wrong-score control to fail its assert:\n{out}"
    );
}

/// The interpreter refuses the call the build cannot link: the test fails
/// with the evaluator's refusal, never runs the assert to a pass.
fn assert_interpreter_refuses(stem: &str, program: &str) {
    let (code, out) = mindc_test(stem, program, 5);
    assert_ne!(
        code, 0,
        "the build cannot link this call, so mindc test must refuse it:\n{out}"
    );
    assert!(
        out.contains(&format!("test {stem}::parity ... FAILED"))
            && out.contains("unsupported operation"),
        "expected the refusal an unlinked call gives:\n{out}"
    );
}

/// Write a Mind.toml project whose entry is `program` plus a `main`.
fn native_project(stem: &str, program: &str) -> PathBuf {
    let dir = scratch(&format!("{stem}-native"));
    fs::create_dir_all(dir.join(".git")).expect("project boundary");
    fs::write(
        dir.join("Mind.toml"),
        format!(
            "[package]\nname = \"{stem}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("write manifest");
    fs::write(
        dir.join("main.mind"),
        format!("{program}\nfn main() -> i64 {{\n    return score();\n}}\n"),
    )
    .expect("write entry");
    dir
}

/// The native leg: build, run, and compare the exit status with `expected`.
fn assert_native(stem: &str, program: &str, expected: i64) {
    assert!((0..256).contains(&expected), "an exit status holds 0..=255");
    let dir = native_project(stem, program);
    let built = common::run_build_captured(&common::require_mindc(), &dir, &[]);
    if !common::gate::compiled(&format!("std_import_admission_parity::{stem}"), &built) {
        return;
    }
    let artifact = common::reported_artifact(&built);
    let ran = Command::new(&artifact)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e}", artifact.display()));
    assert_eq!(
        ran.status.code(),
        Some(expected as i32),
        "the compiled {stem} program must exit with score {expected}:\n{}",
        text(&ran)
    );
}

/// The native leg of a refused call: the build must fail to link `symbol`.
/// An executable names a MIND function `mind.<name>` at the object level.
fn assert_native_link_refuses(stem: &str, program: &str, symbol: &str) {
    let dir = native_project(stem, program);
    let built = common::run_build_captured(&common::require_mindc(), &dir, &[]);
    let log = text(&built);
    if built.status.success() {
        panic!("the compiled build must not link {symbol}:\n{log}");
    }
    if common::gate::is_capability_gap(&String::from_utf8_lossy(&built.stderr)) {
        assert!(
            !common::gate::enforce_real_backend(),
            "MIND_BENCH_REQUIRE=1: the native leg of {stem} hit a capability gap:\n{log}"
        );
        return;
    }
    assert!(
        log.contains(&format!("undefined reference to `mind.{symbol}'")),
        "expected the linker's missing-symbol refusal for {symbol}:\n{log}"
    );
}

fn assert_parity(stem: &str, program: &str, expected: i64) {
    assert_interpreter(stem, program, expected);
    assert_native(stem, program, expected);
}

/// pr365 explicit-import-readmits-unlinkable-ambient-std and pr346
/// std-import-transitive-admits-unlinkable-ambient: `import std.toml` plus
/// `vec_free` passed `mindc test` while `mindc build` failed with
/// "undefined reference to `vec_free'" (`mind.vec_free` since the MIND
/// object symbol namespace).
#[test]
fn an_import_reaching_std_vec_does_not_admit_its_unlinkable_fns() {
    assert_interpreter_refuses("toml_vec_free", TOML_VEC_FREE);
    assert_native_link_refuses("toml_vec_free", TOML_VEC_FREE, "vec_free");
    // The linkable control keeps passing on both legs.
    assert_parity("toml_vec_len", TOML_VEC_LEN, 5);
}

/// The unlinkable ambient names the audit named, through every shipped std
/// module that imports an ambient one (std.toml, std.cli, std.http, std.tui).
#[test]
fn every_import_reaching_an_ambient_module_refuses_its_unlinkable_fns() {
    let calls = [
        (
            "vec_free",
            "let v: Vec = vec_new();\n    return vec_free(v) + 5;",
        ),
        ("map_str_key_eq", "return map_str_key_eq(0, 0) * 0 + 5;"),
    ];
    for module in ["toml", "cli", "http", "tui"] {
        for (name, body) in calls {
            // `ambient_` keeps toml/vec_free off the stem the test above owns.
            let stem = format!("ambient_{module}_{name}");
            let program = format!("import std.{module};\n\nfn score() -> i64 {{\n    {body}\n}}\n");
            assert_interpreter_refuses(&stem, &program);
        }
    }
    let program =
        "import std.toml;\n\nfn score() -> i64 {\n    return map_str_key_eq(0, 0) * 0 + 5;\n}\n";
    assert_native_link_refuses("toml_map_str_key_eq", program, "map_str_key_eq");
}

/// The review case on #365: ambient admission depended on the order of a
/// breadth-first walk. Admission is a membership test now: the same refusal
/// in either import order, next to an import whose graph is several levels
/// deep (std.tls13_handshake reaches std.sha256 through std.tls13_finished,
/// std.tls13_keyschedule and std.hkdf).
#[test]
fn ambient_admission_does_not_depend_on_import_order_or_depth() {
    let body =
        "\nfn score() -> i64 {\n    let v: Vec = vec_new();\n    return vec_free(v) + 5;\n}\n";
    let orders = [
        (
            "order_deep_first",
            "import std.tls13_handshake;\nimport std.toml;\n",
        ),
        (
            "order_deep_last",
            "import std.toml;\nimport std.tls13_handshake;\n",
        ),
    ];
    for (stem, imports) in orders {
        assert_interpreter_refuses(stem, &format!("{imports}{body}"));
    }
    assert_native_link_refuses(
        "order_deep_first",
        &format!("{}{body}", orders[0].1),
        "vec_free",
    );
}

/// pr365 transitive-std-helper-hijacked-by-file-fn: a file fn `ch` replaced
/// std.sha256's private `ch` (now `sha_ch`) inside std code reached through std.hkdf, so
/// `mindc test` computed a wrong HMAC while the build (internal linkage for
/// the std helper) exits 182.
#[test]
fn a_file_fn_does_not_replace_a_transitive_std_helper() {
    assert_parity("hkdf_user_ch", HKDF_FILE_CH, 182);
}

/// The same hijack through a DIRECT `import std.sha256`: the build exits 227.
#[test]
fn a_file_fn_does_not_replace_a_directly_imported_std_helper() {
    assert_parity("sha256_user_ch", SHA256_FILE_CH, 227);
}

/// A std module's call into ANOTHER std module (std.hkdf calling `sha256`)
/// binds to the std definition the build links, not to a file fn of that
/// name: the build makes the file's `sha256` internal and exits 182.
#[test]
fn a_file_fn_does_not_replace_a_std_function_another_std_module_calls() {
    assert_parity("hkdf_user_sha256", HKDF_FILE_SHA256, 182);
}
