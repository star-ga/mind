// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Project alias resolution in lowering is the one on main.
//!
//! `type_aliases::resolve_active` feeds the lowering cast widths, the
//! declared widths, narrow arithmetic and struct resolution. On main it
//! resolves the module's own aliases, then reads a top-level owner-qualified
//! alias (`defs.Small`) through the project registry. Merging main into the
//! string-comparison branch routed it through `resolve_with_imports`
//! instead, which rewrites every nested project alias and type. Compiled
//! project builds changed outside the string scope: a function returning an
//! imported array alias (`Banks = [Bank; 2]`) gave an internal MLIR lowering
//! error with no diagnostic code where main refuses it with `E6009`, and a
//! cast to a bare imported alias (`big as Small`) gave another value.
//!
//! Each case is a two-module project (`defs.mind` declares the struct and
//! the aliases) built with `mindc build --emit cdylib`; the value is read
//! from the built library. The expected results are main's.
//!
//! Two string-pass blockers of the same review are pinned here too: a
//! string stored into the `i64` payload of an imported enum must be refused
//! in the declaring module as it is in one file, and `mindc test` must keep
//! the raw-memory escapes of each merged std module apart, as the build
//! analyses each std module alone.
//!
//! Gate: `cargo test --features "mlir-build std-surface cross-module-imports" \
//!        --test string_ops_native_r5`

#![cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]

mod common;
use common::{mindc_bin, scratch_dir};

use std::path::Path;
use std::process::{Command, Output};

const MANIFEST: &str = "[package]\nname = \"xmod\"\nversion = \"0.1.0\"\n\n[build]\n\
                        entry = \"main.mind\"\nemit = \"cdylib\"\n\n[targets.cpu]\n\
                        backend = \"cpu\"\nsources = [\"main.mind\", \"defs.mind\"]\n";

const DEFS: &str = "pub struct Bank { id: i64, cash: i64 }\n\npub type Small = i32\n\npub type Banks = [Bank; 2]\n";

/// What main gives for a project build of one case.
enum Expected {
    /// The build succeeds and `probe()` returns this value.
    Value(i64),
    /// The build is refused with this diagnostic code and writes nothing.
    Refused(&'static str),
}

/// `(case, main.mind, main's result)`.
const CASES: &[(&str, &str, Expected)] = &[
    (
        "array_of_imported_struct",
        "import defs\n\nfn banks() -> [Bank; 2] {\n    \
         return [Bank { id: 1, cash: 10 }, Bank { id: 2, cash: 20 }]\n}\n\n\
         pub fn probe() -> i64 {\n    return banks()[1].cash\n}\n",
        Expected::Value(20),
    ),
    (
        "imported_array_alias",
        "import defs\n\nfn banks() -> Banks {\n    \
         return [Bank { id: 1, cash: 10 }, Bank { id: 2, cash: 20 }]\n}\n\n\
         pub fn probe() -> i64 {\n    let xs = banks()\n    return xs[1].cash\n}\n",
        Expected::Refused("E6009"),
    ),
    (
        "cast_to_bare_imported_alias",
        "import defs\n\npub fn probe() -> i64 {\n    let big: i64 = 4294967298\n    \
         let n = big as Small\n    return n\n}\n",
        Expected::Value(4294967298),
    ),
    (
        "cast_to_qualified_imported_alias",
        "import defs\n\npub fn probe() -> i64 {\n    let big: i64 = 4294967298\n    \
         let n = big as defs.Small\n    return n\n}\n",
        Expected::Value(2),
    ),
    (
        "result_of_bare_imported_alias",
        "import defs\n\nfn narrow(x: i64) -> Small {\n    return x\n}\n\n\
         pub fn probe() -> i64 {\n    return narrow(4294967298)\n}\n",
        Expected::Value(4294967298),
    ),
];

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn write_project(dir: &Path, main: &str) {
    std::fs::write(dir.join("Mind.toml"), MANIFEST).expect("write manifest");
    std::fs::write(dir.join("defs.mind"), DEFS).expect("write defs");
    std::fs::write(dir.join("main.mind"), main).expect("write main");
}

/// `probe()` of the built library, in a child process so a crash names its
/// case.
fn probe_value(case: &str, so: &Path) -> i64 {
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
        "{case}: probe crashed:\n{}",
        text(&ran)
    );
    String::from_utf8_lossy(&ran.stdout)
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("{case}: probe printed no i64:\n{}", text(&ran)))
}

#[test]
fn project_alias_resolution_in_lowering_is_mains() {
    for (case, main, expected) in CASES {
        let dir = scratch_dir(&format!("string_ops_native_r5_{case}"));
        write_project(&dir, main);
        let main_path = dir.join("main.mind");
        let check = Command::new(mindc_bin())
            .args(["check", "--no-fmt", main_path.to_str().unwrap()])
            .output()
            .expect("spawn mindc check");
        assert!(
            check.status.success(),
            "{case}: check refused:\n{}",
            text(&check)
        );
        let so = dir.join("x.so");
        let _ = std::fs::remove_file(&so);
        let build = Command::new(mindc_bin())
            .args([
                "build",
                "--backend",
                "mlir",
                "--emit",
                "cdylib",
                "--out",
                so.to_str().unwrap(),
            ])
            .current_dir(&dir)
            .output()
            .expect("spawn mindc build");
        let build_text = text(&build);
        assert!(!build_text.contains("panicked"), "{case}:\n{build_text}");
        match expected {
            Expected::Refused(code) => {
                assert!(
                    !build.status.success(),
                    "{case}: build accepted:\n{build_text}"
                );
                assert!(
                    build_text.contains(code),
                    "{case}: no {code} refusal:\n{build_text}"
                );
                assert!(
                    !build_text.contains("MLIR lowering failed"),
                    "{case}: an internal lowering error instead of {code}:\n{build_text}"
                );
                assert!(!so.exists(), "{case}: a refused build left an artifact");
            }
            Expected::Value(value) => {
                if !common::gate::compiled("string_ops_native_r5", &build) {
                    return;
                }
                assert_eq!(probe_value(case, &so), *value, "{case}: probe() value");
            }
        }
    }
}

/// The module of blocker `cross-module-variant-payload-string-compares-as-handle`:
/// it declares the enum and compares two `i64` payloads.
const EVENTS: &str = concat!(
    "pub enum XmlEvent { Start, Text(i64), End }\n\n",
    "pub fn same_text(a: XmlEvent, b: XmlEvent) -> i64 {\n",
    "    let x = match a {\n",
    "        XmlEvent::Text(n) => n,\n",
    "        _ => 0,\n",
    "    }\n",
    "    let y = match b {\n",
    "        XmlEvent::Text(n) => n,\n",
    "        _ => 1,\n",
    "    }\n",
    "    if x == y {\n",
    "        return 1\n",
    "    }\n",
    "    return 0\n",
    "}\n",
);

/// Stores a string into each payload and compares them through `same_text`.
const PAYLOAD_PROBE: &str = concat!(
    "pub fn probe() -> i64 {\n",
    "    let a = XmlEvent::Text(\"abc\")\n",
    "    let b = XmlEvent::Text(\"abc\")\n",
    "    return same_text(a, b)\n",
    "}\n",
);

/// The test `mindc test` runs on a payload project's entry.
const PAYLOAD_TEST: &str =
    "#[test]\nfn probed() {\n    let v: i64 = probe()\n    assert v != 1, \"value is 1.\"\n}\n";

/// A private enum of the same name: the registry keys both by their owner.
const SHADOW: &str = "export { shadow_code }\n\nenum XmlEvent { Open, Close }\n\n\
                      fn shadow_code() -> i64 {\n    return 7\n}\n";

/// The test of blocker `mindc-test-refuses-std-code-after-user-string-escape`.
const STD_ESCAPE: &str = concat!(
    "use std.vec;\nuse std.ecdsa_p256;\n\n",
    "fn keep(v: Vec, s: string) -> Vec { return vec_push(v, s) }\n\n",
    "pub fn probe() -> i64 { return vec_len(keep(vec_new(), \"abc\")) }\n\n",
    "#[test]\nfn probed() {\n    let v: i64 = probe()\n    assert v != 1, \"value is 1.\"\n}\n",
);

/// A string the user's own code stores to raw memory, read back by std code.
const RAW_STRING_READ_BY_STD: &str = concat!(
    "use std.vec;\n\n",
    "pub fn probe() -> i64 {\n",
    "    let v = vec_push(vec_new(), 0)\n",
    "    __mind_store_i64(vec_addr(v), \"abc\")\n",
    "    if vec_get(v, 0) == vec_get(v, 0) {\n",
    "        return 1\n",
    "    }\n",
    "    return 0\n",
    "}\n\n",
    "#[test]\nfn probed() {\n    let v: i64 = probe()\n    assert v != 1, \"value is 1.\"\n}\n",
);

/// A string handed to `vec_push` escapes into a `Vec`; `string_len` takes
/// no `Vec`, so its results stay integers on every backend.
const RECORD_ESCAPE: &str = concat!(
    "use std.vec;\nuse std.string;\n\n",
    "fn keep(v: Vec, s: string) -> Vec { return vec_push(v, s) }\n\n",
    "pub fn probe() -> i64 {\n",
    "    let v = keep(vec_new(), \"abc\")\n",
    "    if string_len(\"ab\") == string_len(\"ab\") {\n",
    "        return vec_len(v)\n",
    "    }\n",
    "    return 0\n",
    "}\n\n",
    "#[test]\nfn probed() {\n    let v: i64 = probe()\n    assert v != 1, \"value is 1.\"\n}\n",
);

/// `mindc <args>` run in `dir`.
fn mindc_in(dir: &Path, args: &[&str]) -> Output {
    Command::new(mindc_bin())
        .args(args)
        .current_dir(dir)
        .output()
        .expect("spawn mindc")
}

/// `out` is a refusal with `E2038`, not a crash.
fn assert_refused(case: &str, step: &str, out: &Output) {
    let log = text(out);
    assert!(
        !out.status.success() && log.contains("E2038") && !log.contains("panicked"),
        "{case}: {step} not refused with E2038:\n{log}"
    );
}

/// One module stores a string into the `i64` payload of another module's
/// enum, and the declaring module compares two such payloads. The same
/// source in one file is refused with `E2038`; the project must be refused
/// too, not built to compare the two string handles (`0` for equal text).
#[test]
fn a_string_stored_in_an_imported_variant_payload_is_tracked() {
    let one = scratch_dir("string_ops_native_r5_payload_one_file");
    std::fs::write(one.join("one.mind"), format!("{EVENTS}\n{PAYLOAD_PROBE}")).expect("write");
    let check = mindc_in(&one, &["check", "--no-fmt", "one.mind"]);
    assert_refused("one_file", "check", &check);
    for (case, shadow) in [
        ("payload_two_modules", false),
        ("payload_colliding_enum", true),
    ] {
        let dir = scratch_dir(&format!("string_ops_native_r5_{case}"));
        std::fs::create_dir_all(dir.join(".git")).expect("project boundary");
        let mut sources = vec!["main.mind", "events.mind"];
        let mut imports = String::from("import events\n");
        if shadow {
            sources.push("shadow.mind");
            imports.push_str("import shadow\n");
            std::fs::write(dir.join("shadow.mind"), SHADOW).expect("write shadow");
        }
        let manifest = format!(
            "[package]\nname = \"xmod\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
             emit = \"cdylib\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = {sources:?}\n"
        );
        std::fs::write(dir.join("Mind.toml"), manifest).expect("write manifest");
        std::fs::write(dir.join("events.mind"), EVENTS).expect("write events");
        let main = format!("{imports}\n{PAYLOAD_PROBE}\n{PAYLOAD_TEST}");
        std::fs::write(dir.join("main.mind"), main).expect("write main");
        let check = mindc_in(&dir, &["check", "--no-fmt", "main.mind"]);
        assert_refused(case, "check", &check);
        let so = dir.join("x.so");
        let _ = std::fs::remove_file(&so);
        let build = mindc_in(
            &dir,
            &[
                "build",
                "--backend",
                "mlir",
                "--emit",
                "cdylib",
                "--out",
                so.to_str().unwrap(),
            ],
        );
        assert_refused(case, "build", &build);
        assert!(!so.exists(), "{case}: a refused build left an artifact");
        let test = mindc_in(&dir, &["test", "main.mind"]);
        assert_refused(case, "mindc test", &test);
        // Positive control: integers in the payloads build and compare equal.
        let ints = format!("{imports}\n{}", PAYLOAD_PROBE.replace("\"abc\"", "7"));
        std::fs::write(dir.join("main.mind"), ints).expect("write main");
        let build = mindc_in(
            &dir,
            &[
                "build",
                "--backend",
                "mlir",
                "--emit",
                "cdylib",
                "--out",
                so.to_str().unwrap(),
            ],
        );
        if common::gate::compiled("string_ops_native_r5", &build) {
            assert_eq!(probe_value(case, &so), 1, "{case}: integer payloads");
        }
    }
}

/// `probe()` of `file` in `dir` built with the MLIR backend.
fn mlir_probe(case: &str, dir: &Path, file: &str) -> Option<i64> {
    let so = dir.join(format!("{case}.so"));
    let _ = std::fs::remove_file(&so);
    let build = mindc_in(dir, &[file, "--emit-shared", so.to_str().unwrap()]);
    common::gate::compiled("string_ops_native_r5", &build).then(|| probe_value(case, &so))
}

/// A string handed to `vec_push` escapes into a `Vec` record only:
/// `string_len`, which keeps no `Vec`, gives integers back. The build and
/// `mindc test` both compute 1.
#[test]
fn record_escape_does_not_taint_recordless_callees() {
    let dir = scratch_dir("string_ops_native_r5_record_escape");
    std::fs::write(dir.join("record_test.mind"), RECORD_ESCAPE).expect("write");
    if let Some(value) = mlir_probe("record_escape", &dir, "record_test.mind") {
        assert_eq!(value, 1, "record_escape: build value");
    }
    let log = text(&mindc_in(&dir, &["test", "record_test.mind"]));
    assert!(
        log.contains("value is 1.") && !log.contains("E2038"),
        "record_escape: mindc test gave no value 1:\n{log}"
    );
}

/// A string handed to `vec_push` escapes to raw memory in std.vec's merged
/// body; `mindc test` must still run std.ecdsa_p256's integer comparisons of
/// raw loads, as the build does. A string the user's code stores to raw
/// memory is still one when std code loads it back.
#[test]
fn mindc_test_keeps_the_escapes_of_each_std_module_apart() {
    let dir = scratch_dir("string_ops_native_r5_std_escapes");
    std::fs::write(dir.join("escape_test.mind"), STD_ESCAPE).expect("write");
    if let Some(value) = mlir_probe("std_escape", &dir, "escape_test.mind") {
        assert_eq!(value, 1, "std_escape: build value");
    }
    let log = text(&mindc_in(&dir, &["test", "escape_test.mind"]));
    assert!(
        log.contains("value is 1.") && !log.contains("E2038"),
        "std_escape: mindc test gave no value 1:\n{log}"
    );
    std::fs::write(dir.join("raw_test.mind"), RAW_STRING_READ_BY_STD).expect("write");
    let raw = mindc_in(&dir, &["test", "raw_test.mind"]);
    assert_refused("raw_string_read_by_std", "mindc test", &raw);
}
