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
            .args(["build", "--emit", "cdylib", "--out", so.to_str().unwrap()])
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
