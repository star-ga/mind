// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc check` reports formatting drift that `mindc fmt --check` reports.
//!
//! `check` formats each file inside the project's scope, where the parser
//! rewrote an imported type `a.E` to its registry key. The printed file lost
//! `a.`, the formatter refused to write it (it would delete source), and
//! `check` read that refusal as a parse error and skipped the file: a badly
//! laid-out file passed `check` while `fmt --check` rejected it. The formatter
//! now parses without cross-module resolution, as a standalone `mindc fmt`
//! does, and a refusal that remains is reported as a warning instead of
//! passing the file unchecked.

mod common;

use std::path::Path;
use std::process::Command;

fn project(target: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let root = common::scratch_dir(target);
    std::fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"qenum\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Mind.toml");
    for (name, text) in files {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, text).expect("write source");
    }
    root
}

fn run(root: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(common::mindc_bin())
        .args(args)
        .current_dir(root)
        .output()
        .expect("run mindc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

const ENUM_MODULE: &str = "enum E {\n    A,\n    B,\n}\n";
const MAIN: &str = "fn main() -> i64 {\n    return 0;\n}\n";

#[test]
fn a_badly_laid_out_file_with_an_imported_type_fails_check() {
    let bad = "import a;\n\nfn f(e: a.E) -> i64 {\n  return   1;\n}\n";
    let root = project(
        "check_fmt_drift_qualified_type",
        &[
            ("src/a.mind", ENUM_MODULE),
            ("src/main.mind", MAIN),
            ("src/en.mind", bad),
        ],
    );
    let (ok, text) = run(&root, &["check", "src/en.mind"]);
    assert!(!ok, "check must fail like `fmt --check`:\n{text}");
    assert!(text.contains("fmt::drift"), "{text}");
    // And the rewrite `check` asks for keeps `a.E`.
    let (ok, text) = run(&root, &["fmt", "src/en.mind"]);
    assert!(ok, "{text}");
    let formatted = std::fs::read_to_string(root.join("src/en.mind")).expect("read");
    assert_eq!(
        formatted,
        "import a;\n\nfn f(e: a.E) -> i64 {\n    return 1;\n}\n"
    );
    // The formatted file has no drift left. (Without `cross-module-imports`
    // the qualified type itself is a separate type-check error.)
    let (_, text) = run(&root, &["check", "src/en.mind"]);
    assert!(!text.contains("fmt::drift"), "{text}");
}

#[test]
fn a_file_the_formatter_refuses_is_reported_not_passed_silently() {
    // A bool literal in a match pattern still prints as `1` / `0`, so the
    // formatter refuses the file; `check` must say so.
    let boolpat = "fn f(b: bool) -> i64 {\n  match b {\n    true => {\n      return 1;\n    },\n    false => {\n      return 0;\n    },\n  }\n}\n";
    let root = project("check_fmt_unformattable", &[("boolpat.mind", boolpat)]);
    let (_, text) = run(&root, &["check", "boolpat.mind"]);
    assert!(text.contains("fmt::unformattable"), "{text}");
}
