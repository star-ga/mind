// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `assert(cond, "msg")` asserts `cond`.
//!
//! The parenthesised form parsed as `assert` of ONE tuple `(cond, "msg")`, and a
//! tuple is truthy, so a native build passed every such assert whatever `cond`
//! was (the test evaluator refused the tuple instead). The parser now splits the
//! pair into the condition and the message, as `assert cond, "msg"` spells them,
//! and refuses any other tuple condition, including `assert(cond, 9)`. The
//! formatter keeps printing the form as `assert (cond, "msg")`, as it always did.

mod common;

use std::process::Command;

fn write(target: &str, name: &str, src: &str) -> std::path::PathBuf {
    let dir = common::scratch_dir(target);
    let path = dir.join(name);
    std::fs::write(&path, src).expect("write source");
    path
}

#[test]
fn a_non_string_second_operand_is_refused() {
    let path = write(
        "assert_paren_code",
        "main.mind",
        "fn main() -> i64 {\n    let x: i64 = 3;\n    assert(x == 2, 9);\n    return 5;\n}\n",
    );
    let out = Command::new(common::mindc_bin())
        .arg("check")
        .arg(&path)
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "must not pass: {text}");
    assert!(
        text.contains("second operand must be a string message"),
        "{text}"
    );
}

#[test]
fn a_tuple_condition_is_refused() {
    let path = write(
        "assert_paren_triple",
        "main.mind",
        "fn main() -> i64 {\n    let x: i64 = 3;\n    assert(x == 2, x == 3, x == 4);\n    return 5;\n}\n",
    );
    let out = Command::new(common::mindc_bin())
        .arg("check")
        .arg(&path)
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "must not pass: {text}");
    assert!(text.contains("not a tuple"), "{text}");
}

#[test]
fn the_formatter_keeps_the_parenthesised_spelling() {
    // Canonical output for the form is unchanged: `assert (cond, "msg")`, so a
    // file the formatter accepted before still formats to the same bytes.
    let canonical = "fn main() -> i64 {\n    let x: i64 = 3;\n    assert (x == 3, \"x must be 3\");\n    assert x == 3, \"bare\";\n    assert (x == 3);\n    return 5;\n}\n";
    let path = write("assert_paren_fmt", "main.mind", canonical);
    let out = Command::new(common::mindc_bin())
        .args(["fmt", "--check"])
        .arg(&path)
        .output()
        .expect("run mindc fmt --check");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let tight = canonical.replace("assert (x == 3, ", "assert(x == 3, ");
    let path = write("assert_paren_fmt_tight", "main.mind", &tight);
    let out = Command::new(common::mindc_bin())
        .arg("fmt")
        .arg(&path)
        .output()
        .expect("run mindc fmt");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read_to_string(&path).expect("read"), canonical);
}

#[cfg(all(unix, feature = "mlir-build", feature = "std-surface"))]
#[test]
fn a_native_build_aborts_on_a_false_parenthesised_assert() {
    use std::os::unix::process::ExitStatusExt;
    for (target, cond, aborts) in [
        ("assert_paren_native_false", "x == 2", true),
        ("assert_paren_native_true", "x == 3", false),
    ] {
        let dir = common::scratch_dir(target);
        std::fs::write(
            dir.join("Mind.toml"),
            "[package]\nname = \"ap\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"ap\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
        )
        .expect("write Mind.toml");
        std::fs::write(
            dir.join("main.mind"),
            format!("fn main() -> i64 {{\n    let x: i64 = 3;\n    assert({cond}, \"x must be 2\");\n    return 5;\n}}\n"),
        )
        .expect("write main.mind");
        let out = Command::new(common::mindc_bin())
            .arg("build")
            .current_dir(&dir)
            .output()
            .expect("run mindc build");
        if !common::gate::compiled(target, &out) {
            return;
        }
        let run = Command::new(dir.join("target").join("debug").join("ap"))
            .output()
            .expect("run the built executable");
        if aborts {
            assert!(
                run.status.signal() == Some(6) || run.status.code() == Some(134),
                "{target}: a false assert must abort, got {:?}",
                run.status
            );
        } else {
            assert_eq!(run.status.code(), Some(5), "{target}");
        }
    }
}
