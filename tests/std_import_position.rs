// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A `std` import is a module item: at the top level of a file or in a
//! `module` block (spec/v1.0/grammar-syntax.ebnf, `ModuleItem`).
//!
//! The defect: the parser also accepts an import inside a function or
//! statement body, and `mindc check`, `mindc test` and `--emit-shared` read a
//! program's std imports at module scope only. A body import of a shipped
//! module was accepted and every call into it then refused (`mindc check`
//! E2003 `unsupported call`, `mindc test` `unsupported operation`), and a body
//! import of a module that does not exist was accepted outright. Both are now
//! refused at the import. A sibling of the same class: `--emit-shared` linked
//! the objects of top-level std imports only, so a call through an import in a
//! `module` block checked clean and then could not link.

#![cfg(feature = "std-surface")]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn write_program(case: &str, source: &str) -> PathBuf {
    let dir = common::scratch_dir("std_import_position").join(case);
    std::fs::create_dir_all(dir.join(".git")).expect("create project boundary");
    std::fs::write(
        dir.join("Mind.toml"),
        format!("[package]\nname = \"{case}\"\nversion = \"0.0.1\"\n"),
    )
    .expect("write Mind.toml");
    let file = dir.join("p.mind");
    std::fs::write(&file, source).expect("write p.mind");
    file
}

fn mindc(args: &[&str], dir: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn mindc {args:?}: {e}"))
}

fn run(sub: &str, file: &Path) -> Output {
    let name = file
        .file_name()
        .expect("file name")
        .to_str()
        .expect("utf-8");
    mindc(&[sub, name], file.parent().expect("program dir"))
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// One call into `std.keccak`, which is not an implicit module: its name
/// resolves only through an import. `keccak_sha3_512` returns 0.
const CALL: &str = "let out: i64 = __mind_alloc(64);\n    return keccak_sha3_512(out, 0, out);";

/// A `module` block whose function calls `std.keccak` through the block's own
/// import, plus a `main`.
fn module_block_program() -> String {
    let body = CALL.replace("\n    ", "\n        ");
    format!(
        "module inner {{\n    import std.keccak;\n\n    pub fn digest() -> i64 {{\n        \
         {body}\n    }}\n}}\n\nfn main() -> i64 {{\n    return 0;\n}}\n"
    )
}

#[test]
fn mindc_check_refuses_a_std_import_inside_a_body_at_the_import() {
    // Controls: the same import at the top level and in a module block checks
    // clean, so each refusal below is about the import's position alone.
    for (case, source) in [
        (
            "check_top_level",
            format!("import std.keccak;\n\nfn main() -> i64 {{\n    {CALL}\n}}\n"),
        ),
        ("check_module_block", module_block_program()),
    ] {
        let out = run("check", &write_program(case, &source));
        assert!(
            out.status.success(),
            "{case}: must check clean:\n{}",
            text(&out)
        );
    }

    let cases = [
        (
            "check_body_shipped",
            format!("fn main() -> i64 {{\n    import std.keccak;\n    {CALL}\n}}\n"),
            "p.mind:2:",
            "std.keccak",
            "is not a module item",
        ),
        (
            "check_body_unknown",
            "fn main() -> i64 {\n    import std.no_such_module;\n    return 0;\n}\n".to_string(),
            "p.mind:2:",
            "std.no_such_module",
            "unknown standard-library module",
        ),
        (
            "check_statement_body",
            "fn main() -> i64 {\n    if 1 == 1 {\n        import std.keccak;\n    }\n    \
             return 0;\n}\n"
                .to_string(),
            "p.mind:3:",
            "std.keccak",
            "is not a module item",
        ),
        (
            "check_body_implicit",
            "fn main() -> i64 {\n    import std.vec;\n    return 0;\n}\n".to_string(),
            "p.mind:2:",
            "std.vec",
            "is not a module item",
        ),
    ];
    for (case, source, at, module, why) in cases {
        let out = run("check", &write_program(case, &source));
        let all = text(&out);
        assert!(
            !out.status.success()
                && all.contains(at)
                && all.contains("E2002")
                && all.contains(module)
                && all.contains(why),
            "{case}: must be refused with E2002 at the import ({at}) naming {module} \
             (`{why}`):\n{all}"
        );
        // A call through the refused import may still be reported after it,
        // but the first error is the import: the cause, at its own line.
        let first = all.lines().find(|line| line.contains(": error"));
        assert!(
            first.is_some_and(|line| line.starts_with(at) && line.contains("E2002")),
            "{case}: the first error must be the refusal at the import ({at}):\n{all}"
        );
    }
}

#[test]
fn mindc_test_refuses_a_std_import_inside_a_test_body() {
    let control = write_program(
        "test_top_level",
        "import std.keccak;\n\n#[test]\nfn t() {\n    let out: i64 = __mind_alloc(64);\n    \
         assert keccak_sha3_512(out, 0, out) == 0;\n}\n",
    );
    let out = run("test", &control);
    let all = text(&out);
    assert!(
        out.status.success() && all.contains("1 passed; 0 failed"),
        "control: a top-level import must let the test call std.keccak:\n{all}"
    );

    for (case, source, module, why) in [
        (
            "test_body_shipped",
            "#[test]\nfn t() {\n    import std.keccak;\n    let out: i64 = __mind_alloc(64);\n    \
             assert keccak_sha3_512(out, 0, out) == 0;\n}\n",
            "std.keccak",
            "is not a module item",
        ),
        (
            "test_body_unknown",
            "#[test]\nfn t() {\n    import std.no_such_module;\n    assert 1 == 1;\n}\n",
            "std.no_such_module",
            "unknown standard-library module",
        ),
    ] {
        let out = run("test", &write_program(case, source));
        let all = text(&out);
        assert!(
            !out.status.success() && all.contains(module) && all.contains(why),
            "{case}: `mindc test` must refuse the import naming {module} (`{why}`):\n{all}"
        );
        assert!(
            !all.contains("unsupported operation"),
            "{case}: the refusal must be at the import, not at the call:\n{all}"
        );
    }
}

/// A linked module of a manifest project is type-checked before any test
/// runs, so its body import is refused there too.
#[cfg(feature = "cross-module-imports")]
#[test]
fn mindc_test_refuses_a_std_import_inside_a_linked_module_body() {
    let root = common::scratch_dir("std_import_position_project").join("linked");
    std::fs::create_dir_all(root.join(".git")).expect("create project boundary");
    std::fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"linked\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"consumer.mind\"\n\n\
         [targets.cpu]\nbackend = \"cpu\"\nsources = [\"provider.mind\", \"consumer.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        root.join("provider.mind"),
        format!(
            "export {{ digest }}\nfn digest() -> i64 {{\n    import std.keccak;\n    {CALL}\n}}\n"
        ),
    )
    .expect("write provider.mind");
    std::fs::write(
        root.join("consumer.mind"),
        "import provider;\n\n#[test]\nfn t() {\n    assert digest() == 0;\n}\n",
    )
    .expect("write consumer.mind");
    let out = mindc(&["test", "consumer.mind"], &root);
    let all = text(&out);
    assert!(
        !out.status.success()
            && all.contains("line 3 column 5")
            && all.contains("import std.keccak")
            && all.contains("is not a module item"),
        "the provider's body import must be refused at the import:\n{all}"
    );
}

/// `--emit-shared` type-checks first, so a body import is refused there.
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
#[test]
fn emit_shared_refuses_a_std_import_inside_a_body() {
    let file = write_program(
        "emit_body",
        &format!("pub fn digest() -> i64 {{\n    import std.keccak;\n    {CALL}\n}}\n"),
    );
    let out = mindc(
        &["p.mind", "--emit-shared", "p.so"],
        file.parent().expect("program dir"),
    );
    let all = text(&out);
    assert!(
        !out.status.success() && all.contains("std.keccak") && all.contains("is not a module item"),
        "`--emit-shared` must refuse the body import at the import:\n{all}"
    );
}

/// The build links the object of every module-scope std import, a `module`
/// block's included: the executable runs the block's call into `std.keccak`
/// and exits with the first byte of SHA3-512 of the empty message (0xa6).
#[cfg(all(unix, feature = "mlir-build", feature = "cross-module-imports"))]
#[test]
fn a_std_import_in_a_module_block_links_and_runs() {
    let dir = common::scratch_dir("std_import_position_build").join("module_block");
    std::fs::create_dir_all(&dir).expect("create build project dir");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"p\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n\
         output = \"p\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\n\
         sources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        dir.join("main.mind"),
        "module inner {\n    import std.keccak;\n\n    pub fn first_byte() -> i64 {\n        \
         let out: i64 = __mind_alloc(64);\n        let rc: i64 = keccak_sha3_512(out, 0, out);\n        \
         return rc + (__mind_load_i8(out) & 255);\n    }\n}\n\n\
         fn main() -> i64 {\n    return first_byte();\n}\n",
    )
    .expect("write main.mind");
    let out = mindc(&["build", "--backend", "mlir"], &dir);
    if !common::gate::compiled("std_import_position_build", &out) {
        return;
    }
    let run = Command::new(dir.join("target").join("debug").join("p"))
        .output()
        .expect("run the built executable");
    assert_eq!(
        run.status.code(),
        Some(0xa6),
        "the module block's call into std.keccak must link and compute the digest"
    );
}
