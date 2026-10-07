// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Two modules may each define a non-`pub` fn of the same name.
//!
//! A multi-module executable compiles each source to its own object, and every
//! fn was a global symbol, so `a.mind` and `b.mind` each defining
//! `fn helper` failed to link with "multiple definition of `helper`". The
//! build now gives such colliding module-private fns internal linkage
//! (`src/project/private_linkage.rs`) when no other module refers to them.
//!
//! The control keeps the other half of the contract: a non-`pub` fn that
//! another module calls stays a global symbol, because `check` accepts that
//! call today and internal linkage would leave it undefined.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::path::Path;
use std::process::Command;

const A: &str = "fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n\npub fn fa(x: i64) -> i64 {\n    return helper(x);\n}\n";
const B: &str = "fn helper(x: i64) -> i64 {\n    return x * 10;\n}\n\npub fn fb(x: i64) -> i64 {\n    return helper(x);\n}\n";

/// Write a project whose sources are `files`, build it, and run the binary.
/// `None` when the build was routed through the capability-skip gate.
fn build_and_run(dir: &Path, files: &[(&str, &str)]) -> Option<i32> {
    let sources: Vec<String> = files.iter().map(|(n, _)| format!("\"{n}\"")).collect();
    std::fs::write(
        dir.join("Mind.toml"),
        format!(
            "[package]\nname = \"prv\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"prv\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [{}]\n",
            sources.join(", ")
        ),
    )
    .expect("write Mind.toml");
    for (name, text) in files {
        std::fs::write(dir.join(name), text).expect("write source");
    }
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(dir)
        .output()
        .expect("run mindc build");
    if !common::gate::compiled("private_fn_link_collision_run", &out) {
        return None;
    }
    let run = Command::new(dir.join("target").join("debug").join("prv"))
        .output()
        .expect("run the built executable");
    Some(run.status.code().expect("exit code"))
}

#[test]
fn same_private_fn_name_in_two_modules_links_and_each_module_calls_its_own() {
    let dir = common::scratch_dir("private_fn_link_collision");
    let main = "import a;\nimport b;\n\nfn main() -> i64 {\n    return a.fa(1) + b.fb(2);\n}\n";
    let Some(code) = build_and_run(&dir, &[("main.mind", main), ("a.mind", A), ("b.mind", B)])
    else {
        return;
    };
    // a.helper(1) = 2, b.helper(2) = 20.
    assert_eq!(code, 22, "each module must call its own `helper`");
}

#[test]
fn test_fns_of_the_same_name_in_two_modules_link() {
    // `#[test]` fns are run by `mindc test`, never by symbol, so a name two
    // modules share must not fail the executable's link.
    let dir = common::scratch_dir("private_fn_link_test_fns");
    let main = "import a;\nimport b;\n\nfn main() -> i64 {\n    return a.fa(1) + b.fb(2);\n}\n";
    let a = format!("{A}\n#[test]\nfn t_one() {{\n    assert fa(1) == 2;\n}}\n");
    let b = format!("{B}\n#[test]\nfn t_one() {{\n    assert fb(2) == 20;\n}}\n");
    let Some(code) = build_and_run(&dir, &[("main.mind", main), ("a.mind", &a), ("b.mind", &b)])
    else {
        return;
    };
    assert_eq!(code, 22);
}

#[test]
fn a_private_fn_called_from_another_module_still_links() {
    let dir = common::scratch_dir("private_fn_link_cross_call");
    let main = "import a;\n\nfn main() -> i64 {\n    return a.helper(4);\n}\n";
    let Some(code) = build_and_run(&dir, &[("main.mind", main), ("a.mind", A)]) else {
        return;
    };
    assert_eq!(code, 5);
}
