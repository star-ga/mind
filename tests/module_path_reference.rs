// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `m::f(x)` and `m::K` on an imported module are `m.f(x)` and `m.K`.
//!
//! The `::` spelling reached MLIR as the symbol `@helper::take_f64`, which
//! `mlir-opt` rejects ("expected '('"), with its `f64` parameter typed `i64`;
//! the dot spelling built and ran. The parser now desugars both alike
//! (`src/parser/paths.rs`), and the formatter keeps the `::`.

mod common;

use std::process::Command;

const HELPER: &str =
    "pub const K: i64 = 10;\n\npub fn take_f64(x: f64) -> f64 {\n    return x + 1.0;\n}\n";
const MAIN: &str = "import helper;\n\nfn main() -> i64 {\n    return helper::take_f64(3.5) as i64 + helper::K;\n}\n";

fn project(target: &str) -> std::path::PathBuf {
    let dir = common::scratch_dir(target);
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"mp\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"mp\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"helper.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(dir.join("helper.mind"), HELPER).expect("write helper.mind");
    std::fs::write(dir.join("main.mind"), MAIN).expect("write main.mind");
    dir
}

#[test]
fn the_formatter_keeps_the_path_spelling() {
    let dir = project("module_path_fmt");
    let out = Command::new(common::mindc_bin())
        .args(["fmt", "--check", "main.mind"])
        .current_dir(&dir)
        .output()
        .expect("run mindc fmt --check");
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#[test]
fn a_path_call_into_an_imported_module_builds_and_runs() {
    let dir = project("module_path_run");
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .output()
        .expect("run mindc build");
    if !common::gate::compiled("module_path_run", &out) {
        return;
    }
    let run = Command::new(dir.join("target").join("debug").join("mp"))
        .output()
        .expect("run the built executable");
    // (3.5 + 1.0) as i64 + 10
    assert_eq!(run.status.code(), Some(14));
}
