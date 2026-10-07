// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A sibling module's `const` reaches the flat `--emit-*` lowering.
//!
//! `mindc src/c.mind --emit-mlir` (and `--emit-shared`) on a file reading
//! `a.K` from a sibling module panicked with exit 101: "undefined identifier
//! `K` reached lowering". Only the executable build seeded the cross-module
//! const table; the project scope the flat path and the cdylib path install
//! now seeds it too (`src/project/single_file_scope.rs`).

#![cfg(all(
    feature = "cross-module-imports",
    any(feature = "mlir-lowering", feature = "mlir-build")
))]

mod common;

use std::process::Command;

#[test]
fn emit_mlir_inlines_a_sibling_const() {
    let root = common::scratch_dir("flat_emit_sibling_const");
    std::fs::create_dir_all(root.join("src")).expect("mkdir src");
    std::fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"iv\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        root.join("src/a.mind"),
        "pub const K: i64 = 5;\n\npub fn inc(x: i64) -> i64 {\n    return x + K;\n}\n",
    )
    .expect("write a.mind");
    std::fs::write(
        root.join("src/main.mind"),
        "fn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("write main.mind");
    std::fs::write(
        root.join("src/c.mind"),
        "import a;\n\npub fn use_k() -> i64 {\n    return a.K + a.inc(1);\n}\n",
    )
    .expect("write c.mind");
    let out = Command::new(common::mindc_bin())
        .args(["src/c.mind", "--emit-mlir"])
        .current_dir(&root)
        .output()
        .expect("run mindc --emit-mlir");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !common::gate::compiled("flat_emit_sibling_const", &out) {
        return;
    }
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stdout.contains("func.func @use_k"), "{stdout}{stderr}");
    // `a.K` inlined as the constant 5.
    assert!(stdout.contains("arith.constant 5"), "{stdout}");
}
