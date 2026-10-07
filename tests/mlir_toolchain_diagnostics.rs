// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A build without `mlir-opt` on `PATH` names the missing tool.
//!
//! It failed with only "public CPU executable requires every source module to
//! compile natively; refusing to link a runtime-JIT fallback object", although
//! the cause was a missing tool and the tool was often installed under
//! `/usr/lib/llvm-<N>/bin`. The refusal now names the tool and where an
//! installed copy is (`src/diagnostics/toolchain.rs`).

#![cfg(all(unix, feature = "mlir-build"))]

mod common;

use std::process::Command;

#[test]
fn a_missing_mlir_opt_is_named_in_the_refusal() {
    let dir = common::scratch_dir("mlir_tool_missing");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"tm\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"tm\"\ntarget = \"cpu\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\"]\n",
    )
    .expect("write Mind.toml");
    std::fs::write(
        dir.join("main.mind"),
        "fn main() -> i64 {\n    return 3;\n}\n",
    )
    .expect("write main.mind");
    // The system bin directories only: LLVM's own bin directories hold mlir-opt.
    let out = Command::new(common::mindc_bin())
        .arg("build")
        .current_dir(&dir)
        .env("PATH", "/usr/bin:/bin")
        .env_remove("MLIR_OPT")
        .env_remove("MLIR_TRANSLATE")
        .env_remove("CLANG")
        .output()
        .expect("run mindc build");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if which_in("/usr/bin:/bin", "mlir-opt") {
        // A host with an unversioned mlir-opt in /usr/bin builds normally.
        return;
    }
    assert!(!out.status.success(), "{text}");
    assert!(text.contains("`mlir-opt` is not on PATH"), "{text}");
}

fn which_in(path: &str, tool: &str) -> bool {
    path.split(':')
        .any(|dir| std::path::Path::new(dir).join(tool).is_file())
}
