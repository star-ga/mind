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
//!
//! A tensor program built with an `mlir-opt` older than the bufferization it
//! needs failed inside `mlir-opt` ("cannot bufferize a FuncOp with tensors and
//! without a unique ReturnOp"). It is now refused before `mlir-opt` runs, with
//! the version found, the version required and the path of the tool in use.

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

/// A stub `mlir-opt` that reports LLVM 18 and fails anything else, so the
/// check is exercised identically on every host whatever LLVM it has.
#[test]
fn an_old_mlir_opt_is_refused_by_version_before_bufferizing() {
    let dir = common::scratch_dir("mlir_tool_too_old");
    let stub = dir.join("mlir-opt");
    std::fs::write(
        &stub,
        concat!(
            "#!",
            "/bin/sh\n",
            "if [ \"$1\" = --version ]; then echo 'Ubuntu LLVM version 18.1.3'; exit 0; fi\n",
            "echo 'stub mlir-opt must not be run past the version check' >&2\nexit 3\n"
        ),
    )
    .expect("write stub");
    let mut perm = std::fs::metadata(&stub)
        .expect("stub metadata")
        .permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
    std::fs::set_permissions(&stub, perm).expect("chmod stub");
    let src = dir.join("t.mind");
    std::fs::write(
        &src,
        "pub fn a(i: i64) -> i64 {\n    let t: tensor<f64[3]> = [1.5, 2.5, 3.5];\n    \
         return __mind_f64_to_bits(t[i]);\n}\n",
    )
    .expect("write source");
    let so = dir.join("t.so");
    // Retried when the version probe could not exec the stub (ETXTBSY: another
    // test thread can briefly hold its write descriptor across a fork); five
    // misses in a row still fail the test.
    let text = (0..5)
        .map(|_| {
            let out = Command::new(common::mindc_bin())
                .arg(&src)
                .arg("--emit-shared")
                .arg(&so)
                .env("MLIR_OPT", &stub)
                .output()
                .expect("run mindc");
            assert!(!out.status.success(), "built with an LLVM 18 mlir-opt");
            String::from_utf8_lossy(&out.stderr).into_owned()
        })
        .find(|t| !t.contains("does not state"))
        .expect("stub never became executable");
    assert!(
        text.contains("requires MLIR/LLVM 20 or newer"),
        "the refusal must name the required version:\n{text}"
    );
    assert!(
        text.contains(&format!("({}) is LLVM 18", stub.display())),
        "the refusal must name the tool in use and its version:\n{text}"
    );
    assert!(
        text.contains("error[build]: [E5004] incompatible mlir-opt"),
        "the refusal must carry the toolchain cause code:\n{text}"
    );
    assert!(
        !text.contains("must not be run past the version check"),
        "mlir-opt ran before the version was checked:\n{text}"
    );
}
