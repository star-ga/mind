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
//! A tensor program built with an `mlir-opt` older than LLVM 20 is NOT refused
//! (MLIR is an optional backend; many tensor programs build fine on LLVM 18).
//! It draws a one-line warning; and when `mlir-opt` then fails — it used to
//! fail opaquely ("cannot bufferize a FuncOp with tensors and without a unique
//! ReturnOp") — the error carries the E5004 explanation: the version found,
//! the version needed and where a usable one is installed.

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

/// A shell stub named `mlir-opt` in its own scratch dir that reports LLVM 18
/// for `--version` and runs `body` for everything else.
fn llvm18_stub(target: &str, body: &str) -> std::path::PathBuf {
    let dir = common::scratch_dir(target);
    let stub = dir.join("mlir-opt");
    let script = format!(
        "{}\nif [ \"$1\" = --version ]; then echo 'Ubuntu LLVM version 18.1.3'; exit 0; fi\n{body}\n",
        concat!("#!", "/bin/sh")
    );
    std::fs::write(&stub, script).expect("write stub");
    let mut perm = std::fs::metadata(&stub)
        .expect("stub metadata")
        .permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
    std::fs::set_permissions(&stub, perm).expect("chmod stub");
    stub
}

const TENSOR_SRC: &str = "pub fn a(i: i64) -> i64 {\n    let t: tensor<f64[3]> = [1.5, 2.5, 3.5];\n    \
     return __mind_f64_to_bits(t[i]);\n}\n";

/// `mindc <src> --emit-shared <so>` with `MLIR_OPT=stub`. Retried when the
/// version probe could not exec the stub (ETXTBSY: another test thread can
/// briefly hold its write descriptor across a fork); five misses still fail.
fn build_with(stub: &std::path::Path, target: &str) -> (bool, String, std::path::PathBuf) {
    let dir = common::scratch_dir(target);
    let src = dir.join("t.mind");
    std::fs::write(&src, TENSOR_SRC).expect("write source");
    let so = dir.join("t.so");
    for _ in 0..5 {
        let _ = std::fs::remove_file(&so);
        let out = Command::new(common::mindc_bin())
            .arg(&src)
            .arg("--emit-shared")
            .arg(&so)
            .env("MLIR_OPT", stub)
            .output()
            .expect("run mindc");
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        if !text.contains("does not state") {
            return (out.status.success(), text, so);
        }
    }
    panic!("the version probe never ran the stub at {}", stub.display());
}

/// An LLVM-18 `mlir-opt` that WORKS (the stub forwards to the real tool) must
/// build the tensor program, print the warning once, and record LLVM 18.
#[test]
fn an_old_mlir_opt_that_works_builds_with_one_warning() {
    let tools = match libmind::eval::resolve_mlir_build_tools() {
        Ok(tools) => tools,
        Err(err) => {
            common::gate::skipped("mlir_toolchain_diagnostics", &format!("{err}"));
            return;
        }
    };
    let real = which::which(&tools.mlir_opt).expect("resolve the real mlir-opt");
    let stub = llvm18_stub(
        "mlir_tool_old_ok",
        &format!("exec '{}' \"$@\"", real.display()),
    );
    let (ok, text, so) = build_with(&stub, "mlir_tool_old_ok");
    assert!(
        ok,
        "an LLVM 18 mlir-opt that works must not be refused:\n{text}"
    );
    assert!(
        so.metadata().map(|m| m.len() > 4096).unwrap_or(false),
        "no real library"
    );
    let warning = format!("warning[build]: mlir-opt {} is LLVM 18;", stub.display());
    assert_eq!(
        text.matches(&warning).count(),
        1,
        "exactly one old-toolchain warning expected:\n{text}"
    );
    assert!(
        text.contains("building anyway"),
        "the warning must say the build proceeds:\n{text}"
    );
    assert!(
        text.contains(&format!("Toolchain: mlir-opt {} (LLVM 18)", stub.display())),
        "the toolchain line must record the LLVM 18 mlir-opt:\n{text}"
    );
    assert!(
        !text.contains("[E5004]"),
        "no failure, so no E5004 note:\n{text}"
    );
}

/// An LLVM-18 `mlir-opt` that FAILS must leave its own error in place and
/// append the E5004 explanation.
#[test]
fn an_old_mlir_opt_that_fails_carries_the_e5004_note() {
    let stub = llvm18_stub(
        "mlir_tool_old_fail",
        "echo 'error: cannot bufferize a FuncOp with tensors and without a unique ReturnOp' >&2\nexit 1",
    );
    let (ok, text, _) = build_with(&stub, "mlir_tool_old_fail");
    assert!(!ok, "the failing stub cannot have built:\n{text}");
    assert!(
        text.contains("cannot bufferize a FuncOp with tensors and without a unique ReturnOp"),
        "mlir-opt's own error must stay:\n{text}"
    );
    assert!(
        text.contains(
            "note: [E5004] the tensor pipeline (one-shot-bufferize) needs MLIR/LLVM 20 or newer"
        ),
        "the failure must carry the E5004 explanation:\n{text}"
    );
    assert!(
        text.contains(&format!("({}) is LLVM 18", stub.display())),
        "the explanation must name the tool in use and its version:\n{text}"
    );
}
