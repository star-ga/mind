// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");

//! `[build] backend` route selection, split out of the parent suite verbatim.

use super::*;

// ---------------------------------------------------------------------------
// `[build] backend`: the manifest and `--backend` select ONE route.
// ---------------------------------------------------------------------------

/// Run `mindc <args>` in a fresh project whose manifest ends with `extra`
/// (appended to `[build]`). Returns (exit code, stderr, project dir).
fn backend_case(extra: &str, args: &[&str]) -> (Option<i32>, String, tempfile::TempDir) {
    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/main.mind"), HELLO_MIND).unwrap();
    let manifest = minimal_manifest("be", "src/main.mind") + extra + "\n";
    fs::write(td.path().join("Mind.toml"), manifest).unwrap();
    let out = Command::new(require_mindc())
        .args(args)
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status.code(), stderr, td)
}

/// Whether an ELF64 image has a `PT_INTERP` program header (a dynamic
/// executable, which the MLIR route links through clang).
fn has_pt_interp(elf: &[u8]) -> bool {
    let word = |at: usize, n: usize| {
        elf[at..at + n]
            .iter()
            .rev()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b))
    };
    let (phoff, phentsize, phnum) = (word(32, 8), word(54, 2), word(56, 2));
    (0..phnum).any(|k| word(phoff + k * phentsize, 4) == 3)
}

/// Positive control: `backend = "mlir"` builds the MLIR artifact, identical to
/// the explicit `--backend mlir` build, and it runs.
#[test]
fn manifest_backend_mlir_builds_the_mlir_artifact() {
    if !mlir_available() {
        crate::common::gate::skipped("mindc_build_phase_a", "MLIR tools not available");
        return;
    }
    let (code, stderr, td) = backend_case("backend = \"mlir\"", &["build", "--out", "m.elf"]);
    assert_eq!(code, Some(0), "manifest mlir build failed:\n{stderr}");
    let elf = fs::read(td.path().join("m.elf")).expect("artifact");
    assert!(
        has_pt_interp(&elf),
        "manifest mlir artifact is not the MLIR route's ELF"
    );
    let explicit = Command::new(require_mindc())
        .args(["build", "--backend", "mlir", "--no-cache", "--out", "x.elf"])
        .current_dir(td.path())
        .status()
        .expect("spawn mindc");
    assert!(explicit.success());
    assert_eq!(elf, fs::read(td.path().join("x.elf")).unwrap());
    let ran = Command::new(td.path().join("m.elf")).status().expect("run");
    assert_eq!(ran.code(), Some(42), "manifest mlir artifact exit");
}

/// An unknown `[build] backend` is refused at manifest load (E2043) by every
/// command, never read as "unset".
#[test]
fn manifest_backend_unknown_value_is_refused() {
    for args in [&["build"][..], &["run"][..]] {
        let (code, stderr, td) = backend_case("backend = \"llvm\"", args);
        assert_ne!(code, Some(0), "{args:?} accepted backend = llvm");
        assert!(stderr.contains("E2043"), "{args:?}: no E2043:\n{stderr}");
        assert!(!td.path().join("target").exists(), "{args:?} built");
    }
}

/// A flag that disagrees with the manifest is refused (E2044), in either order.
#[test]
fn backend_flag_disagreeing_with_manifest_is_refused() {
    for (decl, flag) in [("native", "mlir"), ("mlir", "native")] {
        let extra = format!("backend = \"{decl}\"");
        let (code, stderr, _td) = backend_case(&extra, &["build", "--backend", flag]);
        assert_eq!(code, Some(2), "{decl}/{flag}:\n{stderr}");
        assert!(stderr.contains("E2044"), "{decl}/{flag}:\n{stderr}");
    }
}

/// Neither route builds what it cannot honor: `run` refuses a native manifest by
/// name, and the MLIR pipeline refuses a code-generation backend written in the
/// target-class key (E2043), which used to fall through to `cpu` and build MLIR.
/// Mutant: drop `admit_mlir_route`'s checks and the second row builds.
#[test]
fn mlir_project_pipeline_refuses_a_native_declaration() {
    let mlir = &["build", "--backend=mlir"][..];
    let rows = [
        ("backend = \"native\"", &["run"][..], "(backend.native_run)"),
        ("\n[targets.cpu]\nbackend = \"native\"", mlir, "E2043"),
    ];
    for (extra, args, key) in rows {
        let (code, stderr, _td) = backend_case(extra, args);
        assert_ne!(code, Some(0), "{extra:?} {args:?} was built");
        assert!(stderr.contains(key), "{extra:?}:\n{stderr}");
    }
}
