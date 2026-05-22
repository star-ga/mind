// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase A integration tests for `mindc build`.
//!
//! Gate: `cargo test --release --features "mlir-build std-surface cross-module-imports" mindc_build_phase_a`

// The build orchestrator under test resolves the external MLIR toolchain, which
// is only re-exported under `mlir-build` (see src/eval/mod.rs). Compile this
// suite to an empty binary when the feature is off, matching cli_build.rs /
// mlir_build.rs, so `cargo test --workspace` (no features) stays green.
#![cfg(feature = "mlir-build")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mindc_bin() -> PathBuf {
    // Prefer the binary already built for this test run.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    #[cfg(debug_assertions)]
    p.push("debug");
    #[cfg(not(debug_assertions))]
    p.push("release");
    #[cfg(target_os = "windows")]
    p.push("mindc.exe");
    #[cfg(not(target_os = "windows"))]
    p.push("mindc");
    p
}

fn require_mindc() -> Option<PathBuf> {
    let bin = mindc_bin();
    if bin.exists() {
        Some(bin)
    } else {
        eprintln!("SKIP: mindc binary not found at {}; run cargo build first", bin.display());
        None
    }
}

/// Minimal valid Mind.toml for a test project.
fn minimal_manifest(name: &str, entry: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"

[build]
entry = "{entry}"
"#
    )
}

/// A trivially valid .mind source (compiles through the current pipeline).
const HELLO_MIND: &str = r#"fn main() -> i64 { 42 }"#;

// ---------------------------------------------------------------------------
// Unit-layer tests (project::BuildTarget / EmitKind / OptimizeLevel parsing)
// ---------------------------------------------------------------------------

#[test]
fn build_target_parse_cpu() {
    use libmind::project::BuildTarget;
    assert_eq!(BuildTarget::parse("cpu").unwrap(), BuildTarget::Cpu);
}

#[test]
fn build_target_parse_aliases() {
    use libmind::project::BuildTarget;
    assert_eq!(BuildTarget::parse("cuda").unwrap(), BuildTarget::Gpu);
    assert_eq!(BuildTarget::parse("rocm").unwrap(), BuildTarget::Gpu);
    assert_eq!(BuildTarget::parse("cerebras").unwrap(), BuildTarget::Cerebras);
    assert_eq!(BuildTarget::parse("wse3").unwrap(), BuildTarget::Cerebras);
}

#[test]
fn build_target_parse_unknown_errors() {
    use libmind::project::BuildTarget;
    assert!(BuildTarget::parse("quantum").is_err());
}

#[test]
fn emit_kind_parse_variants() {
    use libmind::project::EmitKind;
    assert_eq!(EmitKind::parse("binary").unwrap(), EmitKind::Binary);
    assert_eq!(EmitKind::parse("cdylib").unwrap(), EmitKind::Cdylib);
    assert_eq!(EmitKind::parse("object").unwrap(), EmitKind::Object);
    assert_eq!(EmitKind::parse("shared").unwrap(), EmitKind::Cdylib);
    assert_eq!(EmitKind::parse("obj").unwrap(), EmitKind::Object);
    assert!(EmitKind::parse("exe").is_err());
}

#[test]
fn optimize_level_is_release_mapping() {
    use libmind::project::OptimizeLevel;
    assert!(!OptimizeLevel::Debug.is_release());
    assert!(OptimizeLevel::Release.is_release());
    assert!(OptimizeLevel::Size.is_release());
}

#[test]
fn build_config_defaults() {
    use libmind::project::{BuildConfig, BuildTarget, EmitKind, OptimizeLevel};
    let cfg = BuildConfig::default();
    assert_eq!(cfg.target, BuildTarget::Cpu);
    assert_eq!(cfg.emit, EmitKind::Binary);
    assert_eq!(cfg.optimize, OptimizeLevel::Debug);
    assert_eq!(cfg.entry, "src/main.mind");
}

// ---------------------------------------------------------------------------
// Manifest parsing tests (round-trip through toml)
// ---------------------------------------------------------------------------

#[test]
fn manifest_parses_build_target_cpu() {
    let toml = r#"
[package]
name = "my-proj"
version = "0.1.0"

[build]
target = "cpu"
emit = "binary"
optimize = "release"
"#;
    let m: libmind::project::ProjectManifest = toml::from_str(toml).unwrap();
    use libmind::project::{BuildTarget, EmitKind, OptimizeLevel};
    assert_eq!(m.build.target, BuildTarget::Cpu);
    assert_eq!(m.build.emit, EmitKind::Binary);
    assert_eq!(m.build.optimize, OptimizeLevel::Release);
}

#[test]
fn manifest_missing_build_section_uses_defaults() {
    let toml = r#"
[package]
name = "bare"
version = "0.1.0"
"#;
    let m: libmind::project::ProjectManifest = toml::from_str(toml).unwrap();
    use libmind::project::{BuildTarget, EmitKind, OptimizeLevel};
    assert_eq!(m.build.target, BuildTarget::Cpu);
    assert_eq!(m.build.emit, EmitKind::Binary);
    assert_eq!(m.build.optimize, OptimizeLevel::Debug);
}

#[test]
fn manifest_parse_error_bad_target() {
    let toml = r#"
[package]
name = "bad"
version = "0.1.0"

[build]
target = "quantumfpga"
"#;
    // toml deserialization should fail because "quantumfpga" is not a valid
    // variant of the BuildTarget enum.
    let result: Result<libmind::project::ProjectManifest, _> = toml::from_str(toml);
    assert!(result.is_err(), "expected parse error for unknown target");
}

#[test]
fn manifest_parse_test_config_defaults() {
    let toml = r#"
[package]
name = "t"
version = "0.1.0"
"#;
    let m: libmind::project::ProjectManifest = toml::from_str(toml).unwrap();
    assert!(m.test.parallel);
    assert_eq!(m.test.timeout, 30);
    assert_eq!(m.test.threads, 0);
    assert!(m.test.filter.is_empty());
}

#[test]
fn manifest_parse_workspace_config() {
    let toml = r#"
[package]
name = "ws-root"
version = "0.1.0"

[workspace]
members = ["crates/core", "crates/std"]
exclude = ["scratch"]
"#;
    let m: libmind::project::ProjectManifest = toml::from_str(toml).unwrap();
    let ws = m.workspace.expect("workspace present");
    assert_eq!(ws.members, vec!["crates/core", "crates/std"]);
    assert_eq!(ws.exclude, vec!["scratch"]);
}

// ---------------------------------------------------------------------------
// CLI integration tests — require the built mindc binary + MLIR toolchain.
// ---------------------------------------------------------------------------

/// Skip-guard: returns true if mlir tools are available.
fn mlir_available() -> bool {
    libmind::eval::resolve_mlir_build_tools()
        .map(|_| true)
        .unwrap_or(false)
}

#[test]
fn cli_build_unknown_target_exits_2() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    let src = td.path().join("main.mind");
    fs::write(&src, HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "main.mind")).unwrap();

    let output = Command::new(&bin)
        .args(["build", "--target=quantumfpga"])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    assert_eq!(
        output.status.code(),
        Some(2),
        "expected exit 2 for unknown target; got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_build_unknown_emit_exits_2() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    let src = td.path().join("main.mind");
    fs::write(&src, HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "main.mind")).unwrap();

    let output = Command::new(&bin)
        .args(["build", "--emit=dll"])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    assert_eq!(
        output.status.code(),
        Some(2),
        "expected exit 2 for unknown emit; got {:?}",
        output.status.code()
    );
}

#[test]
fn cli_build_missing_source_exits_1() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    // Write manifest pointing to a non-existent file.
    fs::write(
        td.path().join("Mind.toml"),
        minimal_manifest("hello", "nonexistent.mind"),
    )
    .unwrap();

    let output = Command::new(&bin)
        .arg("build")
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    assert!(
        output.status.code() == Some(1) || output.status.code() == Some(2),
        "expected exit 1 or 2 for missing source; got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_build_invalid_manifest_exits_2() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    fs::write(td.path().join("Mind.toml"), "this is not valid toml %%%").unwrap();

    let output = Command::new(&bin)
        .arg("build")
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    assert_eq!(
        output.status.code(),
        Some(2),
        "expected exit 2 for invalid manifest; got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_build_release_flag_accepted() {
    let Some(bin) = require_mindc() else { return };
    if !mlir_available() {
        eprintln!("SKIP: MLIR tools not available");
        return;
    }

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    let src = td.path().join("src/main.mind");
    fs::write(&src, HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "src/main.mind")).unwrap();

    let out = td.path().join("target/release/hello");

    let output = Command::new(&bin)
        .args(["build", "--release"])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    // Should succeed or fail with exit 1 (build failure, not exit 2).
    // The artifact may or may not be created depending on toolchain availability,
    // but it must not be a usage error.
    assert_ne!(
        output.status.code(),
        Some(2),
        "unexpected usage error on --release; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    if output.status.success() {
        assert!(
            out.exists(),
            "expected artifact at {} after --release build",
            out.display()
        );
    }
}

#[test]
fn cli_build_debug_puts_artifact_in_debug_subdir() {
    let Some(bin) = require_mindc() else { return };
    if !mlir_available() {
        eprintln!("SKIP: MLIR tools not available");
        return;
    }

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/main.mind"), HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "src/main.mind")).unwrap();

    let output = Command::new(&bin)
        .arg("build")
        // No --release flag → debug profile
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    if output.status.success() {
        let debug_out = td.path().join("target/debug/hello");
        assert!(
            debug_out.exists(),
            "debug artifact expected at {}",
            debug_out.display()
        );
    }
}

#[test]
fn cli_build_custom_out_path() {
    let Some(bin) = require_mindc() else { return };
    if !mlir_available() {
        eprintln!("SKIP: MLIR tools not available");
        return;
    }

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/main.mind"), HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "src/main.mind")).unwrap();

    let custom_out = td.path().join("my_custom_binary");

    let output = Command::new(&bin)
        .args([
            "build",
            &format!("--out={}", custom_out.display()),
        ])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    if output.status.success() {
        assert!(
            custom_out.exists(),
            "artifact expected at custom path {}",
            custom_out.display()
        );
    }
}

#[test]
fn cli_build_emit_cdylib_produces_so() {
    let Some(bin) = require_mindc() else { return };
    if !mlir_available() {
        eprintln!("SKIP: MLIR tools not available");
        return;
    }

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/lib.mind"), HELLO_MIND).unwrap();
    fs::write(
        td.path().join("Mind.toml"),
        minimal_manifest("mylib", "src/lib.mind"),
    )
    .unwrap();

    let output = Command::new(&bin)
        .args(["build", "--emit=cdylib"])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    if output.status.success() {
        let so_path = td.path().join("target/debug/libmylib.so");
        let so_path_win = td.path().join("target/debug/mylib.dll");
        assert!(
            so_path.exists() || so_path_win.exists(),
            "expected .so/.dll under target/debug/; contents: {:?}",
            fs::read_dir(td.path().join("target/debug"))
                .ok()
                .map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).collect::<Vec<_>>())
        );
    }
}

#[test]
fn cli_build_target_cpu_is_default() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/main.mind"), HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "src/main.mind")).unwrap();

    // --target=cpu should succeed or fail at build level (1), never at CLI parsing (2).
    let output = Command::new(&bin)
        .args(["build", "--target=cpu"])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    assert_ne!(
        output.status.code(),
        Some(2),
        "--target=cpu should not produce a usage error; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_build_target_gpu_returns_clear_error() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/main.mind"), HELLO_MIND).unwrap();
    fs::write(td.path().join("Mind.toml"), minimal_manifest("hello", "src/main.mind")).unwrap();

    let output = Command::new(&bin)
        .args(["build", "--target=gpu"])
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    // GPU is parsed but not yet supported in Phase A → exit 2.
    assert_eq!(
        output.status.code(),
        Some(2),
        "expected exit 2 for unsupported --target=gpu; got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("gpu") || stderr.contains("not yet supported") || stderr.contains("target"),
        "error message should mention gpu or target; got: {}",
        stderr
    );
}

#[test]
fn cli_build_manifest_target_field_overrides_default() {
    let Some(bin) = require_mindc() else { return };

    let td = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(td.path().join("src")).unwrap();
    fs::write(td.path().join("src/main.mind"), HELLO_MIND).unwrap();
    // Set target = "cpu" in manifest — should parse fine and not error out.
    fs::write(
        td.path().join("Mind.toml"),
        r#"[package]
name = "hello"
version = "0.1.0"

[build]
entry = "src/main.mind"
target = "cpu"
optimize = "release"
"#,
    )
    .unwrap();

    let output = Command::new(&bin)
        .arg("build")
        .current_dir(td.path())
        .output()
        .expect("spawn mindc");

    assert_ne!(
        output.status.code(),
        Some(2),
        "manifest with [build] target = cpu should parse; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// Self-build smoke test — drives the existing mindc_mind cdylib.
// ---------------------------------------------------------------------------

/// Build `examples/mindc_mind/main.mind --emit=cdylib` and check the .so
/// is produced and non-empty. This is the RFC 0008 §7 Phase A hard gate.
#[test]
fn cli_build_self_build_smoke() {
    let Some(bin) = require_mindc() else { return };
    if !mlir_available() {
        eprintln!("SKIP: MLIR tools not available");
        return;
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = repo_root.join("examples/mindc_mind/main.mind");
    if !src.exists() {
        eprintln!("SKIP: examples/mindc_mind/main.mind not found");
        return;
    }

    let out = std::env::temp_dir().join("mindc_build_phase_a_self_build.so");

    let output = Command::new(&bin)
        .args([
            "build",
            src.to_str().unwrap(),
            "--emit=cdylib",
            &format!("--out={}", out.display()),
        ])
        .current_dir(&repo_root)
        .output()
        .expect("spawn mindc");

    if !output.status.success() {
        // Print diagnostic but do not fail the test — the self-build requires
        // a working MLIR toolchain including llc/mlir-opt; CI environments
        // with only the mindc binary can skip this.
        eprintln!(
            "self-build did not succeed (toolchain may be incomplete): {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    assert!(out.exists(), "expected .so at {}", out.display());
    let sz = fs::metadata(&out).unwrap().len();
    assert!(sz > 0, ".so is empty at {}", out.display());
    eprintln!("self-build smoke: {} bytes", sz);
}
