// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! VNNI declaration follows the resolved translator, including when `PATH`
//! holds a different LLVM major.
//!
//! `gemm_i8_vnni_translates_on_the_resolved_toolchain` is the compile-only
//! leg that used to live in `cross_substrate_identity`. It runs on every
//! cross-substrate cell. `gemm_i8_vnni_builds_when_path_major_differs` is the
//! mismatch leg: `MLIR_TRANSLATE` and `CLANG` (and `MLIR_OPT`, which the
//! driver also resolves) point at one major while `PATH` holds the other.
//! A build that probed `PATH` for the declaration fails that leg. The second
//! major is installed only on the avx2 LLVM 23 cell (`MIND_DUAL_LLVM=1`);
//! elsewhere the absence is an optional input.

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]

mod common;
use common::mindc_bin;

use std::path::{Path, PathBuf};
use std::process::Command;

const TARGET: &str = "vnni_resolved_toolchain";

/// The int8 GEMM entry the cross-substrate VNNI workload lowers. Only this
/// kernel emits `vpdpbusd.512`.
const VNNI_SRC: &str = "\
pub fn gemmi8(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_i8_v(a, b, c, m, k, n)
}
";

fn scratch() -> PathBuf {
    crate::common::scratch_dir(TARGET)
}

fn dual_llvm_promised() -> bool {
    std::env::var("MIND_DUAL_LLVM").ok().as_deref() == Some("1")
}

fn versioned_tool(major: u32, name: &str) -> PathBuf {
    PathBuf::from(format!("/usr/lib/llvm-{major}/bin/{name}"))
}

/// Both LLVM 20 and LLVM 23, or a counted skip. Under `MIND_BENCH_REQUIRE=1`
/// the skip is a failure only when this job promised both majors.
fn require_both_majors() -> Option<(u32, u32)> {
    let tools = [20, 23].map(|major| {
        ["mlir-opt", "mlir-translate", "clang"].map(|name| versioned_tool(major, name))
    });
    let present = tools.iter().flatten().all(|path| path.is_file());
    if !present || !cfg!(target_arch = "x86_64") {
        if dual_llvm_promised() {
            crate::common::gate::skipped(
                TARGET,
                "this job promises LLVM 20 and LLVM 23 on x86_64, but one \
                 toolchain is absent or this host is not x86_64",
            );
        } else {
            // OPTIONAL INPUT: a second installed LLVM major (20 and 23) on
            // x86_64. The avx2 LLVM 23 cross-substrate leg installs both and
            // sets MIND_DUAL_LLVM=1. Every other job keeps one major on PATH.
            crate::common::gate::skipped_optional(
                TARGET,
                "LLVM 20 and LLVM 23 are not both installed on x86_64; the \
                 PATH-versus-resolved-toolchain leg did not run",
            );
        }
        return None;
    }
    let majors_match = [20, 23].into_iter().all(|major| {
        ["mlir-translate", "clang"].into_iter().all(|name| {
            libmind::mlir::gemm_tuning::llvm_major_of_tool(
                versioned_tool(major, name)
                    .to_str()
                    .expect("utf8 tool path"),
            ) == Some(major)
        })
    });
    if !majors_match {
        if dual_llvm_promised() {
            crate::common::gate::skipped(
                TARGET,
                "this job promises LLVM 20 and LLVM 23, but a versioned tool \
                 did not report that major",
            );
        } else {
            // OPTIONAL INPUT: versioned LLVM 20 and LLVM 23 binaries whose
            // reported major matches the directory. The avx2 LLVM 23 leg
            // is the supplier.
            crate::common::gate::skipped_optional(
                TARGET,
                "a versioned LLVM tool did not report major 20 or 23; the \
                 PATH-versus-resolved-toolchain leg did not run",
            );
        }
        return None;
    }
    Some((20, 23))
}

fn path_for(major: u32) -> std::ffi::OsString {
    let mut path = std::ffi::OsString::from(format!("/usr/lib/llvm-{major}/bin"));
    path.push(":");
    path.push("/usr/bin:/bin");
    path
}

/// `env_major`'s tools via the driver overrides, `path_major` first on `PATH`.
fn emit_obj_with_split_toolchain(path_major: u32, env_major: u32, out: &Path) -> usize {
    let src = scratch().join(format!("vnni_split_{path_major}_{env_major}.mind"));
    std::fs::write(&src, VNNI_SRC).unwrap_or_else(|err| panic!("write {}: {err}", src.display()));
    let tool = |major, name| versioned_tool(major, name);
    let emitted = Command::new(mindc_bin())
        .env("PATH", path_for(path_major))
        .env("MIND_INTDOT", "vnni")
        .env("MINDC_BUILD_TOOL_TIMEOUT_SECS", "180")
        .env("MLIR_OPT", tool(env_major, "mlir-opt"))
        .env("MLIR_TRANSLATE", tool(env_major, "mlir-translate"))
        .env("CLANG", tool(env_major, "clang"))
        .args([
            src.to_str().expect("utf8 src"),
            "--emit-obj",
            out.to_str().expect("utf8 obj"),
        ])
        .output()
        .unwrap_or_else(|err| panic!("spawn mindc --emit-obj: {err}"));
    assert!(
        emitted.status.success(),
        "VNNI --emit-obj failed with PATH LLVM {path_major} and resolved LLVM {env_major}\n\
         stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&emitted.stdout),
        String::from_utf8_lossy(&emitted.stderr),
    );
    let dump = Command::new("objdump")
        .args(["-d", out.to_str().expect("utf8 obj")])
        .output()
        .unwrap_or_else(|err| panic!("spawn objdump: {err}"));
    assert!(
        dump.status.success(),
        "objdump failed for PATH LLVM {path_major} env LLVM {env_major}: {}",
        String::from_utf8_lossy(&dump.stderr),
    );
    let text = String::from_utf8_lossy(&dump.stdout);
    let zmm = text
        .lines()
        .filter(|line| line.contains("vpdpbusd") && line.contains("zmm"))
        .count();
    assert!(
        zmm > 0,
        "PATH LLVM {path_major} env LLVM {env_major}: no zmm vpdpbusd in {}",
        out.display()
    );
    println!("PATH LLVM {path_major} env LLVM {env_major}: zmm vpdpbusd = {zmm}");
    zmm
}

/// Translate the VNNI module to LLVM IR on the toolchain `resolve_tools`
/// selects. Execution needs AVX-512-VNNI silicon; translation does not, so
/// this runs on every CI toolchain, including a host that defers the
/// reproducibility gate. The emitter's text stays the LLVM 20 declaration.
/// The build driver rewrites it from the resolved `mlir-translate` before
/// that tool checks the intrinsic signature.
#[test]
fn gemm_i8_vnni_translates_on_the_resolved_toolchain() {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            crate::common::gate::skipped(
                TARGET,
                &format!(
                    "{tool} not on PATH; install the MLIR toolchain \
                     (mlir-opt / mlir-translate / clang) on this runner"
                ),
            );
            return;
        }
    }
    let src_path = scratch().join("vnni_translate.mind");
    std::fs::write(&src_path, VNNI_SRC)
        .unwrap_or_else(|err| panic!("write {}: {err}", src_path.display()));
    let emitted = Command::new(mindc_bin())
        .env("MIND_INTDOT", "vnni")
        .args([src_path.to_str().unwrap(), "--emit-mlir"])
        .output()
        .expect("spawn mindc --emit-mlir");
    assert!(
        emitted.status.success(),
        "mindc --emit-mlir failed for the VNNI translate leg\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&emitted.stdout),
        String::from_utf8_lossy(&emitted.stderr),
    );
    let mlir = String::from_utf8(emitted.stdout).expect("emit-mlir utf8");
    let emit_calls: Vec<&str> = mlir
        .lines()
        .filter(|line| line.contains("vpdpbusd.512"))
        .collect();
    assert!(
        !emit_calls.is_empty(),
        "MIND_INTDOT=vnni did not emit vpdpbusd.512"
    );
    for line in &emit_calls {
        assert!(
            line.contains("(vector<16xi32>, vector<16xi32>, vector<16xi32>)"),
            "emitter declaration is not the stable LLVM 20 form: {line}"
        );
    }

    let tools = libmind::eval::mlir_build::resolve_tools().expect("resolved build tools");
    let out = scratch().join("vnni_translate.ll");
    let opts = libmind::eval::mlir_build::BuildOptions {
        preset: libmind::eval::mlir_build::preset_for_mlir(&mlir),
        emit_mlir_file: None,
        emit_llvm_file: Some(out.as_path()),
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };
    libmind::eval::mlir_build::build_all(&mlir, &tools, &opts)
        .unwrap_or_else(|err| panic!("VNNI translate failed for {}: {err}", tools.mlir_translate));
    let ir = std::fs::read_to_string(&out).expect("read llvm ir");
    let calls: Vec<&str> = ir
        .lines()
        .filter(|line| line.contains("vpdpbusd.512"))
        .collect();
    assert!(!calls.is_empty(), "translated LLVM IR has no vpdpbusd.512");
    let major = libmind::mlir::gemm_tuning::llvm_major_of_tool(&tools.mlir_translate);
    let byte = libmind::mlir::gemm_tuning::vpdpbusd_512_byte_operands(major);
    for line in &calls {
        if byte {
            assert!(
                line.contains("<64 x i8>"),
                "LLVM {major:?} vpdpbusd.512 call is missing byte operands: {line}"
            );
        } else {
            assert!(
                line.contains("<16 x i32>") && !line.contains("<64 x i8>"),
                "LLVM {major:?} vpdpbusd.512 call is not the i32 declaration: {line}"
            );
        }
    }
}

/// Both mismatch directions must produce a real `zmm` `vpdpbusd` when LLVM 20
/// and LLVM 23 are installed. `PATH` carries one major; `MLIR_TRANSLATE` and
/// `CLANG` carry the other.
#[test]
fn gemm_i8_vnni_builds_when_path_major_differs() {
    let Some((older, newer)) = require_both_majors() else {
        return;
    };
    let older_then_newer = scratch().join("vnni_path20_env23.o");
    let newer_then_older = scratch().join("vnni_path23_env20.o");
    let zmm_newer = emit_obj_with_split_toolchain(older, newer, &older_then_newer);
    let zmm_older = emit_obj_with_split_toolchain(newer, older, &newer_then_older);
    assert!(zmm_newer > 0 && zmm_older > 0);
}
