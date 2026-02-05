#![cfg(feature = "mlir-build")]

// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::fs;
use std::time::Duration;

use libmind::eval;

use libmind::parser;

use tempfile::tempdir;

fn parse_and_lower(src: &str) -> (String, eval::MlirLowerPreset) {
    let module = parser::parse_with_diagnostics(src).expect("parse module");
    let ir = eval::lower_to_ir(&module);
    let plain_mlir = eval::emit_mlir_string(&ir, eval::MlirLowerPreset::None);
    (plain_mlir, eval::MlirLowerPreset::None)
}

fn resolve_or_skip() -> Option<eval::MlirBuildTools> {
    match eval::resolve_mlir_build_tools() {
        Ok(tools) => Some(tools),
        Err(err) => {
            eprintln!("Skipping MLIR build test: {err}");
            None
        }
    }
}

#[test]
fn build_emits_mlir_and_llvm() {
    let Some(tools) = resolve_or_skip() else {
        return;
    };
    let (mlir_src, preset) = parse_and_lower("let x = 1; x + 2");
    let dir = tempdir().expect("tempdir");
    let mlir_path = dir.path().join("out.mlir");
    let llvm_path = dir.path().join("out.ll");

    let opts = eval::MlirBuildOptions {
        preset: preset.as_str(),
        emit_mlir_file: Some(mlir_path.as_path()),
        emit_llvm_file: Some(llvm_path.as_path()),
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };

    let products =
        eval::build_mlir_artifacts(&mlir_src, &tools, &opts).expect("mlir build succeeds");

    assert!(mlir_path.exists(), "mlir file missing");
    assert!(llvm_path.exists(), "llvm file missing");

    let mlir_contents = fs::read_to_string(&mlir_path).expect("read mlir file");
    assert!(mlir_contents.contains("module"));

    assert!(products.optimized_mlir.contains("module"));

    let llvm_contents = fs::read_to_string(&llvm_path).expect("read llvm file");
    assert!(llvm_contents.contains("define"));
    assert!(products.llvm_ir.contains("define"));
}

#[test]
fn build_emits_object_file() {
    let Some(tools) = resolve_or_skip() else {
        return;
    };
    let (mlir_src, preset) = parse_and_lower("let x = 1; x * 3");
    let dir = tempdir().expect("tempdir");
    let obj_path = dir.path().join("out.o");

    let opts = eval::MlirBuildOptions {
        preset: preset.as_str(),
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: Some(obj_path.as_path()),
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };

    let _ = eval::build_mlir_artifacts(&mlir_src, &tools, &opts).expect("object build succeeds");
    let metadata = fs::metadata(&obj_path).expect("object metadata");
    assert!(metadata.len() > 0, "object file is empty");
}

#[test]
fn build_emits_shared_library() {
    let Some(tools) = resolve_or_skip() else {
        return;
    };
    let (mlir_src, preset) = parse_and_lower("let x = 1; x");
    let dir = tempdir().expect("tempdir");
    let lib_name = format!("test_artifact{}", std::env::consts::DLL_SUFFIX);
    let lib_path = dir.path().join(lib_name);

    let opts = eval::MlirBuildOptions {
        preset: preset.as_str(),
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: Some(lib_path.as_path()),
        opt_pipeline: None,
        target_triple: None,
    };

    let _ = eval::build_mlir_artifacts(&mlir_src, &tools, &opts).expect("shared build succeeds");
    let metadata = fs::metadata(&lib_path).expect("shared metadata");
    assert!(metadata.len() > 0, "shared library is empty");
}

#[test]
fn build_reports_missing_tools() {
    let (mlir_src, preset) = parse_and_lower("let x = 1; x + 1");
    let fake_tools = eval::MlirBuildTools {
        mlir_opt: "definitely_missing_tool".into(),
        mlir_translate: "definitely_missing_translate".into(),
        clang: "definitely_missing_clang".into(),
        timeout: Duration::from_secs(1),
    };
    let opts = eval::MlirBuildOptions {
        preset: preset.as_str(),
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };

    let err = eval::build_mlir_artifacts(&mlir_src, &fake_tools, &opts)
        .expect_err("expected missing tool error");
    match err {
        eval::MlirBuildError::ToolMissing(name) => assert_eq!(name, "mlir-translate"),
        other => panic!("unexpected error: {other:?}"),
    }
}
