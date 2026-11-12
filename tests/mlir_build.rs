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

mod common;

fn parse_and_lower(src: &str) -> (String, eval::MlirLowerPreset) {
    let module = parser::parse_with_diagnostics(src).expect("parse module");
    let ir = eval::lower_to_ir(&module).expect("lowering");
    let plain_mlir = eval::emit_mlir_string(&ir, eval::MlirLowerPreset::None);
    (plain_mlir, eval::MlirLowerPreset::None)
}

/// The MLIR/clang tools, or a COUNTED skip when the host genuinely lacks them.
///
/// The `eprintln!("Skipping MLIR build test: ...")` this replaced never read
/// `MIND_BENCH_REQUIRE`, so a tier that demanded a real backend could not turn
/// the absence into a failure — and cargo swallows a passing test's output, so
/// the announcement was not even observable. `gate::skipped` panics under
/// enforcement and otherwise emits the `ran=0` marker the tier script counts.
fn resolve_or_skip() -> Option<eval::MlirBuildTools> {
    match eval::resolve_mlir_build_tools() {
        Ok(tools) => Some(tools),
        Err(err) => {
            crate::common::gate::skipped("mlir_build", &format!("{err}"));
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
        // The build pipeline now always runs `mlir-opt` first (even on
        // the `"none"` preset) so the IR is fully lowered to the LLVM
        // dialect before `mlir-translate` runs. Either tool may be the
        // one that's reported missing depending on the preset's pipeline
        // contents; both are acceptable outcomes for this test, which
        // only cares that the error type is `ToolMissing`.
        eval::MlirBuildError::ToolMissing(name) => {
            assert!(
                name == "mlir-opt" || name == "mlir-translate",
                "expected mlir-opt or mlir-translate, got {name}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

/// Builds `let x: Tensor[<dt>,(4)] = <fill>; x` on the MLIR export route
/// (`emit_mlir_string` + `build_mlir_artifacts`, the `mind --emit-shared` path),
/// runs it, and returns element 2 as raw bits (f64/i64 as is, f32/i32
/// zero-extended). Readback harness: `@main` returns that element bitcast to
/// i64, and `mlir-runner` prints it. `Err` is the build refusal text.
fn fill_bits_on_export_route(dt: &str, fill: &str) -> Option<Result<u64, String>> {
    let tools = resolve_or_skip()?;
    let Ok(runner) = which::which("mlir-runner").or_else(|_| which::which("mlir-cpu-runner"))
    else {
        crate::common::gate::skipped("mlir_build", "mlir-runner not on PATH");
        return None;
    };
    let module = parser::parse_with_diagnostics(&format!("let x: Tensor[{dt},(4)] = {fill}; x"))
        .expect("parse module");
    let ir = eval::lower_to_ir(&module).expect("lowering");
    let ids: Vec<usize> = ir
        .instrs
        .iter()
        .filter_map(|i| match i {
            libmind::ir::Instr::ConstTensor(id, ..) => Some(id.0),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 1, "one fill tensor expected: {:?}", ir.instrs);
    let to_i64 = match dt {
        "f64" => "%rb = arith.bitcast %re : f64 to i64",
        "f32" => "%rw = arith.bitcast %re : f32 to i32\n    %rb = arith.extui %rw : i32 to i64",
        "i32" => "%rb = arith.extui %re : i32 to i64",
        _ => "%rb = arith.ori %re, %re : i64",
    };
    let plain = eval::emit_mlir_string(&ir, eval::MlirLowerPreset::None);
    let (head, tail) = plain.rsplit_once("    return\n").expect("main returns");
    let text = format!(
        "{}    %ri = arith.constant 2 : index\n    \
         %re = tensor.extract %{}[%ri] : tensor<4x{dt}>\n    {to_i64}\n    return %rb : i64\n{tail}",
        head.replacen("func.func @main() -> () {", "func.func @main() -> i64 {", 1),
        ids[0],
    );
    let opts = eval::MlirBuildOptions {
        preset: eval::MlirLowerPreset::None.as_str(),
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };
    let products = match eval::build_mlir_artifacts(&text, &tools, &opts) {
        Ok(p) => p,
        Err(err) => return Some(Err(format!("{err}"))),
    };
    let dir = tempdir().expect("tempdir");
    let llvm_mlir = dir.path().join("fill.mlir");
    fs::write(&llvm_mlir, &products.optimized_mlir).expect("write lowered mlir");
    let out = std::process::Command::new(runner)
        .args(["-e", "main", "--entry-point-result=i64"])
        .arg(&llvm_mlir)
        .output()
        .expect("spawn mlir-runner");
    assert!(
        out.status.success(),
        "runner: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8_lossy(&out.stdout);
    Some(Ok(printed.trim().parse::<i64>().expect("i64 result") as u64))
}

/// A float tensor fill must carry the exact binary64 value, and for f32 the
/// binary64 value rounded to binary32 (nearest, ties to even). Expected bits are
/// data, taken outside the repo from `struct.pack('<d'|'<f', x)`; f32 `1e300`
/// overflows to +inf under ties-to-even (`struct.pack('<f', inf)`).
#[test]
fn float_fill_runs_exact_bits_on_export_route() {
    let cases: [(&str, &str, u64); 18] = [
        ("f64", "1.5", 0x3FF8_0000_0000_0000),
        ("f64", "-0.25", 0xBFD0_0000_0000_0000),
        ("f64", "1e300", 0x7E37_E43C_8800_759C),
        ("f64", "0.1", 0x3FB9_9999_9999_999A),
        ("f64", "1", 0x3FF0_0000_0000_0000),
        ("f64", "0", 0x0),
        ("f64", "-0.0", 0x8000_0000_0000_0000),
        ("f32", "1.5", 0x3FC0_0000),
        ("f32", "-0.25", 0xBE80_0000),
        ("f32", "1e300", 0x7F80_0000),
        ("f32", "0.1", 0x3DCC_CCCD),
        ("f32", "1", 0x3F80_0000),
        ("f32", "-0.0", 0x8000_0000),
        // 1 + 2^-24 and 1 + 3*2^-24 are f32 ties: each rounds to the even neighbour.
        ("f32", "1.0000000596046448", 0x3F80_0000),
        ("f32", "1.0000001788139343", 0x3F80_0002),
        ("f32", "1.000000059604645", 0x3F80_0001),
        // Positive controls: integer fills keep their integer arm.
        ("i32", "-7", 0xFFFF_FFF9),
        ("i64", "-3", 0xFFFF_FFFF_FFFF_FFFD),
    ];
    let mut wrong = Vec::new();
    for (dt, fill, want) in cases {
        let Some(got) = fill_bits_on_export_route(dt, fill) else {
            return;
        };
        match got {
            Ok(got) if got == want => {}
            Ok(got) => wrong.push(format!("{dt} fill {fill}: got {got:#x}, want {want:#x}")),
            Err(e) => wrong.push(format!("{dt} fill {fill}: build refused: {e}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 18 fills wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
