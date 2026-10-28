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

//! The concurrent `mlir-opt | mlir-translate` stage must be indistinguishable
//! from the serial one: same `BuildProducts` bytes, same files, same error
//! attribution.

use std::path::Path;
use std::time::Duration;

use super::pipe::{run_mlir_opt_into_translate, tee_trimmed_for_test};
use super::*;

/// The serial reference: the opt/translate leg of `build_all_with_objects` as
/// it was before the two tools ran concurrently.
fn build_serial(
    mlir_src: &str,
    tools: &BuildTools,
    opts: &BuildOptions<'_>,
) -> Result<BuildProducts, BuildError> {
    let lowered = crate::mlir::nobuiltin::lower_for_native_build(mlir_src, opts.preset)
        .map_err(BuildError::Internal)?;
    let lowered = crate::mlir::gemm_tuning::for_translator(lowered, &tools.mlir_translate);
    let optimized_mlir = if let Some(pipeline) = combine_pipelines(opts.preset, opts.opt_pipeline) {
        run_mlir_opt(&lowered, &pipeline, tools)?
    } else {
        lowered
    };
    if let Some(path) = opts.emit_mlir_file {
        write_text_file(path, &optimized_mlir)?;
    }
    let llvm_ir = run_mlir_translate(&optimized_mlir, tools)?;
    if let Some(path) = opts.emit_llvm_file {
        write_text_file(path, &llvm_ir)?;
    }
    Ok(BuildProducts {
        optimized_mlir,
        llvm_ir,
    })
}

/// The three `e2e_compile_shared` bench programs plus one tensor program that
/// selects the `arith-linalg` pipeline.
const FIXTURES: [(&str, &str); 4] = [
    ("e2e_small", "pub fn add(a: i64, b: i64) -> i64 { a + b }\n"),
    (
        "e2e_medium",
        "pub fn poly(x: i64) -> i64 {\n    let a = x * x;\n    let b = a * x;\n    \
         let c = b + a;\n    c + x * 7 - 3\n}\n",
    ),
    (
        "e2e_large",
        "pub fn f0(x: i64) -> i64 { x * x + 1 }\n\
         pub fn f1(x: i64) -> i64 { let a = f0(x); a * a - x }\n\
         pub fn f2(x: i64) -> i64 { let a = f1(x); let b = f0(a); a + b }\n\
         pub fn f3(x: i64) -> i64 { let a = f2(x); let b = f1(a); a * 3 + b - x }\n\
         pub fn driver(x: i64) -> i64 {\n    let a = f3(x);\n    let b = f2(a);\n    \
         let c = f1(b);\n    a + b + c\n}\n",
    ),
    (
        "arith_linalg",
        "pub fn u(x: tensor<f32[2]>) -> f32 {\n    return x[0]\n}\n",
    ),
];

/// The build tools, or `None` (a visible skip) when the host lacks them.
/// `MIND_BENCH_REQUIRE=1` turns the absence into a failure.
fn tools_or_skip(test: &str) -> Option<BuildTools> {
    match resolve_tools() {
        Ok(tools) => Some(tools),
        Err(err) => {
            assert!(
                std::env::var("MIND_BENCH_REQUIRE").as_deref() != Ok("1"),
                "{test}: MLIR tools required but unavailable: {err}"
            );
            eprintln!("{test}: skipped ran=0 ({err})");
            None
        }
    }
}

fn mlir_for(src: &str) -> String {
    let products =
        crate::pipeline::compile_source(src, &crate::pipeline::CompileOptions::default())
            .expect("fixture compiles");
    #[cfg(feature = "autodiff")]
    let mlir = crate::pipeline::lower_to_mlir(&products.ir, None);
    #[cfg(not(feature = "autodiff"))]
    let mlir = crate::pipeline::lower_to_mlir(&products.ir);
    mlir.expect("fixture lowers to MLIR").primal_mlir
}

fn opts_into<'a>(preset: &'a str, mlir: &'a Path, llvm: &'a Path) -> BuildOptions<'a> {
    BuildOptions {
        preset,
        emit_mlir_file: Some(mlir),
        emit_llvm_file: Some(llvm),
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    }
}

fn subprocess_failure(err: BuildError) -> (&'static str, String) {
    match err {
        BuildError::Subprocess { tool, stderr } => (tool, stderr),
        other => panic!("expected a subprocess failure, got {other:?}"),
    }
}

#[test]
fn tee_forwards_exactly_the_trimmed_text_at_every_chunk_split() {
    let samples: [&[u8]; 7] = [
        b"",
        b" \n\t ",
        b"module {\n}\n",
        b"\n\n  module {\n  llvm.func @f() \n}\n\n \r\n",
        b"a",
        b" a \x0b\x0c b \n",
        b"x\n\n\ny   \n",
    ];
    for sample in samples {
        let expected = decode_to_string(sample);
        for split_a in 0..=sample.len() {
            for split_b in split_a..=sample.len() {
                let chunks = [
                    &sample[..split_a],
                    &sample[split_a..split_b],
                    &sample[split_b..],
                ];
                let chunks: Vec<&[u8]> = chunks.into_iter().filter(|c| !c.is_empty()).collect();
                let (raw, sent) = tee_trimmed_for_test(&chunks);
                assert_eq!(raw, sample);
                assert_eq!(
                    sent,
                    expected.as_bytes(),
                    "sample {sample:?} split at {split_a}/{split_b}"
                );
            }
        }
    }
    // A non-ASCII whitespace edge is NOT trimmed by the tee, so the forwarded
    // bytes differ from the decoded text: this is the case the runner detects
    // and answers with a serial re-run of mlir-translate.
    let nbsp = "\u{a0}module {}\u{a0}".as_bytes();
    let (_, sent) = tee_trimmed_for_test(&[nbsp]);
    assert_ne!(sent, decode_to_string(nbsp).as_bytes());
}

#[test]
fn pipelined_build_products_are_byte_equal_to_serial() {
    let Some(tools) = tools_or_skip("pipelined_build_products_are_byte_equal_to_serial") else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let mut presets = Vec::new();
    for (name, src) in FIXTURES {
        let mlir = mlir_for(src);
        let preset = preset_for_mlir(&mlir);
        presets.push(preset);

        let (sm, sl) = (
            dir.path().join(format!("{name}.serial.mlir")),
            dir.path().join(format!("{name}.serial.ll")),
        );
        let (pm, pl) = (
            dir.path().join(format!("{name}.pipe.mlir")),
            dir.path().join(format!("{name}.pipe.ll")),
        );
        let serial =
            build_serial(&mlir, &tools, &opts_into(preset, &sm, &sl)).expect("serial build");
        let piped =
            build_all(&mlir, &tools, &opts_into(preset, &pm, &pl)).expect("pipelined build");

        assert!(
            serial.llvm_ir.contains("define"),
            "{name}: no LLVM function"
        );
        assert_eq!(
            piped.optimized_mlir, serial.optimized_mlir,
            "{name}: optimized_mlir"
        );
        assert_eq!(piped.llvm_ir, serial.llvm_ir, "{name}: llvm_ir");
        for (a, b) in [(&pm, &sm), (&pl, &sl)] {
            assert_eq!(
                std::fs::read(a).unwrap(),
                std::fs::read(b).unwrap(),
                "{name}: {a:?}"
            );
        }

        // Positive control: the result above must come from the concurrent
        // translate, not from the serial fallback (which would make the
        // equality hold trivially).
        let lowered = crate::mlir::nobuiltin::lower_for_native_build(&mlir, preset).unwrap();
        let lowered = crate::mlir::gemm_tuning::for_translator(lowered, &tools.mlir_translate);
        let pipeline = combine_pipelines(preset, None).expect("preset has a pipeline");
        let out = run_mlir_opt_into_translate(&lowered, &pipeline, &tools).expect("opt ok");
        assert!(out.pipelined, "{name}: fell back to the serial translate");
        assert_eq!(out.llvm_ir.expect("translate ok"), serial.llvm_ir, "{name}");
    }
    assert_eq!(presets, ["core", "core", "core", "arith-linalg"]);
}

#[test]
fn broken_pipeline_is_attributed_to_mlir_opt() {
    let Some(tools) = tools_or_skip("broken_pipeline_is_attributed_to_mlir_opt") else {
        return;
    };
    let mlir = mlir_for(FIXTURES[0].1);
    let opts = BuildOptions {
        preset: "core",
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: None,
        opt_pipeline: Some("no-such-mind-pass"),
        target_triple: None,
    };
    let piped = subprocess_failure(build_all(&mlir, &tools, &opts).unwrap_err());
    let serial = subprocess_failure(build_serial(&mlir, &tools, &opts).unwrap_err());
    assert_eq!(piped.0, "mlir-opt");
    assert_eq!(piped, serial);

    // An mlir-opt failure still outranks a missing mlir-translate.
    let no_translate = BuildTools {
        mlir_translate: "/nonexistent/mlir-translate".into(),
        ..tools.clone()
    };
    let piped = subprocess_failure(build_all(&mlir, &no_translate, &opts).unwrap_err());
    assert_eq!(piped, serial);
}

#[test]
fn translate_failure_is_attributed_to_mlir_translate() {
    let Some(tools) = tools_or_skip("translate_failure_is_attributed_to_mlir_translate") else {
        return;
    };
    // A tensor parameter under the scalar `core` pipeline survives mlir-opt and
    // is rejected by mlir-translate.
    let mlir = mlir_for("pub fn s(x: tensor<f32[2]>) -> i64 {\n    return 7\n}\n");
    let dir = tempfile::tempdir().expect("tempdir");
    let (sm, sl) = (dir.path().join("s.mlir"), dir.path().join("s.ll"));
    let (pm, pl) = (dir.path().join("p.mlir"), dir.path().join("p.ll"));
    let serial =
        subprocess_failure(build_serial(&mlir, &tools, &opts_into("core", &sm, &sl)).unwrap_err());
    let piped =
        subprocess_failure(build_all(&mlir, &tools, &opts_into("core", &pm, &pl)).unwrap_err());
    assert_eq!(piped.0, "mlir-translate");
    assert_eq!(piped, serial);
    // The optimized MLIR is still written when translate fails, as before.
    assert_eq!(std::fs::read(&pm).unwrap(), std::fs::read(&sm).unwrap());
    assert!(!pl.exists() && !sl.exists());
}

#[test]
fn missing_translate_and_timeout_match_serial() {
    let Some(tools) = tools_or_skip("missing_translate_and_timeout_match_serial") else {
        return;
    };
    let mlir = mlir_for(FIXTURES[0].1);
    let dir = tempfile::tempdir().expect("tempdir");
    let (m, l) = (dir.path().join("m.mlir"), dir.path().join("m.ll"));
    let opts = opts_into("core", &m, &l);

    let no_translate = BuildTools {
        mlir_translate: "/nonexistent/mlir-translate".into(),
        ..tools.clone()
    };
    for build in [build_all, build_serial] {
        match build(&mlir, &no_translate, &opts) {
            Err(BuildError::ToolMissing(tool)) => assert_eq!(tool, "mlir-translate"),
            other => panic!("expected ToolMissing(mlir-translate), got {other:?}"),
        }
    }

    let instant = BuildTools {
        timeout: Duration::from_nanos(1),
        ..tools
    };
    for build in [build_all, build_serial] {
        match build(&mlir, &instant, &opts) {
            Err(BuildError::Timeout(tool)) => assert_eq!(tool, "mlir-opt"),
            other => panic!("expected Timeout(mlir-opt), got {other:?}"),
        }
    }
}
