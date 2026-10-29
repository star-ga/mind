// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `build_all_with_objects`: the MLIR leg (lower, mlir-opt, mlir-translate,
//! object emit) with the runtime-support object compiled on an overlapping
//! scoped thread and joined before the shared link.

use super::{
    BuildError, BuildOptions, BuildProducts, BuildTools, combine_pipelines,
    compile_runtime_support_obj, run_clang_codegen, run_mlir_translate, write_text_file,
};

/// Like [`super::build_all`], but links additional caller-provided object files into
/// the `emit_shared` output (after the runtime-support stub). Used for native
/// cross-module linking: a consumer cdylib that imports a self-contained `std`
/// substrate module passes that module's compiled `.o` here so its symbols
/// resolve natively. With an empty `extra_objects` slice this is identical to
/// [`super::build_all`].
pub fn build_all_with_objects(
    mlir_src: &str,
    tools: &BuildTools,
    opts: &BuildOptions<'_>,
    extra_objects: &[std::path::PathBuf],
) -> Result<BuildProducts, BuildError> {
    // The runtime-support object depends only on `tools` and the target
    // triple — never on the MLIR — so it is started on its own thread before
    // the mlir-opt / mlir-translate leg and joined right before the shared
    // link that consumes it. Its cache probe (or cold clang compile) then
    // overlaps the MLIR leg instead of running after it. Output-identical by
    // construction: the same inputs reach the same tool, and the object is
    // still linked in full. Each subprocess keeps its own `tools.timeout`.
    // A scoped thread is joined on every exit path, so an mlir-opt /
    // mlir-translate error returns exactly as before (the overlapped object
    // is discarded and its temp file removed).
    std::thread::scope(|scope| {
        let runtime_job = opts
            .emit_shared
            .map(|_| scope.spawn(|| compile_runtime_support_obj(tools, opts.target_triple)));

        let preset_name = opts.preset;
        let lowered = crate::mlir::nobuiltin::lower_for_native_build(mlir_src, preset_name)
            .map_err(BuildError::Internal)?;
        let lowered = crate::mlir::gemm_tuning::for_translator(lowered, &tools.mlir_translate);
        let combined_pipeline = combine_pipelines(preset_name, opts.opt_pipeline);
        // mlir-opt and mlir-translate run concurrently (see `pipe`): translate's
        // startup overlaps opt's run, and it still receives exactly the text that
        // becomes `optimized_mlir`. The `.mlir` file is written before translate's
        // outcome is surfaced, so a translate failure leaves it behind as before.
        let (optimized_mlir, llvm_ir) = if let Some(pipeline) = combined_pipeline {
            let piped = super::pipe::run_mlir_opt_into_translate(&lowered, &pipeline, tools)?;
            if let Some(path) = opts.emit_mlir_file {
                write_text_file(path, &piped.optimized_mlir)?;
            }
            let llvm_ir = piped.llvm_ir?;
            (piped.optimized_mlir, llvm_ir)
        } else {
            if let Some(path) = opts.emit_mlir_file {
                write_text_file(path, &lowered)?;
            }
            let llvm_ir = run_mlir_translate(&lowered, tools)?;
            (lowered, llvm_ir)
        };

        if let Some(path) = opts.emit_llvm_file {
            write_text_file(path, &llvm_ir)?;
        }

        if let Some(path) = opts.emit_obj_file {
            run_clang_codegen(&llvm_ir, tools, opts.target_triple, path, false, &[])?;
        }

        if let (Some(path), Some(job)) = (opts.emit_shared, runtime_job) {
            // Phase 6.5 Stage 1b: link the runtime-support stub into the
            // shared library so vec_new / vec_push / __mind_load_i64 etc. are
            // resolved without an external dependency. Join first: a clang
            // failure in the overlapped compile surfaces here as the same
            // `BuildError` the serial path returned.
            let runtime_obj = job.join().map_err(|_| {
                BuildError::Internal("runtime-support compile thread panicked".into())
            })??;
            let mut extra = vec![runtime_obj.path().to_path_buf()];
            extra.extend(extra_objects.iter().cloned());
            run_clang_codegen(&llvm_ir, tools, opts.target_triple, path, true, &extra)?;
        }

        Ok(BuildProducts {
            optimized_mlir,
            llvm_ir,
        })
    })
}
