// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! The `mlir-opt` leg of the native build, run in two invocations when the
//! pipeline bufferizes.
//!
//! WHY TWO. `one-shot-bufferize{bufferize-function-boundaries=true}` creates
//! the module globals of dense tensor constants in an order read out of a
//! pointer-keyed map (see [`crate::mlir::const_globals`]). Pinning that order
//! needs the set of constants bufferization will actually see — and that set
//! is only known AFTER `canonicalize`/`cse`, which fold tensor arithmetic on
//! constants into new constants and delete the folded operands. So the
//! pipeline is cut just before its first `one-shot-bufferize`:
//!
//! 1. stage A runs the passes before the cut and prints the module;
//! 2. the still-used dense constants of that output are declared as ordered
//!    globals ([`crate::mlir::const_globals::predeclare_dense_constant_globals`]);
//! 3. stage B runs the rest.
//!
//! The pass sequence is the preset's, unchanged; printing and re-parsing
//! between two passes is lossless. A pipeline with no `one-shot-bufferize`
//! (the scalar `core` preset) still runs as ONE invocation, byte for byte as
//! before.
//!
//! DIAGNOSTICS. Stage A reads the input from a file and prints debug
//! locations, so every op stage B sees still carries the line and column of
//! the text the build handed to `mlir-opt`; when stage B fails, `mlir-opt`
//! re-opens that file and shows the original line. The temporary path is then
//! written back as `<stdin>` — what a one-invocation run printed — so a
//! diagnostic reads exactly as before and never cites a generated line.
//!
//! TOOLCHAIN. Bufferizing needs MLIR
//! [`crate::diagnostics::toolchain::MIN_MLIR_MAJOR`] or newer; an older
//! `mlir-opt` is refused up front, naming the version found, the one required
//! and where a usable one is installed, instead of failing inside
//! bufferization.

use std::io::Write;

use crate::diagnostics::toolchain;
use crate::eval::mlir_build::{BuildError, BuildTools, decode_to_string, run_command};

/// The pass at which the pipeline is cut.
const BUFFERIZE_PASS: &str = "one-shot-bufferize";

/// What a one-invocation run called its input in diagnostics.
const STDIN_NAME: &str = "<stdin>";

/// Run `pipeline` (comma-separated passes, anchored on `builtin.module`) over
/// `input` and return the printed result.
pub fn run_pipeline(input: &str, pipeline: &str, tools: &BuildTools) -> Result<String, BuildError> {
    let passes = split_passes(pipeline);
    let Some(cut) = passes.iter().position(|p| p.starts_with(BUFFERIZE_PASS)) else {
        return run_opt(tools, pipeline, Some(input), &[], None);
    };
    toolchain::require_bufferizing_mlir_opt(&tools.mlir_opt)
        .map_err(BuildError::ToolchainIncompatible)?;

    let mut file = tempfile::Builder::new()
        .prefix("mind-opt-")
        .suffix(".mlir")
        .tempfile()?;
    file.write_all(input.as_bytes())?;
    file.flush()?;
    let path = file.path().to_string_lossy().into_owned();
    let stage_a_args = [String::from("--mlir-print-debuginfo"), path.clone()];
    let head = passes[..cut].join(",");
    let folded = run_opt(tools, &head, None, &stage_a_args, Some(&path))?;
    let declared = crate::mlir::const_globals::predeclare_dense_constant_globals(&folded);
    let tail = passes[cut..].join(",");
    let out = run_opt(tools, &tail, Some(&declared), &[], Some(&path));
    drop(file);
    out
}

/// One `mlir-opt` run. `input` goes on stdin when given (else `extra` names
/// the input file); `rename` is a path to report as `<stdin>` on failure.
fn run_opt(
    tools: &BuildTools,
    pipeline: &str,
    input: Option<&str>,
    extra: &[String],
    rename: Option<&str>,
) -> Result<String, BuildError> {
    // MLIR 18+ requires the pass pipeline to be wrapped with an anchor op.
    let mut args = vec![format!("--pass-pipeline=builtin.module({pipeline})")];
    args.extend_from_slice(extra);
    let output = run_command(
        &tools.mlir_opt,
        &args,
        input.map(str::as_bytes),
        tools.timeout,
        "mlir-opt",
    )?;
    if !output.status.success() {
        let mut stderr = decode_to_string(&output.stderr);
        if let Some(path) = rename {
            stderr = stderr.replace(path, STDIN_NAME);
        }
        return Err(BuildError::Subprocess {
            tool: "mlir-opt",
            stderr: toolchain::with_bufferize_hint(&tools.mlir_opt, stderr),
        });
    }
    Ok(decode_to_string(&output.stdout))
}

/// Split a pass pipeline at its top-level commas (commas inside `{...}` pass
/// options or `(...)` nested pipelines stay with their pass).
fn split_passes(pipeline: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0usize);
    for (i, ch) in pipeline.char_indices() {
        match ch {
            '{' | '(' => depth += 1,
            '}' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(pipeline[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(pipeline[start..].trim());
    out.retain(|p| !p.is_empty());
    out
}

#[cfg(test)]
mod tests {
    use super::split_passes;

    #[test]
    fn splits_only_at_top_level_commas() {
        assert_eq!(
            split_passes(
                "canonicalize,cse,one-shot-bufferize{bufferize-function-boundaries=true},\
                 func.func(a,b),convert-to-llvm"
            ),
            [
                "canonicalize",
                "cse",
                "one-shot-bufferize{bufferize-function-boundaries=true}",
                "func.func(a,b)",
                "convert-to-llvm"
            ]
        );
        assert_eq!(split_passes(" a , b{x=1,y=2} "), ["a", "b{x=1,y=2}"]);
        assert!(split_passes("").is_empty());
    }
}
