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
//! STREAMING. [`prepare`] runs every stage before the last and returns the
//! last as a [`LastStage`]. The native build streams that stage's output into
//! a concurrently running `mlir-translate` (`src/eval/mlir_build_pipe.rs`);
//! [`run_pipeline`] runs it serially. Both read its arguments, its stdin text
//! and its failure report from the same value, so stage B is defined once.
//!
//! DIAGNOSTICS. Stage A reads the input from a file and prints debug
//! locations, so every op stage B sees still carries the line and column of
//! the text the build handed to `mlir-opt`; when stage B fails, `mlir-opt`
//! re-opens that file and shows the original line. The temporary path is then
//! written back as `<stdin>` — what a one-invocation run printed — so a
//! diagnostic reads exactly as before and never cites a generated line.
//!
//! TOOLCHAIN. The tensor pipeline is supported on MLIR
//! [`crate::diagnostics::toolchain::MIN_MLIR_MAJOR`] or newer. An older
//! `mlir-opt` is NOT refused — MLIR is an optional backend and many tensor
//! programs build correctly on it — but it draws a one-line warning, and if
//! `mlir-opt` then fails, the error carries the E5004 explanation: the version
//! found, the one needed and where a usable one is installed.

use std::borrow::Cow;
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
    prepare(input, pipeline, tools)?.run(tools)
}

/// The last `mlir-opt` invocation of a pipeline, every earlier stage already
/// run: stage B of a cut pipeline, or the whole of one that is not cut.
/// [`run_pipeline`] runs it to completion; the native build streams its output
/// into `mlir-translate` instead (`src/eval/mlir_build_pipe.rs`). Both take the
/// arguments, the stdin text and the failure report from here, so the serial
/// and the streamed leg cannot drift apart.
pub struct LastStage<'a> {
    pipeline: Cow<'a, str>,
    input: Cow<'a, str>,
    /// Stage A's input file and its path. A stage B diagnostic re-opens the
    /// file, so it lives as long as this value. `None` when nothing was cut.
    source: Option<(tempfile::NamedTempFile, String)>,
    note: Option<String>,
}

impl LastStage<'_> {
    /// The `mlir-opt` arguments; the input goes on stdin.
    pub fn args(&self) -> Vec<String> {
        vec![pipeline_arg(&self.pipeline)]
    }

    /// The text to feed `mlir-opt` on stdin.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// The error a failed run reports, given `mlir-opt`'s stderr.
    pub fn failure(&self, tools: &BuildTools, stderr: &[u8]) -> BuildError {
        opt_failure(tools, stderr, self.rename(), self.note.as_deref())
    }

    /// Run it to completion and return the printed module.
    pub fn run(self, tools: &BuildTools) -> Result<String, BuildError> {
        let (input, rename, note) = (Some(&*self.input), self.rename(), self.note.as_deref());
        run_opt(tools, &self.pipeline, input, &[], rename, note)
    }

    fn rename(&self) -> Option<&str> {
        self.source.as_ref().map(|(_, path)| path.as_str())
    }
}

/// Run every stage of `pipeline` before the last over `input` and return the
/// last one. A pipeline with no `one-shot-bufferize` is a single stage, so
/// nothing runs here.
pub fn prepare<'a>(
    input: &'a str,
    pipeline: &'a str,
    tools: &BuildTools,
) -> Result<LastStage<'a>, BuildError> {
    let passes = split_passes(pipeline);
    let Some(cut) = passes.iter().position(|p| p.starts_with(BUFFERIZE_PASS)) else {
        return Ok(LastStage {
            pipeline: Cow::Borrowed(pipeline),
            input: Cow::Borrowed(input),
            source: None,
            note: None,
        });
    };
    let note = toolchain::tensor_pipeline_note(&tools.mlir_opt);

    let mut file = tempfile::Builder::new()
        .prefix("mind-opt-")
        .suffix(".mlir")
        .tempfile()?;
    file.write_all(input.as_bytes())?;
    file.flush()?;
    let path = file.path().to_string_lossy().into_owned();
    let stage_a_args = [String::from("--mlir-print-debuginfo"), path.clone()];
    let head = passes[..cut].join(",");
    let note_ref = note.as_deref();
    let folded = run_opt(tools, &head, None, &stage_a_args, Some(&path), note_ref)?;
    let declared = crate::mlir::const_globals::predeclare_dense_constant_globals(&folded);
    Ok(LastStage {
        pipeline: Cow::Owned(passes[cut..].join(",")),
        input: Cow::Owned(declared),
        source: Some((file, path)),
        note,
    })
}

/// One `mlir-opt` run. `input` goes on stdin when given (else `extra` names
/// the input file); `rename` is a path to report as `<stdin>` on failure, and
/// `note` an old-toolchain explanation to append to a failure.
fn run_opt(
    tools: &BuildTools,
    pipeline: &str,
    input: Option<&str>,
    extra: &[String],
    rename: Option<&str>,
    note: Option<&str>,
) -> Result<String, BuildError> {
    let mut args = vec![pipeline_arg(pipeline)];
    args.extend_from_slice(extra);
    let output = run_command(
        &tools.mlir_opt,
        &args,
        input.map(str::as_bytes),
        tools.timeout,
        "mlir-opt",
    )?;
    if !output.status.success() {
        return Err(opt_failure(tools, &output.stderr, rename, note));
    }
    Ok(decode_to_string(&output.stdout))
}

/// The `--pass-pipeline` argument. MLIR 18+ requires the pass pipeline to be
/// wrapped with an anchor op.
fn pipeline_arg(pipeline: &str) -> String {
    format!("--pass-pipeline=builtin.module({pipeline})")
}

/// A failed `mlir-opt` run as the build reports it: `rename` written back as
/// `<stdin>`, then `note` appended, or the bufferize hint when there is none.
fn opt_failure(
    tools: &BuildTools,
    stderr: &[u8],
    rename: Option<&str>,
    note: Option<&str>,
) -> BuildError {
    let mut stderr = decode_to_string(stderr);
    if let Some(path) = rename {
        stderr = stderr.replace(path, STDIN_NAME);
    }
    let stderr = match note {
        Some(note) => format!("{stderr}\nnote: {note}"),
        None => toolchain::with_bufferize_hint(&tools.mlir_opt, stderr),
    };
    BuildError::Subprocess {
        tool: "mlir-opt",
        stderr,
    }
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
