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

#[cfg(feature = "mlir-build")]
use std::io::Write;
#[cfg(feature = "mlir-build")]
use std::path::Path;
#[cfg(feature = "mlir-build")]
use std::process::Command;
#[cfg(feature = "mlir-build")]
use std::process::Stdio;
#[cfg(feature = "mlir-build")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "mlir-build")]
use std::time::Duration;
#[cfg(feature = "mlir-build")]
use std::time::Instant;

#[cfg(feature = "mlir-build")]
use tempfile::NamedTempFile;

/// RFC 0005 Phase 6.5 Stage 1b — runtime-support C stub.
///
/// Bundled at compile time (like std/*.mind in Phase C) so the --emit-shared
/// cdylib path needs no external file at build time.  The text is compiled to
/// a temporary .o by `compile_runtime_support_obj` and statically linked into
/// every --emit-shared output, making the .so self-contained.
#[cfg(feature = "mlir-build")]
const MIND_RUNTIME_SUPPORT_C: &str = include_str!("../../runtime-support/mind_intrinsics.c");

#[cfg(feature = "mlir-build")]
#[derive(Clone, Debug)]
pub struct BuildTools {
    pub mlir_opt: String,
    pub mlir_translate: String,
    pub clang: String,
    pub timeout: Duration,
}

#[cfg(feature = "mlir-build")]
pub struct BuildOptions<'a> {
    pub preset: &'a str,
    pub emit_mlir_file: Option<&'a Path>,
    pub emit_llvm_file: Option<&'a Path>,
    pub emit_obj_file: Option<&'a Path>,
    pub emit_shared: Option<&'a Path>,
    pub opt_pipeline: Option<&'a str>,
    pub target_triple: Option<&'a str>,
}

#[cfg(feature = "mlir-build")]
#[derive(Debug, Clone)]
pub struct BuildProducts {
    pub optimized_mlir: String,
    pub llvm_ir: String,
}

#[cfg(feature = "mlir-build")]
#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error("tool not found: {0}")]
    ToolMissing(&'static str),
    #[error("subprocess {tool} failed: {stderr}")]
    Subprocess { tool: &'static str, stderr: String },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("timeout while running {0}")]
    Timeout(&'static str),
    #[error("internal: {0}")]
    Internal(String),
}

#[cfg(feature = "mlir-build")]
pub fn resolve_tools() -> Result<BuildTools, BuildError> {
    fn resolve(env: &str, default: &str, label: &'static str) -> Result<String, BuildError> {
        if let Ok(value) = std::env::var(env) {
            if !value.trim().is_empty() {
                if which::which(&value).is_ok() {
                    return Ok(value);
                } else {
                    return Err(BuildError::ToolMissing(label));
                }
            }
        }
        let path = which::which(default).map_err(|_| BuildError::ToolMissing(label))?;
        Ok(path.to_string_lossy().into_owned())
    }

    Ok(BuildTools {
        mlir_opt: resolve("MLIR_OPT", "mlir-opt", "mlir-opt")?,
        mlir_translate: resolve("MLIR_TRANSLATE", "mlir-translate", "mlir-translate")?,
        clang: resolve("CLANG", "clang", "clang")?,
        timeout: Duration::from_secs(60),
    })
}

#[cfg(feature = "mlir-build")]
pub fn build_all(
    mlir_src: &str,
    tools: &BuildTools,
    opts: &BuildOptions<'_>,
) -> Result<BuildProducts, BuildError> {
    let preset_name = opts.preset;
    let lowered = crate::eval::mlir_export::apply_lowering(mlir_src, preset_name)
        .map_err(BuildError::Internal)?;

    let combined_pipeline = combine_pipelines(preset_name, opts.opt_pipeline);
    let optimized_mlir = if let Some(pipeline) = combined_pipeline {
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

    if let Some(path) = opts.emit_obj_file {
        run_clang_codegen(&llvm_ir, tools, opts.target_triple, path, false, &[])?;
    }

    if let Some(path) = opts.emit_shared {
        // Phase 6.5 Stage 1b: compile the runtime-support stub to a temp .o
        // and link it into the shared library so vec_new / vec_push /
        // __mind_load_i64 etc. are resolved without an external dependency.
        let runtime_obj = compile_runtime_support_obj(tools)?;
        let extra = [runtime_obj.path().to_path_buf()];
        run_clang_codegen(&llvm_ir, tools, opts.target_triple, path, true, &extra)?;
    }

    Ok(BuildProducts {
        optimized_mlir,
        llvm_ir,
    })
}

#[cfg(feature = "mlir-build")]
fn combine_pipelines(preset: &str, extra: Option<&str>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(default) = preset_default_pipeline(preset) {
        parts.push(default.to_string());
    }
    if let Some(extra) = extra {
        let trimmed = extra.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

#[cfg(feature = "mlir-build")]
fn preset_default_pipeline(preset: &str) -> Option<&'static str> {
    // The pipelines must leave the IR entirely in the `llvm` dialect so
    // `mlir-translate --mlir-to-llvmir` can lower it without complaining
    // about missing dialect registrations. Stock `mlir-translate` only
    // registers the LLVM family at link time; every other dialect MIND
    // emits (arith, cf, scf, func, memref, tensor, linalg, …) has to be
    // converted by `mlir-opt` first.
    match preset {
        // Scalar-only pipeline. Enough for the `fn f(x, y) { x + y }`
        // class of programs that exercise `arith` + `func` + `cf` + `scf`.
        // `none` shares this path because `mlir-translate` cannot handle
        // raw `arith` / `func` ops — it only registers the LLVM dialect
        // family. Skipping the dialect conversion would force every
        // caller of `build_mlir_artifacts(preset = "none")` to fail at
        // the translate step.
        // `convert-vector-to-llvm` lowers RFC 0006 Track B `vector`-dialect
        // ops (vector.load / vector.fma / vector.reduction) emitted by the
        // `dot_f32_v` path. It is a no-op on IR that contains no vector
        // ops, so the scalar `fn f(x, y) { x + y }` class is unaffected
        // and the default `cargo build` (which never runs mlir-opt — this
        // is the `mlir-build` feature path) is byte-identical.
        "none" | "core" | "cpu-demo" | "jit-cpu" => Some(
            "canonicalize,cse,\
             convert-scf-to-cf,\
             convert-vector-to-llvm,\
             expand-strided-metadata,\
             finalize-memref-to-llvm,\
             convert-cf-to-llvm,\
             convert-arith-to-llvm,\
             convert-func-to-llvm,\
             reconcile-unrealized-casts",
        ),
        // Tensor-aware pipeline. Adds bufferization + linalg lowering
        // before the scalar leg. `convert-to-llvm` picks up any
        // remaining vector / memref / index ops at the tail.
        "arith-linalg" | "gpu-default" => Some(
            "canonicalize,cse,\
             one-shot-bufferize{bufferize-function-boundaries=true},\
             convert-linalg-to-loops,\
             convert-scf-to-cf,\
             expand-strided-metadata,\
             finalize-memref-to-llvm,\
             convert-cf-to-llvm,\
             convert-arith-to-llvm,\
             convert-func-to-llvm,\
             convert-to-llvm,\
             reconcile-unrealized-casts",
        ),
        _ => None,
    }
}

#[cfg(feature = "mlir-build")]
fn run_mlir_opt(input: &str, pipeline: &str, tools: &BuildTools) -> Result<String, BuildError> {
    // MLIR 18+ requires pass pipeline to be wrapped with anchor operation
    let args = vec![format!("--pass-pipeline=builtin.module({pipeline})")];
    let output = run_command(
        &tools.mlir_opt,
        &args,
        Some(input.as_bytes()),
        tools.timeout,
        "mlir-opt",
    )?;
    if !output.status.success() {
        return Err(BuildError::Subprocess {
            tool: "mlir-opt",
            stderr: decode_to_string(&output.stderr),
        });
    }
    Ok(decode_to_string(&output.stdout))
}

#[cfg(feature = "mlir-build")]
fn run_mlir_translate(input: &str, tools: &BuildTools) -> Result<String, BuildError> {
    let args = vec![String::from("--mlir-to-llvmir")];
    let output = run_command(
        &tools.mlir_translate,
        &args,
        Some(input.as_bytes()),
        tools.timeout,
        "mlir-translate",
    )?;
    if !output.status.success() {
        return Err(BuildError::Subprocess {
            tool: "mlir-translate",
            stderr: decode_to_string(&output.stderr),
        });
    }
    Ok(decode_to_string(&output.stdout))
}

/// Compile the bundled runtime-support C stub to a temporary .o file.
///
/// The returned `NamedTempFile` must be kept alive until the link step
/// completes; dropping it removes the .o from the filesystem.
///
/// RFC 0015 bit-identity: the C source is written with the **fixed**
/// basename `mind_intrinsics.c` (inside a per-build random directory, so
/// concurrent builds never collide). clang records only that basename as
/// the `STT_FILE` symbol in the emitted object, so the symbol — and thus
/// every byte of the linked `.so` — is reproducible across builds. A
/// random `NamedTempFile` source name (the previous behaviour) leaked into
/// `.symtab`/`.strtab`, making the keystone artifact non-deterministic and
/// silently breaking the cross-substrate byte-identity claim.
#[cfg(feature = "mlir-build")]
fn compile_runtime_support_obj(tools: &BuildTools) -> Result<NamedTempFile, BuildError> {
    let src_dir = tempfile::tempdir()?;
    let src_path = src_dir.path().join("mind_intrinsics.c");
    std::fs::write(&src_path, MIND_RUNTIME_SUPPORT_C.as_bytes())?;

    let obj_tmp = NamedTempFile::with_suffix(".o")?;

    // RFC 0015 bit-identity contract for this clang invocation:
    //   * NO `-g` — debug info would embed the absolute random tempdir path
    //     via DWARF `DW_AT_comp_dir` / line tables and re-break byte-identity.
    //   * `-ffile-prefix-map` maps the random parent dir to `.` so that even
    //     if a path were embedded, it normalises to a stable prefix. Belt and
    //     suspenders behind the fixed `mind_intrinsics.c` basename above.
    let prefix_map = format!("-ffile-prefix-map={}=.", src_dir.path().to_string_lossy());
    let args: Vec<String> = vec![
        "-x".into(),
        "c".into(),
        src_path.to_string_lossy().into_owned(),
        "-c".into(),
        "-fPIC".into(),
        "-O2".into(),
        prefix_map,
        "-o".into(),
        obj_tmp.path().to_string_lossy().into_owned(),
    ];

    let output = run_command(&tools.clang, &args, None, tools.timeout, "clang")?;
    if !output.status.success() {
        return Err(BuildError::Subprocess {
            tool: "clang",
            stderr: decode_to_string(&output.stderr),
        });
    }

    // `src_dir` must live until clang finishes reading the source above;
    // drop it explicitly now that the object is written.
    drop(src_dir);

    Ok(obj_tmp)
}

#[cfg(feature = "mlir-build")]
fn run_clang_codegen(
    llvm_ir: &str,
    tools: &BuildTools,
    target_triple: Option<&str>,
    output_path: &Path,
    shared: bool,
    extra_objs: &[std::path::PathBuf],
) -> Result<(), BuildError> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut temp = NamedTempFile::new()?;
    temp.write_all(llvm_ir.as_bytes())?;

    let mut args: Vec<String> = Vec::new();
    args.push("-x".into());
    args.push("ir".into());
    args.push(temp.path().to_string_lossy().into_owned());
    if let Some(triple) = target_triple {
        let trimmed = triple.trim();
        if !trimmed.is_empty() {
            args.push(format!("--target={trimmed}"));
        }
    }
    if shared {
        args.push("-shared".into());
        args.push("-fPIC".into());
    } else {
        args.push("-c".into());
    }
    // Extra object files (e.g. runtime-support stub).
    // Reset -x before adding them so clang treats each as a native object,
    // not as LLVM IR (the -x ir flag set above applies to all subsequent
    // inputs unless overridden).
    if !extra_objs.is_empty() {
        args.push("-x".into());
        args.push("none".into());
        for obj in extra_objs {
            args.push(obj.to_string_lossy().into_owned());
        }
    }
    args.push("-o".into());
    args.push(output_path.to_string_lossy().into_owned());

    let output = run_command(&tools.clang, &args, None, tools.timeout, "clang")?;
    if !output.status.success() {
        return Err(BuildError::Subprocess {
            tool: "clang",
            stderr: decode_to_string(&output.stderr),
        });
    }

    Ok(())
}

#[cfg(feature = "mlir-build")]
struct CommandOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: std::process::ExitStatus,
}

/// Spawn `program` with `args`, optionally feed `input` on stdin, and collect
/// stdout + stderr while respecting `timeout`.
///
/// The previous implementation wrote all of stdin before draining stdout, which
/// deadlocked whenever the child's stdout pipe filled up (> 64 KiB OS buffer).
/// The parser MLIR is ~61 KiB in and ~95 KiB out, which exceeds the pipe buffer
/// and caused a permanent hang.  This version drains stdout and stderr on
/// dedicated threads while the main thread writes stdin and polls for
/// termination, eliminating the deadlock.
#[cfg(feature = "mlir-build")]
fn run_command(
    program: &str,
    args: &[String],
    input: Option<&[u8]>,
    timeout: Duration,
    label: &'static str,
) -> Result<CommandOutput, BuildError> {
    use std::io::Read;
    use std::thread;

    let mut cmd = Command::new(program);
    cmd.args(args.iter().map(|s| s.as_str()));
    if input.is_some() {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => BuildError::ToolMissing(label),
        _ => BuildError::Io(err),
    })?;

    // Drain stdout and stderr on background threads to prevent pipe-buffer
    // deadlock when child output exceeds the OS pipe buffer (typically 64 KiB).
    let stdout_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_buf_clone = Arc::clone(&stdout_buf);
    let stdout_thread = stdout_pipe.map(|mut pipe| {
        thread::spawn(move || {
            let _ = pipe.read_to_end(&mut stdout_buf_clone.lock().unwrap());
        })
    });

    let stderr_buf_clone = Arc::clone(&stderr_buf);
    let stderr_thread = stderr_pipe.map(|mut pipe| {
        thread::spawn(move || {
            let _ = pipe.read_to_end(&mut stderr_buf_clone.lock().unwrap());
        })
    });

    // Write stdin after launching the drain threads so the child can read its
    // input while we drain its output — mutual progress guaranteed.
    if let Some(data) = input {
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(data)?;
            // Drop stdin to signal EOF to the child.
        }
    }

    // Poll for child termination with timeout.
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(BuildError::Timeout(label));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(err) => return Err(BuildError::Io(err)),
        }
    };

    // Join drain threads before reading the buffers.
    if let Some(t) = stdout_thread {
        let _ = t.join();
    }
    if let Some(t) = stderr_thread {
        let _ = t.join();
    }

    let stdout = Arc::try_unwrap(stdout_buf)
        .unwrap_or_default()
        .into_inner()
        .unwrap_or_default();
    let stderr = Arc::try_unwrap(stderr_buf)
        .unwrap_or_default()
        .into_inner()
        .unwrap_or_default();

    Ok(CommandOutput {
        stdout,
        stderr,
        status,
    })
}

#[cfg(feature = "mlir-build")]
fn write_text_file(path: &Path, contents: &str) -> Result<(), BuildError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, contents)?;
    Ok(())
}

#[cfg(feature = "mlir-build")]
fn decode_to_string(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        String::new()
    } else {
        String::from_utf8_lossy(bytes).trim().to_string()
    }
}
