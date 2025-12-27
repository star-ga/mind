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
use std::process::Child;
#[cfg(feature = "mlir-build")]
use std::process::Command;
#[cfg(feature = "mlir-build")]
use std::process::Stdio;
#[cfg(feature = "mlir-build")]
use std::time::Duration;
#[cfg(feature = "mlir-build")]
use std::time::Instant;

#[cfg(feature = "mlir-build")]
use tempfile::NamedTempFile;

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
        run_clang_codegen(&llvm_ir, tools, opts.target_triple, path, false)?;
    }

    if let Some(path) = opts.emit_shared {
        run_clang_codegen(&llvm_ir, tools, opts.target_triple, path, true)?;
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
    match preset {
        "core" | "arith-linalg" | "cpu-demo" | "jit-cpu" | "gpu-default" => {
            Some("canonicalize,cse")
        }
        _ => None,
    }
}

#[cfg(feature = "mlir-build")]
fn run_mlir_opt(input: &str, pipeline: &str, tools: &BuildTools) -> Result<String, BuildError> {
    let args = vec![format!("--pass-pipeline={pipeline}")];
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

#[cfg(feature = "mlir-build")]
fn run_clang_codegen(
    llvm_ir: &str,
    tools: &BuildTools,
    target_triple: Option<&str>,
    output_path: &Path,
    shared: bool,
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

#[cfg(feature = "mlir-build")]
fn run_command(
    program: &str,
    args: &[String],
    input: Option<&[u8]>,
    timeout: Duration,
    label: &'static str,
) -> Result<CommandOutput, BuildError> {
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

    if let Some(data) = input {
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(data)?;
        }
    }

    wait_with_timeout(&mut child, timeout, label)?;
    collect_child_output(child)
}

#[cfg(feature = "mlir-build")]
fn wait_with_timeout(
    child: &mut Child,
    timeout: Duration,
    label: &'static str,
) -> Result<(), BuildError> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
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
    }
}

#[cfg(feature = "mlir-build")]
fn collect_child_output(mut child: Child) -> Result<CommandOutput, BuildError> {
    use std::io::Read;

    let mut stdout = Vec::new();
    if let Some(mut out) = child.stdout.take() {
        out.read_to_end(&mut stdout)?;
    }

    let mut stderr = Vec::new();
    if let Some(mut err) = child.stderr.take() {
        err.read_to_end(&mut stderr)?;
    }

    let status = child.wait()?;

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
