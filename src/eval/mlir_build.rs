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

    // ---------------------------------------------------------------------
    // Disk cache (perf only — output-neutral by construction).
    //
    // The emitted .o is a pure function of: (1) the embedded C source bytes,
    // (2) the clang argument vector, (3) the clang binary itself, and
    // (4) the subset of the environment clang honours. The cache key mixes
    // ALL FOUR:
    //   * the source bytes;
    //   * every literal arg EXCEPT the two random per-build path args (the
    //     source path input and the `-o` output path) — these vary per
    //     invocation but never affect the object bytes thanks to the fixed
    //     `mind_intrinsics.c` basename + `-ffile-prefix-map` determinism
    //     contract above (and the C source carries no `__FILE__`/`__DATE__`
    //     /`__TIME__`/`__COUNTER__`);
    //   * the clang *binary identity* — resolved path + size + mtime + the
    //     `--version` banner (NOT the banner alone: the cache dir is
    //     machine-global, so a same-banner `CLANG=` swap or in-place rebuild
    //     must still miss);
    //   * the clang-honoured env vars that can change codegen
    //     (`SOURCE_DATE_EPOCH`, `CPATH`, include-path vars, …).
    // Any change to any of these yields a different key, so a stale/foreign
    // .o is never reused. The cached bytes are byte-identical to what the
    // non-cached clang invocation below produces.
    // ---------------------------------------------------------------------
    let cache_path = runtime_obj_cache_path(tools, &args);
    if let Some(ref cached) = cache_path {
        if let Ok(bytes) = std::fs::read(cached) {
            std::fs::write(obj_tmp.path(), &bytes)?;
            drop(src_dir);
            return Ok(obj_tmp);
        }
    }

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

    // Populate the cache (best-effort, atomic temp-then-rename). A cache
    // write failure never fails the build — the freshly compiled obj_tmp is
    // already valid.
    if let Some(ref cached) = cache_path {
        if let Ok(obj_bytes) = std::fs::read(obj_tmp.path()) {
            let _ = write_runtime_obj_cache(cached, &obj_bytes);
        }
    }

    Ok(obj_tmp)
}

/// Environment variables clang reads that can alter emitted object bytes.
/// Folded into the runtime-obj cache key so an env change never yields a
/// false cache hit (the key is conservative: a different value — or set vs
/// unset — produces a different key, never a wrong reuse).
#[cfg(feature = "mlir-build")]
const CLANG_CACHE_ENV_VARS: &[&str] = &[
    "SOURCE_DATE_EPOCH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "CCC_OVERRIDE_OPTIONS",
    "COMPILER_PATH",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
];

/// Compute the on-disk cache path for the runtime-support `.o`, or `None`
/// if no cache directory is available.
///
/// Key = sha256(version-prefix ++ clang binary identity ++ clang-honoured
/// env ++ canonical arg vector ++ embedded C source bytes). The two random
/// per-build path args (the source-file input and the `-o` output) are
/// excluded — they vary per invocation but cannot change the emitted object
/// bytes under the fixed-basename + `-ffile-prefix-map` determinism contract.
/// The cache directory is machine-global, so the key pins the clang *binary*
/// (path + size + mtime + version), not merely its `--version` banner.
#[cfg(feature = "mlir-build")]
fn runtime_obj_cache_path(tools: &BuildTools, args: &[String]) -> Option<std::path::PathBuf> {
    let cache_dir = dirs::cache_dir()?.join("mind").join("runtime-obj");

    let clang_identity = clang_identity_string(&tools.clang);

    let mut data: Vec<u8> = Vec::with_capacity(512 + MIND_RUNTIME_SUPPORT_C.len());
    data.extend_from_slice(b"mind-runtime-obj-v2\n");
    data.extend_from_slice(b"clang-identity=\n");
    data.extend_from_slice(clang_identity.as_bytes());
    data.push(b'\n');
    data.extend_from_slice(b"env=\n");
    for var in CLANG_CACHE_ENV_VARS {
        data.extend_from_slice(var.as_bytes());
        data.push(b'=');
        if let Ok(v) = std::env::var(var) {
            data.extend_from_slice(v.as_bytes());
        }
        data.push(b'\n');
    }
    data.extend_from_slice(b"args=\n");
    // Exclude the two random per-build path args (the source input that
    // follows `-x c`, and the `-o` output path). They never affect the
    // object bytes but would otherwise defeat the cache.
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "-o" {
            // Skip "-o" and its operand.
            i += 2;
            continue;
        }
        if a == "-x" {
            // Keep "-x c" (the mode), but the source path is the arg
            // immediately after "c"; the prefix-map arg also embeds the
            // random dir, so normalise it out below.
            data.extend_from_slice(a.as_bytes());
            data.push(b'\n');
            if i + 1 < args.len() {
                data.extend_from_slice(args[i + 1].as_bytes());
                data.push(b'\n');
            }
            // Skip the source-path operand that follows "-x c".
            i += 3;
            continue;
        }
        if let Some(stripped) = a.strip_prefix("-ffile-prefix-map=") {
            // The map RHS (`=.`) is what lands in the object; the LHS is the
            // random tempdir. Record only the stable RHS so the key is
            // path-independent yet flag-sensitive.
            data.extend_from_slice(b"-ffile-prefix-map=");
            if let Some(rhs) = stripped.split_once('=').map(|(_, r)| r) {
                data.extend_from_slice(rhs.as_bytes());
            }
            data.push(b'\n');
            i += 1;
            continue;
        }
        data.extend_from_slice(a.as_bytes());
        data.push(b'\n');
        i += 1;
    }
    data.extend_from_slice(b"source=\n");
    data.extend_from_slice(MIND_RUNTIME_SUPPORT_C.as_bytes());

    let key = crate::build::cache::sha256_hex(&data);
    Some(cache_dir.join(format!("{key}.o")))
}

/// Atomically write the cached runtime `.o` (temp file in the same dir +
/// rename, so a concurrent reader never observes a partial write).
#[cfg(feature = "mlir-build")]
fn write_runtime_obj_cache(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = NamedTempFile::new_in(dest.parent().unwrap_or_else(|| Path::new(".")))?;
    tmp.write_all(bytes)?;
    tmp.flush()?;
    // `persist` performs the rename atomically on POSIX.
    tmp.persist(dest).map_err(|e| e.error)?;
    Ok(())
}

/// Return clang's `--version` output as a single normalised string, or a
/// stable sentinel if it cannot be queried. Used to invalidate the runtime
/// `.o` cache (and the module cache) when the toolchain changes.
#[cfg(feature = "mlir-build")]
pub fn clang_version_string(clang: &str) -> String {
    match Command::new(clang).arg("--version").output() {
        Ok(out) if out.status.success() => decode_to_string(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

/// Return a stable *binary-identity* string for clang: resolved path, file
/// size, mtime (nanos since epoch) and the `--version` banner, joined by
/// `|`. Folded into the runtime-obj cache key so a different clang binary —
/// even one reporting an identical `--version` (a vendor/distro rebuild, an
/// in-place upgrade, or a `CLANG=` override to a same-banner binary) — yields
/// a different key and is never reused from the machine-global cache. The
/// `--version` banner alone is NOT a content hash of the compiler, so the
/// path+size+mtime triple guards the realistic same-banner-swap cases; a
/// binary deliberately forged to identical path+size+mtime yet different
/// codegen is an out-of-scope attack, not a determinism failure mode.
#[cfg(feature = "mlir-build")]
pub fn clang_identity_string(clang: &str) -> String {
    let version = clang_version_string(clang);
    // clang may be a bare command name resolved via PATH; resolve to the
    // real binary so the identity is path-stable.
    let resolved = which::which(clang)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| clang.to_string());
    let (size, mtime_ns) = std::fs::metadata(&resolved)
        .map(|m| {
            let mtime_ns = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            (m.len(), mtime_ns)
        })
        .unwrap_or((0, 0));
    format!("{resolved}|{size}|{mtime_ns}|{version}")
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
                std::thread::sleep(Duration::from_millis(1));
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
