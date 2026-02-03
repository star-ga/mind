// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! Project management for MIND - reads Mind.toml and builds projects.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

/// Project manifest from Mind.toml
#[derive(Debug, Deserialize, Clone)]
pub struct ProjectManifest {
    pub package: PackageInfo,
    #[serde(default)]
    pub build: BuildConfig,
    #[serde(default)]
    pub features: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub targets: HashMap<String, TargetConfig>,
    #[serde(default)]
    pub dependencies: HashMap<String, DependencySpec>,
    #[serde(default)]
    pub profile: HashMap<String, ProfileConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct BuildConfig {
    #[serde(default = "default_entry")]
    pub entry: String,
    #[serde(default = "default_output")]
    pub output: String,
    #[serde(default = "default_optimization")]
    pub optimization: String,
}

fn default_entry() -> String {
    "src/main.mind".to_string()
}

fn default_output() -> String {
    "app".to_string()
}

fn default_optimization() -> String {
    "aggressive".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct TargetConfig {
    pub backend: String,
    #[serde(default)]
    pub simd: Option<String>,
    #[serde(default)]
    pub compute: Option<String>,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum DependencySpec {
    Simple(String),
    Detailed {
        version: String,
        #[serde(default)]
        features: Vec<String>,
    },
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ProfileConfig {
    #[serde(rename = "opt-level", default)]
    pub opt_level: Option<u8>,
    #[serde(default)]
    pub debug: Option<bool>,
    #[serde(default)]
    pub lto: Option<bool>,
    #[serde(rename = "codegen-units", default)]
    pub codegen_units: Option<u8>,
}

/// Build options for project compilation
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    pub release: bool,
    pub target: Option<String>,
    pub verbose: bool,
}

/// Build result
#[derive(Debug)]
pub struct BuildResult {
    pub output_path: PathBuf,
    pub target: String,
    pub success: bool,
}

/// Find the project root by looking for Mind.toml
pub fn find_project_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;
    loop {
        let manifest_path = current.join("Mind.toml");
        if manifest_path.exists() {
            return Ok(current);
        }
        if !current.pop() {
            return Err(anyhow!(
                "Could not find Mind.toml in current directory or any parent"
            ));
        }
    }
}

/// Load project manifest from Mind.toml
pub fn load_manifest(project_root: &Path) -> Result<ProjectManifest> {
    let manifest_path = project_root.join("Mind.toml");
    let content = fs::read_to_string(&manifest_path)
        .with_context(|| format!("Failed to read {}", manifest_path.display()))?;

    let manifest: ProjectManifest = toml::from_str(&content)
        .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;

    Ok(manifest)
}

/// Collect all .mind source files from a project
pub fn collect_sources(project_root: &Path, entry: &str) -> Result<Vec<PathBuf>> {
    let entry_path = project_root.join(entry);
    if !entry_path.exists() {
        return Err(anyhow!("Entry file not found: {}", entry_path.display()));
    }

    let src_dir = entry_path.parent().unwrap_or(project_root);
    let mut sources = Vec::new();

    fn collect_recursive(dir: &Path, sources: &mut Vec<PathBuf>) -> Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    collect_recursive(&path, sources)?;
                } else if path.extension().map(|e| e == "mind").unwrap_or(false) {
                    sources.push(path);
                }
            }
        }
        Ok(())
    }

    collect_recursive(src_dir, &mut sources)?;
    Ok(sources)
}

/// Build a MIND project
pub fn build_project(opts: &BuildOptions) -> Result<BuildResult> {
    let project_root = find_project_root()?;
    let manifest = load_manifest(&project_root)?;

    // Determine target
    let target_name = opts.target.clone().unwrap_or_else(|| "cpu".to_string());
    let target_config = manifest.targets.get(&target_name);

    // Determine output name
    let output_name = if let Some(cfg) = target_config {
        cfg.output
            .clone()
            .unwrap_or_else(|| manifest.build.output.clone())
    } else {
        manifest.build.output.clone()
    };

    // Create target directory
    let profile_dir = if opts.release { "release" } else { "debug" };
    let target_dir = project_root.join("target").join(profile_dir);
    fs::create_dir_all(&target_dir)?;

    let output_path = target_dir.join(&output_name);

    // Collect sources
    let sources = collect_sources(&project_root, &manifest.build.entry)?;

    if opts.verbose {
        println!(
            "Building {} v{}",
            manifest.package.name, manifest.package.version
        );
        println!("  Target: {}", target_name);
        println!("  Profile: {}", profile_dir);
        println!("  Sources: {} files", sources.len());
    }

    // Resolve backend from target config
    let backend = if let Some(cfg) = target_config {
        cfg.backend.clone()
    } else {
        target_name.clone()
    };

    // Build each source file and link
    let compiled = compile_sources(&project_root, &sources, &backend, opts)?;

    // Link into final binary
    link_binary(&compiled, &output_path, &backend, opts)?;

    if opts.verbose {
        println!("  Output: {}", output_path.display());
    }

    Ok(BuildResult {
        output_path,
        target: target_name,
        success: true,
    })
}

/// Compile source files to object files
fn compile_sources(
    project_root: &Path,
    sources: &[PathBuf],
    backend: &str,
    opts: &BuildOptions,
) -> Result<Vec<PathBuf>> {
    let obj_dir = project_root.join("target").join("obj");
    fs::create_dir_all(&obj_dir)?;

    // Determine which file is the entry point
    let manifest = load_manifest(project_root)?;
    let entry_path = project_root.join(&manifest.build.entry);
    let entry_canonical = entry_path.canonicalize().unwrap_or(entry_path.clone());

    let mut objects = Vec::new();

    for source in sources {
        let source_name = source
            .file_stem()
            .ok_or_else(|| anyhow!("Invalid source file: {}", source.display()))?
            .to_string_lossy();

        let obj_path = obj_dir.join(format!("{}.o", source_name));

        if opts.verbose {
            println!("  Compiling: {}", source.display());
        }

        // Check if this is the entry point
        let source_canonical = source.canonicalize().unwrap_or(source.clone());
        let is_entry = source_canonical == entry_canonical;

        // Compile with appropriate mode
        compile_single_source(source, &obj_path, backend, opts, is_entry)?;

        objects.push(obj_path);
    }

    Ok(objects)
}

/// Compile a single source file to native object code
#[allow(clippy::needless_return)]
fn compile_single_source(
    source: &Path,
    output: &Path,
    backend: &str,
    opts: &BuildOptions,
    is_entry: bool, // true if this is the main entry point
) -> Result<()> {
    use crate::eval;
    use crate::parser;
    use crate::pipeline::{compile_source_with_name, CompileOptions};
    use crate::runtime::types::BackendTarget;

    // Read source
    let source_code = fs::read_to_string(source)
        .with_context(|| format!("Failed to read source: {}", source.display()))?;

    // Parse and compile to IR
    let target = match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" | "rocm"
        | "rocm-mi300" | "metal" | "metal-m4" | "webgpu" | "directx" | "oneapi" => {
            BackendTarget::Gpu
        }
        _ => BackendTarget::Cpu,
    };

    let compile_opts = CompileOptions {
        func: None,
        enable_autodiff: false,
        target,
    };

    // Try to compile - if parser doesn't support all syntax, fall back to embedding
    let _products = match compile_source_with_name(
        &source_code,
        Some(&source.to_string_lossy()),
        &compile_opts,
    ) {
        Ok(p) => p,
        Err(_) => {
            // Fall back to embedding source for runtime JIT
            return compile_embedded_source(source, &source_code, output, backend, opts, is_entry);
        }
    };

    // Parse again to get AST for IR lowering
    let module = match parser::parse_with_diagnostics(&source_code) {
        Ok(m) => m,
        Err(_) => {
            return compile_embedded_source(source, &source_code, output, backend, opts, is_entry);
        }
    };

    // Lower AST to IR, then to MLIR
    let ir_module = eval::lower_to_ir(&module);
    let preset_str = if opts.release { "arith-linalg" } else { "core" };
    let mlir_opts = eval::MlirEmitOptions {
        lower_preset: Some(preset_str.to_string()),
        mode: eval::MlirEmitMode::Executable,
    };
    let mlir = eval::emit_mlir_with_opts(&ir_module, &mlir_opts);

    // Use mlir-build if available
    #[cfg(feature = "mlir-build")]
    {
        use crate::eval::mlir_build;

        match mlir_build::resolve_tools() {
            Ok(tools) => {
                let build_opts = mlir_build::BuildOptions {
                    preset: if opts.release { "arith-linalg" } else { "core" },
                    emit_mlir_file: None,
                    emit_llvm_file: None,
                    emit_obj_file: Some(output),
                    emit_shared: None,
                    opt_pipeline: if opts.release {
                        Some("canonicalize,cse,loop-invariant-code-motion")
                    } else {
                        None
                    },
                    target_triple: Some(get_target_triple(backend)),
                };

                mlir_build::build_all(&mlir, &tools, &build_opts)
                    .map_err(|e| anyhow!("MLIR build failed: {}", e))?;

                return Ok(());
            }
            Err(_) => {
                // MLIR tools not available, fall back to embedded source
                return compile_embedded_source(
                    source,
                    &source_code,
                    output,
                    backend,
                    opts,
                    is_entry,
                );
            }
        }
    }

    #[cfg(not(feature = "mlir-build"))]
    {
        let _ = mlir; // Suppress unused variable warning
        compile_embedded_source(source, &source_code, output, backend, opts, is_entry)
    }
}

/// Compile source by embedding it for runtime JIT
fn compile_embedded_source(
    source: &Path,
    source_code: &str,
    output: &Path,
    backend: &str,
    _opts: &BuildOptions,
    is_entry: bool, // true if this is the main entry point
) -> Result<()> {
    // Create a C file that embeds the MIND source
    let escaped_source = source_code
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");

    let module_name = source
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .replace('-', "_");

    let c_code = if is_entry {
        // Entry point: include main() that calls mind_main
        format!(
            r#"
/* Auto-generated MIND entry point wrapper */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

static const char MIND_SOURCE_{module}[] = "{source}";
static const char MIND_BACKEND[] = "{backend}";
static const char MIND_FILE[] = "{file}";

/* Forward declarations for other modules */
extern const char* mind_get_module_source(const char* name);

typedef int (*mind_main_fn)(int argc, char** argv, const char* source, const char* backend);

int main(int argc, char** argv) {{
    void* lib = dlopen("libmind_{backend}_linux-x64.so", RTLD_NOW);
    if (!lib) {{
        lib = dlopen("libmind_cpu_linux-x64.so", RTLD_NOW);
    }}
    if (!lib) {{
        fprintf(stderr, "Error: MIND runtime not found\\n");
        fprintf(stderr, "Run: curl -fsSL https://nikolachess.com/install.sh | bash\\n");
        return 1;
    }}

    mind_main_fn mind_main_ptr = (mind_main_fn)dlsym(lib, "mind_main");
    if (!mind_main_ptr) {{
        fprintf(stderr, "Error: mind_main not found in runtime\\n");
        dlclose(lib);
        return 1;
    }}

    int result = mind_main_ptr(argc, argv, MIND_SOURCE_{module}, MIND_BACKEND);
    dlclose(lib);
    return result;
}}
"#,
            module = module_name,
            source = escaped_source,
            backend = backend,
            file = source.file_name().unwrap_or_default().to_string_lossy(),
        )
    } else {
        // Non-entry module: just export the source
        format!(
            r#"
/* Auto-generated MIND module: {file} */
static const char MIND_MODULE_{module}_SOURCE[] = "{source}";

const char* mind_module_{module}_get_source(void) {{
    return MIND_MODULE_{module}_SOURCE;
}}
"#,
            module = module_name,
            source = escaped_source,
            file = source.file_name().unwrap_or_default().to_string_lossy(),
        )
    };

    // Write C file and compile with cc
    let c_path = output.with_extension("c");
    fs::write(&c_path, &c_code)?;

    let status = Command::new("cc")
        .args(["-c", "-fPIC", "-O2"])
        .arg(&c_path)
        .arg("-o")
        .arg(output)
        .status()
        .with_context(|| "Failed to run C compiler")?;

    if !status.success() {
        return Err(anyhow!("C compilation failed for {}", source.display()));
    }

    // Clean up intermediate C file
    let _ = fs::remove_file(&c_path);

    Ok(())
}

/// Get the LLVM target triple for a backend
#[allow(dead_code)]
fn get_target_triple(backend: &str) -> &'static str {
    match backend {
        // GPU backends use host triple for the launcher
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" | "rocm"
        | "rocm-mi300" | "webgpu" | "directx" | "oneapi" => {
            #[cfg(target_os = "linux")]
            {
                "x86_64-unknown-linux-gnu"
            }
            #[cfg(target_os = "macos")]
            {
                "aarch64-apple-darwin"
            }
            #[cfg(target_os = "windows")]
            {
                "x86_64-pc-windows-msvc"
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                "x86_64-unknown-linux-gnu"
            }
        }
        "metal" | "metal-m4" => "aarch64-apple-darwin",
        _ => {
            #[cfg(target_os = "linux")]
            {
                "x86_64-unknown-linux-gnu"
            }
            #[cfg(target_os = "macos")]
            {
                "aarch64-apple-darwin"
            }
            #[cfg(target_os = "windows")]
            {
                "x86_64-pc-windows-msvc"
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                "x86_64-unknown-linux-gnu"
            }
        }
    }
}

/// Link object files into a native executable binary
fn link_binary(
    objects: &[PathBuf],
    output: &Path,
    backend: &str,
    opts: &BuildOptions,
) -> Result<()> {
    // Check for runtime library
    let lib_dir = find_runtime_lib(backend)?;

    if opts.verbose {
        println!("  Linking with runtime: {}", lib_dir.display());
    }

    // Determine runtime library name
    let (_runtime_lib, runtime_link) = get_runtime_lib_names(backend);

    // Try native linking first
    let link_result = native_link(objects, output, &lib_dir, runtime_link, opts);

    if link_result.is_ok() {
        return Ok(());
    }

    // Fallback to wrapper script if native linking fails
    if opts.verbose {
        println!("  Native linking failed, creating launcher script");
    }

    let project_root = find_project_root()?;
    let manifest = load_manifest(&project_root)?;
    let entry_path = project_root.join(&manifest.build.entry);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let script = format!(
            r#"#!/bin/bash
# {name} v{version} - Built with MIND
# Backend: {backend}
# Runtime: {runtime}

MIND_ROOT="{project_root}"
MIND_ENTRY="{entry}"
MIND_BACKEND="{backend}"
MIND_LIB_DIR="{lib_dir}"

export LD_LIBRARY_PATH="$MIND_LIB_DIR:$LD_LIBRARY_PATH"
export MIND_SOURCE_ROOT="$MIND_ROOT/src"
export MIND_BACKEND="$MIND_BACKEND"

# Execute the runtime interpreter
if [ -f "$MIND_LIB_DIR/mind-runtime" ]; then
    exec "$MIND_LIB_DIR/mind-runtime" --backend "$MIND_BACKEND" --entry "$MIND_ENTRY" "$@"
elif [ -f "$MIND_LIB_DIR/{runtime}" ]; then
    # Direct library execution via ld.so
    exec /lib64/ld-linux-x86-64.so.2 --library-path "$MIND_LIB_DIR" "$MIND_LIB_DIR/{runtime}" --entry "$MIND_ENTRY" "$@" 2>/dev/null || \
    exec /lib/ld-linux-x86-64.so.2 --library-path "$MIND_LIB_DIR" "$MIND_LIB_DIR/{runtime}" --entry "$MIND_ENTRY" "$@" 2>/dev/null || \
    {{
        echo "Error: MIND runtime not executable"
        echo "Run: curl -fsSL https://nikolachess.com/install.sh | bash"
        exit 1
    }}
else
    echo "Error: MIND runtime not found at $MIND_LIB_DIR/{runtime}"
    echo "Run: curl -fsSL https://nikolachess.com/install.sh | bash"
    exit 1
fi
"#,
            name = manifest.package.name,
            version = manifest.package.version,
            backend = backend,
            runtime = runtime_lib,
            project_root = project_root.display(),
            entry = entry_path.display(),
            lib_dir = lib_dir.display(),
        );

        fs::write(output, script)?;

        let mut perms = fs::metadata(output)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(output, perms)?;
    }

    #[cfg(not(unix))]
    {
        let script = format!(
            "@echo off\r\nrem {} v{} - Built with MIND\r\nset MIND_BACKEND={}\r\nset PATH=%PATH%;{}\r\n\"{}/mind-runtime.exe\" --backend {} --entry \"{}\" %*\r\n",
            manifest.package.name,
            manifest.package.version,
            backend,
            lib_dir.display(),
            lib_dir.display(),
            backend,
            entry_path.display(),
        );
        fs::write(output, script)?;
    }

    Ok(())
}

/// Get runtime library names for a backend
fn get_runtime_lib_names(backend: &str) -> (&'static str, &'static str) {
    match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_cuda_linux-x64.so", "mind_cuda_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_cuda_windows-x64.dll", "mind_cuda_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_cuda_linux-x64.so", "mind_cuda_linux-x64")
            }
        }
        "rocm" | "rocm-mi300" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_rocm_linux-x64.so", "mind_rocm_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_rocm_windows-x64.dll", "mind_rocm_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_rocm_linux-x64.so", "mind_rocm_linux-x64")
            }
        }
        "metal" | "metal-m4" => ("libmind_metal_macos-arm64.dylib", "mind_metal_macos-arm64"),
        "webgpu" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_webgpu_linux-x64.so", "mind_webgpu_linux-x64")
            }
            #[cfg(target_os = "macos")]
            {
                (
                    "libmind_webgpu_macos-arm64.dylib",
                    "mind_webgpu_macos-arm64",
                )
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_webgpu_windows-x64.dll", "mind_webgpu_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                ("libmind_webgpu_linux-x64.so", "mind_webgpu_linux-x64")
            }
        }
        "directx" => ("mind_directx_windows-x64.dll", "mind_directx_windows-x64"),
        "oneapi" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_oneapi_linux-x64.so", "mind_oneapi_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_oneapi_windows-x64.dll", "mind_oneapi_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_oneapi_linux-x64.so", "mind_oneapi_linux-x64")
            }
        }
        _ => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_cpu_linux-x64.so", "mind_cpu_linux-x64")
            }
            #[cfg(target_os = "macos")]
            {
                ("libmind_cpu_macos-arm64.dylib", "mind_cpu_macos-arm64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_cpu_windows-x64.dll", "mind_cpu_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                ("libmind_cpu_linux-x64.so", "mind_cpu_linux-x64")
            }
        }
    }
}

/// Attempt native linking with clang/gcc
fn native_link(
    objects: &[PathBuf],
    output: &Path,
    lib_dir: &Path,
    runtime_link: &str,
    opts: &BuildOptions,
) -> Result<()> {
    // Find a C compiler for linking
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());

    let mut cmd = Command::new(&cc);

    // Add object files
    for obj in objects {
        cmd.arg(obj);
    }

    // Output path
    cmd.arg("-o").arg(output);

    // Link against runtime library
    cmd.arg(format!("-L{}", lib_dir.display()));
    cmd.arg(format!("-l{}", runtime_link));

    // Link standard libraries
    cmd.arg("-ldl"); // For dlopen on Linux
    cmd.arg("-lpthread");
    cmd.arg("-lm");

    // Add rpath for runtime library lookup
    #[cfg(target_os = "linux")]
    {
        cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));
        cmd.arg("-Wl,-rpath,$ORIGIN/../lib");
    }

    #[cfg(target_os = "macos")]
    {
        cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));
        cmd.arg("-Wl,-rpath,@executable_path/../lib");
    }

    // Optimization flags
    if opts.release {
        cmd.arg("-O3");
        cmd.arg("-flto");
    }

    if opts.verbose {
        println!("  Link command: {:?}", cmd);
    }

    let status = cmd
        .status()
        .with_context(|| format!("Failed to run linker: {}", cc))?;

    if !status.success() {
        return Err(anyhow!(
            "Linking failed with exit code: {:?}",
            status.code()
        ));
    }

    Ok(())
}

/// Find the MIND runtime library for a backend
fn find_runtime_lib(backend: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Cannot determine home directory"))?;
    let nikolachess_lib = home.join(".nikolachess").join("lib");
    let mind_lib = home.join(".mind").join("lib");

    // Map backend to library name
    let lib_name = match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" => {
            "libmind_cuda_linux-x64.so"
        }
        "rocm" | "rocm-mi300" => "libmind_rocm_linux-x64.so",
        "metal" | "metal-m4" => "libmind_metal_macos-arm64.dylib",
        "webgpu" => "libmind_webgpu_linux-x64.so",
        _ => "libmind_cpu_linux-x64.so",
    };

    // Search in known locations
    for lib_dir in [&nikolachess_lib, &mind_lib] {
        let lib_path = lib_dir.join(lib_name);
        if lib_path.exists() {
            return Ok(lib_dir.clone());
        }
    }

    // Fallback: check if any lib exists
    if nikolachess_lib.exists() {
        return Ok(nikolachess_lib);
    }
    if mind_lib.exists() {
        return Ok(mind_lib);
    }

    Err(anyhow!(
        "MIND runtime not found for backend '{}'. Run the install script first.",
        backend
    ))
}

/// Run a built project
pub fn run_project(args: &[String], opts: &BuildOptions) -> Result<i32> {
    // First build
    let result = build_project(opts)?;

    if !result.success {
        return Err(anyhow!("Build failed"));
    }

    // Then run
    let status = Command::new(&result.output_path)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to execute {}", result.output_path.display()))?;

    Ok(status.code().unwrap_or(1))
}
