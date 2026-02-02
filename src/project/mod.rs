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
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub release: bool,
    pub target: Option<String>,
    pub verbose: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            release: false,
            target: None,
            verbose: false,
        }
    }
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
            return Err(anyhow!("Could not find Mind.toml in current directory or any parent"));
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
        cfg.output.clone().unwrap_or_else(|| manifest.build.output.clone())
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
        println!("Building {} v{}", manifest.package.name, manifest.package.version);
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

    let mut objects = Vec::new();

    for source in sources {
        let source_name = source.file_stem()
            .ok_or_else(|| anyhow!("Invalid source file: {}", source.display()))?
            .to_string_lossy();

        let obj_path = obj_dir.join(format!("{}.o", source_name));

        if opts.verbose {
            println!("  Compiling: {}", source.display());
        }

        // For now, we'll create a stub object file
        // In a full implementation, this would invoke the MIND compiler pipeline
        compile_single_source(source, &obj_path, backend, opts)?;

        objects.push(obj_path);
    }

    Ok(objects)
}

/// Compile a single source file
fn compile_single_source(
    source: &Path,
    output: &Path,
    backend: &str,
    opts: &BuildOptions,
) -> Result<()> {
    // Read source
    let source_code = fs::read_to_string(source)
        .with_context(|| format!("Failed to read source: {}", source.display()))?;

    // For CUDA/GPU targets, we need to generate appropriate code
    // This is a simplified implementation - in production this would use
    // the full MIND compiler pipeline

    let _ = (&source_code, backend, opts);

    // Create a placeholder object file for now
    // Real implementation would generate actual machine code
    fs::write(output, b"MIND_OBJ\0")?;

    Ok(())
}

/// Link object files into a binary
fn link_binary(
    _objects: &[PathBuf],
    output: &Path,
    backend: &str,
    opts: &BuildOptions,
) -> Result<()> {
    // Check for runtime library
    let lib_dir = find_runtime_lib(backend)?;

    if opts.verbose {
        println!("  Linking with runtime: {}", lib_dir.display());
    }

    // For now, create a stub executable
    // Real implementation would invoke the linker

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        // Create a shell script wrapper that loads the runtime and executes
        let script = format!(
            "#!/bin/bash\n# MIND {} Binary\nexec \"$0.real\" \"$@\"\n",
            backend
        );
        fs::write(output, script)?;

        let mut perms = fs::metadata(output)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(output, perms)?;
    }

    #[cfg(not(unix))]
    {
        fs::write(output, b"MIND_EXE")?;
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
