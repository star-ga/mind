// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase A — `mindc build` single-crate orchestrator.
//!
//! Public entry point: [`run_build`].
//!
//! This module drives the existing pure-MIND compile pipeline
//! (`compile_sources` / `link_binary` in `src/project/mod.rs`) through a
//! typed options layer that replaces the unstructured flags the legacy
//! `build_project` function received. Phases B–G (test runner, workspace,
//! path deps, git deps, incremental cache, bootstrap) are not implemented
//! here; this is the single-crate, no-deps foundation.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::project::{
    build_project, find_project_root, load_manifest, BuildOptions as LegacyBuildOptions,
    BuildTarget, EmitKind, OptimizeLevel,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// All options that govern a `mindc build` invocation.
///
/// CLI flags shadow `Mind.toml [build]` fields via `Option<T>`: `None` means
/// "use the manifest default"; `Some(x)` means "override the manifest with x".
#[derive(Debug, Clone, Default)]
pub struct BuildOpts {
    /// Positional source paths from the CLI. When empty, source resolution
    /// falls back to the manifest `[build].entry` field.
    pub paths: Vec<PathBuf>,
    /// Override `[build].target`.
    pub target: Option<BuildTarget>,
    /// Override `[build].emit`.
    pub emit: Option<EmitKind>,
    /// Override `[build].optimize`.
    pub optimize: Option<OptimizeLevel>,
    /// Custom output path (`--out`). Overrides the default
    /// `target/<profile>/<name>` layout.
    pub out: Option<PathBuf>,
    /// Print each compile + link invocation to stderr.
    pub verbose: bool,
}

/// Successful build result returned by [`run_build`].
#[derive(Debug)]
pub struct BuildOutput {
    /// Absolute path to the produced artifact.
    pub artifact_path: PathBuf,
    /// Resolved target string (e.g. `"cpu"`).
    pub target: String,
    /// Resolved emit kind.
    pub emit: EmitKind,
    /// Artifact size in bytes.
    pub byte_count: u64,
}

/// Typed errors from the build orchestrator.
///
/// `Invalid` maps to exit code 2 (bad usage / bad manifest).
/// `Failed`  maps to exit code 1 (compile / link error).
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Failed(String),
}

impl BuildError {
    /// Suggested process exit code per RFC 0008 §6.
    pub fn exit_code(&self) -> i32 {
        match self {
            BuildError::Invalid(_) => 2,
            BuildError::Failed(_) => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Build a single-crate MIND project.
///
/// Loads `Mind.toml`, resolves the source list, drives the existing compile
/// pipeline, and returns the path to the produced artifact.
///
/// When explicit source file paths are provided and no `Mind.toml` can be
/// located, a synthetic single-file manifest is synthesised so that one-off
/// `mindc build <file.mind>` invocations work without a project setup.
///
/// # Exit code semantics
/// The caller should call `BuildError::exit_code()` when propagating errors
/// to `process::exit`.
pub fn run_build(opts: &BuildOpts) -> Result<BuildOutput, BuildError> {
    use crate::project::ProjectManifest;

    // 1. Locate the project root and load the manifest.
    //    When explicit source paths are given and no Mind.toml is found,
    //    synthesise a minimal manifest so single-file builds work without
    //    any project scaffolding (e.g. `mindc build foo.mind --emit=cdylib`).
    let (project_root, manifest) = match find_project_root() {
        Ok(root) => {
            let m = load_manifest(&root)
                .map_err(|e| BuildError::Invalid(format!("manifest error: {e}")))?;
            (root, m)
        }
        Err(_) if !opts.paths.is_empty() => {
            // No Mind.toml; derive project root from cwd so that relative
            // path resolution in resolve_entry remains correct.
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let first_path = if opts.paths[0].is_absolute() {
                opts.paths[0].clone()
            } else {
                cwd.join(&opts.paths[0])
            };
            // Use cwd as the root so relative path joins work correctly.
            let root = cwd;
            let stem = first_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .replace('-', "_");
            // Safe package name: must start with a letter.
            let pkg_name = if stem.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false) {
                stem
            } else {
                format!("pkg_{}", stem)
            };
            let toml_src = format!(
                "[package]\nname = \"{}\"\nversion = \"0.1.0\"\n",
                pkg_name
            );
            let m: ProjectManifest = toml::from_str(&toml_src)
                .map_err(|e| BuildError::Invalid(format!("synthetic manifest: {e}")))?;
            (root, m)
        }
        Err(e) => return Err(BuildError::Invalid(format!("cannot locate Mind.toml: {e}"))),
    };

    // Validate [package].name per RFC 0008 §3.
    validate_package_name(&manifest.package.name)?;

    // 2. Resolve effective build parameters (CLI > manifest > default).
    let eff_target = opts
        .target
        .unwrap_or(manifest.build.target);
    let eff_emit = opts
        .emit
        .unwrap_or(manifest.build.emit);
    let eff_optimize = opts
        .optimize
        .unwrap_or(manifest.build.optimize);

    // 3. Reject targets that have no backend implementation yet.
    validate_target(eff_target)?;

    // 4. Resolve the entry / source file(s).
    //
    // RFC 0008 §4 Phase A: single-crate only; no `[dependencies]` traversal.
    // Source resolution follows the spec priority order:
    //   a) positional PATHS from CLI
    //   b) [build].entry from manifest (may be the default "src/main.mind")
    //   c) auto-detect src/main.mind → binary, src/lib.mind → cdylib
    let entry_path = resolve_entry(opts, &project_root, &manifest.build.entry, eff_emit)?;

    // 5. Build the output path.
    let artifact_path = match &opts.out {
        Some(p) => p.clone(),
        None => default_artifact_path(
            &project_root,
            &manifest.package.name,
            eff_emit,
            eff_optimize,
        ),
    };

    // Ensure the parent directory exists.
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))
            .map_err(|e| BuildError::Failed(e.to_string()))?;
    }

    if opts.verbose {
        eprintln!(
            "   Compiling {} v{} ({})",
            manifest.package.name,
            manifest.package.version,
            project_root.display()
        );
        eprintln!("   Target:   {}", eff_target.as_str());
        eprintln!("   Emit:     {}", eff_emit.as_str());
        eprintln!("   Optimize: {}", eff_optimize.as_str());
        eprintln!("   Entry:    {}", entry_path.display());
        eprintln!("   Out:      {}", artifact_path.display());
    }

    // 6. Drive the existing pipeline via the legacy `build_project` function.
    //    RFC 0008 §4.5: shared-library linking is handled by passing
    //    `emit_shared` through `LegacyBuildOptions`.
    let legacy_opts = legacy_opts_from(
        eff_target,
        eff_emit,
        eff_optimize,
        &manifest.exports.c_abi,
        &entry_path,
        &artifact_path,
        opts.verbose,
    );

    // The legacy `build_project` reads `Mind.toml` from the project root.
    // We ensure a correctly-configured manifest exists, patching it with the
    // resolved entry path if needed.  The original is always restored.
    let manifest_path = project_root.join("Mind.toml");
    let manifest_existed = manifest_path.exists();

    let entry_rel = entry_path
        .strip_prefix(&project_root)
        .unwrap_or(&entry_path)
        .to_string_lossy()
        .replace('\\', "/");

    let orig_manifest_text: Option<String> = if manifest_existed {
        Some(
            fs::read_to_string(&manifest_path)
                .map_err(|e| BuildError::Failed(format!("cannot read manifest: {e}")))?,
        )
    } else {
        None
    };

    // Decide whether to write/patch the on-disk manifest.
    let need_write = match &orig_manifest_text {
        Some(text) => {
            // Manifest exists; only rewrite if the entry needs updating.
            entry_rel != manifest.build.entry
                || text.find("entry = ").is_none()
        }
        None => true, // No manifest on disk; write a synthetic one.
    };

    if need_write {
        let toml_to_write = match &orig_manifest_text {
            Some(text) => patch_manifest_entry(text, &entry_rel),
            None => build_synthetic_manifest(&manifest.package.name, &entry_rel),
        };
        fs::write(&manifest_path, &toml_to_write)
            .map_err(|e| BuildError::Failed(format!("cannot write manifest: {e}")))?;
    }

    // Run the build, then unconditionally restore the manifest state.
    let build_result = build_project(&legacy_opts);

    if need_write {
        match &orig_manifest_text {
            Some(text) => {
                let _ = fs::write(&manifest_path, text);
            }
            None => {
                // We created a synthetic manifest; remove it.
                let _ = fs::remove_file(&manifest_path);
            }
        }
    }

    let build_result = build_result
        .map_err(|e| BuildError::Failed(format!("{e}")))?;

    // Move/rename the legacy output to the requested artifact_path if needed.
    let final_path = if artifact_path != build_result.output_path {
        // Legacy produced a different path; move it.
        if let Some(parent) = artifact_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::rename(&build_result.output_path, &artifact_path)
            .or_else(|_| fs::copy(&build_result.output_path, &artifact_path).map(|_| ()))
            .map_err(|e| BuildError::Failed(format!("cannot move artifact: {e}")))?;
        artifact_path.clone()
    } else {
        artifact_path.clone()
    };

    // Handle shared-library emit separately when mlir-build is unavailable and
    // the legacy path produced a binary instead.
    let final_path = match eff_emit {
        EmitKind::Cdylib => ensure_cdylib_extension(final_path),
        EmitKind::Object => ensure_object_extension(final_path),
        EmitKind::Binary => final_path,
    };

    let byte_count = fs::metadata(&final_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(BuildOutput {
        artifact_path: final_path,
        target: eff_target.as_str().to_string(),
        emit: eff_emit,
        byte_count,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_package_name(name: &str) -> Result<(), BuildError> {
    let valid = name
        .chars()
        .enumerate()
        .all(|(i, c)| {
            if i == 0 {
                c.is_ascii_alphabetic()
            } else {
                c.is_ascii_alphanumeric() || c == '_' || c == '-'
            }
        });
    if name.is_empty() || !valid {
        return Err(BuildError::Invalid(format!(
            "invalid package name '{}': must match [a-zA-Z][a-zA-Z0-9_-]*",
            name
        )));
    }
    Ok(())
}

fn validate_target(target: BuildTarget) -> Result<(), BuildError> {
    match target {
        BuildTarget::Cpu => Ok(()),
        BuildTarget::Cerebras => Ok(()),
        BuildTarget::Gpu
        | BuildTarget::Tpu
        | BuildTarget::Npu
        | BuildTarget::Lpu
        | BuildTarget::Dpu
        | BuildTarget::Fpga
        | BuildTarget::Wasm => Err(BuildError::Invalid(format!(
            "target '{}' is not yet supported in Phase A (only cpu and cerebras are available)",
            target.as_str()
        ))),
    }
}

fn resolve_entry(
    opts: &BuildOpts,
    project_root: &Path,
    manifest_entry: &str,
    eff_emit: EmitKind,
) -> Result<PathBuf, BuildError> {
    // a) Explicit CLI paths take priority.
    if let Some(first) = opts.paths.first() {
        let p = if first.is_absolute() {
            first.clone()
        } else {
            project_root.join(first)
        };
        if !p.exists() {
            return Err(BuildError::Failed(format!(
                "source file not found: {}",
                p.display()
            )));
        }
        return Ok(p);
    }

    // b) Manifest [build].entry (which defaults to "src/main.mind").
    let manifest_path = project_root.join(manifest_entry);
    if manifest_path.exists() {
        return Ok(manifest_path);
    }

    // c) Auto-detect src/main.mind or src/lib.mind.
    let main_mind = project_root.join("src/main.mind");
    if main_mind.exists() {
        return Ok(main_mind);
    }

    let lib_mind = project_root.join("src/lib.mind");
    if lib_mind.exists() {
        if eff_emit != EmitKind::Cdylib {
            return Err(BuildError::Invalid(
                "found src/lib.mind but emit is not 'cdylib'; pass --emit=cdylib or set [build] emit = \"cdylib\" in Mind.toml".to_string()
            ));
        }
        return Ok(lib_mind);
    }

    Err(BuildError::Failed(
        "no source files provided and no src/main.mind or src/lib.mind found".to_string(),
    ))
}

/// Default artifact path mirrors cargo's convention so both can coexist.
fn default_artifact_path(
    project_root: &Path,
    package_name: &str,
    emit: EmitKind,
    optimize: OptimizeLevel,
) -> PathBuf {
    let profile_dir = if optimize.is_release() { "release" } else { "debug" };
    let base = project_root.join("target").join(profile_dir);
    match emit {
        EmitKind::Binary => base.join(package_name),
        EmitKind::Cdylib => {
            #[cfg(target_os = "windows")]
            let name = format!("{}.dll", package_name);
            #[cfg(not(target_os = "windows"))]
            let name = format!("lib{}.so", package_name);
            base.join(name)
        }
        EmitKind::Object => base.join(format!("{}.o", package_name)),
    }
}

/// Build the `LegacyBuildOptions` used to call the existing `build_project`.
fn legacy_opts_from(
    target: BuildTarget,
    emit: EmitKind,
    optimize: OptimizeLevel,
    manifest_exports: &[String],
    _entry_path: &Path,
    _artifact_path: &Path,
    verbose: bool,
) -> LegacyBuildOptions {
    let target_str = match target {
        BuildTarget::Cpu => None,
        other => Some(other.as_str().to_string()),
    };

    LegacyBuildOptions {
        release: optimize.is_release(),
        target: target_str,
        verbose,
        manifest_exports: if emit == EmitKind::Cdylib {
            manifest_exports.to_vec()
        } else {
            Vec::new()
        },
    }
}

/// Build a minimal `Mind.toml` text for a single-file build that has no
/// existing project manifest.
fn build_synthetic_manifest(pkg_name: &str, entry: &str) -> String {
    format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"{}\"\n",
        pkg_name, entry
    )
}

/// Patch a raw `Mind.toml` text to change `entry = "..."`.
/// If no `entry =` line is present, appends the field under `[build]`.
fn patch_manifest_entry(toml_text: &str, new_entry: &str) -> String {
    // Try to replace an existing `entry = "..."` line.
    let entry_pattern = "entry = \"";
    if let Some(start) = toml_text.find(entry_pattern) {
        let after = &toml_text[start + entry_pattern.len()..];
        if let Some(end_quote) = after.find('"') {
            let prefix = &toml_text[..start];
            let suffix = &toml_text[start + entry_pattern.len() + end_quote + 1..];
            return format!("{}entry = \"{}\"{}", prefix, new_entry, suffix);
        }
    }
    // No existing entry = line; append under [build].
    if let Some(pos) = toml_text.find("[build]") {
        let after = &toml_text[pos + 7..];
        // Find end of [build] section (next section header or EOF).
        let section_end = after.find("\n[").map(|p| pos + 7 + p + 1).unwrap_or(toml_text.len());
        let (before_end, rest) = toml_text.split_at(section_end);
        return format!("{}\nentry = \"{}\"\n{}", before_end, new_entry, rest);
    }
    // No [build] section at all; append one.
    format!("{}\n[build]\nentry = \"{}\"\n", toml_text, new_entry)
}

fn ensure_cdylib_extension(path: PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    let ext = "dll";
    #[cfg(not(target_os = "windows"))]
    let ext = "so";

    if path.extension().map(|e| e == ext).unwrap_or(false) {
        return path;
    }
    let stem = path.file_name().unwrap_or_default().to_string_lossy();
    let parent = path.parent().unwrap_or(Path::new("."));
    #[cfg(target_os = "windows")]
    let new_name = format!("{}.{}", stem, ext);
    #[cfg(not(target_os = "windows"))]
    let new_name = if stem.starts_with("lib") {
        format!("{}.{}", stem, ext)
    } else {
        format!("lib{}.{}", stem, ext)
    };
    parent.join(new_name)
}

fn ensure_object_extension(path: PathBuf) -> PathBuf {
    if path.extension().map(|e| e == "o").unwrap_or(false) {
        return path;
    }
    path.with_extension("o")
}
