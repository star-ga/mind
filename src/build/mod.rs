// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase A+F — `mindc build` single-crate orchestrator.
//!
//! Public entry point: [`run_build`].
//!
//! This module drives the existing pure-MIND compile pipeline
//! (`compile_sources` / `link_binary` in `src/project/mod.rs`) through a
//! typed options layer that replaces the unstructured flags the legacy
//! `build_project` function received.
//!
//! **Phase F** adds an incremental SHA-256-keyed object cache.  Cache
//! logic lives in [`cache`]; this module integrates it into the build flow.

pub mod cache;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::project::{
    BuildOptions as LegacyBuildOptions, BuildTarget, EmitKind, OptimizeLevel, build_project,
    find_project_root, find_project_root_for_file, load_manifest,
};

use cache::{
    BuildDecision, BuildManifest, CacheProbe, ObjectMeta, cache_root, module_cache_key, probe,
    write_object,
};

// ---------------------------------------------------------------------------
// Cache-key construction (issue #96: compiler + toolchain binary identity)
// ---------------------------------------------------------------------------

/// Emit-kind discriminator entries folded into the module cache key. A
/// `cdylib` shared object, a `binary` PIE, and a relocatable `object` are
/// distinct artifacts from identical source/target/optimize inputs and must
/// never share a slot. `cdylib` contributes NO entry (historical value);
/// `binary` / `object` get a labelled `emit=` entry.
fn emit_discriminator(emit: EmitKind) -> Vec<String> {
    match emit {
        EmitKind::Cdylib => Vec::new(),
        other => vec![format!("emit={}", other.as_str())],
    }
}

/// Toolchain-identity dep entries (clang / mlir-opt / mlir-translate). Each is
/// `toolchain=<name>|<resolved-path>|<size>|<mtime-ns>|<--version banner>` so a
/// toolchain swap — even one reporting an identical `--version` — invalidates
/// the cache (scan-finding S4, sibling of the runtime-obj cache in
/// `mlir_build::clang_identity_string`). Computed once per process. When the
/// `mlir-build` feature is off (no real backend) this is empty.
pub fn toolchain_dep_entries() -> Vec<String> {
    #[cfg(feature = "mlir-build")]
    {
        use std::sync::OnceLock;
        static ENTRIES: OnceLock<Vec<String>> = OnceLock::new();
        ENTRIES
            .get_or_init(|| match crate::eval::mlir_build::resolve_tools() {
                Ok(tools) => vec![
                    format!(
                        "toolchain=clang|{}",
                        crate::eval::mlir_build::tool_identity_string(&tools.clang)
                    ),
                    format!(
                        "toolchain=mlir-opt|{}",
                        crate::eval::mlir_build::tool_identity_string(&tools.mlir_opt)
                    ),
                    format!(
                        "toolchain=mlir-translate|{}",
                        crate::eval::mlir_build::tool_identity_string(&tools.mlir_translate)
                    ),
                ],
                // Tools unresolvable => the full compile would fail anyway and
                // no cache is written, so an empty toolchain set is safe here.
                Err(_) => Vec::new(),
            })
            .clone()
    }
    #[cfg(not(feature = "mlir-build"))]
    {
        Vec::new()
    }
}

/// Full dep-hash entry set for a module cache key: emit-kind discriminator plus
/// toolchain identity. `module_cache_key` sorts these, so order is irrelevant.
pub fn cache_dep_entries(emit: EmitKind) -> Vec<String> {
    let mut deps = emit_discriminator(emit);
    deps.extend(toolchain_dep_entries());
    deps
}

/// Compute the full module cache key for a compile of `source_bytes` produced
/// by the `mindc` binary at `mindc_exe`, or `None` (fail-closed) when that
/// binary's identity cannot be probed.
///
/// This is the single source of truth for the key shared by the build path
/// (which passes `std::env::current_exe()`) and the integration tests (which
/// pass `CARGO_BIN_EXE_mindc`): both MUST derive byte-identical keys or a
/// subprocess-populated cache would spuriously miss on an in-process probe
/// (issue #96 lockstep hazard). A `None` return MUST be treated as a cache MISS
/// — the cache is neither read nor written under any sentinel key.
pub fn compile_cache_key(
    source_bytes: &[u8],
    target: BuildTarget,
    optimize: OptimizeLevel,
    emit: EmitKind,
    mindc_exe: &Path,
    edition: u32,
) -> Option<String> {
    let identity = cache::compiler_identity_string(mindc_exe)?;
    let compiler_version = format!("{}+{}", env!("CARGO_PKG_VERSION"), identity);
    let deps = cache_dep_entries(emit);
    Some(module_cache_key(
        source_bytes,
        target,
        optimize,
        &deps,
        &compiler_version,
        edition,
    ))
}

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
    /// Bypass the incremental cache for this build (still writes new entries).
    /// Useful for debugging: "is my source the problem, or is it a stale cache?".
    pub no_cache: bool,
}

/// Per-build incremental cache statistics.
#[derive(Debug, Default)]
pub struct IncrementalStats {
    /// Modules that were found in cache and skipped.
    pub hits: u32,
    /// Modules that required a fresh compile.
    pub misses: u32,
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
    /// Incremental cache statistics for this build.
    pub cache_stats: IncrementalStats,
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
/// Loads `Mind.toml`, resolves the source list, probes the incremental object
/// cache (Phase F), drives the existing compile pipeline on misses, and returns
/// the path to the produced artifact.
///
/// When explicit source file paths are provided and no `Mind.toml` can be
/// located, a synthetic single-file manifest is synthesised so that one-off
/// `mindc build <file.mind>` invocations work without a project setup.
///
/// # Cache behaviour (Phase F)
///
/// - By default the cache is consulted.  A hit means the source + flags + deps
///   have not changed since the last build; the previous artifact is reused and
///   the compile pipeline is skipped.
/// - `opts.no_cache = true` bypasses the hit check but still writes the new
///   object to cache so subsequent runs can benefit.
///
/// # Exit code semantics
/// The caller should call `BuildError::exit_code()` when propagating errors
/// to `process::exit`.
pub fn run_build(opts: &BuildOpts) -> Result<BuildOutput, BuildError> {
    use crate::project::ProjectManifest;

    // 1. Locate the project root and load the manifest.
    //
    // Explicit-file builds (`mindc build <file>`) resolve the root with the
    // BOUNDED, git-aware `find_project_root_for_file` so a stray ancestor
    // `Mind.toml` (e.g. a leftover `/tmp/Mind.toml`) can never hijack the build
    // into walking a foreign tree — that hijack was a >30× compile-latency
    // regression (a one-file build under a poisoned `/tmp` walked all 116k dirs
    // of `/tmp`, ~2.3s vs ~60ms). No-explicit-path project builds keep the
    // classic cwd-anchored ancestor scan (`find_project_root`), unchanged.
    let (project_root, manifest) = if !opts.paths.is_empty() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let first_path = if opts.paths[0].is_absolute() {
            opts.paths[0].clone()
        } else {
            cwd.join(&opts.paths[0])
        };
        let entry_dir = first_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.clone());
        match find_project_root_for_file(&entry_dir) {
            Some(root) => {
                let m = load_manifest(&root)
                    .map_err(|e| BuildError::Invalid(format!("manifest error: {e}")))?;
                (root, m)
            }
            None => {
                // No governing manifest in-bounds — synthesise a single-file
                // manifest rooted at the entry file's OWN directory (never cwd
                // nor a distant ancestor), so source collection stays scoped to
                // it rather than to whatever tree happens to sit above.
                let stem = first_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .replace('-', "_");
                let pkg_name = if stem
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_alphabetic())
                    .unwrap_or(false)
                {
                    stem
                } else {
                    format!("pkg_{}", stem)
                };
                let toml_src = format!("[package]\nname = \"{}\"\nversion = \"0.1.0\"\n", pkg_name);
                let m: ProjectManifest = toml::from_str(&toml_src)
                    .map_err(|e| BuildError::Invalid(format!("synthetic manifest: {e}")))?;
                (entry_dir, m)
            }
        }
    } else {
        match find_project_root() {
            Ok(root) => {
                let m = load_manifest(&root)
                    .map_err(|e| BuildError::Invalid(format!("manifest error: {e}")))?;
                (root, m)
            }
            Err(e) => {
                return Err(BuildError::Invalid(format!("cannot locate Mind.toml: {e}")));
            }
        }
    };

    // Validate [package].name per RFC 0008 §3.
    validate_package_name(&manifest.package.name)?;

    // 2. Resolve effective build parameters (CLI > manifest > default).
    let eff_target = opts.target.unwrap_or(manifest.build.target);
    let eff_emit = opts.emit.unwrap_or(manifest.build.emit);
    let eff_optimize = opts.optimize.unwrap_or(manifest.build.optimize);

    // 3. Reject targets that have no backend implementation yet.
    validate_target(eff_target)?;

    // 4. Resolve the entry / source file(s).
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

    // Normalise the cdylib extension up front so the value threaded into the
    // build (where the shared library is linked directly to this path) is the
    // same path reported back to the caller. Otherwise the link would write to
    // the un-suffixed path while the post-build `ensure_cdylib_extension`
    // returns the suffixed path, leaving the artifact at the wrong name.
    let artifact_path = match eff_emit {
        EmitKind::Cdylib => ensure_cdylib_extension(artifact_path),
        EmitKind::Object => ensure_object_extension(artifact_path),
        EmitKind::Binary => artifact_path,
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

    // -------------------------------------------------------------------------
    // Phase F — incremental cache probe
    //
    // Scope: Phase A builds have one entry module per invocation.  We key on
    // the full entry source bytes + all effective build flags.  A hit means
    // the previous artifact (cached in the .cache/objects/ directory) is
    // byte-identical to what a fresh compile would produce, so we can copy it
    // to the output path and skip the entire compile + link pipeline.
    // -------------------------------------------------------------------------

    let edition: u32 = 2024;

    let source_bytes = fs::read(&entry_path).map_err(|e| {
        BuildError::Failed(format!("cannot read source {}: {e}", entry_path.display()))
    })?;

    // Cache key = source + build flags + emit-kind discriminator + compiler
    // BINARY IDENTITY + toolchain identity (issue #96 / scan-finding S4).
    //
    // The compiler identity is derived from THIS process's own binary
    // (`current_exe`): resolved path + size + mtime. A dev rebuild of `mindc`
    // at the same `CARGO_PKG_VERSION` bumps the mtime, so the key changes and a
    // stale cached `.so` is never served. If the identity cannot be probed we
    // FAIL CLOSED — `cache_key` is `None`, the cache is neither read nor
    // written, and the build recompiles rather than key on a stable sentinel
    // (which would reintroduce exactly the staleness this fixes).
    //
    // The emit kind is part of the artifact identity: a `cdylib`, a `binary`
    // PIE, and a relocatable `object` are three distinct artifacts from the
    // same source/target/optimize inputs and must NEVER share a slot (a
    // `cdylib` copied out as a "binary" is Type DYN, entry 0x0, no PT_INTERP →
    // segfault). `cdylib` contributes NO emit entry; `binary` / `object` get a
    // distinct `emit=` entry.
    //
    // Discriminator + compiler/toolchain identity are assembled by
    // `compile_cache_key`, the single source of truth the Phase G warm-cache
    // keystone probe (test 5) also calls against `CARGO_BIN_EXE_mindc` — so a
    // subprocess-populated cache and an in-process probe derive byte-identical
    // keys.
    let current_exe = std::env::current_exe().ok();
    let compiler_identity = current_exe
        .as_deref()
        .and_then(cache::compiler_identity_string);
    let cache_key: Option<String> = current_exe.as_deref().and_then(|exe| {
        compile_cache_key(
            &source_bytes,
            eff_target,
            eff_optimize,
            eff_emit,
            exe,
            edition,
        )
    });

    let c_root = cache_root(&project_root, eff_target, eff_optimize);

    // Probe the cache (skipped when --no-cache is set, or when the compiler's
    // own binary identity could not be probed — fail-closed: recompile rather
    // than risk serving a stale hit).
    let decision = match (opts.no_cache, cache_key.as_deref()) {
        (false, Some(cache_key)) => match probe(&c_root, cache_key) {
            CacheProbe::Hit {
                ref key,
                ref object_path,
            } => {
                if opts.verbose {
                    eprintln!("   [CACHE HIT] {} ({})", entry_path.display(), &key[..8]);
                }
                // Copy cached object to the requested artifact path.
                if let Err(e) = copy_or_rename(object_path, &artifact_path) {
                    // Cache read failed; treat as miss and recompile.
                    if opts.verbose {
                        eprintln!("   [CACHE] read failed ({}); recompiling", e);
                    }
                    BuildDecision::CacheMiss
                } else {
                    // Update manifest.
                    update_manifest(&c_root, &project_root, &entry_path, cache_key, opts.verbose);

                    let final_path = match eff_emit {
                        EmitKind::Cdylib => ensure_cdylib_extension(artifact_path),
                        EmitKind::Object => ensure_object_extension(artifact_path),
                        EmitKind::Binary => artifact_path,
                    };
                    // A cached artifact is restored by copying bytes out of the
                    // cache store (0644); a `binary` must be executable to run,
                    // so re-assert the executable bit on the warm-cache path.
                    // The cold link path already yields a `+x` file from `cc`.
                    ensure_executable_if_binary(&final_path, eff_emit);
                    let byte_count = fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
                    return Ok(BuildOutput {
                        artifact_path: final_path,
                        target: eff_target.as_str().to_string(),
                        emit: eff_emit,
                        byte_count,
                        cache_stats: IncrementalStats { hits: 1, misses: 0 },
                    });
                }
            }
            CacheProbe::Miss { .. } => BuildDecision::CacheMiss,
        },
        _ => BuildDecision::CacheMiss,
    };

    let _ = decision; // CacheMiss — fall through to full compile

    // -------------------------------------------------------------------------
    // Full compile path (cache miss or --no-cache)
    // -------------------------------------------------------------------------

    let legacy_opts = legacy_opts_from(
        eff_target,
        eff_emit,
        eff_optimize,
        &manifest.exports.c_abi,
        &entry_path,
        &artifact_path,
        opts.verbose,
        &project_root,
    );

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

    let need_write = match &orig_manifest_text {
        Some(text) => entry_rel != manifest.build.entry || text.find("entry = ").is_none(),
        None => true,
    };

    if need_write {
        let toml_to_write = match &orig_manifest_text {
            Some(text) => patch_manifest_entry(text, &entry_rel),
            None => build_synthetic_manifest(&manifest.package.name, &entry_rel),
        };
        fs::write(&manifest_path, &toml_to_write)
            .map_err(|e| BuildError::Failed(format!("cannot write manifest: {e}")))?;
    }

    let build_result = build_project(&legacy_opts);

    if need_write {
        match &orig_manifest_text {
            Some(text) => {
                let _ = fs::write(&manifest_path, text);
            }
            None => {
                let _ = fs::remove_file(&manifest_path);
            }
        }
    }

    let build_result = build_result.map_err(|e| BuildError::Failed(format!("{e}")))?;

    // Move/rename the legacy output to the requested artifact_path if needed.
    let final_path = if artifact_path != build_result.output_path {
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

    let final_path = match eff_emit {
        EmitKind::Cdylib => ensure_cdylib_extension(final_path),
        EmitKind::Object => ensure_object_extension(final_path),
        EmitKind::Binary => final_path,
    };
    // The cold link path yields a `+x` file from `cc`, but a move/copy to the
    // requested `--out` path can land on a filesystem that drops the bit;
    // re-assert it for `binary` so the fresh artifact is directly runnable.
    ensure_executable_if_binary(&final_path, eff_emit);

    // -------------------------------------------------------------------------
    // Phase F — write compiled artifact to cache (always, even with --no-cache)
    // -------------------------------------------------------------------------
    if let Some(cache_key) = cache_key.as_deref() {
        if final_path.exists() {
            if let Ok(artifact_bytes) = fs::read(&final_path) {
                // `cache_key` is `Some` only when the compiler identity probed
                // successfully, so the fingerprint below is always populated.
                let identity = compiler_identity.clone().unwrap_or_default();
                let mut dep_hashes_sorted = cache_dep_entries(eff_emit);
                dep_hashes_sorted.sort();
                let meta = ObjectMeta {
                    source_path: entry_rel.clone(),
                    cache_key: cache_key.to_string(),
                    target: eff_target.as_str().to_string(),
                    optimize: eff_optimize.as_str().to_string(),
                    compiler_version: format!("{}+{}", env!("CARGO_PKG_VERSION"), identity),
                    compiler_fingerprint: identity,
                    dep_hashes: dep_hashes_sorted,
                };
                // Best-effort: cache write failure does not fail the build.
                let _ = write_object(&c_root, cache_key, &artifact_bytes, &meta);
                update_manifest(&c_root, &project_root, &entry_path, cache_key, opts.verbose);
            }
        }
    }

    let byte_count = fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);

    Ok(BuildOutput {
        artifact_path: final_path,
        target: eff_target.as_str().to_string(),
        emit: eff_emit,
        byte_count,
        cache_stats: IncrementalStats { hits: 0, misses: 1 },
    })
}

// ---------------------------------------------------------------------------
// Phase F helpers
// ---------------------------------------------------------------------------

/// Ensure a `binary` artifact carries the owner/group/other execute bits.
///
/// A `cdylib`/`object` is never executed directly, so this is a no-op for them;
/// only a `binary` (a PIE executable) needs `+x`. This is idempotent and covers
/// both the warm-cache restore (bytes copied out of the 0644 cache store) and a
/// cross-filesystem move of the freshly linked artifact.
fn ensure_executable_if_binary(path: &Path, emit: EmitKind) {
    if emit != EmitKind::Binary {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            let mode = perms.mode();
            // Add execute wherever the matching read bit is set (0644 -> 0755).
            let new_mode = mode | ((mode & 0o444) >> 2);
            if new_mode != mode {
                perms.set_mode(new_mode);
                let _ = fs::set_permissions(path, perms);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// Copy `src` to `dest`, trying rename first (faster, same filesystem).
fn copy_or_rename(src: &Path, dest: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    // Try a hard-link copy first for zero-copy on same fs, fall back to fs::copy.
    fs::copy(src, dest).map(|_| ())
}

/// Update `manifest.json` to record `source_path -> cache_key`.
fn update_manifest(
    c_root: &Path,
    project_root: &Path,
    entry_path: &Path,
    cache_key: &str,
    verbose: bool,
) {
    let mpath = cache::manifest_path(c_root);
    let mut manifest = BuildManifest::load(&mpath).unwrap_or_default();
    let rel = entry_path
        .strip_prefix(project_root)
        .unwrap_or(entry_path)
        .to_string_lossy()
        .replace('\\', "/");
    manifest.entries.insert(rel, cache_key.to_string());
    if let Err(e) = manifest.save(&mpath) {
        if verbose {
            eprintln!("   [CACHE] manifest write failed: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_package_name(name: &str) -> Result<(), BuildError> {
    let valid = name.chars().enumerate().all(|(i, c)| {
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
    // a) Explicit CLI paths take priority. A relative path is resolved against
    //    the CURRENT DIRECTORY (where the user typed it), never `project_root`:
    //    the bounded root may be an ancestor (git repo root) or a synthesised
    //    entry dir, and joining a cwd-relative path onto it would resolve the
    //    file in the wrong place (e.g. `mindc build t.mind` from a subdir whose
    //    governing Mind.toml sits at the repo root would look for
    //    `<repo>/t.mind`). Absolute paths are used verbatim.
    if let Some(first) = opts.paths.first() {
        let p = if first.is_absolute() {
            first.clone()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(first)
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
    let profile_dir = if optimize.is_release() {
        "release"
    } else {
        "debug"
    };
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
#[allow(clippy::too_many_arguments)]
fn legacy_opts_from(
    target: BuildTarget,
    emit: EmitKind,
    optimize: OptimizeLevel,
    manifest_exports: &[String],
    _entry_path: &Path,
    artifact_path: &Path,
    verbose: bool,
    project_root: &Path,
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
        // Thread the emit kind so `build_project` routes `cdylib` through the
        // single-entry shared-library link path (which links the runtime
        // support shim) instead of the whole-directory executable link path.
        emit,
        // Thread the resolved `--out` so the cdylib link writes directly to
        // the final path — no shared `target/<profile>/<name>` intermediary
        // for concurrent builds to collide on.
        out_path: Some(artifact_path.to_path_buf()),
        // Thread the ALREADY-resolved (bounded) project root so `build_project`
        // reuses it instead of re-running the unbounded `find_project_root()`
        // from cwd — otherwise the legacy path could re-ascend to a stray
        // ancestor `Mind.toml` and reintroduce the foreign-tree walk.
        project_root: Some(project_root.to_path_buf()),
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
        let section_end = after
            .find("\n[")
            .map(|p| pos + 7 + p + 1)
            .unwrap_or(toml_text.len());
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
