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

/// Artifact PATH assembly and the legacy-option bridge — the emit-dependent
/// decoration and the `[targets.*]` block name this build carries, kept out of
/// the driver so the driver reads as one pass.
mod artifact;
mod atomic_publish;
pub mod cache;
mod driver_error;
mod error;
pub mod native_bridge;
pub mod native_image;
pub mod native_scope;
mod project_transaction;

/// Incremental-cache KEY construction — every input that can change the
/// artifact, fingerprinted in one place and failing closed when it cannot be.
mod source_key;

pub use error::BuildError;
pub use source_key::{
    CacheKeyFlags, cache_dep_entries, compile_cache_key, compile_cache_material,
    source_set_dep_entries, toolchain_dep_entries,
};

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use crate::project::build_lock::temporary_manifest;
use artifact::{default_artifact_path, legacy_opts_from, legacy_target_name};
use driver_error::classify_driver_error;
use project_transaction::{
    ManifestEdit, open_explicit, open_project, probe_root, resolve_explicit_entry,
};
use source_key::compile_cache_material_from_snapshots as cache_material_from;

use crate::project::{
    BuildTarget, EmitKind, OptimizeLevel, build_input_snapshot::BuildInputSnapshot,
    build_project_locked_with_snapshot as build_snapshot,
};

use cache::{
    BuildDecision, BuildManifest, CacheProbe, ObjectMeta, cache_root, probe, write_object,
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
    /// Override `[build].target` — the RAW `--target` name (a declared
    /// `[targets.<name>]` block name OR a backend class), resolved against the
    /// manifest by `resolve_build_target`. Raw (not pre-parsed to a
    /// [`BuildTarget`]) so a declared block name that is not a backend class
    /// (e.g. `windows`) survives to the manifest-aware resolver instead of being
    /// rejected before the manifest is loaded.
    pub target: Option<String>,
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
    // 1. Locate the project root and load the manifest.
    //
    // Explicit-file builds (`mindc build <file>`) resolve the root with the
    // BOUNDED, git-aware `find_project_root_for_file` so a stray ancestor
    // `Mind.toml` (e.g. a leftover `/tmp/Mind.toml`) can never hijack the build
    // into walking a foreign tree — that hijack was a >30× compile-latency
    // regression (a one-file build under a poisoned `/tmp` walked all 116k dirs
    // of `/tmp`, ~2.3s vs ~60ms). No-explicit-path project builds keep the
    // classic cwd-anchored ancestor scan (`find_project_root`), unchanged.
    // `single_file` marks an explicit `mindc build <file>` with NO governing
    // Mind.toml in bounds — the manifest is synthesised and the project root is
    // the entry file's OWN directory. Such a build compiles only the named entry
    // (see `BuildOptions::single_file`), never a whole-directory sibling walk.
    let explicit_entry = resolve_explicit_entry(&opts.paths)?;
    let (project_root, manifest, build_lock, single_file) =
        if let Some((entry_dir, first_path)) = explicit_entry.clone() {
            // The root is probed before any lock is held, by the rule of the
            // lock (`probe_root`), and re-audited after it, inside
            // `open_explicit` — see that function for the window that ordering
            // opens.
            let opened = open_explicit(probe_root(&entry_dir)?, &entry_dir, &first_path)?;
            (
                opened.root,
                opened.manifest,
                opened.lock,
                opened.single_file,
            )
        } else {
            let opened = open_project()?;
            (opened.root, opened.manifest, opened.lock, false)
        };

    // Validate [package].name per RFC 0008 §3.
    validate_package_name(&manifest.package.name)?;

    // 2. Resolve effective build parameters (CLI > manifest > default). The raw
    //    --target name resolves against `[targets.<name>]` blocks FIRST (declared
    //    block wins), then backend classes; an undeclared name is a hard error,
    //    never a silent host build. `sel_block` is the selected block name,
    //    threaded to `build_project` so it picks THAT block's sources /
    //    native_sources / `.target` triple.
    let (eff_target, sel_block) = resolve_build_target(opts.target.as_deref(), &manifest)?;
    let eff_emit = opts.emit.unwrap_or(manifest.build.emit);
    let eff_optimize = opts.optimize.unwrap_or(manifest.build.optimize);

    // The legacy target string `build_project` will see, resolved ONCE here so
    // the block whose `output` this orchestrator honours is the same block the
    // compile path honours.
    let legacy_target = legacy_target_name(eff_target, sel_block);

    // 3. Reject targets that have no backend implementation yet.
    validate_target(eff_target)?;

    // 4. Resolve the entry / source file(s). An explicit CLI path was already
    //    resolved to the transaction's canonical identity above; re-deriving it
    //    from `opts.paths` here would reintroduce a spelling `project_root` is
    //    not comparable with.
    let entry_path = resolve_entry(
        explicit_entry.as_ref().map(|(_, entry)| entry.as_path()),
        &project_root,
        &manifest.build.entry,
        eff_emit,
    )?;

    // 5. Build the output path.
    let artifact_path = match &opts.out {
        Some(p) => p.clone(),
        None => default_artifact_path(
            &project_root,
            crate::project::artifact_stem(&manifest, legacy_target.as_deref()),
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
            .map_err(|e| BuildError::failed(e.to_string()))?;
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
    // The key also fingerprints EVERY OTHER SOURCE the build compiles, resolved
    // here through the same single selector `build_project` uses. Keying on the
    // entry alone was a silent-staleness hole: editing a sibling module left the
    // key unchanged, the probe HIT, and `mindc build` copied out a binary built
    // from the OLD sibling — exit 0, no diagnostic, wrong program.
    //
    // The entry fed to the selector is the RESOLVED one, not `manifest.build.entry`:
    // an explicit `mindc build <file>` rewrites that manifest key further down, so
    // the pre-patch value would fingerprint a different project than the one that
    // is about to be compiled.
    //
    // A resolution failure is a PROJECT-DRIVER error (a declared source that does
    // not exist, an escaping path), so it is typed by `classify_driver_error` —
    // the one seam that turns a driver error into a `BuildError` while KEEPING its
    // cause code. Hand-rolling a refusal here would strip that code and an
    // undiagnosed refusal fails closed at every consumer.
    let entry_rel = entry_relative_to_root(&entry_path, &project_root)?;
    let selected_block = legacy_target
        .as_deref()
        .unwrap_or(crate::project::DEFAULT_TARGET_BLOCK);
    let (build_sources, _explicit_sources) = crate::project::resolve_sources(
        &project_root,
        &entry_rel,
        manifest
            .targets
            .get(selected_block)
            .and_then(|t| t.sources.as_deref()),
        single_file,
    )
    .map_err(classify_driver_error)?;
    let source_snapshot = crate::project::source_snapshot::SourceSnapshot::capture(&build_sources)
        .map_err(classify_driver_error)?;
    let build_input_snapshot = BuildInputSnapshot::capture(
        &project_root,
        &manifest,
        selected_block,
        eff_emit,
        &entry_rel,
    )
    .map_err(classify_driver_error)?;

    let current_exe = std::env::current_exe().ok();
    let compiler_identity = current_exe
        .as_deref()
        .and_then(cache::compiler_identity_string);
    let cache_material = current_exe.as_deref().and_then(|exe| {
        cache_material_from(
            &source_snapshot,
            &build_input_snapshot,
            &entry_path,
            CacheKeyFlags {
                target: eff_target,
                optimize: eff_optimize,
                emit: eff_emit,
                edition,
            },
            exe,
            &project_root,
        )
    });

    let c_root = cache_root(&project_root, eff_target, eff_optimize);

    // Probe the cache (skipped when --no-cache is set, or when the compiler's
    // own binary identity could not be probed — fail-closed: recompile rather
    // than risk serving a stale hit).
    let decision = match (opts.no_cache, cache_material.as_ref()) {
        (false, Some(cache_material)) => match probe(&c_root, cache_material) {
            CacheProbe::Hit {
                ref key,
                ref object_bytes,
            } => {
                if opts.verbose {
                    eprintln!("   [CACHE HIT] {} ({})", entry_path.display(), &key[..8]);
                }
                // Restore the bytes that probe already verified.
                if let Err(e) = cache::restore_object(&artifact_path, object_bytes) {
                    // Cache read failed; treat as miss and recompile.
                    if opts.verbose {
                        eprintln!("   [CACHE] read failed ({}); recompiling", e);
                    }
                    BuildDecision::CacheMiss
                } else {
                    // Update manifest.
                    update_manifest(&c_root, &entry_rel, key, opts.verbose);

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

    #[cfg(test)]
    crate::project::source_snapshot::run_test_hook();

    // Full compile path (cache miss or --no-cache)
    // -------------------------------------------------------------------------

    let legacy_opts = legacy_opts_from(
        legacy_target,
        eff_emit,
        eff_optimize,
        build_input_snapshot.exports(),
        &entry_path,
        &artifact_path,
        opts.verbose,
        &project_root,
        single_file,
    );

    let manifest_path = project_root.join("Mind.toml");
    let manifest_existed = manifest_path.exists();

    let orig_manifest_text: Option<String> = if manifest_existed {
        Some(
            fs::read_to_string(&manifest_path)
                .map_err(|e| BuildError::failed(format!("cannot read manifest: {e}")))?,
        )
    } else {
        None
    };

    let need_write = match &orig_manifest_text {
        Some(text) => entry_rel != manifest.build.entry || !manifest_has_entry_key(text),
        None => true,
    };

    let manifest_edit = if need_write {
        let toml_to_write = match &orig_manifest_text {
            Some(text) => patch_manifest_entry(text, &entry_rel),
            None => temporary_manifest(&manifest.package.name, &entry_rel),
        };
        Some(
            ManifestEdit::replace(
                &manifest_path,
                orig_manifest_text
                    .as_ref()
                    .map(|text| text.as_bytes().to_vec()),
                toml_to_write.as_bytes(),
            )
            .map_err(|e| BuildError::failed(format!("cannot write manifest: {e}")))?,
        )
    } else {
        None
    };

    let build_result = build_snapshot(
        &legacy_opts,
        &build_lock,
        &source_snapshot,
        &build_input_snapshot,
    );

    if let Some(edit) = manifest_edit {
        edit.restore()
            .map_err(|e| BuildError::failed(format!("cannot restore manifest after build: {e}")))?;
    }

    let build_result = build_result.map_err(classify_driver_error)?;

    error::refuse_if_fell_back(&build_result)?;

    // Move/rename the legacy output to the requested artifact_path if needed.
    let final_path = if artifact_path != build_result.output_path {
        if let Some(parent) = artifact_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::rename(&build_result.output_path, &artifact_path)
            .or_else(|_| fs::copy(&build_result.output_path, &artifact_path).map(|_| ()))
            .map_err(|e| BuildError::failed(format!("cannot move artifact: {e}")))?;
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
    if let Some(cache_material) = cache_material.as_ref() {
        if final_path.exists() {
            if let Ok(artifact_bytes) = fs::read(&final_path) {
                // Cache material exists only when the compiler identity probed
                // successfully, so the fingerprint below is always populated.
                let identity = compiler_identity.clone().unwrap_or_default();
                let meta = ObjectMeta {
                    source_path: entry_rel.clone(),
                    cache_key: cache_material.key.clone(),
                    target: cache_material.input_inventory.target.clone(),
                    optimize: cache_material.input_inventory.optimize.clone(),
                    compiler_version: format!("{}+{}", env!("CARGO_PKG_VERSION"), identity),
                    compiler_fingerprint: identity,
                    dep_hashes: cache_material.input_inventory.dependencies.clone(),
                    input_inventory: cache_material.input_inventory.clone(),
                };
                // Best-effort: cache write failure does not fail the build.
                let _ = write_object(&c_root, cache_material, &artifact_bytes, &meta);
                update_manifest(&c_root, &entry_rel, &cache_material.key, opts.verbose);
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

/// Update `manifest.json` to record `source_path -> cache_key`.
fn update_manifest(c_root: &Path, entry_rel: &str, cache_key: &str, verbose: bool) {
    let mpath = cache::manifest_path(c_root);
    let mut manifest = BuildManifest::load(&mpath).unwrap_or_default();
    manifest
        .entries
        .insert(entry_rel.to_string(), cache_key.to_string());
    if let Err(e) = manifest.save(&mpath) {
        if verbose {
            eprintln!("   [CACHE] manifest write failed: {e}");
        }
    }
}

/// Derive the one project-relative spelling used by source selection, cache
/// identity, and temporary manifest edits. A root/entry mismatch is an
/// invariant failure and must never become an absolute cache or manifest key.
fn entry_relative_to_root(entry_path: &Path, project_root: &Path) -> Result<String, BuildError> {
    let relative = entry_path.strip_prefix(project_root).map_err(|_| {
        BuildError::Invalid(format!(
            "entry path {} is outside project root {}",
            entry_path.display(),
            project_root.display()
        ))
    })?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
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

/// Resolve the raw `--target` name into (backend class, selected block name).
///
/// Declared-block-name wins (the Cargo-profile shape): a `--target <name>` that
/// matches a `[targets.<name>]` block uses the block's `backend` field as the
/// class and threads `<name>` to `build_project` (which reads the block's
/// sources / native_sources / `.target` triple). Otherwise `<name>` must itself
/// be a backend class. A name that is NEITHER is a hard error — the fail-loud
/// closure of the undeclared-`--target`-name hole (never a silent host build).
/// When no `--target` is given, the manifest `[build].target` class is used with
/// no block override (byte-identical to the historical default path).
///
/// deferred: the incremental cache root/key (`cache_root` / `compile_cache_key`)
/// still key on the backend CLASS only, so two blocks with the same backend but
/// different `.target` triples (e.g. a `[targets.cpu]` linux target and a
/// `[targets.windows]` windows-gnu target, both `backend = "cpu"`) would share a
/// `target/cpu/<opt>/.cache` slot. This is DORMANT today: a non-host triple fails
/// loud at the "not yet wired" cross-compilation guard in `build_project` before
/// any artifact is produced or cached. Upgrade path (lands WITH the windows-gnu
/// PE slice, when a cross artifact is actually emitted): thread the resolved
/// triple into `cache_root` + `compile_cache_key` so per-OS artifacts occupy
/// distinct cache slots.
fn resolve_build_target(
    name: Option<&str>,
    manifest: &crate::project::ProjectManifest,
) -> Result<(BuildTarget, Option<String>), BuildError> {
    let Some(n) = name else {
        return Ok((manifest.build.target, None));
    };
    if let Some(block) = manifest.targets.get(n) {
        // Declared block wins; its `backend` field must be a real class.
        let bt = BuildTarget::parse(&block.backend).map_err(|_| {
            BuildError::Invalid(format!(
                "targets.{n}.backend `{}` is not a backend class \
                 (cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras|wasm)",
                block.backend
            ))
        })?;
        Ok((bt, Some(n.to_string())))
    } else if let Ok(bt) = BuildTarget::parse(n) {
        // Legacy class path (no matching block).
        Ok((bt, None))
    } else {
        let mut declared: Vec<&str> = manifest.targets.keys().map(String::as_str).collect();
        declared.sort_unstable();
        let declared = if declared.is_empty() {
            "none".to_string()
        } else {
            declared.join(", ")
        };
        Err(BuildError::Invalid(format!(
            "unknown target `{n}`: neither a declared [targets.*] block \
             (declared: {declared}) nor a backend class \
             (cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras|wasm)"
        )))
    }
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
    explicit: Option<&Path>,
    project_root: &Path,
    manifest_entry: &str,
    eff_emit: EmitKind,
) -> Result<PathBuf, BuildError> {
    // a) An explicit CLI path takes priority, in the canonical spelling
    //    [`canonical_explicit_entry`] produced. A relative path was resolved
    //    against the CURRENT DIRECTORY (where the user typed it), never
    //    `project_root`: the bounded root may be an ancestor (git repo root) or
    //    a synthesised entry dir, and joining a cwd-relative path onto it would
    //    resolve the file in the wrong place (e.g. `mindc build t.mind` from a
    //    subdir whose governing Mind.toml sits at the repo root would look for
    //    `<repo>/t.mind`).
    if let Some(p) = explicit {
        if !p.exists() {
            return Err(BuildError::failed(format!(
                "source file not found: {}",
                p.display()
            )));
        }
        return Ok(p.to_path_buf());
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

    Err(BuildError::failed(
        "no source files provided and no src/main.mind or src/lib.mind found".to_string(),
    ))
}

/// Patch a raw `Mind.toml` text to change `entry = "..."`.
/// If no `entry =` line is present, appends the field under `[build]`.
/// True when `toml_text` declares an `entry` key on some line, tolerating any
/// amount of whitespace around `=` — aligned manifests write
/// `entry         = "..."`. The previous single-space textual probe
/// (`"entry = "`) missed those, so `run_build` "patched" a second `entry`
/// key into a `[build]` section that already had one; the duplicate TOML key
/// then made `build_project`'s re-parse fail with an opaque
/// `Failed to parse Mind.toml` (#85 — mind-nerve's aligned manifest).
fn manifest_has_entry_key(toml_text: &str) -> bool {
    toml_text.lines().any(is_entry_key_line)
}

/// Line-level test: optional leading whitespace, the key `entry`, then `=`
/// after optional whitespace. Never matches `entrypoint = ...`,
/// `entry_file = ...`, or a `# entry = ...` comment line. Like the previous
/// textual probe, this matches an `entry` key in ANY section — by contract
/// the key lives only in `[build]`.
fn is_entry_key_line(line: &str) -> bool {
    line.trim_start()
        .strip_prefix("entry")
        .map(|rest| rest.trim_start().starts_with('='))
        .unwrap_or(false)
}

fn patch_manifest_entry(toml_text: &str, new_entry: &str) -> String {
    // Replace an existing `entry` key line, tolerating aligned padding and
    // either quote style. Line-based: the first matching line is replaced
    // wholesale, so a padded or single-quoted key can never end up
    // double-declared (duplicate TOML keys are a parse error).
    if manifest_has_entry_key(toml_text) {
        let mut out = String::with_capacity(toml_text.len() + new_entry.len() + 16);
        let mut replaced = false;
        for line in toml_text.lines() {
            if !replaced && is_entry_key_line(line) {
                out.push_str("entry = \"");
                out.push_str(new_entry);
                out.push('"');
                replaced = true;
            } else {
                out.push_str(line);
            }
            out.push('\n');
        }
        return out;
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

#[cfg(test)]
mod manifest_patch_tests {
    use super::{is_entry_key_line, manifest_has_entry_key, patch_manifest_entry};

    /// Parse helper: the patched text must be valid TOML (a duplicate `entry`
    /// key is a parse error — the exact #85 failure mode) and carry the
    /// expected `[build].entry` value.
    fn assert_entry(patched: &str, expected: &str) {
        let v: toml::Value = toml::from_str(patched)
            .unwrap_or_else(|e| panic!("patched manifest is invalid TOML: {e}\n---\n{patched}"));
        let entry = v
            .get("build")
            .and_then(|b| b.get("entry"))
            .and_then(|e| e.as_str())
            .expect("patched manifest has no [build].entry");
        assert_eq!(entry, expected);
    }

    #[test]
    fn detects_aligned_entry_key() {
        // The #85 shape: aligned padding defeats a single-space textual probe.
        assert!(manifest_has_entry_key(
            "[build]\nentry         = \"cli/main.mind\"\n"
        ));
        assert!(manifest_has_entry_key("[build]\nentry=\"x.mind\"\n"));
        assert!(manifest_has_entry_key("[build]\nentry = 'x.mind'\n"));
    }

    #[test]
    fn rejects_lookalike_keys_and_comments() {
        assert!(!manifest_has_entry_key("[build]\nentrypoint = \"x\"\n"));
        assert!(!manifest_has_entry_key("[build]\nentry_file = \"x\"\n"));
        assert!(!manifest_has_entry_key("[build]\n# entry = \"x\"\n"));
        assert!(!is_entry_key_line("entry"));
    }

    #[test]
    fn patch_replaces_aligned_entry_without_duplicating() {
        let src = "[package]\nname = \"n\"\nversion = \"0.1.0\"\n\n[build]\nentry         = \"cli/main.mind\"\noutput        = \"n\"\n\n[targets.cpu]\nbackend = \"cpu\"\n";
        let patched = patch_manifest_entry(src, "cli/main.mind");
        assert_entry(&patched, "cli/main.mind");
    }

    #[test]
    fn patch_replaces_no_space_and_single_quoted_entry() {
        let patched = patch_manifest_entry("[build]\nentry=\"old.mind\"\n", "new.mind");
        assert_entry(&patched, "new.mind");
        let patched = patch_manifest_entry("[build]\nentry = 'old.mind'\n", "new.mind");
        assert_entry(&patched, "new.mind");
    }

    #[test]
    fn patch_appends_when_build_section_has_no_entry() {
        let patched = patch_manifest_entry("[build]\noutput = \"n\"\n\n[test]\n", "m.mind");
        assert_entry(&patched, "m.mind");
    }

    #[test]
    fn patch_appends_build_section_when_absent() {
        let patched =
            patch_manifest_entry("[package]\nname = \"n\"\nversion = \"0.1.0\"\n", "m.mind");
        assert_entry(&patched, "m.mind");
    }
}

#[cfg(all(test, unix))]
mod transaction_identity_tests;
