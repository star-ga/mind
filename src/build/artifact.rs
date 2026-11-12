// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! The seam between `run_build`'s resolved inputs and the file on disk: the
//! artifact PATH, the `[targets.*]` block name that build carries, and the
//! `LegacyBuildOptions` that thread both into the compile pipeline.
//!
//! None of this decides the artifact NAME — `crate::project::artifact_stem` is
//! the one owner of that, and `crate::project::artifact_dir` is the one owner of
//! the `target/<profile>/` directory. What is left here is the emit-dependent
//! decoration and the option bridge, i.e. exactly the facts that belong to the
//! `mindc build` driver and to nothing else.

use std::path::{Path, PathBuf};

use crate::project::{
    BuildOptions as LegacyBuildOptions, BuildTarget, EmitKind, OptimizeLevel, artifact_dir,
};

/// The default artifact path: [`artifact_dir`] plus the emit-dependent
/// decoration (`lib….so` / `.o`).
///
/// `stem` comes from [`crate::project::artifact_stem`] — the single owner of the
/// artifact NAME — and the directory from [`artifact_dir`]. This function owns
/// ONLY the decoration; it must never re-derive either, which is precisely the
/// split it used to be half of.
pub(super) fn default_artifact_path(
    project_root: &Path,
    stem: &str,
    emit: EmitKind,
    optimize: OptimizeLevel,
) -> PathBuf {
    let base = artifact_dir(project_root, optimize.is_release());
    match emit {
        EmitKind::Binary => base.join(stem),
        EmitKind::Cdylib => {
            #[cfg(target_os = "windows")]
            let name = format!("{}.dll", stem);
            #[cfg(not(target_os = "windows"))]
            let name = format!("lib{}.so", stem);
            base.join(name)
        }
        EmitKind::Object => base.join(format!("{}.o", stem)),
    }
}

/// The native backend's artifact path with no `--out` (ruling 1.9): for a
/// governed build (`governed` = root and manifest), the MLIR path function
/// above (`target/<profile>/<name>`), so a native build replaces the file an
/// MLIR build wrote there and a stale binary is never the one that runs; with
/// no governing manifest, `a.out`. `optimize` is `--release` / `--optimize`,
/// else the manifest's level.
pub(crate) fn native_artifact_path(
    governed: Option<&(PathBuf, crate::project::ProjectManifest)>,
    optimize: Option<OptimizeLevel>,
) -> String {
    match governed {
        None => "a.out".to_string(),
        Some((root, manifest)) => default_artifact_path(
            root,
            crate::project::artifact_stem(manifest, None),
            EmitKind::Binary,
            optimize.unwrap_or(manifest.build.optimize),
        )
        .display()
        .to_string(),
    }
}

/// The `[targets.<name>]` block name `build_project` will select for this build.
///
/// Prefer the explicitly-selected block so `build_project` picks THAT block (its
/// sources / native_sources / `.target` triple). With no block selected this
/// falls back to the historical class mapping (`Cpu => None`, so the default
/// host path is byte-identical).
///
/// ONE owner: both the legacy options handed to `build_project` and the source
/// set the cache key fingerprints resolve the block through this function, so
/// they cannot select different `[targets.*].sources` lists.
pub(super) fn legacy_target_name(
    target: BuildTarget,
    block_name: Option<String>,
) -> Option<String> {
    block_name.or(match target {
        BuildTarget::Cpu => None,
        other => Some(other.as_str().to_string()),
    })
}

/// Build the `LegacyBuildOptions` used to call the existing `build_project`.
#[allow(clippy::too_many_arguments)]
pub(super) fn legacy_opts_from(
    target_str: Option<String>,
    emit: EmitKind,
    optimize: OptimizeLevel,
    manifest_exports: &[String],
    _entry_path: &Path,
    artifact_path: &Path,
    verbose: bool,
    project_root: &Path,
    single_file: bool,
) -> LegacyBuildOptions {
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
        // Explicit single-file build (no governing `Mind.toml`): compile ONLY
        // the entry, never the whole entry-directory walk (which would pull in
        // unrelated sibling `.mind` files from a shared dir).
        single_file,
    }
}
