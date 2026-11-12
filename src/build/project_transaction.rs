// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Preparation shared by manifest-backed and standalone build transactions.

use super::BuildError;
use crate::project::{ProjectManifest, find_project_root_for_file_checked, load_manifest};
use anyhow::Result;
use std::path::{Path, PathBuf};

pub(super) use crate::project::build_lock::ManifestEdit;
use crate::project::build_lock::{
    ProjectBuildLock, lock_cwd_project, remove_leftover, temporary_manifest,
};

/// What the project-transaction boundary hands the rest of a build.
pub(super) struct BuildTransaction {
    /// Canonical project root this build compiles under.
    pub(super) root: PathBuf,
    /// The manifest governing it — loaded from disk, or synthesised.
    pub(super) manifest: ProjectManifest,
    /// Held for the whole build; dropped when the caller's binding is.
    pub(super) lock: ProjectBuildLock,
    /// The "project" is exactly the named entry, never its directory.
    pub(super) single_file: bool,
}

/// Transaction identity of an explicit `mindc build <file>` source: the
/// canonical directory holding it, plus the entry re-spelled inside that
/// directory.
///
/// Only the directory is canonicalised. The leaf keeps the spelling the
/// operator asked for, because a `.mind` entry may legitimately be a symlink to
/// a source elsewhere in the project and resolving it would retarget the build
/// at a file the requested directory's manifest does not govern.
///
/// Sharing [`crate::project::canonical_dir`] with `find_project_root_for_file`
/// is the point: root and entry directory are then two results of one function,
/// so `root == entry_dir` tests directory identity. A lexically derived entry
/// directory compares unequal to the root whenever a component is a symlink,
/// which defeats the vanished-temporary-manifest fallback and mis-classifies a
/// co-located bare manifest as a whole-directory project.
pub(super) fn canonical_explicit_entry(requested: &Path, cwd: &Path) -> Result<(PathBuf, PathBuf)> {
    let absolute = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        cwd.join(requested)
    };
    let lexical_dir = absolute
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| cwd.to_path_buf());
    let entry_dir = crate::project::canonical_dir(&lexical_dir)?
        .ok_or_else(|| anyhow::anyhow!("cannot resolve current directory for explicit entry"))?;
    let entry_path = match absolute.file_name() {
        Some(name) => entry_dir.join(name),
        None => absolute,
    };
    Ok((entry_dir, entry_path))
}

/// Open the build transaction for an explicit `mindc build <file>`.
///
/// `probed_root` is `find_project_root_for_file(entry_dir)` evaluated before any
/// lock is held. It is a parameter rather than a call inside this function
/// because the gap between probe and lock is the window a concurrent standalone
/// build lives in: that build holds this same lock while its temporary
/// `Mind.toml` exists and removes the file before releasing. Passing the probe
/// in lets a test interpose an explicit barrier in that window and reproduce the
/// disappearance against the real lock, with no timing-dependent sleep.
///
/// `entry_dir` must be the canonical directory of `entry_path`
/// (`canonical_explicit_entry`) — the same function `probed_root` was derived
/// through, so the `root == entry_dir` tests below compare directory identity.
pub(super) fn open_explicit(
    probed_root: Option<PathBuf>,
    entry_dir: &Path,
    entry_path: &Path,
) -> Result<BuildTransaction, BuildError> {
    match probed_root {
        Some(root) => {
            let lock = lock_project(&root)?;
            // A temporary manifest is never a project. Its writer removes it
            // before releasing this lock, so a marked one still here belongs to
            // a killed build and is removed now. A manifest in the entry's own
            // directory that vanished while we waited was a live standalone
            // build's. Either way the search runs again from the entry, upward
            // past this directory, so a real project above is still found and
            // none at all gives a standalone build. Each pass removes a file or
            // finds one absent, so the search ends. A vanished manifest of any
            // other root is a real project losing it, and `load_manifest`
            // fails loudly. `root` and `entry_dir` must be one identity; a
            // lexical `entry_dir` made the co-location test false for every
            // standalone build under a symlinked parent.
            let manifest_path = root.join("Mind.toml");
            let removed =
                remove_leftover(&manifest_path).map_err(|e| BuildError::failed(e.to_string()))?;
            if removed || (root == *entry_dir && !manifest_path.exists()) {
                drop(lock);
                let reprobed = find_project_root_for_file_checked(entry_dir).map_err(|e| {
                    BuildError::Invalid(format!("cannot discover project root: {e:#}"))
                })?;
                return open_explicit(reprobed, entry_dir, entry_path);
            }
            let manifest = load_manifest(&root)
                .map_err(|e| BuildError::Invalid(format!("manifest error: {e}")))?;
            // One build rule: a named entry inside a project builds THAT
            // project, with the same module graph as `mindc build` with no
            // path, only the entry changes. Scoping a co-located bare manifest
            // to the entry alone dropped every module the entry imports, and
            // the build failed at the first call into one (`E2003`). The
            // leftover temporary manifest that scoping guarded against carries
            // the marker and was removed above, so only a real manifest is here.
            Ok(BuildTransaction {
                root,
                manifest,
                lock,
                single_file: false,
            })
        }
        None => {
            // The project lock also protects standalone builds. The manifest
            // is synthesised, so the selected scope is exactly the
            // named entry: a `Mind.toml` appearing in this directory while we
            // waited cannot widen this build's source set. Later stages still
            // read the root — the input snapshot, the cache root, the manifest
            // write — but as the root of that synthesised single-file project.
            let lock = lock_project(entry_dir)?;
            standalone(entry_dir.to_path_buf(), lock, entry_path)
        }
    }
}

/// Open the build transaction for `mindc build` with no path: the project the
/// current directory is in, by the same rule as [`open_explicit`] (see
/// `build_lock::lock_cwd_project`): a temporary manifest is never a project.
pub(super) fn open_project() -> Result<BuildTransaction, BuildError> {
    let lock = lock_cwd_project()
        .map_err(|e| BuildError::Invalid(format!("cannot locate Mind.toml: {e}")))?;
    let root = lock.root().to_path_buf();
    let manifest =
        load_manifest(&root).map_err(|e| BuildError::Invalid(format!("manifest error: {e}")))?;
    Ok(BuildTransaction {
        root,
        manifest,
        lock,
        single_file: false,
    })
}

/// The resolved explicit entry of `mindc build [PATH]`, or `None` when no path
/// was named.
///
/// One path at most. The modules an entry imports come from its project, so a
/// second path could only be ignored or read as an undeclared source set; both
/// are refused as bad usage.
pub(super) fn resolve_explicit_entry(
    paths: &[PathBuf],
) -> Result<Option<(PathBuf, PathBuf)>, BuildError> {
    let first = match paths {
        [] => return Ok(None),
        [first] => first,
        _ => {
            let named: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
            return Err(BuildError::Invalid(format!(
                "usage: `mindc build [PATH]` takes at most one entry path, got {}: {}. \
                 Modules the entry imports come from its project (Mind.toml), never from \
                 extra paths on the command line",
                paths.len(),
                named.join(" ")
            )));
        }
    };
    std::env::current_dir()
        .map_err(|error| anyhow::Error::new(error).context("cannot read current directory"))
        .and_then(|cwd| canonical_explicit_entry(first, &cwd))
        .map(Some)
        .map_err(|error| BuildError::Invalid(format!("cannot resolve explicit entry: {error:#}")))
}

/// A build with no governing manifest: a synthesised single-file project rooted
/// at the entry's own directory (never cwd nor a distant ancestor), so source
/// collection stays scoped to the entry rather than whatever tree sits above.
fn standalone(
    root: PathBuf,
    lock: ProjectBuildLock,
    entry_path: &Path,
) -> Result<BuildTransaction, BuildError> {
    refuse_local_imports(entry_path, &root)?;
    Ok(BuildTransaction {
        root,
        manifest: single_file_manifest(entry_path)?,
        lock,
        single_file: true,
    })
}

/// Refuse a standalone entry that imports a local (non-`std`) module.
///
/// A standalone build compiles exactly the named entry, so a local import has
/// no module to bind to. Without this refusal the build failed later, at the
/// first call into that module (`E2003 unsupported call`), and never said that
/// a project was missing. An unreadable or unparsable entry is left to the
/// stage that already reports it.
fn refuse_local_imports(entry_path: &Path, root: &Path) -> Result<(), BuildError> {
    let Ok(source) = std::fs::read_to_string(entry_path) else {
        return Ok(());
    };
    let Ok((_, imports)) = crate::parser::parse_with_imports(&source) else {
        return Ok(());
    };
    let local: Vec<String> = imports
        .into_iter()
        .filter(|path| path.split('.').next() != Some("std"))
        .collect();
    if local.is_empty() {
        return Ok(());
    }
    Err(BuildError::refused(
        "E2042",
        format!(
            "local import(s) {} in {} require an enclosing Mind.toml project, and no Mind.toml \
         governs {}: add a Mind.toml there (or in a parent directory) so the build compiles \
         the imported module(s) with the entry",
            local.join(", "),
            entry_path.display(),
            root.display()
        ),
    ))
}

fn single_file_manifest(entry: &Path) -> Result<ProjectManifest, BuildError> {
    let stem = entry
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
        format!("pkg_{stem}")
    };
    toml::from_str(&temporary_manifest(&pkg_name, "src/main.mind"))
        .map_err(|e| BuildError::Invalid(format!("synthetic manifest: {e}")))
}

fn lock_project(root: &Path) -> Result<ProjectBuildLock, BuildError> {
    ProjectBuildLock::acquire(root)
        .map_err(|e| BuildError::failed(format!("cannot lock project build: {e}")))
}
