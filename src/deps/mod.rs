// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase D + E — dependency resolution, lockfile, and cache.
//!
//! ## Overview
//!
//! - **Phase D**: external path dependencies with `tree_sha256` drift detection.
//! - **Phase E**: git dependencies, `~/.mindenv/cache/` content-addressed cache,
//!   `Mind.lock` TOML lockfile with mandatory enforcement (AP-2), and the
//!   `mindc lock`, `mindc fetch`, and `mindc clean` subcommands.
//!
//! ### Anti-pattern mitigations
//!
//! - **AP-1** (URL-only identity): git deps are identified by the triple
//!   `(git_url, rev, tree_sha256)`. A bare URL or rev is not sufficient.
//! - **AP-2** (optional lockfile): `Mind.lock` is *mandatory*. `mindc build`
//!   fails with a clear message if it is absent or stale.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::project::ProjectManifest;

// ---------------------------------------------------------------------------
// Public option types
// ---------------------------------------------------------------------------

/// Options for `mindc lock`.
#[derive(Debug, Clone, Default)]
pub struct LockOpts {
    /// If true, only verify the existing lockfile; do not write it.
    pub check: bool,
    /// Re-resolve only this package. `None` means resolve all.
    pub update_pkg: Option<String>,
}

/// Options for `mindc fetch`.
#[derive(Debug, Clone, Default)]
pub struct FetchOpts {
    /// Re-fetch git deps even if already cached. Does NOT modify lockfile.
    pub update: bool,
}

/// Options for `mindc clean`.
#[derive(Debug, Clone, Default)]
pub struct CleanOpts {
    /// Wipe `~/.mindenv/cache/git/<this-repo-deps>`.
    pub cache: bool,
    /// Wipe both `target/` and the entire `~/.mindenv/cache/`.
    pub all: bool,
}

/// Result of resolving + verifying dependencies.
#[derive(Debug, Default)]
pub struct ResolvedDeps {
    /// Ordered list of resolved entries (matches lockfile ordering).
    pub entries: Vec<LockEntry>,
}

// ---------------------------------------------------------------------------
// Mind.lock TOML schema
// ---------------------------------------------------------------------------

/// The on-disk `Mind.lock` file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MindLock {
    /// Format schema version — currently always 1.
    pub schema_version: u32,
    /// Ordered dep entries (sorted by name for determinism).
    #[serde(default)]
    pub dependencies: Vec<LockEntry>,
}

impl Default for MindLock {
    fn default() -> Self {
        Self {
            schema_version: 1,
            dependencies: Vec::new(),
        }
    }
}

/// A single resolved dependency entry in `Mind.lock`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct LockEntry {
    /// The dep key name from `[dependencies]`.
    pub name: String,
    /// Source URI:
    /// - `"path:../some-lib"` for path deps (relative to manifest).
    /// - `"git+https://github.com/owner/repo"` for git deps.
    pub source: String,
    /// Full 40-char commit SHA (git deps only; empty string for path deps).
    #[serde(default)]
    pub rev: String,
    /// SHA-256 of the canonical directory tree (both path and git deps).
    pub tree_sha256: String,
    /// Absolute resolved path on this machine.
    #[serde(default)]
    pub source_resolved: String,
}

// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------

/// Typed errors from the dependency subsystem.
#[derive(Debug, thiserror::Error)]
pub enum DepError {
    #[error("{0}")]
    LockMissing(String),
    #[error("{0}")]
    LockStale(String),
    #[error("{0}")]
    TreeDrifted(String),
    #[error("{0}")]
    FetchFailed(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Io(String),
}

impl DepError {
    /// Process exit code per RFC 0008 §6.
    pub fn exit_code(&self) -> i32 {
        match self {
            DepError::Invalid(_) => 2,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Dependency kind (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum DepKind {
    Path(String),
    Git {
        url: String,
        rev: Option<String>,
        tag: Option<String>,
        branch: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Verify that the lockfile is present, up-to-date, and that no dep has
/// drifted (tree_sha256 mismatch). Called by `mindc build`.
///
/// For projects with no `[dependencies]` entries (or only simple version
/// deps), this is a no-op (returns empty `ResolvedDeps`).
pub fn resolve_and_verify_deps(
    project_root: &Path,
    manifest: &ProjectManifest,
) -> Result<ResolvedDeps, DepError> {
    let relevant = collect_relevant_deps(manifest);
    if relevant.is_empty() {
        return Ok(ResolvedDeps::default());
    }

    // AP-2: lockfile MUST exist.
    let lock_path = project_root.join("Mind.lock");
    if !lock_path.exists() {
        return Err(DepError::LockMissing(
            "Mind.lock not found. Run 'mindc lock' to generate it.".to_string(),
        ));
    }

    let lock = load_lockfile(&lock_path)
        .map_err(|e| DepError::Io(format!("cannot read Mind.lock: {e}")))?;

    check_lockfile_staleness(&relevant, &lock)?;

    let mut entries: Vec<LockEntry> = Vec::new();
    for entry in &lock.dependencies {
        verify_entry(project_root, entry)?;
        entries.push(entry.clone());
    }

    Ok(ResolvedDeps { entries })
}

/// Regenerate `Mind.lock` from the current manifest.
pub fn run_lock(
    project_root: &Path,
    manifest: &ProjectManifest,
    opts: &LockOpts,
) -> Result<(), DepError> {
    let relevant = collect_relevant_deps(manifest);
    let lock_path = project_root.join("Mind.lock");

    if relevant.is_empty() && !opts.check {
        let lock = MindLock::default();
        write_lockfile(&lock_path, &lock)?;
        println!("   Locked 0 dependencies.");
        return Ok(());
    }

    let existing_lock: Option<MindLock> = if opts.update_pkg.is_some() && lock_path.exists() {
        load_lockfile(&lock_path).ok()
    } else {
        None
    };

    let mut new_entries: Vec<LockEntry> = Vec::new();

    for (name, kind) in &relevant {
        if let Some(update_name) = &opts.update_pkg {
            if name != update_name {
                if let Some(e) = existing_lock
                    .as_ref()
                    .and_then(|l| l.dependencies.iter().find(|e| &e.name == name))
                {
                    new_entries.push(e.clone());
                    continue;
                }
            }
        }
        let entry = resolve_dep_to_lock_entry(project_root, name, kind)?;
        new_entries.push(entry);
    }

    new_entries.sort_by(|a, b| a.name.cmp(&b.name));

    let new_lock = MindLock {
        schema_version: 1,
        dependencies: new_entries,
    };

    if opts.check {
        if !lock_path.exists() {
            return Err(DepError::LockMissing(
                "Mind.lock not found (--check). Run 'mindc lock' to generate it.".to_string(),
            ));
        }
        let existing = load_lockfile(&lock_path)
            .map_err(|e| DepError::Io(format!("cannot read Mind.lock: {e}")))?;
        if existing.dependencies != new_lock.dependencies {
            return Err(DepError::LockStale(
                "Mind.lock is stale (--check). Run 'mindc lock' to update it.".to_string(),
            ));
        }
        println!("   Mind.lock is up to date.");
        return Ok(());
    }

    write_lockfile(&lock_path, &new_lock)?;
    println!("   Locked {} dependencies.", new_lock.dependencies.len());
    Ok(())
}

/// Populate `~/.mindenv/cache/` from `Mind.lock`. Idempotent.
pub fn run_fetch(project_root: &Path, opts: &FetchOpts) -> Result<(), DepError> {
    let lock_path = project_root.join("Mind.lock");
    if !lock_path.exists() {
        return Err(DepError::LockMissing(
            "Mind.lock not found. Run 'mindc lock' to generate it.".to_string(),
        ));
    }

    let lock = load_lockfile(&lock_path)
        .map_err(|e| DepError::Io(format!("cannot read Mind.lock: {e}")))?;

    let mut fetched = 0usize;
    let mut cached_count = 0usize;

    for entry in &lock.dependencies {
        if entry.source.starts_with("git+") {
            let cache_dir = git_cache_dir(entry)
                .map_err(|e| DepError::Io(format!("cache dir for '{}': {e}", entry.name)))?;
            if cache_dir.exists() && !opts.update {
                cached_count += 1;
                continue;
            }
            fetch_git_dep_into(entry, &cache_dir)?;
            fetched += 1;
        }
    }

    println!("   Fetched {fetched} deps, {cached_count} already cached.");
    Ok(())
}

/// Remove build artifacts and/or cache entries.
pub fn run_clean(project_root: &Path, opts: &CleanOpts) -> Result<(), DepError> {
    let mindenv_cache = mindenv_cache_root();

    if opts.all {
        let target_dir = project_root.join("target");
        if target_dir.exists() {
            fs::remove_dir_all(&target_dir)
                .map_err(|e| DepError::Io(format!("cannot remove target/: {e}")))?;
            println!("   Removed {}", target_dir.display());
        }
        if mindenv_cache.exists() {
            match fs::remove_dir_all(&mindenv_cache) {
                Ok(()) => println!("   Removed {}", mindenv_cache.display()),
                Err(e) => eprintln!(
                    "warning: could not fully remove cache ({}): {e}",
                    mindenv_cache.display()
                ),
            }
        }
        return Ok(());
    }

    if opts.cache {
        let lock_path = project_root.join("Mind.lock");
        if lock_path.exists() {
            if let Ok(lock) = load_lockfile(&lock_path) {
                for entry in &lock.dependencies {
                    if entry.source.starts_with("git+") {
                        if let Ok(cache_dir) = git_cache_dir(entry) {
                            // H1 fail-closed: `cache_dir` is derived from
                            // attacker-controlled lockfile fields. Even with the
                            // component sanitisation in `git_cache_dir`,
                            // canonicalize the target and REQUIRE it to live
                            // strictly under the cache root before a recursive
                            // delete — a crafted `Mind.lock` must never make
                            // `mindc clean --cache` remove a directory outside
                            // `~/.mindenv/cache` (e.g. `~/.ssh`).
                            if cache_dir.exists() {
                                let root = mindenv_cache_root();
                                match (cache_dir.canonicalize(), root.canonicalize()) {
                                    (Ok(cd), Ok(r)) if cd.starts_with(&r) && cd != r => {
                                        let _ = fs::remove_dir_all(&cd);
                                        println!("   Removed cache for {}", entry.name);
                                    }
                                    _ => eprintln!(
                                        "warning[clean]: refusing to remove out-of-cache path for `{}` — lockfile entry resolves outside {}",
                                        entry.name,
                                        root.display()
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    // Default: wipe target/ only.
    let target_dir = project_root.join("target");
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir)
            .map_err(|e| DepError::Io(format!("cannot remove target/: {e}")))?;
        println!("   Removed {}", target_dir.display());
    } else {
        println!("   Nothing to clean.");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers — dep collection
// ---------------------------------------------------------------------------

fn collect_relevant_deps(manifest: &ProjectManifest) -> Vec<(String, DepKind)> {
    use crate::project::DependencySpec;

    let mut out: Vec<(String, DepKind)> = manifest
        .dependencies
        .iter()
        .filter_map(|(name, spec)| match spec {
            DependencySpec::Path { path, .. } => Some((name.clone(), DepKind::Path(path.clone()))),
            DependencySpec::Git {
                git,
                rev,
                tag,
                branch,
                ..
            } => Some((
                name.clone(),
                DepKind::Git {
                    url: git.clone(),
                    rev: rev.clone(),
                    tag: tag.clone(),
                    branch: branch.clone(),
                },
            )),
            _ => None,
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

// ---------------------------------------------------------------------------
// Internal helpers — lockfile I/O
// ---------------------------------------------------------------------------

fn load_lockfile(path: &Path) -> Result<MindLock> {
    let text =
        fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let lock: MindLock =
        toml::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))?;
    Ok(lock)
}

fn write_lockfile(path: &Path, lock: &MindLock) -> Result<(), DepError> {
    let header = "# Auto-generated by mindc lock. DO NOT edit by hand.\n";
    let body = toml::to_string_pretty(lock)
        .map_err(|e| DepError::Io(format!("cannot serialise lockfile: {e}")))?;
    let full = format!("{header}{body}");
    fs::write(path, full).map_err(|e| DepError::Io(format!("cannot write {}: {e}", path.display())))
}

// ---------------------------------------------------------------------------
// Internal helpers — staleness and verification
// ---------------------------------------------------------------------------

fn check_lockfile_staleness(
    relevant: &[(String, DepKind)],
    lock: &MindLock,
) -> Result<(), DepError> {
    let lock_names: BTreeSet<&str> = lock.dependencies.iter().map(|e| e.name.as_str()).collect();
    let manifest_names: BTreeSet<&str> = relevant.iter().map(|(n, _)| n.as_str()).collect();

    let missing: Vec<&str> = manifest_names
        .iter()
        .copied()
        .filter(|n| !lock_names.contains(n))
        .collect();
    let extra: Vec<&str> = lock_names
        .iter()
        .copied()
        .filter(|n| !manifest_names.contains(n))
        .collect();

    if !missing.is_empty() || !extra.is_empty() {
        let mut msg = "Mind.lock is stale relative to Mind.toml.".to_string();
        if !missing.is_empty() {
            msg.push_str(&format!(
                "\n  Missing from Mind.lock: {}",
                missing.join(", ")
            ));
        }
        if !extra.is_empty() {
            msg.push_str(&format!(
                "\n  Extra in Mind.lock (not in manifest): {}",
                extra.join(", ")
            ));
        }
        msg.push_str("\nRun 'mindc lock' to update.");
        return Err(DepError::LockStale(msg));
    }

    for (name, kind) in relevant {
        if let Some(entry) = lock.dependencies.iter().find(|e| &e.name == name) {
            let expected = match kind {
                DepKind::Path(_) => "path:",
                DepKind::Git { .. } => "git+",
            };
            if !entry.source.starts_with(expected) {
                return Err(DepError::LockStale(format!(
                    "Mind.lock entry for '{}' has source '{}' but manifest expects a {} dep. \
                     Run 'mindc lock' to update.",
                    name, entry.source, expected
                )));
            }
        }
    }
    Ok(())
}

fn verify_entry(project_root: &Path, entry: &LockEntry) -> Result<(), DepError> {
    if entry.source.starts_with("path:") {
        let path_spec = entry.source.strip_prefix("path:").unwrap_or("");
        let dep_path = if Path::new(path_spec).is_absolute() {
            PathBuf::from(path_spec)
        } else {
            project_root.join(path_spec)
        };

        if !dep_path.exists() {
            return Err(DepError::TreeDrifted(format!(
                "dependency '{}' not found at path '{}'. Run 'mindc lock' to update.",
                entry.name,
                dep_path.display()
            )));
        }

        let actual = compute_tree_sha256(&dep_path)
            .map_err(|e| DepError::Io(format!("hash tree '{}': {e}", entry.name)))?;

        if actual != entry.tree_sha256 {
            return Err(DepError::TreeDrifted(format!(
                "dependency '{}' has drifted from Mind.lock (tree_sha256 mismatch).\n  \
                 expected: {}\n  actual:   {}\n\
                 Run 'mindc lock' to update.",
                entry.name, entry.tree_sha256, actual
            )));
        }
    } else if entry.source.starts_with("git+") {
        let cache_dir = git_cache_dir(entry)
            .map_err(|e| DepError::Io(format!("cache dir '{}': {e}", entry.name)))?;

        if !cache_dir.exists() {
            return Err(DepError::FetchFailed(format!(
                "dependency '{}' not in cache ({}). Run 'mindc fetch' first.",
                entry.name,
                cache_dir.display()
            )));
        }

        let actual = compute_tree_sha256(&cache_dir)
            .map_err(|e| DepError::Io(format!("hash tree '{}': {e}", entry.name)))?;

        if actual != entry.tree_sha256 {
            return Err(DepError::TreeDrifted(format!(
                "cached dependency '{}' has drifted from Mind.lock (tree_sha256 mismatch).\n  \
                 expected: {}\n  actual:   {}\n\
                 Run 'mindc lock' and 'mindc fetch' to update.",
                entry.name, entry.tree_sha256, actual
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers — dep resolution to lock entry
// ---------------------------------------------------------------------------

fn resolve_dep_to_lock_entry(
    project_root: &Path,
    name: &str,
    kind: &DepKind,
) -> Result<LockEntry, DepError> {
    match kind {
        DepKind::Path(path_spec) => {
            let dep_path = if Path::new(path_spec).is_absolute() {
                PathBuf::from(path_spec)
            } else {
                project_root.join(path_spec)
            };

            if !dep_path.exists() {
                return Err(DepError::Invalid(format!(
                    "path dependency '{}' not found at '{}'",
                    name,
                    dep_path.display()
                )));
            }

            let tree_sha256 = compute_tree_sha256(&dep_path)
                .map_err(|e| DepError::Io(format!("hash tree '{name}': {e}")))?;

            let source_resolved = dep_path
                .canonicalize()
                .unwrap_or_else(|_| dep_path.clone())
                .to_string_lossy()
                .to_string();

            Ok(LockEntry {
                name: name.to_string(),
                source: format!("path:{path_spec}"),
                rev: String::new(),
                tree_sha256,
                source_resolved,
            })
        }
        DepKind::Git {
            url,
            rev,
            tag,
            branch,
        } => resolve_git_dep(name, url, rev.as_deref(), tag.as_deref(), branch.as_deref()),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — git fetching
// ---------------------------------------------------------------------------

fn resolve_git_dep(
    name: &str,
    url: &str,
    rev: Option<&str>,
    tag: Option<&str>,
    branch: Option<&str>,
) -> Result<LockEntry, DepError> {
    let fetch_ref = rev
        .or(tag)
        .or(branch)
        .ok_or_else(|| {
            DepError::Invalid(format!(
                "git dep '{}' must specify rev, tag, or branch",
                name
            ))
        })?
        .to_string();

    let tmp = tempfile::tempdir().map_err(|e| DepError::Io(format!("temp dir: {e}")))?;
    let clone_dir = tmp.path().join("repo");

    let ok = shallow_clone(url, &fetch_ref, &clone_dir).is_ok();
    if !ok {
        full_clone_and_checkout(url, &fetch_ref, &clone_dir).map_err(|e| {
            DepError::FetchFailed(format!(
                "cannot fetch git dep '{}' ({}@{}): {e}",
                name, url, fetch_ref
            ))
        })?;
    }

    let full_sha = git_rev_parse(&clone_dir, "HEAD")
        .map_err(|e| DepError::FetchFailed(format!("rev-parse HEAD for '{}': {e}", name)))?;

    let tree_sha256 = compute_tree_sha256(&clone_dir)
        .map_err(|e| DepError::Io(format!("hash tree '{name}': {e}")))?;

    // Stage into permanent cache.
    let provisional_entry = LockEntry {
        name: name.to_string(),
        source: format!("git+{url}"),
        rev: full_sha.clone(),
        tree_sha256: tree_sha256.clone(),
        source_resolved: String::new(),
    };
    let cache_dir =
        git_cache_dir(&provisional_entry).map_err(|e| DepError::Io(format!("cache dir: {e}")))?;

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .map_err(|e| DepError::Io(format!("create cache dir: {e}")))?;
        copy_dir_all(&clone_dir, &cache_dir)
            .map_err(|e| DepError::Io(format!("copy to cache: {e}")))?;
        fs::write(cache_dir.join(".mind_tree_sha256"), &tree_sha256)
            .map_err(|e| DepError::Io(format!("write sentinel: {e}")))?;
    }

    Ok(LockEntry {
        name: name.to_string(),
        source: format!("git+{url}"),
        rev: full_sha,
        tree_sha256,
        source_resolved: cache_dir.to_string_lossy().to_string(),
    })
}

fn fetch_git_dep_into(entry: &LockEntry, cache_dir: &Path) -> Result<(), DepError> {
    let url = entry.source.strip_prefix("git+").unwrap_or(&entry.source);

    let tmp = tempfile::tempdir().map_err(|e| DepError::Io(format!("temp dir: {e}")))?;
    let clone_dir = tmp.path().join("repo");

    let ok = shallow_clone(url, &entry.rev, &clone_dir).is_ok();
    if !ok {
        full_clone_and_checkout(url, &entry.rev, &clone_dir)
            .map_err(|e| DepError::FetchFailed(format!("cannot fetch '{}': {e}", entry.name)))?;
    }

    if cache_dir.exists() {
        fs::remove_dir_all(cache_dir)
            .map_err(|e| DepError::Io(format!("remove old cache: {e}")))?;
    }
    fs::create_dir_all(cache_dir).map_err(|e| DepError::Io(format!("create cache dir: {e}")))?;
    copy_dir_all(&clone_dir, cache_dir).map_err(|e| DepError::Io(format!("copy to cache: {e}")))?;
    fs::write(cache_dir.join(".mind_tree_sha256"), &entry.tree_sha256)
        .map_err(|e| DepError::Io(format!("write sentinel: {e}")))?;
    Ok(())
}

fn shallow_clone(url: &str, git_ref: &str, dest: &Path) -> Result<()> {
    let out = Command::new("git")
        .args([
            "clone",
            "--depth=1",
            "--branch",
            git_ref,
            "--",
            url,
            &dest.to_string_lossy(),
        ])
        .output()
        .context("git clone")?;
    if !out.status.success() {
        return Err(anyhow!(
            "git clone --depth=1 failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn full_clone_and_checkout(url: &str, sha: &str, dest: &Path) -> Result<()> {
    let out = Command::new("git")
        .args(["clone", "--", url, &dest.to_string_lossy()])
        .output()
        .context("git clone")?;
    if !out.status.success() {
        return Err(anyhow!(
            "git clone failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let out = Command::new("git")
        .args(["checkout", sha])
        .current_dir(dest)
        .output()
        .context("git checkout")?;
    if !out.status.success() {
        return Err(anyhow!(
            "git checkout failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn git_rev_parse(repo_dir: &Path, rev: &str) -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", rev])
        .current_dir(repo_dir)
        .output()
        .context("git rev-parse")?;
    if !out.status.success() {
        return Err(anyhow!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// Cache path helpers
// ---------------------------------------------------------------------------

/// Root of the MIND environment cache: `~/.mindenv/cache/`.
pub fn mindenv_cache_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".mindenv")
        .join("cache")
}

/// `~/.mindenv/cache/git/<hostname>/<owner>/<repo>/<rev>/`
fn git_cache_dir(entry: &LockEntry) -> Result<PathBuf> {
    let url = entry
        .source
        .strip_prefix("git+")
        .ok_or_else(|| anyhow!("not a git source: {}", entry.source))?;

    let (hostname, path_part) = parse_git_url(url)?;
    let path_part = path_part
        .trim_start_matches('/')
        .trim_end_matches('/')
        .trim_end_matches(".git");

    let rev = if entry.rev.is_empty() {
        "unknown"
    } else {
        &entry.rev
    };
    // H1: `hostname` / `path_part` / `rev` come verbatim from an untrusted
    // `Mind.lock`. Reject any component that could escape the cache root BEFORE
    // it is joined into a `remove_dir_all` target: `..` traversal, an embedded
    // NUL, or an absolute path — the last is critical because `PathBuf::join`
    // silently DISCARDS the accumulated prefix when the joined component is
    // absolute (so `rev = "/home/<op>/.ssh"` would make the target that path
    // verbatim). The sink in `run_clean` also canonicalizes-and-contains.
    for comp in [hostname.as_str(), path_part, rev] {
        // `is_absolute()` is platform-specific: on Windows a Unix-style
        // "/home/…" is NOT absolute (no drive letter), yet `PathBuf::join` still
        // lets a leading "/" or "\\" REPLACE the accumulated prefix. Reject a
        // leading separator explicitly so a `/`-rooted `rev` is caught on every
        // host — not only where `is_absolute()` happens to flag it.
        if comp.contains("..")
            || comp.contains('\0')
            || comp.starts_with('/')
            || comp.starts_with('\\')
            || std::path::Path::new(comp).is_absolute()
        {
            return Err(anyhow!(
                "illegal path component in Mind.lock entry (possible cache-path traversal): {comp:?}"
            ));
        }
    }
    Ok(mindenv_cache_root()
        .join("git")
        .join(hostname)
        .join(path_part)
        .join(rev))
}

fn parse_git_url(url: &str) -> Result<(String, String)> {
    if let Some(rest) = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
    {
        let slash = rest.find('/').unwrap_or(rest.len());
        return Ok((rest[..slash].to_string(), rest[slash..].to_string()));
    }
    if let Some(rest) = url.strip_prefix("git@") {
        let colon = rest.find(':').unwrap_or(rest.len());
        return Ok((rest[..colon].to_string(), rest[colon + 1..].to_string()));
    }
    // Local paths (bare repos in tests, file:// URLs).
    Ok(("localhost".to_string(), url.to_string()))
}

// ---------------------------------------------------------------------------
// Tree SHA-256 computation
// ---------------------------------------------------------------------------

/// Compute a canonical SHA-256 of a directory tree.
///
/// Algorithm:
///   1. Walk the tree in sorted order (by relative POSIX path).
///   2. Skip `.git/` and `.mind_tree_sha256`.
///   3. For each file: `file_sha256 = SHA256(file_bytes)`.
///   4. Accumulate `"<rel_path>\0<file_sha256_hex>\0"` into outer hasher.
///   5. Return lower-hex of the outer digest.
pub fn compute_tree_sha256(dir: &Path) -> Result<String> {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    collect_files(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut outer_data: Vec<u8> = Vec::new();
    for (rel, contents) in &files {
        let file_hash = sha256_bytes(contents);
        outer_data.extend_from_slice(rel.as_bytes());
        outer_data.push(b'\0');
        outer_data.extend_from_slice(file_hash.as_bytes());
        outer_data.push(b'\0');
    }
    Ok(sha256_bytes(&outer_data))
}

fn collect_files(base: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();

        if name == ".git" || name == ".mind_tree_sha256" {
            continue;
        }

        let rel = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        if path.is_dir() {
            // Skip a subdir we cannot read (e.g. a root-owned `/tmp/systemd-private-*`
            // sibling of a source file compiled from `/tmp`) instead of failing the
            // whole build with EACCES — an unreadable sibling directory is never part
            // of the MIND module tree, so silently excluding it is correct and keeps
            // `mindc` usable from any working directory.
            if let Err(e) = collect_files(base, &path, out) {
                eprintln!("mindc: skipping unreadable dir {}: {e}", path.display());
            }
        } else if path.is_file() {
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            out.push((rel, bytes));
        }
    }
    Ok(())
}

/// SHA-256 of raw bytes, returned as a lower-hex string.
fn sha256_bytes(data: &[u8]) -> String {
    mini_sha256(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Minimal self-contained SHA-256 (FIPS 180-4).
// Avoids any feature-gated external dependency.
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (i, v) in [a, b, c, d, e, f, g, h].iter().enumerate() {
        state[i] = state[i].wrapping_add(*v);
    }
}

/// FIPS 180-4 SHA-256 of raw bytes. Used by the dependency-lock machinery
/// (`sha256_bytes`) and by RFC 0016 evidence-chain `trace_hash` computation
/// (`crate::ir::compact::v2::evidence`). The pure-MIND `std.sha256` runs the
/// identical algorithm over the identical bytes, so the two are bit-identical
/// — that equivalence is what makes the trace_hash substrate-portable.
pub(crate) fn mini_sha256(data: &[u8]) -> [u8; 32] {
    let mut state = SHA256_H0;
    let bit_len = (data.len() as u64) * 8;

    // Build padded message.
    let mut msg: Vec<u8> = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        sha256_compress(&mut state, chunk.try_into().unwrap());
    }

    let mut out = [0u8; 32];
    for (i, &word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Utility: recursive directory copy (skipping .git)
// ---------------------------------------------------------------------------

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)?.flatten() {
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod h1_cache_traversal_tests {
    use super::*;

    fn entry(source: &str, rev: &str) -> LockEntry {
        LockEntry {
            name: "victim".into(),
            source: source.into(),
            rev: rev.into(),
            tree_sha256: String::new(),
            source_resolved: String::new(),
        }
    }

    // H1: a crafted `Mind.lock` must not let `git_cache_dir` build a
    // `remove_dir_all` target that escapes the cache root. An absolute `rev`
    // (PathBuf::join would discard the cache-root prefix), a `..`-traversal
    // component, or `..` smuggled through the git URL are all rejected.
    #[test]
    fn git_cache_dir_rejects_absolute_rev() {
        let e = entry("git+https://github.com/o/r", "/home/victim/.ssh");
        assert!(
            git_cache_dir(&e).is_err(),
            "absolute rev must be rejected as cache-path traversal"
        );
    }

    #[test]
    fn git_cache_dir_rejects_dotdot_rev() {
        let e = entry("git+https://github.com/o/r", "../../../../victim");
        assert!(
            git_cache_dir(&e).is_err(),
            "..-traversal rev must be rejected as cache-path traversal"
        );
    }

    #[test]
    fn git_cache_dir_rejects_dotdot_in_git_path() {
        let e = entry("git+https://github.com/../../../etc", "abcdef");
        assert!(
            git_cache_dir(&e).is_err(),
            ".. in the git path must be rejected"
        );
    }

    #[test]
    fn git_cache_dir_accepts_benign_entry() {
        let e = entry(
            "git+https://github.com/star-ga/mind",
            "0123456789abcdef0123456789abcdef01234567",
        );
        let p = git_cache_dir(&e).expect("benign entry must resolve");
        assert!(
            p.starts_with(mindenv_cache_root()),
            "benign cache dir must live under the cache root"
        );
    }
}
