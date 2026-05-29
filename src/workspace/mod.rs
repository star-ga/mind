// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase C — workspace support for `mindc build` and `mindc test`.
//!
//! ## Overview
//!
//! A workspace is a collection of MIND packages that share a common root
//! `Mind.toml` containing a `[workspace]` block. The root manifest may
//! optionally carry a `[package]` block (non-virtual) or omit it entirely
//! (virtual manifest).
//!
//! Public entry points:
//! - [`resolve_workspace_members`] — load and validate all member manifests.
//! - [`toposort_members`]          — topological sort by intra-workspace deps.
//!
//! Phase C scope: path deps *within* the workspace. External path deps (paths
//! that resolve outside the workspace root) are recorded but not built.
//! Git deps and semver deps are deferred to Phases D/E.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A resolved workspace member.
#[derive(Debug, Clone)]
pub struct WorkspaceMember {
    /// Package name from the member's `[package].name` field.
    pub name: String,
    /// Absolute path to the directory containing the member's `Mind.toml`.
    pub root: PathBuf,
    /// Path dependencies declared in the member's `[dependencies]` table.
    pub path_deps: Vec<PathDep>,
}

/// A `path = "..."` dependency entry.
#[derive(Debug, Clone)]
pub struct PathDep {
    /// The dependency key name (as declared in `[dependencies]`).
    pub name: String,
    /// Absolute resolved path to the dependency root.
    pub resolved_path: PathBuf,
    /// `true` when the resolved path is within the workspace root
    /// and the target directory has a valid `Mind.toml`.
    pub is_workspace_member: bool,
}

/// Options that modify workspace-level `mindc build` / `mindc test` behavior.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceOpts {
    /// When `Some(name)`, restrict the build/test to `name` and its
    /// transitive workspace dependencies. Corresponds to `--package <name>`
    /// / `-p <name>` on the CLI.
    pub package_filter: Option<String>,
}

impl WorkspaceOpts {
    /// Return the subset of `sorted` (already in topo order) that should be
    /// built given `self.package_filter`.
    ///
    /// When `package_filter` is `None`, `sorted` is returned unchanged.
    /// When it names a package, only that package and its transitive
    /// workspace-internal prerequisites are included, preserving their
    /// original topo order.
    pub fn filter_members<'a>(
        &self,
        all_members: &'a [WorkspaceMember],
        sorted: &'a [WorkspaceMember],
    ) -> Vec<&'a WorkspaceMember> {
        let target_name = match &self.package_filter {
            None => return sorted.iter().collect(),
            Some(n) => n.as_str(),
        };

        // Build name→member index.
        let by_name: HashMap<&str, &WorkspaceMember> =
            all_members.iter().map(|m| (m.name.as_str(), m)).collect();

        // BFS to collect the target and all its workspace-member prerequisites.
        let mut needed: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        queue.push_back(target_name);

        while let Some(name) = queue.pop_front() {
            if needed.contains(name) {
                continue;
            }
            needed.insert(name.to_string());
            if let Some(member) = by_name.get(name) {
                for dep in &member.path_deps {
                    if dep.is_workspace_member {
                        queue.push_back(dep.name.as_str());
                    }
                }
            }
        }

        // Preserve the topo order from `sorted`.
        sorted.iter().filter(|m| needed.contains(&m.name)).collect()
    }
}

/// Typed errors from workspace resolution and topological sort.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    /// A member declared in `[workspace.members]` is missing a `Mind.toml`.
    ///
    /// Exit code 1 (build failure — configuration is present but a required
    /// artifact is missing).
    #[error("member manifest not found: {0}")]
    MissingMemberManifest(PathBuf),

    /// A dependency cycle was detected in the workspace dep graph.
    ///
    /// Exit code 2 (invalid usage / configuration — the cycle must be broken
    /// before building can proceed).
    #[error("dependency cycle detected: {0}")]
    DependencyCycle(String),

    /// The workspace root `Mind.toml` could not be read or parsed.
    #[error("workspace manifest error: {0}")]
    ManifestError(String),
}

impl WorkspaceError {
    /// Process exit code per RFC 0008 §6.
    ///
    /// `DependencyCycle` → 2 (invalid usage/config).
    /// All others → 1 (build failure).
    pub fn exit_code(&self) -> i32 {
        match self {
            WorkspaceError::DependencyCycle(_) => 2,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// TOML schema (internal, not re-exported)
// ---------------------------------------------------------------------------

/// Partial manifest used only for workspace resolution — we need `[package]`
/// name, `[workspace]` block, and `[dependencies]` path entries.
#[derive(Deserialize)]
struct RawManifest {
    #[serde(default)]
    package: Option<RawPackage>,
    #[serde(default)]
    workspace: Option<RawWorkspace>,
    #[serde(default)]
    dependencies: HashMap<String, RawDep>,
}

#[derive(Debug, Deserialize)]
struct RawPackage {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RawWorkspace {
    #[serde(default)]
    members: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
}

/// Dependency spec — we only care about `path = "..."` entries for Phase C.
/// Other forms (semver string, git, etc.) are recorded but treated as
/// non-workspace deps.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawDep {
    /// Simple version string `foo = "1.0"` — no path.  The String is
    /// intentionally unused here; we only care about the Table variant.
    Simple(#[allow(dead_code)] String),
    /// Inline table — may contain `path = "..."`.
    Table(RawDepTable),
}

#[derive(Debug, Deserialize, Default)]
struct RawDepTable {
    #[serde(default)]
    path: Option<String>,
    // Other fields (version, git, rev, features) are intentionally ignored
    // here — Phase C only cares about path resolution.
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Load and validate all workspace members for the workspace rooted at `root`.
///
/// # Resolution algorithm
///
/// 1. Read `<root>/Mind.toml`; require a `[workspace]` block.
/// 2. For each pattern in `[workspace].members`:
///    a. If it contains a glob character (`*` or `?`), expand against the
///       filesystem; otherwise treat as a literal relative path.
/// 3. Remove any path whose relative form (from root) starts with an entry
///    in `[workspace].exclude`.
/// 4. For each surviving path, read its `Mind.toml` and extract `[package].name`
///    and `[dependencies]` path entries.
/// 5. Cross-reference path deps against the workspace member set to mark
///    `PathDep::is_workspace_member`.
///
/// Returns an unsorted list of members. Call [`toposort_members`] to get the
/// build order.
pub fn resolve_workspace_members(root: &Path) -> Result<Vec<WorkspaceMember>, WorkspaceError> {
    let manifest_path = root.join("Mind.toml");
    let text = fs::read_to_string(&manifest_path).map_err(|e| {
        WorkspaceError::ManifestError(format!("cannot read {}: {e}", manifest_path.display()))
    })?;
    let raw: RawManifest = toml::from_str(&text).map_err(|e| {
        WorkspaceError::ManifestError(format!("parse error in {}: {e}", manifest_path.display()))
    })?;

    let workspace_cfg = raw.workspace.ok_or_else(|| {
        WorkspaceError::ManifestError("no [workspace] block in root Mind.toml".into())
    })?;

    // Expand member patterns to concrete directories.
    let mut member_dirs: Vec<PathBuf> = Vec::new();
    for pattern in &workspace_cfg.members {
        let expanded = expand_glob_pattern(root, pattern);
        member_dirs.extend(expanded);
    }

    // Apply exclude patterns.
    member_dirs.retain(|dir| {
        let rel = dir
            .strip_prefix(root)
            .unwrap_or(dir)
            .to_string_lossy()
            .replace('\\', "/");
        for excl in &workspace_cfg.exclude {
            if glob_match_prefix(excl, &rel) {
                return false;
            }
        }
        true
    });

    // Deduplicate (glob expansion may produce duplicates).
    member_dirs.sort();
    member_dirs.dedup();

    // Collect absolute member roots.
    let abs_member_roots: HashSet<PathBuf> = member_dirs
        .iter()
        .map(|d| canonicalize_best_effort(d))
        .collect();

    // Load each member manifest.
    let mut members: Vec<WorkspaceMember> = Vec::new();
    for dir in &member_dirs {
        let member = load_member(dir, root, &abs_member_roots)?;
        members.push(member);
    }

    Ok(members)
}

/// Topologically sort workspace members by their intra-workspace path
/// dependencies using Kahn's algorithm.
///
/// Returns `Err(WorkspaceError::DependencyCycle)` when the dep graph contains
/// a cycle. The error message names the packages involved in the cycle.
pub fn toposort_members(
    members: &[WorkspaceMember],
) -> Result<Vec<WorkspaceMember>, WorkspaceError> {
    // Index: package name → position in `members`.
    let idx: HashMap<&str, usize> = members
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.as_str(), i))
        .collect();

    let n = members.len();

    // Build adjacency list (edges point from dep to dependent).
    // in_degree[i] = number of workspace deps that i depends on.
    let mut in_degree: Vec<usize> = vec![0; n];
    // edges[i] = list of packages that depend on i (i must build before them).
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, m) in members.iter().enumerate() {
        for dep in &m.path_deps {
            if !dep.is_workspace_member {
                continue;
            }
            if let Some(&j) = idx.get(dep.name.as_str()) {
                // i depends on j → j must be built before i.
                in_degree[i] += 1;
                edges[j].push(i);
            }
        }
    }

    // Kahn's algorithm.
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    let mut sorted: Vec<usize> = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        sorted.push(node);
        for &dependent in &edges[node] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if sorted.len() != n {
        // Find packages still in cycle (those with in_degree > 0).
        let cycle_names: Vec<&str> = (0..n)
            .filter(|&i| in_degree[i] > 0)
            .map(|i| members[i].name.as_str())
            .collect();
        return Err(WorkspaceError::DependencyCycle(cycle_names.join(" -> ")));
    }

    Ok(sorted.into_iter().map(|i| members[i].clone()).collect())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Load one member from `dir`, resolving its path deps against the workspace.
fn load_member(
    dir: &Path,
    _workspace_root: &Path,
    abs_member_roots: &HashSet<PathBuf>,
) -> Result<WorkspaceMember, WorkspaceError> {
    let manifest_path = dir.join("Mind.toml");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|_| WorkspaceError::MissingMemberManifest(dir.to_path_buf()))?;

    let raw: RawManifest = toml::from_str(&text).map_err(|e| {
        WorkspaceError::ManifestError(format!("parse error in {}: {e}", manifest_path.display()))
    })?;

    let pkg_name = raw
        .package
        .ok_or_else(|| {
            WorkspaceError::ManifestError(format!(
                "member at {} has no [package] block",
                dir.display()
            ))
        })?
        .name;

    // Resolve path deps.
    let mut path_deps: Vec<PathDep> = Vec::new();
    for (dep_name, dep_spec) in &raw.dependencies {
        let path_str = match dep_spec {
            RawDep::Simple(_) => continue, // no path, skip
            RawDep::Table(t) => match &t.path {
                Some(p) => p.clone(),
                None => continue,
            },
        };

        // Resolve path relative to the member directory.
        let resolved = canonicalize_best_effort(&dir.join(&path_str));
        let is_workspace_member = abs_member_roots.contains(&resolved);

        path_deps.push(PathDep {
            name: dep_name.clone(),
            resolved_path: resolved,
            is_workspace_member,
        });
    }

    Ok(WorkspaceMember {
        name: pkg_name,
        root: dir.to_path_buf(),
        path_deps,
    })
}

/// Expand a member pattern (which may contain glob characters) to a list of
/// existing directories under `workspace_root`.
///
/// For literal paths (no `*` or `?`), return a single entry without
/// filesystem access — even if it doesn't exist yet (the caller validates).
fn expand_glob_pattern(workspace_root: &Path, pattern: &str) -> Vec<PathBuf> {
    if !pattern.contains('*') && !pattern.contains('?') {
        // Literal path — no expansion needed.
        return vec![workspace_root.join(pattern)];
    }

    // Split the pattern at the first glob character to find the base dir.
    // e.g. "crates/*" → base = "crates", tail_glob = "*"
    let slash_pos = pattern
        .rfind(|c| c == '/')
        .map(|p| {
            // Walk back to find the last non-glob segment.
            let before = &pattern[..p];
            if before.contains('*') || before.contains('?') {
                // Complex pattern: walk from workspace root.
                None
            } else {
                Some(p)
            }
        })
        .flatten();

    let (base_dir, tail_pattern) = match slash_pos {
        Some(p) => (workspace_root.join(&pattern[..p]), &pattern[p + 1..]),
        None => (workspace_root.to_path_buf(), pattern),
    };

    // Walk base_dir and collect entries that match the tail glob.
    let mut result = Vec::new();
    let read = match fs::read_dir(&base_dir) {
        Ok(r) => r,
        Err(_) => return result,
    };

    for entry in read.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if glob_match_simple(tail_pattern, &name) {
            result.push(path);
        }
    }
    result.sort(); // deterministic order
    result
}

/// Simple single-component glob match (`*` matches any chars except `/`;
/// `?` matches one char except `/`; `**` is not needed for single-level
/// expansion).
fn glob_match_simple(pattern: &str, text: &str) -> bool {
    glob_inner_bytes(pattern.as_bytes(), text.as_bytes())
}

fn glob_inner_bytes(pat: &[u8], txt: &[u8]) -> bool {
    // `**` — matches everything including slashes.
    if pat.starts_with(b"**") {
        let rest = pat[2..].strip_prefix(b"/").unwrap_or(&pat[2..]);
        if rest.is_empty() {
            return true;
        }
        for i in 0..=txt.len() {
            if glob_inner_bytes(rest, &txt[i..]) {
                return true;
            }
        }
        return false;
    }

    match (pat.first(), txt.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(b'*'), _) => {
            let rest = &pat[1..];
            for i in 0..=txt.len() {
                if i > 0 && txt[i - 1] == b'/' {
                    break;
                }
                if glob_inner_bytes(rest, &txt[i..]) {
                    return true;
                }
            }
            false
        }
        (Some(b'?'), Some(&c)) if c != b'/' => glob_inner_bytes(&pat[1..], &txt[1..]),
        (Some(b'?'), _) => false,
        (Some(&p), Some(&t)) if p == t => glob_inner_bytes(&pat[1..], &txt[1..]),
        _ => false,
    }
}

/// Check whether `rel_path` starts with (or equals) `prefix`, where `prefix`
/// is an exclude pattern.
///
/// Examples:
///   `glob_match_prefix("scratch", "scratch/ignore-me")` → true
///   `glob_match_prefix("scratch", "crates/keep1")` → false
fn glob_match_prefix(prefix: &str, rel_path: &str) -> bool {
    // Exact match
    if rel_path == prefix {
        return true;
    }
    // Path starts with `<prefix>/`
    if rel_path.starts_with(&format!("{prefix}/")) {
        return true;
    }
    // Glob match
    glob_match_simple(prefix, rel_path)
}

/// Canonicalize a path using `canonicalize` when the path exists, falling
/// back to `std::path::absolute`-style normalization when it does not.
fn canonicalize_best_effort(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| normalize_path(path))
}

/// Normalize a path without hitting the filesystem (removes `.` and `..`
/// components lexically). Used as a fallback when `canonicalize` fails.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components: Vec<std::path::Component> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if matches!(components.last(), Some(std::path::Component::Normal(_))) {
                    components.pop();
                } else {
                    components.push(component);
                }
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

// ---------------------------------------------------------------------------
// Unit tests (module-private)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_simple_star_matches() {
        assert!(glob_match_simple("*", "anything"));
        assert!(glob_match_simple("lib_*", "lib_one"));
        assert!(glob_match_simple("lib_*", "lib_two"));
        assert!(!glob_match_simple("lib_*", "other_thing"));
    }

    #[test]
    fn glob_simple_question_matches() {
        assert!(glob_match_simple("lib?", "lib1"));
        assert!(!glob_match_simple("lib?", "libb_extra"));
    }

    #[test]
    fn glob_match_prefix_works() {
        assert!(glob_match_prefix("scratch", "scratch"));
        assert!(glob_match_prefix("scratch", "scratch/sub/dir"));
        assert!(!glob_match_prefix("scratch", "crates/scratch"));
    }

    #[test]
    fn normalize_path_removes_dotdot() {
        let p = PathBuf::from("/a/b/../c");
        assert_eq!(normalize_path(&p), PathBuf::from("/a/c"));
    }

    #[test]
    fn normalize_path_removes_curdot() {
        let p = PathBuf::from("/a/./b");
        assert_eq!(normalize_path(&p), PathBuf::from("/a/b"));
    }
}
