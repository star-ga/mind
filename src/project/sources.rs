// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! Project source-set resolution — the single owner of "which `.mind` files
//! are translation units of this build, and in what ORDER".
//!
//! Source ORDER is artifact-visible: it fixes the per-file compile sequence,
//! the object list handed to the linker, and (under `cross-module-imports`)
//! the fold order of the whole-project enum/const registries. A source set
//! that depends on `readdir` order therefore makes the emitted bytes depend on
//! filesystem layout — the cross-substrate byte-identity claim would hold only
//! by luck on two machines that happened to hash directory entries the same
//! way. [`collect_sources`] closes that by sorting the walked set on full path
//! bytes before it is ever returned.
//!
//! Two orders exist, and only one is sorted:
//!  - the DECLARED order of `[targets.*].sources` — the author's contract,
//!    preserved verbatim (see `resolve_declared_sources`);
//!  - the WALKED order of the entry-parent directory tree — no author intent
//!    exists, so it is canonicalised by sorting.
//!
//! [`resolve_sources`] is the one place that chooses between them; both the
//! legacy `build_project` compile path and the `run_build` cache key derive
//! their source set from it, so a build can never be keyed on one source set
//! and compiled from another.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Result, anyhow};

pub(super) fn canonical_project_root(project_root: &Path) -> Result<PathBuf> {
    // Path::parent("main.mind") is empty; discovery uses it for a manifest in
    // the current directory. Normalize that spelling before canonicalization.
    let project_root = if project_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        project_root
    };
    project_root.canonicalize().map_err(|e| {
        anyhow!(
            "cannot resolve project root {}: {e}",
            project_root.display()
        )
    })
}

fn project_path(project_root: &Path, raw: &str, kind: &str) -> Result<(PathBuf, PathBuf)> {
    let root = canonical_project_root(project_root)?;
    let relative = Path::new(raw);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(anyhow!(
            "{kind} \"{raw}\" must be project-root-relative (no absolute or \"..\")"
        ));
    }
    let joined = root.join(relative);
    let canonical = joined
        .canonicalize()
        .map_err(|e| anyhow!("{kind} \"{raw}\" cannot be resolved inside project root: {e}"))?;
    if !canonical.starts_with(&root) {
        return Err(anyhow!(
            "{kind} \"{raw}\" resolves outside project root {}",
            root.display()
        ));
    }
    Ok((root, canonical))
}

/// Resolve a manifest entry while enforcing the project-root boundary.
pub(crate) fn resolve_project_entry(project_root: &Path, entry: &str) -> Result<PathBuf> {
    let (_, path) = project_path(project_root, entry, "manifest entry")?;
    if !path.is_file() {
        return Err(anyhow!("Entry file not found: {}", path.display()));
    }
    Ok(path)
}

/// True only for the project-root `tests` directory, the `mindc test` tree.
/// A nested directory named `tests` is program source.
fn is_project_root_tests(path: &Path, root: &Path) -> bool {
    path == root.join("tests")
}

/// Collect all .mind source files from a project.
pub fn collect_sources(project_root: &Path, entry: &str) -> Result<Vec<PathBuf>> {
    let (root, entry_path) = project_path(project_root, entry, "manifest entry")?;
    if !entry_path.is_file() {
        return Err(anyhow!("Entry file not found: {}", entry_path.display()));
    }

    let src_dir = entry_path.parent().unwrap_or(&root);
    let mut sources = Vec::new();
    let mut visited_dirs = BTreeSet::new();
    let mut visited_files = BTreeSet::new();

    fn collect_recursive(
        dir: &Path,
        root: &Path,
        sources: &mut Vec<PathBuf>,
        visited_dirs: &mut BTreeSet<PathBuf>,
        visited_files: &mut BTreeSet<PathBuf>,
    ) -> Result<()> {
        let canonical_dir = dir
            .canonicalize()
            .map_err(|e| anyhow!("cannot resolve source directory {}: {e}", dir.display()))?;
        if !canonical_dir.starts_with(root) {
            return Err(anyhow!(
                "source directory {} resolves outside project root {}",
                dir.display(),
                root.display()
            ));
        }
        if !visited_dirs.insert(canonical_dir.clone()) {
            return Ok(());
        }

        // An unreadable directory remains a non-source subtree, as before. A
        // path that exists but cannot be canonicalised is an error because it
        // may be a broken or escaping symlink and must not be silently omitted.
        let rd = match fs::read_dir(&canonical_dir) {
            Ok(rd) => rd,
            Err(e) => {
                eprintln!(
                    "mindc: skipping unreadable dir {}: {e}",
                    canonical_dir.display()
                );
                return Ok(());
            }
        };
        // Refusals must be deterministic too: filesystem enumeration order must
        // not choose which escaping or broken source is reported first.
        let mut entries = rd.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                // Preserve the walk's existing pruning contract before following
                // a directory link. Metadata identifies a directory without
                // opening it, so a hidden/target link cannot redirect the walk.
                // Non-source asset links remain outside the source closure.
                let entry_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let metadata = match fs::metadata(&path) {
                    Ok(metadata) => metadata,
                    Err(e) => {
                        if path.extension().map(|ext| ext == "mind").unwrap_or(false) {
                            return Err(anyhow!(
                                "source symlink {} cannot be resolved: {e}",
                                path.display()
                            ));
                        }
                        continue;
                    }
                };
                if metadata.is_dir()
                    && (entry_name == "target"
                        || is_project_root_tests(&path, root)
                        || entry_name.starts_with('.'))
                {
                    continue;
                }
                let target = path.canonicalize().map_err(|e| {
                    anyhow!("source symlink {} cannot be resolved: {e}", path.display())
                })?;
                // The target, rather than the link spelling, determines whether
                // a regular file is a source. This preserves `.mind` source
                // aliases while keeping ordinary asset links outside the
                // source closure.
                if metadata.is_file()
                    && !target.extension().map(|ext| ext == "mind").unwrap_or(false)
                {
                    continue;
                }
                if !target.starts_with(root) {
                    return Err(anyhow!(
                        "source symlink {} resolves outside project root {}",
                        path.display(),
                        root.display()
                    ));
                }
                if target.is_dir() {
                    collect_recursive(&target, root, sources, visited_dirs, visited_files)?;
                } else if target.is_file()
                    && target.extension().map(|e| e == "mind").unwrap_or(false)
                    && visited_files.insert(target.clone())
                {
                    sources.push(target);
                }
                continue;
            }
            if file_type.is_dir() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                // `<project_root>/tests` is the `mindc test` tree, not a program
                // translation unit. A nested directory that is also named
                // `tests` is ordinary program source and stays in the walk.
                // The walk starts at the entry's parent, so an entry inside
                // the project-root tests tree still collects that directory.
                if name == "target" || is_project_root_tests(&path, root) || name.starts_with('.') {
                    continue;
                }
                collect_recursive(&path, root, sources, visited_dirs, visited_files)?;
            } else if file_type.is_file()
                && path.extension().map(|e| e == "mind").unwrap_or(false)
                && visited_files.insert(path.clone())
            {
                sources.push(path);
            }
        }
        Ok(())
    }

    collect_recursive(
        src_dir,
        &root,
        &mut sources,
        &mut visited_dirs,
        &mut visited_files,
    )?;

    // Canonicalise the walked order. `fs::read_dir` yields entries in whatever
    // order the filesystem stores them — creation order on tmpfs, name-HASH
    // order on ext4/xfs (seeded per filesystem), arbitrary elsewhere — so the
    // pre-sort order of this set was a property of the DISK, not of the
    // project. Every downstream consumer is order-sensitive (compile sequence,
    // link object order, the whole-project enum/const registry fold), so the
    // same sources on two machines could emit different bytes.
    //
    // Sort on the FULL path as raw `OsStr` bytes, not on `Path`'s
    // component-wise `Ord` and not on any locale collation: byte order is the
    // one total order that is identical on every host and every locale, which
    // is exactly the property the cross-substrate byte-identity claim needs.
    // The traversal order above is therefore irrelevant by construction — this
    // is the single point that fixes it.
    sources.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    Ok(sources)
}

/// Resolve a target's explicitly declared `sources = [...]` list against the
/// project root, preserving DECLARED order (no sorting — the manifest order is
/// the author's contract). Every declared path must exist: a missing declared
/// source fails the build loudly, because a silently dropped module is exactly
/// the class of bug an explicit source list exists to prevent. The manifest
/// entry is appended when the list omits it, so the entry is always a
/// translation unit of the build (the compile loop keys `is_entry` off it).
fn resolve_declared_sources(
    project_root: &Path,
    declared: &[String],
    entry: &str,
) -> Result<Vec<PathBuf>> {
    let root = canonical_project_root(project_root)?;
    let mut sources: Vec<PathBuf> = Vec::with_capacity(declared.len() + 1);
    for decl in declared {
        // Contract: project-root-relative, inside the root, no duplicates.
        // Symlink targets are canonicalised before the containment check.
        let path = project_path(&root, decl, "declared source")?.1;
        if !path.is_file() {
            return Err(anyhow!(
                "declared source not found: {} (listed as \"{}\" in Mind.toml [targets.*].sources)",
                path.display(),
                decl
            ));
        }
        if sources.contains(&path) {
            return Err(anyhow!(
                "declared source \"{}\" appears more than once in Mind.toml [targets.*].sources",
                decl
            ));
        }
        sources.push(path);
    }
    let entry_path = resolve_project_entry(&root, entry)?;
    let entry_canonical = entry_path.clone();
    let entry_listed = sources
        .iter()
        .any(|s| s.canonicalize().unwrap_or_else(|_| s.clone()) == entry_canonical);
    if !entry_listed {
        sources.push(entry_canonical);
    }
    Ok(sources)
}

/// Resolve the source set for ONE build: the ordered list of `.mind` files that
/// are translation units of it, plus whether that list came from an explicit
/// `[targets.*].sources` declaration (which changes cross-module keying to
/// project-root-relative paths — see `compile_sources`).
///
/// This is the ONLY selector. `build_project` compiles what it returns and
/// `run_build` fingerprints what it returns into the incremental cache key, so
/// the set a build is keyed on and the set it is compiled from cannot drift
/// apart — the drift that let an edit to a sibling module hit a cache entry
/// keyed on the entry alone and hand back a stale binary at exit 0.
///
/// - `single_file`: an explicit `mindc build <file>` with no governing project
///   manifest — the "project" is exactly the named entry, never its directory.
/// - `declared`: the target block's `sources = [...]` list, when present.
/// - otherwise: the entry-parent directory walk, sorted (see
///   [`collect_sources`]).
pub fn resolve_sources(
    project_root: &Path,
    entry: &str,
    declared: Option<&[String]>,
    single_file: bool,
) -> Result<(Vec<PathBuf>, bool)> {
    if single_file {
        let entry_path = project_root.join(entry);
        if !entry_path.exists() {
            return Err(anyhow!("Entry file not found: {}", entry_path.display()));
        }
        return Ok((vec![entry_path], false));
    }
    match declared {
        Some(list) => Ok((resolve_declared_sources(project_root, list, entry)?, true)),
        None => Ok((collect_sources(project_root, entry)?, false)),
    }
}

#[cfg(test)]
mod tests {
    use super::{collect_sources, resolve_sources};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn rejects_parent_and_absolute_manifest_entries() {
        let project = tempdir().expect("temp project");
        fs::create_dir_all(project.path().join("src")).expect("src");
        fs::write(project.path().join("src/main.mind"), "").expect("entry");

        assert!(resolve_sources(project.path(), "../src/main.mind", None, false).is_err());
        assert!(resolve_sources(project.path(), "/tmp/main.mind", None, false).is_err());
        let no_declared: &[String] = &[];
        assert!(
            resolve_sources(project.path(), "../src/main.mind", Some(no_declared), false).is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_outside_symlink_file_and_directory() {
        use std::os::unix::fs::symlink;

        let project = tempdir().expect("temp project");
        let outside = tempdir().expect("outside");
        fs::create_dir_all(project.path().join("src")).expect("src");
        fs::write(project.path().join("src/main.mind"), "").expect("entry");
        fs::write(outside.path().join("outside.mind"), "").expect("outside file");
        fs::create_dir(outside.path().join("modules")).expect("outside dir");
        fs::write(outside.path().join("modules/other.mind"), "").expect("outside module");
        symlink(
            outside.path().join("outside.mind"),
            project.path().join("src/file-link.mind"),
        )
        .expect("file link");
        assert!(collect_sources(project.path(), "src/main.mind").is_err());
        assert!(
            resolve_sources(
                project.path(),
                "src/main.mind",
                Some(&["src/file-link.mind".to_string()]),
                false
            )
            .is_err()
        );
        fs::remove_file(project.path().join("src/file-link.mind")).expect("remove file link");
        symlink(
            outside.path().join("modules"),
            project.path().join("src/dir-link"),
        )
        .expect("directory link");
        assert!(collect_sources(project.path(), "src/main.mind").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn accepts_in_root_symlinks_and_terminates_directory_cycle() {
        use std::os::unix::fs::symlink;

        let project = tempdir().expect("temp project");
        fs::create_dir_all(project.path().join("src/nested")).expect("src");
        fs::write(project.path().join("src/main.mind"), "").expect("entry");
        fs::write(project.path().join("src/nested/dep.mind"), "").expect("dep");
        symlink(
            project.path().join("src/nested/dep.mind"),
            project.path().join("src/dep-link.mind"),
        )
        .expect("file link");
        symlink(
            project.path().join("src"),
            project.path().join("src/nested/back"),
        )
        .expect("cycle link");

        let walked = collect_sources(project.path(), "src/main.mind").expect("bounded walk");
        assert_eq!(walked.len(), 2);
        assert!(walked.iter().any(|path| path.ends_with("main.mind")));
        assert!(walked.iter().any(|path| path.ends_with("dep.mind")));

        let (declared, explicit) = resolve_sources(
            project.path(),
            "src/main.mind",
            Some(&["src/dep-link.mind".to_string()]),
            false,
        )
        .expect("in-root declared link");
        assert!(explicit);
        assert_eq!(declared.len(), 2);
        assert!(declared[0].ends_with("dep.mind"));
        assert!(declared[1].ends_with("main.mind"));
    }

    #[cfg(unix)]
    #[test]
    fn prunes_hidden_target_links_and_ignores_asset_links() {
        use std::os::unix::fs::symlink;

        let project = tempdir().expect("temp project");
        let outside = tempdir().expect("outside");
        fs::create_dir_all(project.path().join("src")).expect("src");
        fs::write(project.path().join("src/main.mind"), "").expect("entry");
        fs::write(project.path().join("src/a.mind"), "").expect("a source");
        fs::write(project.path().join("src/z.mind"), "").expect("z source");
        fs::create_dir(outside.path().join("hidden")).expect("hidden target");
        fs::create_dir(outside.path().join("target")).expect("target");
        fs::write(outside.path().join("hidden/escaped.mind"), "").expect("hidden source");
        fs::write(outside.path().join("target/escaped.mind"), "").expect("target source");
        fs::write(outside.path().join("settings.toml"), "").expect("asset");
        fs::write(project.path().join("src/settings-local.toml"), "").expect("in-root asset");
        symlink(
            outside.path().join("hidden"),
            project.path().join("src/.git"),
        )
        .expect("hidden directory link");
        symlink(
            outside.path().join("target"),
            project.path().join("src/target"),
        )
        .expect("target directory link");
        symlink(
            outside.path().join("settings.toml"),
            project.path().join("src/settings-outside.toml"),
        )
        .expect("asset link");
        symlink(
            project.path().join("src/settings-local.toml"),
            project.path().join("src/settings-inside.toml"),
        )
        .expect("in-root asset link");
        symlink(
            project.path().join("src/a.mind"),
            project.path().join("src/source-alias"),
        )
        .expect("source extension alias");

        let walked = collect_sources(project.path(), "src/main.mind").expect("pruned walk");
        assert_eq!(
            walked.len(),
            3,
            "pruned/asset links leaked into sources: {walked:?}"
        );
        assert!(walked[0].ends_with("a.mind"));
        assert!(walked[1].ends_with("main.mind"));
        assert!(walked[2].ends_with("z.mind"));
    }

    #[cfg(unix)]
    #[test]
    fn source_escape_diagnostic_is_independent_of_creation_order() {
        use std::os::unix::fs::symlink;

        let project = tempdir().expect("temp project");
        let outside = tempdir().expect("outside");
        let src = project.path().join("src");
        fs::create_dir(&src).expect("src");
        fs::write(src.join("main.mind"), "").expect("entry");
        let target = outside.path().join("external.mind");
        fs::write(&target, "").expect("external source");
        let mut diagnostics = Vec::new();
        for order in [["z.mind", "a.mind"], ["a.mind", "z.mind"]] {
            for name in order {
                symlink(&target, src.join(name)).expect("escaping source link");
            }
            let diagnostic = collect_sources(project.path(), "src/main.mind")
                .expect_err("source escape must be refused")
                .to_string();
            assert!(diagnostic.contains("a.mind"), "{diagnostic}");
            diagnostics.push(diagnostic);
            for name in order {
                fs::remove_file(src.join(name)).expect("remove source link");
            }
        }
        assert_eq!(diagnostics[0], diagnostics[1]);
    }

    #[test]
    fn skips_real_project_root_tests_and_keeps_nested_tests() {
        let project = tempdir().expect("temp project");
        let root = project.path();
        fs::create_dir_all(root.join("src/tests")).expect("nested tests");
        fs::create_dir_all(root.join("tests")).expect("root tests");
        fs::write(root.join("main.mind"), "").expect("entry");
        fs::write(root.join("src/tests/kept.mind"), "").expect("nested source");
        fs::write(root.join("tests/hidden.mind"), "").expect("harness file");

        let walked = collect_sources(root, "main.mind").expect("walk");
        assert!(
            walked.iter().any(|path| path.ends_with("kept.mind")),
            "nested tests directory was dropped: {walked:?}"
        );
        assert!(
            walked.iter().any(|path| path.ends_with("main.mind")),
            "entry missing: {walked:?}"
        );
        assert!(
            walked.iter().all(|path| !path.ends_with("hidden.mind")),
            "project-root tests directory was compiled: {walked:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn skips_project_root_tests_symlink_and_keeps_nested_tests() {
        use std::os::unix::fs::symlink;

        let project = tempdir().expect("temp project");
        let outside = tempdir().expect("outside");
        let root = project.path();
        fs::write(outside.path().join("hidden.mind"), "").expect("hidden source");
        fs::create_dir_all(root.join("src/tests")).expect("nested tests");
        fs::write(root.join("main.mind"), "").expect("entry");
        fs::write(root.join("src/tests/kept.mind"), "").expect("nested source");
        symlink(outside.path(), root.join("tests")).expect("root tests link");

        let walked = collect_sources(root, "main.mind").expect("walk");
        assert!(
            walked.iter().any(|path| path.ends_with("kept.mind")),
            "nested tests directory was dropped: {walked:?}"
        );
        assert!(
            walked.iter().all(|path| !path.ends_with("hidden.mind")),
            "project-root tests link was followed: {walked:?}"
        );
    }
}
