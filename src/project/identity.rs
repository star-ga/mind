// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Canonical directory identity and bounded explicit-file project discovery.

use std::path::{Path, PathBuf};

use anyhow::Result;

/// The one directory identity used at a build-transaction boundary.
pub(crate) fn canonical_dir(dir: &Path) -> Result<Option<PathBuf>> {
    let absolute = if dir.is_absolute() {
        dir.to_path_buf()
    } else {
        let Ok(cwd) = std::env::current_dir() else {
            return Ok(None);
        };
        cwd.join(dir)
    };
    Ok(Some(canonicalize_dir_with_missing_fallback(
        absolute,
        |path| path.canonicalize(),
    )?))
}

/// Preserve missing-parent discovery while propagating all other identity errors.
pub(crate) fn canonicalize_dir_with_missing_fallback<F>(
    absolute: PathBuf,
    canonicalize: F,
) -> Result<PathBuf>
where
    F: FnOnce(&Path) -> std::io::Result<PathBuf>,
{
    match canonicalize(&absolute) {
        Ok(canonical) => Ok(canonical),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(absolute),
        Err(error) => Err(anyhow::Error::new(error).context(format!(
            "cannot canonicalize directory {}",
            absolute.display()
        ))),
    }
}

/// `find_project_root` by the rule of the build lock (a temporary manifest is
/// never a project, `build_lock::governing_root`), for a command that reads the
/// project of the current directory without holding a build lock. Never call
/// it under a held lock: an in-doubt manifest takes that lock, and a build's
/// own temporary manifest is its project (`link_binary` stays on the raw search).
pub fn governing_project_root() -> Result<PathBuf> {
    // The search answers a root or an error, never `None`.
    let root = super::build_lock::governing_root(|| super::find_project_root().map(Some))?;
    root.ok_or_else(|| {
        anyhow::anyhow!("Could not find Mind.toml in current directory or any parent")
    })
}

/// Resolve the governing project root for an explicit source within a bounded
/// repository or a co-located scratch manifest.
pub fn find_project_root_for_file(entry_dir: &Path) -> Option<PathBuf> {
    find_project_root_for_file_checked(entry_dir).ok().flatten()
}

/// Fallible companion used by build and project-scope callers.
pub(crate) fn find_project_root_for_file_checked(entry_dir: &Path) -> Result<Option<PathBuf>> {
    let invocation_dir = std::env::current_dir().ok();
    find_project_root_for_file_checked_at(entry_dir, invocation_dir.as_deref())
}

/// Resolve an explicit entry against a caller-supplied invocation directory.
///
/// A source tree extracted from a git archive has no `.git` boundary. In that
/// one case, an invocation made from a directory containing `Mind.toml` may
/// still govern a nested entry, but only when the canonical entry directory
/// remains below that invocation directory. Passing `None` preserves the
/// historical standalone behavior and disables this archive-only fallback.
pub(crate) fn find_project_root_for_file_checked_at(
    entry_dir: &Path,
    invocation_dir: Option<&Path>,
) -> Result<Option<PathBuf>> {
    let Some(entry_dir) = canonical_dir(entry_dir)? else {
        return Ok(None);
    };

    let invocation_dir = invocation_dir.map(canonical_dir).transpose()?.flatten();

    let mut git_root: Option<PathBuf> = None;
    let mut probe = entry_dir.clone();
    loop {
        if probe.join(".git").exists() {
            git_root = Some(probe.clone());
            break;
        }
        if !probe.pop() {
            break;
        }
    }

    match git_root {
        Some(root) => {
            let mut current = entry_dir.clone();
            loop {
                if current.join("Mind.toml").exists() {
                    return Ok(Some(current));
                }
                if current == root {
                    return Ok(None);
                }
                if !current.pop() {
                    return Ok(None);
                }
            }
        }
        None => {
            if entry_dir.join("Mind.toml").exists() {
                Ok(Some(entry_dir))
            } else if let Some(invocation_root) = invocation_dir {
                // The invocation root is the explicit trust boundary for an
                // archive checkout. Do not climb to an unrelated parent, and
                // canonicalise the manifest so a symlink cannot import a file
                // from outside that boundary.
                let manifest = invocation_root.join("Mind.toml");
                let manifest_inside = manifest
                    .canonicalize()
                    .map(|path| path.starts_with(&invocation_root))
                    .unwrap_or(false);
                if entry_dir.starts_with(&invocation_root) && manifest_inside {
                    Ok(Some(invocation_root))
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::find_project_root_for_file_checked_at;
    use std::fs;
    use tempfile::TempDir;

    fn manifest(root: &std::path::Path) {
        fs::write(
            root.join("Mind.toml"),
            "[package]\nname = \"archive\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
    }

    #[test]
    fn archive_invocation_root_adopts_nested_entry_without_git() {
        let td = TempDir::new().unwrap();
        manifest(td.path());
        let entry = td.path().join("docs").join("mindc-repros");
        fs::create_dir_all(&entry).unwrap();

        let got = find_project_root_for_file_checked_at(&entry, Some(td.path()))
            .unwrap()
            .expect("invocation-root manifest governs nested archive entry");
        assert_eq!(got, fs::canonicalize(td.path()).unwrap());
    }

    #[test]
    fn archive_invocation_root_does_not_adopt_entry_outside_boundary() {
        let td = TempDir::new().unwrap();
        manifest(td.path());
        let outside = TempDir::new().unwrap();
        let entry = outside.path().join("docs");
        fs::create_dir_all(&entry).unwrap();

        assert!(
            find_project_root_for_file_checked_at(&entry, Some(td.path()))
                .unwrap()
                .is_none(),
            "a manifest cannot govern an entry outside the invocation root"
        );
    }

    #[test]
    fn archive_invocation_does_not_adopt_unrelated_parent_manifest() {
        let td = TempDir::new().unwrap();
        manifest(td.path());
        let invocation = td.path().join("project");
        let entry = invocation.join("docs");
        fs::create_dir_all(&entry).unwrap();

        assert!(
            find_project_root_for_file_checked_at(&entry, Some(&invocation))
                .unwrap()
                .is_none(),
            "a parent manifest outside the invocation root must remain unrelated"
        );
    }

    #[cfg(unix)]
    #[test]
    fn archive_invocation_rejects_entry_symlink_escape() {
        use std::os::unix::fs::symlink;

        let td = TempDir::new().unwrap();
        manifest(td.path());
        let outside = TempDir::new().unwrap();
        fs::create_dir_all(outside.path().join("docs")).unwrap();
        symlink(outside.path(), td.path().join("docs")).unwrap();

        assert!(
            find_project_root_for_file_checked_at(&td.path().join("docs"), Some(td.path()))
                .unwrap()
                .is_none(),
            "a symlinked entry outside the invocation root must be refused"
        );
    }
}
