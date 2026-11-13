// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Serializes a project's shared objects, manifest edits, cleanup, and cache
//! publication. Other projects remain independent. The persistent lock file
//! lives outside `target/`: cleanup removes that whole directory, and unlinking
//! a held lock lets a later process lock a different inode.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

pub(crate) struct ProjectBuildLock {
    _file: File,
    root: PathBuf,
}

impl ProjectBuildLock {
    pub(crate) fn acquire(root: &Path) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(".mind-build.lock"))?;
        // Qualified trait call preserves the declared Rust 1.85 MSRV.
        fs2::FileExt::lock_exclusive(&file)?;
        Ok(Self {
            _file: file,
            root: root.to_path_buf(),
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }
}

/// First line of every temporary manifest, so a leftover of a killed build is
/// told apart from a real project manifest (which never carries it).
const TEMPORARY_MANIFEST_MARKER: &str =
    "# mindc: temporary manifest of a standalone build; removed when that build returns\n";

/// The temporary `Mind.toml` a standalone build writes for its duration.
pub fn temporary_manifest(pkg_name: &str, entry: &str) -> String {
    format!(
        "{TEMPORARY_MANIFEST_MARKER}[package]\nname = \"{pkg_name}\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"{entry}\"\n"
    )
}

fn is_temporary_manifest(path: &Path) -> bool {
    fs::read_to_string(path).is_ok_and(|text| text.starts_with(TEMPORARY_MANIFEST_MARKER))
}

/// Remove `path` when it is a leftover temporary manifest; true when it was.
///
/// Call only under the project lock of its directory: a temporary manifest's
/// writer removes it before releasing that lock, so one still present under
/// the lock belongs to a killed build.
pub(crate) fn remove_leftover(path: &Path) -> io::Result<bool> {
    if !is_temporary_manifest(path) {
        return Ok(false);
    }
    fs::remove_file(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "cannot remove leftover temporary manifest {}: {error}",
                path.display()
            ),
        )
    })?;
    Ok(true)
}

/// True when `root` holds no project manifest, so a caller searches again: a
/// temporary one, which is then gone (a leftover is removed under the lock, a
/// live build's is removed by its writer before the lock is granted), or none
/// at all (a live build removed it after the search saw it).
///
/// The lock is taken only for a manifest in doubt, so a real project pays one
/// read and no lock here. An empty file is in doubt: a live build's temporary
/// manifest is empty between its creation and its write.
pub(crate) fn clear_temporary_manifest(root: &Path) -> io::Result<bool> {
    let path = root.join("Mind.toml");
    match fs::read_to_string(&path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(true),
        Ok(text) if text.is_empty() || text.starts_with(TEMPORARY_MANIFEST_MARKER) => {
            let _lock = ProjectBuildLock::acquire(root)?;
            Ok(remove_leftover(&path)? || !path.exists())
        }
        // A real manifest, or one whose read error `load_manifest` reports.
        _ => Ok(false),
    }
}

/// The root `probe` finds by the rule of `lock_cwd_project`, for a search made
/// before the build holds its lock: a root `clear_temporary_manifest` settles
/// as no project is searched past. Every pass removes a file or finds one
/// absent, so the search ends.
pub(crate) fn governing_root(
    mut probe: impl FnMut() -> anyhow::Result<Option<PathBuf>>,
) -> anyhow::Result<Option<PathBuf>> {
    loop {
        let Some(root) = probe()? else {
            return Ok(None);
        };
        if !clear_temporary_manifest(&root)? {
            return Ok(Some(root));
        }
    }
}

/// Lock the project the current directory is in, by the one rule every build
/// form follows: a temporary manifest is never a project. A leftover is removed
/// under the lock, and a manifest that vanished while we waited (a live
/// standalone build's) is absent too; either way the search runs again, so the
/// result is what a directory without that file gives. Every pass removes a
/// file or finds one absent, so the search ends.
pub(crate) fn lock_cwd_project() -> anyhow::Result<ProjectBuildLock> {
    lock_searched_project(super::find_project_root)
}

/// `lock_cwd_project` with its search given, so a test can play that search.
fn lock_searched_project(
    mut search: impl FnMut() -> anyhow::Result<PathBuf>,
) -> anyhow::Result<ProjectBuildLock> {
    loop {
        let root = search()?;
        let lock = ProjectBuildLock::acquire(&root)?;
        let manifest = root.join("Mind.toml");
        if remove_leftover(&manifest)? || !manifest.exists() {
            continue;
        }
        return Ok(lock);
    }
}

/// Restores an exact manifest snapshot on every normal or unwinding exit.
///
/// Callers still invoke [`restore`](Self::restore) explicitly so an I/O failure
/// is surfaced. `Drop` is the panic/error backstop; it deliberately retries a
/// failed explicit restore instead of leaving a temporary entry behind without
/// another attempt.
pub(crate) struct ManifestEdit {
    path: PathBuf,
    original: Option<Vec<u8>>,
    restore_on_drop: bool,
}

impl ManifestEdit {
    pub(crate) fn replace(
        path: &Path,
        original: Option<Vec<u8>>,
        replacement: &[u8],
    ) -> io::Result<Self> {
        let mut edit = Self {
            path: path.to_path_buf(),
            original,
            restore_on_drop: true,
        };
        if let Err(write_error) = fs::write(path, replacement) {
            let restore_error = edit.restore_inner().err();
            if restore_error.is_none() {
                edit.restore_on_drop = false;
            }
            let detail = match restore_error {
                Some(error) => format!("; restoring the manifest also failed: {error}"),
                None => "; the original manifest was restored".to_string(),
            };
            return Err(io::Error::new(
                write_error.kind(),
                format!("cannot write temporary manifest: {write_error}{detail}"),
            ));
        }
        Ok(edit)
    }

    pub(crate) fn restore(mut self) -> io::Result<()> {
        let result = self.restore_inner();
        if result.is_ok() {
            self.restore_on_drop = false;
        }
        result
    }

    fn restore_inner(&self) -> io::Result<()> {
        match &self.original {
            Some(bytes) => fs::write(&self.path, bytes),
            None => match fs::remove_file(&self.path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error),
            },
        }
    }
}

impl Drop for ManifestEdit {
    fn drop(&mut self) {
        if self.restore_on_drop {
            let _ = self.restore_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ManifestEdit, ProjectBuildLock};
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn same_project_waits_for_the_existing_transaction() {
        let root = tempfile::tempdir().unwrap();
        let first = ProjectBuildLock::acquire(root.path()).unwrap();
        let path = root.path().to_path_buf();
        let (tx, rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let lock = ProjectBuildLock::acquire(&path).unwrap();
            tx.send(()).unwrap();
            drop(lock);
        });

        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        drop(first);
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn different_projects_do_not_share_a_lock() {
        let first_root = tempfile::tempdir().unwrap();
        let second_root = tempfile::tempdir().unwrap();
        let first = ProjectBuildLock::acquire(first_root.path()).unwrap();

        let second = ProjectBuildLock::acquire(second_root.path()).unwrap();

        drop(second);
        drop(first);
    }

    #[test]
    fn an_error_releases_the_transaction() {
        fn fail_while_locked(root: &Path) -> io::Result<()> {
            let _lock = ProjectBuildLock::acquire(root)?;
            Err(io::Error::other("injected build failure"))
        }

        let root = tempfile::tempdir().unwrap();
        assert!(fail_while_locked(root.path()).is_err());
        let replacement = ProjectBuildLock::acquire(root.path()).unwrap();
        drop(replacement);
    }

    #[test]
    fn target_cleanup_cannot_replace_the_held_lock_inode() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("target")).unwrap();
        let first = ProjectBuildLock::acquire(root.path()).unwrap();
        fs::remove_dir_all(root.path().join("target")).unwrap();
        let path = root.path().to_path_buf();
        let (tx, rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let lock = ProjectBuildLock::acquire(&path).unwrap();
            tx.send(()).unwrap();
            drop(lock);
        });

        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        drop(first);
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn manifest_edit_restores_exact_bytes_during_unwind() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("Mind.toml");
        let original = b"[build]\r\nentry='src/main.mind'\r\n";
        fs::write(&path, original).unwrap();
        let _lock = ProjectBuildLock::acquire(root.path()).unwrap();

        let result = std::panic::catch_unwind(|| {
            let _edit = ManifestEdit::replace(
                &path,
                Some(original.to_vec()),
                b"[build]\nentry = \"tests/probe.mind\"\n",
            )
            .unwrap();
            panic!("injected compiler panic");
        });

        assert!(result.is_err());
        assert_eq!(fs::read(path).unwrap(), original);
    }

    /// The search of `lock_cwd_project` sees a temporary manifest below a real
    /// project: a live standalone build's (its lock held, then the file
    /// removed and the lock released) or a killed build's leftover. Either way
    /// the root it locked is no project, so the search runs again and locks
    /// the real project. Read as a project, the live build's root has no
    /// manifest any more (`manifest error: Failed to read <dir>/Mind.toml`).
    #[test]
    fn a_temporary_manifest_seen_by_the_search_is_no_project() {
        for live in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let above = dir.path().to_path_buf();
            fs::write(
                above.join("Mind.toml"),
                "[package]\nname = \"real\"\nversion = \"0.1.0\"\n",
            )
            .unwrap();
            let sub = above.join("sub");
            fs::create_dir(&sub).unwrap();
            let held = live.then(|| ProjectBuildLock::acquire(&sub).unwrap());
            let manifest = sub.join("Mind.toml");
            fs::write(&manifest, super::temporary_manifest("alpha", "alpha.mind")).unwrap();
            let (seen_tx, seen_rx) = mpsc::channel();
            let (tx, rx) = mpsc::channel();
            let (probe, below, real) = (manifest.clone(), sub.clone(), above.clone());
            let search = thread::spawn(move || {
                let found = super::lock_searched_project(|| {
                    let root = if probe.exists() { &below } else { &real };
                    let _ = seen_tx.send(root.clone());
                    Ok(root.clone())
                });
                let found = found.map(|lock| lock.root().to_path_buf());
                tx.send(found.map_err(|e| e.to_string())).unwrap();
            });
            let seen = seen_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            assert_eq!(
                seen, sub,
                "live {live}: the search did not see the manifest"
            );
            if let Some(held) = held {
                let early = rx.recv_timeout(Duration::from_millis(300)).ok();
                assert_eq!(early, None, "the search did not wait on the live build");
                fs::remove_file(&manifest).unwrap();
                drop(held);
            }
            let found = rx.recv_timeout(Duration::from_secs(10)).unwrap();
            search.join().unwrap();
            assert_eq!(found, Ok(above), "live {live}");
            assert!(
                !manifest.exists(),
                "live {live}: the manifest is still there"
            );
        }
    }
}

/// The route of a build is selected before that build takes its lock, so the
/// search for the governing manifest follows the rule on its own.
#[cfg(test)]
mod governing_tests {
    use super::{ProjectBuildLock, temporary_manifest};
    use crate::project::toolchain_pin::{Backend, governing_backend, governing_manifest};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    const REAL: &str = "[package]\nname = \"real\"\nversion = \"0.1.0\"\n\n\
                        [build]\nentry = \"sub/beta.mind\"\nbackend = \"mlir\"\n";

    fn entry_in(dir: &Path) -> (PathBuf, String) {
        let root = dir.canonicalize().unwrap();
        fs::write(root.join("beta.mind"), "fn main() -> i64 { 0 }\n").unwrap();
        let entry = root.join("beta.mind").display().to_string();
        (root, entry)
    }

    /// A live standalone build's manifest, written or still empty (between its
    /// creation and its write), is no project. Its writer holds the lock and
    /// removes it before release, so the search waits and finds none. Only a
    /// search that skips the rule answers while the lock is held.
    #[test]
    fn route_selection_waits_out_a_live_temporary_manifest() {
        for text in [temporary_manifest("alpha", "alpha.mind"), String::new()] {
            let dir = tempfile::tempdir().unwrap();
            let (root, entry) = entry_in(dir.path());
            let manifest = root.join("Mind.toml");
            let lock = ProjectBuildLock::acquire(&root).unwrap();
            fs::write(&manifest, &text).unwrap();
            let (tx, rx) = mpsc::channel();
            let search = thread::spawn(move || {
                let found = governing_manifest(&[entry]).map(|f| f.map(|(_, m)| m.package.name));
                tx.send(found.map_err(|e| format!("{e:#}"))).unwrap();
            });
            let early = rx.recv_timeout(Duration::from_millis(300)).ok();
            fs::remove_file(&manifest).unwrap();
            drop(lock);
            let found = early.unwrap_or_else(|| rx.recv_timeout(Duration::from_secs(10)).unwrap());
            search.join().unwrap();
            assert_eq!(found, Ok(None), "manifest text {text:?}");
        }
    }

    /// A killed build's leftover in a sub-directory does not hide the real
    /// project above it: the route is that project's `[build] backend`.
    #[test]
    fn a_leftover_temporary_manifest_never_selects_the_route() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("Mind.toml"), REAL).unwrap();
        let (sub, entry) = entry_in(&root.join("sub"));
        let leftover = sub.join("Mind.toml");
        fs::write(&leftover, temporary_manifest("killed", "beta.mind")).unwrap();
        assert_eq!(governing_backend(&[entry]).unwrap(), Some(Backend::Mlir));
        assert!(!leftover.exists(), "the leftover was not removed");
    }

    /// A real project pays one read and no lock: the search does not wait for
    /// a build that holds that project's lock.
    #[test]
    fn a_real_project_is_found_without_its_lock() {
        let dir = tempfile::tempdir().unwrap();
        let (root, entry) = entry_in(dir.path());
        fs::write(root.join("Mind.toml"), REAL).unwrap();
        let _held = ProjectBuildLock::acquire(&root).unwrap();
        let (tx, rx) = mpsc::channel();
        let _search =
            thread::spawn(move || tx.send(governing_backend(&[entry]).map_err(|e| e.to_string())));
        let found = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the search waited");
        assert_eq!(found, Ok(Some(Backend::Mlir)));
    }

    /// A manifest the search saw and that is gone at the check (a live
    /// standalone build removed it in between) is no project: the search runs
    /// again. Nothing is in doubt, so no lock is taken.
    #[test]
    fn a_manifest_gone_since_the_search_is_no_project() {
        let dir = tempfile::tempdir().unwrap();
        assert!(super::clear_temporary_manifest(dir.path()).unwrap());
        assert!(!dir.path().join(".mind-build.lock").exists());
        let mut probes = vec![None, Some(dir.path().to_path_buf())];
        assert_eq!(
            super::governing_root(|| Ok(probes.pop().flatten())).unwrap(),
            None
        );
        assert!(probes.is_empty(), "the search did not run again");
    }
}
