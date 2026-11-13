// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Project identity and temporary-manifest race controls.

use super::project_transaction::canonical_explicit_entry;
use super::{entry_relative_to_root, open_explicit, temporary_manifest};
use crate::project::build_lock::ProjectBuildLock;
use crate::project::{find_project_root_for_file, resolve_sources};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;

const ENTRY_SOURCE: &str = "fn main() -> i64 { 42 }\n";

/// A *bare* manifest: no `[targets.*].sources` and no temporary-manifest
/// marker, so it is a real one-file project. The package name is
/// distinctive so a test can tell a loaded manifest from a synthesised one,
/// which takes its name from the entry's file stem.
const BARE_MANIFEST: &str =
    "[package]\nname = \"preceding\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"alpha.mind\"\n";

/// One directory, two spellings: canonical `real/` and the symlink `link/`.
struct Aliased {
    _tmp: tempfile::TempDir,
    real: PathBuf,
    link: PathBuf,
}

fn aliased_dir() -> Aliased {
    let tmp = tempfile::tempdir().expect("temp dir");
    // Pin the canonical spelling of the temp root itself; on macOS it is
    // already reached through a symlink, and the only aliasing under test
    // must be the one this fixture creates.
    let base = tmp.path().canonicalize().expect("canonical temp root");
    let real = base.join("real");
    fs::create_dir(&real).expect("real dir");
    let link = base.join("link");
    symlink(&real, &link).expect("directory symlink");
    fs::write(real.join("alpha.mind"), ENTRY_SOURCE).expect("entry");
    Aliased {
        _tmp: tmp,
        real,
        link,
    }
}

fn cwd() -> PathBuf {
    std::env::current_dir().expect("cwd")
}

#[test]
fn entry_relative_path_is_stable() {
    assert_eq!(
        entry_relative_to_root(
            Path::new("/tmp/project/src/main.mind"),
            Path::new("/tmp/project")
        )
        .expect("entry is under root"),
        "src/main.mind"
    );
}

#[test]
fn entry_relative_path_refuses_escape_without_absolute_fallback() {
    let error = entry_relative_to_root(Path::new("/tmp/outside.mind"), Path::new("/tmp/project"))
        .expect_err("entry outside root must fail closed");
    assert_eq!(
        error.to_string(),
        "entry path /tmp/outside.mind is outside project root /tmp/project"
    );
}

/// Both spellings resolve to one directory identity, under which the entry
/// is project-root-relative.
///
/// The relative spelling is load-bearing: `run_build` derives the manifest
/// `entry` key by stripping the root prefix off the entry path. Canonicalising
/// the directory makes both spellings produce the same checked relative key.
#[test]
fn symlink_and_canonical_spellings_open_one_directory_identity() {
    let d = aliased_dir();
    fs::write(d.real.join("Mind.toml"), BARE_MANIFEST).expect("manifest");

    let (canonical_dir, canonical_entry) =
        canonical_explicit_entry(&d.real.join("alpha.mind"), &cwd()).expect("canonical entry");
    let (alias_dir, alias_entry) =
        canonical_explicit_entry(&d.link.join("alpha.mind"), &cwd()).expect("alias entry");

    assert_eq!(canonical_dir, alias_dir, "one directory, two spellings");
    assert_eq!(canonical_entry, alias_entry);
    assert_eq!(alias_dir, d.real, "the identity is the canonical spelling");
    assert_eq!(
        find_project_root_for_file(&alias_dir).as_deref(),
        Some(d.real.as_path()),
        "the root probe and the entry directory must agree by construction"
    );
    assert_eq!(
        alias_entry.strip_prefix(&alias_dir).ok(),
        Some(Path::new("alpha.mind")),
        "the manifest entry key must be project-root-relative"
    );
}

/// A co-located *bare* manifest is a project: through either spelling the
/// named entry builds that project, with the same whole-directory source set a
/// no-path build uses, so a sibling module the entry imports is compiled.
#[test]
fn adjacent_bare_manifest_builds_the_project_through_either_spelling() {
    let d = aliased_dir();
    fs::write(d.real.join("Mind.toml"), BARE_MANIFEST).expect("manifest");
    fs::write(d.real.join("sibling.mind"), "fn other() -> i64 { 7 }\n").expect("sibling");

    // The no-path build's source set: the reference both spellings must equal.
    let (no_path, _) = resolve_sources(&d.real, "alpha.mind", None, false).expect("walk");
    assert!(
        no_path.contains(&d.real.join("sibling.mind")),
        "positive control: the no-path walk does not see the sibling"
    );
    for spelling in [&d.real, &d.link] {
        let (entry_dir, entry_path) =
            canonical_explicit_entry(&spelling.join("alpha.mind"), &cwd())
                .expect("canonical entry");
        let opened = open_explicit(
            find_project_root_for_file(&entry_dir),
            &entry_dir,
            &entry_path,
        )
        .expect("transaction opens");
        assert!(
            !opened.single_file,
            "{} narrowed a project to its entry",
            spelling.display()
        );
        assert_eq!(opened.root, d.real);
        assert_eq!(opened.manifest.package.name, "preceding");
        let (sources, _) = resolve_sources(&opened.root, "alpha.mind", None, opened.single_file)
            .expect("source set");
        assert_eq!(
            sources,
            no_path,
            "{} built a different module graph than the no-path build",
            spelling.display()
        );
    }
}

/// The dc85648e hazard: a killed standalone build leaves its temporary
/// manifest behind, here planted with the exact text the build writes. It must
/// not turn the next named build into a project build, next to the entry or in
/// a parent directory, and the leftover is removed under the lock.
#[test]
fn leftover_temporary_manifest_is_never_a_project() {
    let d = aliased_dir();
    fs::write(d.real.join("sibling.mind"), "fn other() -> i64 { 7 }\n").expect("sibling");
    let leftover = d.real.join("Mind.toml");
    fs::write(&leftover, temporary_manifest("killed", "alpha.mind")).expect("leftover");
    let (entry_dir, entry_path) =
        canonical_explicit_entry(&d.link.join("alpha.mind"), &cwd()).expect("canonical entry");
    let probed = find_project_root_for_file(&entry_dir);
    assert_eq!(
        probed.as_deref(),
        Some(d.real.as_path()),
        "probe saw no leftover"
    );
    let opened = open_explicit(probed, &entry_dir, &entry_path).expect("transaction opens");
    assert!(
        opened.single_file,
        "a leftover temporary manifest made a project"
    );
    assert_eq!(opened.manifest.package.name, "alpha");
    assert!(!leftover.exists(), "the leftover was not removed");
    let (sources, _) =
        resolve_sources(&opened.root, "alpha.mind", None, opened.single_file).expect("source set");
    assert_eq!(sources, vec![d.real.join("alpha.mind")]);
    // Release this transaction's project lock: the next one locks the same
    // directory, and a shadowed binding would hold it until the test ends.
    drop(opened);

    // In a parent directory inside a repository: the build is standalone in
    // the entry's own directory, not a walk of the parent.
    fs::create_dir(d.real.join(".git")).expect("repo boundary");
    fs::create_dir(d.real.join("sub")).expect("sub");
    fs::write(d.real.join("sub/beta.mind"), ENTRY_SOURCE).expect("nested entry");
    fs::write(&leftover, temporary_manifest("killed", "alpha.mind")).expect("leftover");
    let (entry_dir, entry_path) =
        canonical_explicit_entry(&d.real.join("sub/beta.mind"), &cwd()).expect("nested entry");
    let probed = find_project_root_for_file(&entry_dir);
    assert_eq!(
        probed.as_deref(),
        Some(d.real.as_path()),
        "probe saw no parent leftover"
    );
    let opened = open_explicit(probed, &entry_dir, &entry_path).expect("transaction opens");
    assert!(opened.single_file);
    assert_eq!(opened.root, d.real.join("sub"));
    assert!(!leftover.exists(), "the parent leftover was not removed");
    drop(opened);

    // A real project above the leftover is still found: the search continues
    // upward past the removed file, both for an entry next to the leftover and
    // for one below it.
    fs::write(&leftover, BARE_MANIFEST).expect("real project manifest");
    fs::create_dir(d.real.join("sub/deep")).expect("deep");
    fs::write(d.real.join("sub/deep/gamma.mind"), ENTRY_SOURCE).expect("deep entry");
    let sub_leftover = d.real.join("sub/Mind.toml");
    for entry in ["sub/beta.mind", "sub/deep/gamma.mind"] {
        fs::write(&sub_leftover, temporary_manifest("killed", "beta.mind")).expect("leftover");
        let (entry_dir, entry_path) =
            canonical_explicit_entry(&d.real.join(entry), &cwd()).expect("entry below");
        let probed = find_project_root_for_file(&entry_dir);
        assert_eq!(
            probed.as_deref(),
            Some(d.real.join("sub").as_path()),
            "{entry}"
        );
        let opened = open_explicit(probed, &entry_dir, &entry_path).expect("transaction opens");
        assert!(
            !opened.single_file,
            "{entry}: the real project above was lost"
        );
        assert_eq!(opened.root, d.real, "{entry}");
        assert_eq!(opened.manifest.package.name, "preceding", "{entry}");
        assert!(
            !sub_leftover.exists(),
            "{entry}: the leftover was not removed"
        );
    }
}

/// With no governing manifest a named entry that imports a local module is
/// refused at the import, by name, with the fix; a `std` import is not.
#[test]
fn standalone_local_import_names_the_missing_project() {
    let d = aliased_dir();
    fs::write(
        d.real.join("alpha.mind"),
        "import std.io;\nimport ledger_lib;\nfn main() -> i64 { return 0; }\n",
    )
    .expect("entry");
    let (entry_dir, entry_path) =
        canonical_explicit_entry(&d.real.join("alpha.mind"), &cwd()).expect("canonical entry");
    let err = match open_explicit(None, &entry_dir, &entry_path) {
        Ok(_) => panic!("a standalone local import was accepted"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("ledger_lib"), "{err}");
    assert!(err.contains("add a Mind.toml"), "{err}");
    assert!(!err.contains("std.io"), "{err}");

    fs::write(
        d.real.join("alpha.mind"),
        "import std.io;\nfn main() -> i64 { return 0; }\n",
    )
    .expect("std-only entry");
    let opened = open_explicit(None, &entry_dir, &entry_path).expect("std-only standalone");
    assert!(opened.single_file);
}

/// A vanished temporary manifest, reproduced against the real lock with
/// explicit barriers.
///
/// A concurrent standalone build holds the project lock while its temporary
/// `Mind.toml` exists and removes that file before releasing — the ordering
/// `run_build` uses, where `ManifestEdit::restore` runs under the lock. The
/// build under test therefore probes a root that exists and, once it owns
/// the lock, finds no manifest there.
///
/// No sleep and no timing assumption: the entering transaction cannot take
/// the lock before the preceding one releases it, and the removal precedes
/// that release, so the manifest is gone at every observation made here.
#[test]
fn vanished_temporary_manifest_under_the_lock_is_not_a_manifest_error() {
    let d = aliased_dir();
    let manifest_path = d.real.join("Mind.toml");
    fs::write(&manifest_path, BARE_MANIFEST).expect("temporary manifest");
    let (entry_dir, entry_path) =
        canonical_explicit_entry(&d.link.join("alpha.mind"), &cwd()).expect("canonical entry");

    let gate = Arc::new(Barrier::new(2));
    let preceding = {
        let gate = Arc::clone(&gate);
        let root = d.real.clone();
        let manifest_path = manifest_path.clone();
        thread::spawn(move || {
            let lock = ProjectBuildLock::acquire(&root).expect("preceding transaction lock");
            gate.wait(); // 1: the manifest exists and the lock is held
            gate.wait(); // 2: the other build has probed; wind this one down
            fs::remove_file(&manifest_path).expect("remove the temporary manifest");
            drop(lock);
        })
    };

    gate.wait(); // 1
    let probed = find_project_root_for_file(&entry_dir);
    assert_eq!(
        probed.as_deref(),
        Some(d.real.as_path()),
        "the probe must take the Some(root) branch"
    );
    gate.wait(); // 2

    let opened = open_explicit(probed, &entry_dir, &entry_path)
        .expect("a vanished temporary manifest is a standalone build, not a manifest error");
    assert!(opened.single_file);
    assert_eq!(opened.root, d.real);
    assert_eq!(
        opened.manifest.package.name, "alpha",
        "the manifest must be synthesised from the entry, not the vanished file"
    );
    drop(opened);

    preceding.join().expect("preceding transaction");
    assert!(
        !manifest_path.exists(),
        "no temporary manifest is left behind"
    );
}

/// The fallback above is bounded to the entry's own directory: a genuine
/// project whose manifest disappears is still a hard, loud error.
#[test]
fn a_vanished_ancestor_project_manifest_is_still_an_error() {
    let d = aliased_dir();
    fs::create_dir(d.real.join("src")).expect("src");
    fs::write(d.real.join("src/main.mind"), ENTRY_SOURCE).expect("entry");
    // A repo boundary, so the bounded root scan may ascend out of `src/`.
    fs::write(d.real.join(".git"), "gitdir: elsewhere\n").expect("repo boundary");
    fs::write(d.real.join("Mind.toml"), BARE_MANIFEST).expect("manifest");

    let (entry_dir, entry_path) =
        canonical_explicit_entry(&d.link.join("src/main.mind"), &cwd()).expect("canonical entry");
    let probed = find_project_root_for_file(&entry_dir);
    assert_eq!(probed.as_deref(), Some(d.real.as_path()));
    assert_ne!(probed.as_deref(), Some(entry_dir.as_path()));

    fs::remove_file(d.real.join("Mind.toml")).expect("manifest disappears");
    // Matched rather than `expect_err`: the Ok payload owns a held build
    // lock, and rendering it for a panic message is not worth a `Debug`
    // impl on a transaction that must never be printed.
    match open_explicit(probed, &entry_dir, &entry_path) {
        Ok(_) => panic!("a real project losing its manifest was silently synthesised"),
        Err(err) => assert!(
            err.to_string().contains("manifest error"),
            "unexpected diagnostic: {err}"
        ),
    }
}

/// A real multi-file project keeps its whole-directory walk through the
/// symlink spelling: canonicalising the boundary must not narrow a genuine
/// project to its entry.
#[test]
fn a_real_project_keeps_its_whole_directory_scope_through_the_symlink() {
    let d = aliased_dir();
    fs::create_dir(d.real.join("src")).expect("src");
    fs::write(d.real.join("src/main.mind"), ENTRY_SOURCE).expect("entry");
    fs::write(d.real.join("src/helper.mind"), "fn helper() -> i64 { 1 }\n").expect("sibling");
    fs::write(d.real.join(".git"), "gitdir: elsewhere\n").expect("repo boundary");
    fs::write(
        d.real.join("Mind.toml"),
        "[package]\nname = \"real_project\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"src/main.mind\"\n",
    )
    .expect("manifest");

    let (entry_dir, entry_path) =
        canonical_explicit_entry(&d.link.join("src/main.mind"), &cwd()).expect("canonical entry");
    let opened = open_explicit(
        find_project_root_for_file(&entry_dir),
        &entry_dir,
        &entry_path,
    )
    .expect("transaction opens");
    assert!(
        !opened.single_file,
        "a real project was narrowed to its entry"
    );
    assert_eq!(opened.root, d.real);
    assert_eq!(opened.manifest.package.name, "real_project");

    let entry_rel = entry_path
        .strip_prefix(&opened.root)
        .expect("entry is project-root-relative")
        .to_string_lossy()
        .replace('\\', "/");
    assert_eq!(entry_rel, "src/main.mind");
    let (sources, _) =
        resolve_sources(&opened.root, &entry_rel, None, opened.single_file).expect("walk");
    assert!(sources.contains(&d.real.join("src/main.mind")));
    assert!(sources.contains(&d.real.join("src/helper.mind")));
}
