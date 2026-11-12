// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Directory identity at the `mindc build` transaction boundary.
//!
//! An explicit `mindc build <file>` relates three directory spellings: the
//! project root the bounded scan resolved, the directory the named entry sits
//! in, and the directory whose build lock is held. Naming the entry through a
//! path with a symlinked component makes those spellings diverge, and every
//! decision that compares them inverts.
//!
//! macOS exposes the same shape through its system directory symlinks: the temporary
//! directory is handed out as `/var/folders/...` and canonicalises to
//! `/private/var/folders/...` — so two concurrent standalone builds under
//! `tempfile::tempdir()` meet it every time. These tests write the same
//! mismatch down with an explicit directory symlink, which enforces the
//! invariant on every Unix runner.

#![cfg(unix)]

mod common;

use common::require_mindc;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;

const ALPHA: &str = "fn main() -> i64 { 42 }\n";
const BETA: &str = "fn main() -> i64 { 99 }\n";
/// A *bare* manifest — no `[targets.*].sources`. A named entry next to it
/// builds the project it declares, with the whole-directory module graph.
const BARE_MANIFEST: &str =
    "[package]\nname = \"adjacent\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"alpha.mind\"\n";

const HELPER: &str = "pub fn graph_anchor(x: i64) -> i64 {\n    return x + 1;\n}\n";
const IMPORTING_MAIN: &str = "import helper;\n\npub fn entry(x: i64) -> i64 {\n    \
                              return graph_anchor(x);\n}\n\nfn main() -> i32 {\n    \
                              return entry(1);\n}\n";

/// One directory reachable under two spellings: the canonical `real/` and the
/// directory symlink `link/`.
struct Aliased {
    _tmp: tempfile::TempDir,
    real: PathBuf,
    link: PathBuf,
}

fn aliased_dir() -> Aliased {
    let tmp = tempfile::tempdir().expect("temp dir");
    // Pin the canonical spelling of the temp root itself; on macOS it is
    // already reached through a symlink, and the only aliasing under test must
    // be the one this fixture creates.
    let base = tmp.path().canonicalize().expect("canonical temp root");
    let real = base.join("real");
    fs::create_dir(&real).expect("real dir");
    let link = base.join("link");
    symlink(&real, &link).expect("directory symlink");
    Aliased {
        _tmp: tmp,
        real,
        link,
    }
}

fn build(mindc: &Path, cwd: &Path, source: &Path, out: &Path) -> Output {
    Command::new(mindc)
        .args(["build", "--backend", "mlir", "--emit=cdylib", "--out"])
        .arg(out)
        .arg(source)
        .current_dir(cwd)
        .output()
        .expect("spawn standalone build")
}

/// Two concurrent standalone builds in one directory, named through the
/// symlink spelling.
///
/// The first build writes a temporary `Mind.toml` under the project lock and
/// removes it before releasing; the second observes that manifest, waits for
/// the lock, and must then recognise the vanished file as the first build's
/// temporary manifest rather than report
/// `manifest error: Failed to read .../Mind.toml`.
///
/// Whether the two processes interleave is up to the scheduler, so this is the
/// end-to-end control, not the deterministic one. The deterministic
/// reproduction — probe and lock coordinated by explicit barriers, no sleep —
/// is
/// `libmind::build::transaction_identity_tests::vanished_temporary_manifest_under_the_lock_is_not_a_manifest_error`.
#[test]
fn concurrent_standalone_builds_through_a_symlinked_directory_leave_no_temporary_manifest() {
    let d = aliased_dir();
    fs::write(d.real.join("alpha.mind"), ALPHA).expect("alpha");
    fs::write(d.real.join("beta.mind"), BETA).expect("beta");
    let mindc = require_mindc();

    let spawn = |source: PathBuf, out: PathBuf| {
        let binary = mindc.clone();
        let cwd = d.link.clone();
        thread::spawn(move || build(&binary, &cwd, &source, &out))
    };
    let first = spawn(d.link.join("alpha.mind"), d.link.join("alpha.so"));
    let second = spawn(d.link.join("beta.mind"), d.link.join("beta.so"));
    let first = first.join().expect("alpha build panicked");
    let second = second.join().expect("beta build panicked");

    let capable_first = common::gate::compiled("mindc_symlinked_project_identity", &first);
    let capable_second = common::gate::compiled("mindc_symlinked_project_identity", &second);
    assert_eq!(capable_first, capable_second);
    assert!(d.real.join("alpha.mind").is_file());
    assert!(d.real.join("beta.mind").is_file());
    assert!(
        !d.real.join("Mind.toml").exists(),
        "a temporary manifest was left behind"
    );
}

/// Both spellings select the same project scope next to a genuine adjacent
/// bare manifest, and that scope holds the sibling the entry imports.
///
/// The entry calls into the sibling, so a build that narrowed the project to
/// its entry fails loudly here instead of quietly producing different bytes.
#[test]
fn symlink_and_canonical_spellings_build_the_same_project_artifact() {
    let d = aliased_dir();
    fs::write(d.real.join("alpha.mind"), IMPORTING_MAIN).expect("alpha");
    fs::write(d.real.join("helper.mind"), HELPER).expect("imported sibling");
    fs::write(d.real.join("Mind.toml"), BARE_MANIFEST).expect("adjacent bare manifest");
    let manifest_before = fs::read(d.real.join("Mind.toml")).expect("manifest bytes");
    let mindc = require_mindc();

    let canonical_out = d.real.join("canonical.so");
    let canonical = build(&mindc, &d.real, &d.real.join("alpha.mind"), &canonical_out);
    let alias_out = d.real.join("alias.so");
    let alias = build(&mindc, &d.link, &d.link.join("alpha.mind"), &alias_out);

    let capable_canonical = common::gate::compiled("mindc_symlinked_project_identity", &canonical);
    let capable_alias = common::gate::compiled("mindc_symlinked_project_identity", &alias);
    assert_eq!(
        capable_canonical, capable_alias,
        "one spelling reached the native backend and the other did not"
    );

    // The adjacent manifest is a real project file: the build borrows it and
    // must hand back exactly the bytes it found.
    assert_eq!(
        fs::read(d.real.join("Mind.toml")).expect("manifest bytes"),
        manifest_before,
        "the adjacent manifest was not restored byte-for-byte"
    );

    if !capable_canonical {
        return;
    }
    // Native evidence, not a check-only pass: both artifacts exist and are the
    // same program.
    let canonical_bytes = fs::read(&canonical_out).expect("canonical artifact");
    let alias_bytes = fs::read(&alias_out).expect("alias artifact");
    assert!(!canonical_bytes.is_empty(), "empty canonical artifact");
    assert_eq!(
        canonical_bytes, alias_bytes,
        "the symlink spelling selected a different source scope than the canonical one"
    );
    let library =
        unsafe { libloading::Library::new(&canonical_out) }.expect("load emitted shared library");
    let entry: libloading::Symbol<'_, unsafe extern "C" fn(i64) -> i64> =
        unsafe { library.get(b"entry") }.expect("resolve entry");
    assert_eq!(
        unsafe { entry(1) },
        2,
        "the imported sibling body did not execute"
    );
}

/// A real multi-file project still resolves its imports and executes when its
/// entry is named through a symlinked directory.
///
/// Canonicalising the transaction boundary must not narrow a genuine project to
/// its entry: `graph_anchor` lives in a sibling module, so an over-narrowed
/// build cannot link and an unresolved import cannot return 2.
#[test]
fn a_real_project_resolves_imports_through_a_symlinked_directory() {
    let d = aliased_dir();
    fs::write(d.real.join("main.mind"), IMPORTING_MAIN).expect("entry");
    fs::write(d.real.join("helper.mind"), HELPER).expect("helper");
    fs::write(
        d.real.join("Mind.toml"),
        "[package]\nname = \"aliased_project\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"main.mind\"\nemit = \"cdylib\"\n\n\
         [targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"helper.mind\"]\n",
    )
    .expect("manifest");
    let mindc = require_mindc();

    let out = d.real.join("aliased.so");
    let output = build(&mindc, &d.link, &d.link.join("main.mind"), &out);
    if !common::gate::compiled("mindc_symlinked_project_identity", &output) {
        return;
    }

    let library = unsafe { libloading::Library::new(&out) }.expect("load emitted shared library");
    let entry: libloading::Symbol<'_, unsafe extern "C" fn(i64) -> i64> =
        unsafe { library.get(b"entry") }.expect("resolve entry");
    assert_eq!(
        unsafe { entry(1) },
        2,
        "the imported sibling body did not execute"
    );
}
