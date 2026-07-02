#![cfg(feature = "pkg")]

// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Package-install path-traversal security regression.
//!
//! `install_package` only rejected `..` (`Component::ParentDir`) entries, so a
//! crafted `.mindpkg` could still escape the install dir via (a) a symlink
//! entry pointing outside `target_root` followed by a write through it, or
//! (b) an absolute-path member. The fix routes every non-manifest entry through
//! `tar::Entry::unpack_in`, which refuses to write outside the destination, and
//! rejects absolute / root-prefixed members up front. This test builds a
//! malicious archive and asserts nothing lands outside the target dir.

use std::fs;
use std::io::Cursor;

use flate2::write::GzEncoder;
use flate2::Compression;
use libmind::package::install_package;
use tar::{Builder, EntryType, Header};
use tempfile::tempdir;

/// Build a `.mindpkg` (gzip tar) containing a valid `package.toml` plus a
/// symlink entry that escapes to `escape_target`, then a file written *through*
/// that symlink. Returns the package path.
fn build_malicious_pkg(dir: &std::path::Path, escape_target: &std::path::Path) -> std::path::PathBuf {
    let manifest = b"name = \"evil\"\nversion = \"0.1.0\"\nauthors = [\"x\"]\nfiles = []\n";

    let buf = Vec::new();
    let enc = GzEncoder::new(buf, Compression::default());
    let mut builder = Builder::new(enc);

    // 1) manifest
    let mut h = Header::new_gnu();
    h.set_size(manifest.len() as u64);
    h.set_mode(0o644);
    builder
        .append_data(&mut h, "package.toml", Cursor::new(&manifest[..]))
        .expect("append manifest");

    // 2) a symlink "link" -> escape_target's PARENT directory
    let link_dir = escape_target.parent().expect("escape parent");
    let mut lh = Header::new_gnu();
    lh.set_entry_type(EntryType::Symlink);
    lh.set_size(0);
    lh.set_mode(0o777);
    builder
        .append_link(&mut lh, "link", link_dir)
        .expect("append symlink");

    // 3) a file written through the symlink: "link/<escape filename>"
    let escape_name = escape_target.file_name().expect("escape name");
    let through = std::path::Path::new("link").join(escape_name);
    let payload = b"PWNED";
    let mut fh = Header::new_gnu();
    fh.set_size(payload.len() as u64);
    fh.set_mode(0o644);
    builder
        .append_data(&mut fh, &through, Cursor::new(&payload[..]))
        .expect("append through-link file");

    let enc = builder.into_inner().expect("finish tar");
    let gz = enc.finish().expect("finish gz");
    let pkg = dir.join("evil.mindpkg");
    fs::write(&pkg, gz).expect("write pkg");
    pkg
}

#[test]
fn install_rejects_symlink_escape() {
    let work = tempdir().expect("workdir");
    let target = work.path().join("install_root");
    fs::create_dir_all(&target).expect("mk target");

    // The file the malicious package tries to plant OUTSIDE target_root.
    let escape_dir = work.path().join("escaped");
    fs::create_dir_all(&escape_dir).expect("mk escape dir");
    let escape_file = escape_dir.join("pwned");
    assert!(!escape_file.exists());

    let pkg = build_malicious_pkg(work.path(), &escape_file);

    // Install may error (preferred) or skip the entry, but it MUST NOT write
    // the payload outside target_root.
    let _ = install_package(
        pkg.to_str().unwrap(),
        target.to_str().unwrap(),
    );

    assert!(
        !escape_file.exists(),
        "SECURITY: package escaped install dir — wrote {}",
        escape_file.display()
    );
}
