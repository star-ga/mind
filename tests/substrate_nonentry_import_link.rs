// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Substrate-object link when a NON-ENTRY module imports a substrate module.
//!
//! REGRESSION for the reported pure-MIND-from-source native-link failure where
//! `hash`/`sha256` were undefined at link time. `std.sha256` IS a substrate
//! module (its `.o` must be compiled + linked when imported), but the
//! substrate-object BFS was seeded from the ENTRY FILE ONLY. When the entry
//! does not `import std.sha256` and only a NON-entry sibling module does
//! (mirroring the reported `src/ir.mind` doing `import std.sha256;
//! sha256.hash(x)` while the entry imports nothing substrate), the parser
//! rewrites the module-qualified call to a bare `hash` symbol but
//! `__std_sha256.o` was never compiled — so `hash`/`sha256` stayed undefined at
//! native link and the executable failed to link.
//!
//! The fix widens the substrate-BFS SEED from the entry alone to the UNION of
//! every project source's substrate imports (`src/project/mod.rs`:
//! `compile_substrate_objects` on the executable path and the extra-objects scan
//! in `build_cdylib_from_entry`). The transitive substrate→substrate walk is
//! unchanged, and an empty seed set still leaves the link byte-identical to the
//! single-entry path (the keystone imports no substrate module in any source).
//!
//! This builds a two-source EXECUTABLE project where `src/hasher.mind` imports
//! `std.sha256` and the entry `src/main.mind` does NOT, links a real native
//! binary, runs it, and asserts the process exit code equals the first byte of
//! the known SHA-256("abc") digest (0xba = 186) — proving both that the
//! substrate object linked (pre-fix: undefined `hash` → link failure) and that
//! the pure-MIND SHA-256 path is genuinely exercised.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test substrate_nonentry_import_link`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// NON-entry module: the only source that imports the substrate module. Hashes
// b"abc" and returns byte `i` of the 32-byte digest. The parser rewrites
// `sha256.hash(b)` to a bare `hash` symbol whose `.o` must be linked.
const HASHER_MIND: &str = r#"import std.sha256

pub fn digest_byte(i: i64) -> i64 {
    let mut b: bytes = []
    b.push(97)
    b.push(98)
    b.push(99)
    let d = sha256.hash(b)
    return __mind_load_i8(d + i) & 255
}
"#;

// Entry module: imports the SIBLING user module (never `std.sha256`) and
// returns the first digest byte as the process exit code. SHA-256("abc")[0] =
// 0xba = 186.
const MAIN_MIND: &str = r#"import hasher

fn main() -> i64 {
    return digest_byte(0)
}
"#;

const MANIFEST: &str = r#"[package]
name = "xshalink"
version = "0.1.0"

[build]
entry = "src/main.mind"
output = "xshalink"

[targets.cpu]
backend = "cpu"
"#;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

#[test]
fn non_entry_substrate_import_links_and_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        println!("substrate-nonentry-import-link: mindc not found; skipping");
        return;
    }

    // Isolated project dir (unique per process to avoid cross-test races).
    let proj = std::env::temp_dir().join(format!("mind_substrate_nonentry_{}", std::process::id()));
    let src = proj.join("src");
    let _ = std::fs::remove_dir_all(&proj);
    std::fs::create_dir_all(&src).expect("mkdir src");
    std::fs::write(proj.join("Mind.toml"), MANIFEST).expect("write Mind.toml");
    std::fs::write(src.join("hasher.mind"), HASHER_MIND).expect("write hasher.mind");
    std::fs::write(src.join("main.mind"), MAIN_MIND).expect("write main.mind");

    let out = Command::new(&mindc)
        .arg("build")
        .current_dir(&proj)
        .output()
        .expect("run mindc build");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("mlir-build") && stderr.contains("requires") {
            println!("substrate-nonentry-import-link: needs mlir-build; skipping");
            return;
        }
        // The exact failure this test guards against: `hash`/`sha256` undefined
        // at native link because the non-entry `import std.sha256` never seeded
        // the substrate-object BFS.
        panic!(
            "substrate-nonentry-import-link: `mindc build` failed (this is the \
             undefined-`hash`/`sha256` native-link regression):\n{stderr}"
        );
    }

    // Locate the produced native executable under target/.
    let mut bin = None;
    for prof in ["debug", "release", "cpu"] {
        let cand = proj.join("target").join(prof).join("xshalink");
        if cand.exists() {
            bin = Some(cand);
            break;
        }
    }
    // Fall back to a recursive scan (profile dir naming can vary).
    if bin.is_none() {
        if let Ok(rd) = std::fs::read_dir(proj.join("target")) {
            for e in rd.flatten() {
                let cand = e.path().join("xshalink");
                if cand.exists() {
                    bin = Some(cand);
                    break;
                }
            }
        }
    }
    let bin = bin.expect("substrate-nonentry-import-link: no executable produced");

    let run = Command::new(&bin).output().expect("run linked executable");
    let code = run.status.code().expect("executable exited via signal");
    assert_eq!(
        code,
        186,
        "expected SHA-256(\"abc\")[0] = 0xba = 186 as exit code; got {code} \
         (stdout: {}, stderr: {})",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );

    let _ = std::fs::remove_dir_all(&proj);
}
