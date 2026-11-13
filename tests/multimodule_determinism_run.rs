// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Multi-module build determinism + incremental-cache completeness.
//!
//! Two properties, both load-bearing for a project that does NOT declare an
//! explicit `[targets.*].sources` list (the default):
//!
//! 1. **The source ORDER must not come from the filesystem.** The entry-parent
//!    walk iterated `fs::read_dir` verbatim, so the compile order — and with it
//!    the object order handed to the linker and the fold order of the
//!    whole-project enum/const registries — was whatever order the filesystem
//!    happened to store directory entries in (creation order on tmpfs, name-hash
//!    order on ext4/xfs). Two machines holding byte-identical sources could
//!    therefore emit different artifacts, which is precisely the claim the
//!    cross-substrate byte-identity wedge makes impossible.
//!
//! 2. **The incremental cache key must cover every participating source.** The
//!    key fingerprinted the ENTRY source alone, so editing a SIBLING module left
//!    it unchanged: the probe HIT and `mindc build` copied the previous artifact
//!    out — exit 0, no diagnostic, a binary built from the old sibling. That is
//!    a silent wrong-program failure, indistinguishable from a correct build.
//!
//! Gate: `cargo test --no-default-features
//!        --features "mlir-build std-surface cross-module-imports"
//!        --test multimodule_determinism_run`

#![cfg(unix)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

/// Root for the walk-order fixtures. `CARGO_TARGET_TMPDIR` (not the system temp
/// dir) so the fixtures live on the SAME filesystem as the build tree: directory
/// ordering is a property of the filesystem, and the system temp dir may be a
/// tmpfs whose ordering differs from the one real projects are built on.
fn walk_tmp_root(tag: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{tag}_{}", std::process::id()))
}

/// Module file names used by the walk-order tests, deliberately spread across a
/// subdirectory so the assertion covers the RECURSIVE walk, not just one level.
const WALK_MODULES: [&str; 8] = [
    "alpha.mind",
    "bravo.mind",
    "charlie.mind",
    "delta.mind",
    "nested/echo.mind",
    "nested/foxtrot.mind",
    "nested/golf.mind",
    "main.mind",
];

/// Raw `fs::read_dir` order of the `.mind` files directly in `dir`, unsorted —
/// the order the walk used to inherit.
fn raw_readdir_mind(dir: &Path) -> Vec<String> {
    fs::read_dir(dir)
        .expect("read_dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "mind").unwrap_or(false))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect()
}

/// Create a project whose `.mind` files are written in `order` (an index
/// permutation of [`WALK_MODULES`]). On a filesystem that stores directory
/// entries in creation order (tmpfs) this is what makes two otherwise identical
/// projects list their sources differently.
fn make_walk_project(root: &Path, order: &[usize]) {
    let _ = fs::remove_dir_all(root);
    fs::create_dir_all(root.join("src").join("nested")).expect("mkdir");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"walkorder\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"src/main.mind\"\noutput = \"walkorder\"\n",
    )
    .expect("write Mind.toml");
    for &i in order {
        let rel = WALK_MODULES[i];
        let body = if rel == "main.mind" {
            "fn main() -> i64 {\n    return 3\n}\n".to_string()
        } else {
            let stem = Path::new(rel).file_stem().unwrap().to_string_lossy();
            format!("pub fn {stem}_value() -> i64 {{\n    return {i}\n}}\n")
        };
        fs::write(root.join("src").join(rel), body).expect("write module");
    }
}

// ---------------------------------------------------------------------------
// 1. The walked source set is sorted on path bytes, not left in readdir order.
// ---------------------------------------------------------------------------

#[test]
fn walked_source_order_is_sorted_not_readdir_order() {
    let root = walk_tmp_root("mind_walkorder_unit");
    // Written in REVERSE name order: on a creation-order filesystem this
    // guarantees the raw listing is not already sorted, so the assertion below
    // cannot pass vacuously.
    let reverse: Vec<usize> = (0..WALK_MODULES.len()).rev().collect();
    make_walk_project(&root, &reverse);

    let src_dir = root.join("src");
    let raw = raw_readdir_mind(&src_dir);
    let mut raw_sorted = raw.clone();
    raw_sorted.sort();
    if raw == raw_sorted {
        // Strength disclosure, not a skip: the assertion below still runs, but on
        // a filesystem that returns directory entries in name order it cannot
        // DISTINGUISH the sorted walk from the raw one. Say so rather than let a
        // weakened run read as a full-strength pass.
        eprintln!(
            "NOTE: {} lists directory entries in name order, so this run cannot \
             distinguish the sorted walk from the raw readdir order; the sort \
             invariant is still asserted.",
            src_dir.display()
        );
    }

    let sources = libmind::project::collect_sources(&root, "src/main.mind")
        .expect("collect_sources on the walk project");
    assert_eq!(
        sources.len(),
        WALK_MODULES.len(),
        "every module must be collected; got {sources:?}"
    );

    let mut expected = sources.clone();
    expected.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    assert_eq!(
        sources, expected,
        "collect_sources must return the walked set in FULL-PATH-BYTE order, not \
         in filesystem order — source order fixes compile order, link object \
         order and the whole-project registry fold order, so a readdir-ordered \
         set makes the emitted artifact a function of the disk.\n\
         raw readdir order was: {raw:?}"
    );

    let _ = fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------
// 2. Two identical projects whose directories list in DIFFERENT orders build
//    byte-identical artifacts.
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#[test]
fn readdir_order_does_not_change_artifact_bytes() {
    use std::process::Command;

    let mindc = common::mindc_bin();
    if !mindc.exists() {
        common::gate::skipped("multimodule_determinism_run", "mindc binary not built");
        return;
    }

    let base = walk_tmp_root("mind_walkorder_bytes");
    let a = base.join("a");
    let b = base.join("b");
    let forward: Vec<usize> = (0..WALK_MODULES.len()).collect();
    let reverse: Vec<usize> = (0..WALK_MODULES.len()).rev().collect();
    make_walk_project(&a, &forward);
    make_walk_project(&b, &reverse);

    let art_a = base.join("walkorder_a");
    let art_b = base.join("walkorder_b");
    for (proj, art) in [(&a, &art_a), (&b, &art_b)] {
        let out = Command::new(&mindc)
            .current_dir(proj)
            .args([
                "build",
                "--backend=mlir",
                "--release",
                "--emit=binary",
                "--no-cache",
                &format!("--out={}", art.display()),
            ])
            .output()
            .expect("run mindc build");
        // One fail-closed decision, keyed on the refusal's stable CAUSE CODE.
        // Reading the prose here ("needs mlir-build") would be a second,
        // hand-written classifier: it grades a real build regression as a
        // capability skip the moment the wording changes, and it cannot be
        // made to hard-fail under `MIND_BENCH_REQUIRE=1`.
        if !common::gate::compiled("multimodule_determinism_run", &out) {
            return;
        }
    }

    let listing_a = raw_readdir_mind(&a.join("src"));
    let listing_b = raw_readdir_mind(&b.join("src"));
    if listing_a == listing_b {
        // This filesystem derives directory order from the NAMES alone (ext4/xfs
        // hash order), so creation order cannot make the two projects diverge.
        // The byte-identity assertion below still runs; assert the underlying
        // invariant directly as well so this test never degrades into a pair of
        // builds that could not have differed.
        let sources = libmind::project::collect_sources(&a, "src/main.mind").expect("collect");
        let mut sorted = sources.clone();
        sorted.sort_by(|x, y| x.as_os_str().cmp(y.as_os_str()));
        assert_eq!(
            sources, sorted,
            "walked source order must be sorted on path bytes"
        );
        eprintln!(
            "multimodule-determinism: filesystem orders {listing_a:?} by name, not \
             by creation — byte-identity asserted plus the sort invariant directly"
        );
    }

    let bytes_a = fs::read(&art_a).expect("read artifact a");
    let bytes_b = fs::read(&art_b).expect("read artifact b");
    // Compared with `assert!`, not `assert_eq!`: a failing `assert_eq!` on two
    // multi-kilobyte artifacts dumps both byte vectors into the log and buries
    // the diagnosis under 30 KB of decimal numbers.
    assert!(
        bytes_a == bytes_b,
        "BYTE-IDENTITY VIOLATION: the same {} sources built into {} bytes vs {} \
         bytes because the two project directories list their entries in \
         different orders ({listing_a:?} vs {listing_b:?}). Source order must \
         come from the sorted walk, never from the filesystem.",
        WALK_MODULES.len(),
        bytes_a.len(),
        bytes_b.len(),
    );

    let _ = fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// 3. Editing a SIBLING module invalidates the incremental cache.
// ---------------------------------------------------------------------------

// The fixtures below feed the build-executing test only, so they carry its
// feature gate: without a native backend they would be dead code, and a
// warning in one feature set is a warning the zero-warning gate must see.

#[cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
const SIBLING_MAIN: &str = "import helper\n\nfn main() -> i64 {\n    return helper_value()\n}\n";

#[cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
fn helper_src(value: i64) -> String {
    format!("pub fn helper_value() -> i64 {{\n    return {value}\n}}\n")
}

#[cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
const SIBLING_MANIFEST: &str = r#"[package]
name = "siblingcache"
version = "0.1.0"

[build]
entry = "src/main.mind"
output = "siblingcache"

[targets.cpu]
backend = "cpu"
"#;

#[cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#[test]
fn editing_a_sibling_module_invalidates_the_cache() {
    use std::process::Command;

    let mindc = common::mindc_bin();
    if !mindc.exists() {
        common::gate::skipped("multimodule_determinism_run", "mindc binary not built");
        return;
    }

    let proj = std::env::temp_dir().join(format!("mind_sibling_cache_{}", std::process::id()));
    let src = proj.join("src");
    let _ = fs::remove_dir_all(&proj);
    fs::create_dir_all(&src).expect("mkdir src");
    fs::write(proj.join("Mind.toml"), SIBLING_MANIFEST).expect("write Mind.toml");
    fs::write(src.join("main.mind"), SIBLING_MAIN).expect("write main.mind");
    fs::write(src.join("helper.mind"), helper_src(7)).expect("write helper.mind");

    let artifact = proj.join("out_bin");

    // The cache is deliberately ENABLED (no `--no-cache`): a cache that cannot
    // see a sibling edit is exactly what this test exists to catch.
    let build = |label: &str| -> Vec<u8> {
        let out = Command::new(&mindc)
            .current_dir(&proj)
            .args([
                "build",
                "--backend=mlir",
                "--release",
                "--emit=binary",
                &format!("--out={}", artifact.display()),
            ])
            .output()
            .expect("run mindc build");
        assert!(
            out.status.success(),
            "sibling-cache: {label} build failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        fs::read(&artifact).expect("read artifact")
    };

    let run_exit = || -> i32 {
        Command::new(&artifact)
            .status()
            .expect("run built binary")
            .code()
            .expect("exit code")
    };

    let bytes_before = build("first");
    assert_eq!(
        run_exit(),
        7,
        "the first build must observe helper_value() == 7"
    );

    // Edit the SIBLING module only; the entry source is untouched.
    fs::write(src.join("helper.mind"), helper_src(9)).expect("rewrite helper.mind");

    let bytes_after = build("rebuild after sibling edit");
    // `assert!` rather than `assert_ne!` — see the byte-identity test above.
    assert!(
        bytes_before != bytes_after,
        "STALE ARTIFACT: rebuilding after a sibling module changed its return \
         value produced a BYTE-IDENTICAL binary ({} bytes), so the incremental \
         cache key does not cover sibling sources and `mindc build` handed back \
         a binary compiled from the OLD helper — exit 0, no diagnostic.",
        bytes_before.len()
    );
    assert_eq!(
        run_exit(),
        9,
        "the rebuilt binary must observe the EDITED helper_value() == 9, not the \
         cached 7"
    );

    let _ = fs::remove_dir_all(&proj);
}

/// Guard the fail-closed half of the fingerprint: a participating source that
/// cannot be read must yield NO key at all, so the cache is neither read nor
/// written. A key that silently skipped the unreadable file would claim a
/// reproducibility it cannot have.
#[test]
fn unreadable_source_yields_no_cache_key() {
    let root = std::env::temp_dir().join(format!("mind_srckey_failclosed_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("mkdir");
    let present = root.join("present.mind");
    fs::write(&present, "fn main() -> i64 { return 0 }\n").expect("write");
    let missing: PathBuf = root.join("absent.mind");

    assert!(
        libmind::build::source_set_dep_entries(&root, std::slice::from_ref(&present)).is_some(),
        "a readable source set must fingerprint"
    );
    assert!(
        libmind::build::source_set_dep_entries(&root, &[present, missing]).is_none(),
        "an unreadable participating source must fail closed (no key), never be \
         skipped"
    );

    let _ = fs::remove_dir_all(&root);
}

/// Source ORDER is artifact-visible, so a reordered source list must produce a
/// different fingerprint even though `module_cache_key` sorts dep entries.
#[test]
fn source_order_changes_the_fingerprint() {
    let root = std::env::temp_dir().join(format!("mind_srckey_order_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("mkdir");
    let a = root.join("a.mind");
    let b = root.join("b.mind");
    fs::write(&a, "pub fn a_value() -> i64 { return 1 }\n").expect("write a");
    fs::write(&b, "pub fn b_value() -> i64 { return 2 }\n").expect("write b");

    let forward = libmind::build::source_set_dep_entries(&root, &[a.clone(), b.clone()])
        .expect("fingerprint forward");
    let reverse =
        libmind::build::source_set_dep_entries(&root, &[b, a]).expect("fingerprint reverse");

    // Compare the entries SORTED, because that is the only comparison that can
    // fail. `module_cache_key` sorts the dep set before it digests it, so two
    // entry lists that differ only by permutation collapse to the SAME key.
    // Asserting `forward != reverse` on the returned vectors therefore proves
    // nothing: they are permutations of each other by construction and differ
    // whether or not position is encoded — measured by deleting the `src{:04}=`
    // ordinal, which left that assertion green. Sorting first is what makes the
    // ordinal load-bearing here, exactly as the real key makes it load-bearing.
    let mut forward_sorted = forward.clone();
    forward_sorted.sort();
    let mut reverse_sorted = reverse.clone();
    reverse_sorted.sort();
    assert_ne!(
        forward_sorted, reverse_sorted,
        "reordering the source list must change the fingerprint — dep entries are \
         SORTED into the key, so position must be encoded in each entry or a \
         reordered source list hashes identically while emitting different bytes"
    );

    let _ = fs::remove_dir_all(&root);
}
