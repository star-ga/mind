// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase F integration tests — incremental compilation cache.
//!
//! Gate:
//! ```
//! cargo test --release --features "mlir-build std-surface cross-module-imports" mindc_cache_phase_f
//! ```
//!
//! Tests:
//!  1. Cold build (no cache) — all misses, objects written.
//!  2. Re-build unchanged — all hits (hit count == module count).
//!  3. Touch one module — one miss, rest hit, binary byte-identical to cold build.
//!  4. `--no-cache` — all misses even if cache exists; entries still written.
//!  5. `--target=cerebras` — all misses (cross-target isolation).
//!  6. `--optimize=release` — all misses (release vs debug separate).
//!  7. Compiler version bump (mock) — all misses.
//!  8. `mindc clean --cache` — removes `.cache/` but leaves binary intact.
//!  9. Determinism — two cold builds on the same source produce byte-identical artifacts.
//!
//! Concurrent project transactions have focused coverage in `mindc_project_lock.rs`.

mod common;
use common::{mindc_bin, reported_artifact, require_mindc, run_build_captured};

use std::fs;
use std::path::Path;
use std::process::Command;

use libmind::build::cache::{
    BuildManifest, CacheProbe, cache_root, module_cache_key, module_cache_key_material,
    object_path, probe,
};
use libmind::project::{BuildTarget, EmitKind, OptimizeLevel};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// Create a minimal single-source MIND project in `dir`.
fn make_project(dir: &Path, name: &str, source: &str) {
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n"
        ),
    )
    .unwrap();
    fs::write(dir.join("src/main.mind"), source).unwrap();
}

/// Run `mindc build` in `dir`. Returns exit status.
fn run_build(mindc: &Path, dir: &Path, extra_args: &[&str]) -> std::process::ExitStatus {
    Command::new(mindc)
        .args(["build", "--backend", "mlir"])
        .args(extra_args)
        .current_dir(dir)
        .status()
        .expect("failed to spawn mindc")
}

/// Probe the cache for the entry source.
fn probe_for_source(
    project_root: &Path,
    source: &[u8],
    target: BuildTarget,
    optimize: OptimizeLevel,
) -> CacheProbe {
    // `run_build` invokes `mindc build` with no `--emit`, whose default emit is
    // `binary`. Recompute the key with the SAME helper the build path uses — it
    // folds in the `emit=binary` discriminator AND the compiler/toolchain binary
    // identity (issue #96), fingerprinting the mindc SUBPROCESS binary
    // (`mindc_bin()` == CARGO_BIN_EXE_mindc), NOT this test executable. A bare
    // `module_cache_key` with `CARGO_PKG_VERSION` would never match the
    // subprocess-populated cache once the fingerprint is in the key.
    // The key also fingerprints every source the build compiles, so the probe
    // must hand over the SAME set `run_build` resolved — here the single
    // `src/main.mind` these fixtures contain.
    let material = libmind::build::compile_cache_material(
        source,
        libmind::build::CacheKeyFlags {
            target,
            optimize,
            emit: EmitKind::Binary,
            edition: 2024,
        },
        &mindc_bin(),
        project_root,
        &[project_root.join("src").join("main.mind")],
    )
    .expect("compiler identity for CARGO_BIN_EXE_mindc");
    let c_root = cache_root(project_root, target, optimize);
    probe(&c_root, &material)
}

const SIMPLE_MIND: &str = "fn main() -> i64 { 42 }\n";
const MODIFIED_MIND: &str = "fn main() -> i64 { 99 }\n";

// ---------------------------------------------------------------------------
// Test 1: cold build — no cache exists → all misses, objects written
// ---------------------------------------------------------------------------

#[test]
fn phase_f_01_cold_build_all_miss() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "cold_project", SIMPLE_MIND);

    // Before build: cache miss.
    let pre = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(pre, CacheProbe::Miss { .. }),
        "expected miss before build"
    );

    // After build: cache entry must exist (regardless of whether mlir-build ran).
    let mindc = require_mindc();
    let out = run_build_captured(&mindc, dir, &[]);

    // A build that FAILS is classified, not assumed environmental. The bare
    // `if status.success() { ..asserts.. }` this replaced ran no assertion at
    // all on failure and still graded `ok`, so a real build regression was
    // indistinguishable from a host without the backend. gate::compiled either
    // panics (regression, or a toolchain gap under MIND_BENCH_REQUIRE=1) or
    // emits the countable `ran=0` marker.
    if !crate::common::gate::compiled("mindc_cache_phase_f", &out) {
        return;
    }
    let post = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(post, CacheProbe::Hit { .. }),
        "expected cache hit after successful build"
    );
}

// ---------------------------------------------------------------------------
// Test 2: re-build unchanged — all hits
// ---------------------------------------------------------------------------

#[test]
fn phase_f_02_rebuild_unchanged_all_hit() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "rebuild_project", SIMPLE_MIND);

    let mindc = require_mindc();

    // First build.
    // A failure is classified, never silently returned: `run_build` inherits
    // stdio, so the bare `if !s1.success() { return; }` this replaced could not
    // even see the diagnostic it was grading as a capability gap.
    let s1 = run_build_captured(&mindc, dir, &[]);
    if !crate::common::gate::compiled("mindc_cache_phase_f", &s1) {
        return;
    }

    // Verify cache is populated after first build.
    let after_first = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(after_first, CacheProbe::Hit { .. }),
        "cache must be populated after first build"
    );

    // Second build with no source change must hit.
    let s2 = run_build(&mindc, dir, &["--verbose"]);
    assert!(s2.success(), "second build should succeed");

    // Cache is still populated and key unchanged.
    let after_second = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(after_second, CacheProbe::Hit { .. }),
        "cache must remain hit after no-op rebuild"
    );
}

// ---------------------------------------------------------------------------
// Test 3: touch one module — one miss, binary still produced
// ---------------------------------------------------------------------------

#[test]
fn phase_f_03_touch_source_causes_miss() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "touch_project", SIMPLE_MIND);

    let mindc = require_mindc();

    // First build.
    // A failure is classified, never silently returned: `run_build` inherits
    // stdio, so the bare `if !s1.success() { return; }` this replaced could not
    // even see the diagnostic it was grading as a capability gap.
    let s1 = run_build_captured(&mindc, dir, &[]);
    if !crate::common::gate::compiled("mindc_cache_phase_f", &s1) {
        return;
    }

    // Verify original source is cached.
    let original_hit = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(matches!(original_hit, CacheProbe::Hit { .. }));

    // Modify source.
    fs::write(dir.join("src/main.mind"), MODIFIED_MIND).unwrap();

    // New source key should be a miss before the rebuild.
    let modified_miss = probe_for_source(
        dir,
        MODIFIED_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(modified_miss, CacheProbe::Miss { .. }),
        "modified source must be a cache miss before rebuild"
    );

    // Rebuild.
    let s2 = run_build(&mindc, dir, &[]);
    assert!(s2.success(), "rebuild after source change should succeed");

    // After rebuild, modified source is cached.
    let modified_hit = probe_for_source(
        dir,
        MODIFIED_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(modified_hit, CacheProbe::Hit { .. }),
        "modified source must be cached after rebuild"
    );
}

// ---------------------------------------------------------------------------
// Test 4: --no-cache bypasses hit check; entries still written
// ---------------------------------------------------------------------------

#[test]
fn phase_f_04_no_cache_bypasses_hit_but_still_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "nocache_project", SIMPLE_MIND);

    let mindc = require_mindc();

    // First build (warm-up).
    // A failure is classified, never silently returned: `run_build` inherits
    // stdio, so the bare `if !s1.success() { return; }` this replaced could not
    // even see the diagnostic it was grading as a capability gap.
    let s1 = run_build_captured(&mindc, dir, &[]);
    if !crate::common::gate::compiled("mindc_cache_phase_f", &s1) {
        return;
    }

    // Confirm hit exists.
    let hit = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(matches!(hit, CacheProbe::Hit { .. }));

    // Second build with --no-cache: must NOT short-circuit (will run full compile)
    // but must still write to cache afterwards.
    let s2 = run_build(&mindc, dir, &["--no-cache"]);
    assert!(s2.success(), "--no-cache build must succeed");

    // The cache entry should still be present (written by the full compile pass).
    let post = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(post, CacheProbe::Hit { .. }),
        "--no-cache build must still write entry to cache"
    );
}

// ---------------------------------------------------------------------------
// Test 5: cross-target isolation — changing --target produces all misses
// ---------------------------------------------------------------------------

#[test]
fn phase_f_05_cross_target_isolation() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "target_project", SIMPLE_MIND);

    // Cache key for cpu target.
    let key_cpu = module_cache_key(
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        env!("CARGO_PKG_VERSION"),
        2024,
    );

    // Cache key for cerebras target.
    let key_cerebras = module_cache_key(
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cerebras,
        OptimizeLevel::Debug,
        &[],
        env!("CARGO_PKG_VERSION"),
        2024,
    );

    // Keys must differ across targets.
    assert_ne!(
        key_cpu, key_cerebras,
        "cpu and cerebras keys must not collide"
    );

    // Cache root paths must also differ.
    let root_cpu = cache_root(dir, BuildTarget::Cpu, OptimizeLevel::Debug);
    let root_cerebras = cache_root(dir, BuildTarget::Cerebras, OptimizeLevel::Debug);
    assert_ne!(
        root_cpu, root_cerebras,
        "cache roots must be target-segregated"
    );

    // cpu probe must not return a hit in the cerebras cache root and vice versa.
    let obj_cpu = object_path(&root_cpu, &key_cpu);
    let obj_cerebras_wrong = object_path(&root_cerebras, &key_cpu);
    assert_ne!(obj_cpu, obj_cerebras_wrong);
}

// ---------------------------------------------------------------------------
// Test 6: optimize level isolation — release vs debug separate cache dirs
// ---------------------------------------------------------------------------

#[test]
fn phase_f_06_optimize_level_isolation() {
    let key_debug = module_cache_key(
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        env!("CARGO_PKG_VERSION"),
        2024,
    );
    let key_release = module_cache_key(
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Release,
        &[],
        env!("CARGO_PKG_VERSION"),
        2024,
    );
    assert_ne!(key_debug, key_release, "debug and release keys must differ");

    let tmp = tempfile::tempdir().unwrap();
    let root_debug = cache_root(tmp.path(), BuildTarget::Cpu, OptimizeLevel::Debug);
    let root_release = cache_root(tmp.path(), BuildTarget::Cpu, OptimizeLevel::Release);
    assert_ne!(
        root_debug, root_release,
        "debug and release cache roots must differ"
    );
}

// ---------------------------------------------------------------------------
// Test 7: compiler version change → all misses
// ---------------------------------------------------------------------------

#[test]
fn phase_f_07_compiler_version_bump_invalidates() {
    let key_v1 = module_cache_key(
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.8",
        2024,
    );
    let key_v2 = module_cache_key(
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.9",
        2024,
    );
    assert_ne!(
        key_v1, key_v2,
        "different compiler versions must produce different cache keys"
    );
}

// ---------------------------------------------------------------------------
// Test 8: mindc clean --cache removes .cache/ but leaves binary intact
// ---------------------------------------------------------------------------

#[test]
fn phase_f_08_clean_cache_preserves_binary() {
    use libmind::build::cache::{ObjectMeta, clean_all_caches, write_object};
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let dir = tmp.path();

    // Simulate a populated cache.
    let c_root = cache_root(dir, BuildTarget::Cpu, OptimizeLevel::Debug);
    let material = module_cache_key_material(
        b"fixture source",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.8+/x|1|2|0.6.8",
        2024,
    );
    let key = material.key();
    let meta = ObjectMeta {
        source_path: "src/main.mind".to_string(),
        cache_key: key.to_string(),
        target: "Cpu".to_string(),
        optimize: "Debug".to_string(),
        compiler_version: "0.6.8+/x|1|2|0.6.8".to_string(),
        compiler_fingerprint: "/x|1|2|0.6.8".to_string(),
        dep_hashes: vec![],
        input_inventory: material.input_inventory().clone(),
    };
    write_object(&c_root, &material, b"\x7fELF stub", &meta).unwrap();
    assert!(c_root.exists(), ".cache/ must exist after write");

    // Simulate a linked binary sitting next to .cache/.
    let binary_dir = dir.join("target").join("cpu").join("debug");
    fs::create_dir_all(&binary_dir).unwrap();
    let binary = binary_dir.join("myapp");
    fs::write(&binary, b"ELF binary").unwrap();

    // Run clean_all_caches.
    clean_all_caches(dir).unwrap();

    // .cache/ removed.
    assert!(
        !c_root.exists(),
        ".cache/ must be removed after clean --cache"
    );

    // Binary still intact.
    assert!(binary.exists(), "linked binary must survive clean --cache");
    assert_eq!(fs::read(&binary).unwrap(), b"ELF binary");
    assert!(
        dir.join(".mind-build.lock").is_file(),
        "the persistent project lock must survive target cleanup"
    );
    assert!(
        !dir.join("target/.mind-build.lock").exists(),
        "cleanup must not create a replaceable lock inside target"
    );
}

// ---------------------------------------------------------------------------
// Test 9: determinism — same source + same flags → byte-identical artifact
// ---------------------------------------------------------------------------

#[test]
fn phase_f_09_deterministic_cache_key() {
    // The cache key itself must be deterministic (same inputs → same SHA-256).
    let k1 = module_cache_key(
        b"fn main() -> i64 { 0 }",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &["dep_a_hash".to_string(), "dep_b_hash".to_string()],
        "0.6.8",
        2024,
    );
    let k2 = module_cache_key(
        b"fn main() -> i64 { 0 }",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &["dep_b_hash".to_string(), "dep_a_hash".to_string()], // reversed order
        "0.6.8",
        2024,
    );
    // Dep order must not affect the key.
    assert_eq!(k1, k2, "cache key must be independent of dep hash ordering");

    // Verify that a build on unchanged source produces the same artifact.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "determinism_project", SIMPLE_MIND);

    let mindc = require_mindc();

    // Build 1. A failure is classified, never silently returned: the bare
    // `if !s1.success() { return; }` this replaced graded a real build
    // regression as a pass.
    let s1 = run_build_captured(&mindc, dir, &[]);
    if !crate::common::gate::compiled("mindc_cache_phase_f", &s1) {
        return;
    }

    // The artifact is named by the builder, so the builder is asked. A `binary`
    // with no declared `[build] output` is `<root>/target/<profile>/<package.name>`
    // — here `determinism_project` — on the native path and on the
    // launcher-refusal path alike (project::artifact_stem owns that name). This
    // line once hand-typed "app", the spelling only the backend-less fallback
    // then wrote, so on every host that HAS the native backend the assertion
    // below panicked instead of comparing bytes; before that it hand-typed
    // `target/cpu/debug/determinism_project`, a path nothing writes, which made
    // the check vacuous while it reported a pass. Two hand-typed spellings, two
    // failure modes, one cause: reading the reported path removes the spelling.
    let artifact = reported_artifact(&s1);
    assert!(
        artifact.exists(),
        "the build succeeded but wrote no artifact at {}; the cache \
         byte-identity check cannot run",
        artifact.display()
    );
    let bytes1 = fs::read(&artifact).unwrap();

    // Clean the artifact but keep the cache.
    let _ = fs::remove_file(&artifact);

    // Build 2 — cache hit → artifact restored from cache.
    let s2 = run_build(&mindc, dir, &[]);
    assert!(s2.success(), "second build should succeed");

    // Unconditional. `if artifact.exists() { .. }` guarded the ONE comparison
    // this test exists to make, so a rebuild that restored nothing graded as a
    // pass having asserted nothing — the same vacuity the hand-typed path
    // above produced, one line further down. Build 2 exited 0, so the artifact
    // it was asked to produce has to be there.
    assert!(
        artifact.exists(),
        "the cache-hit rebuild exited 0 but restored no artifact at {}; the \
         byte-identity comparison cannot run",
        artifact.display()
    );
    let bytes2 = fs::read(&artifact).unwrap();
    assert_eq!(
        bytes1, bytes2,
        "artifact must be byte-identical across cache-hit rebuild"
    );
}

// ---------------------------------------------------------------------------
// Unit tests — cache key properties (pure, no I/O)
// ---------------------------------------------------------------------------

#[test]
fn cache_key_empty_source_is_stable() {
    let k1 = module_cache_key(
        b"",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.8",
        2024,
    );
    let k2 = module_cache_key(
        b"",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.8",
        2024,
    );
    assert_eq!(k1, k2);
    assert_eq!(k1.len(), 64);
}

#[test]
fn cache_key_edition_change_invalidates() {
    let k2024 = module_cache_key(
        b"fn x(){}",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.8",
        2024,
    );
    let k2025 = module_cache_key(
        b"fn x(){}",
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
        &[],
        "0.6.8",
        2025,
    );
    assert_ne!(k2024, k2025, "edition change must produce different key");
}

#[test]
fn manifest_entries_are_sorted() {
    let tmp = tempfile::tempdir().unwrap();
    let mut m = BuildManifest::default();
    m.entries.insert("src/z.mind".to_string(), "zz".to_string());
    m.entries.insert("src/a.mind".to_string(), "aa".to_string());
    let path = tmp.path().join("manifest.json");
    m.save(&path).unwrap();
    let loaded = BuildManifest::load(&path).unwrap();
    // BTreeMap iterates in sorted order.
    let keys: Vec<&str> = loaded.entries.keys().map(|s| s.as_str()).collect();
    assert_eq!(keys, vec!["src/a.mind", "src/z.mind"]);
}

// ---------------------------------------------------------------------------
// Test 11 (issue #96): a rebuilt mindc binary invalidates the cache.
//
// The correctness bug: the cache key folded only `CARGO_PKG_VERSION`, which
// does not change between dev rebuilds of `mindc` at the same version — so a
// stale `.so` was served after a compiler change. The fix folds the compiler
// BINARY IDENTITY (path|size|mtime) into the key. This test proves a
// different mindc binary (fresh path + bumped mtime, identical version) keys
// differently and MISSES a cache the real binary populated.
// ---------------------------------------------------------------------------

#[test]
fn phase_f_11_rebuilt_compiler_binary_invalidates_cache() {
    let bin = require_mindc();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    make_project(dir, "fp_project", SIMPLE_MIND);

    // Cold build populates the cache under the real mindc's fingerprint. A
    // missing backend may skip; a build regression must NOT grade as a pass.
    let cold = run_build_captured(&bin, dir, &[]);
    if !crate::common::gate::compiled("mindc_cache_phase_f", &cold) {
        return;
    }
    let hit = probe_for_source(
        dir,
        SIMPLE_MIND.as_bytes(),
        BuildTarget::Cpu,
        OptimizeLevel::Debug,
    );
    assert!(
        matches!(hit, CacheProbe::Hit { .. }),
        "cold build must populate the cache under the real binary's key"
    );

    // Simulate a compiler rebuild: same version, different binary identity.
    // Copy the mindc binary to a fresh path and stamp a later mtime.
    let copied = dir.join("mindc-rebuilt");
    std::fs::copy(&bin, &copied).unwrap();
    let f = std::fs::File::options().write(true).open(&copied).unwrap();
    f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(300))
        .unwrap();
    drop(f);

    let material_real = libmind::build::compile_cache_material(
        SIMPLE_MIND.as_bytes(),
        libmind::build::CacheKeyFlags {
            target: BuildTarget::Cpu,
            optimize: OptimizeLevel::Debug,
            emit: EmitKind::Binary,
            edition: 2024,
        },
        &bin,
        dir,
        &[dir.join("src").join("main.mind")],
    )
    .unwrap();
    let material_rebuilt = libmind::build::compile_cache_material(
        SIMPLE_MIND.as_bytes(),
        libmind::build::CacheKeyFlags {
            target: BuildTarget::Cpu,
            optimize: OptimizeLevel::Debug,
            emit: EmitKind::Binary,
            edition: 2024,
        },
        &copied,
        dir,
        &[dir.join("src").join("main.mind")],
    )
    .unwrap();
    assert_ne!(
        material_real.key(),
        material_rebuilt.key(),
        "a rebuilt mindc binary (new path/mtime, same version) must key differently"
    );

    let c_root = cache_root(dir, BuildTarget::Cpu, OptimizeLevel::Debug);
    assert!(
        matches!(probe(&c_root, &material_rebuilt), CacheProbe::Miss { .. }),
        "cache populated by the real binary must MISS for the rebuilt binary's key"
    );
    assert!(
        matches!(probe(&c_root, &material_real), CacheProbe::Hit { .. }),
        "the real binary's key must still hit its own cache entry"
    );
}
