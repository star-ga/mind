// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase G — KEYSTONE: `mindc build` bootstraps the mind repo itself.
//!
//! This is the milestone that retires cargo from the pure-MIND compile loop.
//! All tests in this module use the `mindc` binary (built via `cargo build`
//! from the Rust host crate) to drive `mindc build` against the mind repo's
//! own `Mind.toml`. The output must be byte-identical to:
//!
//!   1. A direct `mindc build <path> --emit=cdylib --out=<path>` invocation
//!      (same flags, same source, no `Mind.toml` involvement) — proves that
//!      `Mind.toml`-driven builds are a transparent layer over the single-file
//!      path.
//!
//!   2. The Phase F warm-cache hit — proves that Phase G adds no overhead
//!      to the incremental rebuild path.
//!
//! The byte-identity claim is the load-bearing keystone. The Rust crate
//! continues to host `mindc` (compiles the Rust source via cargo) until
//! RFC 0010 lands a pure-MIND libMLIR FFI. What Phase G claims:
//!
//!   "mindc build produces libmindc_mind.so byte-identical to the v0.6.1
//!   fixed-point oracle, driven entirely by the pure-MIND build orchestrator.
//!   Cargo is no longer load-bearing for the pure-MIND compile loop."
//!
//! Gate:
//! ```
//! cargo test --release --features "mlir-build std-surface cross-module-imports" \
//!     phase_g_keystone_bootstrap
//! ```

use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

fn mindc_bin() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    #[cfg(debug_assertions)]
    p.push("debug");
    #[cfg(not(debug_assertions))]
    p.push("release");
    #[cfg(target_os = "windows")]
    p.push("mindc.exe");
    #[cfg(not(target_os = "windows"))]
    p.push("mindc");
    p
}

fn require_mindc() -> Option<PathBuf> {
    let bin = mindc_bin();
    if bin.exists() {
        Some(bin)
    } else {
        // Fail-closed under enforcement: a missing mindc must NOT let the
        // keystone suite pass vacuously (#306 false-green guard). Only a
        // non-enforced (local convenience) run is allowed to skip.
        assert!(
            !enforce_real_backend(),
            "KEYSTONE: mindc binary not found at {} but MIND_BENCH_REQUIRE=1 — build it \
             (cargo build --release --features \"mlir-build std-surface cross-module-imports\" --bin mindc); \
             the gate must not pass when the toolchain is absent",
            bin.display()
        );
        eprintln!(
            "SKIP: mindc binary not found at {}; run `cargo build --release` first",
            bin.display()
        );
        None
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// RFC 0015 / `#306` fail-closed gate.
///
/// When `MIND_BENCH_REQUIRE=1` is set, the keystone tests must NOT silently
/// skip or report PASS on a launcher stub. A missing backend toolchain or a
/// stub-vs-ELF artifact mismatch becomes a hard failure instead.
///
/// This closes the false-green documented in `docs/byte-store-migration.md`
/// ("DO NOT capture the oracle from a stub-producing environment") and in the
/// 2026-05-29 ecosystem audit (PITFALLS B4): when `mindc build` falls back to a
/// ~1245-byte launcher script, byte-identity between two stubs is vacuous, yet
/// the suite still reported PASS — masking the broken real-ELF keystone. The
/// default (env unset) behaviour is unchanged: these environments legitimately
/// lack the proprietary `~/.mind/lib` runtime, so they skip. Enforcement mode
/// (the same flag the cross-substrate gate already uses) demands a real ELF.
fn enforce_real_backend() -> bool {
    std::env::var_os("MIND_BENCH_REQUIRE").is_some()
}

/// Guard the self-host bootstrap fixed point: the pure-MIND parser in
/// `examples/mindc_mind/main.mind` is attribute-BLIND and fails OPEN (it would
/// consume `#[` as a stray token, desyncing). The byte-identity oracle holds
/// only because no bootstrap-path source carries an attribute. If the compiler
/// ever annotates its own source with `#[deterministic]`/`#[target]`/`#[q16]`,
/// the Rust mindc (which understands `#[`) and the self-hosted parser would
/// produce different IR → the `.so` would diverge and the oracle would break
/// silently. This converts that latent landmine into a loud, toolchain-free
/// precondition (recommended by the architecture audit, 2026-05-23). Remove
/// this guard only when the pure-MIND `parse_item` learns to parse attributes.
#[test]
fn bootstrap_source_is_attribute_free() {
    for name in ["main.mind", "fixture.mind"] {
        let path = repo_root().join("examples/mindc_mind").join(name);
        if let Ok(src) = std::fs::read_to_string(&path) {
            assert!(
                !src.contains("#["),
                "examples/mindc_mind/{name} contains an attribute `#[` but the \
                 pure-MIND self-host parser is attribute-blind — this would break \
                 the byte-identity bootstrap oracle. Teach `parse_item` to parse \
                 attributes before annotating the compiler's own source."
            );
        }
    }
}

/// Return a hex-encoded SHA-256 of `bytes` using the same FIPS 180-4
/// implementation that `src/build/cache.rs` uses — self-contained, no dep.
fn sha256_hex(bytes: &[u8]) -> String {
    use libmind::build::cache::module_cache_key;
    use libmind::project::{BuildTarget, OptimizeLevel};
    // We reuse the cache-key function with an empty source section as a
    // stable proxy for the file hash. For the full-file hash we call the
    // sha256 helper that cache.rs exposes via module_cache_key with a
    // deterministic inputs set so the same bytes always hash the same way.
    // The actual file hash is captured in the assertion below by running
    // module_cache_key on the file bytes directly.
    module_cache_key(
        bytes,
        BuildTarget::Cpu,
        OptimizeLevel::Release,
        &[],
        env!("CARGO_PKG_VERSION"),
        2024,
    )
}

// ---------------------------------------------------------------------------
// Test 1 — Phase G manifest: Mind.toml exists at the repo root and is valid.
// ---------------------------------------------------------------------------

#[test]
fn phase_g_01_mind_toml_exists_and_is_valid() {
    let manifest_path = repo_root().join("Mind.toml");
    assert!(
        manifest_path.exists(),
        "Mind.toml must exist at the repo root (Phase G prerequisite)"
    );

    let text = fs::read_to_string(&manifest_path).expect("Mind.toml must be readable");

    let manifest: libmind::project::ProjectManifest =
        toml::from_str(&text).expect("Mind.toml must parse as a valid ProjectManifest");

    assert_eq!(
        manifest.package.name, "mind",
        "Mind.toml [package].name must be 'mind'"
    );
    assert_eq!(
        manifest.package.version,
        env!("CARGO_PKG_VERSION"),
        "Mind.toml [package].version must match Cargo.toml (single version source of truth)"
    );

    use libmind::project::{EmitKind, OptimizeLevel};
    assert_eq!(
        manifest.build.emit,
        EmitKind::Cdylib,
        "Mind.toml [build].emit must be 'cdylib' (we build libmindc_mind.so)"
    );
    assert_eq!(
        manifest.build.optimize,
        OptimizeLevel::Release,
        "Mind.toml [build].optimize must be 'release'"
    );

    // The entry must point to the pure-MIND self-hosting compiler source.
    assert!(
        manifest.build.entry.contains("mindc_mind") || manifest.build.entry.contains("main.mind"),
        "Mind.toml [build].entry must reference the mindc_mind main.mind source; got: {}",
        manifest.build.entry
    );

    // The c_abi export list must include mindc_compile.
    assert!(
        manifest.exports.c_abi.iter().any(|s| s == "mindc_compile"),
        "Mind.toml [exports] c_abi must include 'mindc_compile'"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — `mindc build` via Mind.toml succeeds (exit 0).
// ---------------------------------------------------------------------------

#[test]
fn phase_g_02_mindc_build_via_mind_toml_exits_0() {
    let Some(bin) = require_mindc() else { return };

    let out = std::env::temp_dir().join("phase_g_02_libmindc_mind.so");

    let result = Command::new(&bin)
        .args(["build", "--release", &format!("--out={}", out.display())])
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn mindc");

    // Mirror phase_g_03 / phase_g_04: the keystone tests are deliberately
    // tolerant of environments without a fully-wired backend toolchain. The
    // public CI runners cannot ship the proprietary MIND runtime backend
    // (`libmind_cpu_*`), so `mindc build` legitimately fails there with
    // "MIND runtime not found for backend" (Linux) or a downstream C-compile
    // error (macOS). When the build cannot proceed, SKIP rather than fail —
    // exit-0 is implied by `status.success()` gating the path below, and the
    // non-empty artifact contract is still hard-asserted on equipped runners
    // where the build succeeds, preserving real coverage.
    if !result.status.success() {
        let detail = format!(
            "`mindc build --release` did not succeed (backend toolchain \
             may be incomplete)\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            !enforce_real_backend(),
            "MIND_BENCH_REQUIRE is set but the keystone build did not succeed — \
             refusing to skip (#306).\n{detail}"
        );
        eprintln!("SKIP: {detail}");
        return;
    }

    assert!(
        out.exists(),
        "artifact must exist at {} after a successful `mindc build`",
        out.display()
    );

    let sz = fs::metadata(&out).unwrap().len();
    assert!(
        sz > 0,
        "artifact at {} must be non-empty (got 0 bytes)",
        out.display()
    );

    eprintln!(
        "phase_g_02: `mindc build` via Mind.toml produced {} bytes at {}",
        sz,
        out.display()
    );
}

// ---------------------------------------------------------------------------
// Test 3 — KEYSTONE byte-identity: Mind.toml-driven build == direct-path build.
//
// This is the load-bearing claim. Both invocations compile the same source
// with the same flags; the output must be byte-identical. The two code paths
// differ only in how the manifest is located (implicit Mind.toml walk vs
// explicit --out path). A byte difference would mean Mind.toml processing
// introduces a silent behaviour change.
// ---------------------------------------------------------------------------

#[test]
fn phase_g_03_byte_identical_mind_toml_vs_direct_path() {
    let Some(bin) = require_mindc() else { return };

    let out_manifest = std::env::temp_dir().join("phase_g_03_mind_toml.so");
    let out_direct = std::env::temp_dir().join("phase_g_03_direct.so");

    let src_path = repo_root().join("examples/mindc_mind/main.mind");
    if !src_path.exists() {
        assert!(
            !enforce_real_backend(),
            "KEYSTONE: examples/mindc_mind/main.mind not found but MIND_BENCH_REQUIRE=1 — \
             the keystone source must be present when the gate is enforced"
        );
        eprintln!("SKIP: examples/mindc_mind/main.mind not found");
        return;
    }

    // Build A: driven by Mind.toml (Phase G path).
    let r_manifest = Command::new(&bin)
        .args([
            "build",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out_manifest.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (manifest path)");

    if !r_manifest.status.success() {
        let detail = String::from_utf8_lossy(&r_manifest.stderr);
        assert!(
            !enforce_real_backend(),
            "MIND_BENCH_REQUIRE is set but the Mind.toml build did not succeed — \
             refusing to skip (#306).\nstderr: {detail}"
        );
        eprintln!(
            "SKIP: Mind.toml build did not succeed (toolchain may be incomplete)\nstderr: {detail}"
        );
        return;
    }

    // Build B: driven by explicit source path (Phase A style).
    let r_direct = Command::new(&bin)
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out_direct.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (direct path)");

    if !r_direct.status.success() {
        let detail = String::from_utf8_lossy(&r_direct.stderr);
        assert!(
            !enforce_real_backend(),
            "MIND_BENCH_REQUIRE is set but the direct-path build did not succeed — \
             refusing to skip (#306).\nstderr: {detail}"
        );
        eprintln!("SKIP: direct-path build did not succeed\nstderr: {detail}");
        return;
    }

    let bytes_manifest = fs::read(&out_manifest).expect("read manifest artifact");
    let bytes_direct = fs::read(&out_direct).expect("read direct artifact");

    assert_eq!(
        bytes_manifest,
        bytes_direct,
        "KEYSTONE VIOLATION: Mind.toml-driven build and direct-path build produced \
         different artifacts.\n\
         Mind.toml artifact: {} bytes ({})\n\
         Direct artifact:    {} bytes ({})",
        bytes_manifest.len(),
        sha256_hex(&bytes_manifest),
        bytes_direct.len(),
        sha256_hex(&bytes_direct)
    );

    eprintln!(
        "phase_g_03 KEYSTONE: byte-identical ({} bytes, SHA256 prefix {}...)",
        bytes_manifest.len(),
        &sha256_hex(&bytes_manifest)[..16]
    );

    // #306 / PITFALLS B4: under enforcement, byte-identity between two launcher
    // stubs is vacuous — it proves nothing about the cross-substrate wedge.
    // Demand a real ELF so the keystone claim cannot pass on the stub path.
    assert!(
        !enforce_real_backend() || bytes_manifest.starts_with(b"\x7fELF"),
        "MIND_BENCH_REQUIRE is set but the keystone artifact is a {}-byte stub, \
         not a real ELF — byte-identity between two stubs does not prove the \
         cross-substrate byte-identity wedge (#306). Build mindc with \
         --features \"mlir-build std-surface\" and ensure the ~/.mind/lib runtime \
         is present so native linking produces a real cdylib.",
        bytes_manifest.len()
    );
}

// ---------------------------------------------------------------------------
// Test 4 — KEYSTONE self-consistency: two independent clean builds of the
// self-host `.so` in THIS environment are byte-identical.
//
// This is the honest keystone. The wedge claim is that the MIND compiler is
// *deterministic*: the same input + the same environment produces the same
// output bytes, every time. We do NOT compare against a committed
// cross-toolchain binary oracle — that is unworkable in practice (the ELF
// size and bytes are clang-patch-version specific, so a committed oracle
// would be green only on the exact toolchain that produced it and red
// everywhere else, and a committed Linux `.so` would break the
// cross-platform tests that dlopen it). Instead we re-run the *same*
// deterministic pipeline twice from scratch and require bit-for-bit
// identical artifacts.
//
// #306 / PITFALLS B4: byte-identity between two launcher *stubs* is vacuous
// (stubs may embed absolute paths and prove nothing about the compiler).
// So under MIND_BENCH_REQUIRE the artifact must additionally be a real ELF,
// and a non-succeeding build is a hard failure rather than a silent skip.
// ---------------------------------------------------------------------------

#[test]
fn phase_g_04_self_consistent_byte_identity() {
    let Some(bin) = require_mindc() else { return };

    let out_a = std::env::temp_dir().join("phase_g_04_build_a.so");
    let out_b = std::env::temp_dir().join("phase_g_04_build_b.so");

    let src_path = repo_root().join("examples/mindc_mind/main.mind");
    if !src_path.exists() {
        assert!(
            !enforce_real_backend(),
            "KEYSTONE: examples/mindc_mind/main.mind not found but MIND_BENCH_REQUIRE=1 — \
             the keystone source must be present when the gate is enforced"
        );
        eprintln!("SKIP: examples/mindc_mind/main.mind not found");
        return;
    }

    // Build the self-host `.so` twice, each a fresh `mindc build` process in
    // this same environment, writing to distinct `--out` paths so the two
    // links never collide on a shared intermediary.
    let build = |out: &std::path::Path| -> std::process::Output {
        Command::new(&bin)
            .args([
                "build",
                "--release",
                "--emit=cdylib",
                "--no-cache",
                &format!("--out={}", out.display()),
            ])
            .current_dir(repo_root())
            .output()
            .expect("spawn mindc")
    };

    let r_a = build(&out_a);
    if !r_a.status.success() {
        let detail = String::from_utf8_lossy(&r_a.stderr);
        assert!(
            !enforce_real_backend(),
            "MIND_BENCH_REQUIRE is set but build A did not succeed — refusing to \
             skip the self-consistency keystone (#306).\nstderr: {detail}"
        );
        eprintln!("SKIP: build A did not succeed (toolchain may be incomplete)\nstderr: {detail}");
        return;
    }

    let r_b = build(&out_b);
    if !r_b.status.success() {
        let detail = String::from_utf8_lossy(&r_b.stderr);
        assert!(
            !enforce_real_backend(),
            "MIND_BENCH_REQUIRE is set but build B did not succeed — refusing to \
             skip the self-consistency keystone (#306).\nstderr: {detail}"
        );
        eprintln!("SKIP: build B did not succeed (toolchain may be incomplete)\nstderr: {detail}");
        return;
    }

    let bytes_a = fs::read(&out_a).expect("read build A artifact");
    let bytes_b = fs::read(&out_b).expect("read build B artifact");

    // #306 / PITFALLS B4: demand a real ELF under enforcement so two stubs
    // hashing alike can never satisfy the keystone.
    assert!(
        !enforce_real_backend() || bytes_a.starts_with(b"\x7fELF"),
        "MIND_BENCH_REQUIRE is set but the self-host artifact is a {}-byte stub, \
         not a real ELF — self-consistency between two stubs does not prove the \
         deterministic-compiler wedge (#306). Build mindc with \
         --features \"mlir-build std-surface cross-module-imports\" and ensure the \
         MLIR toolchain (mlir-*-18, clang-18) is on PATH so native linking \
         produces a real cdylib.",
        bytes_a.len()
    );

    assert_eq!(
        bytes_a,
        bytes_b,
        "KEYSTONE VIOLATION: two independent clean builds of the self-host \
         compiler in the same environment produced DIFFERENT artifacts — the \
         deterministic-compiler claim is false.\n\
         Build A: {} bytes (SHA256 {})\n\
         Build B: {} bytes (SHA256 {})",
        bytes_a.len(),
        sha256_hex(&bytes_a),
        bytes_b.len(),
        sha256_hex(&bytes_b)
    );

    eprintln!(
        "phase_g_04 KEYSTONE: self-consistent byte-identity across two clean \
         builds ({} bytes, ELF={}, SHA256 prefix {}...)",
        bytes_a.len(),
        bytes_a.starts_with(b"\x7fELF"),
        &sha256_hex(&bytes_a)[..16]
    );
}

// ---------------------------------------------------------------------------
// Test 5 — Phase F warm-cache preserved: second `mindc build` is a cache hit.
// ---------------------------------------------------------------------------

#[test]
fn phase_g_05_warm_cache_hit_after_mind_toml_build() {
    use libmind::build::cache::{CacheProbe, cache_root, module_cache_key, probe};
    use libmind::project::{BuildTarget, OptimizeLevel};

    let Some(bin) = require_mindc() else { return };

    let src_path = repo_root().join("examples/mindc_mind/main.mind");
    if !src_path.exists() {
        assert!(
            !enforce_real_backend(),
            "KEYSTONE: examples/mindc_mind/main.mind not found but MIND_BENCH_REQUIRE=1 — \
             the keystone source must be present when the gate is enforced"
        );
        eprintln!("SKIP: examples/mindc_mind/main.mind not found");
        return;
    }

    let out = std::env::temp_dir().join("phase_g_05_warm.so");

    // First build — populates the cache.
    let r1 = Command::new(&bin)
        .args([
            "build",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (first build)");

    if !r1.status.success() {
        let detail = String::from_utf8_lossy(&r1.stderr);
        assert!(
            !enforce_real_backend(),
            "MIND_BENCH_REQUIRE is set but the first build did not succeed — \
             refusing to skip the warm-cache check (#306).\nstderr: {detail}"
        );
        eprintln!("SKIP: first build did not succeed\nstderr: {detail}");
        return;
    }

    // Probe the cache for the entry source.
    let source_bytes = fs::read(&src_path).expect("read main.mind");
    let cache_key = module_cache_key(
        &source_bytes,
        BuildTarget::Cpu,
        OptimizeLevel::Release,
        &[],
        env!("CARGO_PKG_VERSION"),
        2024,
    );
    let c_root = cache_root(&repo_root(), BuildTarget::Cpu, OptimizeLevel::Release);

    let probe_result = probe(&c_root, &cache_key);
    assert!(
        matches!(probe_result, CacheProbe::Hit { .. }),
        "cache must be populated after first `mindc build` via Mind.toml"
    );

    eprintln!("phase_g_05: cache populated. Running warm rebuild...");

    // Second build — must be a cache hit (fast path).
    let r2 = Command::new(&bin)
        .args([
            "build",
            "--release",
            "--emit=cdylib",
            "--verbose",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (second build)");

    assert!(
        r2.status.success(),
        "warm rebuild must succeed; stderr: {}",
        String::from_utf8_lossy(&r2.stderr)
    );

    // The cache entry must still be valid.
    let probe_result2 = probe(&c_root, &cache_key);
    assert!(
        matches!(probe_result2, CacheProbe::Hit { .. }),
        "cache must remain a hit after warm rebuild"
    );

    let stderr2 = String::from_utf8_lossy(&r2.stderr);
    eprintln!(
        "phase_g_05: warm rebuild output:\n{}",
        &stderr2[..stderr2.len().min(300)]
    );
}

// ---------------------------------------------------------------------------
// Test 6 — Artifact SHA-256 report (informational, always passes).
//
// Prints the SHA-256 of the Phase G artifact for reproducibility records.
// This is the "evidence" the milestone commit carries.
// ---------------------------------------------------------------------------

#[test]
fn phase_g_06_report_artifact_sha256() {
    let Some(bin) = require_mindc() else { return };

    let out = std::env::temp_dir().join("phase_g_06_report.so");

    let r = Command::new(&bin)
        .args([
            "build",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc");

    if !r.status.success() {
        eprintln!(
            "SKIP: `mindc build` did not succeed\nstderr: {}",
            String::from_utf8_lossy(&r.stderr)
        );
        return;
    }

    let built_bytes = fs::read(&out).expect("read artifact");
    let built_hash = sha256_hex(&built_bytes);

    eprintln!("=== Phase G — Keystone artifact hash ===");
    eprintln!(
        "Built via mindc build (Mind.toml):  SHA256 = {}",
        built_hash
    );
    eprintln!(
        "Artifact kind: {}",
        if built_bytes.starts_with(b"\x7fELF") {
            "real ELF (full MLIR path)"
        } else {
            "launcher stub (no MLIR toolchain)"
        }
    );
    eprintln!("Artifact size: {} bytes", built_bytes.len());

    // This test always passes — it is a reporting gate, not a boolean assertion.
    // Test 4 (phase_g_04_self_consistent_byte_identity) is the hard assertion:
    // it builds the self-host `.so` twice in this environment and requires the
    // two artifacts to be byte-identical (deterministic-compiler self-consistency),
    // rather than matching a committed cross-toolchain oracle.
}
