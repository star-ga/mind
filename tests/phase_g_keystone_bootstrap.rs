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

mod common;
use common::gate;
use common::mindc_bin;

/// RFC 0015 / `#306` fail-closed gate, re-exported from its ONE owner.
///
/// When `MIND_BENCH_REQUIRE=1` is set, the keystone tests must NOT silently
/// skip or report PASS on a launcher stub: a missing backend toolchain or a
/// stub-vs-ELF artifact mismatch becomes a hard failure. This closes the
/// false-green documented in `docs/byte-store-migration.md` ("DO NOT capture
/// the oracle from a stub-producing environment") — when `mindc build` falls
/// back to a ~1245-byte launcher script, byte-identity between two stubs is
/// vacuous, yet the suite still reported PASS.
///
/// This file used to define its own copy reading
/// `var_os("MIND_BENCH_REQUIRE").is_some()`, which made `MIND_BENCH_REQUIRE=0`
/// ENFORCE — the same defect `gate::bless_mode` documents, where a value that
/// reads as "off" switched a mode ON. `gate::enforce_real_backend` requires the
/// value to be exactly `1`, and there is now one spelling of the predicate.
use common::gate::enforce_real_backend;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

fn require_mindc() -> Option<PathBuf> {
    let bin = mindc_bin();
    if bin.exists() {
        Some(bin)
    } else {
        // Fail-closed under enforcement: a missing mindc must NOT let the
        // keystone suite pass vacuously (#306 false-green guard). Only a
        // non-enforced (local convenience) run is allowed to skip, and the
        // skip is COUNTED (`ran=0`) rather than announced into cargo's capture.
        gate::skipped(
            "phase_g_keystone_bootstrap",
            &format!(
                "mindc binary not found at {}; build it (cargo build --release \
                 --features \"mlir-build std-surface cross-module-imports\" --bin mindc)",
                bin.display()
            ),
        );
        None
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The artifact root for this target: private to the test BINARY and to the
/// PROCESS (`common::scratch_dir` appends target + pid).
///
/// The keystone artifacts used a FIXED shared temp path
/// (`/tmp/phase_g_03_direct.so` and siblings). Two runs of this gate on one box
/// — two agents, two CI jobs on one runner, a `cargo test` beside a preflight —
/// write the SAME file, so one build can truncate the artifact the other is
/// about to read and byte-compare. The load-bearing byte-identity claim would
/// then be reported as violated by a collision it never made. File names are
/// unchanged; only the directory moves off the shared root.
fn scratch() -> PathBuf {
    crate::common::scratch_dir("phase_g_keystone_bootstrap")
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
fn has_attribute_token(src: &str) -> bool {
    // Match MIND's double-quoted strings and line comments. A mention in
    // either is data, while `#[` in code remains a bootstrap precondition.
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'#' if bytes.get(i + 1) == Some(&b'[') => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

#[test]
fn bootstrap_source_is_attribute_free() {
    for src in [
        "// expands #[bimap]\r\nfn main() {}",
        r##"fn text() { "#[target]"; }"##,
        r##"fn text() { "escaped \" #[target] // λ"; } // #[test]"##,
    ] {
        assert!(
            !has_attribute_token(src),
            "prose is not an attribute: {src}"
        );
    }
    for src in [
        "#[deterministic]\nfn main() {}",
        "// #[comment]\n#[test]\nfn main() {}",
        r#"fn text() { "escaped \\"; } #[target] fn main() {}"#,
        "fn main() { #[collapse] for i in 0..2 {} }",
    ] {
        assert!(has_attribute_token(src), "missed an attribute: {src}");
    }
    for name in ["main.mind", "fixture.mind"] {
        let path = repo_root().join("examples/mindc_mind").join(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
        assert!(
            !has_attribute_token(&src),
            "examples/mindc_mind/{name} contains an attribute `#[` but the \
                 pure-MIND self-host parser is attribute-blind — this would break \
                 the byte-identity bootstrap oracle. Teach `parse_item` to parse \
                 attributes before annotating the compiler's own source."
        );
    }
}

/// Return the lowercase hex-encoded FIPS 180-4 SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    format!("{:x}", Sha256::digest(bytes))
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

    let out = scratch().join("phase_g_02_libmindc_mind.so");

    let result = Command::new(&bin)
        .args([
            "build",
            "--backend",
            "mlir",
            "--release",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn mindc");

    // WHICH failure this was decides whether it may skip, and the CAUSE CODE
    // decides that — never the call site. A host that genuinely cannot ship the
    // backend still skips (`[E5003]` no native backend, `[E5004]` no
    // `mlir-opt`/`clang` on PATH) and is counted `ran=0`; every other failure
    // panics with the stderr quoted, on EVERY run and not only under
    // `MIND_BENCH_REQUIRE=1`.
    //
    // The six build sites in this file used to name the absence themselves —
    // `gate::skipped(.., "backend toolchain may be incomplete")` — which is a
    // GUESS about a failure whose cause the call site is holding. Measured with
    // a stray token in the tracked `examples/mindc_mind/main.mind` and
    // `MIND_BENCH_REQUIRE` unset: `error[parse][E1001]` printed
    // `ran=0 class=toolchain` here and this test reported `ok`. The rule has one
    // owner, `common::gate::compiled`, and `tests/fail_open_skip_site_ratchet.rs`
    // now forbids an exit-status skip that does not reach it.
    if !gate::compiled("phase_g_keystone_bootstrap", &result) {
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

    let out_manifest = scratch().join("phase_g_03_mind_toml.so");
    let out_direct = scratch().join("phase_g_03_direct.so");

    let src_path = repo_root().join("examples/mindc_mind/main.mind");
    // The keystone source is TRACKED in this repo, so its absence is a broken
    // checkout, never a capability gap. The MIND_BENCH_REQUIRE-conditional skip
    // this replaced graded a broken checkout as a PASS on every run that did not
    // set the variable.
    assert!(
        src_path.exists(),
        "KEYSTONE: {} is tracked in this repo but missing; the keystone gate \
         cannot run",
        src_path.display()
    );

    // Build A: driven by Mind.toml (Phase G path).
    let r_manifest = Command::new(&bin)
        .args([
            "build",
            "--backend=mlir",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out_manifest.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (manifest path)");

    // Cause decides, exactly as in phase_g_02: a capability gap skips and is
    // counted, a compiler regression panics with the stderr quoted.
    if !gate::compiled("phase_g_keystone_bootstrap", &r_manifest) {
        return;
    }

    // Build B: driven by explicit source path (Phase A style).
    let r_direct = Command::new(&bin)
        .args([
            "build",
            "--backend=mlir",
            src_path.to_str().unwrap(),
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out_direct.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (direct path)");

    if !gate::compiled("phase_g_keystone_bootstrap", &r_direct) {
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

    let out_a = scratch().join("phase_g_04_build_a.so");
    let out_b = scratch().join("phase_g_04_build_b.so");

    let src_path = repo_root().join("examples/mindc_mind/main.mind");
    // The keystone source is TRACKED in this repo, so its absence is a broken
    // checkout, never a capability gap. The MIND_BENCH_REQUIRE-conditional skip
    // this replaced graded a broken checkout as a PASS on every run that did not
    // set the variable.
    assert!(
        src_path.exists(),
        "KEYSTONE: {} is tracked in this repo but missing; the keystone gate \
         cannot run",
        src_path.display()
    );

    // Build the self-host `.so` twice, each a fresh `mindc build` process in
    // this same environment, writing to distinct `--out` paths so the two
    // links never collide on a shared intermediary.
    let build = |out: &std::path::Path| -> std::process::Output {
        Command::new(&bin)
            .args([
                "build",
                "--backend=mlir",
                "--release",
                "--emit=cdylib",
                "--no-cache",
                &format!("--out={}", out.display()),
            ])
            .current_dir(repo_root())
            .output()
            .expect("spawn mindc")
    };

    // Cause decides for both builds: only `[E5003]`/`[E5004]` may skip here.
    let r_a = build(&out_a);
    if !gate::compiled("phase_g_keystone_bootstrap", &r_a) {
        return;
    }

    let r_b = build(&out_b);
    if !gate::compiled("phase_g_keystone_bootstrap", &r_b) {
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
    use libmind::build::cache::{CacheProbe, cache_root, probe};
    use libmind::project::{BuildTarget, EmitKind, OptimizeLevel};

    let Some(bin) = require_mindc() else { return };

    let src_path = repo_root().join("examples/mindc_mind/main.mind");
    // The keystone source is TRACKED in this repo, so its absence is a broken
    // checkout, never a capability gap. The MIND_BENCH_REQUIRE-conditional skip
    // this replaced graded a broken checkout as a PASS on every run that did not
    // set the variable.
    assert!(
        src_path.exists(),
        "KEYSTONE: {} is tracked in this repo but missing; the keystone gate \
         cannot run",
        src_path.display()
    );

    let out = scratch().join("phase_g_05_warm.so");

    // First build — populates the cache.
    let r1 = Command::new(&bin)
        .args([
            "build",
            "--backend=mlir",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (first build)");

    // Cause decides: a cache probe over a build that FAILED for any reason
    // other than a capability gap would be measuring the previous run.
    if !gate::compiled("phase_g_keystone_bootstrap", &r1) {
        return;
    }

    // Probe the cache for the entry source.
    let source_bytes = fs::read(&src_path).expect("read main.mind");
    // Recompute the key with the SAME helper the build path uses, fingerprinting
    // the mindc SUBPROCESS binary (`bin` == CARGO_BIN_EXE_mindc), NOT this test
    // executable — otherwise the compiler fingerprint (issue #96) would never
    // match the subprocess-populated cache and this warm-cache probe would
    // spuriously miss.
    // The key fingerprints EVERY source the build compiles, not just the entry
    // (an edit to a sibling module must invalidate). Resolve that set through
    // the compiler's own selector so this probe and the build cannot disagree
    // about which files — or in which order — the key covers.
    let (build_sources, _) = libmind::project::resolve_sources(
        &repo_root(),
        "examples/mindc_mind/main.mind",
        None,
        false,
    )
    .expect("resolve the keystone project sources");
    let cache_material = libmind::build::compile_cache_material(
        &source_bytes,
        libmind::build::CacheKeyFlags {
            target: BuildTarget::Cpu,
            optimize: OptimizeLevel::Release,
            emit: EmitKind::Cdylib,
            edition: 2024,
        },
        &bin,
        &repo_root(),
        &build_sources,
    )
    .expect("compiler identity for CARGO_BIN_EXE_mindc");
    let c_root = cache_root(&repo_root(), BuildTarget::Cpu, OptimizeLevel::Release);

    let probe_result = probe(&c_root, &cache_material);
    assert!(
        matches!(probe_result, CacheProbe::Hit { .. }),
        "cache must be populated after first `mindc build` via Mind.toml"
    );

    eprintln!("phase_g_05: cache populated. Running warm rebuild...");

    // Second build — must be a cache hit (fast path).
    let r2 = Command::new(&bin)
        .args([
            "build",
            "--backend=mlir",
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
    let probe_result2 = probe(&c_root, &cache_material);
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
// Test 6 — Artifact SHA-256 evidence: the reported hash is REPRODUCIBLE.
//
// This test used to be annotated "informational, always passes": it printed a
// hash, asserted nothing, and held the only unguarded skip in this file. It was
// still counted in the suite's "7/7 byte-identical" headline, so one seventh of
// that number was a print statement. A gate that cannot fail is not evidence.
//
// What it now asserts, and why it is not a duplicate of 03/04:
//   * 03 compares the Mind.toml route against the direct-path route, both
//     through the BUILD CACHE.
//   * 04 compares two `--no-cache` builds against each other.
//   * Nothing compared the CACHED route against the CACHE-FREE one — so a cache
//     that returned subtly different bytes would satisfy both, and the hash this
//     test records into the milestone commit would be a hash no cache-free
//     rebuild reproduces. That is exactly the claim an evidence record makes.
//
// So the recorded hash is re-derived from an independent `--no-cache` rebuild
// and must match, bytes and all. The byte compare backs the hash compare so a
// degenerate hash function cannot launder a mismatch, and the artifact hash is
// required to differ from the hash of an empty input so the assertion cannot
// pass on a hash that ignores its argument.
//
// Both build steps route their skip through the one fail-closed capability
// decision (`common::gate::compiled`) rather than open-coding the enforcement
// check: a genuine capability gap still skips, an undiagnosed build failure
// panics with the stderr quoted, and under MIND_BENCH_REQUIRE=1 neither skips.
// ---------------------------------------------------------------------------

#[test]
fn phase_g_06_report_artifact_sha256() {
    let Some(bin) = require_mindc() else { return };

    let out = scratch().join("phase_g_06_report.so");
    let out_recheck = scratch().join("phase_g_06_report_nocache.so");

    let r = Command::new(&bin)
        .args([
            "build",
            "--backend=mlir",
            "--release",
            "--emit=cdylib",
            &format!("--out={}", out.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc");

    // Every sibling keystone test asserts the build; this one printed and passed,
    // so a keystone build regression graded green here. Only a genuine capability
    // gap may skip, and under MIND_BENCH_REQUIRE=1 not even that.
    if !crate::common::gate::compiled("phase_g_keystone_bootstrap", &r) {
        return;
    }

    let built_bytes = fs::read(&out).expect("read artifact");
    let built_hash = sha256_hex(&built_bytes);

    eprintln!("=== Phase G — Keystone artifact hash ===");
    eprintln!("Built via mindc build (Mind.toml):  SHA256 = {built_hash}");
    eprintln!(
        "Artifact kind: {}",
        if built_bytes.starts_with(b"\x7fELF") {
            "real ELF (full MLIR path)"
        } else {
            "launcher stub (no MLIR toolchain)"
        }
    );
    eprintln!("Artifact size: {} bytes", built_bytes.len());

    // The report has to be about a real artifact before it can be about
    // anything: a 0-byte file hashes just fine.
    assert!(
        !built_bytes.is_empty(),
        "the artifact at {} is empty — an evidence record over 0 bytes states nothing",
        out.display()
    );

    // Positive control: a hash that ignores its input would satisfy every
    // equality below. Require the recorded hash to depend on the artifact.
    assert_ne!(
        built_hash,
        sha256_hex(&[]),
        "the recorded artifact hash equals the hash of an EMPTY input — the hash \
         function is not reading the artifact, so the evidence record is vacuous"
    );
    assert!(
        built_hash.len() == 64 && built_hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "the recorded hash must be a 64-character hex SHA-256; got {built_hash:?}"
    );

    // #306 / PITFALLS B4: under enforcement the evidence must be about a real
    // ELF — a launcher stub's hash records the stub, not the compiler.
    assert!(
        !enforce_real_backend() || built_bytes.starts_with(b"\x7fELF"),
        "MIND_BENCH_REQUIRE is set but the recorded artifact is a {}-byte stub, \
         not a real ELF — an evidence hash over a stub proves nothing about the \
         deterministic-compiler wedge (#306).",
        built_bytes.len()
    );

    // Re-derive the recorded hash through the CACHE-FREE route. If the build
    // cache is not byte-transparent, the hash carried by the milestone commit is
    // not the hash a clean rebuild produces.
    let r2 = Command::new(&bin)
        .args([
            "build",
            "--backend=mlir",
            "--release",
            "--emit=cdylib",
            "--no-cache",
            &format!("--out={}", out_recheck.display()),
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn mindc (--no-cache recheck)");

    // Same one fail-closed decision as the first build: a capability gap skips,
    // anything else panics with the stderr quoted.
    if !crate::common::gate::compiled("phase_g_keystone_bootstrap", &r2) {
        return;
    }

    let recheck_bytes = fs::read(&out_recheck).expect("read --no-cache artifact");
    let recheck_hash = sha256_hex(&recheck_bytes);

    assert_eq!(
        built_hash,
        recheck_hash,
        "KEYSTONE EVIDENCE VIOLATION: the recorded artifact hash is not reproducible \
         through a cache-free rebuild — the build cache is not byte-transparent, so \
         the hash this gate records is a hash no clean rebuild produces.\n\
         cached build:   {} bytes (SHA256 {built_hash})\n\
         --no-cache:     {} bytes (SHA256 {recheck_hash})",
        built_bytes.len(),
        recheck_bytes.len()
    );
    assert_eq!(
        built_bytes, recheck_bytes,
        "KEYSTONE EVIDENCE VIOLATION: cached and --no-cache artifacts hash alike but \
         differ byte-for-byte — the recorded hash is not identifying the artifact"
    );

    eprintln!(
        "phase_g_06 EVIDENCE: recorded hash reproduced by a cache-free rebuild \
         ({} bytes, ELF={}, SHA256 prefix {}...)",
        built_bytes.len(),
        built_bytes.starts_with(b"\x7fELF"),
        &built_hash[..16]
    );
}
