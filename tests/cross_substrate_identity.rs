// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0020 §10 — the **internal** mind-bench reproducibility gate.
//!
//! This is the in-tree merge gate that produces the very reference hash the
//! public `mind-bench` CLI (RFC 0020 §3) and the published
//! `mind-spec/wedge-reference-hashes/<version>.txt` manifest will consume —
//! single source of truth, two consumers (RFC 0020 §4.3). It runs a workload's
//! deterministic kernel, serialises the output canonically, sha256-hashes it,
//! and asserts the hash equals the per-substrate reference committed in the
//! workload's `reference_hashes.toml`.
//!
//! The property under test is **byte-identity across builds, machines and
//! time** — stronger than `blas_vec_q16_smoke.rs`, which proves only that the
//! vector path equals its own scalar oracle within a single run. Here the
//! exact output bytes are pinned to a committed constant, so any drift in
//! mindc lowering / std-surface / libc-syscall surfaces as a hash mismatch.
//!
//! Per RFC 0015 §3.1 every substrate listed in a Q16.16 workload's manifest
//! MUST share the SAME content hash; the per-substrate lines in
//! `reference_hashes.toml` therefore carry one identical hash with
//! substrate-specific provenance — cross-substrate bit-identity made
//! inspectable. This host verifies its own substrate (avx2 on x86_64, neon on
//! aarch64); other substrates are verified on their own CI runners (RFC 0020
//! §10) and are `deferred` here, never `pass`.
//!
//! Run: `cargo test --features "mlir-build std-surface cross-module-imports" \
//!       --test cross_substrate_identity`. Self-skips without the MLIR
//! toolchain (mlir-opt / mlir-translate / clang), like the blas smoke tests.
//!
//! Re-bless after an *intentional* lowering change (RFC 0020 §13): run with
//! `MIND_BENCH_BLESS=1` to print the computed hash, then commit it.

#![cfg(all(
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#![cfg(not(windows))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// The host substrate id, per RFC 0014 tier naming. The workload's reference
/// hash is looked up under this key; a substrate the host cannot run is
/// `deferred` (verified on its own runner), never silently passed.
fn host_substrate() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "avx2"
    } else if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "unknown"
    }
}

/// Direct-intrinsic source: the Track B Q16.16 dot path, lowered inside mindc
/// to a native `vector`-dialect reduction (no func.call, no C shim) — the same
/// entry point `blas_vec_q16_smoke.rs` exercises.
const SRC: &str = r#"
pub fn dotq(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_q16_v(a, b, n)
}
"#;

type DotFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

fn mindc_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dbg = manifest_dir.join("target").join("debug").join("mindc");
    if dbg.exists() {
        return dbg;
    }
    let rel = manifest_dir.join("target").join("release").join("mindc");
    assert!(
        rel.exists(),
        "mindc binary not found at {dbg:?} or {rel:?}; build with: \
         cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc"
    );
    rel
}

/// Compile SRC to a temp `.so` once for the whole test binary. Returns `None`
/// if the MLIR toolchain is shadowed (sandbox self-skip, like the smoke tests).
fn build_dot_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                println!("cross_substrate_identity: {tool} not on PATH; skipping");
                return None;
            }
        }
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_xsi_dot_q16.mind");
        let so_path = dir.join("mind_xsi_dot_q16.so");
        std::fs::write(&src_path, SRC).expect("write workload .mind source");
        let status = Command::new(mindc_path())
            .args([src_path.to_str().unwrap(), "--emit-shared", so_path.to_str().unwrap()])
            .status()
            .expect("spawn mindc --emit-shared");
        assert!(status.success(), "mindc --emit-shared failed for the dot-q16 workload");
        Some(so_path)
    })
    .as_ref()
}

/// Deterministic LCG — byte-for-byte the generator `blas_vec_q16_smoke.rs`
/// uses, so the workload's input distribution is shared and reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u32
    }
    fn next_q16(&mut self) -> i32 {
        (self.next_u32() as i32) >> 12
    }
}

/// Regenerate the workload input from its seed (manifest `[input]`).
fn make_pair_q16(len: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i32> = (0..len).map(|_| g.next_q16()).collect();
    let b: Vec<i32> = (0..len).map(|_| g.next_q16()).collect();
    (a, b)
}

/// Track A scalar oracle, byte-for-byte (`mind_blas_dot_q16_scalar`): the
/// independent reference the vector result must match within a run.
fn ref_dot_q16_scalar(a: &[i32], b: &[i32]) -> i64 {
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        acc += ((a[i] as i64) * (b[i] as i64)) >> 16;
    }
    (acc as i32) as i64
}

/// Workload spec, mirroring tests/cross_substrate_identity/dot-l2-q16/manifest.toml.
/// (A full TOML reader lands with the pure-MIND CLI; the internal gate pins the
/// values here and the manifest documents them for the public consumer.)
const WORKLOAD_ID: &str = "dot-l2-q16";
const SEED: u64 = 0xDEADBEEF;
const LENGTH: usize = 65536;

fn workload_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cross_substrate_identity")
        .join(WORKLOAD_ID)
}

/// Read the committed reference hash for a substrate from reference_hashes.toml.
/// Format: `<substrate> = "<sha256>"` lines (minimal parse — no toml dep).
fn reference_hash(substrate: &str) -> Option<String> {
    let path = workload_dir().join("reference_hashes.toml");
    let text = std::fs::read_to_string(&path).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == substrate {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Canonical output encoding (manifest `output_encoding = "i64_le"`): the 8
/// little-endian bytes of the result, then sha256 → lowercase hex.
fn canonical_hash(result: i64) -> String {
    let mut h = Sha256::new();
    h.update(result.to_le_bytes());
    format!("{:x}", h.finalize())
}

#[test]
fn dot_l2_q16_reproducibility_gate() {
    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let dotq: Symbol<DotFn> = unsafe { lib.get(b"dotq").expect("dotq symbol") };

    let (a, b) = make_pair_q16(LENGTH, SEED);

    // 1. Vector path == scalar oracle (within-run exactness; integer
    //    reduction is associative so this is exact, not approximate).
    let vec_result = unsafe { dotq(a.as_ptr() as i64, b.as_ptr() as i64, LENGTH as i64) };
    let oracle = ref_dot_q16_scalar(&a, &b);
    assert_eq!(
        vec_result, oracle,
        "{WORKLOAD_ID}: vector path diverged from scalar oracle within a single run"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference
    //    (across-build / across-machine / across-time byte-identity).
    let computed = canonical_hash(vec_result);
    let substrate = host_substrate();

    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {WORKLOAD_ID} {substrate} {computed}");
        return;
    }

    match reference_hash(substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{WORKLOAD_ID} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result_i64={vec_result}\n\
             If this is an intentional lowering change (RFC 0020 §13), re-bless with \
             MIND_BENCH_BLESS=1 and commit the new reference_hashes.toml."
        ),
        None => panic!(
            "{WORKLOAD_ID}: no reference hash for substrate '{substrate}' in \
             reference_hashes.toml. Computed hash is {computed} (result_i64={vec_result}); \
             bless it with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
}
