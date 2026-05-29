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
pub fn dotl1q(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_l1_q16_v(a, b, n)
}
pub fn mmq(w: i64, x: i64, y: i64, rows: i64, cols: i64) -> i64 {
    __mind_blas_matmul_rmajor_q16_v(w, x, y, rows, cols)
}
fn gemmq_row(a: i64, bt: i64, c: i64, m: i64, k: i64, n: i64, i: i64) -> i64 {
    if i >= m {
        return 0;
    }
    // Q16.16 elements are i32 (4 bytes); row strides are in i32 units.
    let a_i: i64 = a + i * k * 4;
    let c_i: i64 = c + i * n * 4;
    __mind_blas_matmul_rmajor_q16_v(bt, a_i, c_i, n, k);
    gemmq_row(a, bt, c, m, k, n, i + 1)
}
pub fn gemmq(a: i64, bt: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    gemmq_row(a, bt, c, m, k, n, 0)
}
"#;

type DotFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
type MatmulFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

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
                // CI sets MIND_BENCH_REQUIRE=1 so a missing toolchain fails the
                // gate loudly instead of self-skipping. A silent skip would turn
                // the cross-substrate bit-identity check into a vacuous green if
                // the MLIR install ever broke on a runner (RFC 0020 §10) — the
                // whole point of the gate is that it cannot pass without running.
                // Local/sandbox runs without the var keep self-skipping, like the
                // blas smoke tests.
                assert!(
                    std::env::var_os("MIND_BENCH_REQUIRE").is_none(),
                    "MIND_BENCH_REQUIRE is set but '{tool}' is not on PATH: the \
                     cross-substrate gate cannot run. Install the MLIR toolchain \
                     (mlir-opt / mlir-translate / clang) on this runner."
                );
                println!("cross_substrate_identity: {tool} not on PATH; skipping");
                return None;
            }
        }
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_xsi_dot_q16.mind");
        let so_path = dir.join("mind_xsi_dot_q16.so");
        std::fs::write(&src_path, SRC).expect("write workload .mind source");
        let status = Command::new(mindc_path())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc --emit-shared");
        assert!(
            status.success(),
            "mindc --emit-shared failed for the dot-q16 workload"
        );
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

/// Track A scalar oracle for Q16.16 L1, byte-for-byte
/// (`mind_blas_dot_l1_q16_scalar`): `d=|a-b|` accumulated, then `(i64)(i32)acc`.
fn ref_dot_l1_q16_scalar(a: &[i32], b: &[i32]) -> i64 {
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        let mut d = (a[i] as i64) - (b[i] as i64);
        if d < 0 {
            d = -d;
        }
        acc += d;
    }
    (acc as i32) as i64
}

/// A dot-style workload, mirroring tests/cross_substrate_identity/<id>/manifest.toml.
/// (A full TOML reader lands with the pure-MIND CLI; the internal gate pins the
/// values here and the manifest documents them for the public consumer.) Every
/// field except `oracle` is also stated in the manifest — single source of truth.
struct DotWorkload {
    id: &'static str,
    symbol: &'static [u8],
    seed: u64,
    length: usize,
    /// Independent scalar reference the vector path must match within a run.
    oracle: fn(&[i32], &[i32]) -> i64,
}

fn workload_dir(id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("cross_substrate_identity")
        .join(id)
}

/// Read the committed reference hash for a substrate from reference_hashes.toml.
/// Format: `<substrate> = "<sha256>"` lines (minimal parse — no toml dep).
fn reference_hash(id: &str, substrate: &str) -> Option<String> {
    let path = workload_dir(id).join("reference_hashes.toml");
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

/// Run one dot-style workload: build/dlopen the kernel, regenerate the seeded
/// input, run the vector path, cross-check the scalar oracle (within-run
/// exactness — the integer reduction is associative, so this is exact), then
/// pin the canonical output hash to the committed per-substrate reference
/// (across-build / across-machine / across-time byte-identity). Self-skips
/// without the MLIR toolchain.
fn run_dot_workload(w: &DotWorkload) {
    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let dot: Symbol<DotFn> = unsafe { lib.get(w.symbol).expect("workload symbol") };

    let (a, b) = make_pair_q16(w.length, w.seed);

    let vec_result = unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, w.length as i64) };
    let oracle = (w.oracle)(&a, &b);
    assert_eq!(
        vec_result, oracle,
        "{}: vector path diverged from scalar oracle within a single run",
        w.id
    );

    let computed = canonical_hash(vec_result);
    let substrate = host_substrate();

    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {} {substrate} {computed}", w.id);
        return;
    }

    match reference_hash(w.id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result_i64={vec_result}\n\
             If this is an intentional lowering change (RFC 0020 §13), re-bless with \
             MIND_BENCH_BLESS=1 and commit the new reference_hashes.toml.",
            w.id
        ),
        None => panic!(
            "{}: no reference hash for substrate '{substrate}' in reference_hashes.toml. \
             Computed hash is {computed} (result_i64={vec_result}); bless it with \
             MIND_BENCH_BLESS=1 if this host is canonical.",
            w.id
        ),
    }
}

#[test]
fn dot_l2_q16_reproducibility_gate() {
    run_dot_workload(&DotWorkload {
        id: "dot-l2-q16",
        symbol: b"dotq",
        seed: 0xDEADBEEF,
        length: 65536,
        oracle: ref_dot_q16_scalar,
    });
}

#[test]
fn dot_l1_q16_reproducibility_gate() {
    run_dot_workload(&DotWorkload {
        id: "dot-l1-q16",
        symbol: b"dotl1q",
        seed: 0xDEADBEEF,
        length: 65536,
        oracle: ref_dot_l1_q16_scalar,
    });
}

// --- gemv-q16 workload (matrix x vector) -----------------------------------
// The output is a `rows`-length Q16.16 vector (i32 each), not a scalar, so the
// canonical encoding is the y buffer's bytes (rows * 4 LE) → sha256.

/// Regenerate the gemv inputs from a seed: a rows*cols Q16.16 matrix W
/// (row-major) and a cols-length Q16.16 vector x.
fn make_gemv_q16(rows: usize, cols: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let w: Vec<i32> = (0..rows * cols).map(|_| g.next_q16()).collect();
    let x: Vec<i32> = (0..cols).map(|_| g.next_q16()).collect();
    (w, x)
}

/// Scalar Q16.16 gemv oracle: y[r] = dot_q16(W row r, x).
fn ref_gemv_q16_scalar(w: &[i32], x: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows)
        .map(|r| ref_dot_q16_scalar(&w[r * cols..(r + 1) * cols], x) as i32)
        .collect()
}

/// Canonical hash of a Q16.16 vector: each i32 little-endian, then sha256.
fn canonical_hash_i32s(v: &[i32]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update(e.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[test]
fn gemv_q16_reproducibility_gate() {
    let id = "gemv-q16-256x256";
    let (rows, cols, seed) = (256usize, 256usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let mmq: Symbol<MatmulFn> = unsafe { lib.get(b"mmq").expect("mmq symbol") };

    let (w, x) = make_gemv_q16(rows, cols, seed);
    let mut y = vec![0i32; rows];
    let rc = unsafe {
        mmq(
            w.as_ptr() as i64,
            x.as_ptr() as i64,
            y.as_mut_ptr() as i64,
            rows as i64,
            cols as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: kernel returned {rc} (expected 0)");

    // 1. Within-run exactness vs the scalar gemv oracle.
    let oracle = ref_gemv_q16_scalar(&w, &x, rows, cols);
    assert_eq!(
        y, oracle,
        "{id}: gemv vector path diverged from the scalar oracle"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference.
    let computed = canonical_hash_i32s(&y);
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13)."
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed}; \
             bless with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
}

// --- gemm-q16 workload (square matrix x matrix) ----------------------------
// The first matmul-SHAPED workload that is matrix x matrix, not matrix x
// vector. C[M,N] = A[M,K] · B[K,N] in Q16.16. The kernel composes the
// already-proven gemv intrinsic: C[i,:] = gemv(Bᵀ, A[i,:]) where Bᵀ is N×K,
// so byte-identity is inherited from `gemv_q16_reproducibility_gate` — no new
// arithmetic, only a deterministic transpose (exact data movement, done in the
// harness) plus a deterministic ascending row loop. The output is the M×N
// Q16.16 matrix; canonical encoding is its i32 LE bytes → sha256.

/// Regenerate the gemm inputs from a seed: an M×K matrix A and a K×N matrix B,
/// both row-major Q16.16, A generated before B (order is part of the seed
/// contract). Returns (A, B, Bᵀ) where Bᵀ (N×K, row-major) is the exact
/// transpose the kernel consumes.
fn make_gemm_q16(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i32> = (0..m * k).map(|_| g.next_q16()).collect();
    let b: Vec<i32> = (0..k * n).map(|_| g.next_q16()).collect();
    // Bᵀ[j, kk] = B[kk, j] — exact data movement, no arithmetic.
    let mut bt = vec![0i32; n * k];
    for kk in 0..k {
        for j in 0..n {
            bt[j * k + kk] = b[kk * n + j];
        }
    }
    (a, b, bt)
}

/// Scalar Q16.16 GEMM oracle, expressed as M·N scalar dot products over Bᵀ —
/// byte-for-byte the same accumulation the kernel performs via gemv, so the
/// within-run cross-check is exact (integer reduction is associative).
fn ref_gemm_q16_scalar(a: &[i32], bt: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let bt_row = &bt[j * k..(j + 1) * k];
            c[i * n + j] = ref_dot_q16_scalar(a_row, bt_row) as i32;
        }
    }
    c
}

#[test]
fn gemm_q16_reproducibility_gate() {
    let id = "gemm-q16-64x64x64";
    let (m, k, n, seed) = (64usize, 64usize, 64usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };

    let (a, _b, bt) = make_gemm_q16(m, k, n, seed);
    let mut c = vec![0i32; m * n];
    let rc = unsafe {
        gemmq(
            a.as_ptr() as i64,
            bt.as_ptr() as i64,
            c.as_mut_ptr() as i64,
            m as i64,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: kernel returned {rc} (expected 0)");

    // 1. Within-run exactness vs the scalar GEMM oracle.
    let oracle = ref_gemm_q16_scalar(&a, &bt, m, k, n);
    assert_eq!(
        c, oracle,
        "{id}: gemm vector path diverged from the scalar oracle"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference.
    let computed = canonical_hash_i32s(&c);
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13)."
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed}; \
             bless with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
}
