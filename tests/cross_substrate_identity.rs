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
pub fn mmi16(w: i64, x: i64, y: i64, rows: i64, cols: i64) -> i64 {
    __mind_blas_matmul_rmajor_i16_v(w, x, y, rows, cols)
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
pub fn gemmi8(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_i8_v(a, b, c, m, k, n)
}
// RFC 0012 §5.1 — deterministic scalar IEEE-754 f64 elementwise chain. A fixed
// sequence of scalar `+ − × ÷` over four f64 inputs supplied by the harness.
// Lowers to strict IEEE `arith.addf/subf/mulf/divf` (vaddsd/vmulsd/vdivsd/vsubsd
// on avx2; the aarch64 equivalents on neon) — NO FMA fusion (`c * d` stays a
// separate mulf, never contracted into the add), NO fastmath/reassoc flags. The
// operation order is fully fixed by source precedence: `a + b - (c * d / a)`.
// Scalar IEEE `+ − × ÷` are round-to-nearest-even with no contraction or
// reassociation, so unlike a float REDUCTION (order-sensitive) the result is
// byte-identical across x86 avx2 and ARM neon by construction (RFC 0015 §3.1).
pub fn scalar_f64_chain(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a + b - c * d / a
}
"#;

type DotFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
type MatmulFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
/// The scalar-f64 chain: four f64 args in, one f64 result out (System V xmm ABI).
type ScalarF64Fn = unsafe extern "C" fn(f64, f64, f64, f64) -> f64;

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

// --- same-process run-to-run determinism -----------------------------------
// The cross-substrate gates above run each kernel ONCE and pin its output to a
// frozen per-substrate reference. That proves across-build / across-machine /
// across-time byte-identity, but NOT within-process run-to-run determinism: a
// kernel that leaked uninitialised padding into its output, depended on prior
// buffer state, or wrote results in an allocation-order-dependent way could
// still match the frozen reference on the single run that produced it, yet
// differ from run to run. This gate runs the same workload REPEATEDLY in one
// process and asserts every run is byte-identical to the first — turning any
// such non-determinism into a deterministic single-run failure instead of an
// intermittent CI flake. Self-skips without the MLIR toolchain, like the gates
// above.
const DETERMINISM_RUNS: usize = 16;

#[test]
fn same_process_run_to_run_determinism() {
    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };

    // Scalar reduction path (dot-q16): a fixed seeded input, run N times, must
    // hash identically every time.
    {
        let dot: Symbol<DotFn> = unsafe { lib.get(b"dotq").expect("dotq symbol") };
        let (a, b) = make_pair_q16(65536, 0xDEADBEEF);
        let run_once =
            || canonical_hash(unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, a.len() as i64) });
        let first = run_once();
        for run in 1..DETERMINISM_RUNS {
            assert_eq!(
                run_once(),
                first,
                "dot-q16: run {run} diverged from run 0 in the same process \
                 (run-to-run non-determinism)"
            );
        }
    }

    // Buffer-output path (gemv-q16): each run writes into a FRESH zeroed buffer;
    // every run's buffer must hash identically. The fresh buffer per run is the
    // point — it catches a kernel that reads prior buffer state or leaves output
    // bytes unwritten.
    {
        let mmq: Symbol<MatmulFn> = unsafe { lib.get(b"mmq").expect("mmq symbol") };
        let (rows, cols) = (256usize, 256usize);
        let (w, x) = make_gemv_q16(rows, cols, 0xDEADBEEF);
        let run_once = || {
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
            assert_eq!(rc, 0, "gemv-q16: kernel returned {rc} (expected 0)");
            canonical_hash_i32s(&y)
        };
        let first = run_once();
        for run in 1..DETERMINISM_RUNS {
            assert_eq!(
                run_once(),
                first,
                "gemv-q16: run {run} diverged from run 0 in the same process \
                 (run-to-run non-determinism)"
            );
        }
    }

    // Matrix-matrix path (gemm-q16): the headline wedge workload, where any
    // accumulation-order or buffer-state non-determinism is most likely to
    // surface. Fresh M*N output buffer per run; every run must hash identically.
    {
        let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (a, _b, bt) = make_gemm_q16(m, k, n, 0xDEADBEEF);
        let run_once = || {
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
            assert_eq!(rc, 0, "gemm-q16: kernel returned {rc} (expected 0)");
            canonical_hash_i32s(&c)
        };
        let first = run_once();
        for run in 1..DETERMINISM_RUNS {
            assert_eq!(
                run_once(),
                first,
                "gemm-q16: run {run} diverged from run 0 in the same process \
                 (run-to-run non-determinism)"
            );
        }
    }

    // int8 matrix-matrix path (gemm-i8, the det.igemm tier): pure-integer GEMM
    // into a fresh int32 output buffer per run; every run must hash identically.
    {
        let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (a, b) = make_gemm_i8(m, k, n, 0xDEADBEEF);
        let run_once = || {
            let mut c = vec![0i32; m * n];
            let rc = unsafe {
                gemmi8(
                    a.as_ptr() as i64,
                    b.as_ptr() as i64,
                    c.as_mut_ptr() as i64,
                    m as i64,
                    k as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0, "gemm-i8: kernel returned {rc} (expected 0)");
            canonical_hash_i32s(&c)
        };
        let first = run_once();
        for run in 1..DETERMINISM_RUNS {
            assert_eq!(
                run_once(),
                first,
                "gemm-i8: run {run} diverged from run 0 in the same process \
                 (run-to-run non-determinism)"
            );
        }
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

// --- gemm-i8 workload (the "det.igemm" tier) -------------------------------
// A square int8 GEMM C[M,N] = A[M,K] · B[K,N], PURE INTEGER (no fixed-point
// shift). Driven directly by the fused int8 intrinsic
// __mind_blas_matmul_mm_i8_v (A,B int8 1-byte; C int32 4-byte; B is the
// UN-transposed K×N row-major operand). Each output element is the exact int32
// sum (i32) Σ_k (i32)A[i,k]*(i32)B[k,j], accumulated in i64 and truncated once.
// Integer add is associative + commutative, so the result is byte-identical to
// the sequential scalar int32 oracle and the SAME MLIR lowers to vpmaddwd
// (AVX2) / SDOT / SMMLA (aarch64) — both produce the identical int32 sum, so
// avx2 == neon by construction (RFC 0015 §3.1). The output is the M×N int32
// matrix; canonical encoding is its i32 LE bytes → sha256.

/// Regenerate the gemm-i8 inputs from a seed via the SAME LCG, narrowed to
/// int8: an M×K matrix A and a K×N matrix B (both row-major int8), A generated
/// before B (order is part of the seed contract). The sample is the LCG's
/// `next_u32 >> 16` truncated to i8 (full signed-int8 range, deterministic).
fn make_gemm_i8(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i8>, Vec<i8>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i8> = (0..m * k).map(|_| (g.next_u32() >> 16) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|_| (g.next_u32() >> 16) as i8).collect();
    (a, b)
}

/// Independent scalar int32 oracle: C[i,j] = (i32) Σ_k (i32)A[i,k]*(i32)B[k,j],
/// sequential accumulation in i64 then truncate. B is the un-transposed K×N
/// row-major operand (B[k*n+j]), matching the kernel's ABI.
fn ref_gemm_i8_scalar(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i64) * (b[kk * n + j] as i64);
            }
            c[i * n + j] = acc as i32;
        }
    }
    c
}

#[test]
fn gemm_i8_reproducibility_gate() {
    let id = "gemm-i8-64x64x64";
    let (m, k, n, seed) = (64usize, 64usize, 64usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };

    let (a, b) = make_gemm_i8(m, k, n, seed);
    let mut c = vec![0i32; m * n];
    let rc = unsafe {
        gemmi8(
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            c.as_mut_ptr() as i64,
            m as i64,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: kernel returned {rc} (expected 0)");

    // 1. Within-run exactness vs the scalar int32 GEMM oracle.
    let oracle = ref_gemm_i8_scalar(&a, &b, m, k, n);
    assert_eq!(
        c, oracle,
        "{id}: int8 gemm vector path diverged from the scalar int32 oracle"
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

// --- gemv-i16 workload (int16 matrix x vector) -----------------------------
// The "int-dot" tier sibling of gemv-q16: y = W . x over int16 inputs, where W
// is a 256x256 int16 matrix and x a 256-vector, via __mind_blas_matmul_rmajor_i16_v.
// Each output element is an exact integer reduction (sext i16->i64, multiply,
// i64-lane accumulate, narrow once to i32 — NO Q16 shift), so the vectorised
// reduction is bit-identical across substrates by construction (RFC 0015 §3.1).
// The output is the rows-length i32 result vector; canonical encoding is its
// i32 LE bytes → sha256 (same as the Q16.16 vector path).

/// Regenerate the gemv-i16 inputs from a seed: a rows*cols int16 matrix W
/// (row-major) and a cols-length int16 vector x, W generated before x. `next_i16`
/// takes the full int16 range from the shared LCG window.
fn make_gemv_i16(rows: usize, cols: usize, seed: u64) -> (Vec<i16>, Vec<i16>) {
    let mut g = Lcg::new(seed);
    let next_i16 = |g: &mut Lcg| (g.next_u32() >> 16) as i16;
    let w: Vec<i16> = (0..rows * cols).map(|_| next_i16(&mut g)).collect();
    let x: Vec<i16> = (0..cols).map(|_| next_i16(&mut g)).collect();
    (w, x)
}

/// Scalar int16 dot oracle, byte-for-byte the per-row reduction the kernel
/// performs: sext each i16 to i64, multiply-accumulate exactly, narrow once to
/// i32 (raw integer dot — NO Q16 shift).
fn ref_dot_i16_scalar(w: &[i16], x: &[i16]) -> i32 {
    let mut acc: i64 = 0;
    for i in 0..w.len() {
        acc += (w[i] as i64) * (x[i] as i64);
    }
    acc as i32
}

/// Scalar int16 gemv oracle: y[r] = dot_i16(W row r, x).
fn ref_gemv_i16_scalar(w: &[i16], x: &[i16], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows)
        .map(|r| ref_dot_i16_scalar(&w[r * cols..(r + 1) * cols], x))
        .collect()
}

#[test]
fn gemv_i16_reproducibility_gate() {
    let id = "gemv-i16-256x256";
    let (rows, cols, seed) = (256usize, 256usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let mmi16: Symbol<MatmulFn> = unsafe { lib.get(b"mmi16").expect("mmi16 symbol") };

    let (w, x) = make_gemv_i16(rows, cols, seed);
    let mut y = vec![0i32; rows];
    let rc = unsafe {
        mmi16(
            w.as_ptr() as i64,
            x.as_ptr() as i64,
            y.as_mut_ptr() as i64,
            rows as i64,
            cols as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: kernel returned {rc} (expected 0)");

    // 1. Within-run exactness vs the scalar gemv oracle.
    let oracle = ref_gemv_i16_scalar(&w, &x, rows, cols);
    assert_eq!(
        y, oracle,
        "{id}: gemv-i16 vector path diverged from the scalar oracle"
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

// --- scalar-f64 workload (deterministic scalar IEEE-754 float) --------------
// The first NON-INTEGER cross-substrate canary: a fixed scalar `+ − × ÷` chain
// over f64, proving MIND's byte-identity wedge extends from Q16.16 / pure-int to
// strict IEEE-754 scalar float. The kernel `scalar_f64_chain(a,b,c,d)` computes
// `a + b - c * d / a`, lowered to `arith.addf/subf/mulf/divf` on f64 with NO FMA
// fusion and NO fastmath/reassoc flags (verified vaddsd/vmulsd/vdivsd/vsubsd).
//
// Why avx2 == neon here, UNLIKE a float reduction: scalar IEEE `+ − × ÷` are
// individually round-to-nearest-even with a single, fully-specified result —
// there is no accumulation order, no contraction, no reassociation to differ
// across substrates. The source precedence pins one fixed op order, so both x86
// and ARM evaluate the identical IEEE operations on the identical bits and
// produce the identical result (RFC 0015 §3.1). This is scoped to scalar
// elementwise `+ − × ÷` ONLY — float REDUCTIONS remain order-sensitive and are
// deliberately out of this canary's scope.
//
// The inputs are four exact-representable f64 constants supplied by the harness
// (the manifest `[input]` documents them); they are chosen so the division
// `c * d / a = 0.5 * 3.125 / 1.5` is a non-terminating binary fraction, making
// the result `2.708333333333333` (bits 0x4005aaaaaaaaaaaa) — so the canary
// proves the ROUNDING of `÷` is byte-identical, not merely that exact arithmetic
// agrees. The canonical encoding is the result's IEEE-754 bit pattern as an i64,
// then 8 little-endian bytes → sha256 (the same `canonical_hash` the scalar
// dot-q16 gate uses, applied to `f64::to_bits`).

/// The four deterministic scalar-f64 inputs (manifest `[input]`). Exact-
/// representable f64; `c*d/a` is intentionally a non-terminating binary fraction.
const SCALAR_F64_INPUTS: (f64, f64, f64, f64) = (1.5, 2.25, 0.5, 3.125);

/// Independent in-process oracle: the identical IEEE chain evaluated in Rust f64.
/// Because scalar `+ − × ÷` are strict IEEE with the same fixed op order, this is
/// bit-exact to the kernel within a run (the cross-check), and — being IEEE — the
/// same on every substrate (the cross-substrate claim).
fn ref_scalar_f64_chain(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a + b - c * d / a
}

#[test]
fn scalar_float_f64_reproducibility_gate() {
    let id = "scalar-float-f64";

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let chain: Symbol<ScalarF64Fn> = unsafe {
        lib.get(b"scalar_f64_chain")
            .expect("scalar_f64_chain symbol")
    };

    let (a, b, c, d) = SCALAR_F64_INPUTS;
    let result = unsafe { chain(a, b, c, d) };

    // 1. Within-run exactness vs the IEEE oracle: the kernel's strict-IEEE chain
    //    must reproduce the identical bit pattern as the Rust f64 chain.
    let oracle = ref_scalar_f64_chain(a, b, c, d);
    assert_eq!(
        result.to_bits(),
        oracle.to_bits(),
        "{id}: scalar-f64 kernel diverged from the IEEE oracle within a single run \
         (kernel={result} bits={:#018x}, oracle={oracle} bits={:#018x})",
        result.to_bits(),
        oracle.to_bits()
    );

    // 2. Canonical hash of the result's IEEE-754 bit pattern (as i64 LE bytes),
    //    pinned to the committed per-substrate reference. avx2 == neon by IEEE
    //    construction (RFC 0015 §3.1).
    let computed = canonical_hash(result.to_bits() as i64);
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed,
            expected,
            "{id} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result_bits={:#018x}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13).",
            result.to_bits()
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed} \
             (result_bits={:#018x}); bless with MIND_BENCH_BLESS=1 if this host is canonical.",
            result.to_bits()
        ),
    }
}
