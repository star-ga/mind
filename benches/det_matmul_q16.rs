// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Part of the MIND project (Machine Intelligence Native Design).

//! Deterministic Q16.16 GEMM — execution-throughput benchmark with an
//! embedded **cross-substrate byte-identity** assertion (the CPU realisation
//! of the "CUDA-MMM" story).
//!
//! ## Why this is the wedge, not a speed brag
//!
//! `C[M,N] = A[M,K] · B[K,N]` in Q16.16 fixed-point is integer
//! multiply-accumulate. Integer add is **associative**, so the compiler /
//! hardware is free to reorder and vectorise the reduction (the `vector`-dialect
//! widen-multiply-arithmetic-shift-accumulate loop mindc lowers
//! `__mind_blas_matmul_rmajor_q16_v` into) **without changing a single output
//! byte**. That gives us *both* axes at once:
//!
//!   1. **Throughput** — a vectorised reduction, timed here at 16×16, 64×64 and
//!      128×128 square shapes (criterion, microsecond-scale).
//!   2. **Byte-identity** — the exact output bytes are pinned to the committed
//!      per-substrate reference hash in
//!      `tests/cross_substrate_identity/gemm-q16-64x64x64/reference_hashes.toml`
//!      (RFC 0020 §5). avx2 and neon carry **one identical hash** — the
//!      cross-substrate bit-identity claim made inspectable (RFC 0015 §3.1).
//!
//! Float GEMM (cuBLAS / any BLAS) cannot make claim 2: f32 add is
//! non-associative, so reordering the reduction changes the result. Reordering
//! is exactly what makes GEMM fast. MIND keeps the reordering *and* the
//! byte-identity because the accumulation is integer.
//!
//! ## Additive, self-skipping, reuses the existing gate
//!
//! This bench adds **nothing** to `src/`. It drives the same kernel and the
//! same reference the `cross_substrate_identity` test gate uses — the LCG, the
//! seed, the deterministic Bᵀ transpose, the `i32_le → sha256` canonical
//! encoding, and the committed hash are all byte-for-byte what
//! `tests/cross_substrate_identity.rs::gemm_q16_reproducibility_gate` pins. The
//! assertion is therefore *wired to* the merge gate, not reinvented.
//!
//! Like the gated test harnesses, it self-skips when the MLIR toolchain
//! (`mlir-opt` / `mlir-translate` / `clang`) is not on PATH, because it must
//! shell out to `mindc --emit-shared`. The whole file compiles unconditionally
//! (cargo always builds bench targets); the skip is a runtime check, and the
//! byte-identity assertion only runs when the kernel actually built.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench det_matmul_q16 --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// Square shapes the throughput sweep exercises. The 64×64 entry is the
/// byte-identity anchor — its hash is pinned to the committed reference.
const SHAPES: &[usize] = &[16, 64, 128];

/// The committed byte-identity workload (RFC 0020 §5). The 64×64×64 row of the
/// sweep regenerates exactly this input and must hash to this reference.
const ANCHOR_ID: &str = "gemm-q16-64x64x64";
const ANCHOR_N: usize = 64;
const ANCHOR_SEED: u64 = 0xDEAD_BEEF;

/// Kernel ABI: `gemmq(a, bt, c, m, k, n) -> 0`. Computes `C[M,N] = A[M,K]·B[K,N]`
/// in Q16.16 by composing the `__mind_blas_matmul_rmajor_q16_v` intrinsic over
/// rows against Bᵀ — byte-for-byte the source in `cross_substrate_identity.rs`.
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

/// `gemmq(a, bt, c, m, k, n)` plus the row-loop helper it recurses through.
/// Identical to the `cross_substrate_identity.rs` SRC so the `.so` this bench
/// builds is the same artifact the merge gate verifies.
const SRC: &str = r#"
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

/// The host substrate id, per RFC 0014 tier naming — the reference-hash lookup
/// key. Matches `cross_substrate_identity.rs::host_substrate`.
fn host_substrate() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "avx2"
    } else if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "unknown"
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn mindc_path() -> Option<PathBuf> {
    let dbg = manifest_dir().join("target").join("debug").join("mindc");
    if dbg.exists() {
        return Some(dbg);
    }
    let rel = manifest_dir().join("target").join("release").join("mindc");
    if rel.exists() { Some(rel) } else { None }
}

/// Compile the GEMM kernel to a temp `.so` once. Returns `None` (self-skip) if
/// the MLIR toolchain is shadowed or `mindc` is not built — same contract as the
/// gated test harnesses.
fn build_gemm_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("det_matmul_q16: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "det_matmul_q16: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_det_matmul_q16.mind");
        let so_path = dir.join("mind_bench_det_matmul_q16.so");
        if std::fs::write(&src_path, SRC).is_err() {
            eprintln!("det_matmul_q16: could not write workload source; skipping");
            return None;
        }
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => Some(so_path),
            _ => {
                eprintln!("det_matmul_q16: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Deterministic LCG — byte-for-byte the generator the cross-substrate gate
/// uses, so the workload input distribution is the shared, reproducible one.
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

/// Regenerate the seeded GEMM inputs: an M×K matrix A and a K×N matrix B, both
/// row-major Q16.16, A generated before B (order is part of the seed contract).
/// Returns `(A, Bᵀ)` where Bᵀ (N×K, row-major) is the exact transpose the kernel
/// consumes. Byte-for-byte `cross_substrate_identity.rs::make_gemm_q16`.
fn make_gemm_q16(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
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
    (a, bt)
}

/// Track A scalar oracle (byte-for-byte `mind_blas_dot_q16_scalar`): the
/// independent reference the vector reduction must match exactly within a run.
fn ref_dot_q16_scalar(a: &[i32], b: &[i32]) -> i64 {
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        acc += ((a[i] as i64) * (b[i] as i64)) >> 16;
    }
    (acc as i32) as i64
}

/// Scalar GEMM oracle over Bᵀ — M·N scalar dot products, byte-for-byte the
/// accumulation the kernel performs via the gemv intrinsic.
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

/// Canonical output hash of a Q16.16 matrix: each i32 little-endian, then
/// sha256 → lowercase hex (manifest `output_encoding = "i32_le"`). Byte-for-byte
/// `cross_substrate_identity.rs::canonical_hash_i32s`.
fn canonical_hash_i32s(v: &[i32]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update(e.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Read the committed reference hash for `(id, substrate)` from the workload's
/// `reference_hashes.toml` — the same minimal parse the gate uses.
fn reference_hash(id: &str, substrate: &str) -> Option<String> {
    let path = manifest_dir()
        .join("tests")
        .join("cross_substrate_identity")
        .join(id)
        .join("reference_hashes.toml");
    let text = std::fs::read_to_string(&path).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((kk, v)) = line.split_once('=') {
            if kk.trim() == substrate {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Run the GEMM once on host, returning the M×N output matrix.
fn run_gemm(lib: &Library, a: &[i32], bt: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };
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
    assert_eq!(rc, 0, "gemmq kernel returned {rc} (expected 0)");
    c
}

/// THE byte-identity assertion, wired to the existing cross-substrate gate.
///
/// 1. avx2 and neon in the committed `reference_hashes.toml` carry one identical
///    hash (RFC 0015 §3.1 — the cross-substrate claim is that they are equal).
/// 2. The kernel output, canonically encoded, equals the committed reference for
///    this host's substrate (across-build / across-machine / across-time
///    byte-identity — RFC 0020 §5).
/// 3. Within-run exactness vs the independent scalar oracle (sanity belt).
///
/// Panics (fails the bench run) on any mismatch — so a lowering regression that
/// changed the bytes could never be reported as a clean throughput number.
fn assert_byte_identity(lib: &Library) {
    let n = ANCHOR_N;
    let (a, bt) = make_gemm_q16(n, n, n, ANCHOR_SEED);
    let c = run_gemm(lib, &a, &bt, n, n, n);

    // (3) within-run exactness vs the scalar oracle.
    let oracle = ref_gemm_q16_scalar(&a, &bt, n, n, n);
    assert_eq!(
        c, oracle,
        "{ANCHOR_ID}: vector GEMM diverged from the scalar oracle within a single run"
    );

    let computed = canonical_hash_i32s(&c);

    // (1) cross-substrate equality made inspectable: every substrate line in the
    // committed manifest must carry this one hash.
    let avx2 = reference_hash(ANCHOR_ID, "avx2")
        .unwrap_or_else(|| panic!("{ANCHOR_ID}: missing avx2 reference hash"));
    let neon = reference_hash(ANCHOR_ID, "neon")
        .unwrap_or_else(|| panic!("{ANCHOR_ID}: missing neon reference hash"));
    assert_eq!(
        avx2, neon,
        "{ANCHOR_ID}: avx2 and neon reference hashes differ — cross-substrate \
         bit-identity claim is broken in reference_hashes.toml"
    );

    // (2) this host's output is byte-identical to the committed reference.
    let substrate = host_substrate();
    let expected = reference_hash(ANCHOR_ID, substrate)
        .unwrap_or_else(|| panic!("{ANCHOR_ID}: no reference hash for substrate '{substrate}'"));
    assert_eq!(
        computed, expected,
        "{ANCHOR_ID} [{substrate}]: GEMM output hash drifted from the committed reference.\n\
         computed={computed}\n expected={expected}\n\
         If this is an intentional lowering change (RFC 0020 §13), re-bless the \
         cross_substrate_identity gate with MIND_BENCH_BLESS=1 and commit the new hash."
    );

    eprintln!(
        "det_matmul_q16: byte-identity VERIFIED [{substrate}] {ANCHOR_ID} sha256={computed} \
         (avx2==neon committed)"
    );
}

fn bench_det_matmul_q16(c: &mut Criterion) {
    let Some(so) = build_gemm_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("det_matmul_q16: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen GEMM .so") };

    // Correctness gate first: a throughput number for output that drifted from
    // the committed bytes would be a lie. This panics on any mismatch.
    assert_byte_identity(&lib);

    let mut group = c.benchmark_group("det_matmul_q16");
    for &n in SHAPES {
        // One MAC per inner-product term: M·N·K multiply-accumulates.
        let macs = (n as u64) * (n as u64) * (n as u64);
        group.throughput(Throughput::Elements(macs));

        // Seed off the shape so each size has its own reproducible input; the
        // 64×64 entry uses the anchor seed so its bytes match the committed
        // reference exactly (already verified above).
        let seed = if n == ANCHOR_N {
            ANCHOR_SEED
        } else {
            0xDEAD_BEEF_0000_0000 ^ (n as u64)
        };
        let (a, bt) = make_gemm_q16(n, n, n, seed);
        let mut out = vec![0i32; n * n];
        let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };

        group.bench_with_input(
            BenchmarkId::new("q16_square", format!("{n}x{n}x{n}")),
            &n,
            |bencher, &nn| {
                bencher.iter(|| {
                    let rc = unsafe {
                        gemmq(
                            black_box(a.as_ptr() as i64),
                            black_box(bt.as_ptr() as i64),
                            black_box(out.as_mut_ptr() as i64),
                            black_box(nn as i64),
                            black_box(nn as i64),
                            black_box(nn as i64),
                        )
                    };
                    black_box(rc);
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = det_matmul_q16;
    // Real measurement time, not a 5s quick run: warm up + ample sampling so the
    // microsecond-scale GEMMs land with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_det_matmul_q16
}
criterion_main!(det_matmul_q16);
