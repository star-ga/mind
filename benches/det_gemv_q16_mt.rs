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

//! Deterministic **multithreaded** Q16.16 GEMV — the routing SCORE kernel
//! (`scores[M] = catalog[M×K] · query[K]`), with a **byte-identity** assertion
//! and a throughput axis at the real routing shape.
//!
//! ## The claim this bench pins
//!
//! `__mind_blas_gemv_q16_mt` reuses the same owner-computes M-row band wrapper as
//! `__mind_blas_matmul_mm_q16_mt_v` (N pinned to 1), but each band runs the
//! K-vectorised gemv kernel: it accumulates a `vector<8xi64>` of per-product
//! `(catalog[i,k]*query[k]) >> 16` shifted terms, then horizontally reduces (plus
//! a scalar K%8 tail). Because the accumulator is **exact i64 integer**, the lane
//! regrouping + horizontal reduction sum to the identical value as the strictly
//! sequential scalar oracle — so the scores are **byte-for-byte identical** to
//! `Σ_k (catalog[i,k]*query[k])>>16`, and — because every score is written by
//! exactly one thread (no cross-thread reduction / atomics) — **independent of
//! the thread count**.
//!
//! So this bench asserts and panics on any miss:
//!   1. Within-run exactness vs the independent scalar dot oracle (byte-identity).
//!   2. Concurrency stability: the anchor is recomputed `STABILITY_RUNS` times and
//!      every run produces the identical hash (a data race would make it flaky).
//! It then reports GMAC/s at the real mind-nerve routing shape (M=11922, K=384).
//!
//! Self-skips when the MLIR toolchain is shadowed or `mindc` is unbuilt, exactly
//! like `det_matmul_q16_mt`.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench det_gemv_q16_mt --no-default-features
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// The real mind-nerve routing catalog shape: M rows × K dims, one query of K.
const ROUTE_M: usize = 11922;
const ROUTE_K: usize = 384;

/// Smaller shapes for the byte-identity anchor + K-tail coverage (K not a
/// multiple of 8 exercises the scalar tail path alongside the vector prefix).
const SHAPES: &[(usize, usize)] = &[(64, 64), (128, 96), (256, 130), (ROUTE_M, ROUTE_K)];

/// Anchor for the exactness + concurrency-stability assertion (a K%8≠0 shape so
/// both the vector prefix and the scalar tail are covered).
const ANCHOR_M: usize = 512;
const ANCHOR_K: usize = 130;
const ANCHOR_SEED: u64 = 0xDEAD_BEEF;

/// Repeated recomputations to flush out any data race in the parallel partition.
const STABILITY_RUNS: usize = 256;

/// Conservative single-core integer-MAC ceiling (GMAC/s), matching the GEMM
/// bench's constant; the all-core roofline denominator is `cores × this`.
fn isa_peak_gmacs_per_core() -> f64 {
    if cfg!(target_arch = "x86_64") {
        56.0
    } else if cfg!(target_arch = "aarch64") {
        24.0
    } else {
        f64::NAN
    }
}

/// Kernel ABI: `gemvq(catalog, query, scores, m, k) -> 0`.
/// `scores[M] = catalog[M×K] · query[K]` in Q16.16 via the MT fused intrinsic.
type GemvFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

/// Thin wrapper over the multithreaded fused Q16.16 GEMV intrinsic.
const SRC: &str = r#"
pub fn gemvq(catalog: i64, query: i64, scores: i64, m: i64, k: i64) -> i64 {
    __mind_blas_gemv_q16_mt(catalog, query, scores, m, k)
}
"#;

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

/// Compile the MT GEMV kernel to a temp `.so` once. `None` (self-skip) if the
/// MLIR toolchain is shadowed or `mindc` is not built.
fn build_gemv_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("det_gemv_q16_mt: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "det_gemv_q16_mt: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_det_gemv_q16_mt.mind");
        let so_path = dir.join("mind_bench_det_gemv_q16_mt.so");
        if std::fs::write(&src_path, SRC).is_err() {
            eprintln!("det_gemv_q16_mt: could not write workload source; skipping");
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
                eprintln!("det_gemv_q16_mt: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Deterministic LCG — byte-for-byte the cross-substrate gate's generator.
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

/// Seeded inputs: M×K catalog then K query, catalog drawn before query.
fn make_gemv_q16(m: usize, k: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let catalog: Vec<i32> = (0..m * k).map(|_| g.next_q16()).collect();
    let query: Vec<i32> = (0..k).map(|_| g.next_q16()).collect();
    (catalog, query)
}

/// Scalar Q16.16 oracle: `scores[i] = trunc_i32( Σ_k (catalog[i,k]*query[k]) >> 16 )`.
fn ref_gemv_q16_scalar(catalog: &[i32], query: &[i32], m: usize, k: usize) -> Vec<i32> {
    let mut s = vec![0i32; m];
    for i in 0..m {
        let mut acc: i64 = 0;
        for kk in 0..k {
            acc += ((catalog[i * k + kk] as i64) * (query[kk] as i64)) >> 16;
        }
        s[i] = acc as i32;
    }
    s
}

/// Canonical i32-little-endian → sha256 hex.
fn canonical_hash_i32s(v: &[i32]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update(e.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Run the MT GEMV once, returning the M scores.
fn run_gemv(lib: &Library, catalog: &[i32], query: &[i32], m: usize, k: usize) -> Vec<i32> {
    let gemvq: Symbol<GemvFn> = unsafe { lib.get(b"gemvq").expect("gemvq symbol") };
    let mut s = vec![0i32; m];
    let rc = unsafe {
        gemvq(
            catalog.as_ptr() as i64,
            query.as_ptr() as i64,
            s.as_mut_ptr() as i64,
            m as i64,
            k as i64,
        )
    };
    assert_eq!(rc, 0, "gemvq kernel returned {rc} (expected 0)");
    s
}

/// (1) exactness vs the scalar oracle (byte-identity of the K-vectorised kernel);
/// (2) concurrency stability across `STABILITY_RUNS` repeats. Panics on any miss.
fn assert_byte_identity(lib: &Library) {
    let (m, k) = (ANCHOR_M, ANCHOR_K);
    let (catalog, query) = make_gemv_q16(m, k, ANCHOR_SEED);
    let s = run_gemv(lib, &catalog, &query, m, k);

    // (1) within-run exactness vs the scalar oracle.
    let oracle = ref_gemv_q16_scalar(&catalog, &query, m, k);
    assert_eq!(
        s, oracle,
        "gemv-q16-{m}x{k}: multithreaded GEMV diverged from the scalar dot oracle"
    );
    let computed = canonical_hash_i32s(&s);

    // (2) concurrency stability: a data race would make the hash flaky.
    for run in 0..STABILITY_RUNS {
        let si = run_gemv(lib, &catalog, &query, m, k);
        let hi = canonical_hash_i32s(&si);
        assert_eq!(
            hi, computed,
            "gemv-q16-{m}x{k}: output changed across runs (run {run}) — a data race \
             in the parallel partition. expected={computed} got={hi}"
        );
    }

    let substrate = host_substrate();
    eprintln!(
        "det_gemv_q16_mt: byte-identity VERIFIED [{substrate}] gemv-q16-{m}x{k} sha256={computed} \
         (== scalar dot oracle; stable across {STABILITY_RUNS} concurrent runs)"
    );
}

/// GMAC/s + `%-of-all-core-ISA-peak` for the GEMV (`MACs = M·K`).
fn report_gmacs_gemv(lib: &Library, m: usize, k: usize, seed: u64) {
    const WARMUP: usize = 4;
    const REPS: usize = 32;
    let (catalog, query) = make_gemv_q16(m, k, seed);
    let mut out = vec![0i32; m];
    let gemvq: Symbol<GemvFn> = unsafe { lib.get(b"gemvq").expect("gemvq symbol") };
    let call = |catalog: &[i32], query: &[i32], out: &mut [i32]| {
        let rc = unsafe {
            gemvq(
                catalog.as_ptr() as i64,
                query.as_ptr() as i64,
                out.as_mut_ptr() as i64,
                m as i64,
                k as i64,
            )
        };
        assert_eq!(rc, 0, "gemvq returned {rc}");
    };
    for _ in 0..WARMUP {
        call(&catalog, &query, &mut out);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        call(black_box(&catalog), black_box(&query), black_box(&mut out));
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = samples[REPS / 2];
    let macs = (m as f64) * (k as f64);
    let gmacs = macs / median / 1e9;
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1) as f64;
    let per_core = isa_peak_gmacs_per_core();
    let pct = if per_core.is_finite() {
        let peak = per_core * cores;
        format!(
            "{:.1}% of all-core ISA peak (~{peak:.0} GMAC/s est., {cores:.0}c)",
            gmacs / peak * 100.0
        )
    } else {
        "ISA peak unknown".to_string()
    };
    eprintln!(
        "det_gemv_q16_mt: ROOFLINE {m}x{k} {gmacs:7.2} GMAC/s  [{pct}]  (median {:.1} µs/call)",
        median * 1e6
    );
}

fn bench_det_gemv_q16_mt(c: &mut Criterion) {
    let Some(so) = build_gemv_so() else {
        eprintln!("det_gemv_q16_mt: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen MT GEMV .so") };

    // Correctness + race gate first.
    assert_byte_identity(&lib);

    let mut group = c.benchmark_group("det_gemv_q16_mt");
    for &(m, k) in SHAPES {
        let macs = (m as u64) * (k as u64);
        group.throughput(Throughput::Elements(macs));
        let seed = 0xDEAD_BEEF_0000_0000 ^ ((m as u64) << 20 ^ k as u64);
        report_gmacs_gemv(&lib, m, k, seed);

        let (catalog, query) = make_gemv_q16(m, k, seed);
        let mut out = vec![0i32; m];
        let gemvq: Symbol<GemvFn> = unsafe { lib.get(b"gemvq").expect("gemvq symbol") };
        group.bench_with_input(
            BenchmarkId::new("q16_mt_gemv", format!("{m}x{k}")),
            &(m, k),
            |bencher, &(mm, kk)| {
                bencher.iter(|| {
                    let rc = unsafe {
                        gemvq(
                            black_box(catalog.as_ptr() as i64),
                            black_box(query.as_ptr() as i64),
                            black_box(out.as_mut_ptr() as i64),
                            black_box(mm as i64),
                            black_box(kk as i64),
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
    name = det_gemv_q16_mt;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_det_gemv_q16_mt
}
criterion_main!(det_gemv_q16_mt);
