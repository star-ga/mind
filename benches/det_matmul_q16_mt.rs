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

//! Deterministic **multithreaded** Q16.16 GEMM — throughput + a
//! **byte-identity-under-concurrency** assertion (the deterministic-parallel
//! compiler made inspectable).
//!
//! ## The claim this bench pins
//!
//! `__mind_blas_matmul_mm_q16_mt_v` partitions the M output rows into contiguous
//! bands, one per worker thread (emitted `pthread_create`/`pthread_join`, static
//! owner-computes schedule baked into the artifact). Because **every output
//! element is written by exactly one thread** — no cross-thread reduction, no
//! atomics, no shared accumulator — the result is **byte-for-byte identical to
//! the single-thread kernel regardless of thread count**, and the integer MAC
//! makes it identical across substrates too.
//!
//! So this bench asserts three things and panics on any miss:
//!   1. The 64×64×64 anchor output hashes to the **same committed reference**
//!      (`92e2cb75…`, `tests/cross_substrate_identity/gemm-q16-64x64x64`) that the
//!      single-thread gate pins — the multithreaded kernel did not change one byte.
//!   2. **Concurrency stability:** the anchor is recomputed `STABILITY_RUNS` times
//!      and every run produces the identical hash. A data race would make this
//!      flaky; a clean owner-computes partition makes it deterministic.
//!   3. Within-run exactness vs the independent scalar oracle.
//!
//! It then sweeps 64/128/256/512 square shapes so the multi-core scaling is
//! visible (the larger shapes amortise the spawn/join overhead). The kernel runs
//! across all cores; this bench is **not** core-pinned.
//!
//! Self-skips when the MLIR toolchain is shadowed or `mindc` is unbuilt, exactly
//! like `det_matmul_q16`.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench det_matmul_q16_mt --no-default-features
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// Square shapes the throughput sweep exercises. 64×64 is the byte-identity
/// anchor; the larger shapes show multi-core scaling past the spawn/join cost.
const SHAPES: &[usize] = &[64, 128, 256, 512];

/// The committed byte-identity workload — the multithreaded kernel must hit the
/// SAME reference hash the single-thread gate pins (RFC 0020 §5).
const ANCHOR_ID: &str = "gemm-q16-64x64x64";
const ANCHOR_N: usize = 64;
const ANCHOR_SEED: u64 = 0xDEAD_BEEF;

/// Repeated anchor recomputations to flush out any data race in the parallel
/// partition: every run must produce the identical hash.
const STABILITY_RUNS: usize = 256;

/// Kernel ABI: `gemmq(a, b, c, m, k, n) -> 0`. `C[M,N]=A[M,K]·B[K,N]` in Q16.16
/// via the multithreaded fused intrinsic, B un-transposed (K×N row-major).
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

/// Thin wrapper over the multithreaded fused outer-product GEMM intrinsic.
const SRC: &str = r#"
pub fn gemmq(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_q16_mt_v(a, b, c, m, k, n)
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

/// Compile the multithreaded GEMM kernel to a temp `.so` once. `None` (self-skip)
/// if the MLIR toolchain is shadowed or `mindc` is not built.
fn build_gemm_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("det_matmul_q16_mt: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "det_matmul_q16_mt: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_det_matmul_q16_mt.mind");
        let so_path = dir.join("mind_bench_det_matmul_q16_mt.so");
        if std::fs::write(&src_path, SRC).is_err() {
            eprintln!("det_matmul_q16_mt: could not write workload source; skipping");
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
                eprintln!("det_matmul_q16_mt: mindc --emit-shared failed; skipping");
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

/// Seeded inputs: M×K matrix A then K×N matrix B (un-transposed), A drawn before
/// B — the same LCG order the cross-substrate gate uses.
fn make_gemm_q16(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i32> = (0..m * k).map(|_| g.next_q16()).collect();
    let b: Vec<i32> = (0..k * n).map(|_| g.next_q16()).collect();
    (a, b)
}

/// Scalar Q16.16 oracle: `C[i,j] = trunc_i32( Σ_k (A[i,k]*B[k,j]) >> 16 )`.
fn ref_gemm_q16_scalar(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for kk in 0..k {
                acc += ((a[i * k + kk] as i64) * (b[kk * n + j] as i64)) >> 16;
            }
            c[i * n + j] = acc as i32;
        }
    }
    c
}

/// Canonical i32-little-endian → sha256 hex.
fn canonical_hash_i32s(v: &[i32]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update(e.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Committed reference hash for `(id, substrate)`.
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

/// Run the MT GEMM once, returning the M×N output. B is K×N row-major.
fn run_gemm(lib: &Library, a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };
    let mut c = vec![0i32; m * n];
    let rc = unsafe {
        gemmq(
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            c.as_mut_ptr() as i64,
            m as i64,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc, 0, "gemmq kernel returned {rc} (expected 0)");
    c
}

/// The three-part assertion: (1) anchor output == the committed single-thread
/// reference hash (the parallel kernel changed no bytes); (2) concurrency
/// stability across `STABILITY_RUNS` repeats (no data race); (3) within-run
/// exactness vs the scalar oracle. Panics on any miss.
fn assert_byte_identity(lib: &Library) {
    let n = ANCHOR_N;
    let (a, b) = make_gemm_q16(n, n, n, ANCHOR_SEED);
    let c = run_gemm(lib, &a, &b, n, n, n);

    // (3) within-run exactness vs the scalar oracle.
    let oracle = ref_gemm_q16_scalar(&a, &b, n, n, n);
    assert_eq!(
        c, oracle,
        "{ANCHOR_ID}: multithreaded GEMM diverged from the scalar oracle"
    );

    let computed = canonical_hash_i32s(&c);

    // (1) the multithreaded output is byte-identical to the committed reference
    // (the same hash the single-thread cross-substrate gate pins).
    let avx2 = reference_hash(ANCHOR_ID, "avx2")
        .unwrap_or_else(|| panic!("{ANCHOR_ID}: missing avx2 reference hash"));
    let neon = reference_hash(ANCHOR_ID, "neon")
        .unwrap_or_else(|| panic!("{ANCHOR_ID}: missing neon reference hash"));
    assert_eq!(
        avx2, neon,
        "{ANCHOR_ID}: avx2 and neon reference hashes differ"
    );
    let substrate = host_substrate();
    let expected = reference_hash(ANCHOR_ID, substrate)
        .unwrap_or_else(|| panic!("{ANCHOR_ID}: no reference hash for substrate '{substrate}'"));
    assert_eq!(
        computed, expected,
        "{ANCHOR_ID} [{substrate}]: multithreaded GEMM output hash drifted from the \
         committed single-thread reference.\n computed={computed}\n expected={expected}"
    );

    // (2) concurrency stability: a data race would make the hash flaky.
    for run in 0..STABILITY_RUNS {
        let ci = run_gemm(lib, &a, &b, n, n, n);
        let hi = canonical_hash_i32s(&ci);
        assert_eq!(
            hi, computed,
            "{ANCHOR_ID}: multithreaded output changed across runs (run {run}) — \
             a data race in the parallel partition. expected={computed} got={hi}"
        );
    }

    eprintln!(
        "det_matmul_q16_mt: byte-identity VERIFIED [{substrate}] {ANCHOR_ID} sha256={computed} \
         (== single-thread reference; stable across {STABILITY_RUNS} concurrent runs)"
    );
}

fn bench_det_matmul_q16_mt(c: &mut Criterion) {
    let Some(so) = build_gemm_so() else {
        eprintln!("det_matmul_q16_mt: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen MT GEMM .so") };

    // Correctness + race gate first.
    assert_byte_identity(&lib);

    let mut group = c.benchmark_group("det_matmul_q16_mt");
    for &n in SHAPES {
        let macs = (n as u64) * (n as u64) * (n as u64);
        group.throughput(Throughput::Elements(macs));

        let seed = if n == ANCHOR_N {
            ANCHOR_SEED
        } else {
            0xDEAD_BEEF_0000_0000 ^ (n as u64)
        };
        let (a, b) = make_gemm_q16(n, n, n, seed);
        let mut out = vec![0i32; n * n];
        let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };

        group.bench_with_input(
            BenchmarkId::new("q16_mt_square", format!("{n}x{n}x{n}")),
            &n,
            |bencher, &nn| {
                bencher.iter(|| {
                    let rc = unsafe {
                        gemmq(
                            black_box(a.as_ptr() as i64),
                            black_box(b.as_ptr() as i64),
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
    name = det_matmul_q16_mt;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(100);
    targets = bench_det_matmul_q16_mt
}
criterion_main!(det_matmul_q16_mt);
