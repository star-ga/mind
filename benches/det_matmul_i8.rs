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

//! Deterministic int8 GEMM (the `det.igemm` tier) — execution-throughput
//! benchmark with an embedded **cross-substrate byte-identity** assertion, the
//! pure-integer sibling of `det_matmul_q16`.
//!
//! ## Why this is the wedge, not a speed brag
//!
//! `C[M,N] = A[M,K] · B[K,N]` over int8 operands is **raw** integer
//! multiply-accumulate: each term `(i32)A[i,k]*(i32)B[k,j]` is summed into an
//! i64 accumulator and truncated once to i32 — there is **no** `>> 16`
//! fixed-point shift (that shift is Q16.16-only). Integer add is *associative*
//! and *commutative*, so the compiler / hardware is free to reorder, tile and
//! vectorise the reduction (the fused `__mind_blas_matmul_mm_i8_v` intrinsic —
//! `vpmaddwd` on the AVX2 path, the `vpdpbusd` ZMM path under
//! `MIND_INTDOT=vnni`, `SDOT`/`SMMLA` on aarch64) **without changing a single
//! output byte**. That gives us *both* axes at once:
//!
//!   1. **Throughput** — a vectorised reduction, timed here at 16/64/128/256/512
//!      square shapes (criterion, microsecond-scale).
//!   2. **Byte-identity** — the exact output bytes are pinned to the committed
//!      per-substrate reference hash in
//!      `tests/cross_substrate_identity/gemm-i8-64x64x64/reference_hashes.toml`
//!      (RFC 0020 §5). avx2 and neon carry **one identical hash** — the
//!      cross-substrate bit-identity claim made inspectable (RFC 0015 §3.1).
//!
//! Float GEMM (cuBLAS / any BLAS) cannot make claim 2: f32 add is
//! non-associative, so reordering the reduction changes the result. Reordering
//! is exactly what makes GEMM fast. MIND keeps the reordering *and* the
//! byte-identity because the accumulation is integer.
//!
//! ## Additive, self-skipping, asserts the same committed output hash
//!
//! This bench adds **nothing** to `src/`. It drives the fused
//! `__mind_blas_matmul_mm_i8_v` kernel (A, B int8 1-byte; C int32 4-byte; B is
//! the **un-transposed** K×N row-major operand) exactly as the
//! `cross_substrate_identity` gate's int8 path does, so it regenerates the same
//! seeded input and asserts the *same committed output hash*: the per-element
//! math `C[i,j] = (i32) Σ_k (i32)A[i,k]*(i32)B[k,j]` is identical, the LCG, the
//! seed and the `i32_le → sha256` canonical encoding are byte-for-byte what
//! `tests/cross_substrate_identity.rs::gemm_i8_reproducibility_gate` pins.
//!
//! Like the gated test harnesses, it self-skips when the MLIR toolchain
//! (`mlir-opt` / `mlir-translate` / `clang`) is not on PATH, because it must
//! shell out to `mindc --emit-shared`. The whole file compiles unconditionally
//! (cargo always builds bench targets); the skip is a runtime check, and the
//! byte-identity assertion only runs when the kernel actually built.
//!
//! The int8 dot-product backend is selected by `MIND_INTDOT` exactly as the
//! kernel reads it: unset (default) measures the AVX2 `vpmaddwd` path — the
//! regression-gate target on an AVX2 host — and `MIND_INTDOT=vnni` measures the
//! ZMM `vpdpbusd` path. The env is read by the kernel at compile time; this
//! bench does not set it, so it times whatever the host invocation selected.
//!
//! Run:
//! ```text
//! cargo build --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench det_matmul_i8 --no-default-features
//! ```
//! (mindc must exist at `target/{debug,release}/mindc`; the line above builds
//! the debug binary the bench discovers.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

/// Square shapes the throughput sweep exercises. The 64×64 entry is the
/// byte-identity anchor — its hash is pinned to the committed reference.
const SHAPES: &[usize] = &[16, 64, 128, 256, 512];

/// Documented, conservative single-core integer multiply-accumulate ceiling for
/// the host ISA — the denominator of the roofline `%-of-ISA-peak` axis (see
/// `docs/benchmarking.md` §3). This is an **estimate against a documented
/// constant**, not a vendor-certified hardware peak: it is the reference clock
/// times the int8 multiply-accumulate MACs the fused outer-product microkernel
/// can retire per cycle on this ISA. Reported as a comparability aid only.
///
/// avx2: i7-5930K @ 3.5 GHz. The default int8 path widens to i16 and uses
/// `vpmaddwd` (16×i16 → 8×i32 fused MAC) ≈ 2 fused ops/cyc → ~3.5e9·16·2 ≈ 112
/// GMAC/s. neon: a conservative 16×i8 `SDOT` MAC at 3.0 GHz ≈ 48 GMAC/s.
/// Single-core. (Under `MIND_INTDOT=vnni`, `vpdpbusd` does 64×i8 → 16×i32 per
/// op, a higher ceiling — but the default AVX2 path is the gated target here.)
fn isa_peak_gmacs() -> f64 {
    if cfg!(target_arch = "x86_64") {
        112.0
    } else if cfg!(target_arch = "aarch64") {
        48.0
    } else {
        f64::NAN
    }
}

/// The committed byte-identity workload (RFC 0020 §5). The 64×64×64 row of the
/// sweep regenerates exactly this input and must hash to this reference.
const ANCHOR_ID: &str = "gemm-i8-64x64x64";
const ANCHOR_N: usize = 64;
const ANCHOR_SEED: u64 = 0xDEADBEEF;

/// Kernel ABI: `gemmi8(a, b, c, m, k, n) -> 0`. Computes `C[M,N] = A[M,K]·B[K,N]`
/// as raw int8 integer MAC via the fused `__mind_blas_matmul_mm_i8_v` intrinsic,
/// with B passed **un-transposed** (K×N row-major). A, B int8 (1 byte each);
/// C int32 (4 bytes each).
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

/// `gemmi8(a, b, c, m, k, n)` — a thin wrapper over the fused int8 GEMM
/// intrinsic. B is K×N row-major (un-transposed); the kernel handles the full
/// M×N output (including N / M tails) in one call.
const SRC: &str = r#"
pub fn gemmi8(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_i8_v(a, b, c, m, k, n)
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
                eprintln!("det_matmul_i8: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "det_matmul_i8: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_det_matmul_i8.mind");
        let so_path = dir.join("mind_bench_det_matmul_i8.so");
        if std::fs::write(&src_path, SRC).is_err() {
            eprintln!("det_matmul_i8: could not write workload source; skipping");
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
                eprintln!("det_matmul_i8: mindc --emit-shared failed; skipping");
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
}

/// Regenerate the seeded gemm-i8 inputs via the SAME LCG, narrowed to int8: an
/// M×K matrix A and a K×N matrix B (both row-major int8), A generated before B
/// (order is part of the seed contract — byte-for-byte the LCG draw order of
/// `cross_substrate_identity.rs::make_gemm_i8`). The sample is the LCG's
/// `next_u32 >> 16` truncated to i8 (full signed-int8 range, deterministic).
/// Returns `(A, B)` with B **un-transposed** (K×N row-major) — the layout the
/// fused int8 intrinsic consumes directly.
fn make_gemm_i8(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i8>, Vec<i8>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i8> = (0..m * k).map(|_| (g.next_u32() >> 16) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|_| (g.next_u32() >> 16) as i8).collect();
    (a, b)
}

/// Independent scalar int32 GEMM oracle over B (K×N row-major):
/// `C[i,j] = (i32) Σ_k (i32)A[i,k]*(i32)B[k,j]`, sequential accumulation in i64
/// then truncate once — byte-for-byte the per-element accumulation the fused
/// kernel performs (PURE INTEGER, no `>> 16` shift). The independent reference
/// the vector kernel must match exactly within a run.
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

/// Canonical output hash of an int32 matrix: each i32 little-endian, then
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

/// Run the int8 GEMM once on host, returning the M×N int32 output matrix. B is
/// K×N row-major (un-transposed), consumed directly by the fused intrinsic.
fn run_gemm(lib: &Library, a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };
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
    assert_eq!(rc, 0, "gemmi8 kernel returned {rc} (expected 0)");
    c
}

/// THE byte-identity assertion, wired to the existing cross-substrate gate.
///
/// 1. avx2 and neon in the committed `reference_hashes.toml` carry one identical
///    hash (RFC 0015 §3.1 — the cross-substrate claim is that they are equal).
/// 2. The kernel output, canonically encoded, equals the committed reference for
///    this host's substrate (across-build / across-machine / across-time
///    byte-identity — RFC 0020 §5).
/// 3. Within-run exactness vs the independent scalar int32 oracle (sanity belt).
///
/// Panics (fails the bench run) on any mismatch — so a lowering regression that
/// changed the bytes could never be reported as a clean throughput number.
fn assert_byte_identity(lib: &Library) {
    let n = ANCHOR_N;
    let (a, b) = make_gemm_i8(n, n, n, ANCHOR_SEED);
    let c = run_gemm(lib, &a, &b, n, n, n);

    // (3) within-run exactness vs the scalar int32 oracle.
    let oracle = ref_gemm_i8_scalar(&a, &b, n, n, n);
    assert_eq!(
        c, oracle,
        "{ANCHOR_ID}: int8 vector GEMM diverged from the scalar int32 oracle within a single run"
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
        "{ANCHOR_ID} [{substrate}]: int8 GEMM output hash drifted from the committed reference.\n\
         computed={computed}\n expected={expected}\n\
         If this is an intentional lowering change (RFC 0020 §13), re-bless the \
         cross_substrate_identity gate with MIND_BENCH_BLESS=1 and commit the new hash."
    );

    eprintln!(
        "det_matmul_i8: byte-identity VERIFIED [{substrate}] {ANCHOR_ID} sha256={computed} \
         (avx2==neon committed)"
    );
}

/// Comparable execution-throughput axis (`docs/benchmarking.md` §3): GMAC/s and
/// `%-of-ISA-peak` for one square `n×n×n` shape, computed from first principles.
///
/// `MACs = n³`; `GMAC/s = MACs / median_seconds / 1e9`; roofline `% = GMAC/s /
/// isa_peak`. The timing here is an independent quick measurement (median of
/// `REPS` calls after a warm-up) printed to stderr next to criterion's native
/// `elem/s` — it does NOT replace criterion's statistics, it makes the result
/// comparable in the unit a roofline analysis uses. Self-contained: derived from
/// the workload size, no BLAS, no external reference.
fn report_gmacs_i8(lib: &Library, n: usize, seed: u64) {
    const WARMUP: usize = 8;
    const REPS: usize = 64;
    let (a, b) = make_gemm_i8(n, n, n, seed);
    let mut out = vec![0i32; n * n];
    let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };
    let call = |a: &[i8], b: &[i8], out: &mut [i32]| {
        let rc = unsafe {
            gemmi8(
                a.as_ptr() as i64,
                b.as_ptr() as i64,
                out.as_mut_ptr() as i64,
                n as i64,
                n as i64,
                n as i64,
            )
        };
        assert_eq!(rc, 0, "gemmi8 returned {rc}");
    };
    for _ in 0..WARMUP {
        call(&a, &b, &mut out);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        call(black_box(&a), black_box(&b), black_box(&mut out));
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = samples[REPS / 2];
    let macs = (n as f64) * (n as f64) * (n as f64);
    let gmacs = macs / median / 1e9;
    let peak = isa_peak_gmacs();
    let pct = if peak.is_finite() {
        format!(
            "{:.1}% of ISA peak (~{peak:.0} GMAC/s est.)",
            gmacs / peak * 100.0
        )
    } else {
        "ISA peak unknown".to_string()
    };
    eprintln!(
        "det_matmul_i8: ROOFLINE {n}x{n}x{n} {gmacs:7.2} GMAC/s  [{pct}]  (median {:.2} µs/call)",
        median * 1e6
    );
}

fn bench_det_matmul_i8(c: &mut Criterion) {
    let Some(so) = build_gemm_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("det_matmul_i8: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen GEMM .so") };

    // Correctness gate first: a throughput number for output that drifted from
    // the committed bytes would be a lie. This panics on any mismatch.
    assert_byte_identity(&lib);

    let mut group = c.benchmark_group("det_matmul_i8");
    for &n in SHAPES {
        // One MAC per inner-product term: M·N·K multiply-accumulates.
        let macs = (n as u64) * (n as u64) * (n as u64);
        group.throughput(Throughput::Elements(macs));

        // Comparable roofline axis (GMAC/s + %-of-ISA-peak), printed alongside
        // criterion's native elem/s. Uses the same per-shape seed as the timed
        // group below so the measured input is identical.
        let report_seed = if n == ANCHOR_N {
            ANCHOR_SEED
        } else {
            0xDEAD_BEEF_0000_0000 ^ (n as u64)
        };
        report_gmacs_i8(&lib, n, report_seed);

        // Seed off the shape so each size has its own reproducible input; the
        // 64×64 entry uses the anchor seed so its bytes match the committed
        // reference exactly (already verified above).
        let seed = if n == ANCHOR_N {
            ANCHOR_SEED
        } else {
            0xDEAD_BEEF_0000_0000 ^ (n as u64)
        };
        let (a, b) = make_gemm_i8(n, n, n, seed);
        let mut out = vec![0i32; n * n];
        let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };

        group.bench_with_input(
            BenchmarkId::new("i8_square", format!("{n}x{n}x{n}")),
            &n,
            |bencher, &nn| {
                bencher.iter(|| {
                    let rc = unsafe {
                        gemmi8(
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
    name = det_matmul_i8;
    // Real measurement time, not a 5s quick run: warm up + ample sampling so the
    // microsecond-scale GEMMs land with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_det_matmul_i8
}
criterion_main!(det_matmul_i8);
