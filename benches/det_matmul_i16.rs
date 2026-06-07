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

//! Deterministic int16 GEMV — execution-throughput benchmark with an embedded
//! **cross-substrate byte-identity** assertion (the int-dot sibling of
//! `det_matmul_q16`).
//!
//! ## Why this is the wedge, not a speed brag
//!
//! `y[r] = Σ_i W[r,i] · x[i]` over int16 inputs is integer multiply-accumulate
//! with an i64 accumulator narrowed once to i32 at the end. Integer add is
//! **associative**, so the compiler / hardware is free to reorder and vectorise
//! the reduction (the `vector`-dialect widen-multiply-accumulate loop mindc
//! lowers `__mind_blas_matmul_rmajor_i16_v` into — the AVX2 `vpmaddwd`
//! idiom at `-march=x86-64-v3`) **without changing a single output byte**. That
//! gives us *both* axes at once:
//!
//!   1. **Throughput** — a vectorised int16 reduction, timed here at 64×64,
//!      256×256 and 512×512 (matrix × vector) shapes (criterion).
//!   2. **Byte-identity** — the exact output bytes are pinned to the committed
//!      per-substrate reference hash in
//!      `tests/cross_substrate_identity/gemv-i16-256x256/reference_hashes.toml`
//!      (RFC 0020 §5). avx2 and neon carry **one identical hash** — the
//!      cross-substrate bit-identity claim made inspectable (RFC 0015 §3.1).
//!
//! Float GEMV (any BLAS) cannot make claim 2: f32 add is non-associative, so
//! reordering the reduction changes the result. Reordering is exactly what makes
//! the kernel fast. MIND keeps the reordering *and* the byte-identity because
//! the accumulation is integer.
//!
//! ## Additive, self-skipping, reuses the existing gate
//!
//! This bench adds **nothing** to `src/`. It drives the int16 matmul/gemv
//! intrinsic and the same workload directory the `cross_substrate_identity` test
//! gate uses — the LCG, the seed, the `i32_le → sha256` canonical encoding, and
//! the committed hash are all byte-for-byte what the
//! `gemv-i16-256x256` cross-substrate workload pins. The assertion is therefore
//! *wired to* the merge gate, not reinvented.
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
//! cargo bench --bench det_matmul_i16 --no-default-features
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

/// Square (rows == cols) GEMV shapes the throughput sweep exercises. The
/// 256×256 entry is the byte-identity anchor — its hash is pinned to the
/// committed reference.
const SHAPES: &[usize] = &[64, 256, 512];

/// Documented, conservative single-core int16 multiply-accumulate ceiling for
/// the host ISA (GMAC/s) — the roofline `%-of-ISA-peak` denominator
/// (`docs/benchmarking.md` §3). An **estimate against a documented constant**,
/// not a certified hardware peak. int16 `vpmaddwd` on AVX2 does 16 i16-MACs/op:
/// at the reference clock (~3.5 GHz, ~2 such ops/cyc) ≈ 112 GMAC/s. neon SDOT/
/// widen-MAC is taken conservatively at ~48 GMAC/s. Single-core; comparability
/// aid only.
fn isa_peak_gmacs() -> f64 {
    if cfg!(target_arch = "x86_64") {
        112.0
    } else if cfg!(target_arch = "aarch64") {
        48.0
    } else {
        f64::NAN
    }
}

/// The committed byte-identity workload (RFC 0020 §5). The 256×256 row of the
/// sweep regenerates exactly this input and must hash to this reference.
const ANCHOR_ID: &str = "gemv-i16-256x256";
const ANCHOR_N: usize = 256;
const ANCHOR_SEED: u64 = 0xDEAD_BEEF;

/// Kernel ABI: `mmi16(w, x, y, rows, cols) -> 0`. Computes `y[r] = Σ_i W[r,i]·x[i]`
/// over int16 W (rows×cols, row-major i16) and int16 x (cols i16), writing a
/// rows-element **i32** result y. Direct call into the int16 matmul/gemv
/// intrinsic — no func.call, no C shim.
type MatmulFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

/// `mmi16(w, x, y, rows, cols)` — the direct int16 row-major matmul/gemv path,
/// lowered inside mindc to a native `vector`-dialect reduction.
const SRC: &str = r#"
pub fn mmi16(w: i64, x: i64, y: i64, rows: i64, cols: i64) -> i64 {
    __mind_blas_matmul_rmajor_i16_v(w, x, y, rows, cols)
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

/// Compile the GEMV kernel to a temp `.so` once. Returns `None` (self-skip) if
/// the MLIR toolchain is shadowed or `mindc` is not built — same contract as the
/// gated test harnesses.
fn build_gemv_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("det_matmul_i16: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "det_matmul_i16: mindc not built; run \
                 `cargo build --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_det_matmul_i16.mind");
        let so_path = dir.join("mind_bench_det_matmul_i16.so");
        if std::fs::write(&src_path, SRC).is_err() {
            eprintln!("det_matmul_i16: could not write workload source; skipping");
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
                eprintln!("det_matmul_i16: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// Deterministic LCG — byte-for-byte the generator the Q16.16 cross-substrate
/// gate uses (same constants/seed contract), so the workload input distribution
/// is the shared, reproducible one. `next_i16` takes the full int16 range.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u32
    }
    fn next_i16(&mut self) -> i16 {
        // Take the high 16 bits of the 32-bit LCG state window as a signed
        // int16 — the full [-32768, 32767] range, deterministic per seed.
        (self.next_u32() >> 16) as i16
    }
}

/// Regenerate the seeded GEMV inputs: a rows×cols int16 matrix W (row-major)
/// and a cols-length int16 vector x, W generated before x (order is part of the
/// seed contract).
fn make_gemv_i16(rows: usize, cols: usize, seed: u64) -> (Vec<i16>, Vec<i16>) {
    let mut g = Lcg::new(seed);
    let w: Vec<i16> = (0..rows * cols).map(|_| g.next_i16()).collect();
    let x: Vec<i16> = (0..cols).map(|_| g.next_i16()).collect();
    (w, x)
}

/// Scalar int16 dot oracle, byte-for-byte the per-row reduction the kernel
/// performs: sign-extend each i16 to i64, multiply-accumulate exactly, then
/// narrow once to i32 (NO Q16 shift — raw integer dot). The independent
/// reference the vector reduction must match exactly within a run.
fn ref_dot_i16_scalar(w: &[i16], x: &[i16]) -> i32 {
    let mut acc: i64 = 0;
    for i in 0..w.len() {
        acc += (w[i] as i64) * (x[i] as i64);
    }
    acc as i32
}

/// Scalar int16 GEMV oracle: y[r] = dot_i16(W row r, x).
fn ref_gemv_i16_scalar(w: &[i16], x: &[i16], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows)
        .map(|r| ref_dot_i16_scalar(&w[r * cols..(r + 1) * cols], x))
        .collect()
}

/// Canonical output hash of an int16 GEMV result: each i32 little-endian, then
/// sha256 → lowercase hex (manifest `output_encoding = "i32_le_vector"`).
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

/// Run the GEMV once on host, returning the rows-length i32 output vector.
fn run_gemv(lib: &Library, w: &[i16], x: &[i16], rows: usize, cols: usize) -> Vec<i32> {
    let mmi16: Symbol<MatmulFn> = unsafe { lib.get(b"mmi16").expect("mmi16 symbol") };
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
    assert_eq!(rc, 0, "mmi16 kernel returned {rc} (expected 0)");
    y
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
    let (w, x) = make_gemv_i16(n, n, ANCHOR_SEED);
    let y = run_gemv(lib, &w, &x, n, n);

    // (3) within-run exactness vs the scalar oracle.
    let oracle = ref_gemv_i16_scalar(&w, &x, n, n);
    assert_eq!(
        y, oracle,
        "{ANCHOR_ID}: vector GEMV diverged from the scalar oracle within a single run"
    );

    let computed = canonical_hash_i32s(&y);

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
        "{ANCHOR_ID} [{substrate}]: GEMV output hash drifted from the committed reference.\n\
         computed={computed}\n expected={expected}\n\
         If this is an intentional lowering change (RFC 0020 §13), re-bless the \
         cross_substrate_identity gate with MIND_BENCH_BLESS=1 and commit the new hash."
    );

    eprintln!(
        "det_matmul_i16: byte-identity VERIFIED [{substrate}] {ANCHOR_ID} sha256={computed} \
         (avx2==neon committed)"
    );
}

/// Comparable execution-throughput axis (`docs/benchmarking.md` §3) for the
/// int16 GEMV: GMAC/s + `%-of-ISA-peak`. `MACs = rows·cols = n²`;
/// `GMAC/s = MACs / median_seconds / 1e9`; roofline `% = GMAC/s / isa_peak`.
/// Quick independent median (after warm-up) printed next to criterion's native
/// elem/s. Self-contained: derived from workload size, no BLAS reference.
fn report_gmacs_i16(lib: &Library, n: usize, seed: u64) {
    const WARMUP: usize = 8;
    const REPS: usize = 64;
    let (w, x) = make_gemv_i16(n, n, seed);
    let mut out = vec![0i32; n];
    let mmi16: Symbol<MatmulFn> = unsafe { lib.get(b"mmi16").expect("mmi16 symbol") };
    let call = |w: &[i16], x: &[i16], out: &mut [i32]| {
        let rc = unsafe {
            mmi16(
                w.as_ptr() as i64,
                x.as_ptr() as i64,
                out.as_mut_ptr() as i64,
                n as i64,
                n as i64,
            )
        };
        assert_eq!(rc, 0, "mmi16 returned {rc}");
    };
    for _ in 0..WARMUP {
        call(&w, &x, &mut out);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        call(black_box(&w), black_box(&x), black_box(&mut out));
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[REPS / 2];
    let macs = (n as f64) * (n as f64);
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
        "det_matmul_i16: ROOFLINE {n}x{n} {gmacs:7.2} GMAC/s  [{pct}]  (median {:.3} µs/call)",
        median * 1e6
    );
}

fn bench_det_matmul_i16(c: &mut Criterion) {
    let Some(so) = build_gemv_so() else {
        // Toolchain shadowed or mindc unbuilt — register no benchmarks, exit clean.
        eprintln!("det_matmul_i16: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen GEMV .so") };

    // Correctness gate first: a throughput number for output that drifted from
    // the committed bytes would be a lie. This panics on any mismatch.
    assert_byte_identity(&lib);

    let mut group = c.benchmark_group("det_matmul_i16");
    for &n in SHAPES {
        // One MAC per inner-product term: rows·cols multiply-accumulates.
        let macs = (n as u64) * (n as u64);
        group.throughput(Throughput::Elements(macs));

        // Comparable roofline axis (GMAC/s + %-of-ISA-peak), printed alongside
        // criterion's native elem/s, with the same per-shape seed.
        let report_seed = if n == ANCHOR_N {
            ANCHOR_SEED
        } else {
            0xDEAD_BEEF_0000_0000 ^ (n as u64)
        };
        report_gmacs_i16(&lib, n, report_seed);

        // Seed off the shape so each size has its own reproducible input; the
        // 256×256 entry uses the anchor seed so its bytes match the committed
        // reference exactly (already verified above).
        let seed = if n == ANCHOR_N {
            ANCHOR_SEED
        } else {
            0xDEAD_BEEF_0000_0000 ^ (n as u64)
        };
        let (w, x) = make_gemv_i16(n, n, seed);
        let mut out = vec![0i32; n];
        let mmi16: Symbol<MatmulFn> = unsafe { lib.get(b"mmi16").expect("mmi16 symbol") };

        group.bench_with_input(
            BenchmarkId::new("i16_gemv", format!("{n}x{n}")),
            &n,
            |bencher, &nn| {
                bencher.iter(|| {
                    let rc = unsafe {
                        mmi16(
                            black_box(w.as_ptr() as i64),
                            black_box(x.as_ptr() as i64),
                            black_box(out.as_mut_ptr() as i64),
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
    name = det_matmul_i16;
    // Real measurement time, not a 5s quick run: warm up + ample sampling so the
    // microsecond-scale GEMVs land with tight CIs and stable p50/p95.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_det_matmul_i16
}
criterion_main!(det_matmul_i16);
