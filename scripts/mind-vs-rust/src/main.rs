// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).
//
//! Apples-to-apples MIND-vs-Rust integer-GEMM benchmark.
//!
//! Times THREE things on the same machine, same shapes, same seeds, same
//! timing method (warmup + median of N, with black_box on the hot args):
//!
//!   (M) the MIND-compiled `.so` kernel (dlopen'd `gemmi8` / `gemmq`),
//!   (R) the equivalent hand-written Rust kernel built into THIS binary.
//!
//! The Rust opt-level (-O2 vs -O3) is whatever this binary was compiled at — we
//! build it twice and run it twice; the MIND number is the same in both runs
//! (it's the same `.so`), so the two runs let us compare MIND vs Rust-O2 and
//! MIND vs Rust-O3 on identical inputs.
//!
//! Correctness: before timing, the Rust kernel output is asserted byte-equal to
//! the MIND kernel output (0 mismatch) on the shared seed — proving they compute
//! the same thing. The math here is the byte-for-byte scalar accumulation the
//! cross_substrate_identity gate pins (i8: pure i32 MAC; q16: per-term `>>16`).
//!
//! Usage: `mind-vs-rust <path-to-mind.so> [--reps N]`

use std::env;
use std::time::Instant;

use libloading::{Library, Symbol};
use sha2::{Digest, Sha256};

const SHAPES: &[usize] = &[256, 512];
const SEED: u64 = 0xDEADBEEF;
const WARMUP: usize = 8;
const DEFAULT_REPS: usize = 64;

/// Kernel ABI shared by gemmi8 and gemmq: (a, b, c, m, k, n) -> 0.
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;

/// Deterministic LCG — byte-for-byte the generator the cross-substrate gate and
/// the det_matmul benches use, so the input distribution is the shared one.
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

// --- input generation: byte-for-byte the scaffolding's draw order -----------

fn make_gemm_i8(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i8>, Vec<i8>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i8> = (0..m * k).map(|_| (g.next_u32() >> 16) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|_| (g.next_u32() >> 16) as i8).collect();
    (a, b)
}

fn make_gemm_q16(m: usize, k: usize, n: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<i32> = (0..m * k).map(|_| g.next_q16()).collect();
    let b: Vec<i32> = (0..k * n).map(|_| g.next_q16()).collect();
    (a, b)
}

// --- the equivalent Rust kernels (SAME algorithm / int types / acc order) ---
// i8 GEMM: C[i,j] = (i32) Σ_k (i32)A[i,k]*(i32)B[k,j], i64 accumulate, B is the
// un-transposed K×N row-major operand — byte-for-byte ref_gemm_i8_scalar.
#[inline(never)]
fn rust_gemm_i8(a: &[i8], b: &[i8], c: &mut [i32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i64) * (b[kk * n + j] as i64);
            }
            c[i * n + j] = acc as i32;
        }
    }
}

// Q16.16 GEMM: C[i,j] = (i32) Σ_k ((A[i,k]*B[k,j]) >> 16) — each term shifted
// before it is summed; truncated once. Byte-for-byte ref_gemm_q16_scalar.
#[inline(never)]
fn rust_gemm_q16(a: &[i32], b: &[i32], c: &mut [i32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for kk in 0..k {
                acc += ((a[i * k + kk] as i64) * (b[kk * n + j] as i64)) >> 16;
            }
            c[i * n + j] = acc as i32;
        }
    }
}

fn canonical_hash_i32s(v: &[i32]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update(e.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// black_box: keep the optimiser from hoisting / dead-coding the timed work.
#[inline(never)]
fn black_box<T>(x: T) -> T {
    unsafe {
        let r = std::ptr::read_volatile(&x);
        std::mem::forget(x);
        r
    }
}

/// Median-of-REPS seconds after WARMUP, mirroring det_matmul_i8::report_gmacs_i8.
fn median_secs<F: FnMut()>(mut run: F, reps: usize) -> f64 {
    for _ in 0..WARMUP {
        run();
    }
    let mut s: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        run();
        s.push(t0.elapsed().as_secs_f64());
    }
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s[reps / 2]
}

fn gmacs(n: usize, median: f64) -> f64 {
    (n as f64) * (n as f64) * (n as f64) / median / 1e9
}

struct Row {
    workload: &'static str,
    n: usize,
    mind_gmacs: f64,
    mind_us: f64,
    rust_gmacs: f64,
    rust_us: f64,
    byte_match: bool,
    mind_hash: String,
}

fn bench_i8(lib: &Library, n: usize, reps: usize) -> Row {
    let (a, b) = make_gemm_i8(n, n, n, SEED);
    let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };

    // MIND
    let mut c_mind = vec![0i32; n * n];
    let mind_median = {
        let call = || {
            let rc = unsafe {
                gemmi8(
                    black_box(a.as_ptr() as i64),
                    black_box(b.as_ptr() as i64),
                    black_box(c_mind.as_mut_ptr() as i64),
                    n as i64,
                    n as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0);
        };
        median_secs(call, reps)
    };
    let mind_hash = canonical_hash_i32s(&c_mind);

    // Rust
    let mut c_rust = vec![0i32; n * n];
    let rust_median = {
        let call = || {
            rust_gemm_i8(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_rust),
                n,
                n,
                n,
            );
        };
        median_secs(call, reps)
    };
    let rust_hash = canonical_hash_i32s(&c_rust);

    Row {
        workload: "gemm-i8 (int8→i32)",
        n,
        mind_gmacs: gmacs(n, mind_median),
        mind_us: mind_median * 1e6,
        rust_gmacs: gmacs(n, rust_median),
        rust_us: rust_median * 1e6,
        byte_match: mind_hash == rust_hash,
        mind_hash,
    }
}

fn bench_q16(lib: &Library, n: usize, reps: usize) -> Row {
    let (a, b) = make_gemm_q16(n, n, n, SEED);
    let gemmq: Symbol<GemmFn> = unsafe { lib.get(b"gemmq").expect("gemmq symbol") };

    let mut c_mind = vec![0i32; n * n];
    let mind_median = {
        let call = || {
            let rc = unsafe {
                gemmq(
                    black_box(a.as_ptr() as i64),
                    black_box(b.as_ptr() as i64),
                    black_box(c_mind.as_mut_ptr() as i64),
                    n as i64,
                    n as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0);
        };
        median_secs(call, reps)
    };
    let mind_hash = canonical_hash_i32s(&c_mind);

    let mut c_rust = vec![0i32; n * n];
    let rust_median = {
        let call = || {
            rust_gemm_q16(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_rust),
                n,
                n,
                n,
            );
        };
        median_secs(call, reps)
    };
    let rust_hash = canonical_hash_i32s(&c_rust);

    Row {
        workload: "gemm-q16 (Q16.16)",
        n,
        mind_gmacs: gmacs(n, mind_median),
        mind_us: mind_median * 1e6,
        rust_gmacs: gmacs(n, rust_median),
        rust_us: rust_median * 1e6,
        byte_match: mind_hash == rust_hash,
        mind_hash,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: mind-vs-rust <path-to-mind.so> [--reps N]");
        std::process::exit(2);
    }
    let so_path = &args[1];
    let mut reps = DEFAULT_REPS;
    if let Some(i) = args.iter().position(|a| a == "--reps") {
        reps = args[i + 1].parse().expect("--reps N");
    }

    let opt = option_env!("MVR_OPT").unwrap_or("?");
    eprintln!("# mind-vs-rust  rustc-opt-level={opt}  reps={reps}  warmup={WARMUP}");
    eprintln!("# so={so_path}");

    let lib = unsafe { Library::new(so_path).expect("dlopen MIND .so") };

    let mut rows: Vec<Row> = Vec::new();
    for &n in SHAPES {
        rows.push(bench_i8(&lib, n, reps));
    }
    for &n in SHAPES {
        rows.push(bench_q16(&lib, n, reps));
    }

    // Machine-readable lines for the results doc (TSV).
    println!(
        "WORKLOAD\tN\tMIND_GMACs\tMIND_us\tRUST_GMACs\tRUST_us\tRATIO_MIND_over_RUST\tBYTE_MATCH\tMIND_HASH"
    );
    for r in &rows {
        let ratio = r.mind_gmacs / r.rust_gmacs;
        println!(
            "{}\t{}\t{:.2}\t{:.1}\t{:.2}\t{:.1}\t{:.3}\t{}\t{}",
            r.workload,
            r.n,
            r.mind_gmacs,
            r.mind_us,
            r.rust_gmacs,
            r.rust_us,
            ratio,
            if r.byte_match { "yes" } else { "NO" },
            &r.mind_hash[..16],
        );
    }

    let all_match = rows.iter().all(|r| r.byte_match);
    if !all_match {
        eprintln!("FATAL: a Rust kernel output did NOT byte-match the MIND kernel.");
        std::process::exit(1);
    }
    eprintln!(
        "# all outputs byte-match (MIND == Rust), {} workloads",
        rows.len()
    );
}
