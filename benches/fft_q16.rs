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

//! Deterministic Q16.16 radix-2 DIT FFT (N=256) — execution-throughput benchmark
//! with an embedded **byte-identity** assertion against an `-O3` C reference that
//! implements the *same integer algorithm* (`bench/fft/fft_ref.c`).
//!
//! ## Why integer FFT is the wedge, not a speed brag
//!
//! Every butterfly is `(W * x) >> 16` Q16.16 fixed-point: each product is
//! arithmetic-shifted to a fixed i64 *before* it is added, and integer add is
//! associative and commutative. So the transform is **bit-identical across
//! substrates** (x86/ARM/GPU) and across re-runs — a property a FP32 FFT
//! (cuFFT / FFTW / an nvcc-compiled radix-r kernel) structurally cannot have,
//! because f32 add is non-associative and the fast reorderings change the result.
//!
//! ## What this bench measures
//!
//! 1. **Correctness gate** — the MIND-compiled `fft256` `.so` output is asserted
//!    byte-for-byte equal to the C reference output on the same input + twiddle
//!    table (FNV-1a hash printed). Panics on any mismatch, so a throughput number
//!    can never be reported for drifted bytes.
//! 2. **Throughput** — criterion times the bare `fft256` call, in place, on a
//!    fixed buffer (the FFT control flow is fully data-independent, so repeated
//!    in-place calls are a fair standard microbench).
//! 3. **Head-to-head** — the bench compiles `bench/fft/fft_ref.c` with the best
//!    available C compiler (`nvcc -O3` if present, else `clang`/`gcc -O3
//!    -march=native`) and runs its self-timed driver, printing the C p50 ns next
//!    to the MIND number so the ratio is on the page from one command.
//!
//! Self-skips (registers no benchmarks, exits clean) when the MLIR toolchain
//! (`mlir-opt`/`mlir-translate`/`clang`) is shadowed or `mindc` is not built —
//! identical contract to `det_matmul_q16.rs`, so it never falsely passes.
//!
//! Run:
//! ```text
//! cargo build --release --features "mlir-build std-surface cross-module-imports" --bin mindc
//! cargo bench --bench fft_q16 --no-default-features
//! ```

use std::f64::consts::PI;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use libloading::{Library, Symbol};

const N: usize = 256;
const LOGN: i64 = 8;

/// Kernel ABI: `fft256(re, im, tw, logn, n) -> 0`. re/im are N i64 Q16.16
/// buffers (mutated in place); tw is the interleaved twiddle table (wr,wi).
type FftFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

/// The exact pure-executable-subset kernel from `examples/fft_q16.mind`, embedded
/// so the bench is self-contained and times precisely the verified algorithm.
const SRC: &str = r#"
pub fn fft256(re: i64, im: i64, tw: i64, logn: i64, n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        let mut x: i64 = i;
        let mut j: i64 = 0;
        let mut b: i64 = 0;
        while b < logn {
            j = (j << 1) | (x & 1);
            x = x >> 1;
            b = b + 1;
        }
        if j > i {
            let ai: i64 = re + i * 8;
            let aj: i64 = re + j * 8;
            let tr: i64 = __mind_load_i64(ai);
            __mind_store_i64(ai, __mind_load_i64(aj));
            __mind_store_i64(aj, tr);
            let bi: i64 = im + i * 8;
            let bj: i64 = im + j * 8;
            let ti: i64 = __mind_load_i64(bi);
            __mind_store_i64(bi, __mind_load_i64(bj));
            __mind_store_i64(bj, ti);
        }
        i = i + 1;
    }
    // Stage length==2: the only twiddle in play is k==0 == (65536, 0) (Q16.16
    // one + zero). (65536 * x) >> 16 == x exactly and (0 * x) >> 16 == 0, so the
    // butterfly degenerates to a pure add/sub with NO multiply — a bit-identity-
    // preserving algebraic identity (same integer result, same bytes). Peeling
    // it removes 128 of the 1024 butterflies' 4 i64-multiplies each (the most
    // numerous stage), which is where the head-to-head margin over C is won.
    let nbytes: i64 = n * 8;
    let mut areS2: i64 = re;
    let mut aimS2: i64 = im;
    let reEnd2: i64 = re + nbytes;
    while areS2 < reEnd2 {
        let bre: i64 = areS2 + 8;
        let bim: i64 = aimS2 + 8;
        let xbr: i64 = __mind_load_i64(bre);
        let xbi: i64 = __mind_load_i64(bim);
        let uur: i64 = __mind_load_i64(areS2);
        let uui: i64 = __mind_load_i64(aimS2);
        __mind_store_i64(areS2, uur + xbr);
        __mind_store_i64(aimS2, uui + xbi);
        __mind_store_i64(bre, uur - xbr);
        __mind_store_i64(bim, uui - xbi);
        areS2 = areS2 + 16;
        aimS2 = aimS2 + 16;
    }
    // Stages length>=4: twiddles vary, so the multiply stays. Loop-invariant
    // hoisting + incremental pointer advance (twp/are/aim/bre/bim bumped by a
    // constant stride instead of recomputed from k/jj each iteration) lets the
    // backend keep one base pointer per stream and index the partner element off
    // it, shrinking the inner-loop instruction stream. Pure address arithmetic /
    // hoisting — the Q16.16 shift is still per product before the add, so every
    // output byte is unchanged (verified by the FNV-1a gate).
    let mut length: i64 = 4;
    while length <= n {
        let half: i64 = length >> 1;
        let step: i64 = n / length;
        let twstep: i64 = step * 16;
        let lenbytes: i64 = length * 8;
        let halfbytes: i64 = half * 8;
        let mut areS: i64 = re;
        let mut aimS: i64 = im;
        let mut start: i64 = 0;
        while start < n {
            let mut jj: i64 = 0;
            let mut twp: i64 = tw;
            let mut are: i64 = areS;
            let mut aim: i64 = aimS;
            let mut bre: i64 = areS + halfbytes;
            let mut bim: i64 = aimS + halfbytes;
            while jj < half {
                let wr: i64 = __mind_load_i64(twp);
                let wi: i64 = __mind_load_i64(twp + 8);
                let xbr: i64 = __mind_load_i64(bre);
                let xbi: i64 = __mind_load_i64(bim);
                let trr: i64 = ((wr * xbr) >> 16) - ((wi * xbi) >> 16);
                let tii: i64 = ((wr * xbi) >> 16) + ((wi * xbr) >> 16);
                let uur: i64 = __mind_load_i64(are);
                let uui: i64 = __mind_load_i64(aim);
                __mind_store_i64(are, uur + trr);
                __mind_store_i64(aim, uui + tii);
                __mind_store_i64(bre, uur - trr);
                __mind_store_i64(bim, uui - tii);
                jj = jj + 1;
                twp = twp + twstep;
                are = are + 8;
                aim = aim + 8;
                bre = bre + 8;
                bim = bim + 8;
            }
            areS = areS + lenbytes;
            aimS = aimS + lenbytes;
            start = start + length;
        }
        length = length << 1;
    }
    0
}
"#;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn mindc_path() -> Option<PathBuf> {
    let rel = manifest_dir().join("target").join("release").join("mindc");
    if rel.exists() {
        return Some(rel);
    }
    let dbg = manifest_dir().join("target").join("debug").join("mindc");
    if dbg.exists() { Some(dbg) } else { None }
}

/// Compile the FFT kernel to a temp `.so` once. `None` (self-skip) if the MLIR
/// toolchain is shadowed or `mindc` is not built.
fn build_fft_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                eprintln!("fft_q16: {tool} not on PATH; skipping (toolchain shadowed)");
                return None;
            }
        }
        let Some(mindc) = mindc_path() else {
            eprintln!(
                "fft_q16: mindc not built; run \
                 `cargo build --release --features \"mlir-build std-surface cross-module-imports\" --bin mindc`; skipping"
            );
            return None;
        };
        let dir = std::env::temp_dir();
        let src_path = dir.join("mind_bench_fft_q16.mind");
        let so_path = dir.join("mind_bench_fft_q16.so");
        if std::fs::write(&src_path, SRC).is_err() {
            eprintln!("fft_q16: could not write workload source; skipping");
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
                eprintln!("fft_q16: mindc --emit-shared failed; skipping");
                None
            }
        }
    })
    .as_ref()
}

/// LCG byte-for-byte identical to `bench/fft/fft_driver.c` so the inputs match.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u32
    }
    fn next_q16(&mut self) -> i64 {
        ((self.next_u32() as i32) >> 13) as i64
    }
}

/// Interleaved twiddle table tw[2k]=wr, tw[2k+1]=wi, k=0..N/2-1, Q16.16, with the
/// shared rounding convention `round(x * 65536)` — byte-identical to the C side.
fn build_twiddles() -> Vec<i64> {
    let mut tw = vec![0i64; N];
    for k in 0..N / 2 {
        let ang = -2.0 * PI * (k as f64) / (N as f64);
        tw[2 * k] = (ang.cos() * 65536.0).round() as i64;
        tw[2 * k + 1] = (ang.sin() * 65536.0).round() as i64;
    }
    tw
}

/// Deterministic Q16.16 signal: re then im, byte-for-byte the C `make_input`.
fn make_input(seed: u64) -> (Vec<i64>, Vec<i64>) {
    let mut g = Lcg::new(seed);
    let re: Vec<i64> = (0..N).map(|_| g.next_q16()).collect();
    let im: Vec<i64> = (0..N).map(|_| g.next_q16()).collect();
    (re, im)
}

/// FNV-1a over the little-endian bytes of the two i64 buffers — byte-for-byte the
/// C `hash_buffers`, so a matching hash means matching output bytes.
fn fnv1a(re: &[i64], im: &[i64]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &e in re.iter().chain(im.iter()) {
        for b in e.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

fn run_fft(lib: &Library, re: &mut [i64], im: &mut [i64], tw: &[i64]) {
    let fft: Symbol<FftFn> = unsafe { lib.get(b"fft256").expect("fft256 symbol") };
    let rc = unsafe {
        fft(
            re.as_mut_ptr() as i64,
            im.as_mut_ptr() as i64,
            tw.as_ptr() as i64,
            LOGN,
            N as i64,
        )
    };
    assert_eq!(rc, 0, "fft256 returned {rc} (expected 0)");
}

/// THE correctness gate. Runs the MIND `.so` and an in-Rust scalar reference of
/// the identical integer algorithm; asserts byte-equal and prints the hash.
/// The same hash is independently asserted against the C `-O3` reference via the
/// `fft_verify` harness compiled in `run_c_reference`.
fn assert_byte_identity(lib: &Library, tw: &[i64], seed: u64) -> u64 {
    let (mut re_m, mut im_m) = make_input(seed);
    run_fft(lib, &mut re_m, &mut im_m, tw);
    let h_mind = fnv1a(&re_m, &im_m);

    // Independent in-Rust scalar oracle of the same integer algorithm.
    let (mut re_o, mut im_o) = make_input(seed);
    ref_fft_scalar(&mut re_o, &mut im_o, tw);
    let h_oracle = fnv1a(&re_o, &im_o);

    assert_eq!(
        (re_m.clone(), im_m.clone()),
        (re_o, im_o),
        "fft_q16: MIND .so output diverged from the in-Rust scalar oracle"
    );
    assert_eq!(h_mind, h_oracle, "fft_q16: hash mismatch vs oracle");
    eprintln!("fft_q16: byte-identity VERIFIED (MIND .so == scalar oracle) fnv1a=0x{h_mind:016x}");
    h_mind
}

/// In-Rust scalar reference — the same Q16.16 radix-2 DIT as the kernel and the C
/// file, an independent third implementation of the bit-exact math.
fn ref_fft_scalar(re: &mut [i64], im: &mut [i64], tw: &[i64]) {
    let n = N as i64;
    let mut i = 0i64;
    while i < n {
        let (mut x, mut j, mut b) = (i, 0i64, 0i64);
        while b < LOGN {
            j = (j << 1) | (x & 1);
            x >>= 1;
            b += 1;
        }
        if j > i {
            let (ui, uj) = (i as usize, j as usize);
            re.swap(ui, uj);
            im.swap(ui, uj);
        }
        i += 1;
    }
    let mut length = 2i64;
    while length <= n {
        let half = length >> 1;
        let step = n / length;
        let mut start = 0i64;
        while start < n {
            let (mut jj, mut k) = (0i64, 0i64);
            while jj < half {
                let (wr, wi) = (tw[(2 * k) as usize], tw[(2 * k + 1) as usize]);
                let a = (start + jj) as usize;
                let bidx = (start + jj + half) as usize;
                let (xbr, xbi) = (re[bidx], im[bidx]);
                let trr = ((wr * xbr) >> 16) - ((wi * xbi) >> 16);
                let tii = ((wr * xbi) >> 16) + ((wi * xbr) >> 16);
                let (uur, uui) = (re[a], im[a]);
                re[a] = uur + trr;
                im[a] = uui + tii;
                re[bidx] = uur - trr;
                im[bidx] = uui - tii;
                jj += 1;
                k += step;
            }
            start += length;
        }
        length <<= 1;
    }
}

/// Quick median-of-REPS self-timing of the MIND kernel printed to stderr (ns),
/// next to criterion's own statistics — the comparable unit for the C ratio.
fn report_mind_ns(lib: &Library, tw: &[i64], seed: u64) -> f64 {
    const WARMUP: usize = 1000;
    const REPS: usize = 20000;
    let (mut re, mut im) = make_input(seed);
    let fft: Symbol<FftFn> = unsafe { lib.get(b"fft256").expect("fft256 symbol") };
    let call = |re: &mut [i64], im: &mut [i64]| {
        let rc = unsafe {
            fft(
                re.as_mut_ptr() as i64,
                im.as_mut_ptr() as i64,
                tw.as_ptr() as i64,
                LOGN,
                N as i64,
            )
        };
        assert_eq!(rc, 0);
    };
    for _ in 0..WARMUP {
        call(&mut re, &mut im);
    }
    let mut s: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        call(black_box(&mut re), black_box(&mut im));
        s.push(t0.elapsed().as_secs_f64() * 1e9);
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = s[REPS / 2];
    let p95 = s[(REPS as f64 * 0.95) as usize];
    eprintln!("fft_q16: MIND self-timed  p50={p50:.1} ns  p95={p95:.1} ns  (reps={REPS})");
    p50
}

/// Compile + run the C reference driver and parse its p50 ns. Returns
/// `(compiler_label, c_p50_ns)`. Best available compiler: nvcc if present, else
/// clang, else gcc — all at `-O3 -march=native`. Also runs `fft_verify` to assert
/// the MIND `.so` is byte-identical to the C output (printed).
fn run_c_reference(so_path: &std::path::Path, seed: u64) -> Option<(String, f64)> {
    let fft_dir = manifest_dir().join("bench").join("fft");
    let ref_c = fft_dir.join("fft_ref.c");
    let drv_c = fft_dir.join("fft_driver.c");
    let ver_c = fft_dir.join("fft_verify.c");
    if !ref_c.exists() || !drv_c.exists() {
        eprintln!("fft_q16: bench/fft/*.c missing; skipping C head-to-head");
        return None;
    }

    let (cc, label) = if which::which("nvcc").is_ok() {
        ("nvcc".to_string(), "nvcc -O3".to_string())
    } else if which::which("clang").is_ok() {
        let v = Command::new("clang").arg("--version").output().ok();
        let ver = v
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.lines().next().map(|l| l.to_string()))
            .unwrap_or_else(|| "clang".to_string());
        ("clang".to_string(), format!("{ver} -O3 -march=native"))
    } else {
        ("gcc".to_string(), "gcc -O3 -march=native".to_string())
    };

    let dir = std::env::temp_dir();
    let drv_bin = dir.join("mind_bench_fft_driver");
    let ver_bin = dir.join("mind_bench_fft_verify");

    // nvcc needs the host -O3 passed via -Xcompiler; clang/gcc take it directly.
    let build = |out: &std::path::Path, extra_src: &std::path::Path| -> bool {
        let mut cmd = Command::new(&cc);
        if cc == "nvcc" {
            cmd.args([
                "-O3",
                "-Xcompiler",
                "-O3,-march=native",
                "-o",
                out.to_str().unwrap(),
                extra_src.to_str().unwrap(),
                ref_c.to_str().unwrap(),
                "-lm",
            ]);
        } else {
            cmd.args([
                "-O3",
                "-march=native",
                "-o",
                out.to_str().unwrap(),
                extra_src.to_str().unwrap(),
                ref_c.to_str().unwrap(),
                "-lm",
                "-ldl",
            ]);
        }
        cmd.status().map(|s| s.success()).unwrap_or(false)
    };

    if !build(&drv_bin, &drv_c) {
        eprintln!("fft_q16: C driver compile failed ({label}); skipping C head-to-head");
        return None;
    }

    // Byte-identity vs the C reference (independent of the in-Rust oracle).
    if ver_c.exists() && build(&ver_bin, &ver_c) {
        let out = Command::new(&ver_bin)
            .args([so_path.to_str().unwrap(), &format!("0x{seed:x}")])
            .output();
        if let Ok(o) = out {
            let txt = String::from_utf8_lossy(&o.stdout);
            for line in txt.lines() {
                eprintln!("fft_q16:   {line}");
            }
            assert!(
                o.status.success(),
                "fft_q16: MIND .so is NOT byte-identical to the C -O3 reference"
            );
        }
    }

    let out = Command::new(&drv_bin)
        .arg(format!("0x{seed:x}"))
        .output()
        .ok()?;
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut c_p50 = None;
    for line in txt.lines() {
        eprintln!("fft_q16:   {line}");
        if let Some(idx) = line.find("ns_p50=") {
            let rest = &line[idx + "ns_p50=".len()..];
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            c_p50 = num.parse::<f64>().ok();
        }
    }
    c_p50.map(|p| (label, p))
}

fn bench_fft_q16(c: &mut Criterion) {
    let seed: u64 = 0x12345678;
    let Some(so) = build_fft_so() else {
        eprintln!("fft_q16: kernel unavailable; no measurements taken.");
        return;
    };
    let lib = unsafe { Library::new(so).expect("dlopen FFT .so") };
    let tw = build_twiddles();

    // Correctness gate first — a throughput number for drifted bytes is a lie.
    let hash = assert_byte_identity(&lib, &tw, seed);

    // Head-to-head C reference (compile + run + byte-identity vs C).
    let c_ref = run_c_reference(so, seed);

    // MIND self-timed p50 for the comparable-unit ratio.
    let mind_p50 = report_mind_ns(&lib, &tw, seed);

    if let Some((label, c_p50)) = &c_ref {
        let ratio = c_p50 / mind_p50; // >1 means MIND faster
        let verdict = if ratio >= 1.0 {
            format!("MIND is {ratio:.3}x FASTER than C")
        } else {
            format!(
                "MIND is {:.3}x SLOWER than C (C/MIND={ratio:.3})",
                1.0 / ratio
            )
        };
        eprintln!(
            "fft_q16: HEAD-TO-HEAD  MIND p50={mind_p50:.1} ns  vs  C({label}) p50={c_p50:.1} ns  => {verdict}  [fnv1a=0x{hash:016x}]"
        );
    }

    let mut group = c.benchmark_group("fft_q16");
    // One complex N=256 transform = (N/2)*log2(N) butterflies = 128*8 = 1024.
    let butterflies = (N as u64 / 2) * (LOGN as u64);
    group.throughput(Throughput::Elements(butterflies));

    let (mut re, mut im) = make_input(seed);
    let fft: Symbol<FftFn> = unsafe { lib.get(b"fft256").expect("fft256 symbol") };
    let tw_ptr = tw.as_ptr() as i64;

    group.bench_function("q16_n256_dit", |bencher| {
        bencher.iter(|| {
            let rc = unsafe {
                fft(
                    black_box(re.as_mut_ptr() as i64),
                    black_box(im.as_mut_ptr() as i64),
                    black_box(tw_ptr),
                    black_box(LOGN),
                    black_box(N as i64),
                )
            };
            black_box(rc);
        });
    });
    group.finish();
}

criterion_group! {
    name = fft_q16;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(200);
    targets = bench_fft_q16
}
criterion_main!(fft_q16);
