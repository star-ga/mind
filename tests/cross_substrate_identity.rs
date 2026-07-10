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

mod common;
use common::mindc_bin;

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
// MULTITHREADED fused int8 GEMM (the "det.igemm" MT surface,
// __mind_blas_matmul_mm_i8_mt_v). Same ABI + byte-for-byte output as the
// single-thread `gemmi8` above: the M output rows are split into contiguous
// owner-computes bands, one per POSIX thread, each running the SAME
// BLIS-blocked int8 macro-kernel over its band — NO cross-thread reduction, NO
// atomic, NO shared accumulator. Integer add is associative + commutative, so
// the thread-band partition is byte-identical to the single-thread reduction
// REGARDLESS of the runtime thread count T (= online CPUs). RANK 4 canary.
pub fn gemmi8mt(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_i8_mt_v(a, b, c, m, k, n)
}
// STRICT-FP f32 vector dot (__mind_blas_dot_f32_v): 8-lane accumulator with the
// FMA UNFUSED (separate mulf+addf) and the horizontal reduction pinned to a
// fixed left-to-right lane fold (NOT a target-defined `vector.reduction`), so
// the dot is bit-exact — a strict-FP tier, not a tolerance path. Result is the
// f32 bits packed into the low 32 bits of an i64 (Option-C ABI). RANK 7 canary:
// avx2 blessed here; neon deferred pending a real-aarch64 bless.
pub fn dotf32(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_f32_v(a, b, n)
}
// STRICT-FP row-major f32 matmul (__mind_blas_matmul_rmajor_f32_v): outer loop
// over rows, the SAME pinned fixed-order f32 dot fold inlined per row, writing
// the rows-length f32 result buffer y. Same strict-FP determinism contract as
// `dotf32`. RANK 7 canary.
pub fn mmf32(w: i64, x: i64, y: i64, rows: i64, cols: i64) -> i64 {
    __mind_blas_matmul_rmajor_f32_v(w, x, y, rows, cols)
}
// Track #16 additions — broaden the cross-substrate canary set with
// determinism-sensitive paths NOT covered by the kernels above.
//
// (1) Bare int16 dot reduction (`int-dot` tier). The gemv-i16 canary exercises
// the matrix-x-vector wrapper; this hits the raw reduction intrinsic directly
// (sext i16->i64, mac, narrow once to i32 — NO Q16 shift). Exact integer add is
// associative, so the result is grouping-/substrate-independent (RFC 0015 §3.1).
pub fn doti16(a: i64, b: i64, n: i64) -> i64 {
    __mind_blas_dot_i16_v(a, b, n)
}
// (2) FUSED Q16.16 GEMM via the outer-product microkernel intrinsic
// __mind_blas_matmul_mm_q16_v (A M×K, B K×N UN-transposed, C M×N) — a DISTINCT
// lowering from the gemv-composed `gemmq` above (register-tiled outer product,
// no horizontal reduction). Per-product `>> 16` then i64 accumulate, byte-
// identical to the scalar oracle Σ_k (A[i,k]*B[k,j])>>16 for all shapes.
pub fn gemmqmm(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_q16_v(a, b, c, m, k, n)
}
// (3) Scalar Q16.16 fixed-point arithmetic chain — exercises mindc's per-element
// fixed-point lowering directly (i64 multiply, arithmetic shift-right by 16,
// add/sub) with NO intrinsic, NO reduction. q16_mul(x,y) = (x*y) >> 16 is the
// fundamental Q16.16 product; the chain composes mul/add/sub in a fixed source
// order. Each op is exact integer arithmetic with a single deterministic
// truncating shift per product, so the result is byte-identical across
// substrates by construction (RFC 0015 §3.1) — the scalar analogue of the
// Q16.16 reduction tiers, isolating the shift+arith lowering.
fn q16_mul(x: i64, y: i64) -> i64 {
    (x * y) >> 16
}
pub fn q16_arith_chain(a: i64, b: i64, c: i64) -> i64 {
    // ((a*b) + (b*c)) - (a*c), all in Q16.16; fixed precedence, no contraction.
    let ab: i64 = q16_mul(a, b);
    let bc: i64 = q16_mul(b, c);
    let ac: i64 = q16_mul(a, c);
    (ab + bc) - ac
}
// (4) Struct-by-handle round-trip: allocate a 4-field i64 record on the heap via
// __mind_alloc, store the inputs through __mind_store_i64, read them back through
// __mind_load_i64, and combine them. Exercises the alloc/store/load handle ABI
// and its address arithmetic (the struct-by-handle path) — deterministic data
// movement, no reduction, no float. The result is a fixed integer function of
// the inputs, so it is byte-identical across substrates by construction.
pub fn struct_handle_roundtrip(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let base: i64 = __mind_alloc(32);
    __mind_store_i64(base, a);
    __mind_store_i64(base + 8, b);
    __mind_store_i64(base + 16, c);
    __mind_store_i64(base + 24, d);
    let r0: i64 = __mind_load_i64(base);
    let r1: i64 = __mind_load_i64(base + 8);
    let r2: i64 = __mind_load_i64(base + 16);
    let r3: i64 = __mind_load_i64(base + 24);
    // Fixed combination: (r0 + r1) * 2 - r2 + r3 * 3.
    ((r0 + r1) * 2 - r2) + r3 * 3
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
// SCALAR int<->float `as`-cast conversion canary (#92/#93 follow-up). This is
// the ONLY cross-substrate fixture that exercises the scalar conversion
// lowering — the coverage hole that let a real float→int WEDGE BREAK ship: a
// bare `arith.fptosi` is target-defined out of range (x86 `cvttsd2si` →
// INT64_MIN for every out-of-range/NaN input; ARM `fcvtzs` SATURATES), so
// `1e30 as i64` / `inf as i64` / `nan as i64` produced DIFFERENT bytes on avx2
// vs neon until `emit_saturating_fp_to_i64` replaced it with a fully
// IEEE-defined clamp (maxnumf/minnumf → in-range fptosi → select on the ≥2^N
// overflow and the NaN predicate). The edge operands are passed in as RUNTIME
// f64 arguments (never in-kernel literals) so mlir-opt cannot constant-fold the
// conversion away — the saturating path provably executes at run time.
//
// Casts exercised: int→f64 (`n as f64`, sitofp) and int→f32 (`n as f32`,
// sitofp+round), bool→f64 (`(inr < povf) as f64`, i1→uitofp → 1.0), and the
// float→i64 SATURATING edges: in-range (9.7→9, truncate toward zero),
// +overflow (1e30→INT64_MAX), −overflow (−1e30→INT64_MIN), NaN (→0), +inf
// (→INT64_MAX), −inf (→INT64_MIN). Because the saturating result is built only
// from IEEE-defined ops it is identical on avx2 and neon BY CONSTRUCTION
// (RFC 0015 §3.1) — that is what this fixture pins. The nine cast results are
// folded in a FIXED source order into one i64 via a wrapping polynomial
// (`acc = acc*K + term`, K = 1000003) so that equal saturated values at
// different positions do NOT cancel — a divergence in any single edge changes
// the final byte. `arith.muli`/`arith.addi` wrap in two's complement (matching
// Rust `wrapping_mul`/`wrapping_add`), so the fold itself is substrate-exact.
pub fn scalar_cast_conv(
    inr: f64,
    povf: f64,
    novf: f64,
    nan: f64,
    pinf: f64,
    ninf: f64,
    n: i64,
) -> i64 {
    let e0: i64 = inr as i64; // in-range 9.7 -> 9 (trunc toward zero)
    let e1: i64 = povf as i64; // +overflow 1e30 -> INT64_MAX
    let e2: i64 = novf as i64; // -overflow -1e30 -> INT64_MIN
    let e3: i64 = nan as i64; // NaN -> 0
    let e4: i64 = pinf as i64; // +inf -> INT64_MAX
    let e5: i64 = ninf as i64; // -inf -> INT64_MIN
    let b0: i64 = (n as f64) as i64; // int -> f64 -> i64 (round-trip)
    let b1: i64 = (n as f32) as i64; // int -> f32 (distinct rounding) -> i64
    let b2: i64 = (inr < povf) as f64 as i64; // bool -> f64 (1.0) -> i64
    let k: i64 = 1000003;
    // Fixed-order wrapping polynomial fold, starting acc = e0.
    let a1: i64 = e0 * k + e1;
    let a2: i64 = a1 * k + e2;
    let a3: i64 = a2 * k + e3;
    let a4: i64 = a3 * k + e4;
    let a5: i64 = a4 * k + e5;
    let a6: i64 = a5 * k + b0;
    let a7: i64 = a6 * k + b1;
    a7 * k + b2
}

// Scalar float→NARROW-int `as`-cast conversion canary (scalar-cast-conv-narrow).
// The width-tier sibling of `scalar_cast_conv`: it pins the float→`i8`/`i16`/
// `i32`/`u8`/`u16`/`u32` lowering, which SATURATES to the NARROW target range
// (`emit_saturating_fp_to_narrow`) — distinct from the full-width float→i64 the
// kernel above pins, and distinct from integer narrowing (which WRAPS). The
// six f64 edge operands arrive as RUNTIME args (no const-fold) and are cast to a
// spread of narrow widths, exercising every saturation edge AT THE NARROW BOUND:
//   inr  9.7   → i8 9,  u32 9            (in-range trunc toward zero)
//   povf 1e30  → i8 127, i16 32767       (+overflow saturates to iN::MAX)
//   novf -1e30 → u8 0,  i32 -2147483648  (−overflow → 0 / iN::MIN)
//   nan  NaN   → i16 0                    (NaN → 0)
//   pinf +inf  → i32 2147483647          (+inf → iN::MAX)
//   ninf -inf  → u16 0                    (−inf → 0 for unsigned)
// Emitting integer shifts on a float SSA value (the historical bug) failed to
// COMPILE (mlir-opt `i64 vs f64`); the saturating clamp is built only from
// IEEE-defined ops (maxnumf/minnumf/fptosi/cmpf/select) + a two's-complement
// integer clamp (maxsi/minsi), so avx2 == neon BY CONSTRUCTION (RFC 0015 §3.1).
// The nine narrow results fold in FIXED source order via the same wrapping
// polynomial (K = 1000003) so a divergence in any single edge changes the byte.
pub fn scalar_cast_conv_narrow(
    inr: f64,
    povf: f64,
    novf: f64,
    nan: f64,
    pinf: f64,
    ninf: f64,
) -> i64 {
    let e0: i64 = (inr as i8) as i64; // 9.7 -> 9
    let e1: i64 = (povf as i8) as i64; // 1e30 -> 127 (i8::MAX)
    let e2: i64 = (novf as u8) as i64; // -1e30 -> 0
    let e3: i64 = (nan as i16) as i64; // NaN -> 0
    let e4: i64 = (pinf as i32) as i64; // +inf -> 2147483647 (i32::MAX)
    let e5: i64 = (ninf as u16) as i64; // -inf -> 0
    let e6: i64 = (inr as u32) as i64; // 9.7 -> 9
    let e7: i64 = (povf as i16) as i64; // 1e30 -> 32767 (i16::MAX)
    let e8: i64 = (novf as i32) as i64; // -1e30 -> -2147483648 (i32::MIN)
    let k: i64 = 1000003;
    let a1: i64 = e0 * k + e1;
    let a2: i64 = a1 * k + e2;
    let a3: i64 = a2 * k + e3;
    let a4: i64 = a3 * k + e4;
    let a5: i64 = a4 * k + e5;
    let a6: i64 = a5 * k + e6;
    let a7: i64 = a6 * k + e7;
    a7 * k + e8
}
"#;

type DotFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
type MatmulFn = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
type GemmFn = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
/// The scalar-f64 chain: four f64 args in, one f64 result out (System V xmm ABI).
type ScalarF64Fn = unsafe extern "C" fn(f64, f64, f64, f64) -> f64;
/// The scalar int↔float cast conv kernel: six f64 edge operands (xmm0-5) + one
/// i64 (rdi) in, one folded i64 out.
type ScalarCastFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64) -> i64;
/// The scalar float→narrow-int cast conv kernel: six f64 edge operands, one
/// folded i64 out (no int source — pins the saturating-to-narrow float path).
type ScalarCastNarrowFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64) -> i64;
/// Track #16: a 3-arg → i64 scalar kernel (the Q16.16 arithmetic chain).
type Arith3Fn = unsafe extern "C" fn(i64, i64, i64) -> i64;
/// Track #16: a 4-arg → i64 scalar kernel (the struct-by-handle round-trip).
type Arith4Fn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
/// The Lorenz integrator: (state_ptr, steps) → final x, buffer mutated in place.
type LorenzFn = unsafe extern "C" fn(i64, i64) -> i64;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

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
        let status = Command::new(mindc_bin())
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

/// Compile `examples/lorenz_q16.mind` to a temp `.so` once for the whole test
/// binary. Separate from `build_dot_so` because the Lorenz kernel lives in the
/// examples/ tree (it is also a shipped demo) and uses a pointer-to-buffer ABI
/// rather than the scalar dot/arith kernels embedded in `SRC`. Same toolchain
/// guard / self-skip discipline: `None` when the MLIR toolchain is shadowed,
/// but a hard failure under `MIND_BENCH_REQUIRE` so the gate can never pass
/// vacuously (RFC 0020 §10).
fn build_lorenz_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
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
        // examples/lorenz_q16.mind, relative to the crate root.
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("lorenz_q16.mind");
        let so_path = std::env::temp_dir().join("mind_xsi_lorenz_q16.so");
        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("spawn mindc --emit-shared for lorenz_q16");
        assert!(
            status.success(),
            "mindc --emit-shared failed for the lorenz-q16 workload"
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

    // RANK 8 — extend run-to-run coverage from 4 canaries to EVERY buffer-
    // writing / result-producing canary in the suite. Each closure runs its
    // kernel and returns the canonical hash of its output; `stable` re-runs it
    // DETERMINISM_RUNS times and asserts a byte-identical hash every time. This
    // turns any latent run-to-run non-determinism (uninitialised padding, prior
    // buffer state, allocation-order-dependent writes, a data race in the MT
    // kernel) into a deterministic single-run failure instead of a CI flake.
    let stable = |label: &str, run_once: &dyn Fn() -> String| {
        let first = run_once();
        for run in 1..DETERMINISM_RUNS {
            assert_eq!(
                run_once(),
                first,
                "{label}: run {run} diverged from run 0 in the same process \
                 (run-to-run non-determinism)"
            );
        }
    };

    // Q16.16 L1 reduction (dot-l1-q16).
    {
        let dot: Symbol<DotFn> = unsafe { lib.get(b"dotl1q").expect("dotl1q symbol") };
        let (a, b) = make_pair_q16(65536, 0xDEADBEEF);
        stable("dot-l1-q16", &|| {
            canonical_hash(unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, a.len() as i64) })
        });
    }

    // Bare int16 dot reduction (dot-i16).
    {
        let dot: Symbol<DotFn> = unsafe { lib.get(b"doti16").expect("doti16 symbol") };
        let (a, b) = make_pair_i16(4096, 0xDEADBEEF);
        stable("dot-i16", &|| {
            canonical_hash(unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, a.len() as i64) })
        });
    }

    // int16 gemv buffer path (gemv-i16).
    {
        let mmi16: Symbol<MatmulFn> = unsafe { lib.get(b"mmi16").expect("mmi16 symbol") };
        let (rows, cols) = (256usize, 256usize);
        let (w, x) = make_gemv_i16(rows, cols, 0xDEADBEEF);
        stable("gemv-i16", &|| {
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
            assert_eq!(rc, 0, "gemv-i16: kernel returned {rc} (expected 0)");
            canonical_hash_i32s(&y)
        });
    }

    // FUSED Q16.16 GEMM outer-product microkernel (gemm-q16-fused) — consumes the
    // UN-transposed B (second return of make_gemm_q16).
    {
        let gemmqmm: Symbol<GemmFn> = unsafe { lib.get(b"gemmqmm").expect("gemmqmm symbol") };
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (a, b, _bt) = make_gemm_q16(m, k, n, 0xDEADBEEF);
        stable("gemm-q16-fused", &|| {
            let mut c = vec![0i32; m * n];
            let rc = unsafe {
                gemmqmm(
                    a.as_ptr() as i64,
                    b.as_ptr() as i64,
                    c.as_mut_ptr() as i64,
                    m as i64,
                    k as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0, "gemm-q16-fused: kernel returned {rc} (expected 0)");
            canonical_hash_i32s(&c)
        });
    }

    // MULTITHREADED int8 GEMM (gemm-i8-mt) — the RANK 4 thread-band kernel. This
    // is the canary MOST likely to expose run-to-run non-determinism (a data race
    // across worker threads would surface as a run-to-run hash difference here),
    // so re-running it N times in-process is load-bearing.
    {
        let gemmi8mt: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8mt").expect("gemmi8mt symbol") };
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (a, b) = make_gemm_i8(m, k, n, 0xDEADBEEF);
        stable("gemm-i8-mt", &|| {
            let mut c = vec![0i32; m * n];
            let rc = unsafe {
                gemmi8mt(
                    a.as_ptr() as i64,
                    b.as_ptr() as i64,
                    c.as_mut_ptr() as i64,
                    m as i64,
                    k as i64,
                    n as i64,
                )
            };
            assert_eq!(rc, 0, "gemm-i8-mt: kernel returned {rc} (expected 0)");
            canonical_hash_i32s(&c)
        });
    }

    // Scalar Q16.16 arithmetic chain (q16-arith-chain).
    {
        let f: Symbol<Arith3Fn> =
            unsafe { lib.get(b"q16_arith_chain").expect("q16_arith_chain symbol") };
        let (a, b, c) = Q16_ARITH_INPUTS;
        stable("q16-arith-chain", &|| canonical_hash(unsafe { f(a, b, c) }));
    }

    // Struct-by-handle round-trip (struct-handle-roundtrip).
    {
        let f: Symbol<Arith4Fn> = unsafe {
            lib.get(b"struct_handle_roundtrip")
                .expect("struct_handle_roundtrip symbol")
        };
        let (a, b, c, d) = STRUCT_HANDLE_INPUTS;
        stable("struct-handle-roundtrip", &|| {
            canonical_hash(unsafe { f(a, b, c, d) })
        });
    }

    // Scalar IEEE-754 f64 chain (scalar-float-f64).
    {
        let f: Symbol<ScalarF64Fn> = unsafe {
            lib.get(b"scalar_f64_chain")
                .expect("scalar_f64_chain symbol")
        };
        let (a, b, c, d) = SCALAR_F64_INPUTS;
        stable("scalar-float-f64", &|| {
            canonical_hash(unsafe { f(a, b, c, d) }.to_bits() as i64)
        });
    }

    // Scalar int↔float `as`-cast conversion (scalar-cast-conv).
    {
        let f: Symbol<ScalarCastFn> = unsafe {
            lib.get(b"scalar_cast_conv")
                .expect("scalar_cast_conv symbol")
        };
        let (inr, povf, novf, nan, pinf, ninf, n) = SCALAR_CAST_INPUTS;
        stable("scalar-cast-conv", &|| {
            canonical_hash(unsafe { f(inr, povf, novf, nan, pinf, ninf, n) })
        });
    }

    // Strict-FP f32 dot (dot-f32-v) — RANK 7.
    {
        let dot: Symbol<DotFn> = unsafe { lib.get(b"dotf32").expect("dotf32 symbol") };
        let (a, b) = make_pair_f32(4093, 0xDEADBEEF);
        stable("dot-f32-v", &|| {
            let packed = unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, a.len() as i64) };
            canonical_hash(f32_from_packed(packed).to_bits() as i64)
        });
    }

    // Strict-FP f32 matmul buffer path (matmul-f32-v) — RANK 7.
    {
        let mm: Symbol<MatmulFn> = unsafe { lib.get(b"mmf32").expect("mmf32 symbol") };
        let (rows, cols) = (64usize, 64usize);
        let (w, x) = make_matvec_f32(rows, cols, 0xDEADBEEF);
        stable("matmul-f32-v", &|| {
            let mut y = vec![0.0f32; rows];
            let rc = unsafe {
                mm(
                    w.as_ptr() as i64,
                    x.as_ptr() as i64,
                    y.as_mut_ptr() as i64,
                    rows as i64,
                    cols as i64,
                )
            };
            assert_eq!(rc, 0, "matmul-f32-v: kernel returned {rc} (expected 0)");
            let mut h = Sha256::new();
            for v in &y {
                h.update(v.to_bits().to_le_bytes());
            }
            format!("{:x}", h.finalize())
        });
    }

    // Q16.16 Lorenz attractor (lorenz-q16) — lives in a SEPARATE .so (examples/
    // tree), so load it independently. Fresh 3-cell state buffer per run.
    if let Some(lso) = build_lorenz_so() {
        let llib = unsafe { Library::new(lso).expect("dlopen lorenz workload .so") };
        let lorenz: Symbol<LorenzFn> =
            unsafe { llib.get(b"lorenz_q16").expect("lorenz_q16 symbol") };
        let (x0, y0, z0) = LORENZ_INIT;
        stable("lorenz-q16", &|| {
            let mut state: [i64; 3] = [x0, y0, z0];
            let r = unsafe { lorenz(state.as_mut_ptr() as i64, LORENZ_STEPS) };
            canonical_hash(r)
        });
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

// --- gemm-i8-mt workload (MULTITHREADED int8 GEMM thread-band reduction) ----
// RANK 4. The multithreaded fused int8 GEMM __mind_blas_matmul_mm_i8_mt_v splits
// the M output rows into contiguous owner-computes bands, one per POSIX thread
// (T = clamp(online-CPUs, 1, M)), each running the SAME BLIS-blocked int8
// macro-kernel over its band. There is NO cross-thread reduction, NO atomic, NO
// shared accumulator — every output element is written by exactly one thread —
// so because integer add is associative + commutative the thread-band partition
// is byte-for-byte identical to the single-thread `gemmi8` kernel REGARDLESS of
// the runtime thread count. This canary PROVES that: it runs the MT kernel on a
// genuinely multi-core host (T > 1 → real threads), cross-checks its output
// buffer against BOTH the single-thread kernel AND the scalar int32 oracle
// within the run, and pins the canonical hash to the committed single-thread
// gemm-i8 reference (917d353b…) — MT output hash == ST output hash is the
// invariant. avx2 == neon holds for the same reason it does single-thread.
//
// Thread-count coverage: T is read at runtime from sysconf(_SC_NPROCESSORS_ONLN)
// and is NOT a compile-time MIND setting, so the canary exercises the real host
// T (here nproc = the CI/dev box's core count, > 1). The output is T-invariant
// by owner-computes construction; the same_process_run_to_run_determinism gate
// additionally re-runs this kernel N times in-process and asserts a stable hash.
// deferred: forcing a *specific* T (e.g. T=1 vs T=64) would require launching
// the kernel in child processes under a restricted cpuset/`taskset` online mask
// (sysconf reads online CPUs, not the affinity mask) — upgrade path: a
// subprocess xnode driver invoked with varied `--cpu-online` cgroups asserting
// the identical hash under each. The in-process MT==ST buffer equality below
// already proves band-partition invariance for the host's real T.

/// The committed single-thread gemm-i8 reference hash (RFC 0015 §3.1). The
/// multithreaded kernel MUST reproduce this exact hash — that equality IS the
/// thread-band-reduction byte-identity claim.
const GEMM_I8_ST_REF: &str = "917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7";

#[test]
fn gemm_i8_mt_reproducibility_gate() {
    let id = "gemm-i8-mt-64x64x64";
    let (m, k, n, seed) = (64usize, 64usize, 64usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let gemmi8mt: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8mt").expect("gemmi8mt symbol") };
    let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol") };

    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("{id}: host online parallelism = {ncpu} (T = clamp(ncpu, 1, {m}))");

    let (a, b) = make_gemm_i8(m, k, n, seed);

    // Multithreaded kernel → fresh int32 output buffer.
    let mut c_mt = vec![0i32; m * n];
    let rc = unsafe {
        gemmi8mt(
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            c_mt.as_mut_ptr() as i64,
            m as i64,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: MT kernel returned {rc} (expected 0)");

    // 1a. Within-run exactness vs the SINGLE-THREAD kernel: the MT thread-band
    //     partition must be byte-for-byte identical to the sequential kernel.
    let mut c_st = vec![0i32; m * n];
    let rc_st = unsafe {
        gemmi8(
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            c_st.as_mut_ptr() as i64,
            m as i64,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc_st, 0, "{id}: ST kernel returned {rc_st} (expected 0)");
    assert_eq!(
        c_mt, c_st,
        "{id}: multithreaded int8 GEMM diverged from the single-thread kernel \
         (thread-band reduction is NOT byte-identical — associativity violated)"
    );

    // 1b. Within-run exactness vs the independent scalar int32 oracle.
    let oracle = ref_gemm_i8_scalar(&a, &b, m, k, n);
    assert_eq!(
        c_mt, oracle,
        "{id}: multithreaded int8 GEMM diverged from the scalar int32 oracle"
    );

    // 2. Canonical hash pinned to the committed reference. It MUST equal the
    //    single-thread gemm-i8 hash — the MT and ST kernels are byte-identical.
    let computed = canonical_hash_i32s(&c_mt);
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    // The MT hash must equal the single-thread reference by construction.
    assert_eq!(
        computed, GEMM_I8_ST_REF,
        "{id} [{substrate}]: multithreaded int8 GEMM hash != the committed \
         single-thread gemm-i8 reference (917d353b…).\ncomputed={computed}\n\
         A drift here means the thread-band partition changed the output — STOP: \
         this is a determinism break, NOT a re-bless (mind-det-gemm owns the fix)."
    );
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

// --- gemm-i8-vnni workload (VPDPBUSD int-dot rung, VNNI hardware only) -------
// RANK 3. The VNNI int-dot rung compiles the int8 GEMM with MIND_INTDOT=vnni so
// the kernel emits the explicit @llvm.x86.avx512.vpdpbusd.256 intrinsic (with
// the signed-input bias correction Σ aₛ·bₛ = Σ(aₛ+128)·bₛ − 128·Σ bₛ, all exact
// i32). By that identity the VPDPBUSD rung is byte-for-byte identical to the
// AVX2 vpmaddwd default, so its output hash MUST equal the committed gemm-i8
// reference (917d353b…). BUT the compiled kernel needs VNNI HARDWARE (Ice Lake /
// Sapphire Rapids+, or an AVX-VNNI part) to RUN — on a non-VNNI host it would
// SIGILL. This canary therefore GATES on runtime VNNI capability and DEFERS
// LOUDLY (prints a skip reason and returns — never a stub-green) when the host
// lacks avx512vnni, exactly as the constitution requires for a rung this
// hardware cannot execute. It must land BEFORE any VNNI auto-select so the wedge
// is guarded the moment the rung can fire. deferred: this box is Haswell (AVX2,
// no VNNI) — upgrade path: run on an avx512vnni host (or CI runner) where the
// gate compiles with MIND_INTDOT=vnni, executes vpdpbusd, and asserts the hash
// equals 917d353b… byte-for-byte.

/// Build SRC with `MIND_INTDOT=vnni` into a DISTINCT `.so`, so the int8 kernel
/// emits the VPDPBUSD rung (and clang links the VNNI target features). Returns
/// `None` on toolchain self-skip. Only called after a VNNI-capability gate.
fn build_dot_so_vnni() -> Option<PathBuf> {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            assert!(
                std::env::var_os("MIND_BENCH_REQUIRE").is_none(),
                "MIND_BENCH_REQUIRE is set but '{tool}' is not on PATH: the \
                 cross-substrate gate cannot run."
            );
            println!("cross_substrate_identity: {tool} not on PATH; skipping");
            return None;
        }
    }
    let dir = std::env::temp_dir();
    let src_path = dir.join("mind_xsi_dot_q16_vnni.mind");
    let so_path = dir.join("mind_xsi_dot_q16_vnni.so");
    std::fs::write(&src_path, SRC).expect("write vnni workload .mind source");
    let status = Command::new(mindc_bin())
        .env("MIND_INTDOT", "vnni")
        .args([
            src_path.to_str().unwrap(),
            "--emit-shared",
            so_path.to_str().unwrap(),
        ])
        .status()
        .expect("spawn mindc --emit-shared (vnni)");
    assert!(
        status.success(),
        "mindc --emit-shared failed for the VNNI int8 workload"
    );
    Some(so_path)
}

/// Runtime VNNI capability of the host. AVX-512-VNNI is the rung the build
/// enables (`-mavx512vnni -mavx512vl`); anything else is a DEFER.
fn host_has_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[test]
fn gemm_i8_vnni_reproducibility_gate() {
    let id = "gemm-i8-vnni-64x64x64";
    let (m, k, n, seed) = (64usize, 64usize, 64usize, 0xDEADBEEFu64);

    // The vpdpbusd rung EXECUTES a lowering whose byte-identity to 917d353b has
    // so far only been ARGUED (the exact signed-bias identity), never MEASURED on
    // real VNNI silicon. GitHub's default runners are a mix (Intel Ice Lake HAS
    // avx512vnni and would execute + hard-assert this; AMD Milan has no AVX-512
    // and defers), so letting default CI be the first place an unmeasured path
    // gates the build breaks "verify green before push / no unmeasured assertion".
    // The rung therefore DEFERS unless BOTH the host has VNNI AND the explicit
    // opt-in MIND_INTDOT_VNNI_VERIFY=1 is set — the flag we set only on a VNNI
    // host where we intend to bless it.
    // deferred: bless on a gcloud Sapphire Rapids / Ice Lake instance with
    // MIND_INTDOT_VNNI_VERIFY=1, confirm the executed hash == 917d353b byte-for-
    // byte, then make the assertion unconditional on any VNNI host.
    if !host_has_vnni() || std::env::var_os("MIND_INTDOT_VNNI_VERIFY").is_none() {
        let why = if host_has_vnni() {
            "VNNI present, but executing the vpdpbusd rung is opt-in (set \
             MIND_INTDOT_VNNI_VERIFY=1 to compile + run + assert it) pending a first \
             bless on real VNNI hardware"
        } else {
            "host lacks AVX-512-VNNI (vpdpbusd) — the rung cannot RUN here"
        };
        // DEFER LOUDLY — never a stub-green.
        println!(
            "DEFER {id}: {why}. The rung is byte-identical to the committed gemm-i8 \
             hash (917d353b…) by the signed-bias identity, but MUST NOT be reported \
             as passing until measured on VNNI silicon. Honest deferral, not a skip."
        );
        return;
    }

    // VNNI-capable host: compile the VPDPBUSD rung and prove it is byte-identical.
    let Some(so) = build_dot_so_vnni() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen vnni workload .so") };
    let gemmi8: Symbol<GemmFn> = unsafe { lib.get(b"gemmi8").expect("gemmi8 symbol (vnni)") };

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
    assert_eq!(rc, 0, "{id}: VNNI kernel returned {rc} (expected 0)");

    let oracle = ref_gemm_i8_scalar(&a, &b, m, k, n);
    assert_eq!(
        c, oracle,
        "{id}: VNNI (vpdpbusd) int8 GEMM diverged from the scalar int32 oracle"
    );

    let computed = canonical_hash_i32s(&c);
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    // The VNNI rung must reproduce the committed AVX2 gemm-i8 hash byte-for-byte.
    assert_eq!(
        computed, GEMM_I8_ST_REF,
        "{id} [{substrate}]: VNNI vpdpbusd int8 GEMM hash != the committed AVX2 \
         gemm-i8 reference (917d353b…).\ncomputed={computed}\n\
         A drift here means the signed-bias VPDPBUSD lowering is not exact — STOP: \
         hand to mind-det-gemm (likely a dropped −128·Σb term or a saturating sibling)."
    );
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: drift from committed reference."
        ),
        None => panic!("{id}: no reference hash for substrate '{substrate}'. Computed {computed}."),
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

// --- scalar-cast-conv workload (scalar int↔float `as`-cast conversion) ------
// The ONLY cross-substrate canary that exercises the scalar conversion
// lowering. It closes the coverage hole that let a real float→int wedge break
// ship (fixed in the saturating-cast commits): NO fixture ran a scalar cast, so
// the byte-identity gate never tested the conversion. A bare `arith.fptosi` is
// target-defined out of range — x86 `cvttsd2si` yields INT64_MIN for every
// out-of-range/NaN input while ARM `fcvtzs` SATURATES — so `1e30 as i64` /
// `inf as i64` / `nan as i64` produced DIFFERENT bytes on avx2 vs neon until the
// fully-IEEE-defined saturating clamp (`emit_saturating_fp_to_i64`).
//
// The kernel `scalar_cast_conv(inr,povf,novf,nan,pinf,ninf,n)` takes the edge
// operands as RUNTIME f64 arguments (never in-kernel literals), so mlir-opt
// cannot constant-fold the conversion away — the saturating path provably runs.
// It exercises int→f64 + int→f32 (sitofp), bool→f64 (i1→uitofp → 1.0), and the
// float→i64 SATURATING edges (in-range 9.7→9, +ovf 1e30→INT64_MAX, −ovf
// −1e30→INT64_MIN, NaN→0, +inf→INT64_MAX, −inf→INT64_MIN), folding all nine cast
// results in fixed source order into one i64 via a wrapping polynomial. Because
// the saturating result is built only from IEEE-defined ops, avx2 == neon BY
// CONSTRUCTION (RFC 0015 §3.1).

/// Deterministic scalar-cast inputs (manifest `[input]`): the six f64 edge
/// operands + the int→float source `n`. `n = 16777219` (= 2^24 + 3) is NOT
/// exactly representable in f32 (ulp = 2 above 2^24), so `n as f32` rounds to
/// 16777220 while `n as f64` stays 16777219 — the fold therefore distinguishes
/// the int→f32 and int→f64 legs, not merely a round-trip.
const SCALAR_CAST_INPUTS: (f64, f64, f64, f64, f64, f64, i64) = (
    9.7,
    1e30,
    -1e30,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    16_777_219,
);

/// Independent in-process oracle: the identical cast+fold in Rust. Rust `as`
/// float→int is saturating since 1.45 (NaN→0, ±ovf→MIN/MAX, truncate toward
/// zero in range) — the exact semantics `emit_saturating_fp_to_i64` mirrors —
/// so this is bit-exact to the compiled kernel within a run AND, being built
/// only from IEEE-defined ops, identical on every substrate. `bool as f64` is
/// spelled `as i64 as f64` here (Rust forbids the direct `bool as f64` the MIND
/// frontend accepts); the value (1.0) and the final i64 are identical. The fold
/// uses `wrapping_*` to match the two's-complement `arith.muli`/`arith.addi`.
fn ref_scalar_cast_conv(
    inr: f64,
    povf: f64,
    novf: f64,
    nan: f64,
    pinf: f64,
    ninf: f64,
    n: i64,
) -> i64 {
    let e0 = inr as i64;
    let e1 = povf as i64;
    let e2 = novf as i64;
    let e3 = nan as i64;
    let e4 = pinf as i64;
    let e5 = ninf as i64;
    let b0 = (n as f64) as i64;
    let b1 = (n as f32) as i64;
    let b2 = (inr < povf) as i64 as f64 as i64;
    let k: i64 = 1_000_003;
    let mut acc = e0;
    for t in [e1, e2, e3, e4, e5, b0, b1, b2] {
        acc = acc.wrapping_mul(k).wrapping_add(t);
    }
    acc
}

#[test]
fn scalar_cast_conv_reproducibility_gate() {
    let id = "scalar-cast-conv";

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let conv: Symbol<ScalarCastFn> = unsafe {
        lib.get(b"scalar_cast_conv")
            .expect("scalar_cast_conv symbol")
    };

    let (inr, povf, novf, nan, pinf, ninf, n) = SCALAR_CAST_INPUTS;
    let result = unsafe { conv(inr, povf, novf, nan, pinf, ninf, n) };

    // 1. Within-run exactness vs the saturating oracle: the compiled kernel must
    //    reproduce the identical i64 the Rust saturating-cast oracle computes.
    //    This also proves the saturating EDGES fired (INT64_MAX/MIN/0), not a
    //    trivial in-range-only path — the oracle bakes in every edge value.
    let oracle = ref_scalar_cast_conv(inr, povf, novf, nan, pinf, ninf, n);
    assert_eq!(
        result, oracle,
        "{id}: scalar-cast kernel diverged from the saturating IEEE oracle within a \
         single run (kernel={result}, oracle={oracle})"
    );

    // 2. Canonical hash of the folded i64, pinned to the committed per-substrate
    //    reference. avx2 == neon by IEEE construction (RFC 0015 §3.1).
    let computed = canonical_hash(result);
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result={result}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13).",
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed} \
             (result={result}); bless with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
}

/// Rust saturating float→narrow oracle for `scalar_cast_conv_narrow`. Rust `f as
/// iN`/`uN` has been SATURATING (NaN→0, ±ovf→iN::MIN/MAX) since 1.45 — exactly
/// the MIND `emit_saturating_fp_to_narrow` contract — so this is bit-exact to the
/// compiled kernel within a run AND identical on every substrate (IEEE-defined
/// ops only). The fold uses `wrapping_*` to match `arith.muli`/`arith.addi`.
fn ref_scalar_cast_conv_narrow(
    inr: f64,
    povf: f64,
    novf: f64,
    nan: f64,
    pinf: f64,
    ninf: f64,
) -> i64 {
    let e0 = (inr as i8) as i64;
    let e1 = (povf as i8) as i64;
    let e2 = (novf as u8) as i64;
    let e3 = (nan as i16) as i64;
    let e4 = (pinf as i32) as i64;
    let e5 = (ninf as u16) as i64;
    let e6 = (inr as u32) as i64;
    let e7 = (povf as i16) as i64;
    let e8 = (novf as i32) as i64;
    let k: i64 = 1_000_003;
    let mut acc = e0;
    for t in [e1, e2, e3, e4, e5, e6, e7, e8] {
        acc = acc.wrapping_mul(k).wrapping_add(t);
    }
    acc
}

#[test]
fn scalar_cast_conv_narrow_reproducibility_gate() {
    let id = "scalar-cast-conv-narrow";

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let conv: Symbol<ScalarCastNarrowFn> = unsafe {
        lib.get(b"scalar_cast_conv_narrow")
            .expect("scalar_cast_conv_narrow symbol")
    };

    let (inr, povf, novf, nan, pinf, ninf, _n) = SCALAR_CAST_INPUTS;
    let result = unsafe { conv(inr, povf, novf, nan, pinf, ninf) };

    // 1. Within-run exactness vs the saturating-to-narrow oracle: proves the
    //    compiled kernel reproduces Rust `f as iN`/`uN` bit-for-bit, and that the
    //    narrow saturation EDGES fired (i8::MAX/i16::MAX/i32::MIN/0), not a
    //    trivial in-range path.
    let oracle = ref_scalar_cast_conv_narrow(inr, povf, novf, nan, pinf, ninf);
    assert_eq!(
        result, oracle,
        "{id}: float→narrow kernel diverged from the saturating IEEE oracle within a \
         single run (kernel={result}, oracle={oracle})"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference.
    let computed = canonical_hash(result);
    pin_or_bless(id, &computed, result);
}

// ===========================================================================
// Track #16 — broaden the cross-substrate canary set (determinism hardening).
//
// Four NEW byte-identity canaries over determinism-sensitive paths the gates
// above do not exercise. Each is a real compiled kernel whose output is pinned
// to a committed per-substrate reference (avx2 == neon, RFC 0015 §3.1) and
// cross-checked against an independent in-process oracle within the run. The
// f32 reductions are deliberately excluded — they use tree-shaped summation
// that reorders, so they are NOT bit-exact and belong in the approximate
// surface (RFC 0020 §8), never as a byte-identity reference (README §"Why only
// Q16.16 integer workloads").
// ===========================================================================

/// Shared hash-pin / bless step for a scalar-i64-result canary. Pins `computed`
/// to the committed per-substrate reference, or prints a `BLESS` line under
/// MIND_BENCH_BLESS. Same contract as the per-test inline blocks above.
fn pin_or_bless(id: &str, computed: &str, result: i64) {
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result_i64={result}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13)."
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed} \
             (result_i64={result}); bless with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
}

// --- dot-i16 workload (bare int16 dot reduction) ---------------------------
// The `int-dot` tier's raw reduction intrinsic __mind_blas_dot_i16_v, exercised
// directly (the gemv-i16 canary only reaches it through the matrix wrapper).
// y = Σ_i (i32 sum) sext(w[i])*sext(x[i]), accumulated in i64, narrowed once to
// i32. Exact integer add is associative, so the vectorised reduction is
// grouping-/substrate-independent (RFC 0015 §3.1). Scalar i64 output.

/// Regenerate a single int16 vector pair from a seed (a before b), full i16
/// range from the shared LCG window — same generator as gemv-i16.
fn make_pair_i16(len: usize, seed: u64) -> (Vec<i16>, Vec<i16>) {
    let mut g = Lcg::new(seed);
    let next_i16 = |g: &mut Lcg| (g.next_u32() >> 16) as i16;
    let a: Vec<i16> = (0..len).map(|_| next_i16(&mut g)).collect();
    let b: Vec<i16> = (0..len).map(|_| next_i16(&mut g)).collect();
    (a, b)
}

#[test]
fn dot_i16_reproducibility_gate() {
    let id = "dot-i16-4096";
    let (len, seed) = (4096usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let dot: Symbol<DotFn> = unsafe { lib.get(b"doti16").expect("doti16 symbol") };

    let (a, b) = make_pair_i16(len, seed);
    let vec_result = unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, len as i64) };

    // Within-run exactness vs the scalar int16 dot oracle (already in this file).
    let oracle = ref_dot_i16_scalar(&a, &b) as i64;
    assert_eq!(
        vec_result, oracle,
        "{id}: int16 dot vector path diverged from the scalar oracle within a run"
    );

    pin_or_bless(id, &canonical_hash(vec_result), vec_result);
}

// --- gemm-q16-fused workload (outer-product microkernel) -------------------
// FUSED Q16.16 GEMM via __mind_blas_matmul_mm_q16_v (A M×K, B K×N UN-transposed,
// C M×N), a DISTINCT lowering from the gemv-composed `gemmq` gate (register-tiled
// outer product, no horizontal reduction). Per-product `>> 16` then i64
// accumulate, byte-identical to Σ_k (A[i,k]*B[k,j])>>16. M=K=N=64. The output is
// the M×N Q16.16 matrix; canonical encoding is its i32 LE bytes → sha256.

/// Scalar Q16.16 GEMM oracle over the UN-transposed B (B[k*n+j]), matching the
/// fused intrinsic's ABI: C[i,j] = Σ_k (A[i,k]*B[k,j]) >> 16, each product
/// shifted before the associative i64 accumulate (exact within a run).
fn ref_gemm_q16_mm_scalar(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
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

#[test]
fn gemm_q16_fused_reproducibility_gate() {
    let id = "gemm-q16-fused-64x64x64";
    let (m, k, n, seed) = (64usize, 64usize, 64usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let gemmqmm: Symbol<GemmFn> = unsafe { lib.get(b"gemmqmm").expect("gemmqmm symbol") };

    // The fused intrinsic consumes the UN-transposed B (K×N); reuse the gemm-q16
    // generator and take its B (the second return), not Bᵀ.
    let (a, b, _bt) = make_gemm_q16(m, k, n, seed);
    let mut c = vec![0i32; m * n];
    let rc = unsafe {
        gemmqmm(
            a.as_ptr() as i64,
            b.as_ptr() as i64,
            c.as_mut_ptr() as i64,
            m as i64,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: kernel returned {rc} (expected 0)");

    let oracle = ref_gemm_q16_mm_scalar(&a, &b, m, k, n);
    assert_eq!(
        c, oracle,
        "{id}: fused q16 gemm vector path diverged from the scalar oracle"
    );

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

// --- q16-arith-chain workload (scalar Q16.16 fixed-point arithmetic) -------
// A scalar Q16.16 arithmetic chain `((a*b)+(b*c))-(a*c)` where each `*` is a
// Q16.16 product `(x*y) >> 16` — exercises mindc's per-element fixed-point
// lowering (i64 multiply, arithmetic shift-right, add/sub) with NO intrinsic and
// NO reduction. Each op is exact integer arithmetic with a single deterministic
// truncating shift per product, so the result is byte-identical across
// substrates (RFC 0015 §3.1). Scalar i64 output → canonical_hash.

/// Independent in-process oracle for the Q16.16 arithmetic chain — the identical
/// fixed-point ops in the identical source order, bit-exact within a run and
/// (being exact integer arithmetic with fixed truncation points) the same on
/// every substrate.
fn ref_q16_arith_chain(a: i64, b: i64, c: i64) -> i64 {
    let q16_mul = |x: i64, y: i64| (x * y) >> 16;
    let ab = q16_mul(a, b);
    let bc = q16_mul(b, c);
    let ac = q16_mul(a, c);
    (ab + bc) - ac
}

/// Three deterministic Q16.16 inputs (manifest `[input]`): 1.5, 2.25, 3.125 in
/// Q16.16 (value << 16). Chosen exact-representable so the oracle is unambiguous.
const Q16_ARITH_INPUTS: (i64, i64, i64) = (
    (1.5 * 65536.0) as i64,   // 98304
    (2.25 * 65536.0) as i64,  // 147456
    (3.125 * 65536.0) as i64, // 204800
);

#[test]
fn q16_arith_chain_reproducibility_gate() {
    let id = "q16-arith-chain";

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let chain: Symbol<Arith3Fn> =
        unsafe { lib.get(b"q16_arith_chain").expect("q16_arith_chain symbol") };

    let (a, b, c) = Q16_ARITH_INPUTS;
    let result = unsafe { chain(a, b, c) };

    let oracle = ref_q16_arith_chain(a, b, c);
    assert_eq!(
        result, oracle,
        "{id}: q16 arithmetic chain diverged from the in-process oracle within a run \
         (kernel={result}, oracle={oracle})"
    );

    pin_or_bless(id, &canonical_hash(result), result);
}

// --- struct-handle-roundtrip workload (alloc/store/load handle ABI) --------
// Allocate a 4-field i64 record via __mind_alloc, store the inputs through
// __mind_store_i64, read them back through __mind_load_i64, and combine them.
// Exercises the struct-by-handle ABI (heap handle + address arithmetic +
// store/load round-trip) — deterministic data movement, no reduction, no float.
// The result is a fixed integer function of the inputs, byte-identical across
// substrates by construction. Scalar i64 output → canonical_hash.

/// Independent oracle for the struct-by-handle round-trip: the round-trip is the
/// identity, so the result is the fixed combination of the inputs.
fn ref_struct_handle_roundtrip(a: i64, b: i64, c: i64, d: i64) -> i64 {
    ((a + b) * 2 - c) + d * 3
}

/// Four deterministic inputs (manifest `[input]`): fixed integers, no LCG.
const STRUCT_HANDLE_INPUTS: (i64, i64, i64, i64) = (1111, 2222, 3333, 4444);

#[test]
fn struct_handle_roundtrip_reproducibility_gate() {
    let id = "struct-handle-roundtrip";

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let f: Symbol<Arith4Fn> = unsafe {
        lib.get(b"struct_handle_roundtrip")
            .expect("struct_handle_roundtrip symbol")
    };

    let (a, b, c, d) = STRUCT_HANDLE_INPUTS;
    let result = unsafe { f(a, b, c, d) };

    let oracle = ref_struct_handle_roundtrip(a, b, c, d);
    assert_eq!(
        result, oracle,
        "{id}: struct-handle round-trip diverged from the in-process oracle within a run \
         (kernel={result}, oracle={oracle})"
    );

    pin_or_bless(id, &canonical_hash(result), result);
}

// --- lorenz-q16 workload (deterministic Q16.16 Lorenz attractor) -----------
// A forward-Euler Lorenz integrator in Q16.16 — the textbook poster child for
// sensitive dependence on initial conditions, made byte-identical across
// substrates. Each derivative/update product is one i64 multiply + one
// arithmetic shift-right (>>16), then integer add/sub, in fixed source order.
// No float, no reduction reorder, no reassociation — so after N chaotic steps
// the final state is byte-identical on avx2 and neon BY CONSTRUCTION (RFC 0015
// §3.1), even though a float Lorenz would have diverged to a different
// trajectory on different hardware hundreds of steps earlier. This is the wedge
// on its hardest workload: reproducible chaos. (This is *a* Q16.16 chaotic
// orbit, not a bit-match of float64's Lorenz — cross-substrate reproducibility,
// not IEEE-754 parity.) Buffer ABI + scalar i64 return → canonical_hash.

/// Q16.16 system constants — the identical values baked into
/// examples/lorenz_q16.mind (SIGMA=10, RHO=28, BETA=8/3 truncated, DT=1/256).
const LORENZ_SIGMA: i64 = 655360; // 10.0
const LORENZ_RHO: i64 = 1835008; // 28.0
const LORENZ_BETA: i64 = 174762; // 8/3 truncated -> 2.666656494140625
const LORENZ_DT: i64 = 256; // 1/256 = 0.00390625

/// Fixed initial state (Q16.16) and step count — the manifest `[input]`.
const LORENZ_INIT: (i64, i64, i64) = (0, 65536, 0); // (0.0, 1.0, 0.0)
const LORENZ_STEPS: i64 = 1000;

/// Independent in-process oracle — the identical fixed-point ops in the
/// identical source order as the .mind kernel. Exact integer arithmetic with
/// one truncating shift per product, so it is bit-exact within a run and (being
/// float-free with fixed truncation points) identical on every substrate.
/// Returns the final x (Q16.16), matching the kernel's return + state[0].
fn ref_lorenz_q16(init: (i64, i64, i64), steps: i64) -> i64 {
    let q16_mul = |a: i64, b: i64| (a * b) >> 16;
    let (mut x, mut y, mut z) = init;
    let mut s = 0i64;
    while s < steps {
        let dx = q16_mul(LORENZ_SIGMA, y - x);
        let dy = q16_mul(x, LORENZ_RHO - z) - y;
        let dz = q16_mul(x, y) - q16_mul(LORENZ_BETA, z);
        x += q16_mul(LORENZ_DT, dx);
        y += q16_mul(LORENZ_DT, dy);
        z += q16_mul(LORENZ_DT, dz);
        s += 1;
    }
    x
}

#[test]
fn lorenz_q16_reproducibility_gate() {
    let id = "lorenz-q16";

    let Some(so) = build_lorenz_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen lorenz workload .so") };
    let lorenz: Symbol<LorenzFn> = unsafe { lib.get(b"lorenz_q16").expect("lorenz_q16 symbol") };

    // The buffer ABI: 3 consecutive i64 Q16.16 cells [x, y, z], mutated in place.
    let (x0, y0, z0) = LORENZ_INIT;
    let mut state: [i64; 3] = [x0, y0, z0];
    let result = unsafe { lorenz(state.as_mut_ptr() as i64, LORENZ_STEPS) };

    // Return value must equal the final x written back to the buffer.
    assert_eq!(
        result, state[0],
        "{id}: kernel return ({result}) != final state[0] ({}) — buffer/return \
         disagreement",
        state[0]
    );

    // ...and both must match the independent in-process oracle.
    let oracle = ref_lorenz_q16(LORENZ_INIT, LORENZ_STEPS);
    assert_eq!(
        result, oracle,
        "{id}: Lorenz orbit diverged from the in-process oracle within a run \
         (kernel={result}, oracle={oracle})"
    );

    pin_or_bless(id, &canonical_hash(result), result);
}

// ===========================================================================
// RANK 7 — STRICT-FP f32 vector canaries (dot_f32_v + matmul_rmajor_f32_v).
//
// A NEW bit-identity class: as of the strict-FP tier the f32 `_v` kernels have
// their FMA UNFUSED (separate mulf + addf) and their horizontal reduction
// replaced by a PINNED fixed-order left-to-right lane fold, so f32 dot/matmul
// are bit-exact — no 1e-4 tolerance. Because scalar IEEE mul/add are
// round-to-nearest-even with a single fully-specified result and the fold ORDER
// is now fixed (not a target-defined `vector.reduction` tree), the value is a
// candidate for cross-substrate byte-identity — BUT unlike the integer tiers
// this rests on strict-FP lowering, so avx2 == neon MUST be blessed on REAL
// aarch64, never asserted from x86 + the associativity argument. This host is
// x86_64 (avx2): the avx2 line is blessed here; the neon line is DEFERRED
// pending a real-aarch64 bless (RFC 0020 §13) and reported LOUDLY, never a
// stub-green. The within-run oracle below mirrors the kernel's exact fold
// (8-lane accumulator, fixed 0..8 horizontal fold, scalar tail) so it is
// bit-exact to the kernel on this substrate.
//
// deferred: neon cross-substrate identity for the strict-f32 tier is UNPROVEN
// here — upgrade path: run this gate on a real aarch64 host with
// MIND_BENCH_BLESS=1 and commit the neon line only if it reproduces the avx2
// hash byte-for-byte (a divergence would be a real wedge break, not a bless).
// ===========================================================================

/// Number of f32 lanes the kernel accumulates over (matches VEC_DOT_F32_LANES).
const F32_LANES: usize = 8;

/// Unpack the Option-C f32 result: the kernel returns the f32 bits in the low
/// 32 bits of an i64.
fn f32_from_packed(bits_i64: i64) -> f32 {
    f32::from_bits((bits_i64 as u64) as u32)
}

/// One deterministic f32 sample in [-1, 1) from the shared LCG window — the
/// SAME construction as `blas_vec_q16_smoke.rs::next_f32_unit`, so the numeric
/// distribution is shared and reproducible.
fn next_f32_unit(g: &mut Lcg) -> f32 {
    ((g.next_u32() as f32) / (u32::MAX as f32)) * 2.0 - 1.0
}

/// Regenerate an f32 vector pair from a seed (a before b), each in [-1, 1).
fn make_pair_f32(len: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut g = Lcg::new(seed);
    let a: Vec<f32> = (0..len).map(|_| next_f32_unit(&mut g)).collect();
    let b: Vec<f32> = (0..len).map(|_| next_f32_unit(&mut g)).collect();
    (a, b)
}

/// Regenerate the f32 matmul inputs: a rows*cols row-major matrix W and a
/// cols-length vector x, W generated before x (order is part of the seed
/// contract).
fn make_matvec_f32(rows: usize, cols: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut g = Lcg::new(seed);
    let w: Vec<f32> = (0..rows * cols).map(|_| next_f32_unit(&mut g)).collect();
    let x: Vec<f32> = (0..cols).map(|_| next_f32_unit(&mut g)).collect();
    (w, x)
}

/// Independent in-process oracle mirroring the kernel's EXACT strict-FP fold:
/// an 8-lane accumulator (mulf then addf, `acc + a*b` per lane), a PINNED
/// left-to-right horizontal fold over lanes 0..8, then a scalar tail. Rust f32
/// `*`/`+` are strict IEEE round-to-nearest with NO auto-FMA, so this is
/// bit-exact to the unfused kernel on a substrate whose FPU is IEEE-conformant.
fn ref_dot_f32_strict(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let ve = (n / F32_LANES) * F32_LANES;
    let mut acc = [0.0f32; F32_LANES];
    let mut i = 0;
    while i < ve {
        for (lane, slot) in acc.iter_mut().enumerate() {
            *slot += a[i + lane] * b[i + lane];
        }
        i += F32_LANES;
    }
    // Fixed left-to-right horizontal fold, lane 0 through lane 7.
    let mut hs = acc[0];
    for &lane in acc.iter().skip(1) {
        hs += lane;
    }
    // Scalar tail for the len % LANES remainder.
    let mut s = hs;
    let mut j = ve;
    while j < n {
        s += a[j] * b[j];
        j += 1;
    }
    s
}

/// Pin a strict-FP canary's hash to the committed per-substrate reference, or —
/// when the host substrate has NO committed hash and is `neon` — DEFER LOUDLY
/// (strict-FP cross-substrate identity must be blessed on real aarch64, never
/// asserted from x86). Under MIND_BENCH_BLESS it prints the bless line instead.
fn pin_or_defer_strict_fp(id: &str, computed: &str, bits: u32) {
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        println!("BLESS {id} {substrate} {computed}");
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: strict-f32 output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result_bits={bits:#010x}\n\
             Re-bless with MIND_BENCH_BLESS=1 only on an intentional lowering change (RFC 0020 §13)."
        ),
        None if substrate == "neon" => {
            // Honest hardware/strict-FP deferral — NEVER a stub-green.
            println!(
                "DEFER {id}: no committed neon reference for the strict-f32 tier. \
                 Cross-substrate bit-identity of strict FP must be blessed on REAL \
                 aarch64 (RFC 0020 §13), not asserted from x86. Computed here would \
                 be {computed} (bits={bits:#010x}); bless it ONLY if it reproduces \
                 the avx2 hash byte-for-byte on real ARM hardware."
            );
        }
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}'. Computed {computed} \
             (bits={bits:#010x}); bless with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
}

#[test]
fn dot_f32_v_reproducibility_gate() {
    let id = "dot-f32-v-4093";
    // 4093 = 511*8 + 5 — exercises BOTH the 8-lane main loop AND the scalar tail.
    let (len, seed) = (4093usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let dot: Symbol<DotFn> = unsafe { lib.get(b"dotf32").expect("dotf32 symbol") };

    let (a, b) = make_pair_f32(len, seed);
    let packed = unsafe { dot(a.as_ptr() as i64, b.as_ptr() as i64, len as i64) };
    let result = f32_from_packed(packed);

    // Within-run exactness vs the strict-FP fold oracle (bit-exact, not tolerance).
    let oracle = ref_dot_f32_strict(&a, &b);
    assert_eq!(
        result.to_bits(),
        oracle.to_bits(),
        "{id}: strict-f32 dot diverged from the fixed-order fold oracle within a run \
         (kernel={result} bits={:#010x}, oracle={oracle} bits={:#010x})",
        result.to_bits(),
        oracle.to_bits()
    );

    // Canonical encoding: the f32 bits as an i64, 8 LE bytes → sha256.
    let computed = canonical_hash(result.to_bits() as i64);
    pin_or_defer_strict_fp(id, &computed, result.to_bits());
}

#[test]
fn matmul_f32_v_reproducibility_gate() {
    let id = "matmul-f32-v-64x64";
    let (rows, cols, seed) = (64usize, 64usize, 0xDEADBEEFu64);

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let mm: Symbol<MatmulFn> = unsafe { lib.get(b"mmf32").expect("mmf32 symbol") };

    let (w, x) = make_matvec_f32(rows, cols, seed);
    let mut y = vec![0.0f32; rows];
    let rc = unsafe {
        mm(
            w.as_ptr() as i64,
            x.as_ptr() as i64,
            y.as_mut_ptr() as i64,
            rows as i64,
            cols as i64,
        )
    };
    assert_eq!(rc, 0, "{id}: kernel returned {rc} (expected 0)");

    // Within-run exactness: each row is the SAME strict-FP dot fold.
    let oracle: Vec<f32> = (0..rows)
        .map(|r| ref_dot_f32_strict(&w[r * cols..(r + 1) * cols], &x))
        .collect();
    let y_bits: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
    let o_bits: Vec<u32> = oracle.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        y_bits, o_bits,
        "{id}: strict-f32 matmul diverged from the per-row fixed-order fold oracle"
    );

    // Canonical encoding: each f32 result's bit pattern, LE, → sha256.
    let mut h = Sha256::new();
    for v in &y {
        h.update(v.to_bits().to_le_bytes());
    }
    let computed = format!("{:x}", h.finalize());
    // A stable per-buffer digest head for the DEFER/drift message.
    let head_bits = y.first().map(|v| v.to_bits()).unwrap_or(0);
    pin_or_defer_strict_fp(id, &computed, head_bits);
}
