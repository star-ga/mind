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
use common::xsi_gate::{self, VnniDecision};

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

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
// avx2 AND neon BOTH blessed on real hardware — the neon line was harvested from
// the ubuntu-24.04-arm CI runner (real aarch64) and reproduces the avx2 hash
// byte-for-byte. The gate is fail-closed; see `pin_strict_fp`.
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

// Unsigned `u64` sign-sensitive op canary (u64-ops). Pins issue #99 Stage 2:
// first-class `ScalarU64` unsigned lowering, where `/ % >> < <= > >=` on a u64
// operand select the UNSIGNED MLIR variants (`divui`/`remui`/`shrui`/`ult`/…)
// instead of the signed ones a bare i64 would use. The inputs are u64 values
// with the HIGH BIT SET, passed as runtime args (their i64 bit pattern), so the
// signed-vs-unsigned choice is observable: as signed i64 `a` is negative, so
// `a > b` / `a / b` / `a >> n` would all give the WRONG answer. Unsigned integer
// ops are exact and order-independent, so avx2 == neon by construction
// (RFC 0015 §3.1) — this canary is the regression pin that the u64 arm keeps
// emitting the unsigned variants. The results fold in fixed source order via the
// same wrapping polynomial (K = 1000003).
pub fn u64_ops(a: u64, b: u64) -> i64 {
    let s: u64 = a >> 3 // logical (unsigned) shift
    let d: u64 = a / b // unsigned div
    let m: u64 = a % b // unsigned rem
    let g: i64 = if a > b { 1 } else { 0 } // unsigned compare
    let l: i64 = if a <= b { 1 } else { 0 }
    let k: i64 = 1000003
    let f0: i64 = i64(s)
    let f1: i64 = f0 * k + i64(d)
    let f2: i64 = f1 * k + i64(m)
    let f3: i64 = f2 * k + g
    return f3 * k + l
}
"#;

// F2 aggregate loop-carry canary source (array-store-loop). Kept in its OWN
// translation unit — NOT appended to `SRC` — because `a[k]=v` emits value-tensor
// ops (`tensor.insert`/`tensor.extract` + a `dense<> : tensor<8xi64>` constant),
// which flips `preset_for_mlir` to the tensor-aware `arith-linalg` pipeline for
// the whole module. The shared `SRC` also holds the int-dot kernels that emit
// `vector.reduction`, and that pipeline drops `convert-vector-to-llvm`, so the
// two cannot legalise in one `.so`. This kernel gets its own arith-linalg `.so`.
//
// Pins #320 Step D: a fixed `[i64; 8]` mutated by `a[k] = v` INSIDE a loop body
// is threaded as a value-semantic `tensor<8xi64>` loop iter-arg (typed
// header/body/^while_after block-args), and every post-loop read observes the
// write. `base` is a RUNTIME arg (rdi) so the stores are not constant-folded;
// the eight results fold in fixed source order via a wrapping polynomial
// (K = 1000003). The bufferized loop body carries NO per-iteration malloc/memcpy
// (the iter-arg stays in-place), and integer store/add/mul are exact +
// order-independent, so avx2 == neon by construction (RFC 0015 §3.1). A
// regression that drops a write (the pre-F2 silent-miscompile) or copies per
// iteration flips this hash.
const ARRAY_STORE_SRC: &str = r#"
pub fn array_store_loop(base: i64) -> i64 {
    let mut a: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
    let mut k: i64 = 0
    while k < 8 {
        a[k] = base * k + k * k
        k = k + 1
    }
    let kk: i64 = 1000003
    let f0: i64 = a[0]
    let f1: i64 = f0 * kk + a[1]
    let f2: i64 = f1 * kk + a[2]
    let f3: i64 = f2 * kk + a[3]
    let f4: i64 = f3 * kk + a[4]
    let f5: i64 = f4 * kk + a[5]
    let f6: i64 = f5 * kk + a[6]
    return f6 * kk + a[7]
}
"#;

// F2 aggregate BRANCH-carry canary source (array-store-branch). Own translation
// unit for the same reason as `ARRAY_STORE_SRC` (value-tensor ops force the
// arith-linalg pipeline, incompatible with the vector-reduction kernels in `SRC`).
//
// Pins the #320 Step D If-MERGE path (distinct from the loop's `^while_after`):
// `a[i] = v` inside `if`/`else` branch bodies is threaded as a value-semantic
// `tensor<8xi64>` `^if_after` merge block-arg (then/else edge = stored incarnation,
// untouched edge = pre-if tensor). `base` is a RUNTIME arg so the branches are
// genuinely taken/not-taken (not const-folded), exercising if/else same-slot,
// no-else (one-sided merge), and a straight-line store in the same fn. Integer
// store/add/mul are exact + order-independent, and the post-bufferize
// memref.alloc/copy profile matches the loop baseline (no per-branch copy), so
// avx2 == neon by construction (RFC 0015 §3.1). A dropped/aliased branch store or
// a wrong merge incarnation flips the hash.
const ARRAY_STORE_BRANCH_SRC: &str = r#"
pub fn array_store_branch(base: i64) -> i64 {
    let mut a: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
    if base > 3 { a[0] = base * 7 } else { a[0] = base + 100 }
    a[1] = base * 2
    if base > 100 { a[2] = 999 }
    if base > 5 { a[3] = base * 11 } else { a[3] = base - 50 }
    let kk: i64 = 1000003
    let f0: i64 = a[0]
    let f1: i64 = f0 * kk + a[1]
    let f2: i64 = f1 * kk + a[2]
    let f3: i64 = f2 * kk + a[3]
    let f4: i64 = f3 * kk + a[4]
    let f5: i64 = f4 * kk + a[5]
    let f6: i64 = f5 * kk + a[6]
    return f6 * kk + a[7]
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
/// The unsigned-u64 op kernel: two u64 operands (passed as their i64 bit
/// patterns), one folded i64 out. Pins the first-class ScalarU64 unsigned lowering.
type U64OpsFn = unsafe extern "C" fn(i64, i64) -> i64;
/// The F2 aggregate loop-carry kernel: one i64 `base` in (rdi), one folded i64
/// out. Pins `a[k]=v` inside a loop (#320 Step D value-semantic tensor iter-arg).
type ArrayStoreFn = unsafe extern "C" fn(i64) -> i64;
/// Track #16: a 3-arg → i64 scalar kernel (the Q16.16 arithmetic chain).
type Arith3Fn = unsafe extern "C" fn(i64, i64, i64) -> i64;
/// Track #16: a 4-arg → i64 scalar kernel (the struct-by-handle round-trip).
type Arith4Fn = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
/// The Lorenz integrator: (state_ptr, steps) → final x, buffer mutated in place.
type LorenzFn = unsafe extern "C" fn(i64, i64) -> i64;

// --- examples/quant/* kernel ABIs -------------------------------------------
// Every quant example is a self-contained package (its own Mind.toml, no
// cross-module f64 import — mindc 0.10.2 cannot carry an f64 across a module
// boundary, roadmap 17.1) built to its OWN `.so` via `build_quant_so`, the
// generic sibling of `build_lorenz_so`/`build_collatz_so` for the examples/
// tree. Each pricing kernel below is a pure scalar function (System V xmm/gpr
// ABI, f64 args before/after i64 args exactly as declared): no pointer, no
// buffer, matching the `ScalarF64Fn`/`ScalarCastFn` precedent above.
type QuantF64x6Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64;
type QuantF64x7Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64;
/// bond_price(face, coupon_rate, ytm, freq: i64, n_periods: i64) -> f64.
type QuantBondPriceFn = unsafe extern "C" fn(f64, f64, f64, i64, i64) -> f64;
/// two_asset_vol(w1, w2, s1, s2, rho) -> f64.
type QuantF64x5Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64) -> f64;
/// bopm_american_put(s, k, r, q, sigma, t, n: i64) -> f64.
type QuantF64x6I64Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64) -> f64;
/// crr_price(s, k, r, q, sigma, t, n: i64, is_call: i64, american: i64) -> f64.
type QuantCrrPriceFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64, i64, i64) -> f64;
/// mc_call(s, k, r, q, sigma, t, npath: i64, seed: i64) -> f64.
type QuantMcCallFn = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, i64, i64) -> f64;

// mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

/// The artifact root for this target: private to the test BINARY and to the
/// PROCESS (`common::scratch_dir` appends target + pid). Every path below moves
/// off the world-writable shared temp root, where a second `cargo test`, a
/// preflight, or another agent compiling the same fixed name truncates the
/// `.so` between `--emit-shared` and `dlopen` — a flake that reads as a
/// compiler regression and does not reproduce. The FILE names are unchanged:
/// only the directory moves, so each artifact's identity is preserved.
fn scratch() -> PathBuf {
    crate::common::scratch_dir("cross_substrate_identity")
}

/// Write an embedded workload source into this target's scratch dir.
fn scratch_src(name: &str, text: &str) -> PathBuf {
    let p = scratch().join(name);
    std::fs::write(&p, text).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
    p
}

/// Compile `src` to `scratch()/out_name` with `mindc --emit-shared`, quoting
/// mindc's OWN stdout and stderr in the panic when it fails.
///
/// Eight builders each hand-rolled the same `Command…status()` +
/// `assert!(status.success(), "…failed for the <x> workload")` pair. `status()`
/// INHERITS the child's stderr, which bypasses libtest's capture and lands
/// unattributed in the raw run output — never in the panic message, and never
/// beside the test that failed. A run of this binary that reported nineteen
/// failures, every one of them the bare sentence `mindc --emit-shared failed
/// for the dot-q16 workload`, therefore carried no cause at all. `.output()`
/// captures both streams and the assertion quotes them, so the next occurrence
/// names itself.
fn emit_shared(workload: &str, src: &Path, out_name: &str, env: &[(&str, &str)]) -> PathBuf {
    let out = scratch().join(out_name);
    let mut cmd = Command::new(mindc_bin());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let r = cmd
        .args([
            src.to_str().unwrap(),
            "--emit-shared",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap_or_else(|e| panic!("spawn mindc --emit-shared for the {workload} workload: {e}"));
    assert!(
        r.status.success(),
        "mindc --emit-shared failed for the {workload} workload\n  \
         src:    {}\n  out:    {}\n  status: {}\n  stdout: {}\n  stderr: {}",
        src.display(),
        out.display(),
        r.status,
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr),
    );
    out
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
                // One owner for the skip decision. This used to be a hand-copied
                // `assert!(var_os(..).is_none())` + `println!` pair at eight sites:
                // `.is_none()` made `MIND_BENCH_REQUIRE=0` ENFORCE (the mirror of the
                // defect `gate::bless_mode` documents), and the announcement was
                // swallowed by cargo's capture, so the skip was uncountable.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = scratch_src("mind_xsi_dot_q16.mind", SRC);
        Some(emit_shared(
            "dot-q16",
            &src_path,
            "mind_xsi_dot_q16.so",
            &[],
        ))
    })
    .as_ref()
}

/// Compile `ARRAY_STORE_SRC` to its OWN temp `.so` once for the whole test
/// binary. Separate from `build_dot_so` because the aggregate loop-carry kernel
/// emits value-tensor ops that force the `arith-linalg` pipeline, which is
/// incompatible in one module with the `vector.reduction` int-dot kernels in
/// `SRC` (see the `ARRAY_STORE_SRC` note). Same toolchain self-skip contract.
fn build_array_store_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                // Same one-owner skip decision as `build_dot_so` above.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = scratch_src("mind_xsi_array_store.mind", ARRAY_STORE_SRC);
        Some(emit_shared(
            "array-store-loop",
            &src_path,
            "mind_xsi_array_store.so",
            &[],
        ))
    })
    .as_ref()
}

/// Compile `ARRAY_STORE_BRANCH_SRC` to its OWN temp `.so` (same arith-linalg
/// isolation rationale as `build_array_store_so`).
fn build_array_store_branch_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                // Same one-owner skip decision as `build_dot_so` above.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = scratch_src("mind_xsi_array_store_branch.mind", ARRAY_STORE_BRANCH_SRC);
        Some(emit_shared(
            "array-store-branch",
            &src_path,
            "mind_xsi_array_store_branch.so",
            &[],
        ))
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
                // Same one-owner skip decision as `build_dot_so` above.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        // examples/lorenz_q16.mind, relative to the crate root.
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("lorenz_q16.mind");
        Some(emit_shared(
            "lorenz-q16",
            &src_path,
            "mind_xsi_lorenz_q16.so",
            &[],
        ))
    })
    .as_ref()
}

/// Compile `examples/quant/<name>/main.mind` to its OWN temp `.so`, memoized
/// per `name` (one `OnceLock`-backed cache shared by every quant fixture, so
/// each of the 10 quant examples is built exactly once for the whole test
/// binary regardless of how many tests reference it). Generic sibling of
/// `build_lorenz_so` for the examples/quant/ tree: same toolchain self-skip /
/// `MIND_BENCH_REQUIRE` fail-closed contract (RFC 0020 §10) — a quant fixture
/// can never pass vacuously off a shadowed MLIR toolchain any more than the
/// integer canaries can.
fn build_quant_so(name: &str) -> Option<PathBuf> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: OnceLock<Mutex<HashMap<String, Option<PathBuf>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("quant .so cache poisoned");
    if let Some(cached) = guard.get(name) {
        return cached.clone();
    }
    let result = (|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                assert!(
                    std::env::var_os("MIND_BENCH_REQUIRE").is_none(),
                    "MIND_BENCH_REQUIRE is set but '{tool}' is not on PATH: the \
                     cross-substrate gate cannot run. Install the MLIR toolchain \
                     (mlir-opt / mlir-translate / clang) on this runner."
                );
                println!("cross_substrate_identity: {tool} not on PATH; skipping quant-{name}");
                return None;
            }
        }
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("quant")
            .join(name)
            .join("main.mind");
        let so_path = std::env::temp_dir().join(format!("mind_xsi_quant_{name}.so"));
        let status = Command::new(mindc_bin())
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("spawn mindc --emit-shared for quant/{name}: {e}"));
        assert!(
            status.success(),
            "mindc --emit-shared failed for the quant-{name} workload"
        );
        Some(so_path)
    })();
    guard.insert(name.to_string(), result.clone());
    result
}

/// The quant examples every test below reads, in stable order.
fn quant_example_dirs() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("quant");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("read {}: {e}", root.display()))
        .map(|entry| entry.expect("quant dir entry").path())
        .filter(|dir| dir.join("main.mind").is_file())
        .collect();
    dirs.sort();
    assert!(
        !dirs.is_empty(),
        "no quant examples under {}",
        root.display()
    );
    dirs
}

/// Brace-matched text of `fn name(...) { ... }` in `src`, or None if absent.
/// A `}` inside a nested block must not end the function early.
fn extract_mind_fn<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let start = src
        .lines()
        .scan(0usize, |offset, line| {
            let at = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find_map(|(at, line)| {
            let rest = line.strip_prefix("pub ").unwrap_or(line);
            let rest = rest.strip_prefix("fn ")?.strip_prefix(name)?;
            rest.trim_start().starts_with('(').then_some(at)
        })?;
    let open = start + src[start..].find('{')?;
    let mut depth = 0usize;
    for (offset, ch) in src[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[start..=open + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every quant example vendors its own copy of the deterministic math kernels
/// on purpose, so each stays readable and runnable alone. That only holds while
/// the copies are IDENTICAL: an edit to one example's `dm_erfc` would make two
/// examples price the same option differently, each with a green KAT, because
/// each KAT checks its own copy. An example may OMIT a kernel it does not use;
/// it may not define a different one.
#[test]
fn quant_shared_kernels_byte_identical_across_examples() {
    const SHARED_KERNELS: [&str; 11] = [
        "dm_exp",
        "dm_log",
        "dm_sqrt",
        "dm_erf",
        "dm_erfc",
        "dm_norm_cdf",
        "dm_norm_pdf",
        "absf",
        "canonical_qnan",
        "bs_d1",
        "bs_d2",
    ];
    let sources: Vec<(String, String)> = quant_example_dirs()
        .into_iter()
        .map(|dir| {
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            let src = std::fs::read_to_string(dir.join("main.mind"))
                .unwrap_or_else(|e| panic!("read quant/{name}/main.mind: {e}"));
            (name, src)
        })
        .collect();

    let mut diverged = Vec::new();
    let mut shared = 0usize;
    for kernel in SHARED_KERNELS {
        let mut variants: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
        for (name, src) in &sources {
            if let Some(body) = extract_mind_fn(src, kernel) {
                variants.entry(body).or_default().push(name);
            }
        }
        if variants.values().map(Vec::len).sum::<usize>() > 1 {
            shared += 1;
        }
        if variants.len() > 1 {
            let owners: Vec<String> = variants.values().map(|v| v.join(", ")).collect();
            diverged.push(format!(
                "{kernel}: {} definitions [{}]",
                variants.len(),
                owners.join(" | ")
            ));
        }
    }
    assert!(
        diverged.is_empty(),
        "quant examples disagree about a shared kernel -- copy the intended \
         definition verbatim into every example that defines it:\n  {}",
        diverged.join("\n  ")
    );
    // Positive control: the check is vacuous if nothing is actually shared.
    assert!(
        shared >= 5,
        "only {shared} kernels are shared; the extractor found nothing to compare"
    );
}

/// Each quant example's `main` is a known-answer suite whose exit code is its
/// count of failed checks. Run every one from a fresh copy so a stale `target/`
/// can never report an earlier artifact's status.
///
/// x86_64 only: the examples' `mindc run` path links the host CPU runtime, which
/// is not built for aarch64 yet. The neon leg is covered by the byte-identity
/// fixtures above, which compile each kernel to a shared object instead.
/// deferred: would also run on aarch64, skipped because the CPU runtime is
/// x86_64-only -- upgrade path: drop this cfg once that runtime builds for ARM.
#[cfg(target_arch = "x86_64")]
#[test]
fn quant_examples_known_answer_suites_pass() {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            assert!(
                std::env::var_os("MIND_BENCH_REQUIRE").is_none(),
                "MIND_BENCH_REQUIRE is set but '{tool}' is not on PATH"
            );
            println!("cross_substrate_identity: {tool} not on PATH; skipping quant KAT runs");
            return;
        }
    }
    let mut failed = Vec::new();
    for dir in quant_example_dirs() {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let work = tempfile::tempdir().expect("tempdir for quant KAT run");
        for file in ["Mind.toml", "main.mind"] {
            std::fs::copy(dir.join(file), work.path().join(file))
                .unwrap_or_else(|e| panic!("copy quant/{name}/{file}: {e}"));
        }
        let out = Command::new(mindc_bin())
            .arg("run")
            .current_dir(work.path())
            .output()
            .unwrap_or_else(|e| panic!("spawn mindc run for quant/{name}: {e}"));
        if !out.status.success() {
            failed.push(format!(
                "{name}: exit {:?} (count of failed checks, or a build error)\n{}{}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "quant known-answer suites red:\n{}",
        failed.join("\n")
    );
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

/// Emit a bless line for `id` on the host substrate (RFC 0020 §13).
///
/// Prints `BLESS <id> <substrate> <hash>` to stdout AND — when
/// `MIND_BENCH_HASHES_OUT` names a file — appends the same line to it, so a
/// bless run on a substrate we do not own locally (the real-aarch64 CI runner)
/// leaves a DURABLE artifact instead of a hash buried in a log that expires.
///
/// This is the mechanism the strict-f32 neon bless needed and did not have: the
/// ARM runner had been computing the correct neon hashes on every CI run and
/// throwing them away into an ephemeral job log, which is precisely why that
/// hole stayed open long after the hardware to close it was already in the
/// matrix. The harvest is now a file download, not log archaeology.
///
/// Writing the sink is strictly NON-load-bearing: a failed write cannot turn a
/// red gate green (it is only reachable under MIND_BENCH_BLESS, which asserts
/// nothing), and its absence changes no assertion.
fn emit_bless(id: &str, substrate: &str, computed: &str) {
    println!("BLESS {id} {substrate} {computed}");
    if let Some(path) = std::env::var_os("MIND_BENCH_HASHES_OUT") {
        use std::io::Write;
        static SINK_LOCK: Mutex<()> = Mutex::new(());
        let _guard = SINK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{id} {substrate} {computed}");
        }
    }
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

    if common::gate::bless_mode() {
        emit_bless(w.id, substrate, &computed);
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
    common::xsi_gate::record_measured(w.id);
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

/// Canonical hash of a grid of f64 scalar results (manifest `output_encoding =
/// "f64_bits_i64_le_grid"`): each grid point's IEEE-754 bit pattern as an i64,
/// 8 little-endian bytes, in fixed grid order, concatenated, then sha256 — the
/// vector-of-scalars extension of `scalar-float-f64`'s single-value
/// `f64_bits_i64_le` encoding (used by every `quant-*` fixture).
fn canonical_hash_f64_grid(v: &[f64]) -> String {
    let mut h = Sha256::new();
    for &e in v {
        h.update((e.to_bits() as i64).to_le_bytes());
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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

    // Collatz (3n+1) hash (collatz) — SEPARATE .so (examples/ tree), pure i64,
    // scalar return; fixed [lo, hi] input, must hash identically every run.
    if let Some(cso) = build_collatz_so() {
        let clib = unsafe { Library::new(cso).expect("dlopen collatz workload .so") };
        let collatz: Symbol<CollatzFn> =
            unsafe { clib.get(b"collatz_hash").expect("collatz_hash symbol") };
        let (lo, hi) = COLLATZ_INPUTS;
        stable("collatz", &|| canonical_hash(unsafe { collatz(lo, hi) }));
    }

    // Galperin billiard-π collision count (galperin-pi) — SEPARATE .so, pure i64,
    // scalar return; fixed n, must hash identically every run.
    if let Some(gso) = build_galperin_so() {
        let glib = unsafe { Library::new(gso).expect("dlopen galperin workload .so") };
        let galperin: Symbol<GalperinFn> = unsafe {
            glib.get(b"galperin_collisions")
                .expect("galperin_collisions symbol")
        };
        stable("galperin-pi", &|| {
            canonical_hash(unsafe { galperin(GALPERIN_INPUT) })
        });
    }

    // --- examples/quant/* pricing kernels — each in its OWN .so (build_quant_so).
    // Every grid point's f64 result is fixed and deterministic (no clock, no
    // OS entropy, no host RNG); a fresh Vec collects the grid each run so a
    // kernel that read prior buffer state or left an output uninitialised
    // would surface as a run-to-run divergence, same rationale as the buffer
    // canaries above.

    if let Some(so) = build_quant_so("black_scholes") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-black-scholes .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"bs_call").expect("bs_call symbol") };
        stable("quant-black-scholes", &|| {
            let out: Vec<f64> = QUANT_BLACK_SCHOLES_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t)| unsafe { f(s, k, r, q, sigma, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("bond_curve") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-bond-curve .so") };
        let f: Symbol<QuantBondPriceFn> =
            unsafe { lib.get(b"bond_price").expect("bond_price symbol") };
        stable("quant-bond-curve", &|| {
            let out: Vec<f64> = QUANT_BOND_CURVE_GRID
                .iter()
                .map(|&(face, cr, ytm, freq, n)| unsafe { f(face, cr, ytm, freq, n) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("greeks_higher_order") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-greeks-higher-order .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"hog_vanna").expect("hog_vanna symbol") };
        stable("quant-greeks-higher-order", &|| {
            let out: Vec<f64> = QUANT_BLACK_SCHOLES_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t)| unsafe { f(s, k, r, q, sigma, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("implied_vol_surface") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol-surface .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe {
            lib.get(b"svi_total_variance")
                .expect("svi_total_variance symbol")
        };
        stable("quant-implied-vol-surface", &|| {
            let out: Vec<f64> = QUANT_SVI_GRID
                .iter()
                .map(|&(k, a, b, rho, m, p_sigma)| unsafe { f(k, a, b, rho, m, p_sigma) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("portfolio_risk") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-portfolio-risk .so") };
        let f: Symbol<QuantF64x5Fn> =
            unsafe { lib.get(b"two_asset_vol").expect("two_asset_vol symbol") };
        stable("quant-portfolio-risk", &|| {
            let out: Vec<f64> = QUANT_PORTFOLIO_RISK_GRID
                .iter()
                .map(|&(w1, w2, s1, s2, rho)| unsafe { f(w1, w2, s1, s2, rho) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("american_binomial") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-american-binomial .so") };
        let f: Symbol<QuantF64x6I64Fn> = unsafe {
            lib.get(b"bopm_american_put")
                .expect("bopm_american_put symbol")
        };
        stable("quant-american-binomial", &|| {
            let out: Vec<f64> = QUANT_AMERICAN_BINOMIAL_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t, n)| unsafe { f(s, k, r, q, sigma, t, n) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("barrier_analytic") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-barrier-analytic .so") };
        let f: Symbol<QuantF64x7Fn> = unsafe { lib.get(b"bar_doc").expect("bar_doc symbol") };
        stable("quant-barrier-analytic", &|| {
            let out: Vec<f64> = QUANT_BARRIER_ANALYTIC_GRID
                .iter()
                .map(|&(s, k, h, r, q, sigma, t)| unsafe { f(s, k, h, r, q, sigma, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("crr_binomial") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-crr-binomial .so") };
        let f: Symbol<QuantCrrPriceFn> =
            unsafe { lib.get(b"crr_price").expect("crr_price symbol") };
        stable("quant-crr-binomial", &|| {
            let out: Vec<f64> = QUANT_CRR_BINOMIAL_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t, n, is_call, american)| unsafe {
                    f(s, k, r, q, sigma, t, n, is_call, american)
                })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("implied_vol") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol .so") };
        let f: Symbol<QuantF64x6Fn> = unsafe {
            lib.get(b"implied_vol_call")
                .expect("implied_vol_call symbol")
        };
        stable("quant-implied-vol", &|| {
            let out: Vec<f64> = QUANT_IMPLIED_VOL_GRID
                .iter()
                .map(|&(price, s, k, r, q, t)| unsafe { f(price, s, k, r, q, t) })
                .collect();
            canonical_hash_f64_grid(&out)
        });
    }

    if let Some(so) = build_quant_so("monte_carlo_gbm") {
        let lib = unsafe { Library::new(&so).expect("dlopen quant-monte-carlo-gbm .so") };
        let f: Symbol<QuantMcCallFn> = unsafe { lib.get(b"mc_call").expect("mc_call symbol") };
        stable("quant-monte-carlo-gbm", &|| {
            let out: Vec<f64> = QUANT_MONTE_CARLO_GBM_GRID
                .iter()
                .map(|&(s, k, r, q, sigma, t, npath, seed)| unsafe {
                    f(s, k, r, q, sigma, t, npath, seed)
                })
                .collect();
            canonical_hash_f64_grid(&out)
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
}

// --- gemm-i8-vnni workload (VPDPBUSD int-dot rung, VNNI hardware only) -------
// RANK 3. This uses the explicit AVX-512 VPDPBUSD lowering with exact signed-i8
// bias correction. A host without AVX-512-VNNI records the one manifest-bound
// deferral; a capable host must opt in and execute rather than skip.

/// Build SRC with `MIND_INTDOT=vnni` into a DISTINCT `.so`, so the int8 kernel
/// emits the VPDPBUSD rung (and clang links the VNNI target features). Returns
/// `None` on toolchain self-skip. Only called after a VNNI-capability gate.
fn build_dot_so_vnni() -> Option<PathBuf> {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            // Same one-owner skip decision as `build_dot_so` above.
            crate::common::gate::skipped(
                "cross_substrate_identity",
                &format!(
                    "{tool} not on PATH; install the MLIR toolchain \
                     (mlir-opt / mlir-translate / clang) on this runner"
                ),
            );
            return None;
        }
    }
    let src_path = scratch_src("mind_xsi_dot_q16_vnni.mind", SRC);
    Some(emit_shared(
        "VNNI int8",
        &src_path,
        "mind_xsi_dot_q16_vnni.so",
        &[("MIND_INTDOT", "vnni")],
    ))
}

#[test]
fn gemm_i8_vnni_reproducibility_gate() {
    let id = "gemm-i8-vnni-64x64x64";
    let (m, k, n, seed) = (64usize, 64usize, 64usize, 0xDEADBEEFu64);

    let policy = xsi_gate::policy_for_id(id).expect("declared VNNI case policy");
    let has_vnni = xsi_gate::host_has_avx512vnni();
    let verify = std::env::var(xsi_gate::VNNI_VERIFY_VAR).ok();
    match xsi_gate::vnni_decision(&policy, has_vnni, verify.as_deref()) {
        Ok(VnniDecision::Run) => {}
        Ok(VnniDecision::DeferUnsupported(isa)) => {
            xsi_gate::record_deferred(id, isa);
            // OPTIONAL INPUT: AVX-512-VNNI silicon supplied by the runner;
            // the receipt writer independently verifies its absence above.
            crate::common::gate::skipped_optional_case(
                "cross_substrate_identity",
                id,
                "host lacks AVX-512-VNNI (vpdpbusd); this rung did not execute",
            );
            return;
        }
        Err(reason) => panic!("{id}: {reason}"),
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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

/// Two u64 operands with the HIGH BIT SET, passed as their i64 bit patterns. As
/// signed i64 `a` is negative, so a SIGNED lowering of `>`/`/`/`>>` would give
/// the wrong answer — the canary catches a regression to signed u64 ops.
const U64_OPS_INPUTS: (u64, u64) = (0x8000_0000_0000_0005, 3);

/// Rust `u64` oracle for `u64_ops` — the compiled kernel must reproduce it
/// bit-for-bit. Rust's native `u64` operators ARE the unsigned semantics
/// (`divui`/`remui`/`shrui`/`ult`) the kernel lowers to, and the interpreter's
/// `apply_int_op_u64` mirror uses the SAME Rust `u64` ops, so this one oracle
/// pins artifact == Rust == interpreter.
fn ref_u64_ops(a: u64, b: u64) -> i64 {
    let s = a >> 3;
    let d = a / b;
    let m = a % b;
    let g = if a > b { 1i64 } else { 0 };
    let l = if a <= b { 1i64 } else { 0 };
    let k: i64 = 1_000_003;
    let mut acc = s as i64;
    acc = acc.wrapping_mul(k).wrapping_add(d as i64);
    acc = acc.wrapping_mul(k).wrapping_add(m as i64);
    acc = acc.wrapping_mul(k).wrapping_add(g);
    acc.wrapping_mul(k).wrapping_add(l)
}

#[test]
fn u64_ops_reproducibility_gate() {
    let id = "u64-ops";

    let Some(so) = build_dot_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let f: Symbol<U64OpsFn> = unsafe { lib.get(b"u64_ops").expect("u64_ops symbol") };

    let (a, b) = U64_OPS_INPUTS;
    let result = unsafe { f(a as i64, b as i64) };

    // 1. Within-run exactness vs the Rust u64 oracle: proves the compiled kernel
    //    used the UNSIGNED op variants (a>b => 1, a>>3 logical, a/b unsigned) —
    //    a signed lowering would diverge because `a` has the high bit set.
    let oracle = ref_u64_ops(a, b);
    assert_eq!(
        result, oracle,
        "{id}: u64 kernel diverged from the Rust unsigned oracle within a single \
         run (kernel={result}, oracle={oracle}) — a signed lowering regressed?"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference. Unsigned
    //    integer ops are exact and order-independent → avx2 == neon by construction.
    let computed = canonical_hash(result);
    pin_or_bless(id, &computed, result);
}

/// Fixed runtime input for `array_store_loop`. A non-trivial `base` so the
/// per-slot store `a[k] = base*k + k*k` produces distinct values across all
/// eight elements (not all zero), making a dropped write observable in the fold.
const ARRAY_STORE_BASE: i64 = 7;

/// Rust oracle for `array_store_loop` — the compiled kernel must reproduce it
/// bit-for-bit. Mirrors the MIND kernel exactly: fill `a[k] = base*k + k*k` for
/// k in 0..8 (a loop-carried fixed `[i64; 8]`), then fold the eight slots in
/// fixed source order via the wrapping polynomial (K = 1_000_003).
fn ref_array_store_loop(base: i64) -> i64 {
    let mut a = [0i64; 8];
    for (k, slot) in a.iter_mut().enumerate() {
        let k = k as i64;
        *slot = base.wrapping_mul(k).wrapping_add(k.wrapping_mul(k));
    }
    let kk: i64 = 1_000_003;
    let mut acc = a[0];
    for &e in &a[1..] {
        acc = acc.wrapping_mul(kk).wrapping_add(e);
    }
    acc
}

#[test]
fn array_store_loop_reproducibility_gate() {
    let id = "array-store-loop";

    let Some(so) = build_array_store_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen array-store workload .so") };
    let f: Symbol<ArrayStoreFn> = unsafe {
        lib.get(b"array_store_loop")
            .expect("array_store_loop symbol")
    };

    let result = unsafe { f(ARRAY_STORE_BASE) };

    // 1. Within-run exactness vs the Rust oracle: proves the loop-body stores
    //    landed (the pre-F2 miscompile dropped them, folding all-zero) and that
    //    the post-loop reads see the mutated aggregate.
    let oracle = ref_array_store_loop(ARRAY_STORE_BASE);
    assert_eq!(
        result, oracle,
        "{id}: aggregate loop-carry kernel diverged from the Rust oracle within a \
         single run (kernel={result}, oracle={oracle}) — a dropped/aliased store regressed?"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference. Integer
    //    store/add/mul are exact and order-independent, and the bufferized loop
    //    body has no per-iteration copy → avx2 == neon by construction.
    let computed = canonical_hash(result);
    pin_or_bless(id, &computed, result);
}

/// Rust oracle for `array_store_branch` — the compiled kernel must reproduce it
/// bit-for-bit. Mirrors the MIND kernel exactly: `if`/`else` + no-else stores
/// into a fixed `[i64; 8]` (a value-semantic If-merge), then a fixed-order fold.
fn ref_array_store_branch(base: i64) -> i64 {
    let mut a = [0i64; 8];
    a[0] = if base > 3 {
        base.wrapping_mul(7)
    } else {
        base.wrapping_add(100)
    };
    a[1] = base.wrapping_mul(2);
    if base > 100 {
        a[2] = 999;
    }
    a[3] = if base > 5 {
        base.wrapping_mul(11)
    } else {
        base.wrapping_sub(50)
    };
    let kk: i64 = 1_000_003;
    let mut acc = a[0];
    for &e in &a[1..] {
        acc = acc.wrapping_mul(kk).wrapping_add(e);
    }
    acc
}

#[test]
fn array_store_branch_reproducibility_gate() {
    let id = "array-store-branch";

    let Some(so) = build_array_store_branch_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen array-store-branch workload .so") };
    let f: Symbol<ArrayStoreFn> = unsafe {
        lib.get(b"array_store_branch")
            .expect("array_store_branch symbol")
    };

    let result = unsafe { f(ARRAY_STORE_BASE) };

    // 1. Within-run exactness vs the Rust oracle: proves the branch-body stores
    //    landed on the taken edges and the `^if_after` merge carried the right
    //    incarnation (a dropped/aliased branch store folds a wrong value).
    let oracle = ref_array_store_branch(ARRAY_STORE_BASE);
    assert_eq!(
        result, oracle,
        "{id}: aggregate branch-carry kernel diverged from the Rust oracle within a \
         single run (kernel={result}, oracle={oracle}) — a dropped/aliased branch store regressed?"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference. Integer
    //    store/add/mul are exact + order-independent, and the bufferized If-merge
    //    has no per-branch copy (profile matches the loop baseline) → avx2 == neon.
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, computed);
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
    common::xsi_gate::record_measured(id);
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
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &computed);
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
    common::xsi_gate::record_measured(id);
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
// aarch64, never asserted from x86 + the associativity argument.
//
// That bless is DONE (2026-07-14): both canaries were executed on the
// GitHub-hosted `ubuntu-24.04-arm` runner (real aarch64 silicon, LLVM 20,
// MIND_BENCH_REQUIRE=1) by the `cross_substrate_identity (neon)` CI job and
// produced hashes byte-for-byte IDENTICAL to the avx2 references, reproduced
// across four independent runs. Both `neon` lines are now committed, and
// `pin_strict_fp` is FAIL-CLOSED: the old self-passing DEFER arm is gone, so a
// substrate without a committed strict-FP hash can no longer green this gate.
// The strict-f32 leg of "bit-identical across CPU and ARM" is therefore proven
// by execution, not argued.
//
// The within-run oracle below mirrors the kernel's exact fold (8-lane
// accumulator, fixed 0..8 horizontal fold, scalar tail) so it is bit-exact to
// the kernel on whichever substrate runs it.
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

/// Pin a strict-FP canary's hash to the committed per-substrate reference.
///
/// FAIL-CLOSED (2026-07-14). This function used to carry a `None if substrate ==
/// "neon"` arm that printed a loud DEFER and *passed* the test, because the
/// strict-FP neon line was unblessed: bit-identity here rests on the strict
/// lowering (FMA unfused + pinned fixed-order fold), NOT on integer
/// associativity, so it could only ever be proven by execution on real ARM and
/// must never be asserted from x86.
///
/// That bless has now HAPPENED: the `cross_substrate_identity (neon)` CI job on
/// the GitHub-hosted `ubuntu-24.04-arm` runner (real aarch64 silicon, LLVM 20,
/// MIND_BENCH_REQUIRE=1) reproduced the avx2 hash byte-for-byte for BOTH
/// strict-f32 canaries across four independent runs, and both `neon` lines are
/// committed (see each fixture's reference_hashes.toml provenance block).
///
/// With both substrates blessed, the deferral arm is DEAD — and keeping it would
/// be actively dangerous: it was a self-passing branch that turned a missing
/// reference into a green test. A missing reference on ANY substrate is now a
/// hard failure, exactly like every other canary in this suite. If a new
/// substrate (e.g. a GPU tier) is added later, it must earn its hash by running
/// on its own real hardware — it does NOT get to self-pass its way in.
///
/// Under MIND_BENCH_BLESS it prints the bless line instead (RFC 0020 §13).
fn pin_strict_fp(id: &str, computed: &str, bits: u32) {
    let substrate = host_substrate();
    if common::gate::bless_mode() {
        emit_bless(id, substrate, computed);
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            computed, expected,
            "{id} [{substrate}]: strict-f32 output hash drifted from the committed reference.\n\
             computed={computed}\n expected={expected}\n result_bits={bits:#010x}\n\
             Both avx2 and neon are BLESSED on real hardware and carry the SAME hash, so a \
             drift here is a strict-FP WEDGE BREAK (an FMA got contracted, or the horizontal \
             fold stopped being a pinned fixed-order left-to-right lane fold), NOT a re-bless. \
             Re-bless with MIND_BENCH_BLESS=1 ONLY for a deliberate, reviewed lowering change \
             (RFC 0020 §13) — and then only on BOTH substrates, on real hardware each."
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}' — FAIL-CLOSED. Computed \
             {computed} (bits={bits:#010x}). A substrate without a committed strict-FP hash \
             CANNOT pass this gate: strict-FP bit-identity rests on the lowering, not on \
             integer associativity, so it must be earned by execution on that substrate's \
             REAL hardware and blessed with MIND_BENCH_BLESS=1 — never asserted from another \
             substrate, never emulated, never self-skipped into green."
        ),
    }
    common::xsi_gate::record_measured(id);
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
    pin_strict_fp(id, &computed, result.to_bits());
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
    pin_strict_fp(id, &computed, head_bits);
}

// ===========================================================================
// grammar-mask — structured / grammar-constrained decoding canary (roadmap
// Phase 14 §6, slice-1 go/no-go). Source: examples/grammar_mask/main.mind.
//
// The kernel `grammar_mask_fold` compiles two grammars to automata and emits
// their per-state legal-token bitmask TABLES in the runnable integer subset —
// FSM (flat JSON object, 7 states x 1 word = 56 bytes) then PDA (nested JSON
// value, 6 states x 3 stack symbols = 144 bytes) — and folds all 200 bytes with
// FNV-1a 32-bit. This is the DETERMINISTIC tier of guided decoding: the mask is
// a pure integer function of (grammar, vocabulary), with no float, no clock, no
// division, no reassociation, so avx2 == neon by construction (RFC 0015 §3.1).
// The -inf logit mask, softmax, and sampler are OUT of scope (the model's float
// path). A drift here means the mask table or fold picked up a target-sensitive
// or nondeterministic op — a release-blocker.
// ===========================================================================

type GrammarMaskFoldFn = unsafe extern "C" fn() -> i64;

/// Compile `examples/grammar_mask/main.mind` to a temp `.so` once for the whole
/// test binary. Same toolchain guard / self-skip discipline as
/// `build_lorenz_so`: `None` when the MLIR toolchain is shadowed, but a hard
/// failure under `MIND_BENCH_REQUIRE` so the gate can never pass vacuously.
fn build_grammar_mask_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                // Same one-owner skip decision as `build_dot_so` above.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("grammar_mask")
            .join("main.mind");
        Some(emit_shared(
            "grammar-mask",
            &src_path,
            "mind_xsi_grammar_mask.so",
            &[],
        ))
    })
    .as_ref()
}

/// Independent Rust oracle for `grammar_mask_fold`. The mask words are derived
/// from the grammar semantics ALONE (not read back from the .mind): the FSM
/// legal-token bitmask per state, then the PDA legal-token bitmask per
/// (state, stack-top). Vocabulary bit order:
///   0 `{`  1 `}`  2 `[`  3 `]`  4 `:`  5 `,`  6 STRING  7 NUMBER
///   8 TRUE 9 FALSE 10 NULL 11 END.
/// Serialised exactly as the kernel does — 8 little-endian bytes per u64 word,
/// FSM table then PDA table — and folded with FNV-1a 32-bit over UNSIGNED bytes
/// (matching `__mind_load_i8`'s zero-extend). If the compiled table drifts from
/// the grammar this fires before the hash pin.
fn ref_grammar_mask_fold() -> i64 {
    const FSM: [u64; 7] = [0x001, 0x042, 0x010, 0x7c0, 0x022, 0x800, 0x040];
    const PDA: [u64; 18] = [
        0x7c5, 0x7c5, 0x7c5, // V0: value start (`{` `[` scalars)
        0x7c5, 0x7c5, 0x7cd, // A0: + `]` when ARR is on top (empty array)
        0x040, 0x042, 0x040, // K0: STRING key, + `}` when OBJ on top (empty object)
        0x040, 0x040, 0x040, // K1: STRING key after `,`
        0x010, 0x010, 0x010, // C0: `:`
        0x800, 0x022, 0x028, // E0: END at BOTTOM, `,`/`}` under OBJ, `,`/`]` under ARR
    ];
    let mut bytes: Vec<u8> = Vec::with_capacity(200);
    for w in FSM.iter().chain(PDA.iter()) {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    // FNV-1a 32-bit in the sha256.mind mask-32 discipline: h < 2^32 and b < 256
    // keep every product < 2^57, so the fold never overflows i64.
    let mut h: i64 = 2166136261;
    for b in bytes {
        h = (h ^ b as i64).wrapping_mul(16777619) & 4294967295;
    }
    h
}

#[test]
fn grammar_mask_reproducibility_gate() {
    let id = "grammar-mask";

    let Some(so) = build_grammar_mask_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen workload .so") };
    let f: Symbol<GrammarMaskFoldFn> = unsafe {
        lib.get(b"grammar_mask_fold")
            .expect("grammar_mask_fold symbol")
    };

    let result = unsafe { f() };

    // 1. Within-run oracle: the compiled legal-token mask table reproduces the
    //    grammar semantics reconstructed independently in Rust.
    let oracle = ref_grammar_mask_fold();
    assert_eq!(
        result, oracle,
        "{id}: compiled legal-token mask table diverged from the grammar oracle \
         within a single run (kernel={result:#x}, oracle={oracle:#x})"
    );

    // 2. Canonical hash pinned to the committed per-substrate reference.
    //    Integer-only fold → avx2 == neon by construction (RFC 0015 §3.1).
    let computed = canonical_hash(result);
    pin_or_bless(id, &computed, result);
}

// bimap-phf — the #[bimap] perfect-hash CONSTRUCTION byte-identity canary.
// Source: examples/bimap_currency/main.mind. Task #182.
// ===========================================================================
//
// Distinct in KIND from every other workload in this suite: the others hash a
// kernel's RUNTIME OUTPUT; this one hashes the COMPILER'S CONSTRUCTION of the
// `#[bimap]` derive. The `#[bimap] enum Currency` derive
// (src/parser/expand_bimap.rs `make_from_str_phf`, over the seedless-CHD
// perfect hash in src/phf/mod.rs, phf-v1) synthesises `currency_from_str` as an
// O(1) perfect-hash inverse; the emitted disp/meta/pool table literals plus the
// from_str body ARE the construction under test.
//
// The measured fingerprint is the artifact `trace_hash` — the SHA-256 of the
// canonical mic@3 IR bytes (`trace_hash_kind = mic3-bytes`, MIND-CONSTITUTION
// §IV). That encoding is substrate-independent BY DESIGN, so a substrate-
// dependent construction bug (a table literal emitted differently on ARM vs
// x86) surfaces as a trace_hash mismatch. This is STRICTLY STRONGER than a
// runtime-output hash: two different-but-both-correct PHF tables would answer
// currency_from_str identically (a runtime hash would pass) yet produce
// different mic@3 bytes (this gate fails). The trace_hash pins the TABLE, not
// merely the answers — which is exactly the wedge claim `#[bimap]` rests on.
//
// This is a compile-time gate: it runs the REAL mindc compile of the example
// (no stub, no self-skip — mic@3 emission is the frontend/IR path, so unlike
// the `.so` gates it does not need the MLIR toolchain and therefore NEVER
// self-skips). Fail-closed by other means: a failed compile, a stub/short
// artifact, a missing/invalid trace_hash, or a non-strict fp_mode is a hard
// failure. Pins the trace_hash to the committed per-substrate reference; a
// drift is a hard failure pointing at the RFC 0020 §13 re-bless protocol, and
// there is NO auto-bless path.

/// Compile examples/bimap_currency with `mindc --emit-evidence`, run
/// `mindc verify`, and return the parsed `(trace_hash, trace_hash_valid,
/// fp_mode)`. Fails hard (never self-skips) if the compile fails, the artifact
/// is a stub/short blob, or verify does not report a valid trace_hash — the
/// construction gate is worthless if it did not actually construct.
fn bimap_trace_hash() -> (String, String, String) {
    let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("bimap_currency")
        .join("main.mind");
    // Artifact path private to this target AND this process (scratch_dir):
    // the hand-rolled pid suffix this replaces was a per-file uniqueness
    // policy, and the helper exists so there is exactly one.
    let ev_path = scratch().join("mind_xsi_bimap_ev.json");

    let emit = Command::new(mindc_bin())
        .args([
            src_path.to_str().unwrap(),
            "--emit-evidence",
            ev_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn mindc --emit-evidence for bimap_currency");
    assert!(
        emit.status.success(),
        "bimap-phf: mindc --emit-evidence failed for examples/bimap_currency/main.mind\n\
         stdout: {}\n stderr: {}",
        String::from_utf8_lossy(&emit.stdout),
        String::from_utf8_lossy(&emit.stderr),
    );

    // The artifact must be a real mic@3 evidence blob, not a stub. The genuine
    // artifact is ~4 KB (self-check ok); the #306 launcher-stub failure mode is
    // ~1.2 KB. A short file means the construction did not really happen — that
    // would make the gate vacuous, so reject it loudly.
    let ev_bytes = std::fs::read(&ev_path).expect("read bimap evidence artifact");
    assert!(
        ev_bytes.len() >= 2048,
        "bimap-phf: evidence artifact is only {} bytes — suspected stub/short blob, \
         the construction gate would be vacuous (expected a ~4 KB mic@3 artifact)",
        ev_bytes.len()
    );

    let verify = Command::new(mindc_bin())
        .args(["verify", ev_path.to_str().unwrap()])
        .output()
        .expect("spawn mindc verify for bimap_currency");
    assert!(
        verify.status.success(),
        "bimap-phf: mindc verify failed for the bimap evidence artifact\n\
         stdout: {}\n stderr: {}",
        String::from_utf8_lossy(&verify.stdout),
        String::from_utf8_lossy(&verify.stderr),
    );
    let out = String::from_utf8_lossy(&verify.stdout);

    let field = |key: &str| -> Option<String> {
        out.lines().find_map(|l| {
            let l = l.trim();
            l.strip_prefix(key).map(|rest| rest.trim().to_string())
        })
    };
    let trace_hash =
        field("trace_hash:").expect("bimap-phf: `mindc verify` printed no trace_hash line");
    let valid = field("trace_hash_valid:")
        .expect("bimap-phf: `mindc verify` printed no trace_hash_valid line");
    let fp_mode = field("fp_mode:").expect("bimap-phf: `mindc verify` printed no fp_mode line");

    let _ = std::fs::remove_file(&ev_path);
    (trace_hash, valid, fp_mode)
}

#[test]
fn bimap_phf_construction_identity_gate() {
    let id = "bimap-phf";
    let (trace_hash, valid, fp_mode) = bimap_trace_hash();

    // 1. The construction must be intact and strict: an unverifiable or
    //    non-strict artifact means the fingerprint is not trustworthy.
    assert_eq!(
        valid, "yes",
        "{id}: mindc verify reports trace_hash_valid={valid} (expected yes) — \
         the #[bimap] construction artifact is not self-consistent"
    );
    assert_eq!(
        fp_mode, "strict",
        "{id}: mindc verify reports fp_mode={fp_mode} (expected strict) — the \
         #[bimap] enum-form construction must classify strict"
    );

    // 2. Pin the trace_hash (canonical mic@3 bytes) to the committed
    //    per-substrate reference. Substrate-independent by design, so avx2 ==
    //    neon (RFC 0015 §3.1); a drift is a real construction change and a hard
    //    failure — re-bless only per RFC 0020 §13, never automatically.
    let substrate = host_substrate();
    if common::gate::bless_mode() {
        emit_bless(id, substrate, &trace_hash);
        return;
    }
    match reference_hash(id, substrate) {
        Some(expected) => assert_eq!(
            trace_hash, expected,
            "{id} [{substrate}]: #[bimap] construction trace_hash drifted from the \
             committed reference.\n computed={trace_hash}\n expected={expected}\n\
             This is the phf-v1 O(1) from_str construction fingerprint (canonical \
             mic@3 bytes). A change means the emitted PHF table/body moved. If this \
             is an intentional #[bimap]/phf-v1 lowering change (RFC 0020 §13), \
             re-bless with MIND_BENCH_BLESS=1 on BOTH substrates and commit the new \
             reference_hashes.toml; otherwise it is a construction regression — STOP."
        ),
        None => panic!(
            "{id}: no reference hash for substrate '{substrate}' in \
             reference_hashes.toml. Computed trace_hash is {trace_hash}; bless with \
             MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
    common::xsi_gate::record_measured(id);
}

// ===========================================================================
// collatz + galperin-pi — the two "deterministic != predictable" DEMO canaries
// (Salov C1 / roadmap W0.4). Both are shipped demos in examples/, PURE i64 with
// ZERO floating point: no rounding mode, no FMA contraction, no reduction order
// and no reassociation to differ across substrates, so avx2 == neon BY
// CONSTRUCTION (exact integer arithmetic; RFC 0015 §3.1) — the same footing as
// lorenz-q16, and the whole point of these demos ("compile, run, hash; a
// different hash is a release-blocking bug, not a rounding artifact"). Each
// compiles to its OWN .so (examples/ tree, like lorenz-q16 / grammar-mask), is
// cross-checked within a run against an INDEPENDENT Rust port of the identical
// integer algorithm, and pins its canonical i64_le output hash to the committed
// per-substrate reference.
// ===========================================================================

/// The Collatz hash kernel: (lo, hi) → folded i64 hash of the [lo, hi] orbits.
type CollatzFn = unsafe extern "C" fn(i64, i64) -> i64;
/// The Galperin billiard-π kernel: n → collision count (floor(π·10^(n-1))).
type GalperinFn = unsafe extern "C" fn(i64) -> i64;

/// Compile `examples/collatz.mind` to a temp `.so` once for the whole test
/// binary. Separate .so (examples/ tree, also a shipped demo), same toolchain
/// guard / self-skip discipline as `build_lorenz_so`: `None` when the MLIR
/// toolchain is shadowed, a hard failure under `MIND_BENCH_REQUIRE` so the gate
/// can never pass vacuously (RFC 0020 §10).
fn build_collatz_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                // Same one-owner skip decision as `build_dot_so` above.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("collatz.mind");
        Some(emit_shared(
            "collatz",
            &src_path,
            "mind_xsi_collatz.so",
            &[],
        ))
    })
    .as_ref()
}

/// Compile `examples/galperin_pi.mind` to a temp `.so` once for the whole test
/// binary. Separate .so, same toolchain guard / self-skip discipline as
/// `build_lorenz_so`.
fn build_galperin_so() -> Option<&'static PathBuf> {
    static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
    SO.get_or_init(|| {
        for tool in ["mlir-opt", "mlir-translate", "clang"] {
            if which::which(tool).is_err() {
                // Same one-owner skip decision as `build_dot_so` above.
                crate::common::gate::skipped(
                    "cross_substrate_identity",
                    &format!(
                        "{tool} not on PATH; install the MLIR toolchain \
                         (mlir-opt / mlir-translate / clang) on this runner"
                    ),
                );
                return None;
            }
        }
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("galperin_pi.mind");
        Some(emit_shared(
            "galperin-pi",
            &src_path,
            "mind_xsi_galperin.so",
            &[],
        ))
    })
    .as_ref()
}

/// Fixed Collatz input range (manifest `[input]`): sum + fold the stopping times
/// of every seed in [1, 1000]. Non-trivial (1000 orbits, some reaching ~250504)
/// yet fully in-range; the fold `acc = acc*1000003 + steps + n` wraps in two's
/// complement, matching `arith.muli`/`arith.addi`.
const COLLATZ_INPUTS: (i64, i64) = (1, 1000);

/// Independent in-process oracle for `collatz_hash` — the identical integer
/// Collatz iteration + multiply-mix fold, in the identical source order. Exact
/// integer arithmetic with a two's-complement wrapping fold, so it is bit-exact
/// within a run and (being float-free) the same on every substrate. The inner
/// `v` arithmetic never overflows (|v| stays well under i64 for seeds ≤ 1000);
/// only the `acc` fold wraps, mirrored here with `wrapping_*`.
fn ref_collatz_hash(lo: i64, hi: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut n: i64 = lo;
    while n <= hi {
        let mut v: i64 = n;
        let mut steps: i64 = 0;
        while v > 1 {
            let half = v / 2;
            let is_even = v - half * 2; // 0 if even, 1 if odd
            v = (1 - is_even) * half + is_even * (3 * v + 1);
            steps += 1;
        }
        acc = acc
            .wrapping_mul(1000003)
            .wrapping_add(steps)
            .wrapping_add(n);
        n += 1;
    }
    acc
}

/// Fixed Galperin input (manifest `[input]`): n = 5 → mass ratio 100^4, collision
/// count floor(π·10^4) = 31415. The largest supported non-bignum case (n ≤ 5).
const GALPERIN_INPUT: i64 = 5;

/// Round-to-nearest signed integer divide, byte-for-byte the .mind `round_div_d`.
fn ref_round_div_d(x: i64, d: i64) -> i64 {
    if x >= 0 {
        (2 * x + d) / (2 * d)
    } else {
        (2 * x - d) / (2 * d)
    }
}

/// Independent in-process oracle for `galperin_collisions` — the identical
/// integer phase-space reflection process in the identical source/control-flow
/// order (including the `guard = 4_000_000` termination sentinel). All products
/// stay inside i64 for n ≤ 5 (worst ≈ 2e18 < i64::MAX), so plain i64 ops match
/// the kernel exactly; being float-free it is identical on every substrate.
fn ref_galperin_collisions(n: i64) -> i64 {
    if n < 1 {
        return -1;
    }
    if n > 5 {
        return -1;
    }
    let mut k: i64 = 1;
    let mut e: i64 = 1;
    while e < n {
        k *= 10;
        e += 1;
    }
    let scale: i64 = 1_000_000;
    let dd: i64 = k * k + 1;
    let mut u: i64 = (0 - k) * scale;
    let mut w: i64 = 0;
    let mut count: i64 = 0;
    let mut turn: i64 = 1;
    let mut guard: i64 = 0;
    while guard < 4_000_000 {
        if turn == 1 {
            if u < w * k {
                let a = (k * k - 1) * u;
                let b = (2 * k) * w;
                let nu = a + b;
                let c = (2 * k) * u;
                let d2 = (1 - k * k) * w;
                let nw = c + d2;
                u = ref_round_div_d(nu, dd);
                w = ref_round_div_d(nw, dd);
                count += 1;
                turn = 0;
            } else {
                guard = 4_000_000;
            }
        } else if w < 0 {
            w = 0 - w;
            count += 1;
            turn = 1;
        } else {
            guard = 4_000_000;
        }
        guard += 1;
    }
    count
}

#[test]
fn collatz_reproducibility_gate() {
    let id = "collatz";

    let Some(so) = build_collatz_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen collatz workload .so") };
    let collatz: Symbol<CollatzFn> =
        unsafe { lib.get(b"collatz_hash").expect("collatz_hash symbol") };

    let (lo, hi) = COLLATZ_INPUTS;
    let result = unsafe { collatz(lo, hi) };

    // Within-run exactness vs the independent Rust Collatz oracle.
    let oracle = ref_collatz_hash(lo, hi);
    assert_eq!(
        result, oracle,
        "{id}: Collatz hash diverged from the in-process oracle within a run \
         (kernel={result}, oracle={oracle})"
    );

    pin_or_bless(id, &canonical_hash(result), result);
}

#[test]
fn galperin_pi_reproducibility_gate() {
    let id = "galperin-pi";

    let Some(so) = build_galperin_so() else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(so).expect("dlopen galperin workload .so") };
    let galperin: Symbol<GalperinFn> = unsafe {
        lib.get(b"galperin_collisions")
            .expect("galperin_collisions symbol")
    };

    let result = unsafe { galperin(GALPERIN_INPUT) };

    // Within-run exactness vs the independent Rust billiard-π oracle. n=5 must
    // recover floor(π·10^4) = 31415 — proves the reflection process actually ran,
    // not a trivial early return.
    let oracle = ref_galperin_collisions(GALPERIN_INPUT);
    assert_eq!(
        result, oracle,
        "{id}: Galperin collision count diverged from the in-process oracle within \
         a run (kernel={result}, oracle={oracle})"
    );
    assert_eq!(
        result, 31415,
        "{id}: galperin_collisions(5) = {result}, expected 31415 (floor(π·10^4)) — \
         the demo's headline identity broke"
    );

    pin_or_bless(id, &canonical_hash(result), result);
}

// =============================================================================
// examples/quant/* — cross-substrate byte-identity fixtures (quant-*).
//
// Ten deterministic scalar f64 pricing kernels, one per examples/quant/<name>
// package, each built to its OWN `.so` via `build_quant_so` (the generic
// examples/-tree builder above) because mindc 0.10.2 cannot carry an f64
// across a module boundary (roadmap 17.1) — every quant example is therefore
// fully self-contained, exactly like `examples/quant/*/main.mind` documents.
//
// Every fixture evaluates its kernel over a SMALL (8-point) fixed deterministic
// input grid of exact-representable f64 constants, folds the grid to one hash
// via `canonical_hash_f64_grid` (manifest `output_encoding =
// "f64_bits_i64_le_grid"`), and pins that hash to the committed per-substrate
// reference — same `pin_or_bless` mechanism every other fixture in this file
// uses. See each `tests/cross_substrate_identity/quant-*/manifest.toml` for
// the exact grid, the per-fixture avx2==neon argument, and — for the five
// fixtures with no within-run Rust oracle — the explicit statement of why.
//
// WITHIN-RUN ORACLE COVERAGE. Five of the ten fixtures (black_scholes,
// bond_curve, greeks_higher_order, implied_vol_surface, portfolio_risk) are
// closed-form / fixed-loop-order scalar chains with a straightforward,
// faithful Rust port below (`ref_*`), checked byte-exact against the kernel
// EVERY run, not merely at bless time. The other five (american_binomial,
// barrier_analytic, crr_binomial, implied_vol, monte_carlo_gbm) have NO Rust
// oracle here — each manifest states exactly why (a lattice buffer-walk, a
// large closed-form transcription, or an input-dependent root-find/RNG stream
// whose faithful port would be a large surface of its own) — and rely on the
// pinned committed hash alone, as the remaining gate.
//
// --- shared math kernel oracle, ported verbatim from
// examples/quant/black_scholes/main.mind's KERNEL BEGIN..END block (every
// quant example vendors a byte-identical copy of this block — see
// quant_shared_kernels_byte_identical_across_examples, which enforces that byte-identity
// across the ten .mind sources; this Rust port targets that one shared
// definition, not ten separate ones). Every step below reproduces the MIND
// kernel's exact op order and constants; `sqrt` is MIND's `llvm.intr.sqrt`
// (IEEE-754 correctly-rounded), matching Rust's `f64::sqrt`.

// Ported for parity with the MIND KERNEL BEGIN..END block, though none of the
// oracles below happen to call it directly (each oracle's own branch/qNaN
// checks make absf unnecessary at the call sites used here).
#[allow(dead_code)]
fn ref_absf(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn ref_inf() -> f64 {
    let mut v: f64 = 1.0;
    let mut i: i64 = 0;
    while i < 350 {
        v *= 10.0;
        i += 1;
    }
    v
}

fn ref_canonical_qnan() -> f64 {
    f64::from_bits(9221120237041090560u64)
}

fn ref_rnd_to_i64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        0 - ((0.5 - x) as i64)
    }
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_det_exp(x: f64) -> f64 {
    let ln2_hi: f64 = 0.6931467056274414;
    let ln2_lo: f64 = 0.00000047493250390941725;
    let inv_ln2: f64 = 1.4426950408889634;
    let k: i64 = ref_rnd_to_i64(x * inv_ln2);
    let kf: f64 = k as f64;
    let r: f64 = (x - kf * ln2_hi) - kf * ln2_lo;
    let mut t: f64 = 0.0000000000007647163731819816;
    t = 0.00000000016059043836821613 + r * t;
    t = 0.00000000208767569878681 + r * t;
    t = 0.00000002505210838544172 + r * t;
    t = 0.0000002755731922398589 + r * t;
    t = 0.0000027557319223985893 + r * t;
    t = 0.0000248015873015873 + r * t;
    t = 0.0001984126984126984 + r * t;
    t = 0.001388888888888889 + r * t;
    t = 0.008333333333333333 + r * t;
    t = 0.041666666666666664 + r * t;
    t = 0.16666666666666666 + r * t;
    t = 0.5 + r * t;
    t = 1.0 + r * t;
    t = 1.0 + r * t;
    let mut y: f64 = t * 1.0;
    let mut j: i64 = 0;
    if k >= 0 {
        while j < k {
            y *= 2.0;
            j += 1;
        }
    } else {
        while j < (0 - k) {
            y /= 2.0;
            j += 1;
        }
    }
    y
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_dm_exp(x: f64) -> f64 {
    if x != x {
        return x;
    }
    if x >= 709.7827128933841 {
        return ref_inf();
    }
    if x <= 0.0 - 745.1332191019412 {
        return 0.0;
    }
    ref_det_exp(x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_det_log_core(x: f64) -> f64 {
    let ln2: f64 = 0.6931471805599453;
    let mut k: i64 = 0;
    let mut m: f64 = x * 1.0;
    while m >= 1.4142135623730951 {
        m /= 2.0;
        k += 1;
    }
    while m < 0.7071067811865476 {
        m *= 2.0;
        k -= 1;
    }
    let s: f64 = (m - 1.0) / (m + 1.0);
    let s2: f64 = s * s;
    let mut acc: f64 = 0.0;
    let mut n: i64 = 29;
    while n >= 1 {
        acc = 1.0 / (n as f64) + s2 * acc;
        n -= 2;
    }
    (k as f64) * ln2 + 2.0 * s * acc
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_dm_log(x: f64) -> f64 {
    if x != x {
        return x;
    }
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0 - ref_inf();
    }
    if x >= ref_inf() {
        return ref_inf();
    }
    ref_det_log_core(x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_dm_sqrt(x: f64) -> f64 {
    if x != x {
        return ref_canonical_qnan();
    }
    if x < 0.0 {
        return ref_canonical_qnan();
    }
    x.sqrt()
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_erf_series(x: f64) -> f64 {
    let two_over_sqrtpi: f64 = 1.1283791670955126;
    let x2: f64 = x * x;
    let mut t: f64 = x * 1.0;
    let mut s: f64 = x * 1.0;
    let mut n: i64 = 1;
    while n < 60 {
        t = t * (0.0 - x2) / (n as f64);
        s += t / ((2 * n + 1) as f64);
        n += 1;
    }
    two_over_sqrtpi * s
}

fn ref_erfc_cf(x: f64) -> f64 {
    let sqrtpi: f64 = 1.7724538509055159;
    let mut acc: f64 = 0.0;
    let mut n: i64 = 200;
    while n >= 1 {
        acc = ((n as f64) * 0.5) / (x + acc);
        n -= 1;
    }
    ref_det_exp(0.0 - x * x) / sqrtpi / (x + acc)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_dm_erfc(x: f64) -> f64 {
    if x != x {
        return x;
    }
    if x >= ref_inf() {
        return 0.0;
    }
    if x <= 0.0 - ref_inf() {
        return 2.0;
    }
    if x < 0.0 {
        return 2.0 - ref_dm_erfc(0.0 - x);
    }
    if x < 1.0 {
        return 1.0 - ref_erf_series(x);
    }
    ref_erfc_cf(x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_dm_norm_cdf(x: f64) -> f64 {
    if x != x {
        return x;
    }
    let inv_sqrt2: f64 = 0.7071067811865476;
    0.5 * ref_dm_erfc(0.0 - x * inv_sqrt2)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_dm_norm_pdf(x: f64) -> f64 {
    if x != x {
        return x;
    }
    let inv_sqrt_2pi: f64 = 0.3989422804014327;
    inv_sqrt_2pi * ref_det_exp(0.0 - 0.5 * x * x)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_bs_valid(s: f64, k: f64, sigma: f64, t: f64) -> bool {
    !(s != s
        || k != k
        || sigma != sigma
        || t != t
        || s <= 0.0
        || k <= 0.0
        || sigma <= 0.0
        || t <= 0.0)
}

fn ref_bs_d1(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sq: f64 = sigma * ref_dm_sqrt(t);
    (ref_dm_log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sq
}

// --- quant-black-scholes: ref_bs_call ---------------------------------------
fn ref_bs_call(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if !ref_bs_valid(s, k, sigma, t) {
        return ref_canonical_qnan();
    }
    let d1: f64 = ref_bs_d1(s, k, r, q, sigma, t);
    let d2: f64 = d1 - sigma * ref_dm_sqrt(t);
    let df_r: f64 = ref_dm_exp(0.0 - r * t);
    let df_q: f64 = ref_dm_exp(0.0 - q * t);
    s * df_q * ref_dm_norm_cdf(d1) - k * df_r * ref_dm_norm_cdf(d2)
}

const QUANT_BLACK_SCHOLES_GRID: [(f64, f64, f64, f64, f64, f64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
    (120.0, 100.0, 0.05, 0.0, 0.2, 1.0),
    (80.0, 100.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 100.0, 0.05, 0.03, 0.25, 2.0),
    (50.0, 100.0, 0.05, 0.0, 0.2, 0.25),
    (100.0, 100.0, 0.01, 0.0, 0.05, 0.5),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.0027397260273972603),
    (100.0, 80.0, 0.05, 0.0, 0.3, 1.5),
];

#[test]
fn quant_black_scholes_reproducibility_gate() {
    let id = "quant-black-scholes";
    let Some(so) = build_quant_so("black_scholes") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-black-scholes .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"bs_call").expect("bs_call symbol") };

    let mut out = Vec::with_capacity(QUANT_BLACK_SCHOLES_GRID.len());
    for &(s, k, r, q, sigma, t) in QUANT_BLACK_SCHOLES_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t) };
        let oracle = ref_bs_call(s, k, r, q, sigma, t);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: bs_call({s},{k},{r},{q},{sigma},{t}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-bond-curve: ref_bond_price ---------------------------------------
// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_bond_valid(face: f64, coupon_rate: f64, ytm: f64, freq: i64, n_periods: i64) -> bool {
    if face != face || coupon_rate != coupon_rate || ytm != ytm {
        return false;
    }
    if freq <= 0 || n_periods <= 0 || face <= 0.0 || coupon_rate < 0.0 {
        return false;
    }
    let f: f64 = freq as f64;
    !(1.0 + ytm / f <= 0.0)
}

fn ref_df_single(ytm: f64, freq: f64) -> f64 {
    1.0 / (1.0 + ytm / freq)
}

fn ref_df_pow(v: f64, i: i64) -> f64 {
    let mut acc: f64 = 1.0 * 1.0;
    let mut j: i64 = 0;
    while j < i {
        acc *= v;
        j += 1;
    }
    acc
}

fn ref_bond_price(face: f64, coupon_rate: f64, ytm: f64, freq: i64, n_periods: i64) -> f64 {
    if !ref_bond_valid(face, coupon_rate, ytm, freq, n_periods) {
        return ref_canonical_qnan();
    }
    let f: f64 = freq as f64;
    let v: f64 = ref_df_single(ytm, f);
    let cpn: f64 = face * coupon_rate / f;
    let mut acc: f64 = 0.0 * 1.0;
    let mut i: i64 = 1;
    while i <= n_periods {
        acc += cpn * ref_df_pow(v, i);
        i += 1;
    }
    acc += face * ref_df_pow(v, n_periods);
    acc
}

const QUANT_BOND_CURVE_GRID: [(f64, f64, f64, i64, i64); 8] = [
    (100.0, 0.05, 0.05, 2, 10),
    (100.0, 0.04, 0.05, 2, 10),
    (1000.0, 0.03, 0.02, 1, 5),
    (100.0, 0.0, 0.05, 2, 10),
    (100.0, 0.06, 0.06, 4, 20),
    (100.0, 0.05, -0.01, 2, 6),
    (100.0, 0.05, 0.05, 12, 3),
    (500.0, 0.045, 0.03, 2, 8),
];

#[test]
fn quant_bond_curve_reproducibility_gate() {
    let id = "quant-bond-curve";
    let Some(so) = build_quant_so("bond_curve") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-bond-curve .so") };
    let f: Symbol<QuantBondPriceFn> = unsafe { lib.get(b"bond_price").expect("bond_price symbol") };

    let mut out = Vec::with_capacity(QUANT_BOND_CURVE_GRID.len());
    for &(face, cr, ytm, freq, n) in QUANT_BOND_CURVE_GRID.iter() {
        let got = unsafe { f(face, cr, ytm, freq, n) };
        let oracle = ref_bond_price(face, cr, ytm, freq, n);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: bond_price({face},{cr},{ytm},{freq},{n}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-greeks-higher-order: ref_hog_vanna -------------------------------
fn ref_hog_vanna(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    if !ref_bs_valid(s, k, sigma, t) {
        return ref_canonical_qnan();
    }
    let d1: f64 = ref_bs_d1(s, k, r, q, sigma, t);
    let d2: f64 = d1 - sigma * ref_dm_sqrt(t);
    0.0 - ref_dm_exp(0.0 - q * t) * ref_dm_norm_pdf(d1) * d2 / sigma
}

#[test]
fn quant_greeks_higher_order_reproducibility_gate() {
    let id = "quant-greeks-higher-order";
    let Some(so) = build_quant_so("greeks_higher_order") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-greeks-higher-order .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe { lib.get(b"hog_vanna").expect("hog_vanna symbol") };

    let mut out = Vec::with_capacity(QUANT_BLACK_SCHOLES_GRID.len());
    for &(s, k, r, q, sigma, t) in QUANT_BLACK_SCHOLES_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t) };
        let oracle = ref_hog_vanna(s, k, r, q, sigma, t);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: hog_vanna({s},{k},{r},{q},{sigma},{t}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-implied-vol-surface: ref_svi_total_variance ----------------------
fn ref_svi_w_min_raw(a: f64, b: f64, rho: f64, p_sigma: f64) -> f64 {
    a + b * p_sigma * ref_dm_sqrt(1.0 - rho * rho)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_svi_params_valid(a: f64, b: f64, rho: f64, m: f64, p_sigma: f64) -> bool {
    if a != a || b != b || rho != rho || m != m || p_sigma != p_sigma {
        return false;
    }
    if b < 0.0 || rho <= 0.0 - 1.0 || rho >= 1.0 || p_sigma <= 0.0 {
        return false;
    }
    !(ref_svi_w_min_raw(a, b, rho, p_sigma) < 0.0)
}

// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_svi_total_variance(k: f64, a: f64, b: f64, rho: f64, m: f64, p_sigma: f64) -> f64 {
    if !ref_svi_params_valid(a, b, rho, m, p_sigma) {
        return ref_canonical_qnan();
    }
    if k != k {
        return ref_canonical_qnan();
    }
    let km: f64 = k - m;
    a + b * (rho * km + ref_dm_sqrt(km * km + p_sigma * p_sigma))
}

const QUANT_SVI_GRID: [(f64, f64, f64, f64, f64, f64); 8] = [
    (0.0, 0.04, 0.4, -0.3, 0.0, 0.2),
    (0.1, 0.04, 0.4, -0.3, 0.0, 0.2),
    (-0.1, 0.04, 0.4, -0.3, 0.0, 0.2),
    (0.2, 0.05, 0.3, -0.5, 0.05, 0.15),
    (-0.2, 0.05, 0.3, -0.5, 0.05, 0.15),
    (0.0, 0.02, 0.2, 0.0, 0.0, 0.3),
    (0.05, 0.03, 0.25, -0.2, -0.02, 0.1),
    (0.3, 0.04, 0.4, -0.3, 0.0, 0.2),
];

#[test]
fn quant_implied_vol_surface_reproducibility_gate() {
    let id = "quant-implied-vol-surface";
    let Some(so) = build_quant_so("implied_vol_surface") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol-surface .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe {
        lib.get(b"svi_total_variance")
            .expect("svi_total_variance symbol")
    };

    let mut out = Vec::with_capacity(QUANT_SVI_GRID.len());
    for &(k, a, b, rho, m, p_sigma) in QUANT_SVI_GRID.iter() {
        let got = unsafe { f(k, a, b, rho, m, p_sigma) };
        let oracle = ref_svi_total_variance(k, a, b, rho, m, p_sigma);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: svi_total_variance({k},{a},{b},{rho},{m},{p_sigma}) diverged from the \
             Rust oracle within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-portfolio-risk: ref_two_asset_vol ---------------------------------
// Literal transcription of the .mind kernel: constants and NaN checks kept as written.
#[allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::neg_cmp_op_on_partial_ord
)]
fn ref_two_asset_vol(w1: f64, w2: f64, s1: f64, s2: f64, rho: f64) -> f64 {
    if s1 != s1 || s2 != s2 || rho != rho {
        return ref_canonical_qnan();
    }
    if s1 < 0.0 || s2 < 0.0 {
        return ref_canonical_qnan();
    }
    if rho < 0.0 - 1.0 || rho > 1.0 {
        return ref_canonical_qnan();
    }
    let v: f64 = w1 * w1 * s1 * s1 + w2 * w2 * s2 * s2 + 2.0 * w1 * w2 * rho * s1 * s2;
    ref_dm_sqrt(v)
}

const QUANT_PORTFOLIO_RISK_GRID: [(f64, f64, f64, f64, f64); 8] = [
    (0.5, 0.5, 0.2, 0.3, 0.4),
    (0.7, 0.3, 0.15, 0.25, -0.2),
    (0.3, 0.7, 0.2, 0.2, 1.0),
    (0.5, 0.5, 0.2, 0.3, -1.0),
    (1.0, 0.0, 0.2, 0.3, 0.0),
    (0.6, 0.4, 0.1, 0.5, 0.6),
    (0.25, 0.75, 0.3, 0.1, -0.5),
    (0.9, 0.1, 0.18, 0.35, 0.1),
];

#[test]
fn quant_portfolio_risk_reproducibility_gate() {
    let id = "quant-portfolio-risk";
    let Some(so) = build_quant_so("portfolio_risk") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-portfolio-risk .so") };
    let f: Symbol<QuantF64x5Fn> =
        unsafe { lib.get(b"two_asset_vol").expect("two_asset_vol symbol") };

    let mut out = Vec::with_capacity(QUANT_PORTFOLIO_RISK_GRID.len());
    for &(w1, w2, s1, s2, rho) in QUANT_PORTFOLIO_RISK_GRID.iter() {
        let got = unsafe { f(w1, w2, s1, s2, rho) };
        let oracle = ref_two_asset_vol(w1, w2, s1, s2, rho);
        assert_eq!(
            got.to_bits(),
            oracle.to_bits(),
            "{id}: two_asset_vol({w1},{w2},{s1},{s2},{rho}) diverged from the Rust oracle \
             within a run (kernel={got} bits={:#018x}, oracle={oracle} bits={:#018x})",
            got.to_bits(),
            oracle.to_bits()
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-american-binomial: NO within-run Rust oracle (see manifest) -----
const QUANT_AMERICAN_BINOMIAL_GRID: [(f64, f64, f64, f64, f64, f64, i64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100),
    (90.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100),
    (110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100),
    (100.0, 100.0, 0.1, 0.02, 0.3, 0.5, 50),
    (100.0, 100.0, 0.05, 0.0, 0.4, 2.0, 200),
    (80.0, 100.0, 0.08, 0.01, 0.25, 1.0, 100),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.1, 30),
    (120.0, 100.0, 0.06, 0.02, 0.35, 1.5, 150),
];

#[test]
fn quant_american_binomial_reproducibility_gate() {
    let id = "quant-american-binomial";
    let Some(so) = build_quant_so("american_binomial") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-american-binomial .so") };
    let f: Symbol<QuantF64x6I64Fn> = unsafe {
        lib.get(b"bopm_american_put")
            .expect("bopm_american_put symbol")
    };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (lattice backward-induction buffer walk). Every result must still be
    // FINITE (not NaN/inf), which at least proves the lattice ran to
    // completion rather than degenerating.
    let mut out = Vec::with_capacity(QUANT_AMERICAN_BINOMIAL_GRID.len());
    for &(s, k, r, q, sigma, t, n) in QUANT_AMERICAN_BINOMIAL_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t, n) };
        assert!(
            got.is_finite(),
            "{id}: bopm_american_put({s},{k},{r},{q},{sigma},{t},{n}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-barrier-analytic: NO within-run Rust oracle (see manifest) ------
const QUANT_BARRIER_ANALYTIC_GRID: [(f64, f64, f64, f64, f64, f64, f64); 8] = [
    (100.0, 100.0, 90.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 90.0, 85.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 110.0, 95.0, 0.05, 0.0, 0.25, 0.5),
    (100.0, 100.0, 95.0, 0.05, 0.02, 0.3, 2.0),
    (100.0, 100.0, 99.0, 0.05, 0.0, 0.2, 1.0),
    (100.0, 105.0, 90.0, 0.03, 0.0, 0.2, 1.0),
    (100.0, 100.0, 80.0, 0.05, 0.0, 0.2, 0.25),
    (100.0, 95.0, 92.0, 0.05, 0.01, 0.22, 1.0),
];

#[test]
fn quant_barrier_analytic_reproducibility_gate() {
    let id = "quant-barrier-analytic";
    let Some(so) = build_quant_so("barrier_analytic") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-barrier-analytic .so") };
    let f: Symbol<QuantF64x7Fn> = unsafe { lib.get(b"bar_doc").expect("bar_doc symbol") };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (large Reiner-Rubinstein block transcription).
    let mut out = Vec::with_capacity(QUANT_BARRIER_ANALYTIC_GRID.len());
    for &(s, k, h, r, q, sigma, t) in QUANT_BARRIER_ANALYTIC_GRID.iter() {
        let got = unsafe { f(s, k, h, r, q, sigma, t) };
        assert!(
            got.is_finite(),
            "{id}: bar_doc({s},{k},{h},{r},{q},{sigma},{t}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-crr-binomial: NO within-run Rust oracle (see manifest) ---------
#[allow(clippy::type_complexity)]
const QUANT_CRR_BINOMIAL_GRID: [(f64, f64, f64, f64, f64, f64, i64, i64, i64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, 1, 0),
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 100, 0, 1),
    (90.0, 100.0, 0.05, 0.0, 0.2, 1.0, 200, 1, 0),
    (110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 200, 0, 1),
    (100.0, 100.0, 0.1, 0.02, 0.3, 0.5, 50, 1, 1),
    (80.0, 100.0, 0.08, 0.01, 0.25, 1.0, 100, 0, 0),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.1, 30, 1, 0),
    (120.0, 100.0, 0.06, 0.02, 0.35, 1.5, 150, 0, 1),
];

#[test]
fn quant_crr_binomial_reproducibility_gate() {
    let id = "quant-crr-binomial";
    let Some(so) = build_quant_so("crr_binomial") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-crr-binomial .so") };
    let f: Symbol<QuantCrrPriceFn> = unsafe { lib.get(b"crr_price").expect("crr_price symbol") };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (lattice backward-induction buffer walk, distinct implementation from
    // american_binomial).
    let mut out = Vec::with_capacity(QUANT_CRR_BINOMIAL_GRID.len());
    for &(s, k, r, q, sigma, t, n, is_call, american) in QUANT_CRR_BINOMIAL_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t, n, is_call, american) };
        assert!(
            got.is_finite(),
            "{id}: crr_price({s},{k},{r},{q},{sigma},{t},{n},{is_call},{american}) = {got} \
             is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-implied-vol: NO within-run Rust oracle (see manifest) ----------
const QUANT_IMPLIED_VOL_GRID: [(f64, f64, f64, f64, f64, f64); 8] = [
    (10.450583572185568, 100.0, 100.0, 0.05, 0.0, 1.0),
    (26.169043946847296, 120.0, 100.0, 0.05, 0.0, 1.0),
    (1.859419572812186, 80.0, 100.0, 0.05, 0.0, 1.0),
    (14.883718105064894, 100.0, 100.0, 0.05, 0.03, 2.0),
    (
        0.42448595543281803,
        100.0,
        100.0,
        0.05,
        0.0,
        0.0027397260273972603,
    ),
    (5.0, 100.0, 100.0, 0.05, 0.0, 1.0),
    (20.0, 100.0, 100.0, 0.05, 0.0, 1.0),
    (2.5, 90.0, 100.0, 0.03, 0.0, 0.5),
];

#[test]
fn quant_implied_vol_reproducibility_gate() {
    let id = "quant-implied-vol";
    let Some(so) = build_quant_so("implied_vol") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-implied-vol .so") };
    let f: Symbol<QuantF64x6Fn> = unsafe {
        lib.get(b"implied_vol_call")
            .expect("implied_vol_call symbol")
    };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (input-dependent Newton/bisection branch sequence over a fixed
    // iteration count).
    let mut out = Vec::with_capacity(QUANT_IMPLIED_VOL_GRID.len());
    for &(price, s, k, r, q, t) in QUANT_IMPLIED_VOL_GRID.iter() {
        let got = unsafe { f(price, s, k, r, q, t) };
        assert!(
            got.is_finite(),
            "{id}: implied_vol_call({price},{s},{k},{r},{q},{t}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

// --- quant-monte-carlo-gbm: NO within-run Rust oracle (see manifest) ------
#[allow(clippy::type_complexity)]
const QUANT_MONTE_CARLO_GBM_GRID: [(f64, f64, f64, f64, f64, f64, i64, i64); 8] = [
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 12345),
    (100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 7),
    (90.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 42),
    (110.0, 100.0, 0.05, 0.0, 0.2, 1.0, 500, 99),
    (100.0, 100.0, 0.1, 0.02, 0.3, 0.5, 500, 2024),
    (80.0, 100.0, 0.08, 0.01, 0.25, 1.0, 500, 555),
    (100.0, 100.0, 0.05, 0.0, 0.2, 0.1, 500, 1),
    (120.0, 100.0, 0.06, 0.02, 0.35, 1.5, 500, 8675309),
];

#[test]
fn quant_monte_carlo_gbm_reproducibility_gate() {
    let id = "quant-monte-carlo-gbm";
    let Some(so) = build_quant_so("monte_carlo_gbm") else {
        return; // toolchain shadowed — self-skip
    };
    let lib = unsafe { Library::new(&so).expect("dlopen quant-monte-carlo-gbm .so") };
    let f: Symbol<QuantMcCallFn> = unsafe { lib.get(b"mc_call").expect("mc_call symbol") };

    // No within-run Rust oracle for this fixture — see manifest.toml header
    // (in-kernel deterministic LCG path generator, not the host RNG).
    let mut out = Vec::with_capacity(QUANT_MONTE_CARLO_GBM_GRID.len());
    for &(s, k, r, q, sigma, t, npath, seed) in QUANT_MONTE_CARLO_GBM_GRID.iter() {
        let got = unsafe { f(s, k, r, q, sigma, t, npath, seed) };
        assert!(
            got.is_finite(),
            "{id}: mc_call({s},{k},{r},{q},{sigma},{t},{npath},{seed}) = {got} is not finite"
        );
        out.push(got);
    }
    pin_or_bless_hash(id, &canonical_hash_f64_grid(&out));
}

/// `pin_or_bless` (the i64-result helper above) takes an `i64 result` for its
/// error message; the quant fixtures have no single scalar result (they fold
/// an 8-point grid), so this is the grid-shaped sibling: same bless/pin
/// mechanism, no `result_i64` in the panic text.
fn pin_or_bless_hash(id: &str, computed: &str) {
    let substrate = host_substrate();
    if std::env::var("MIND_BENCH_BLESS").is_ok() {
        emit_bless(id, substrate, computed);
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
             bless it with MIND_BENCH_BLESS=1 if this host is canonical."
        ),
    }
    common::xsi_gate::record_measured(id);
}
