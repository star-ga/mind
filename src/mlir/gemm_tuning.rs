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

//! Cache-blocking knobs for the fused Q16.16 GEMM macro-kernel.
//!
//! These constants drive the BLIS-style loop nest
//! `jc → ic → pc → jr → ir → microkernel` emitted by
//! `emit_vec_matmul_mm_q16` (single-thread) and `emit_mm_q16_row_band` (the
//! per-thread band kernel the multithreaded wrapper composes). They are the
//! ONLY thing an autotuner needs to sweep — the kernel logic is invariant to
//! the values; only locality (and therefore throughput) changes.
//!
//! ## Why blocking and why these numbers (Haswell: L1d 32 KiB, L2 256 KiB/core)
//!
//! Without K-blocking the inner reduction streams a `K`-deep B column panel for
//! every output tile, so the working set scales with `K`; at `K = 512` the
//! K×N B panel is 512 KiB and falls out of L2, collapsing throughput
//! (measured 4.5 → 2.56 GMAC/s, 64³ → 512³). Splitting the K reduction into
//! `KC`-deep panels and accumulating panel-partials in an i64 C-scratch tile
//! bounds the resident set independently of `K`.
//!
//! Bit-identity holds because each product is shifted `>> 16` to a fixed i64
//! BEFORE summing, and i64 add is associative + commutative: the panel-partial
//! sums add to exactly the full-K sum, truncated once at the end. Packing the
//! A/B panels into contiguous scratch is pure data movement, so it does not
//! perturb a single output byte either.
//!
//! Working-set budget at the chosen point (`MC=64, KC=256, NC=128`):
//!   * i64 C-scratch tile  `MC*NC*8 = 64 KiB`  → resides across the pc loop
//!   * packed B panel       `KC*NC*4 = 128 KiB` → reused across the ic loop
//!   * packed A panel       `MC*KC*4 = 64 KiB`  → reused across the jr loop
//! The packed B panel (128 KiB) is the largest reused tile and the A panel +
//! C-scratch (128 KiB together) fit alongside it inside the 256 KiB L2; the
//! `MR×NR` microkernel inputs stream from L1. Total per-call scratch is
//! `64 + 128 + 64 = 256 KiB`, allocated once per kernel invocation on the
//! stack (`llvm.alloca`, compile-time-constant extent — statically reserved,
//! no pointer bits leak into the artifact). In the multithreaded path each
//! worker is a distinct function activation, so its scratch is private by
//! construction (no shared accumulator, no data race).
//!
//! An autotuner may sweep `Q16_MC / Q16_KC / Q16_NC` freely; the emitter
//! handles every `M%MC / N%NC / K%KC / N%NR / M%MR` remainder. `Q16_MR` and
//! `Q16_NR` are the register-tile dimensions (4 rows × an 8-wide i64 column
//! vector) and are wired into the microkernel's accumulator shape; changing
//! them requires the matching accumulator-vector edits, so they are pinned.

/// Row block — rows of C (and of the packed A panel) held resident per `ic`
/// step. `MC * NC * 8` (the i64 C-scratch) and `MC * KC * 4` (packed A) must
/// fit L2 alongside the packed B panel.
pub const Q16_MC: usize = 64;

/// K panel depth — the reduction is split into `KC`-deep panels so the resident
/// B/A working set is independent of the full `K`. THE knob that was missing.
pub const Q16_KC: usize = 256;

/// Column block — columns of C (and of the packed B panel) held resident per
/// `jc` step. `KC * NC * 4` (packed B) is the largest reused tile.
pub const Q16_NC: usize = 128;

/// Size-threshold for the BLIS-vs-simple GEMM dispatch (Q16 and int8 tiers).
///
/// The BLIS macro-kernel (`emit_mm_q16_blocked` / `emit_mm_i8_blocked`) pays a
/// fixed packing + C-scratch-init cost per call that only amortises once the
/// problem is large enough that the K-deep B panel would otherwise spill cache.
/// For small shapes — where the whole problem already fits L1/L2 — that
/// overhead dominates and the simpler fused outer-product kernel
/// (`emit_mm_q16_row_band` over the full `[0, M)` row range, the pre-BLIS form)
/// is faster. The emitted GEMM therefore branches at runtime on
/// `max(M, N, K) < Q16_BLIS_MIN_DIM`: below the threshold it runs the simple
/// kernel, at or above it runs the BLIS-blocked kernel.
///
/// CRITICAL: the two paths are BYTE-IDENTICAL by construction — same per-term
/// `arith.shrsi >> 16` (int8: none), same associative i64 reduction, single
/// i64→i32 truncation at the store. The dispatch only changes tiling, never the
/// math, so the cross-substrate canary hashes (`92e2cb75` Q16, `917d353b` int8)
/// are independent of which path a given shape takes. An autotuner may sweep
/// this value freely without perturbing a single output byte.
pub const Q16_BLIS_MIN_DIM: usize = 256;

/// Register-tile rows: independent `MR` accumulator chains in the microkernel.
/// Pinned — the accumulator shape in the emitter depends on it.
pub const Q16_MR: usize = 4;

/// Register-tile columns: width of the `vector<NRxi64>` column accumulator.
/// Pinned — the accumulator vector type in the emitter depends on it.
pub const Q16_NR: usize = 8;

// ── int8 "det.igemm" tier ────────────────────────────────────────────────────
//
// The fused int8 GEMM (`__mind_blas_matmul_mm_i8_v`) reuses the EXACT BLIS loop
// nest as the Q16 kernel, with two differences that do not change the working-set
// budget: (1) A/B source elements are 1-byte i8 (sign-extended once during the
// pack into the i32 packed panels — the panels stay i32, identical extent to the
// Q16 panels), and (2) the microkernel performs a pure integer multiply-add with
// NO `>> 16` shift (int8 is integer, not fixed-point). The C-scratch accumulates
// i64 and is truncated to i32 once at the store. Because the packed panels and
// C-scratch keep the Q16 element sizes (i32 / i64), the L2 budget reasoned about
// for Q16_MC/KC/NC holds unchanged, so the int8 tier reuses the same numbers.
//
// Byte-identity / overflow: each product is `(i32)a*(i32)b` with |a|,|b| ≤ 128,
// so |product| ≤ 16384 (well under 2^15). The running per-element sum is carried
// in i64 throughout (the C-scratch is i64), so the full-K reduction is exact for
// any K up to the i64 range; the single i64→i32 truncation at the store yields
// the exact int32 sum (it fits i32 for K ≤ ~2^31/2^14 ≈ 131072 with worst-case
// saturated inputs, far beyond realistic K). i64 add is associative + commutative
// ⇒ any tiling / lane order gives the identical int32 result, and the SAME MLIR
// lowers to `vpmaddwd` (x86 AVX2) or `SDOT`/`SMMLA` (aarch64) producing the same
// exact int32 sum — cross-substrate bit-identity by construction.

/// int8 tier row block — mirrors `Q16_MC`. The packed A panel and i64 C-scratch
/// have the same extent as the Q16 tier (panels stay i32, scratch i64).
pub const I8_MC: usize = 64;

/// int8 tier K-panel depth — mirrors `Q16_KC`.
pub const I8_KC: usize = 256;

/// int8 tier column block — mirrors `Q16_NC`.
pub const I8_NC: usize = 128;

/// int8 tier register-tile rows — mirrors `Q16_MR`. Pinned (accumulator shape).
/// This is the AVX2 `vpmaddwd` rung's row tile and is **hash-frozen** (the
/// `917d353b` int8 cross-substrate canary): do not change without a rebless.
pub const I8_MR: usize = 4;

/// int8 tier register-tile columns — mirrors `Q16_NR`. Pinned (accumulator
/// vector width). AVX2 `vpmaddwd` produces 8 i32 partials per 256-bit op; this
/// is **hash-frozen** (`917d353b`).
pub const I8_NR: usize = 8;

/// VNNI rung register-tile columns — the ZMM `vpdpbusd.512` width. One
/// `vpdpbusd.512` produces 16 i32 partials (16xi32 = 512-bit), so each A-row
/// accumulator is a single ZMM `vector<16xi32>`. This DOUBLES the column-tile
/// width vs the 256-bit form (the ~2× throughput lever). It does not touch the
/// AVX2 canary — only the opt-in `MIND_INTDOT=vnni` path reads it — and it is a
/// pure tiling choice over an associative i64 reduction, so the exact int32 sum
/// (and therefore byte-identity with the AVX2 rung) is unchanged.
pub const I8_VNNI_NR: usize = 16;

/// VNNI rung register-tile rows — number of independent `vpdpbusd` accumulator
/// chains. 8 main chains (+ 1 shared column-sum chain = 9 in flight) hide the
/// 4–5-cycle `vpdpbusd` latency; pure reordering of independent integer adds, so
/// the exact int32 sum is unchanged. `I8_MC` (64) and `I8_NC` (128) stay
/// multiples of these dims, so the C-scratch / packed-panel extents are unchanged.
pub const I8_VNNI_MR: usize = 8;

/// int8 GEMM int-dot rung selector. Selects the vector instruction the int8
/// macro-kernel uses to contract the K dimension. **Every rung produces the
/// byte-identical exact int32 sum** — they differ only in how many K-steps one
/// instruction fuses, never in the value:
///
/// * `Avx2` (default, this box + CI) — `vpmaddwd` (`_mm256_madd_epi16`), 2 K-steps
///   per instruction over NR=8 i32 partials. Lowers to `vpmaddwd` on AVX2 and to
///   `SDOT`/`SMMLA` on aarch64. This is the committed, hash-pinned path.
/// * `Vnni` — AVX-512-VNNI `vpdpbusd` (256-bit form, `@llvm.x86.avx512.vpdpbusd.256`),
///   4 K-steps per instruction over the same NR=8 i32 partials. VPDPBUSD is
///   `u8 × s8 → i32`; our inputs are SIGNED int8, so the kernel applies the exact
///   integer bias identity `Σ aₛ·bₛ = Σ (aₛ+128)·bₛ − 128·Σ bₛ` (the `+128` is the
///   `xor 0x80` byte reinterpretation, `−128·Σ bₛ` is one `vpsubd` of a per-column
///   `Σ bₛ` computed with a `vpdpbusd` against an all-ones u8 vector and shifted
///   `<< 7`). All terms are exact i32 ⇒ the result equals the `Avx2` rung's exact
///   sum bit-for-bit. Requires VNNI hardware (Ice Lake / Sapphire Rapids+) to RUN;
///   the explicit intrinsic still *emits* and disassembles to `vpdpbusd` anywhere.
///
/// Selected at emit time by the `MIND_INTDOT` environment variable (`vnni` ⇒
/// `Vnni`, anything else / unset ⇒ `Avx2`). The default is `Avx2` so this box and
/// CI — and the pinned `917d353b` int8 / `92e2cb75` Q16 cross-substrate hashes —
/// are unchanged. `mlir_build.rs` reads the same selector to add the VNNI target
/// features to the clang invocation when the VNNI rung is active.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntDotMode {
    /// `vpmaddwd` (AVX2) — the default, hash-pinned path.
    Avx2,
    /// `vpdpbusd` (AVX-512-VNNI) with the signed-input bias correction.
    Vnni,
}

impl IntDotMode {
    /// Resolve the int-dot rung from the `MIND_INTDOT` environment variable.
    /// `vnni` (case-insensitive) selects the VPDPBUSD rung; everything else
    /// (including unset) keeps the committed AVX2 `vpmaddwd` default.
    pub fn from_env() -> Self {
        match std::env::var("MIND_INTDOT") {
            Ok(v) if v.trim().eq_ignore_ascii_case("vnni") => IntDotMode::Vnni,
            _ => IntDotMode::Avx2,
        }
    }
}
