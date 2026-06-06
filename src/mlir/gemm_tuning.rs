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

/// int8 tier row block. Held at the champion 64. The row axis is a double dead
/// end — MC=80 (iter 9) and MC=32 (iter 7) both regressed — and the speculative
/// MC=96 trial is reverted here: at MC=96 the i64 C-scratch is `MC*NC*8 =
/// 96*256*8 = 192 KiB`, which together with packed-B spills L2, the same
/// C-scratch-pressure failure mode the larger-MC moves keep hitting. The
/// residency budget is better spent on the COLUMN axis (see `I8_NC`), so MC
/// stays at the measured best. 64 = 16 * MR=4, a clean register-tile multiple.
/// Byte-identity is independent of the row tile: it only repartitions the outer
/// row loop and never reorders an integer product.
pub const I8_MC: usize = 64;

/// int8 tier K-panel depth. Reverted to the champion 128 (the speculative
/// KC=160 wide-NC trial is rolled back here): the KC axis has a SHARP measured
/// peak at 128 — the NC=256 sweep gave 64→12.99, 128→14.13, 256→13.23,
/// 512→13.17, a single-point spike that fell off ~6% in BOTH directions, and
/// the inverse KC=64 (iter 14) and KC=512 (iter 3) points are both logged dead
/// ends. Deepening to 160 trades that known optimum for an unmeasured RMW-traffic
/// argument; with the column peak still un-pinpointed it is the weaker bet, so KC
/// stays at the champion to isolate the NC bisection below. 128 keeps packed-B at
/// `KC*NC*4 = 128*400*4 = 200 KiB` (under the 256 KiB L2). Byte-identity
/// preserved: the i64 panel-partial reduction is associative/commutative, so any
/// K-split sums to the identical exact int32 result.
pub const I8_KC: usize = 128;

/// int8 tier column block. Stepped 384 → 392 — the SMALLEST clean increment
/// int8 tier column block. Stepped 384 → 376 — the SMALLEST clean increment
/// BELOW the champion (one register-tile column-group, NR=8) — probing the
/// lower side of the 384 crest, which is entirely unexplored. The KC=128 column
/// sweep measured NC=256→14.126, NC=384→14.162 (champion), NC=416→13.811 (lost),
/// NC=448→lost, NC=512→13.352: the function rose monotonically 256→384 then fell
/// 384→416, so 384 is a measured maximum — but every point in `(256, 384)` was
/// skipped (the sweep jumped in 32+ strides) and the immediate −8-column
/// neighbour has never been tested. This is the symmetric counterpart to the
/// +8 step above: if 384 is a sharp single-point peak, NC=376 falls off and
/// confirms the edge; if the 384 measurement actually sat on a short plateau
/// whose left shoulder reaches 376, the slightly smaller column block trims
/// cache pressure (each C-scratch RMW pass and packed-B fetch touches fewer
/// lines) at negligible loss of A-strip amortization, and may match or edge the
/// champion with more residency headroom. 376 = 47 * NR=8 is the finest
/// admissible step down. At NC=376 the packed-B panel is `KC*NC*4 =
/// 128*376*4 = 188 KiB` (under the 256 KiB L2) and the i64 C-scratch is
/// `MC*NC*8 = 64*376*8 = 188 KiB`, ~4 KiB less of each reused tile than the 384
/// champion. Byte-identity is unaffected: the i64 panel-partial reduction is
/// associative/commutative, so the column block width never perturbs the exact
/// int32 sum.
pub const I8_NC: usize = 376;

/// int8 tier register-tile rows — mirrors `Q16_MR`. Pinned (accumulator shape).
pub const I8_MR: usize = 4;

/// int8 tier register-tile columns — mirrors `Q16_NR`. Pinned (accumulator
/// vector width).
pub const I8_NR: usize = 8;

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
