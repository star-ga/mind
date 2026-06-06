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

/// int8 tier row block. exp143 EXPLORE — the MODEL-DISCRIMINATOR probe at the one unmeasured
/// INTERIOR point of the 2-strip band, settling WHICH law governs the band rather than re-probing
/// its cliff. The full-K insight reframes the constraint: at KC=1024 the i64 C-scratch is a
/// WRITE-ONCE STREAM (exp128 — written once, never read back), so it is NOT a resident tile; the
/// only cache-RESIDENT working set is the pair packed-A (`MC*1024*4`) + packed-B (`4 MiB`, full-N).
/// That reframing splits the data into two rival predictors that AGREE on every measured point but
/// DISAGREE on the interior. (1) STRIP-COUNT model: throughput is a step function of `ceil(1024/MC)`
/// (4→3→2→1 strips = packed-B re-streams), so the whole 2-strip band `MC∈[512,1023]` is a FLAT
/// plateau and 512 is merely the smallest of many tied optima. (2) RESIDENT-PAIR-PRESSURE model:
/// within a fixed strip count throughput slopes monotonically with the resident pair size, so
/// 512 (6 MiB pair) strictly beats 640 (~6.5) beats 768 (~7), a continuous decline — and the
/// champion wins because it is the SMALLEST 2-strip MC, not because the band is flat. Every prior
/// 2-strip datapoint sits at a band ENDPOINT (512 = lower edge; the in-flight 768/896 = upper/cliff
/// edge); NONE bisects the interior, so the two models have never been told apart. MC=640
/// (`ceil(1024/640)=2` strips of 640+384 — SAME packed-B traffic as the champion, zero strip-count
/// penalty) is exactly that bisector: packed-A `640*1024*4 = 2.5 MiB` + packed-B `4 MiB` = 6.5 MiB
/// resident, comfortably mid-L3 (the streaming C-scratch `640*1024*8 = 5 MiB` writes through, not
/// resident). If 640 ≈ 17.303 the plateau is flat → strip-count law, the 1024^3 search is converged
/// and 512 is one of a tied family; if 640 lands strictly below the champion the band slopes →
/// resident-pair law, 512 is the unique edge optimum and NO 2-strip MC can beat it. This is a
/// different QUESTION (which law, not where the cliff), orthogonal to exp141/142's upper-edge cliff
/// localization, not a magnitude retread. Byte-identity is independent of the row tile: MC only
/// repartitions the outer row loop, never reordering a single product, so the seed-42 sha256 gate
/// is untouched.
pub const I8_MC: usize = 640;

/// int8 tier K-panel depth. exp128 EXPLORE pivot — drives the deep-K lever to its ABSOLUTE
/// ENDPOINT: I8_KC 256→1024 = full K, so the pc K-panel loop collapses to a SINGLE iteration
/// (1024/1024=1, zero K-remainder). This is a qualitatively different kernel behavior, not a
/// magnitude retread of exp126's KC step: with one pc panel the i64 C-scratch is no longer
/// read-modify-written at all — it is WRITTEN EXACTLY ONCE with the complete full-K sum, never
/// read back to accumulate a later panel's partial. The whole read-modify-write traffic
/// category over the 768 KiB L3 C-scratch is ELIMINATED, not merely halved. The progression
/// exp126 validated as first-order at MC=256 (KC=128→8 RMW sweeps, KC=256→4) is taken to its
/// floor (KC=512→2, KC=1024→1 = write-only), testing whether removing the champion's single
/// largest L3 traffic stream entirely beats the price: both packed panels balloon into L3
/// (packed-A `256*1024*4 = 1 MiB`, packed-B `1024*384*4 = 1.5 MiB`), but with only one pc
/// panel each is built ONCE per its loop level and never re-streamed across a K-loop that no
/// longer exists. The K-axis endpoint is unprobed in this regime — distinct from every dead
/// KC≥256 test (exp117/exp82/exp72/iter-3, all MC≤128 / small L2-resident C-scratch where
/// killing RMW bought nothing) and the mirror-image endpoint probes on the OTHER axes
/// (exp113 NC=1024 full-N, exp124 MC=512). I8_NC is reverted from the unmeasured exp127
/// NC=512 back to the proven champion's 384 so KC is the SOLE moving axis vs the 15.560
/// champion. Byte-identity is independent of KC: a single full-K pass is the trivial K-split,
/// and the i64 reduction is order-invariant.
pub const I8_KC: usize = 1024;

/// int8 tier column block. exp129 EXPLORE pivot — collapses the jc column loop to a SINGLE
/// full-N block (I8_NC 384→1024 = full N, N%1024=0, zero column epilogue), the SOLE moving axis
/// vs the 15.791 full-K champion (MC=256/KC=1024/NC=384). The mechanism is specific to the new
/// full-K regime exp128 just opened, and is NOT the dead exp113 (NC=1024 at the shallow KC=128
/// corner, where the 8× C-scratch RMW dominated and masked any jc benefit, 14.189). In the
/// full-K nest `jc → ic → pc(=1) → jr → ir`, packed-A (MC×K = `256*1024*4` = 1 MiB) is rebuilt
/// for EVERY jc block, so with NC=384's 3 jc blocks packed-A is re-streamed 3× per ic strip —
/// the dominant L3 traffic now that full-K eliminated the C-scratch RMW stream. Collapsing to a
/// single jc block (NC=1024) cuts that packed-A re-streaming 3×→1× (built once per ic strip),
/// directly attacking the champion's largest remaining L3 read traffic. The cost: packed-B
/// balloons to `1024*1024*4` = 4 MiB and the i64 C-scratch to `256*1024*8` = 2 MiB — but with a
/// single jc block each is built exactly ONCE total and reused across all 4 row strips (minimum
/// possible B traffic), and the ~7 MiB packed-A+packed-B+C working set fits this box's 15 MB
/// Haswell-E L3. The full-N endpoint is unprobed in the full-K/MC=256 regime — distinct from
/// every dead wide-NC test (all at shallow KC where RMW, not packed-A re-streaming, was binding).
/// Byte-identity unaffected: the column-block width never reorders a product nor perturbs the
/// int32 sum.
pub const I8_NC: usize = 1024;

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
