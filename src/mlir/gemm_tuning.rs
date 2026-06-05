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
//! Working-set budget at the chosen point (`MC=128, KC=256, NC=128`), as seen
//! by the 128³ target (so the effective packed-panel K depth is K=128, not the
//! KC=256 cap):
//!   * i64 C-scratch tile  `MC*NC*8   = 128 KiB` → resides across the pc loop
//!   * packed B panel       `K*NC*4    = 64 KiB`  → reused across the ic loop
//!   * packed A panel       `MC*K*4    = 64 KiB`  → reused across the jr loop
//! At `MC=128` the `ic` row-band loop collapses to a single band for M=128, so
//! the packed B panel is built and streamed once instead of being re-read for a
//! second band as it is at `MC=64`. The C-scratch (128 KiB) plus A panel +
//! B panel (128 KiB together) fit inside the 256 KiB L2; the
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
pub const Q16_MC: usize = 128;

/// K panel depth — the reduction is split into `KC`-deep panels so the resident
/// B/A working set is independent of the full `K`. THE knob that was missing.
pub const Q16_KC: usize = 256;

/// Column block — columns of C (and of the packed B panel) held resident per
/// `jc` step. `KC * NC * 4` (packed B) is the largest reused tile.
///
/// At the 128³ target N=128, so `jc` already collapses to a single column block
/// at NC≥128; the touched footprint (C-scratch `MC*N*8`, packed B `K*N*4`) is
/// pinned to the live N=128 regardless of NC, since the emitter fills only the
/// live columns. Doubling NC 128→256 already won (+0.090 GMAC/s) by leaving the
/// resident cache set unchanged (untouched reserved pages never load) while
/// widening the packed-B / C-scratch block stride. This pushes that same proven
/// gradient one further doubling 256→512 to locate its knee: `jc` still
/// collapses to one column block (NC≥128) and the live N=128 footprint is still
/// pinned, so it remains a pure stride/reservation change — the i64 C-scratch
/// row stride grows to `NC*8 = 4096 B` and the packed-B block stride to
/// `NC*4 = 2048 B`, shifting where each live row maps in the L2 sets while never
/// touching the reserved-but-dead columns. Reduction order and Q16.16
/// byte-identity are untouched: NC only blocks columns, never the k-reduction.
pub const Q16_NC: usize = 512;

/// Register-tile rows: independent `MR` accumulator chains in the microkernel.
/// Pinned — the accumulator shape in the emitter depends on it.
pub const Q16_MR: usize = 4;

/// Register-tile columns: width of the `vector<NRxi64>` column accumulator.
/// Pinned — the accumulator vector type in the emitter depends on it.
pub const Q16_NR: usize = 8;
