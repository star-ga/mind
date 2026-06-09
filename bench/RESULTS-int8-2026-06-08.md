# MIND int8 VNNI GEMM — single-core vs OpenBLAS f32 (2026-06-08)

**Headline: MIND int8 VNNI beats single-core OpenBLAS f32 at ~2.0× @1024³, byte-exact.**

> Do NOT conflate with the GPU result: the "4096³" / cuBLAS numbers are a SEPARATE win — the
> GPU int8 tensor-core kernel (Determinant / MindBLAS) beating cuBLAS int8 1.28× @4096³ (see
> `mind-ecosystem-audit/gpu/`). This file is CPU int8 VNNI @1024³ vs OpenBLAS f32.

## Method
- Single physical core, `taskset`-pinned, median-of-many reps.
- Byte-identity gate PASSED on every config (cross-substrate canaries unchanged,
  keystone 7/7).
- Kernel: `emit_i8_microkernel_vnni` (VPDPBUSD), `MIND_INTDOT=vnni`.

## Cache-blocking sweep (the MC knob)
Root cause of the original deficit: at MC=64 the B-panel was re-streamed 16× →
memory-bound, not compute-bound. Correct row-block size makes the gain fall out.

| MC    | MIND int8 (GMAC/s) | vs OpenBLAS f32 | scratch  |
|-------|--------------------|-----------------|----------|
| 64    | 49.2 (baseline)    | 0.74×           | ~0.6 MB  |
| 128   | —                  | 1.27×           | ~1.2 MB  |
| 256   | 109                | 1.65×           | ~3.5 MB  |
| 512   | 123.3              | 1.85×           | ~5 MB    |
| 768   | 121                | 1.81×           | ~7 MB    |
| 1024  | 134.8 (heap)       | 2.02×           | ~9.5 MB  |

All measured @ **1024³** (the perf shape), single core, median of 64, pinned. OpenBLAS f32
single-core reference: ~66.6 GMAC/s. The MC=1024 row is the **c3-confirmed value WITH the
heap-scratch** (134.8 = 2.02×); the earlier 130.76 was the stack-scratch measurement.

Total internal gain: **49.2 → 134.8 GMAC/s = 2.74×** @1024³, byte-exact throughout — the
byte-identity gate PASSED on all 14 K-straddling/tail shapes at every config.

## Tuning chain (all byte-exact, committed in ~/mind)
- `f380fe9` — I8_KC 256→512 (+6%)
- `276455d` — I8_VNNI_NC 384→768 (+5.5%)
- `5a82cde` — I8_MC 64→256 (1.65×)
- `ccaf7a6` — heap-allocate the GEMM scratch (robust for large MC; unlocks MC≥512;
  one `@free` per `@malloc` on the single return path, leak-free)

Live consts (`src/mlir/gemm_tuning.rs`): `I8_MC=1024`, `I8_KC=512`, `I8_VNNI_NC=768`.

## Supersedes
- `mind-ecosystem-audit/bench/RESULTS-int8-2026-06-08.md` (51.0 GMAC/s = 0.754×, "not
  beating yet") — that was the pre-cache-blocking datum. Now stale.
- The 83.48 / 1.23× intermediate (MC=128 on top of NC=768).

## Diminishing-returns note
Above MC=512 gains flatten (768 dips, 1024 = 2.01×) while scratch cost climbs.
The heap-scratch commit removes the stack-overflow risk that previously capped MC,
so MC=1024 is a safe default. Stop chasing MC alone past here — the next lever is
the microkernel / pipeline level.
