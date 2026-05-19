# RFC 0006: mind-blas — native BLAS surface for MIND

| Field | Value |
|---|---|
| RFC | 0006 |
| Title | mind-blas — native BLAS surface (Track A + Track B increment 1) |
| Status | Track A landed; Track B increment 1 landed |
| Authors | STARGA Inc. |
| Created | 2026-05-18 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0002 (pub-fn C exports), RFC 0005 (pure-MIND std surface) |

## 1. Summary

mind-blas is the native MIND BLAS surface: a small, deliberately scoped set
of dense-vector primitives (`dot`, `matmul`, `dot_l1`, `dot_linf`, plus
Q16.16 variants) that any MIND program can call without depending on an
external BLAS implementation. The same surface targets five sub-backends
(scalar / x86-SIMD / ARM-NEON / CUDA / photonic-projected Q16.16) so the
cross-arch determinism story holds end-to-end without `if cpu == "x86" {
... } else { ... }` branches in user code.

This RFC documents **Track A**, the runtime-support SIMD bridge that lands
the six AVX2-accelerated intrinsics today. Track B (native MLIR vector
dialect lowering in mindc) is the follow-on, thesis-pure implementation
and is sketched at the end of this document.

## 2. Motivation

The mind-nerve A1.5 measurement (commit `mind-nerve@b9b6401`) returned
PARTIAL: pure-MIND tail-recursive scalar matmul on the 11,922-row STARGA
catalog ran at 14.4 ms p50 / 15.1 ms p95 — roughly 40× slower than the
numpy + BLAS reference on the same shape. A single-cycle scalar inner
loop cannot close that gap; AVX2 FMA on a single Skylake-era core supplies
~16 ops/cycle of theoretical throughput. mind-blas is the smallest possible
surface that closes the gap without breaking the L1-substrate /
cross-arch-bit-identity thesis.

A non-goal is to outperform BLAS on every measurement axis. The honest
headline framing (per the bench spec) is: *mind-blas reaches X% of
idealised BLAS while preserving cross-arch Q16.16 bit-identity, which
BLAS does not offer.*

## 3. The five sub-backends

mind-blas is one surface with five interchangeable sub-backends. Every
sub-backend ships all three metric flavors (`l2`, `l1`, `linf`) as
first-class:

| Sub-backend | Trigger | Q16.16 bit-identity | Thesis role |
|---|---|---|---|
| `mind-blas:scalar` | reference / oracle | yes, cross-arch | Q16.16 oracle for the bit-identity gate |
| `mind-blas:simd-x86` | host CPU AVX2 / AVX512 | within substrate | criterion bench target (Track A) |
| `mind-blas:simd-arm` | Apple / ARM NEON | within substrate | mobile reach |
| `mind-blas:cuda` | NVIDIA GPU | within substrate | rfn-mind training, mind-inference |
| `mind-blas:q16-photonic` | photonic-projected | yes, substrate-natural | L1-substrate paper |

Sub-backend selection is **not** part of the public MIND surface at this
RFC. Track B introduces a `@target("simd-x86" | "cuda" | ...)` annotation
that lowers per-call; Track A picks `simd-x86` once at `.so` load time via
`__builtin_cpu_supports` and falls back to `scalar` when AVX2 + FMA are
not both present. This is the same runtime-dispatch pattern OpenBLAS uses
internally; the difference is that mind-blas never reorders integer (Q16.16)
reductions across sub-backends — the bit-identity contract is the moat.

## 4. Track A: the surface

Six intrinsics added to `runtime-support/mind_intrinsics.c`. All take and
return `int64_t` only — the Option-C ABI from RFC 0005. Pointers are
opaque `int64_t` addresses; f32 results are packed into the low 32 bits
of the i64 return value; Q16.16 results are sign-extended into the low
32 bits.

```c
int64_t __mind_blas_dot_f32(int64_t a_addr, int64_t b_addr, int64_t len);
int64_t __mind_blas_dot_l1_f32(int64_t a_addr, int64_t b_addr, int64_t len);
int64_t __mind_blas_dot_linf_f32(int64_t a_addr, int64_t b_addr, int64_t len);

int64_t __mind_blas_matmul_rmajor_f32(
    int64_t w_addr, int64_t x_addr, int64_t y_addr,
    int64_t rows, int64_t cols
);  // y = W · x; row-major W. Returns 0 on OK, -1 on null pointer.

int64_t __mind_blas_dot_q16(int64_t a_addr, int64_t b_addr, int64_t len);
int64_t __mind_blas_dot_l1_q16(int64_t a_addr, int64_t b_addr, int64_t len);
```

Pure-MIND callers go through `std/blas.mind`, which is registered in the
`STDLIB_MIND_SOURCES` table alongside `std.vec` / `std.string` / `std.map` /
`std.io`:

```mind
use std.blas

let s = dot_f32(a_addr, b_addr, len);          // f32 bits in low 32 of i64
let m = dot_l1_q16(a_addr, b_addr, len);       // Q16.16 result sign-ext to i64
let rc = matmul_rmajor_f32(w, x, y, rows, cols);
```

## 5. Implementation: AVX2 + scalar fallback

Every intrinsic ships two implementations:

- A portable C scalar reference. This is the cross-arch oracle: the
  Q16.16 variants must produce a byte-identical i64 to this fallback on
  every backend (task #57).
- An AVX2 path compiled with `__attribute__((target("avx2,fma")))` so the
  translation unit stays buildable on hosts without AVX2 in the build
  flags — the SIMD code is emitted into the object but only executed
  when the runtime dispatcher selects it.

A constructor (`__attribute__((constructor))`) runs at `.so` load time
and sets a `static int mind_blas_use_avx2` flag based on
`__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")`. Every
intrinsic branches on this flag exactly once per call; the branch
predictor pins it after the first call.

### 5.1 Q16.16 SIMD: why `_mm256_mul_epi32`, not `_mm256_mullo_epi32`

The Q16.16 intermediate product needs the full 64 bits. `_mm256_mullo_epi32`
truncates to 32 bits and loses the high half — that destroys both the
numeric value and the bit-identity contract. The implementation issues
two `_mm256_mul_epi32` instructions per iteration (even lanes, then odd
lanes shifted into position) and accumulates into an i64 vector register.

The arithmetic right shift by 16 on each i64 lane needs `_mm256_srai_epi64`,
which is AVX-512 only. AVX2 supplies only `_mm256_srli_epi64` (logical).
The implementation emulates signed shift via `_mm256_cmpgt_epi64(0, x)` to
derive an all-1s sign mask, shifts it into the top 16 bits, and OR-merges
with the logical-shifted value. The final i64 result is bit-identical to
the scalar `x >> 16` under LLVM `ashr` semantics.

### 5.2 Cross-arch bit-identity proof obligation

The smoke test at `tests/blas_smoke.rs` forces the dispatcher to scalar
and to AVX2 in turn and asserts byte-identical i64 results on the Q16.16
intrinsics at lengths {0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1024,
4096, 65537}. This is the cross-arch bit-identity gate (task #57)
restricted to a single x86 host; the full cross-arch contract (Linux x86
vs macOS ARM vs CUDA-x86 vs simulated photonic) is in the bench-spec
publication.

## 6. Numerical contract

| Path | Tolerance | Test |
|---|---|---|
| f32 `dot_f32` | 1e-6 relative on 1M-element vectors | `dot_f32_avx2_within_1e6_on_1m_elements` |
| f32 `dot_l1_f32` | 1e-5 relative on 1024-element vectors | `dot_l1_f32_avx2_matches_scalar_within_tolerance` |
| f32 `dot_linf_f32` | byte-identical (max is associative) | `dot_linf_f32_avx2_byte_identical_to_scalar` |
| f32 `matmul_rmajor_f32` | 1e-5 relative per-row | `matmul_rmajor_f32_close_to_scalar_per_row` |
| Q16.16 `dot_q16` | byte-identical | `dot_q16_byte_identical_scalar_vs_avx2_all_lengths` |
| Q16.16 `dot_l1_q16` | byte-identical | `dot_l1_q16_byte_identical_scalar_vs_avx2_all_lengths` |

## 7. Default-build hot path

This RFC adds zero default-build hot-path code:

- `std/blas.mind` lives behind the existing `std-surface` feature gate
  (the `STDLIB_MIND_SOURCES` table is gated at the registration site, not
  on the `include_str!` line — but every consumer of the table is also
  gated, so the bundled blob is dead code on the default build).
- The six `__mind_blas_*` names are registered in `STD_SURFACE_INTRINSICS`
  inside a `#[cfg(feature = "std-surface")]` block — invisible to the
  default-build type checker.
- The C runtime-support file is only linked into the cdylib when
  `--emit-shared` is used, which itself is the `mlir-build` feature path.
  Default `cargo build` produces no `.so` and links no runtime-support.
- The clang invocation that compiles `mind_intrinsics.c` is unchanged
  (`-O2`, no `-march=native`, no `-mavx2`); the SIMD code is opt-in via
  the per-function `target` attribute, not the compiler-wide flag.

The +7% bench-gate cap (`.bench-baseline-2026-05-18-rfc0005.txt`, mlp,
small_matmul, full-pipeline) is therefore expected to hold trivially.

## 8. What lands and what does NOT

Lands in this RFC:

1. Six `__mind_blas_*` intrinsics in `runtime-support/mind_intrinsics.c`.
2. `std/blas.mind` and its registration in `STDLIB_MIND_SOURCES`.
3. Type-checker registration in `STD_SURFACE_INTRINSICS`.
4. `tests/blas_smoke.rs` exercising every intrinsic under both dispatch
   legs.
5. This RFC.

Does NOT land:

- A `bitcast<f32>(i64)` op in the MIND type system. The i64-packed return
  value is what pure-MIND callers consume today; the bitcast lands when
  scalar f32 enters the type system at a future phase.
- `@target("simd-x86" | "cuda")` annotations. Per-call sub-backend
  selection is Track B.
- A native MLIR `vector` dialect lowering inside mindc. That is Track B.
- An ARM NEON sub-backend. The translation unit only compiles AVX2 paths
  under `__x86_64__` / `_M_X64`; ARM hosts take the scalar fallback path
  unconditionally until Track B brings native vector-dialect lowering.

## 9. Track B — native MLIR vector-dialect lowering

Track B replaces the runtime-support C bridge with a thesis-pure path:
dense f32 reductions vectorize *through mindc itself* via the MLIR
`vector` dialect. LLVM's vector legalisation maps the ops to the host
SIMD width with no per-target code in mindc, no `-fPIC` shim object,
and no Windows-MSVC symbol-export problem.

### 9.1 Increment 1 (landed, v0.6.3)

Three new `#[cfg(feature = "std-surface")]`-gated IR primitives in
`src/ir/mod.rs`:

- `Instr::VecLoad { dst, base, offset, lanes }` — load `lanes`
  contiguous f32 from an opaque i64 heap address.
- `Instr::VecFma { dst, a, b, acc, lanes }` — element-wise fused
  multiply-add across lanes.
- `Instr::VecReduceAdd { dst, src, lanes }` — horizontal sum to an
  i64-packed f32 scalar.

`src/mlir/lowering.rs` emits the MLIR `vector` dialect:

- `vector.load` is realised as `llvm.inttoptr` (recover the pointer
  from the Option-C i64 address) + a byte `llvm.getelementptr` + a
  vector-typed `llvm.load` of `vector<lanes x f32>`.
- `vector.fma` lowers (via `convert-vector-to-llvm`) to the
  `llvm.intr.fmuladd` intrinsic.
- `vector.reduction <add>` lowers to `llvm.intr.vector.reduce.fadd`.

The `core` build pipeline registers `convert-vector-to-llvm` alongside
the existing `arith` / `scf` / `cf` / `func` conversions — a no-op on
vector-free IR, so every existing scalar program and the default
`cargo build` (which never runs `mlir-opt`) are byte-identical.

Surface: `std/blas.mind` exposes `pub fn dot_f32_v(a, b, len) -> i64`
over the new `__mind_blas_dot_f32_v` intrinsic. The intrinsic's
`Instr::Call` is intercepted and emitted as a fused 8-lane
`vector.fma` main loop + `vector.reduction <add>` horizontal sum +
scalar tail for the `len % 8` remainder. Track A's
`__mind_blas_dot_f32` extern path is unchanged and still the
runtime-support scalar/AVX2 fallback — Track B is strictly additive.

Numerical contract (`tests/blas_vec_smoke.rs`): within 1e-4 relative
of an f64 oracle on 1024- and 1M-element vectors (the pairwise
`vector.reduction` reorders summation exactly like Track A's AVX2
path); byte-identical to a sequential scalar reference for sub-lane
lengths. The bench-gate +7% cap (`.bench-baseline-2026-05-18-rfc0005.txt`)
held: small_matmul −0.5%, medium_mlp +0.1%, large_network +3.0%
(inside the documented large_network jitter band). v0.6.1 bootstrap
fixed-point unchanged (10,889 bytes / next_id 206) — the bootstrap
source uses no vector ops.

### 9.2 Deferred to increment 2

- `@target("simd-x86" | "simd-arm" | "cuda" | "q16-photonic")` per-call
  substrate annotation. Increment 1 lets the host target triple drive
  LLVM's vector legalisation; the portable `vector<8xf32>` width needs
  no explicit hint on x86/ARM, so no no-op marker token was added (it
  would imply more than it does).
- Q16.16 vector path (must stay byte-identical scalar-vs-vector for the
  cross-arch bit-identity gate #57), `Instr::VecStore`, and vector
  lowering for `dot_l1` / `dot_linf` / `matmul_rmajor`.
- Cross-module std-wrapper inlining. The `use std.blas` path emits a
  `func.func private @dot_f32_v` forward decl exactly as Track A's
  `dot_f32` does today; the working codegen entry point is the direct
  `__mind_blas_dot_f32_v` intrinsic call.
- A *dense reduction throughput* bench sub-category (p95 µs for an 8K-row
  matmul). The compile-time frontend gate (2.80–17.10 µs) MUST continue
  to hold; vector-dialect lowering stays module-gated.

## 10. References

- Bench spec: `mind-internal/plans/2026-05-18-mind-blas-bench-spec.md`
- A1.5 measurement: `mind-nerve@b9b6401` PARTIAL result
- RFC 0005: pure-MIND std surface (Phase 6.5 fixed-point precursor)
- RFC 0002: `pub fn` C exports (the C-ABI ground truth this RFC builds on)
- task #57: cross-arch Q16.16 bit-identity gate
- Landauer-floor framing: `mindlang.dev/docs/...` (TBD link once Track B
  publication lands)
