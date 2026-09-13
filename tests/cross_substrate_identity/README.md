# cross_substrate_identity — the internal mind-bench reproducibility gate

This directory is the **internal** half of mind-bench (RFC 0020 §10): the
in-tree merge gate that pins MIND's cross-substrate bit-identity property to
committed reference hashes. It is the single source of truth that the future
public `mind-bench` CLI and versioned reference bundle (RFC 0020 §§3–5) will
consume. Public packaging and signed-reference integration remain pending.

## What it proves

Stronger than `tests/blas_vec_q16_smoke.rs` (which proves only that the vector
path equals its own scalar oracle *within a single run*): here the exact output
bytes are pinned to a committed constant, so byte-identity is enforced **across
builds, machines and time**. Any drift in mindc lowering, std-surface, or the
libc syscall surface shows up as a hash mismatch.

Per RFC 0015 §3.1, every substrate listed for a Q16.16 workload MUST yield the
**same** content hash — that shared hash *is* the cross-substrate bit-identity
claim, made inspectable in `reference_hashes.toml`.

## Layout

```
<workload-id>/
  manifest.toml          # deterministic spec: seed, length, kernel symbol, output encoding
  reference_hashes.toml  # <substrate> = "<sha256>" — the load-bearing committed hashes
```

The harness is `tests/cross_substrate_identity.rs`.

## Workloads

The table is a selected subset. The per-workload manifests and execution
receipts define the complete inventory and supported/deferred coverage.

| id | shape | kernel | output |
|----|-------|--------|--------|
| `dot-l2-q16` | dot product, len 65536 | `__mind_blas_dot_q16_v` | scalar i64 |
| `dot-l1-q16` | L1 distance, len 65536 | `__mind_blas_dot_l1_q16_v` | scalar i64 |
| `gemv-q16-256x256` | matrix×vector, 256×256 | `__mind_blas_matmul_rmajor_q16_v` | 256-vector |
| `gemm-q16-64x64x64` | matrix×matrix, 64×64×64 | composed (M gemv calls over Bᵀ) | 64×64 matrix |
| `gemm-i8-64x64x64` | int8 matrix×matrix, 64×64×64 | `__mind_blas_matmul_mm_i8_v` | 64×64 i32 matrix |
| `gemv-i16-256x256` | int16 matrix×vector, 256×256 | `__mind_blas_matmul_rmajor_i16_v` | 256-vector |
| `scalar-float-f64` | scalar IEEE chain | `scalar_f64_chain` (`a+b-c*d/a`) | scalar f64 |
| `dot-f32-v-4093` | strict f32 dot, len 4093 | `__mind_blas_dot_f32_v` | f32 bits packed into i64 |
| `matmul-f32-v-64x64` | strict f32 matrix×vector, 64×64 | `__mind_blas_matmul_rmajor_f32_v` | 64 f32 bit patterns |
| `dot-i16-4096` | int16 dot, len 4096 | `__mind_blas_dot_i16_v` | scalar i64 |
| `gemm-q16-fused-64x64x64` | fused matrix×matrix, 64×64×64 | `__mind_blas_matmul_mm_q16_v` | 64×64 matrix |
| `q16-arith-chain` | scalar Q16.16 arith chain | `q16_arith_chain` (`(x*y)>>16`) | scalar i64 |
| `struct-handle-roundtrip` | alloc/store/load round-trip | `struct_handle_roundtrip` | scalar i64 |
| `collatz` | integer Collatz (3n+1) hash over [1, 1000] | `collatz_hash` (examples/collatz.mind) | scalar i64 |
| `galperin-pi` | billiard-π collision count, n=5 → 31415 | `galperin_collisions` (examples/galperin_pi.mind) | scalar i64 |

The GEMM is the first matmul-MATMUL workload; it composes the proven gemv
intrinsic (`C[i,:] = gemv(Bᵀ, A[i,:])`), so its byte-identity is inherited from
`gemv-q16-256x256` — no new arithmetic, only an exact transpose + deterministic
row loop.

The Track #16 additions broaden the canary set to determinism-sensitive paths
the original four did not cover: the bare int16 dot reduction (`dot-i16-4096`),
the FUSED Q16.16 GEMM via the outer-product microkernel (`gemm-q16-fused-…`,
whose committed hash is **intentionally identical** to `gemm-q16-64x64x64` —
proving two different lowerings of the same GEMM produce the same bytes:
cross-LOWERING bit-identity, on top of cross-substrate), the scalar Q16.16
fixed-point arithmetic chain (`q16-arith-chain`, isolating the shift+integer
lowering), and the struct-by-handle alloc/store/load round-trip
(`struct-handle-roundtrip`, the heap-handle ABI + address-arithmetic path).

## Running

```
MIND_BENCH_REQUIRE=1 cargo test --no-default-features \
      --features "mlir-build std-surface cross-module-imports" \
      --test cross_substrate_identity
```

Run with `MIND_BENCH_BLESS` unset. `MIND_BENCH_REQUIRE=1` makes a missing
MLIR toolchain (`mlir-opt` / `mlir-translate` / `clang`) fail closed, as in CI.
An optional local run without that requirement can self-skip and does not
establish a pass. `avx2` is verified on x86_64 hosts; `neon` on aarch64.
CI also runs `cross_substrate_receipts` to verify coverage, and keeps the
non-asserting bless-mode harvest separate from the asserting gate.

## Adding a workload

1. Create `<id>/manifest.toml` (copy an existing one; set seed/length/kernel).
2. Add a `DotWorkload` entry + a `#[test]` in `cross_substrate_identity.rs`
   (the harness is table-driven; an exact-integer dot path is a few lines).
3. Bless the reference hash:
   `MIND_BENCH_BLESS=1 cargo test --features "..." --test cross_substrate_identity <name> -- --nocapture`
   then write the printed hash into `<id>/reference_hashes.toml`.
4. Re-run without `MIND_BENCH_BLESS` — it must pass.

Re-bless only on an **intentional** lowering change (RFC 0020 §13); document the
transition in the release notes. Public reference signing must use the exact
PQC hybrid in RFC 0020 §5.3. The compiler capability is implemented opt-in;
public harness integration, trust-anchor publication and operational release
signing remain pending. These committed reference files are unsigned.

## Floating-point scope

Exact-integer workloads and the three strict floating-point output fixtures
above have matching committed `avx2`/`neon` hashes (RFC 0015 §5A). The f32
dot and matrix-vector kernels use separate multiply/add operations and a
fixed lane fold; the f64 fixture pins a strict scalar operation sequence.
Each compares IEEE bit patterns exactly, without a tolerance. This coverage
does not establish bit-identity for arbitrary floating-point reductions,
transcendental functions, all inputs, or GPU execution. New cases require
their own strict computation, encoding and real-substrate evidence.
