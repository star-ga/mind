# cross_substrate_identity — the internal mind-bench reproducibility gate

This directory is the **internal** half of mind-bench (RFC 0020 §10): the
in-tree merge gate that pins MIND's cross-substrate bit-identity property to
committed reference hashes. It is the single source of truth that the future
public `mind-bench` CLI (RFC 0020 §3) and the published
`mind-spec/wedge-reference-hashes/<version>.txt` manifest both consume.

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

## Running

```
cargo test --features "mlir-build std-surface cross-module-imports" \
      --test cross_substrate_identity
```

Self-skips if the MLIR toolchain (`mlir-opt` / `mlir-translate` / `clang`) is
not on PATH — the property is verified by STARGA engineers on a
toolchain-equipped host and on the per-substrate CI runners (RFC 0020 §10), not
on stock runners. `avx2` is verified on x86_64 hosts; `neon` on aarch64.

## Adding a workload

1. Create `<id>/manifest.toml` (copy an existing one; set seed/length/kernel).
2. Add a `DotWorkload` entry + a `#[test]` in `cross_substrate_identity.rs`
   (the harness is table-driven; an exact-integer dot path is a few lines).
3. Bless the reference hash:
   `MIND_BENCH_BLESS=1 cargo test --features "..." --test cross_substrate_identity <name> -- --nocapture`
   then write the printed hash into `<id>/reference_hashes.toml`.
4. Re-run without `MIND_BENCH_BLESS` — it must pass.

Re-bless only on an **intentional** lowering change (RFC 0020 §13); document the
transition in the release notes. Ed25519 signing of the hashes (RFC 0020 §5.3)
lands with the pure-MIND CLI once std-crypto exists.

## Why only Q16.16 integer workloads

Only exact-integer reductions qualify as byte-identity workloads. The f32 L1/L∞
paths use tree-shaped reductions that reorder summation, so they are *not*
bit-exact across substrates — they belong in the approximate-comparison surface
(RFC 0020 §8 `compare`), never as a byte-identity reference.
