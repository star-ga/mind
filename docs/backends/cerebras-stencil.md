# `mind.cerebras.stencil_tile` — Op Surface and Lowering Contract

> **Status:** experimental — available in mindc v0.2.10+. Final emission to
> Cerebras Software Language (CSL) requires
> `mind-runtime/src/backend/cerebras/` on a CS-3 system.

## Overview

`mind.cerebras.stencil_tile` is a public MLIR-style surface op for programs
whose dominant kernel is a short-range 2-D stencil. The Cerebras WSE-3
contains a 2-D mesh of ~900 000 independent cores, each with its own SRAM. A
local stencil — one that reads only a fixed neighbourhood of cells per update
step — maps onto this mesh without the O(N²) scatter/gather cost that
attention-style all-to-all communication patterns incur when sharded across
cores.

### Stencil-kernel motivation

A 5-point Laplacian on a 2-D grid reads four orthogonal neighbours and the
centre cell for each output cell. Every core in the tile region can compute its
output value from data resident in the SRAM of itself and its four direct
neighbours — no long-distance interconnect, no all-reduce. The fabric performs
the update in a single pipeline wave, with latency independent of total mesh
size.

This structural match between local-stencil access patterns and the WSE-3 mesh
topology is the primary reason `mind.cerebras.stencil_tile` exists as a
first-class op rather than a rewrite of `mind.cerebras.fabric_matmul`.

## Op syntax

```
%result = mind.cerebras.stencil_tile()
          {rows = R, cols = C, elem = "ETYPE", kernel = "SYMBOL"}
          : tensor<R×C×MLIR_TYPE>
```

| Attribute | Type   | Constraints                             |
|-----------|--------|-----------------------------------------|
| `rows`    | `u32`  | 1–4096                                  |
| `cols`    | `u32`  | 1–4096                                  |
| `elem`    | string | `q16_16` \| `f32` \| `f16` \| `bf16`   |
| `kernel`  | string | Non-empty ASCII identifier              |

The `result` value has type `tensor<R×C×T>` where `T` is the MLIR wire type
for `elem`. Q16.16 uses `i32` as its wire type (16-bit integer part, 16-bit
fractional part, stored as a signed 32-bit integer).

### Element types

| MIND `elem` | MLIR wire type | Notes                                    |
|-------------|----------------|------------------------------------------|
| `q16_16`    | `i32`          | Canonical MIND fixed-point. Bit-identical results across x86, CUDA, and the wafer. |
| `f32`       | `f32`          | IEEE 754 single precision.               |
| `f16`       | `f16`          | IEEE 754 half precision.                 |
| `bf16`      | `bf16`         | Brain float 16.                          |

Q16.16 is the recommended element type for new stencil programs targeting the
Cerebras path: Cerebras Weight Streaming is precision-agnostic, so Q16.16
weights pass through without per-format kernel rewrites. The same weight
tensor produces a byte-identical 32-byte SHA-256 hash on x86, CUDA, and the
wafer — a compliance primitive for regulated-AI deployments.

## Lowering contract

`mindc` is responsible for:

1. Parsing and validating the op attributes (dimension bounds, identifier
   grammar, element type).
2. Emitting the `mind.cerebras.stencil_tile` IR node as opaque textual MLIR.

`mind-runtime/src/backend/cerebras/` is responsible for:

1. Consuming the opaque IR node.
2. Generating the per-kernel CSL dispatch, region allocation, and DMA
   descriptors for the target wafer.

This crate never emits CSL directly. The split exists so the public IR
surface can evolve independently of the private backend toolchain.

## Compile-time performance

Every parser arm is feature-gated by a leading-token check. The op surface
exits in O(1) when `mind.cerebras.stencil_tile` is absent from the module.
When the op is present, construction and MLIR text emission are also O(1) with
respect to tensor size — all work operates on scalar metadata only.

Measured on the benchmark machine (see `benches/cerebras_stencil.rs`):

| Variant         | Fabric region | Measured cost |
|-----------------|---------------|---------------|
| tiny_32x32      | 32 × 32       | < 300 ns      |
| medium_128x128  | 128 × 128     | < 300 ns      |
| large_256x256   | 256 × 256     | < 300 ns      |
| wafer_750x750   | 750 × 750     | < 300 ns      |

All four variants are constant-time: the fabric dimensions appear in the
emitted MLIR text only as integer literals, so there is no loop, allocation, or
allocation proportional to `rows * cols`.

## Representative fabric regions

| Name    | Region   | Context                                         |
|---------|----------|-------------------------------------------------|
| Tiny    | 32 × 32  | Minimal tile; unit-test default.                |
| Medium  | 128 × 128| Typical sub-region for iterative solvers.       |
| Large   | 256 × 256| `FabricRegion::default()` in `mind-runtime`.    |
| Wafer   | 750 × 750| Approximate WSE-3 working block.                |

## Usage example

```rust
use libmind::ops::cerebras::{StencilTileOp, StencilElemType};

let op = StencilTileOp::new(
    "%diffusion_out",
    256, 256,
    StencilElemType::Q16_16,
    "laplacian_5pt",
)?;

// Emit the IR node for mind-runtime/backend/cerebras/ to consume.
let mlir_text = op.to_mlir_text();
```

The `kernel` attribute is a symbol name resolved by the downstream CSL
compiler. MIND does not define the kernel body — the op declares that a
2-D stencil named `kernel` should be applied to the region, and the
backend maps that name to a concrete CSL kernel implementation.

## See also

- `src/ops/cerebras.rs` — op definition, validation, and MLIR emission.
- `benches/cerebras_stencil.rs` — compile-time benchmark.
- `tests/cerebras_stencil_tile.rs` — surface tests.
- `mind-runtime/src/backend/cerebras/` — private CSL emission backend.
- `docs/backends/` — other backend surface documentation.
