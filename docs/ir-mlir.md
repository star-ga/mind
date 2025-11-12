# IR & MLIR Integration

MIND's compiler lowers high-level programs into a custom SSA intermediate representation (IR) before targeting MLIR dialects and LLVM.

## MIND IR

The IR is block-based SSA with explicit tensor types and shape metadata. Core features include:

- **Structured Ops** – Control flow is expressed with `if`, `loop`, and `switch` ops to maintain analyzable regions.
- **Pattern Matching** – Rewriting infrastructure performs algebraic simplifications, fusion, and layout selection.
- **Metadata Attachments** – Attributes capture tiling hints, memory spaces, and autodiff annotations.

An IR verifier enforces dominance, shape consistency, and effect constraints prior to lowering.

## Lowering Stages

1. **Canonicalization** – Simplify expressions, fold constants, and eliminate redundant casts.
2. **Bufferization** – Convert abstract tensors into explicit buffer descriptors with layout information.
3. **Dialect Mapping** – Translate IR ops into the `mind` MLIR dialect. Composite ops expand into standard MLIR + Linalg dialects.
4. **Target Specialization** – Apply CPU/GPU specific passes, vectorization, and library call rewriting.

Each stage is configurable through pass pipelines exposed on the CLI: `mind pass --pipeline=cpu-release`.

## MLIR Emission

The `mind-mlir` crate registers custom dialects and translation hooks. Highlights:

- Dialect ops mirror the IR 1:1 to keep transformations predictable.
- Conversion to LLVM IR uses MLIR's upstream conversion infrastructure.
- Optional serialization emits `.mlir` artifacts for debugging.

## Diagnostics & Tooling

- `mind dump-ir` – Print the SSA IR after each pass.
- `mind dump-mlir` – Emit the MLIR module prior to LLVM conversion.
- `mind lint` – Run verifier + style checks to catch invalid IR patterns early.

For runtime integration with generated code, continue to [`ffi-runtime.md`](ffi-runtime.md).
