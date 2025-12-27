## MLIR lowering pipeline (public)

MIND can lower its deterministic IR into MLIR textual form for debugging and
future backend integration. The pipeline is intentionally lightweight and does
not invoke any private runtime code.

- Entry point: `compile_ir_to_mlir_text` (behind the `mlir-lowering` feature)
  runs IR verification, canonicalization, and lowering in one call.
- Supported operations in this phase: scalar/tensor constants, integer and
  tensor binary ops (`arith.*`), matrix multiply (`linalg.matmul`), and
  NHWC/HWCF convolutions lowered to `linalg.conv_2d_nhwc_hwcf`.
- The output is deterministic MLIR text suitable for demos or inspection. Any
  unsupported IR instruction results in a structured `MlirLowerError`.

Execution remains the responsibility of private/runtime crates; this pipeline
only produces MLIR text.
