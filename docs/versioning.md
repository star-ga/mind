# MIND Core Stability & Versioning

MIND Core follows the mind-spec Core v1 contract (`mind-spec/spec/v1.0`). This
page documents how stability is applied for the compiler/runtime pipeline and
how versions are communicated in the CLI.

## Semantic Versioning (0.y.z)

MIND Core currently publishes 0.y.z versions with the following rules:

- **Patch (0.y.*)**: bug fixes, documentation updates, and tooling-only changes
  that do not alter IR semantics or CLI contracts.
- **Minor (0.y.*)**: additive IR extensions are allowed, including new
  instructions and flags. Existing IR semantics must remain compatible.

## What counts as breaking

- Changing the semantics of an existing IR instruction, its shape rules, or its
  broadcasting/reduction behavior.
- Altering CLI output formats (textual IR, diagnostics, or stable flags) in ways
  that break downstream tooling.
- Modifying MLIR lowering patterns that are relied on by snapshot tests or
  downstream compiler passes.

## Stability categories

### Stable surfaces

- **Core IR v1**: instruction set, shape rules, reductions, broadcasting, and
  verification guarantees.
- **Autodiff API**: `differentiate_function`, `GradientResult`, and
  `AutodiffError`.
- **Canonicalization guarantees**: deterministic rewrites prior to lowering.
- **`mindc` base flags and textual IR output**.
- **Core v1 GPU profile**: `DeviceKind`/`BackendTarget` enum variants for CPU and
  GPU, the runtime backend-selection error model (e.g., `BackendUnavailable`),
  and the `GPUBackend` trait surface that executes canonical IR on GPU devices.

### Conditionally stable surfaces

- **MLIR lowering** (`mlir-lowering` feature): stable within a minor version but
  may change across minor releases to track backend needs.

### Experimental surfaces

- New operations added to the IR.
- New feature flags exposed by the CLI or libraries.
- Concrete GPU or accelerator backend implementations and device-specific
  lowering pipelines.

## References

- mind-spec Core v1: https://github.com/cputer/mind-spec/tree/main/spec/v1.0
- Error model: see [`docs/errors.md`](errors.md).
