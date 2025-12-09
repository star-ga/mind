# GPU backend profile

MIND exposes a stable Core v1 GPU profile that defines the contract for GPU
backends. The open-core crate does **not** ship a GPU implementation, but the
API surface and error model are intended to remain stable so downstream tooling
can target them.

## Device model

- `DeviceKind` distinguishes CPU and GPU devices. The CPU variant is implemented
  in this crate, and the GPU variant is part of the stable profile for
  downstream backends.
- GPU execution is allowed to be non-deterministic at the bit level (e.g., due
  to floating-point associativity). Backends must still preserve the semantic
  meaning of IR operations.

## Target model

- `BackendTarget::Cpu` is the default compilation target and is fully supported.
- `BackendTarget::Gpu` is defined by the Core v1 GPU profile. No concrete GPU
  backend ships with this crate, but downstream runtimes can implement it.
- Downstream tools can inspect the target to decide whether to dispatch to a
  custom backend via the `GPUBackend` trait.

## Error model

Requesting the GPU target currently returns a structured
`CompileError::BackendUnavailable` error from the pipeline. The `mindc` CLI
surfaces this as:

```
error[backend]: gpu backend not available (experimental interface only)
```

This behavior ensures an explicit failure instead of a panic or implicit
fallback when the GPU path is selected. The presence of a distinct
backend-selection error for unsupported or unavailable GPU targets is part of
the Core v1 GPU profile and must be preserved across compatible releases.
