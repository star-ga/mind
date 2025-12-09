# GPU backend (experimental)

MIND exposes an abstract GPU backend contract for future accelerator
integrations. The open-core crate does **not** ship a GPU implementation; the
surface exists for tooling and downstream runtimes to build upon.

## Device model

- `DeviceKind` distinguishes CPU and GPU devices. CPU is the only stable,
  implemented device in this crate.
- GPU execution is allowed to be non-deterministic at the bit level (e.g., due
  to floating-point associativity). Backends must still preserve the semantic
  meaning of IR operations.

## Target model

- `BackendTarget::Cpu` is the default compilation target and is fully supported.
- `BackendTarget::Gpu` is experimental and currently unimplemented.
- Downstream tools can inspect the target to decide whether to dispatch to a
  custom backend.

## Error model

Requesting the GPU target currently returns a structured
`CompileError::BackendUnavailable` error from the pipeline. The `mindc` CLI
surfaces this as:

```
error[backend]: gpu backend not available (experimental interface only)
```

This behavior ensures an explicit failure instead of a panic or implicit
fallback when the GPU path is selected.
