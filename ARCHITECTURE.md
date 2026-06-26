# MIND Architecture (high level)

- `src/parser`: lexer+parser → `ast`
- `src/types`: dtype/shape/value typing
- `src/eval`: front-end evaluation (preview), IR, MLIR export, optional exec backends
- `src/exec`: optional CPU kernels (feature-gated)
- `src/ffi`: optional C FFI (feature-gated)
- `src/package`: optional packaging (feature-gated)
- `src/main.rs`: CLI (gates subcommands by features)

**Design principles**
- `--no-default-features` always builds & tests
- Feature gates: `cpu-exec`, `cpu-conv`, `mlir-*`, `pkg`, `ffi-c`
- Tests prefer integration tests in `tests/`

### Backend architecture (v0.10.0+)

The **NATIVE-ELF backend** (`src/native`) is the normative self-host target —
determinism-by-construction (the native ELF is a pure function of the IR). The
**MLIR-text backend** is a downstream-interchange and exotic-chip-reach backend,
demoted from the self-host path but retained for broader chip targets. "Target any
chip" is implemented via a pluggable `Backend` trait plus commercial backends in
the private `mind-runtime`.

### Stability
The architecture is stable for v0.10.0.  
Breaking changes will only occur in v1.0 once SDK bindings and developer docs are added.
